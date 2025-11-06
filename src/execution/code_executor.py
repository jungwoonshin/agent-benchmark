"""Code execution handler with retry and error fixing."""

import json
import logging
from typing import Any, Dict, Optional

from ..models import SearchResult
from ..state import InformationStateManager, Subtask
from ..tools import ToolBelt
from ..utils import extract_json_from_text
from .result_analyzer import ExecutionResultAnalyzer


class CodeExecutor:
    """Handles code execution with automatic retry and error fixing."""

    def __init__(
        self,
        tool_belt: ToolBelt,
        llm_service: Any,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize CodeExecutor.

        Args:
            tool_belt: ToolBelt instance with available tools.
            llm_service: LLM service for code fixing.
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.tool_belt = tool_belt
        self.llm_service = llm_service
        self.state_manager = state_manager
        self.logger = logger

    def serialize_result_for_code(self, result: Any) -> Any:
        """
        Serialize result objects for use in LLM reasoning.

        Converts SearchResult dataclass objects to dictionaries to avoid
        iteration errors in RestrictedPython execution environment.

        Args:
            result: The result to serialize (can be dict, list, SearchResult, or other types)

        Returns:
            Serialized result with SearchResult objects converted to dictionaries
        """
        # If result is a SearchResult object, convert to dict
        if isinstance(result, SearchResult):
            return {
                'snippet': result.snippet,
                'url': result.url,
                'title': result.title,
                'relevance_score': result.relevance_score,
            }

        # If result is a dictionary, recursively serialize values
        if isinstance(result, dict):
            serialized = {}
            for key, value in result.items():
                if isinstance(value, SearchResult):
                    serialized[key] = {
                        'snippet': value.snippet,
                        'url': value.url,
                        'title': value.title,
                        'relevance_score': value.relevance_score,
                    }
                elif isinstance(value, list):
                    # Recursively serialize list items
                    serialized[key] = [
                        self.serialize_result_for_code(item) for item in value
                    ]
                elif isinstance(value, dict):
                    # Recursively serialize nested dictionaries
                    serialized[key] = self.serialize_result_for_code(value)
                else:
                    serialized[key] = value
            return serialized

        # If result is a list, recursively serialize items
        if isinstance(result, list):
            return [self.serialize_result_for_code(item) for item in result]

        # For other types (str, int, etc.), return as-is
        return result

    def execute_code_with_retry(
        self,
        code: str,
        context: Dict[str, Any],
        problem: str,
        subtask: Subtask,
        max_retries: int = 10,
    ) -> Any:
        """
        Execute code with automatic retry and error fixing.

        Args:
            code: Python code to execute
            context: Context dictionary with variables
            problem: Original problem description
            subtask: Subtask being executed
            max_retries: Maximum number of retry attempts

        Returns:
            Execution result (successful result or error dict)
        """
        retry_count = subtask.metadata.get('code_fix_retry_count', 0)
        current_code = code
        last_error = None

        # Execute initial code
        result = self.tool_belt.code_interpreter(current_code, context)

        # Keep retrying until success or max retries reached
        while retry_count < max_retries:
            # Check if result is an error
            is_error = ExecutionResultAnalyzer.is_error_result(result)

            if not is_error:
                # Success! Handle success and return
                return self.handle_code_execution_success(result, retry_count, subtask)

            # Error detected - extract error message and try to fix
            error_reason = ExecutionResultAnalyzer.extract_error_message(result)
            if error_reason is None:
                error_reason = str(result)[:500]
            last_error = error_reason[:500]

            self.logger.warning(
                f'Code execution failed (attempt {retry_count + 1}/{max_retries}): {error_reason}'
            )

            # Attempt to fix the code
            fixed_code = self.fix_code_error(
                current_code, error_reason, context, problem, subtask
            )

            if not fixed_code or fixed_code == current_code:
                # Could not fix the code or fix didn't change anything
                self.logger.warning(
                    'Could not fix code error or fix produced identical code. Stopping retries.'
                )
                break

            # Update retry count and metadata
            retry_count += 1
            self.update_code_retry_metadata(
                subtask, code, retry_count, error_reason, fixed_code
            )

            self.logger.info(
                f'Retrying code execution with fixed code (attempt {retry_count}/{max_retries})'
            )

            # Update current_code for next iteration
            current_code = fixed_code

            # Retry with fixed code
            result = self.tool_belt.code_interpreter(fixed_code, context)

        # Check if we still have an error after all retries
        return self.handle_code_execution_failure(
            result, last_error, retry_count, subtask
        )

    def handle_code_execution_success(
        self, result: Any, retry_count: int, subtask: Subtask
    ) -> Any:
        """
        Handle successful code execution.

        Args:
            result: Successful execution result
            retry_count: Number of retries that were made
            subtask: Subtask that was executed

        Returns:
            The successful result
        """
        if retry_count > 0:
            self.logger.info(
                f'Code execution succeeded after {retry_count} retry attempt(s)!'
            )
            # Clear error metadata since we succeeded
            self.clear_code_error_metadata(subtask)

        return result

    def handle_code_execution_failure(
        self,
        result: Any,
        last_error: Optional[str],
        retry_count: int,
        subtask: Subtask,
    ) -> Dict[str, Any]:
        """
        Handle code execution failure after all retries exhausted.

        Args:
            result: The final execution result (should indicate an error)
            last_error: The last error message encountered
            retry_count: Number of retry attempts made
            subtask: Subtask that failed

        Returns:
            Structured error dictionary
        """
        # Check if result is still an error
        is_final_error = ExecutionResultAnalyzer.is_error_result(result)

        if not is_final_error:
            # Unexpected: result is not an error, treat as success
            self.logger.warning(
                'Code execution result is not an error after retries. Treating as success.'
            )
            return result

        # Extract error message and normalize result
        _, normalized_result, error_message = (
            ExecutionResultAnalyzer.normalize_error_result(result)
        )

        # Use normalized error message or fallback to last_error
        error_reason = (
            error_message or last_error or normalized_result or 'Unknown error'
        )
        error_reason = str(error_reason)[:500]

        self.logger.error(
            f'Code execution failed after {retry_count} retry attempt(s). Giving up.'
        )

        # Mark subtask as failed
        self.state_manager.fail_subtask(subtask.id, error_reason)

        # Store error in metadata and classify error type
        error_type = ExecutionResultAnalyzer.classify_error_type(error_reason)
        subtask.metadata['error'] = error_reason
        subtask.metadata['error_type'] = error_type

        # Return structured error dict
        return ExecutionResultAnalyzer.create_error_dict(
            error_reason,
            error_type=error_type,
            subtask_id=subtask.id,
            retry_attempts=retry_count,
        )

    def update_code_retry_metadata(
        self,
        subtask: Subtask,
        original_code: str,
        retry_count: int,
        error_reason: str,
        fixed_code: str,
    ) -> None:
        """
        Update subtask metadata with retry information.

        Args:
            subtask: Subtask being retried
            original_code: Original code that failed
            retry_count: Current retry count
            error_reason: Error message from last attempt
            fixed_code: Fixed code to retry with
        """
        subtask.metadata['code_fix_retry_count'] = retry_count
        if 'original_code' not in subtask.metadata:
            subtask.metadata['original_code'] = original_code
        subtask.metadata['last_error'] = error_reason
        subtask.metadata['last_fixed_code'] = fixed_code

    def clear_code_error_metadata(self, subtask: Subtask) -> None:
        """
        Clear error-related metadata from subtask after successful execution.

        Args:
            subtask: Subtask that succeeded
        """
        metadata_keys_to_remove = ['error', 'error_type', 'code_fix_retry_count']
        for key in metadata_keys_to_remove:
            if key in subtask.metadata:
                del subtask.metadata[key]

    def fix_code_error(
        self,
        code: str,
        error_message: str,
        context: Dict[str, Any],
        problem: str,
        subtask: Subtask,
    ) -> Optional[str]:
        """
        Use LLM to fix code errors by analyzing the error and modifying the code.

        Args:
            code: The original code that failed.
            error_message: The error message from code execution.
            context: The context dictionary available to the code.
            problem: The original problem description.
            subtask: The subtask being executed.

        Returns:
            Fixed code string, or None if fixing failed.
        """
        self.logger.debug(f'Attempting to fix code error: {error_message[:200]}...')

        system_prompt = """You are an expert Python code debugger and fixer.
Given Python code that failed to execute with an error message, analyze the error and fix the code.

CRITICAL CONSTRAINTS:
- **llm_reasoning**: Use for calculations, data processing, and analysis. Provide clear task descriptions.
- **Context access**: Variables from context are available. Use dictionary access: context['key'] NOT context.key
- **For NameError**:
  - FIRST: Check if the NameError is for a module name (e.g., re, math, datetime, json, os, sys, pandas, numpy, etc.). If so, add `import <module_name>` at the top of the code.
  - SECOND: Check if the variable exists in context. If accessing context variables, use dictionary syntax: context['step_1'] not step_1
- **For ImportError**:
  - If the module cannot be imported (doesn't exist or not installed), suggest using an alternative approach or note that the module needs to be installed
- **For SyntaxError**: Fix the syntax error
- **For ExecutionError**: Fix the logic error

Return ONLY the fixed Python code as a JSON object with this exact structure:
{
  "fixed_code": "the fixed Python code here"
}

IMPORTANT:
- Always include necessary import statements at the top of the code
- If code uses `re.search()`, `datetime.now()`, `math.sqrt()`, `pandas.read_csv()`, `numpy.array()`, etc., ensure the corresponding import statements are present (e.g., `import re`, `import datetime`, `import math`, `import pandas`, `import numpy`)
- Return your response as valid JSON only, without any markdown formatting or additional text."""

        # Build context info for the prompt
        context_info = ''
        if context:
            context_keys = list(context.keys())
            context_info = f'\nAvailable context keys: {", ".join(context_keys[:20])}'
            if len(context_keys) > 20:
                context_info += f' (and {len(context_keys) - 20} more)'

        user_prompt = f"""Problem: {problem}

Subtask Description: {subtask.description}

Original Code (that failed):
```python
{code}
```

Error Message:
{error_message}
{context_info}

Analyze the error and fix the code. Return the fixed code in JSON format:
{{
  "fixed_code": "fixed Python code here"
}}

IMPORTANT:
- Fix the specific error mentioned in the error message
- If error mentions a missing variable, check if it's in context and use context['key'] syntax
- If error mentions an import, remove it or use whitelisted alternatives
- Keep the same logic and intent, just fix the error
- Return ONLY the fixed code, no explanations
- Return your response as valid JSON only, without any markdown formatting or additional text"""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for more consistent fixes
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            fixed_data = json.loads(json_text)
            fixed_code = fixed_data.get('fixed_code', '')

            if fixed_code and fixed_code.strip():
                self.logger.info(
                    f'Code fix generated successfully (length: {len(fixed_code)} chars)'
                )
                return fixed_code.strip()
            else:
                self.logger.warning('Code fix returned empty or invalid code')
                return None
        except Exception as e:
            self.logger.error(f'Failed to generate code fix: {e}', exc_info=True)
            return None
