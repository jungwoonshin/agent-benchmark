"""Error detection and analysis utilities for code execution results."""

import re
from typing import Any, Dict, Optional, Tuple


class ExecutionResultAnalyzer:
    """Analyzes code execution results to determine success or failure."""

    # Standard error prefixes that indicate execution failures
    ERROR_PREFIXES = (
        'Name Error:',
        'Execution Error:',
        'Import Error:',
        'Syntax Error:',
        'Compilation Error:',
    )

    # Pattern to match dict-like error strings: {'error': "..."} or {"error": "..."}
    ERROR_DICT_PATTERN = re.compile(
        r"\{['\"]error['\"]\s*:\s*['\"]([^'\"]+)['\"]"
    )

    @classmethod
    def is_error_result(cls, result: Any) -> bool:
        """
        Check if a result indicates an execution error.

        Args:
            result: The execution result (can be dict, str, or other types)

        Returns:
            True if result indicates an error, False otherwise
        """
        if isinstance(result, dict):
            # Check if result is a dict with 'error' key
            return 'error' in result

        if isinstance(result, str):
            # Check for standard error prefixes
            if any(result.startswith(prefix) for prefix in cls.ERROR_PREFIXES):
                return True

            # Check for dict-like error patterns in string
            if cls.ERROR_DICT_PATTERN.search(result):
                return True

        return False

    @classmethod
    def extract_error_message(cls, result: Any) -> Optional[str]:
        """
        Extract error message from a result that indicates an error.

        Args:
            result: The execution result (can be dict, str, or other types)

        Returns:
            Extracted error message, or None if not an error
        """
        if isinstance(result, dict) and 'error' in result:
            return str(result.get('error', 'Unknown error'))

        if isinstance(result, str):
            # Check for dict-like error pattern and extract
            error_dict_match = cls.ERROR_DICT_PATTERN.search(result)
            if error_dict_match:
                return error_dict_match.group(1)

            # Check for standard error prefixes
            if any(result.startswith(prefix) for prefix in cls.ERROR_PREFIXES):
                return result

        return None

    @classmethod
    def normalize_error_result(cls, result: Any) -> Tuple[bool, str, Optional[str]]:
        """
        Analyze result and return normalized error information.

        Args:
            result: The execution result (can be dict, str, or other types)

        Returns:
            Tuple of (is_error, normalized_result_str, error_message)
            - is_error: True if result indicates an error
            - normalized_result_str: Result as string, formatted if error
            - error_message: Extracted error message if error, None otherwise
        """
        is_error = cls.is_error_result(result)
        error_message = None

        if is_error:
            error_message = cls.extract_error_message(result)

            # Normalize to standard error format
            if isinstance(result, dict) and 'error' in result:
                normalized_result = f'Execution Error: {error_message}'
            elif isinstance(result, str):
                # If it's already a standard error format, use it
                if any(result.startswith(prefix) for prefix in cls.ERROR_PREFIXES):
                    normalized_result = result
                else:
                    # Extract and reformat
                    normalized_result = f'Execution Error: {error_message}'
            else:
                normalized_result = f'Execution Error: {error_message}'
        else:
            normalized_result = str(result) if result is not None else ''

        return is_error, normalized_result, error_message

    @classmethod
    def classify_error_type(cls, error_message: str) -> str:
        """
        Classify the type of error from error message.

        Args:
            error_message: The error message string

        Returns:
            Error type: 'import_error', 'name_error', 'syntax_error', 'execution_error', or 'unknown_error'
        """
        error_lower = error_message.lower()

        if 'import error:' in error_lower or 'failed to import' in error_lower:
            return 'import_error'
        elif 'name error:' in error_lower or "name '" in error_lower:
            return 'name_error'
        elif (
            'syntax error:' in error_lower
            or 'compilation error:' in error_lower
        ):
            return 'syntax_error'
        elif 'execution error:' in error_lower:
            return 'execution_error'
        else:
            return 'unknown_error'

    @classmethod
    def create_error_dict(
        cls,
        error_message: str,
        error_type: Optional[str] = None,
        subtask_id: Optional[str] = None,
        retry_attempts: int = 0,
    ) -> Dict[str, Any]:
        """
        Create a standardized error dictionary.

        Args:
            error_message: The error message
            error_type: Optional error type (will be classified if not provided)
            subtask_id: Optional subtask ID
            retry_attempts: Number of retry attempts made

        Returns:
            Standardized error dictionary
        """
        if error_type is None:
            error_type = cls.classify_error_type(error_message)

        error_dict: Dict[str, Any] = {
            'error': error_message[:500],  # Limit length
            'error_type': error_type,
            'status': 'failed',
        }

        if subtask_id:
            error_dict['subtask_id'] = subtask_id

        if retry_attempts > 0:
            error_dict['retry_attempts'] = retry_attempts

        return error_dict

