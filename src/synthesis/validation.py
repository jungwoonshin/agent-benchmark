"""Answer Validation Module for checking if final answers are correct."""

import json
import logging
from typing import Any, Dict, List

from ..llm import LLMService
from ..state import InformationStateManager, Subtask
from ..utils import extract_json_from_text


class AnswerValidator:
    """Validates if final answers are correct based on execution results."""

    def __init__(
        self,
        llm_service: LLMService,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize Answer Validator.

        Args:
            llm_service: LLM service instance.
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.state_manager = state_manager
        self.logger = logger

    def validate_answer(
        self,
        problem: str,
        final_answer: str,
        query_analysis: Dict[str, Any],
        execution_results: Dict[str, Any],
        combined_results: Dict[str, Any],
        plan: List[Subtask],
    ) -> Dict[str, Any]:
        """
        Validate if the final answer is correct based on all previous information.

        Args:
            problem: Original problem description.
            final_answer: The final answer to validate.
            query_analysis: Query analysis results.
            execution_results: Execution results from all subtasks.
            combined_results: Combined execution and reasoning results.
            plan: Execution plan with all subtasks.

        Returns:
            Dictionary with validation result containing:
            - is_correct: bool
            - reason: str (if incorrect)
            - incorrect_subtask_ids: List[str] (if incorrect)
        """
        self.logger.info('Validating answer correctness using LLM...')

        # Prepare context for validation with more complete information
        execution_summary = []
        for subtask_id, result in execution_results.items():
            subtask = self.state_manager.subtasks.get(subtask_id)
            if subtask:
                # Include more complete results (increase limit for better context)
                result_str = str(result)
                # For search results, include more context
                if 'search_results' in result_str or 'SearchResult' in result_str:
                    result_preview = result_str[
                        :5000
                    ]  # More context for search results
                elif 'wikipedia' in result_str.lower() or 'api' in subtask.metadata.get(
                    'tool', ''
                ):
                    # For API results (like Wikipedia), include more context
                    result_preview = result_str[:5000]
                elif 'llm_reasoning' in subtask.metadata.get('tool', ''):
                    # For LLM reasoning results, include full context (they're usually important)
                    result_preview = result_str[:8000]
                else:
                    result_preview = result_str[:3000]  # More context for other results

                execution_summary.append(
                    {
                        'subtask_id': subtask_id,
                        'description': subtask.description,
                        'tool': subtask.metadata.get('tool', 'unknown'),
                        'result': result_preview if result else 'None',
                        'result_length': len(result_str) if result else 0,
                    }
                )

        # Truncate execution summary and reasoning summary to prevent prompt from being too long
        # Limit execution summary to avoid API errors, but allow more space for important results
        execution_summary_json = json.dumps(
            execution_summary, indent=2, ensure_ascii=False
        )
        max_execution_length = (
            20000  # Increased limit to preserve more execution step information
        )
        if len(execution_summary_json) > max_execution_length:
            # Try to preserve complete results by truncating from the end of the JSON
            # rather than cutting off mid-result
            execution_summary_json = (
                execution_summary_json[:max_execution_length] + '\n... [truncated]'
            )

        # Create improved validation prompt
        user_prompt = f"""PROBLEM:
{problem}

EXPLICIT REQUIREMENTS:
{json.dumps(query_analysis.get('explicit_requirements', []), indent=2)}

ANSWER FORMAT REQUIREMENT:
{query_analysis.get('answer_format', '')}

EXECUTION STEPS:
{execution_summary_json}

FINAL ANSWER:
{final_answer}

VALIDATION INSTRUCTIONS:
Your task is to determine if the final answer is CORRECT for the problem, considering:

1. **Answer Correctness**: Does the answer correctly address the problem requirements? Even if some execution steps returned None or had partial results, the answer synthesizer may have successfully combined information from earlier steps.

2. **Format Compliance**: Does the answer match the required format (e.g., if format requires "word", is it a single word? If format requires "number", is it a number? If format requires "zip codes", are they five-digit zip codes?). **CRITICAL: If the problem asks for "the name of a character" or "character name", the answer MUST be a character NAME in words NOT the symbol itself.**

3. **Information Synthesis**: Consider that the answer synthesizer may have successfully extracted and combined information from multiple execution steps, even if later steps didn't produce perfect results. The presence of "None" in a step doesn't necessarily mean the answer is wrong - it may mean the synthesizer used earlier successful steps.

4. **Evidence Evaluation**: Look for evidence that supports the answer in ANY of the execution steps or reasoning summary. The answer doesn't need to be explicitly documented in every step - successful synthesis from partial information is valid.

5. **Logical Consistency with Reasoning**: Critically evaluate whether the FINAL ANSWER is logically consistent with the conclusions, optimal choices, or derived information presented in the REASONING SUMMARY. If the reasoning points to a specific outcome (e.g., a particular ball number as optimal), the final answer MUST reflect this.

6. **Be Lenient on Process**: Focus on whether the ANSWER is correct, not whether every execution step perfectly documented how it was derived. An answer can be correct even if the execution trace is incomplete.

IMPORTANT:
- If the answer appears correct and matches format requirements AND is logically consistent with the reasoning, mark it as correct even if some steps returned None
- Only mark as incorrect if the answer is clearly wrong, doesn't match format, violates explicit requirements, OR is inconsistent with the reasoning summary.
- Consider that zip codes, numbers, words, etc. extracted from earlier search/browse steps may be valid even if later formatting steps didn't complete
- **CRITICAL CHECK FOR CHARACTER NAMES**: If the problem explicitly asks for "the name of a character" or "character name", verify the answer is a NAME not a symbol. Reject answers that return symbols when names are required.

Return a JSON object with:
{{
  "is_correct": {{true/false}},
  "reason": "Brief explanation of why the answer is correct or incorrect. If correct, acknowledge successful synthesis even if some steps were incomplete.",
  "incorrect_subtask_ids": ["subtask_id1", "subtask_id2"] (only if incorrect AND the step truly produced wrong data, empty list if correct or if answer is correct despite incomplete steps)
}}

Note: Set "is_correct" to true or false (boolean), not a string. Use an empty list [] for "incorrect_subtask_ids" if the answer is correct or if steps were incomplete but the answer is still correct.

Only include subtask IDs if they produced FACTUALLY INCORRECT data. Do NOT include steps that returned None or had incomplete results if the final answer is still correct.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.
"""

        system_prompt = """You are an expert validator focused on answer correctness, not execution trace perfection.
Your primary goal is to determine if the final answer correctly solves the problem and matches format requirements.
Consider that successful answer synthesis may combine information from multiple steps, even if some steps had incomplete results.
Be lenient on process quality - focus on answer correctness."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,  # Maximum determinism for consistent validation
                response_format={'type': 'json_object'},
            )
            self.logger.debug(f'Validation LLM response: {response[:500]}...')

            # Extract JSON from response
            validation_data = extract_json_from_text(response)
            validation_data = json.loads(validation_data)
            if not validation_data:
                # Fallback: assume incorrect if we can't parse (safer for validation)
                self.logger.warning(
                    'Could not parse validation response. Assuming answer is incorrect.'
                )
                return {
                    'is_correct': False,
                    'reason': 'Validation response could not be parsed',
                    'incorrect_subtask_ids': [],
                }

            # Ensure all required keys are present with defaults
            return {
                'is_correct': validation_data.get('is_correct', False),
                'reason': validation_data.get('reason', 'No reason provided'),
                'incorrect_subtask_ids': validation_data.get(
                    'incorrect_subtask_ids', []
                ),
            }

        except Exception as e:
            self.logger.error(f'Error during answer validation: {e}', exc_info=True)
            # Fallback: assume incorrect on error (safer for validation)
            return {
                'is_correct': False,
                'reason': f'Validation error: {str(e)}',
                'incorrect_subtask_ids': [],
            }
