"""Agent class for solving complex multi-problem tasks."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ..execution import Executor
from ..llm import LLMService
from ..models import Attachment
from ..planning import Planner, ProblemClassifier, QueryUnderstanding
from ..state import InformationStateManager
from ..synthesis import AnswerSynthesizer, AnswerValidator
from ..tools import ToolBelt

# Load environment variables
load_dotenv()


class Agent:
    """
    Main agent class that solves complex problems using tools and step-by-step reasoning.
    """

    def __init__(
        self,
        tool_belt: ToolBelt,
        logger: logging.Logger,
        llm_model: str = None,
    ):
        """
        Initialize the Agent with a ToolBelt and a logger.

        Args:
            tool_belt: An instance of the ToolBelt.
            logger: A pre-configured logging.Logger instance.
            llm_model: OpenAI model to use (default: from LLM_MODEL env var, or 'openai/gpt-oss-120b').
        """
        self.tool_belt = tool_belt
        self.logger = logger
        # Pass the logger to the toolbelt for unified logging
        self.tool_belt.set_logger(logger)

        # Load model from environment variable if not provided
        model = llm_model or os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')
        # Initialize LLM service
        self.llm_service = LLMService(logger, model=model)
        # Pass LLM service to toolbelt for intelligent extraction
        self.tool_belt.set_llm_service(self.llm_service)

        # Initialize core modules
        self.state_manager = InformationStateManager(logger)
        self.query_understanding = QueryUnderstanding(self.llm_service, logger)
        self.problem_classifier = ProblemClassifier(self.llm_service, logger)
        self.planner = Planner(self.llm_service, self.state_manager, logger)
        self.executor = Executor(
            tool_belt, self.llm_service, self.state_manager, logger
        )
        self.answer_synthesizer = AnswerSynthesizer(
            self.llm_service, self.state_manager, logger
        )
        self.answer_validator = AnswerValidator(
            self.llm_service, self.state_manager, logger
        )

    def solve(
        self, problem: str, attachments: Optional[List[Attachment]] = None
    ) -> str:
        """
        Solves a complex problem using tools and a step-by-step reasoning process.

        Implements the control flow:
        1. PARSE: Decompose query into atomic requirements
        2. CLASSIFY: Determine the problem type
        3. PLAN: Generate execution strategy
        4. EXECUTE: Run tool operations
        5. SYNTHESIZE: Construct answer from components
        6. VALIDATE: Check if answer is correct and retry incorrect execution steps if needed

        Logs its operations to the logger.

        Args:
            problem: The natural language problem description.
            attachments: An optional list of file attachments (images, PDFs, etc.).

        Returns:
            final_answer: str
        """
        self.logger.info('--- New Problem Received ---')
        self.logger.info(f'Problem: {problem}')

        # Initialize attachments list if None (needed for downloaded files)
        if attachments is None:
            attachments = []

        if attachments:
            self.logger.info(f'Attachments: {[a.filename for a in attachments]}')

        try:
            # Step 1: Classify Problem (basic classification without requirements)
            self.logger.info('=== Phase 1: CLASSIFY ===')
            # Create minimal query analysis for classification (without step-tagged requirements)
            minimal_query_analysis = {
                'explicit_requirements': [],
                'dependencies': [],
                'answer_format': '',
                'cross_references': [],
            }
            problem_classification = self.problem_classifier.classify(
                problem, minimal_query_analysis
            )
            self.logger.info(
                f'Problem classified as: {problem_classification.get("primary_type")}'
            )

            # Step 2: PLAN - Generate execution strategy first (before query understanding)
            self.logger.info('=== Phase 2: PLAN ===')
            plan = self.planner.create_plan(
                problem, minimal_query_analysis, problem_classification
            )
            self.logger.info(f'Execution plan created with {len(plan)} subtasks')

            # Step 3: PARSE - Query Understanding (assign step numbers based on generated subtasks)
            self.logger.info('=== Phase 3: PARSE ===')
            query_analysis = self.query_understanding.analyze(
                problem, attachments, subtasks=plan
            )
            self.logger.info(
                f'Query analysis complete: {len(query_analysis.get("explicit_requirements", []))} '
                f'explicit requirements identified and assigned to steps'
            )
            # Log query analysis structure at DEBUG level only (not as final answer)
            self.logger.debug(
                f'Query analysis structure: {json.dumps(query_analysis, indent=2)[:500]}...'
            )

            # Step 3.5: Append tool calls to subtasks based on external requirements
            self.logger.info('=== Phase 3.5: APPEND TOOL CALLS ===')
            plan = self.planner.append_tool_calls_to_subtasks(
                plan, problem, query_analysis
            )
            self.logger.info(
                f'Tool calls appended to {len(plan)} subtasks based on requirements'
            )

            # Step 4: EXECUTE - Run tool operations
            self.logger.info('=== Phase 4: EXECUTE ===')
            execution_results = self.executor.execute_plan(
                plan, problem, attachments, query_analysis
            )

            # Include failed subtasks from state_manager that might not be in execution_results
            for subtask_id, subtask in self.state_manager.subtasks.items():
                if subtask.status == 'failed' and subtask_id not in execution_results:
                    # Include failed subtask in execution_results for completeness
                    error_msg = subtask.metadata.get('error', 'Unknown error')
                    error_type = subtask.metadata.get('error_type', 'unknown')
                    execution_results[subtask_id] = {
                        'error': error_msg,
                        'error_type': error_type,
                        'status': 'failed',
                        'subtask_id': subtask_id,
                    }
                    self.logger.info(
                        f'Included failed subtask {subtask_id} in execution_results'
                    )

            self.logger.info(
                f'Execution complete: {len(execution_results)} results obtained'
            )

            # Combine execution results for synthesis
            combined_results = {
                'execution_summary': execution_results,
            }

            # MONITOR & ADAPT (logging state)
            state_summary = self.state_manager.get_state_summary()
            self.logger.info(f'Current state summary: {state_summary}')

            # Step 5: SYNTHESIZE - Construct answer (with retry on validation failure)
            self.logger.info('=== Phase 5: SYNTHESIZE ===')
            max_retries = 2
            retry_count = 0
            synthesis = None
            final_answer = None

            while retry_count <= max_retries:
                # Log attachments count (including any downloaded files)
                if attachments and retry_count == 0:
                    self.logger.info(
                        f'Passing {len(attachments)} attachment(s) to synthesizer '
                        f'(filenames: {[a.filename for a in attachments]})'
                    )

                if retry_count > 0:
                    self.logger.info(
                        f'Retry attempt {retry_count}/{max_retries} due to validation failure'
                    )

                synthesis = self.answer_synthesizer.synthesize(
                    problem, combined_results, query_analysis, attachments
                )
                final_answer = synthesis.get(
                    'final_answer', 'Unable to determine answer'
                )
                validation_info = synthesis.get('validation_info', {})

                # Check validation results
                is_complete = validation_info.get('is_complete', True)
                data_quality = validation_info.get('data_quality', 'good')
                missing_requirements = validation_info.get('missing_requirements', [])
                warnings = validation_info.get('warnings', [])

                self.logger.info(
                    f'Validation: is_complete={is_complete}, data_quality={data_quality}, '
                    f'missing_requirements={len(missing_requirements)}, warnings={len(warnings)}'
                )

                # If validation passes or we've exhausted retries, break
                if is_complete and data_quality in ('good', 'fair'):
                    break

                # If validation fails and we have retries left, retry only failed subtasks
                if retry_count < max_retries:
                    self.logger.warning(
                        f'Validation failed: is_complete={is_complete}, data_quality={data_quality}. '
                        f'Retrying only failed subtasks...'
                    )

                    # Check if there are any failed subtasks
                    failed_subtasks = self.state_manager.get_failed_subtasks()

                    if failed_subtasks:
                        self.logger.info(
                            f'Found {len(failed_subtasks)} failed subtask(s) to retry: '
                            f'{[st.id for st in failed_subtasks]}'
                        )

                        # Retry only the failed subtasks
                        self.logger.info(
                            '=== Phase 4 (RETRY): EXECUTE FAILED SUBTASKS ==='
                        )
                        retry_results = self.executor.retry_failed_subtasks(
                            problem, attachments, query_analysis
                        )
                        self.logger.info(
                            f'Retry execution complete: {len(retry_results)} subtask(s) retried'
                        )

                        # Merge retry results with existing execution results
                        execution_results.update(retry_results)

                        combined_results = {
                            'execution_summary': execution_results,
                        }
                    else:
                        # No failed subtasks, but validation still failed
                        # This means we need to create additional subtasks for missing requirements
                        self.logger.info(
                            'No failed subtasks found. Creating additional subtasks for missing requirements...'
                        )

                        if missing_requirements:
                            self.logger.info(
                                f'Creating improved plan to address: {missing_requirements[:3]}...'
                            )

                            # Create improved plan with focus on missing requirements
                            improved_plan = self.planner.create_plan(
                                problem,
                                query_analysis,
                                problem_classification,
                                previous_plan=plan,
                                missing_requirements=missing_requirements or [],
                                validation_warnings=warnings or [],
                            )

                            if improved_plan:
                                self.logger.info(
                                    f'Improved plan created with {len(improved_plan)} subtasks'
                                )

                                # Append tool calls to improved plan
                                self.logger.info(
                                    '=== Phase 3.5 (RETRY): APPEND TOOL CALLS ==='
                                )
                                improved_plan = (
                                    self.planner.append_tool_calls_to_subtasks(
                                        improved_plan, problem, query_analysis
                                    )
                                )
                                self.logger.info(
                                    f'Tool calls appended to {len(improved_plan)} subtasks'
                                )

                                # Execute the new plan
                                self.logger.info(
                                    '=== Phase 4 (RETRY): EXECUTE NEW PLAN ==='
                                )
                                new_execution_results = self.executor.execute_plan(
                                    improved_plan, problem, attachments, query_analysis
                                )
                                self.logger.info(
                                    f'New plan execution complete: {len(new_execution_results)} results'
                                )

                                # Merge with existing results
                                execution_results.update(new_execution_results)

                                combined_results = {
                                    'execution_summary': execution_results,
                                }

                    retry_count += 1
                else:
                    self.logger.warning(
                        'Max retries reached. Proceeding with current results.'
                    )
                    break

            # Log final answer clearly (this is the actual answer, not query analysis)
            self.logger.info(f'FINAL ANSWER: {final_answer}')

            # Step 6: VALIDATE - Check if answer is correct
            self.logger.info('=== Phase 6: VALIDATE ===')
            validation_result = self.answer_validator.validate_answer(
                problem,
                final_answer,
                query_analysis,
                execution_results,
                combined_results,
                plan,
            )

            # Safely get is_correct with default to False if key is missing
            is_correct = validation_result.get('is_correct', False)

            if is_correct:
                self.logger.info('Answer validation passed. Answer is correct.')
                self.logger.info('Problem solved. Returning answer.')
                return final_answer
            else:
                # Answer is incorrect, identify and retry only incorrect execution steps
                self.logger.warning(
                    f'Answer validation failed: {validation_result.get("reason", "Unknown reason")}'
                )
                incorrect_subtask_ids = validation_result.get(
                    'incorrect_subtask_ids', []
                )

                if incorrect_subtask_ids:
                    self.logger.info(
                        f'Identified {len(incorrect_subtask_ids)} incorrect execution step(s): '
                        f'{incorrect_subtask_ids}'
                    )

                    # Retry only the incorrect execution steps
                    self.logger.info(
                        '=== Phase 4 (VALIDATION RETRY): EXECUTE INCORRECT SUBTASKS ==='
                    )
                    retry_results = self._retry_incorrect_subtasks(
                        incorrect_subtask_ids,
                        problem,
                        attachments,
                        query_analysis,
                    )
                    self.logger.info(
                        f'Retry execution complete: {len(retry_results)} subtask(s) retried'
                    )

                    # Merge retry results with existing execution results
                    execution_results.update(retry_results)

                    combined_results = {
                        'execution_summary': execution_results,
                    }

                    # Re-synthesize answer with corrected results
                    self.logger.info('=== Phase 5 (VALIDATION RETRY): SYNTHESIZE ===')
                    synthesis = self.answer_synthesizer.synthesize(
                        problem, combined_results, query_analysis, attachments
                    )
                    final_answer = synthesis.get(
                        'final_answer', 'Unable to determine answer'
                    )

                    self.logger.info(
                        f'FINAL ANSWER (after validation retry): {final_answer}'
                    )
                else:
                    self.logger.warning(
                        'Could not identify specific incorrect subtasks. '
                        'Proceeding with current answer.'
                    )

            self.logger.info('Problem solved. Returning answer.')
            return final_answer

        except Exception as e:
            self.logger.error(f'An error occurred during problem solving: {e}')
            import traceback

            self.logger.error(traceback.format_exc())
            return 'I encountered an error and could not solve the problem.'

    def _retry_incorrect_subtasks(
        self,
        incorrect_subtask_ids: List[str],
        problem: str,
        attachments: Optional[List[Attachment]],
        query_analysis: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Retry only the identified incorrect subtasks.

        Args:
            incorrect_subtask_ids: List of subtask IDs to retry.
            problem: Original problem description.
            attachments: Optional attachments.
            query_analysis: Optional query analysis results.

        Returns:
            Dictionary with retry execution results.
        """
        if not incorrect_subtask_ids:
            return {}

        # Get the subtasks to retry
        subtasks_to_retry = []
        for subtask_id in incorrect_subtask_ids:
            if subtask_id in self.state_manager.subtasks:
                subtask = self.state_manager.subtasks[subtask_id]
                # Reset subtask for retry
                self.state_manager.retry_subtask(subtask_id)
                subtasks_to_retry.append(subtask)
            else:
                self.logger.warning(f'Subtask ID not found: {subtask_id}')

        if not subtasks_to_retry:
            self.logger.warning('No valid subtasks to retry.')
            return {}

        self.logger.info(
            f'Retrying {len(subtasks_to_retry)} incorrect subtask(s): '
            f'{[st.id for st in subtasks_to_retry]}'
        )

        # Re-execute only the incorrect subtasks
        results = {}
        for subtask in subtasks_to_retry:
            subtask.status = 'in_progress'
            try:
                self.logger.info(
                    f'Retrying incorrect subtask: {subtask.id} - {subtask.description}'
                )
                result = self.executor.execute_subtask(
                    subtask, problem, attachments, query_analysis
                )
                results[subtask.id] = result
                self.logger.info(f'Successfully retried subtask: {subtask.id}')
            except Exception as e:
                self.logger.error(f'Failed to retry {subtask.id}: {e}', exc_info=True)
                # Task will remain in failed state
                continue

        return results
