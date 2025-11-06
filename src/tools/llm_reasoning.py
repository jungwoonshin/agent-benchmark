"""LLM reasoning functionality for calculations and problem solving."""

import json
import logging
from typing import List

from src.llm.llm_service import LLMService


class LLMReasoningTool:
    """Tool for performing calculations, data processing, and analysis using LLM reasoning."""

    def __init__(
        self, llm_service: LLMService, logger: logging.Logger, image_recognition=None
    ):
        """
        Initialize LLM reasoning tool.

        Args:
            llm_service: LLM service instance.
            logger: Logger instance.
            image_recognition: Optional image recognition tool for visual processing.
        """
        self.llm_service = llm_service
        self.logger = logger
        self.image_recognition = image_recognition

    def set_llm_service(self, llm_service):
        """Set the LLM service."""
        self.llm_service = llm_service

    def set_image_recognition(self, image_recognition):
        """Set the image recognition tool."""
        self.image_recognition = image_recognition

    def llm_reasoning(self, task_description: str, context: dict = None) -> str:
        """
        Performs calculations, data processing, and analysis using LLM reasoning.
        This replaces code_interpreter with LLM-based problem solving.

        Args:
            task_description: Description of what needs to be calculated/analyzed
            context: Optional context dictionary with variables and data available

        Returns:
            String representation of the reasoning result or answer
        """
        if self.llm_service is None:
            raise ValueError(
                'LLM service not set. Call set_llm_service() before using llm_reasoning.'
            )

        # Build context description
        context_str = ''
        if context:
            # Format context in a readable way
            context_items = []
            for key, value in context.items():
                if key == 'dependency_results':
                    # Handle dependency results specially
                    if isinstance(value, dict):
                        for dep_id, dep_result in value.items():
                            context_items.append(
                                f'{dep_id}: {json.dumps(dep_result, indent=2)[:500]}'
                            )
                elif isinstance(value, (dict, list)):
                    # Serialize complex structures
                    context_items.append(f'{key}: {json.dumps(value, indent=2)[:1000]}')
                else:
                    context_items.append(f'{key}: {str(value)[:500]}')
            if context_items:
                context_str = '\n\nAvailable Context:\n' + '\n'.join(
                    f'- {item}' for item in context_items
                )

        # Separate system and user prompts
        system_prompt = """You are an expert at performing calculations, data analysis, and problem solving.
Given a task description and available context, solve the problem step by step.

Your task:
1. Analyze the problem and identify what needs to be calculated or determined
2. Extract relevant data from the context
3. Perform calculations accurately (mathematical operations, percentages, averages, etc.)
4. Process and analyze data structures (lists, dictionaries, etc.)
5. Format conversions (dates, units, etc.)
6. Provide clear, step-by-step reasoning
7. Return the final answer or result

Be precise with numbers and calculations. Show your work when doing mathematical operations.
If the context contains data from previous steps, use that data in your calculations."""

        user_prompt = f"""Task: {task_description}{context_str}

Solve this task step by step. Show your reasoning and calculations clearly.
Extract and use any relevant data from the context provided.
Return your final answer or result."""

        try:
            self.logger.info(
                f'Performing LLM reasoning for task: {task_description}...'
            )
            # Use call_with_system_prompt() instead of call()
            result = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for consistent, accurate calculations
                max_tokens=8192,
            )
            self.logger.info(f'LLM reasoning completed: {len(result)} chars')
            return result
        except Exception as e:
            error_msg = f'LLM reasoning failed: {e}'
            self.logger.error(error_msg)
            return error_msg

    def llm_reasoning_with_images(
        self, task_description: str, context: dict = None, images: List[bytes] = None
    ) -> str:
        """
        Performs calculations, data processing, and analysis using visual LLM with image inputs.
        Uses GPT-4o or other vision model to process images along with text.

        Args:
            task_description: Description of what needs to be calculated/analyzed
            context: Optional context dictionary with variables and data available
            images: List of image data as bytes for visual processing

        Returns:
            String representation of the reasoning result or answer
        """
        if not self.image_recognition:
            raise ValueError(
                'Image recognition tool not initialized. Call set_image_recognition() first.'
            )

        if not images:
            # Fallback to regular LLM reasoning if no images
            return self.llm_reasoning(task_description, context)

        # Use image recognition tool for processing
        return self.image_recognition.recognize_images(
            images=images,
            task_description=task_description,
            context=context,
            source_type='general',
        )

    def code_interpreter(self, python_code: str, context: dict = None) -> str:
        """
        DEPRECATED: This method is disabled. Use llm_reasoning instead.
        Kept for backward compatibility but redirects to LLM reasoning.
        """
        self.logger.warning(
            'code_interpreter is deprecated. Redirecting to llm_reasoning.'
        )
        # Convert Python code to task description
        task_description = f'Execute the following Python code logic: {python_code}'
        return self.llm_reasoning(task_description, context)
