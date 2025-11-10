"""Problem Classification System for categorizing problem types."""

import json
import logging
from typing import Any, Dict, Optional

from ..llm import LLMService
from ..utils import extract_json_from_text

PROBLEM_TYPES = [
    'Information Retrieval',
    'Logical Deduction',
    'Computational',
    'Cross-Reference',
    'Sequential Discovery',
    'Verification',
]


class ProblemClassifier:
    """Classifies problems into different types."""

    def __init__(self, llm_service: LLMService, logger: logging.Logger):
        """
        Initialize Problem Classifier.

        Args:
            llm_service: LLM service instance.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger

    def classify(
        self, problem: str, query_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a problem into one or more problem types with step-level analysis.

        Args:
            problem: The problem description.
            query_analysis: Optional query analysis from QueryUnderstanding.

        Returns:
            Dictionary containing:
            - primary_type: Primary problem type
            - secondary_types: List of secondary types
            - confidence: Confidence score (0-1)
            - reasoning: Explanation of classification
        """
        self.logger.info('Classifying problem type')

        problem_types_str = ', '.join(PROBLEM_TYPES)

        system_prompt = f"""You are an expert at classifying problem types.
Classify the given problem into one or more of these types:
{problem_types_str}

IMPORTANT CLASSIFICATION GUIDELINES:
- **Information Retrieval** should be PRIMARY when:
  * The problem requires specific counts, statistics, or data from a known source
  * The problem mentions a specific database, archive, or official website by name
  * The answer requires navigating to a website to extract structured data
  * Search alone is insufficient - direct website navigation is needed
  
- **Computational** should be PRIMARY when:
  * The problem is primarily about performing calculations or data processing
  * All required data is provided or easily estimated
  * The main challenge is mathematical/logical computation

- **Logical Deduction** should be PRIMARY when:
  * The problem can be solved through pure reasoning without external data retrieval
  * All information needed is provided in the problem statement
  * The task requires logical inference, pattern matching, or reasoning chains

- If a problem requires BOTH retrieving data from a specific source AND computation:
  * Classify as "Information Retrieval" (primary) with "Computational" as secondary
  * The retrieval task is typically more critical and challenging

Return your classification as a JSON object with:
- primary_type: string (one of the types above)
- secondary_types: list of strings (other applicable types)
- confidence: float between 0 and 1
- reasoning: string explaining the overall classification, specifically noting if the problem requires navigating to a specific database/archive source

Be precise and consider the problem's core characteristics. Break down complex problems into steps to identify which ones need search.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        context_info = ''
        if query_analysis:
            context_info = f'\nQuery Analysis: {json.dumps(query_analysis, indent=2)}'

        user_prompt = f"""Problem: {problem}{context_info}

Classify this problem and explain your reasoning."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Consistent problem classification
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            classification = json.loads(json_text)

            # Log step classifications if available
            # step_classifications = classification.get('step_classifications', [])
            # if step_classifications:
            #     search_steps = [
            #         s for s in step_classifications if s.get('requires_search', False)
            #     ]
            #     llm_only_steps = [
            #         s
            #         for s in step_classifications
            #         if not s.get('requires_search', True)
            #     ]
            #     self.logger.info(
            #         f'Problem classified as: {classification.get("primary_type")} '
            #         f'(confidence: {classification.get("confidence", 0):.2f})'
            #     )
            #     self.logger.info(
            #         f'Step breakdown: {len(search_steps)} step(s) require search, '
            #         f'{len(llm_only_steps)} step(s) LLM-only'
            #     )
            #     for step in step_classifications:
            #         search_indicator = (
            #             '🔍 SEARCH'
            #             if step.get('requires_search', False)
            #             else '🧠 LLM-ONLY'
            #         )
            #         self.logger.debug(
            #             f'  {search_indicator}: {step.get("step_description", "N/A")} '
            #             f'({step.get("step_type", "N/A")})'
            #         )
            # else:
            self.logger.info(
                f'Problem classified as: {classification.get("primary_type")} '
                f'(confidence: {classification.get("confidence", 0):.2f})'
            )

            return classification
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse classification response: {e}')
            return {
                'primary_type': 'Information Retrieval',
                'secondary_types': [],
                'confidence': 0.5,
                'reasoning': 'Fallback classification due to parsing error',
            }
        except Exception as e:
            self.logger.error(f'Problem classification failed: {e}', exc_info=True)
            raise
