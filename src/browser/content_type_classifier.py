"""Content Type Classifier for determining required content type for subtasks."""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..utils import extract_json_from_text

if TYPE_CHECKING:
    from ..llm import LLMService


class ContentTypeClassifier:
    """Classifies what type of content (web page or PDF) is needed for a subtask."""

    def __init__(self, llm_service: 'LLMService', logger: logging.Logger):
        """
        Initialize ContentTypeClassifier.

        Args:
            llm_service: LLM service for classification.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger

    def classify_required_content_type(
        self,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Determine what type of content is needed for the subtask.

        Args:
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            'pdf' if PDF is needed, 'web_page' if web page is needed, or 'either' if both are acceptable.
        """
        self.logger.info(
            f'Classifying required content type for subtask: {subtask_description[:100]}...'
        )

        system_prompt = """You are an expert at analyzing tasks and determining what type of content is needed.

Given a subtask description and problem context, determine whether the task requires:
- A PDF file (academic papers, research documents, downloadable documents)
- A web page (interactive pages, forms, dynamic content, information pages)
- Either (both types could work)

Consider:
- Keywords like "paper", "article", "publication", "download", "PDF" suggest PDF is needed
- Keywords like "navigate", "search", "form", "interactive", "page" suggest web page is needed
- Tasks involving reading academic papers, research documents, or downloadable files typically need PDFs
- Tasks involving searching or interacting with web interfaces typically need web pages
- If the task could be completed with either type, return "either"

Return JSON only with:
{"required_type": "pdf" | "web_page" | "either", "reasoning": "brief explanation"}"""

        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            if explicit_reqs:
                requirements_context = (
                    f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                )

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

What type of content is needed to complete this subtask?"""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            required_type = result_data.get('required_type', 'either')
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'Content type classification: {required_type}. Reasoning: {reasoning}'
            )

            return required_type

        except Exception as e:
            self.logger.warning(
                f'Failed to classify content type using LLM: {e}. Defaulting to "either".'
            )
            return 'either'
