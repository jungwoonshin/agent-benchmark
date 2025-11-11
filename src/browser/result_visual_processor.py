"""Visual analysis processing module for search results.

Handles visual analysis requirements and processing for PDFs and web pages.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..utils import extract_json_from_text

if TYPE_CHECKING:
    from src.llm import LLMService
    from src.tools import ToolBelt


class ResultVisualProcessor:
    """Processes visual analysis for search results."""

    def __init__(
        self,
        llm_service: 'LLMService',
        tool_belt: 'ToolBelt',
        browser: Any,  # Browser instance
        logger: logging.Logger,
    ):
        """Initialize ResultVisualProcessor."""
        self.llm_service = llm_service
        self.tool_belt = tool_belt
        self.browser = browser
        self.logger = logger

    def requires_visual_analysis(
        self,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if the subtask requires visual analysis using LLM.

        Args:
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            True if visual analysis is required, False otherwise.
        """
        try:
            # Extract step number from subtask_description
            step_num = None
            step_match = re.search(
                r'step[_\s]*(\d+)', subtask_description, re.IGNORECASE
            )
            if step_match:
                try:
                    step_num = int(step_match.group(1))
                except (ValueError, IndexError):
                    pass

            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                if explicit_reqs:
                    if step_num is not None:
                        filtered_reqs = []
                        for req in explicit_reqs:
                            req_str = str(req)
                            req_step_match = re.search(
                                r'step[_\s]*(\d+)[:\s]', req_str, re.IGNORECASE
                            )
                            if req_step_match:
                                try:
                                    req_step_num = int(req_step_match.group(1))
                                    if req_step_num == step_num:
                                        filtered_reqs.append(req)
                                except (ValueError, IndexError):
                                    filtered_reqs.append(req)
                        if filtered_reqs:
                            requirements_context += (
                                f'\nExplicit Requirements: {", ".join(filtered_reqs)}'
                            )
                    else:
                        requirements_context += (
                            f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                        )

            system_prompt = """You are an expert at analyzing subtasks to determine if they require visual analysis (image recognition, screenshot analysis, or visual content processing).

Consider the subtask description and requirements when making your determination.

A subtask requires visual analysis if it:
- Explicitly mentions analyzing images, screenshots, photos, diagrams, charts, graphs, or visual content
- Asks to identify, recognize, or extract information from visual elements
- Requires understanding visual layouts, UI elements, or visual patterns
- Mentions visual inspection, image processing, or visual data extraction
- Asks about visual characteristics, colors, shapes, or visual relationships
- The requirements indicate that visual information (figures, charts, diagrams) is needed to answer the question

A subtask does NOT require visual analysis if it:
- Only asks for text-based information extraction
- Only requires reading text content from web pages or documents
- Only asks for calculations or data processing without visual input
- Is about searching, navigating, or downloading without visual analysis needs

Return a JSON object with:
- requires_visual: boolean indicating if visual analysis is needed
- reasoning: brief explanation (1-2 sentences) of why or why not

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Subtask: {subtask_description}
{requirements_context}

Does this subtask require visual analysis (image recognition, screenshot analysis, or visual content processing)?"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            requires_visual = result_data.get('requires_visual', False)
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.debug(
                f'Visual analysis requirement check: {requires_visual}. Reasoning: {reasoning}'
            )

            return requires_visual

        except Exception as e:
            self.logger.warning(
                f'Failed to determine visual analysis requirement using LLM: {e}. Defaulting to False.'
            )
            return False

    def process_visual_analysis(
        self,
        is_file: bool,
        extracted_data: Dict[str, Any],
        attachments: List[Any],  # List[Attachment]
        result: Any,  # SearchResult
        subtask_description: str,
        problem: str,
        full_content: str,
        summarized_content: str,
    ) -> Dict[str, Any]:
        """
        Process visual analysis if required.

        Args:
            is_file: Whether the result is a file.
            extracted_data: Extracted data dictionary.
            attachments: List of attachments.
            result: SearchResult object.
            subtask_description: Subtask description.
            problem: Problem description.
            full_content: Full content text.
            summarized_content: Summarized content text.

        Returns:
            Updated extracted_data with image_analysis if applicable.
        """
        if (
            not hasattr(self.tool_belt, 'image_recognition')
            or not self.tool_belt.image_recognition
        ):
            self.logger.warning(
                'Visual analysis required but image_recognition tool not available'
            )
            return extracted_data

        try:
            if (
                is_file
                and isinstance(extracted_data, dict)
                and extracted_data.get('type') == 'pdf'
            ):
                # Check if image analysis was already done
                existing_image_analysis = extracted_data.get('image_analysis', '')
                if existing_image_analysis:
                    self.logger.info(
                        'Image analysis already completed before relevance check. Skipping duplicate processing.'
                    )
                    return extracted_data

                # Process PDF images
                extracted_images = extracted_data.get('extracted_images', [])
                if extracted_images:
                    attachment = next(
                        (
                            a
                            for a in attachments
                            if a.filename == extracted_data.get('filename', '')
                        ),
                        None,
                    )
                    if attachment:
                        image_analysis = self.tool_belt.image_recognition.process_pdf_images_after_relevance(
                            attachment,
                            extracted_images,
                            problem=problem,
                            context_text=full_content or '',
                        )
                        if image_analysis:
                            extracted_data['image_analysis'] = image_analysis
                            self.logger.info(
                                f'Processed {len(extracted_images)} image(s) from PDF with visual LLM'
                            )

            elif not is_file:
                # Process web page screenshot
                screenshot = self.browser.take_screenshot(as_base64=False)
                if screenshot:
                    task_desc = f'Analyze this webpage screenshot and extract relevant information for: {subtask_description}'
                    image_analysis = (
                        self.tool_belt.image_recognition.recognize_images_from_browser(
                            screenshot,
                            context={
                                'url': result.url,
                                'title': result.title,
                                'text': summarized_content[:1000]
                                if summarized_content
                                else '',
                            },
                            task_description=task_desc,
                        )
                    )
                    if image_analysis:
                        if isinstance(extracted_data, dict):
                            extracted_data['image_analysis'] = image_analysis
                        else:
                            extracted_data = {
                                'content': extracted_data,
                                'image_analysis': image_analysis,
                            }
                        self.logger.info('Processed webpage screenshot with visual LLM')

        except Exception as e:
            self.logger.warning(f'Failed to process visual analysis: {e}')

        return extracted_data
