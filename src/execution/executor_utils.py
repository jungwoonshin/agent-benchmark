"""Utility functions for executor."""

import base64
import json
import logging
from typing import Any, Dict, List, Optional

from ..models import Attachment
from ..state import InformationStateManager, Subtask
from ..utils import extract_json_from_text


def extract_images_from_context(
    context: Dict[str, Any],
    attachments: Optional[List[Attachment]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[bytes]:
    """
    Extract image data from context and attachments for visual LLM processing.

    Args:
        context: Context dictionary that may contain image data.
        attachments: Optional list of attachments that may contain images.
        logger: Optional logger instance.

    Returns:
        List of image data as bytes.
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    images = []

    # Check attachments for image files
    if attachments:
        image_extensions = {
            '.png',
            '.jpg',
            '.jpeg',
            '.gif',
            '.bmp',
            '.webp',
            '.svg',
        }
        for attachment in attachments:
            filename_lower = attachment.filename.lower()
            if any(filename_lower.endswith(ext) for ext in image_extensions):
                images.append(attachment.data)
                logger.debug(f'Found image in attachment: {attachment.filename}')

    # Check context for image data (from browser screenshots, etc.)
    if isinstance(context, dict):
        # Check for screenshot data in context
        if 'screenshot' in context:
            screenshot_data = context.get('screenshot')
            if isinstance(screenshot_data, bytes):
                images.append(screenshot_data)
                logger.debug('Found screenshot in context')
            elif isinstance(screenshot_data, str):
                # Try to decode base64 string
                try:
                    decoded = base64.b64decode(screenshot_data)
                    images.append(decoded)
                    logger.debug('Found base64-encoded screenshot in context')
                except Exception:
                    pass

        # Check for image data in nested structures
        for key, value in context.items():
            if key == 'screenshot' or 'image' in key.lower():
                if isinstance(value, bytes):
                    images.append(value)
                elif isinstance(value, str):
                    try:
                        decoded = base64.b64decode(value)
                        images.append(decoded)
                    except Exception:
                        pass

    return images


def determine_tool_parameters(
    subtask: Subtask,
    problem: str,
    llm_service: Any,
    state_manager: InformationStateManager,
    attachments: Optional[List[Attachment]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Use LLM to determine tool parameters for a subtask.

    Args:
        subtask: Subtask to execute.
        problem: Original problem.
        llm_service: LLM service instance.
        state_manager: State manager instance.
        attachments: Optional attachments.
        logger: Optional logger instance.

    Returns:
        Dictionary of tool parameters.
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    logger.debug(f'Determining parameters for subtask: {subtask.id}')

    system_prompt = """You are an expert at determining tool parameters for task execution.
Given a subtask description, determine the appropriate parameters for the tool.

CRITICAL CONSTRAINTS:
- **llm_reasoning**: Use for calculations, data processing, and analysis. Provide task_description instead of code.
- **Use search for information gathering**: When you need to find PDFs, documents, or information, generate a specific search query instead of reasoning
- **PDF processing**: Use search to locate PDFs, then read_attachment to extract text. Never try to parse PDFs with reasoning.
- **Search query priority**: If search_queries are provided in the subtask information, you MUST use all of them. The system will try all queries and combine results. Do not modify or replace them.

Return a JSON object with parameters specific to the tool type.
For llm_reasoning: {"task_description": "description of what needs to be calculated/analyzed", "context": {...}}

CRITICAL TASK DESCRIPTION RULES FOR LLM_REASONING:
- **Clear task description**: Provide a clear, detailed description of what needs to be calculated or analyzed
  - Example: "Calculate the average of the numbers in the context" instead of "average()"
  - Example: "Extract all dates from the text and convert them to YYYY-MM-DD format" instead of "datetime conversion"
- **Reference context data**: Explicitly mention which context variables contain the data you need
  - Example: "Using the data from step_1, calculate the total and divide by the count"
- **Be specific**: Include details about data formats, units, and expected output format

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.
  - **DEPENDENCY RESULTS**: If this subtask has dependencies, the results from dependency subtasks will be automatically added to the context. For llm_reasoning, describe in the task_description which dependency results to use (e.g., "Use data from step_1 and step_2 to calculate...").
  - **SEARCH RESULTS**: When task_description references search results from dependencies, describe what information to extract from them clearly.
For search: {"query": "search query", "num_results": 5}
  - Generate specific, detailed search queries that include key terms, dates, and requirements
  - Example: Instead of "AI papers", use "arXiv AI regulation papers submitted June 2022"
For browser_navigate: {"url": "https://...", "action": "click_link|extract_text|find_table|search_text|extract_count|extract_statistics", "link_text": "...", "selector": "...", "extraction_query": "..."}
  - url (REQUIRED): Must be a valid URL with scheme (e.g., "https://www.example.com"). This is MANDATORY - you MUST provide a valid URL.
  - action (optional):
    * "click_link": Click a link on the page (requires link_text)
    * "extract_text": Extract text content using CSS selector
    * "find_table": Extract structured data from HTML tables
    * "search_text": Search for specific text terms (requires selector with comma-separated terms)
    * "extract_count" or "extract_statistics": Extract numeric counts/statistics with context awareness (RECOMMENDED when you need to find counts like "number of articles", "total items", etc.)
      - extraction_query (recommended): Clear description of what to extract (e.g., "total number of Nature articles published in 2020"). This enables LLM-based extraction which is more accurate.
      - selector (optional): Comma-separated context keywords to help find relevant numbers (e.g., "article,count,total" for finding article counts)
  - link_text (optional): Text of link to click (required if action="click_link")
  - selector (optional):
    * For "extract_text": CSS selector to target specific elements
    * For "search_text": Comma-separated search terms
    * For "extract_count"/"extract_statistics": Comma-separated context keywords (e.g., "article,count,total")
  - extraction_query (optional, recommended for extract_count/extract_statistics): Clear description of what information to extract from the page

CRITICAL: For browser_navigate, the "url" field is REQUIRED and must be a valid URL with scheme (http:// or https://). If the subtask doesn't specify a URL directly, you must infer or construct a reasonable URL based on the subtask description (e.g., if it mentions "Nature articles 2020", use "https://www.nature.com/nature/articles?year=2020").

For read_attachment: {"attachment_index": 0, "options": {...}}
For analyze_media: {"attachment_index": 0, "analysis_type": "auto"}"""

    attachment_info = ''
    if attachments:
        attachment_info = (
            f'\nAvailable attachments: {[a.filename for a in attachments]}'
        )

    # Get LLM-generated search_queries from metadata if available (prefer new format, fallback to old)
    search_queries = subtask.metadata.get('search_queries', [])
    if not search_queries:
        # Handle backward compatibility: if search_query (singular) exists, convert to array
        old_search_query = subtask.metadata.get('search_query', '')
        if old_search_query:
            search_queries = [old_search_query]

    search_query_info = ''
    if search_queries and subtask.metadata.get('tool') == 'search':
        if len(search_queries) == 1:
            search_query_info = (
                f'\nLLM-generated search_query (MUST USE): {search_queries[0]}'
            )
        else:
            queries_list = ', '.join([f'"{q}"' for q in search_queries[:3]])
            search_query_info = f'\nLLM-generated search_queries (MUST USE ALL {len(search_queries)}): [{queries_list}]'

    # Collect dependency results information for the prompt
    dependency_info = ''
    if subtask.dependencies:
        dependency_results_summary = []
        for dep_id in subtask.dependencies:
            if dep_id in state_manager.subtasks:
                dep_subtask = state_manager.subtasks[dep_id]
                if dep_subtask.status == 'completed':
                    # Include full result (no truncation) for complete context
                    result_text = (
                        str(dep_subtask.result) if dep_subtask.result else 'None'
                    )
                    dependency_results_summary.append(f'  - {dep_id}: {result_text}')
        if dependency_results_summary:
            dependency_info = (
                f'\n\nDependency Results (available in context):\n'
                f'{"".join(dependency_results_summary)}\n'
                f'Note: These results will be automatically added to the context parameter for llm_reasoning.'
            )

    user_prompt = f"""Problem: {problem}

Subtask: {subtask.description}
Tool: {subtask.metadata.get('tool', 'unknown')}{search_query_info}{attachment_info}{dependency_info}

Determine the appropriate tool parameters.
IMPORTANT: If the tool is 'search' and search_queries are provided above, the system will automatically try all of them. You do not need to specify them in parameters.
If the tool is 'llm_reasoning' and dependencies are listed above, describe in the task_description which dependency results to use."""

    try:
        response = llm_service.call_with_system_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,  # Creative but focused for parameter generation
            response_format={'type': 'json_object'},
        )
        json_text = extract_json_from_text(response)
        return json.loads(json_text)
    except Exception as e:
        logger.error(f'Parameter determination failed: {e}')
        return {}
