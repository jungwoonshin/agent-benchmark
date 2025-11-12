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
Given a subtask description and previous step results, determine the appropriate parameters for the tool.

CRITICAL CONSTRAINTS:
- **llm_reasoning**: Use for calculations, data processing, and analysis. Provide task_description instead of code.
- **Use search for information gathering**: When you need to find PDFs, documents, or information, generate a specific search query instead of reasoning
- **PDF processing**: Use search to locate PDFs, then read_attachment to extract text. Never try to parse PDFs with reasoning.
- **Search query priority**: If search_queries are provided in the subtask information, you MUST use all of them. The system will try all queries and combine results. Do not modify or replace them.
- **PREVIOUS STEP RESULTS**: ALWAYS check previous step results provided in the prompt. Extract relevant information (titles, IDs, URLs, data, etc.) from previous step results and use them in the current step's parameters. Use information from previous steps to populate API parameters, refine search queries, or provide context for reasoning tasks.

Return a JSON object with parameters specific to the tool type.
For llm_reasoning: {"task_description": "description of what needs to be calculated/analyzed", "context": {...}}

CRITICAL TASK DESCRIPTION RULES FOR LLM_REASONING:
- **Clear task description**: Provide a clear, detailed description of what needs to be calculated or analyzed, avoiding code-like syntax
- **Reference context data**: Explicitly mention which context variables contain the data you need
- **Be specific**: Include details about data formats, units, and expected output format
- **Use previous step results**: Reference specific information from previous step results in the task_description

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.
  - **PREVIOUS STEP RESULTS**: Previous step execution results will be provided in the prompt. ALWAYS examine them and extract relevant information to use in parameters. For API tools, extract titles, IDs, URLs, etc. from previous results. For llm_reasoning, describe which previous step results to use in the task_description.
  - **SEARCH RESULTS**: When task_description references search results from previous steps, describe what information to extract from them clearly.
For search: {"query": "search query", "num_results": 5}
  - Generate search queries using general, broad keywords that include key terms, dates, and requirements
  - Use core concepts, entities, and topics rather than overly specific measurement or quantification terms
For browser_navigate: {"url": "https://...", "action": "click_link|extract_text|find_table|search_text|extract_count|extract_statistics", "link_text": "...", "selector": "...", "extraction_query": "..."}
  - url (REQUIRED): Must be a valid URL with scheme (e.g., "https://www.example.com"). This is MANDATORY - you MUST provide a valid URL.
  - action (optional):
    * "click_link": Click a link on the page (requires link_text)
    * "extract_text": Extract text content using CSS selector
    * "find_table": Extract structured data from HTML tables
    * "search_text": Search for specific text terms (requires selector with comma-separated terms)
    * "extract_count" or "extract_statistics": Extract numeric counts/statistics with context awareness (RECOMMENDED when you need to find counts or totals)
      - extraction_query (recommended): Clear description of what to extract. This enables LLM-based extraction which is more accurate.
      - selector (optional): Comma-separated context keywords to help find relevant numbers
  - link_text (optional): Text of link to click (required if action="click_link")
  - selector (optional):
    * For "extract_text": CSS selector to target specific elements
    * For "search_text": Comma-separated search terms
    * For "extract_count"/"extract_statistics": Comma-separated context keywords to help locate relevant numeric data
  - extraction_query (optional, recommended for extract_count/extract_statistics): Clear description of what information to extract from the page

CRITICAL: For browser_navigate, the "url" field is REQUIRED and must be a valid URL with scheme (http:// or https://). If the subtask doesn't specify a URL directly, you must infer or construct a reasonable URL based on the subtask description.

For read_attachment: {"attachment_index": 0, "options": {...}}
For analyze_media: {"attachment_index": 0, "analysis_type": "auto"}
For API tools (github_api, wikipedia_api, youtube_api, twitter_api, reddit_api, arxiv_api, wayback_api, google_maps_api):
  - Single API call: {{"method": "method_name", ...method_specific_parameters}}
  - github_api methods:
    * search_issues: {"method": "search_issues", "repo": "owner/repo" (required), "state": "all|open|closed" (optional, default: "all"), "labels": ["label1"] (optional, list), "sort": "created|updated|comments" (optional, default: "created"), "order": "asc|desc" (optional, default: "asc"), "per_page": 100 (optional, default: 100, max: 100)}
    * get_issue: {"method": "get_issue", "repo": "owner/repo" (required), "issue_number": 123 (required, integer)}
    * get_repository_commit: {"method": "get_repository_commit", "repo": "owner/repo" (required), "ref": "commit_sha" (required, string)}
    * get_repository_contents: {"method": "get_repository_contents", "repo": "owner/repo" (required), "path": "file/path" (optional, default: ""), "ref": "branch_name" (optional, string)}
  - wikipedia_api methods:
    * get_page: {"method": "get_page", "title": "Page Title" (required, string), "revision_id": 123 (optional, integer)}
      - **CRITICAL**: If the subtask description mentions a specific year/version (e.g., "2022 version", "as of 2022"), use LIST format with chained calls:
        [{"function": "get_page_revisions", "parameters": {"title": "Page Title", "start_date": "2022-01-01", "end_date": "2022-12-31", "limit": 1}}, {"function": "get_page", "parameters": {"title": "Page Title", "revision_id": "<from previous call>"}}]
      - The system will automatically extract revision_id from the first call and use it in the second call
    * search_pages: {"method": "search_pages", "query": "search terms" (required), "limit": 10 (optional, default: 10)}
    * get_page_revisions: {"method": "get_page_revisions", "title": "Page Title" (required), "start_date": "YYYY-MM-DD" (optional), "end_date": "YYYY-MM-DD" (optional), "limit": 500 (optional, default: 500)}
      - Use this to find revisions from a specific year when the problem requires a historical version
      - For "2022 version": Use start_date="2022-01-01", end_date="2022-12-31" to get revisions from 2022
  - youtube_api methods:
    * get_video_info: {"method": "get_video_info", "video_id": "video_id" (required, string)}
    * search_videos: {"method": "search_videos", "query": "search terms" (required), "max_results": 10 (optional, default: 10, max: 50)}
  - twitter_api methods:
    * get_user_tweets: {"method": "get_user_tweets", "username": "username" (required, without @), "max_results": 10 (optional, default: 10, max: 100), "start_time": "YYYY-MM-DDTHH:MM:SSZ" (optional), "end_time": "YYYY-MM-DDTHH:MM:SSZ" (optional)}
  - reddit_api methods:
    * get_user_posts: {"method": "get_user_posts", "username": "username" (required), "limit": 25 (optional, default: 25, max: 100)}
    * search_posts: {"method": "search_posts", "subreddit": "subreddit_name" (required, without r/), "query": "search terms" (required), "limit": 25 (optional, default: 25, max: 100), "sort": "relevance|hot|top|new" (optional, default: "relevance")}
  - arxiv_api methods:
    * get_metadata: {"method": "get_metadata", "paper_id": "YYMM.NNNNN" or "archive/category/YYMMNNN" (required), "download_pdf": false (optional, boolean, default: false)}
  - wayback_api methods:
    * get_archived_url: {"method": "get_archived_url", "url": "https://example.com" (required), "timestamp": "YYYYMMDD" (optional, string)}
  - google_maps_api methods:
    * get_place_details: {"method": "get_place_details", "place_id": "place_id" (required, string)}
    * get_street_view_image: {"method": "get_street_view_image", "location": "address or coordinates" (required), "size": "600x400" (optional, default: "600x400", format: "WIDTHxHEIGHT"), "heading": 0 (optional, integer 0-360), "pitch": 0 (optional, integer -90 to 90), "fov": 90 (optional, integer, default: 90)}

  - Multiple chained API calls: When a subtask requires multiple API calls where later calls depend on results from earlier calls, use LIST format: [{{"function": "method1", "parameters": {{...}}}}, {{"function": "method2", "parameters": {{...}}}}]
    * Use chained calls when:
      - The subtask description mentions a specific year, version, revision, or date requirement (e.g., "2022 version", "latest 2022 revision", "as of 2022", "version from YYYY")
      - The subtask requires retrieving data in multiple steps
      - One API call's output is required as input for another
      - Historical or versioned data is needed
    * For year/version requirements: First call should retrieve the revision/version ID for the specified year, then use that ID in the second call
    * The system will automatically extract values from previous API call results and use them in subsequent calls
    * Use placeholder "<from previous call>" in parameters to reference results from earlier calls in the chain
    * Extract the appropriate API name, method names, and required parameters from the subtask description and available API documentation
  """
  
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

    # Collect ALL previous step results (dependencies + all previous steps)
    previous_results_info = ''
    previous_results_summary = []

    # Collect direct dependencies
    if subtask.dependencies:
        for dep_id in subtask.dependencies:
            if dep_id in state_manager.subtasks:
                dep_subtask = state_manager.subtasks[dep_id]
                if dep_subtask.status == 'completed' and dep_subtask.result is not None:
                    result_text = str(dep_subtask.result)
                    previous_results_summary.append(f'  - {dep_id}: {result_text}')

    # Collect all previous steps (sequential steps before current)
    try:
        current_step_num = int(subtask.id.split('_')[1]) if '_' in subtask.id else None
    except (ValueError, IndexError):
        current_step_num = None

    if current_step_num is not None:
        for step_num in range(1, current_step_num):
            prev_step_id = f'step_{step_num}'
            if prev_step_id in state_manager.subtasks:
                prev_subtask = state_manager.subtasks[prev_step_id]
                if (
                    prev_subtask.status == 'completed'
                    and prev_subtask.result is not None
                    and prev_step_id not in (subtask.dependencies or [])
                ):
                    result_text = str(prev_subtask.result)
                    previous_results_summary.append(
                        f'  - {prev_step_id}: {result_text}'
                    )

    if previous_results_summary:
        previous_results_info = (
            f'\n\nPrevious Step Results (use these to determine parameters):\n'
            f'{"".join(previous_results_summary)}\n'
            f'IMPORTANT: When determining parameters, reference and use information from these previous step results. '
            f'For example, if a previous step found a page title, use it in API parameters. '
            f'If a previous step found search results, extract relevant information from them for the current step.'
        )

    user_prompt = f"""Problem: {problem}

Subtask: {subtask.description}
Tool: {subtask.metadata.get('tool', 'unknown')}{search_query_info}{attachment_info}{previous_results_info}

Determine the appropriate tool parameters.
IMPORTANT: 
- If the tool is 'search' and search_queries are provided above, the system will automatically try all of them. You do not need to specify them in parameters.
- If the tool is 'llm_reasoning' and previous step results are listed above, describe in the task_description which previous step results to use.
- If the tool is an API tool (ends with '_api'), you MUST include the "method" parameter and all required parameters for that method. 
  **CRITICAL**: Extract information from previous step results to determine API parameters. Reference the previous step results above to find the exact values needed (titles, IDs, URLs, identifiers, etc.) and use them in the appropriate API parameters.
  **CRITICAL FOR YEAR/VERSION REQUIREMENTS**: If the subtask description mentions a specific year, version, revision, or date (e.g., "2022 version", "latest 2022 revision", "as of 2022"), you MUST use chained API calls. First, call the method that retrieves revisions/versions for that year to get the revision_id, then call the method that retrieves the page/content using that revision_id. Use LIST format with placeholder "<from previous call>" for the revision_id parameter.
- If the tool is 'search', you may use information from previous steps to refine search queries, but the system will use the provided search_queries if available.
- For API tools requiring chained calls (when later calls depend on earlier call results), use LIST format with multiple function calls. Use placeholder "<from previous call>" to reference results from earlier calls in the chain. The system will automatically extract and use values from previous API call results."""

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
