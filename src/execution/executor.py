"""Execution Engine for orchestrating tool operations."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..browser.search_result_processor import SearchResultProcessor
from ..llm import LLMService
from ..models import Attachment
from ..state import InformationStateManager, Subtask
from ..tools import ToolBelt
from ..utils.json_utils import extract_json_from_text
from .code_executor import CodeExecutor
from .result_analyzer import ExecutionResultAnalyzer
from .search_handler import SearchHandler


class Executor:
    """Orchestrates tool execution based on execution plan."""

    def __init__(
        self,
        tool_belt: ToolBelt,
        llm_service: LLMService,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize Executor.

        Args:
            tool_belt: ToolBelt instance with available tools.
            llm_service: LLM service for decision-making.
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.tool_belt = tool_belt
        self.llm_service = llm_service
        self.state_manager = state_manager
        self.logger = logger

        # Initialize browser for search result processor
        from ..browser import Browser

        browser = Browser(logger=logger, headless=True)
        # Initialize search result processor for systematic search result handling
        self.search_processor = SearchResultProcessor(
            llm_service=llm_service,
            browser=browser,
            tool_belt=tool_belt,
            logger=logger,
        )

        # Initialize helper classes
        self.code_executor = CodeExecutor(
            tool_belt=tool_belt,
            llm_service=llm_service,
            state_manager=state_manager,
            logger=logger,
        )
        self.search_handler = SearchHandler(
            tool_belt=tool_belt,
            llm_service=llm_service,
            logger=logger,
        )

    def execute_subtask(
        self,
        subtask: Subtask,
        problem: str,
        attachments: Optional[List[Attachment]] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a single subtask using appropriate tool.

        Args:
            subtask: Subtask to execute.
            problem: Original problem description.
            attachments: Optional attachments.
            query_analysis: Optional query analysis results containing requirements and constraints.

        Returns:
            Result of the execution.
        """
        # === BEFORE SUBTASK EXECUTION ===
        state_before = self.state_manager.get_state_summary()
        self.logger.info('=' * 80)
        self.logger.info(f'BEFORE executing subtask: {subtask.id}')
        self.logger.info(f'  Description: {subtask.description}')
        self.logger.info(f'  Status: {subtask.status}')
        self.logger.info(f'  Dependencies: {subtask.dependencies}')
        self.logger.info(
            f'  Dependencies satisfied: {all(dep_id in self.state_manager.completed_subtasks for dep_id in subtask.dependencies)}'
        )
        self.logger.info(f'  Tool: {subtask.metadata.get("tool", "unknown")}')
        self.logger.info(
            f'  Metadata: {json.dumps(subtask.metadata, default=str, indent=2)}'
        )
        self.logger.info(f'  State before: {state_before}')
        self.logger.info(
            f'  Knowledge facts count: {len(self.state_manager.knowledge_graph)}'
        )

        tool_name = subtask.metadata.get('tool', 'unknown')
        parameters = subtask.metadata.get('parameters', {})

        # Use LLM to determine tool parameters if not provided
        if not parameters or tool_name == 'unknown':
            self.logger.debug(
                f'Determining parameters for subtask {subtask.id} (parameters not provided)'
            )
            parameters = self._determine_tool_parameters(subtask, problem, attachments)
            self.logger.debug(
                f'Determined parameters: {json.dumps(parameters, default=str)[:200]}...'
            )

        self.logger.info(
            f'  Parameters: {json.dumps(parameters, default=str)[:300]}...'
        )
        self.logger.info('-' * 80)

        try:
            # IMPORTANT: For most subtasks, always search first, then process results
            # Only skip search for: llm_reasoning, read_attachment, analyze_media
            should_search_first = tool_name not in [
                'llm_reasoning',
                'code_interpreter',  # Backward compatibility
                'read_attachment',
                'analyze_media',
            ]

            # If tool is browser_navigate or unknown, convert to search
            if should_search_first and tool_name != 'search':
                # Get LLM-generated search_queries from metadata (prefer new format, fallback to old)
                search_queries = subtask.metadata.get('search_queries', [])
                if not search_queries:
                    # Handle backward compatibility: if search_query (singular) exists, convert to array
                    old_search_query = subtask.metadata.get('search_query', '')
                    if old_search_query:
                        search_queries = [old_search_query]
                    else:
                        # Fallback to description if no search queries provided
                        self.logger.warning(
                            f'Subtask {subtask.id} missing search_queries in metadata. '
                            f'Using description as fallback.'
                        )
                        search_queries = [subtask.description]

                # Use first query for backward compatibility with parameters
                search_query = (
                    search_queries[0] if search_queries else subtask.description
                )
                self.logger.info(
                    f'Converting tool "{tool_name}" to search-first workflow. '
                    f'Will search with {len(search_queries)} queries, first query: "{search_query}"'
                )
                tool_name = 'search'
                # Use LLM-generated search_queries from metadata
                if 'query' not in parameters:
                    parameters['query'] = (
                        search_query  # Store first query for backward compatibility
                    )

            # Execute based on tool type
            if tool_name == 'code_interpreter' or tool_name == 'llm_reasoning':
                # Support both old 'code_interpreter' and new 'llm_reasoning' tool names
                code = parameters.get('code', '')
                task_description = parameters.get('task_description', '')
                context = parameters.get('context', {})

                # Convert code to task description if needed
                if code and not task_description:
                    task_description = (
                        f'Execute the following Python code logic: {code}'
                    )
                elif not task_description:
                    # Fallback to subtask description
                    task_description = subtask.description

                # Collect results from ALL previous completed subtasks (not just direct dependencies)
                # This ensures step N has access to materials from steps 1..N-1
                previous_results = {}
                missing_dependencies = []

                # First, check direct dependencies (required)
                if subtask.dependencies:
                    for dep_id in subtask.dependencies:
                        if dep_id in self.state_manager.subtasks:
                            dep_subtask = self.state_manager.subtasks[dep_id]
                            if (
                                dep_subtask.status == 'completed'
                                and dep_subtask.result is not None
                            ):
                                result = dep_subtask.result
                                serialized_result = self._serialize_result_for_code(
                                    result
                                )
                                previous_results[dep_id] = serialized_result
                            elif dep_subtask.status == 'failed':
                                self.logger.warning(
                                    f'Dependency {dep_id} failed - not adding to context'
                                )
                                missing_dependencies.append(dep_id)
                            else:
                                self.logger.warning(
                                    f'Dependency {dep_id} has status "{dep_subtask.status}" - not adding to context'
                                )
                                missing_dependencies.append(dep_id)
                        else:
                            self.logger.warning(
                                f'Dependency {dep_id} not found in state manager'
                            )
                            missing_dependencies.append(dep_id)

                    # Check if we have missing dependencies (fail if required dependencies are missing)
                    if missing_dependencies:
                        error_msg = (
                            f'Cannot execute LLM reasoning - missing or incomplete dependencies: '
                            f'{", ".join(missing_dependencies)}. '
                            f'Available dependencies: {", ".join(previous_results.keys()) if previous_results else "none"}'
                        )
                        self.logger.error(error_msg)
                        # Mark subtask as failed
                        self.state_manager.fail_subtask(subtask.id, error_msg)
                        subtask.metadata['error'] = error_msg
                        subtask.metadata['error_type'] = 'missing_dependencies'
                        # Return structured error dict instead of error string
                        return {
                            'error': error_msg,
                            'error_type': 'missing_dependencies',
                            'status': 'failed',
                            'subtask_id': subtask.id,
                        }

                # Add ALL other completed subtasks (for context, not just dependencies)
                # Extract step number from current subtask ID (e.g., "step_3" -> 3)
                try:
                    current_step_num = (
                        int(subtask.id.split('_')[1]) if '_' in subtask.id else None
                    )
                except (ValueError, IndexError):
                    current_step_num = None

                if current_step_num is not None:
                    # Add all previous steps (step_1, step_2, ..., step_{current_step_num-1})
                    for step_num in range(1, current_step_num):
                        prev_step_id = f'step_{step_num}'
                        if prev_step_id in self.state_manager.subtasks:
                            prev_subtask = self.state_manager.subtasks[prev_step_id]
                            if (
                                prev_subtask.status == 'completed'
                                and prev_subtask.result is not None
                                and prev_step_id
                                not in previous_results  # Don't duplicate
                            ):
                                result = prev_subtask.result
                                serialized_result = self._serialize_result_for_code(
                                    result
                                )
                                previous_results[prev_step_id] = serialized_result
                                self.logger.debug(
                                    f'Added previous step {prev_step_id} to context'
                                )

                # Build structured context with materials from each previous step
                for step_id, step_result in previous_results.items():
                    # Extract materials if available
                    materials = []
                    if isinstance(step_result, dict):
                        # Check if result has materials array
                        if 'materials' in step_result:
                            materials = step_result['materials']
                        # Also check for materials in nested structures
                        elif 'processing_summary' in step_result:
                            processing = step_result.get('processing_summary', {})
                            # Build materials from web_pages and downloaded_files
                            for web_page in processing.get('web_pages', []):
                                materials.append(
                                    {
                                        'type': 'web_page',
                                        'title': web_page.get('title', ''),
                                        'url': web_page.get('url', ''),
                                        'content': web_page.get('content', ''),
                                    }
                                )
                            for file_data in processing.get('downloaded_files', []):
                                if (
                                    file_data.get('type') == 'pdf'
                                    and 'sections' in file_data
                                ):
                                    materials.append(
                                        {
                                            'type': 'pdf',
                                            'title': file_data.get('title', '')
                                            or file_data.get('url', '').split('/')[-1]
                                            or 'PDF',
                                            'url': file_data.get('url', ''),
                                            'sections': file_data.get('sections', []),
                                            'image_analysis': file_data.get(
                                                'image_analysis', ''
                                            ),
                                            'content': file_data.get('content', ''),
                                        }
                                    )
                                else:
                                    materials.append(
                                        {
                                            'type': 'file'
                                            if file_data.get('type') != 'pdf'
                                            else 'pdf',
                                            'title': file_data.get('title', '')
                                            or file_data.get('url', '').split('/')[-1]
                                            or 'File',
                                            'url': file_data.get('url', ''),
                                            'content': file_data.get('content', ''),
                                        }
                                    )

                    # Build structured context entry for this step
                    # Include full result text (no truncation) for complete context
                    if isinstance(step_result, dict):
                        summary = step_result.get('content', '') or str(step_result)
                    else:
                        summary = str(step_result)

                    context[step_id] = {
                        'materials': materials,
                        'summary': summary,  # Full summary without truncation
                        'full_result': step_result,  # Include full result for backward compatibility
                    }

                    # Also add with simplified key (e.g., step1, step2) for backward compatibility
                    simplified_key = step_id.replace('step_', 'step')
                    if simplified_key != step_id:
                        context[simplified_key] = context[step_id]

                # Also add dependency_results for backward compatibility
                context['dependency_results'] = previous_results

                self.logger.info(
                    f'Added {len(previous_results)} previous step result(s) to LLM reasoning context '
                    f'({sum(len(ctx.get("materials", [])) for ctx in context.values() if isinstance(ctx, dict) and "materials" in ctx)} materials total)'
                )
                self.logger.debug(
                    f'Context keys available: {", ".join(sorted(context.keys()))}'
                )

                self.logger.debug(
                    f'Executing LLM reasoning with task: {task_description[:100]}..., '
                    f'context keys: {list(context.keys())}'
                )
                # Check if images are present and use visual LLM if needed
                images = self._extract_images_from_context(context, attachments)
                if images:
                    self.logger.info(
                        f'Images detected in subtask {subtask.id}, using visual LLM for processing'
                    )
                    result = self.tool_belt.llm_reasoning_with_images(
                        task_description, context, images
                    )
                else:
                    # Execute LLM reasoning (simplified - no code error fixing needed)
                    result = self.tool_belt.llm_reasoning(task_description, context)

            elif tool_name == 'search':
                # Get search_queries array from metadata (prefer new format, fallback to old)
                search_queries = subtask.metadata.get('search_queries', [])

                # Handle backward compatibility: if search_query (singular) exists, convert to array
                if not search_queries:
                    old_search_query = subtask.metadata.get('search_query', '')
                    if old_search_query:
                        search_queries = [old_search_query]
                        self.logger.debug(
                            f'Subtask {subtask.id} uses old search_query format. '
                            f'Converted to search_queries array with 1 query.'
                        )

                # Fallback to parameters or description if no queries found
                if not search_queries:
                    fallback_query = parameters.get('query', subtask.description)
                    search_queries = [fallback_query]
                    if fallback_query == subtask.description:
                        self.logger.warning(
                            f'Subtask {subtask.id} missing search_queries in metadata. '
                            f'Using description as fallback query: "{fallback_query}"'
                        )

                # Ensure we have exactly 3 queries (pad if needed)
                if len(search_queries) < 3:
                    while len(search_queries) < 3:
                        search_queries.append(
                            search_queries[-1]
                            if search_queries
                            else subtask.description
                        )
                    self.logger.warning(
                        f'Subtask {subtask.id} has only {len([q for q in search_queries if q])} unique search queries. '
                        f'Padded to 3 queries.'
                    )
                elif len(search_queries) > 3:
                    search_queries = search_queries[:3]
                    self.logger.debug(
                        f'Subtask {subtask.id} has {len(search_queries)} search queries. Using first 3.'
                    )

                num_results = parameters.get(
                    'num_results', 5
                )  # Default: 5 results per query
                search_type = parameters.get('search_type', 'web')

                # Try all 3 search queries and combine results
                all_search_results = []
                seen_urls = set()  # Track URLs to avoid duplicates

                self.logger.info(
                    f'Executing {len(search_queries)} different search queries for subtask {subtask.id}'
                )

                for idx, query in enumerate(search_queries, 1):
                    self.logger.info(
                        f'Search query {idx}/{len(search_queries)}: "{query}" '
                        f'(type={search_type}, num_results={num_results})'
                    )
                    query_results = self.tool_belt.search(
                        query, num_results, search_type
                    )

                    # Add results to combined list, avoiding duplicates by URL
                    for result in query_results:
                        if hasattr(result, 'url'):
                            url = result.url
                        elif isinstance(result, dict):
                            url = result.get('url', '')
                        else:
                            url = str(result)

                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_search_results.append(result)

                    self.logger.debug(
                        f'Query {idx} returned {len(query_results)} results, '
                        f'{len(all_search_results)} total unique results so far'
                    )

                search_results = all_search_results
                self.logger.info(
                    f'Combined search results: {len(search_results)} unique results from {len(search_queries)} queries'
                )

                # Use LLM to determine all relevant results from combined set
                was_selected_indices_empty = False
                if search_results:
                    self.logger.info(
                        f'Filtering {len(search_results)} combined results to select all relevant results using LLM'
                    )
                    # Use subtask description as the query context for filtering
                    filtered_results, was_selected_indices_empty = (
                        self._filter_search_results_by_relevance(
                            search_results=search_results,
                            query=subtask.description,  # Use subtask description as context
                            problem=problem,
                            query_analysis=query_analysis,
                        )
                    )
                    search_results = filtered_results
                    self.logger.info(
                        f'Selected {len(search_results)} relevant results from combined search queries'
                    )

                # If selected_indices was empty (0), generate new queries and retry
                # Note: We retry if selected_indices was 0, even if fallback returned all results
                if was_selected_indices_empty:
                    self.logger.info(
                        'LLM selected 0 indices from search results. Generating new search queries and retrying...'
                    )
                    # Generate new search queries
                    new_search_queries = self._generate_new_search_queries(
                        subtask_description=subtask.description,
                        problem=problem,
                        previous_queries=search_queries,
                        query_analysis=query_analysis,
                    )

                    # Perform search again with new queries
                    retry_search_results = []
                    seen_urls_retry = set(
                        seen_urls
                    )  # Keep track of URLs from first attempt

                    self.logger.info(
                        f'Retrying search with {len(new_search_queries)} new queries for subtask {subtask.id}'
                    )

                    for idx, query in enumerate(new_search_queries, 1):
                        self.logger.info(
                            f'Retry search query {idx}/{len(new_search_queries)}: "{query}" '
                            f'(type={search_type}, num_results={num_results})'
                        )
                        query_results = self.tool_belt.search(
                            query, num_results, search_type
                        )

                        # Add results to combined list, avoiding duplicates by URL
                        for result in query_results:
                            if hasattr(result, 'url'):
                                url = result.url
                            elif isinstance(result, dict):
                                url = result.get('url', '')
                            else:
                                url = str(result)

                            if url and url not in seen_urls_retry:
                                seen_urls_retry.add(url)
                                retry_search_results.append(result)

                        self.logger.debug(
                            f'Retry query {idx} returned {len(query_results)} results, '
                            f'{len(retry_search_results)} total unique results so far'
                        )

                    # Filter retry results by relevance
                    if retry_search_results:
                        self.logger.info(
                            f'Filtering {len(retry_search_results)} retry search results to select all relevant results using LLM'
                        )
                        retry_filtered_results, retry_was_empty = (
                            self._filter_search_results_by_relevance(
                                search_results=retry_search_results,
                                query=subtask.description,
                                problem=problem,
                                query_analysis=query_analysis,
                            )
                        )
                        # Use retry results if they have selected indices, otherwise keep original
                        if not retry_was_empty and retry_filtered_results:
                            search_results = retry_filtered_results
                            self.logger.info(
                                f'Retry search successful: Selected {len(search_results)} relevant results from retry queries'
                            )
                        else:
                            self.logger.warning(
                                'Retry search also returned 0 selected indices. Using original results as fallback.'
                            )
                    else:
                        self.logger.warning(
                            'Retry search returned 0 results. Using original results as fallback.'
                        )

                # If search returned 0 results, try to identify downloadable resources
                if not search_results:
                    self.logger.info(
                        'Search returned 0 results. Attempting to identify downloadable resources from problem description...'
                    )
                    try:
                        resources = self._identify_downloadable_resources(
                            problem, subtask.description, query
                        )
                        if resources:
                            for resource in resources:
                                if resource.get('url'):
                                    url = resource['url']
                                    title = resource.get('title', 'Downloaded resource')
                                    try:
                                        attachment = (
                                            self.tool_belt.download_file_from_url(url)
                                        )
                                        if attachments is not None:
                                            attachments.append(attachment)
                                            self.logger.info(
                                                f'Downloaded resource: {title} from {url}'
                                            )
                                            # Add to search results as a SearchResult for consistency
                                            from ..models import SearchResult

                                            search_results.append(
                                                SearchResult(
                                                    title=title,
                                                    snippet=resource.get(
                                                        'description', title
                                                    ),
                                                    url=url,
                                                    relevance_score=resource.get(
                                                        'relevance', 0.9
                                                    ),
                                                )
                                            )
                                    except Exception as e:
                                        self.logger.warning(
                                            f'Failed to download resource from {url}: {e}'
                                        )
                    except Exception as e:
                        self.logger.debug(f'Resource identification failed: {e}')

                # Use SearchResultProcessor to systematically process results
                # This will:
                # 1. Check relevance of each result using LLM
                # 2. Classify as web page vs file
                # 3. Navigate to web pages or download files
                # 4. Extract and structure content
                if search_results:
                    self.logger.info(
                        f'Processing {len(search_results)} search results systematically...'
                    )
                    processing_result = self.search_processor.process_search_results(
                        search_results=search_results,
                        subtask_description=subtask.description,
                        problem=problem,
                        query_analysis=query_analysis,
                        attachments=attachments,
                        max_results_to_process=len(
                            search_results
                        ),  # Process all relevant results
                    )

                    # Build materials array from processed results
                    # We need to get materials from the processed_result which contains the structured data
                    # The web_pages and downloaded_files arrays contain the 'data' field from _process_single_result
                    materials = []

                    # Add web pages as materials
                    for web_page in processing_result.get('web_pages', []):
                        materials.append(
                            {
                                'type': 'web_page',
                                'title': web_page.get('title', ''),
                                'url': web_page.get('url', ''),
                                'content': web_page.get('content', ''),
                            }
                        )

                    # Add downloaded files as materials
                    for file_data in processing_result.get('downloaded_files', []):
                        if file_data.get('type') == 'pdf' and 'sections' in file_data:
                            # Structured PDF with sections
                            materials.append(
                                {
                                    'type': 'pdf',
                                    'title': file_data.get('title', '')
                                    or file_data.get('url', '').split('/')[-1]
                                    or 'PDF',
                                    'url': file_data.get('url', ''),
                                    'sections': file_data.get('sections', []),
                                    'image_analysis': file_data.get(
                                        'image_analysis', ''
                                    ),
                                    'content': file_data.get(
                                        'content', ''
                                    ),  # Full text fallback
                                }
                            )
                        else:
                            # Regular file or PDF without sections
                            materials.append(
                                {
                                    'type': 'file'
                                    if file_data.get('type') != 'pdf'
                                    else 'pdf',
                                    'title': file_data.get('title', '')
                                    or file_data.get('url', '').split('/')[-1]
                                    or 'File',
                                    'url': file_data.get('url', ''),
                                    'content': file_data.get('content', ''),
                                }
                            )

                    # Use LLM to analyze processed search results and determine the answer
                    # based on the subtask description
                    llm_analysis = self._analyze_search_results_with_llm(
                        processing_result=processing_result,
                        materials=materials,
                        subtask_description=subtask.description,
                        problem=problem,
                        query_analysis=query_analysis,
                    )

                    # Return LLM analysis answer as the primary result (string)
                    # This matches the format of llm_reasoning results for consistency
                    result = llm_analysis.get(
                        'content', 'No answer determined from search results.'
                    )

                    # Store full analysis and metadata in subtask metadata for reference
                    subtask.metadata['search_analysis'] = llm_analysis
                    subtask.metadata['search_metadata'] = {
                        'search_results': search_results,  # Original search results (for reference)
                        'materials': materials,  # Structured materials with title + content
                        'processing_summary': processing_result,
                        'relevant_count': processing_result.get('relevant_count', 0),
                        'web_pages': processing_result.get('web_pages', []),
                        'downloaded_files': processing_result.get(
                            'downloaded_files', []
                        ),
                    }
                else:
                    # No search results - return empty analysis
                    llm_analysis = self._analyze_search_results_with_llm(
                        processing_result={
                            'content_summary': '',
                            'web_pages': [],
                            'downloaded_files': [],
                        },
                        materials=[],
                        subtask_description=subtask.description,
                        problem=problem,
                        query_analysis=query_analysis,
                    )
                    result = llm_analysis.get('content', 'No search results found.')
                    subtask.metadata['search_analysis'] = llm_analysis
            elif tool_name == 'browser_navigate':
                url = parameters.get('url', '')
                action = parameters.get('action', None)
                link_text = parameters.get('link_text', None)
                selector = parameters.get('selector', None)
                extraction_query = parameters.get('extraction_query', None)

                # Validate URL is provided
                if not url or not url.strip():
                    error_msg = (
                        f'Invalid parameters for browser_navigate: URL is required but was not provided. '
                        f'Parameters received: {json.dumps(parameters, default=str)}'
                    )
                    self.logger.error(error_msg)
                    result = {
                        'success': False,
                        'error': error_msg,
                        'parameters_received': parameters,
                    }
                else:
                    # If action is extract_count/extract_statistics and no extraction_query,
                    # generate one from subtask description
                    if (
                        action in ('extract_count', 'extract_statistics')
                        and not extraction_query
                    ):
                        extraction_query = subtask.description
                    self.logger.debug(
                        f'Executing browser_navigate: url="{url}", action={action}, extraction_query="{extraction_query}"'
                    )
                    result = self.tool_belt.browser_navigate(
                        url, action, link_text, selector, extraction_query
                    )
            elif tool_name == 'read_attachment':
                if attachments:
                    attachment = attachments[parameters.get('attachment_index', 0)]
                    options = parameters.get('options', {})
                    self.logger.debug(f'Reading attachment: {attachment.filename}')
                    result = self.tool_belt.read_attachment(
                        attachment, options, problem, query_analysis
                    )

                    # Handle structured PDF results with sections
                    # If result is a dict with sections, store them in subtask metadata
                    if (
                        isinstance(result, dict)
                        and result.get('type') == 'pdf'
                        and 'sections' in result
                    ):
                        # Store structured PDF data in subtask metadata for later use
                        if not hasattr(subtask, 'metadata'):
                            subtask.metadata = {}
                        subtask.metadata['pdf_data'] = {
                            'filename': result.get('filename', attachment.filename),
                            'sections': result.get('sections', []),
                            'image_analysis': result.get('image_analysis', ''),
                            'full_text': result.get('full_text', ''),
                        }
                        self.logger.info(
                            f'Extracted {len(result.get("sections", []))} sections from PDF {attachment.filename}'
                        )
                else:
                    result = 'Error: No attachments available'
            elif tool_name == 'analyze_media':
                if attachments:
                    attachment = attachments[parameters.get('attachment_index', 0)]
                    analysis_type = parameters.get('analysis_type', 'auto')
                    self.logger.debug(
                        f'Analyzing media: {attachment.filename}, type={analysis_type}'
                    )
                    result = self.tool_belt.analyze_media(attachment, analysis_type)
                else:
                    result = 'Error: No attachments available'
            else:
                result = f'Error: Unknown tool {tool_name}'

            # === AFTER SUBTASK EXECUTION (SUCCESS) ===
            state_after = self.state_manager.get_state_summary()

            # Handle structured PDF results - convert to string format with sections
            if (
                isinstance(result, dict)
                and result.get('type') == 'pdf'
                and 'sections' in result
            ):
                # Format sections into readable text for the result
                sections_text = []
                for section in result.get('sections', []):
                    section_title = section.get('title', 'Untitled')
                    section_content = section.get('content', '')
                    section_page = section.get('page', '?')
                    sections_text.append(
                        f'[Section: {section_title} (Page {section_page})]\n{section_content}'
                    )

                # Combine sections with image analysis
                formatted_result = '\n\n'.join(sections_text)
                if result.get('image_analysis'):
                    formatted_result += '\n\nIMAGE ANALYSIS (from visual LLM):\n'
                    formatted_result += result.get('image_analysis', '')

                # Use formatted result for storage and logging
                result_str = formatted_result
                result_type = 'pdf_with_sections'
                result_to_store = formatted_result  # Store formatted string
            else:
                # Regular result (string or other)
                result_str = str(result)
                result_type = type(result).__name__
                result_to_store = result  # Store original result

            # Check if result contains image analysis - if so, extract and log it separately
            image_analysis_marker = 'IMAGE ANALYSIS (from visual LLM):'
            has_image_analysis = image_analysis_marker in result_str
            image_analysis = None

            if has_image_analysis:
                # Extract image analysis section
                # Format: "\n\n" + "="*80 + "\n" + "IMAGE ANALYSIS (from visual LLM):\n" + "="*80 + "\n" + content
                marker_idx = result_str.find(image_analysis_marker)
                if marker_idx >= 0:
                    # Marker ends with '\n', so find the next line (separator line)
                    after_marker = marker_idx + len(image_analysis_marker)
                    # Skip the separator line (=====)
                    separator_end = result_str.find('\n', after_marker)
                    if separator_end >= 0:
                        # Content starts after the separator line
                        content_start = separator_end + 1
                        image_analysis = result_str[content_start:].strip()
                    else:
                        # Fallback: content starts right after marker
                        image_analysis = result_str[after_marker:].strip()

                    # Create summary without image analysis for brevity
                    text_before_analysis = result_str[:marker_idx].strip()
                    result_summary = (
                        text_before_analysis[:200] + '...'
                        if len(text_before_analysis) > 200
                        else text_before_analysis
                    )
                    result_summary += (
                        f'\n[+ Image Analysis: {len(image_analysis)} chars - see below]'
                    )
                else:
                    result_summary = (
                        result_str[:200] + '...'
                        if len(result_str) > 200
                        else result_str
                    )
            else:
                result_summary = (
                    result_str[:200] + '...' if len(result_str) > 200 else result_str
                )

            # Check if subtask was already marked as failed (e.g., from LLM reasoning error)
            if subtask.id in self.state_manager.subtasks:
                current_subtask = self.state_manager.subtasks[subtask.id]
                if current_subtask.status == 'failed':
                    # Don't mark as completed if it's already failed
                    self.logger.warning(
                        f'Subtask {subtask.id} already marked as failed, not completing'
                    )
                else:
                    self.state_manager.complete_subtask(subtask.id, result_to_store)
            else:
                # If not in state_manager, add it as completed
                self.state_manager.complete_subtask(subtask.id, result_to_store)

            self.logger.info('-' * 80)
            self.logger.info(f'AFTER executing subtask: {subtask.id}')
            self.logger.info('  Status: completed')
            self.logger.info(f'  Result type: {result_type}')
            self.logger.info(f'  Result summary: {result_summary}')
            self.logger.info(f'  Result length: {len(result_str)} chars')

            # Log full image analysis separately if present
            if has_image_analysis and image_analysis:
                self.logger.info('-' * 80)
                self.logger.info('IMAGE ANALYSIS (from visual LLM):')
                self.logger.info('=' * 80)
                self.logger.info(image_analysis)
                self.logger.info('=' * 80)

            self.logger.info(f'  State after: {state_after}')
            self.logger.info(
                f'  Knowledge facts count: {len(self.state_manager.knowledge_graph)}'
            )

            # Show what changed in state
            state_changes = {
                'knowledge_facts': state_after['knowledge_facts']
                - state_before['knowledge_facts'],
                'subtasks_completed': state_after['subtasks_completed']
                - state_before['subtasks_completed'],
                'subtasks_pending': state_after['subtasks_pending']
                - state_before['subtasks_pending'],
            }
            self.logger.info(f'  State changes: {state_changes}')
            self.logger.info('=' * 80)

            return result_to_store
        except Exception as e:
            # === AFTER SUBTASK EXECUTION (FAILURE) ===
            state_after = self.state_manager.get_state_summary()

            self.state_manager.fail_subtask(subtask.id, str(e))

            self.logger.error('-' * 80)
            self.logger.error(f'AFTER executing subtask: {subtask.id} - FAILED')
            self.logger.error('  Status: failed')
            self.logger.error(f'  Error: {str(e)}')
            self.logger.error(f'  Error type: {type(e).__name__}')
            self.logger.error(f'  State after: {state_after}')
            self.logger.error(f'  Dead ends count: {len(self.state_manager.dead_ends)}')
            self.logger.error('=' * 80)

            self.logger.error(f'Subtask execution failed: {e}', exc_info=True)
            raise

    def _serialize_result_for_code(self, result: Any) -> Any:
        """
        Serialize result objects for use in LLM reasoning.

        Converts SearchResult dataclass objects to dictionaries to avoid
        iteration errors in RestrictedPython execution environment.

        Args:
            result: The result to serialize (can be dict, list, SearchResult, or other types)

        Returns:
            Serialized result with SearchResult objects converted to dictionaries
        """
        from ..models import SearchResult

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
                        self._serialize_result_for_code(item) for item in value
                    ]
                elif isinstance(value, dict):
                    # Recursively serialize nested dictionaries
                    serialized[key] = self._serialize_result_for_code(value)
                else:
                    serialized[key] = value
            return serialized

        # If result is a list, recursively serialize items
        if isinstance(result, list):
            return [self._serialize_result_for_code(item) for item in result]

        # For other types (str, int, etc.), return as-is
        return result

    def _execute_code_with_retry(
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
                return self._handle_code_execution_success(result, retry_count, subtask)

            # Error detected - extract error message and try to fix
            error_reason = ExecutionResultAnalyzer.extract_error_message(result)
            if error_reason is None:
                error_reason = str(result)[:500]
            last_error = error_reason[:500]

            self.logger.warning(
                f'Code execution failed (attempt {retry_count + 1}/{max_retries}): {error_reason}'
            )

            # Attempt to fix the code
            fixed_code = self._fix_code_error(
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
            self._update_code_retry_metadata(
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
        return self._handle_code_execution_failure(
            result, last_error, retry_count, subtask
        )

    def _handle_code_execution_success(
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
            self._clear_code_error_metadata(subtask)

        return result

    def _handle_code_execution_failure(
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

    def _update_code_retry_metadata(
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

    def _clear_code_error_metadata(self, subtask: Subtask) -> None:
        """
        Clear error-related metadata from subtask after successful execution.

        Args:
            subtask: Subtask that succeeded
        """
        metadata_keys_to_remove = ['error', 'error_type', 'code_fix_retry_count']
        for key in metadata_keys_to_remove:
            if key in subtask.metadata:
                del subtask.metadata[key]

    def _fix_code_error(
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

    def _determine_tool_parameters(
        self,
        subtask: Subtask,
        problem: str,
        attachments: Optional[List[Attachment]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to determine tool parameters for a subtask.

        Args:
            subtask: Subtask to execute.
            problem: Original problem.
            attachments: Optional attachments.

        Returns:
            Dictionary of tool parameters.
        """
        self.logger.debug(f'Determining parameters for subtask: {subtask.id}')

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
For search: {"query": "search query", "num_results": 5, "search_type": "web"}
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
                if dep_id in self.state_manager.subtasks:
                    dep_subtask = self.state_manager.subtasks[dep_id]
                    if dep_subtask.status == 'completed':
                        # Include full result (no truncation) for complete context
                        result_text = (
                            str(dep_subtask.result) if dep_subtask.result else 'None'
                        )
                        dependency_results_summary.append(
                            f'  - {dep_id}: {result_text}'
                        )
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
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative but focused for parameter generation
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            return json.loads(json_text)
        except Exception as e:
            self.logger.error(f'Parameter determination failed: {e}')
            return {}

    def _identify_downloadable_resources(
        self, problem: str, subtask_description: str, query: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to identify downloadable resources (PDFs, documents, etc.) from problem description.
        This is a general-purpose method that works for any type of resource.

        Args:
            problem: Original problem description.
            subtask_description: Description of the subtask.
            query: The search query.

        Returns:
            List of dictionaries with resource information:
            - url: Direct URL to the resource
            - title: Descriptive title
            - description: Description of the resource
            - relevance: Relevance score (0-1)
        """
        system_prompt = """You are an expert at identifying downloadable resources from problem descriptions.
Given a problem description, identify any resources that need to be downloaded to solve the problem.

Resources could include:
- Academic papers (preprint repositories, PubMed, etc.)
- Government documents
- Reports or datasets
- Documents mentioned by URL, ID, or citation
- Any downloadable files referenced in the problem

For each resource you identify, provide:
- url: The direct download URL if you can construct it (e.g., for preprint repositories: https://domain.org/pdf/PAPER_ID.pdf)
- title: A descriptive title
- description: Brief description of the resource
- relevance: Relevance score from 0.0 to 1.0

Return a JSON object with key "resources" containing an array of resource objects.
If you cannot construct a direct URL but can identify a resource, include it with url as null and provide as much info as possible.
If no resources are found, return {{"resources": []}}.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.

Examples:
- "Preprint paper 2207.01510" -> {{"url": "https://repository.org/pdf/2207.01510.pdf", "title": "Preprint paper 2207.01510", "description": "Preprint paper with ID 2207.01510", "relevance": 1.0}}
- "Paper submitted to preprint repository in June 2022" -> {{"url": null, "title": "June 2022 preprint submission", "description": "Paper submitted to preprint repository in June 2022", "relevance": 0.8}}
- "Document at https://example.com/doc.pdf" -> {{"url": "https://example.com/doc.pdf", "title": "Document from example.com", "description": "PDF document", "relevance": 1.0}}"""

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}

Search Query: {query}

Identify any downloadable resources (papers, documents, PDFs, etc.) mentioned in the above text.
Return as JSON with "resources" key containing an array of resource objects."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative but focused for resource identification
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            data = json.loads(json_text)

            resources = data.get('resources', [])
            if not isinstance(resources, list):
                resources = []

            # Filter to only include resources with URLs (can be downloaded)
            downloadable_resources = [
                r for r in resources if isinstance(r, dict) and r.get('url')
            ]

            if downloadable_resources:
                self.logger.info(
                    f'Identified {len(downloadable_resources)} downloadable resource(s)'
                )
            return downloadable_resources
        except Exception as e:
            self.logger.debug(f'Resource identification failed: {e}')
            return []

    def _filter_search_results_by_relevance(
        self,
        search_results: List[Any],
        query: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[Any], bool]:
        """
        Use LLM to filter and rank search results by relevance to the query.

        Args:
            search_results: List of SearchResult objects from search tool.
            query: The search query that produced these results.
            problem: The original problem being solved.
            query_analysis: Optional query analysis results containing requirements and constraints.

        Returns:
            Tuple of (filtered list of most relevant SearchResult objects, was_selected_indices_empty).
            was_selected_indices_empty is True if the LLM returned 0 selected indices.
        """
        from ..models import SearchResult

        if not search_results:
            return search_results, False

        self.logger.info(
            f'Filtering {len(search_results)} search results by relevance using LLM'
        )

        # Build a formatted list of search results for the LLM
        results_list = []
        for i, result in enumerate(search_results):
            if isinstance(result, SearchResult):
                results_list.append(
                    {
                        'index': i,
                        'title': result.title,
                        'snippet': result.snippet,
                        'url': result.url,
                    }
                )
            elif isinstance(result, dict):
                # Handle already-serialized SearchResult dictionaries
                results_list.append(
                    {
                        'index': i,
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'url': result.get('url', ''),
                    }
                )

        # Extract requirement information from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            implicit_reqs = query_analysis.get('implicit_requirements', [])
            constraints = query_analysis.get('constraints', {})
            answer_format = query_analysis.get('answer_format', '')

            requirements_context = '\n\nKey Requirements from Query Analysis:\n'
            if explicit_reqs:
                requirements_context += (
                    f'- Explicit Requirements: {", ".join(explicit_reqs)}\n'
                )
            if implicit_reqs:
                requirements_context += (
                    f'- Implicit Requirements: {", ".join(implicit_reqs)}\n'
                )
            if constraints:
                constraints_str = []
                if constraints.get('temporal'):
                    constraints_str.append(
                        f'Temporal: {", ".join(constraints["temporal"])}'
                    )
                if constraints.get('spatial'):
                    constraints_str.append(
                        f'Spatial: {", ".join(constraints["spatial"])}'
                    )
                if constraints.get('categorical'):
                    constraints_str.append(
                        f'Categorical: {", ".join(constraints["categorical"])}'
                    )
                if constraints_str:
                    requirements_context += (
                        f'- Constraints: {"; ".join(constraints_str)}\n'
                    )
            if answer_format:
                requirements_context += f'- Expected Answer Format: {answer_format}\n'

        system_prompt = """You are an expert at evaluating search result relevance.
Given a search query/subtask description, query requirements analysis, and a list of search results, identify which results are relevant to answering the query.

Consider the explicit and implicit requirements from the query analysis when determining relevance.
A result is relevant if it helps satisfy any of the stated requirements or constraints.

Return a JSON object with:
- selected_indices: list of integers representing the indices (0-based) of ALL relevant results, ordered by relevance (most relevant first)
- reasoning: brief explanation of why these results were selected, specifically referencing which requirements they address

IMPORTANT:
- Select ALL results that are relevant to the query, not just a fixed number
- Include any result that could help answer the query or satisfy the requirements
- Order results by relevance (most relevant first)
- Return your response as valid JSON only, without any markdown formatting or additional text

Focus on:
- Direct relevance to the query/subtask terms and intent
- Alignment with explicit and implicit requirements from query analysis
- How well each result addresses the specific requirements and constraints
- Information quality and usefulness for satisfying the problem requirements
- For aggregate/statistical queries (e.g., "how many articles"), prioritize archive/browse pages over individual articles"""

        user_prompt = f"""Original Problem: {problem}

Subtask/Query: {query}
{requirements_context}

Search Results (combined from multiple search queries):
{json.dumps(results_list, indent=2)}

Identify ALL relevant search results for answering the query/subtask. Pay special attention to which results satisfy the requirements identified in the query analysis. Return the indices of all relevant results, ordered by relevance (most relevant first). Include any result that could help answer the query or satisfy the requirements."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative but focused for search result filtering
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            selected_indices = result_data.get('selected_indices', [])
            reasoning = result_data.get('reasoning', 'No reasoning provided')
            was_selected_indices_empty = len(selected_indices) == 0

            self.logger.info(
                f'LLM selected {len(selected_indices)} most relevant result(s) out of {len(search_results)}. Reasoning: {reasoning}'
            )

            # Filter results to only include selected indices, preserving order
            filtered_results = []
            seen_indices = set()  # Avoid duplicates
            for idx in selected_indices:
                if 0 <= idx < len(search_results) and idx not in seen_indices:
                    filtered_results.append(search_results[idx])
                    seen_indices.add(idx)
                elif idx not in seen_indices:
                    self.logger.warning(
                        f'Invalid index {idx} in selected_indices (valid range: 0-{len(search_results) - 1})'
                    )

            # If no valid results were selected, return all original results as fallback
            if not filtered_results:
                self.logger.warning(
                    'No valid results selected by LLM, using all original results as fallback'
                )
                return search_results, was_selected_indices_empty

            self.logger.info(
                f'LLM filtered {len(search_results)} results down to {len(filtered_results)} most relevant'
            )
            return filtered_results, was_selected_indices_empty

        except Exception as e:
            self.logger.error(
                f'Failed to filter search results by relevance: {e}', exc_info=True
            )
            # Fallback to original results if filtering fails
            self.logger.warning('Using all original search results as fallback')
            return search_results, False

    def _generate_new_search_queries(
        self,
        subtask_description: str,
        problem: str,
        previous_queries: List[str],
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate new search queries when initial search returned 0 selected indices.

        Args:
            subtask_description: Description of the subtask.
            problem: Original problem being solved.
            previous_queries: List of previous search queries that didn't yield results.
            query_analysis: Optional query analysis results.

        Returns:
            List of 3 new search queries.
        """
        self.logger.info(
            f'Generating new search queries for subtask. Previous queries: {previous_queries}'
        )

        system_prompt = """You are an expert at creating effective search queries.
Given a subtask description, problem context, and previous search queries that didn't yield relevant results, create 3 NEW and DIFFERENT search queries.

CRITICAL REQUIREMENTS:
- Create exactly 3 different search queries
- Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
- Keep each query SHORT: 3-8 keywords maximum (typically 5-6 words)
- Remove filler words like "article", "submitted", "descriptors", "about", "related to"
- Use dates in format: "August 11 2016" or "2016-08-11" or "August 2016"
- Separate keywords with spaces, NOT commas or special formatting
- Make queries DIFFERENT from the previous ones - try alternative keyword combinations, synonyms, or different phrasings
- Focus on the core information needed to complete the subtask

Return a JSON object with:
- search_queries: array of exactly 3 different search queries in keyword-only format

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        # Build context about previous queries
        previous_queries_context = ''
        if previous_queries:
            previous_queries_context = (
                '\n\nPrevious search queries that did not yield relevant results:\n'
            )
            for i, query in enumerate(previous_queries, 1):
                previous_queries_context += f'{i}. {query}\n'
            previous_queries_context += '\nCreate NEW queries with different keyword combinations or phrasings.\n'

        # Extract requirement information from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            implicit_reqs = query_analysis.get('implicit_requirements', [])
            constraints = query_analysis.get('constraints', {})

            if explicit_reqs or implicit_reqs or constraints:
                requirements_context = '\n\nKey Requirements:\n'
                if explicit_reqs:
                    requirements_context += (
                        f'- Explicit Requirements: {", ".join(explicit_reqs)}\n'
                    )
                if implicit_reqs:
                    requirements_context += (
                        f'- Implicit Requirements: {", ".join(implicit_reqs)}\n'
                    )
                if constraints:
                    constraints_str = []
                    if constraints.get('temporal'):
                        constraints_str.append(
                            f'Temporal: {", ".join(constraints["temporal"])}'
                        )
                    if constraints.get('spatial'):
                        constraints_str.append(
                            f'Spatial: {", ".join(constraints["spatial"])}'
                        )
                    if constraints.get('categorical'):
                        constraints_str.append(
                            f'Categorical: {", ".join(constraints["categorical"])}'
                        )
                    if constraints_str:
                        requirements_context += (
                            f'- Constraints: {"; ".join(constraints_str)}\n'
                        )

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}
{previous_queries_context}

Create 3 NEW search queries (different from previous ones) that will help find information to complete this subtask. Use keyword-only format (3-8 keywords each)."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative for generating alternative queries
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            new_queries = result_data.get('search_queries', [])
            if not new_queries or len(new_queries) < 3:
                self.logger.warning(
                    f'LLM generated only {len(new_queries)} queries, padding to 3'
                )
                # Pad with variations of the subtask description
                while len(new_queries) < 3:
                    new_queries.append(subtask_description)
            elif len(new_queries) > 3:
                new_queries = new_queries[:3]
                self.logger.debug(
                    f'LLM generated {len(new_queries)} queries, using first 3'
                )

            self.logger.info(
                'Generated %d new search queries: %s', len(new_queries), new_queries
            )
            return new_queries

        except Exception as e:
            self.logger.error(
                f'Failed to generate new search queries: {e}', exc_info=True
            )
            # Fallback: create variations from subtask description
            self.logger.warning(
                'Using fallback: creating query variations from subtask description'
            )
            # Simple fallback: use subtask description with slight variations
            base_query = subtask_description
            return [base_query, base_query, base_query]

    def execute_plan(
        self,
        plan: List[Subtask],
        problem: str,
        attachments: Optional[List[Attachment]] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete plan, respecting dependencies.

        Automatically downloads files from search results and adds them to attachments.

        Args:
            plan: List of subtasks.
            problem: Original problem.
            attachments: Optional attachments list (modified in-place if provided).
            query_analysis: Optional query analysis results containing requirements and constraints.

        Returns:
            Dictionary with execution results.
        """
        self.logger.info(f'Executing plan with {len(plan)} subtasks')

        # Initialize attachments list if None
        if attachments is None:
            attachments = []
        elif not isinstance(attachments, list):
            # Convert to list if needed
            attachments = list(attachments)

        results = {}
        completed_ids = set()

        # Topological sort: execute tasks respecting dependencies
        while len(completed_ids) < len(plan):
            progress_made = False

            for subtask in plan:
                if subtask.id in completed_ids:
                    continue

                # Check if dependencies are satisfied
                if all(dep_id in completed_ids for dep_id in subtask.dependencies):
                    subtask.status = 'in_progress'
                    try:
                        result = self.execute_subtask(
                            subtask, problem, attachments, query_analysis
                        )
                        results[subtask.id] = result

                        # Note: Search result processing is now handled within execute_subtask
                        # via SearchResultProcessor, so no additional processing needed here

                        completed_ids.add(subtask.id)
                        progress_made = True
                    except Exception as e:
                        self.logger.error(f'Failed to execute {subtask.id}: {e}')
                        # Include failed subtask in results with structured error format
                        # Note: fail_subtask() already called in execute_subtask(), so state is consistent
                        results[subtask.id] = {
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'status': 'failed',
                            'subtask_id': subtask.id,
                        }
                        # Try to continue with other tasks
                        continue

            if not progress_made:
                self.logger.warning(
                    'Execution stalled - circular dependencies or all remaining tasks failed'
                )
                break

        return results

    def retry_failed_subtasks(
        self,
        problem: str,
        attachments: Optional[List[Attachment]] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retry only the failed subtasks instead of re-executing the entire plan.

        Args:
            problem: Original problem.
            attachments: Optional attachments list.
            query_analysis: Optional query analysis results.

        Returns:
            Dictionary with retry execution results for failed subtasks only.
        """
        # Get failed subtasks from state manager
        failed_subtasks = self.state_manager.get_failed_subtasks()

        if not failed_subtasks:
            self.logger.info('No failed subtasks to retry')
            return {}

        self.logger.info(
            f'Retrying {len(failed_subtasks)} failed subtask(s): '
            f'{[st.id for st in failed_subtasks]}'
        )

        # Reset failed subtasks to pending status
        for subtask in failed_subtasks:
            self.state_manager.retry_subtask(subtask.id)

        # Re-execute only the failed subtasks
        results = {}
        for subtask in failed_subtasks:
            subtask.status = 'in_progress'
            try:
                self.logger.info(
                    f'Retrying failed subtask: {subtask.id} - {subtask.description}'
                )
                result = self.execute_subtask(
                    subtask, problem, attachments, query_analysis
                )
                results[subtask.id] = result
                self.logger.info(f'Successfully retried subtask: {subtask.id}')
            except Exception as e:
                self.logger.error(f'Failed to retry {subtask.id}: {e}')
                # Task will remain in failed state
                continue

        return results

    def _extract_images_from_context(
        self,
        context: Dict[str, Any],
        attachments: Optional[List[Attachment]] = None,
    ) -> List[bytes]:
        """
        Extract image data from context and attachments for visual LLM processing.

        Args:
            context: Context dictionary that may contain image data.
            attachments: Optional list of attachments that may contain images.

        Returns:
            List of image data as bytes.
        """
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
                    self.logger.debug(
                        f'Found image in attachment: {attachment.filename}'
                    )

        # Check context for image data (from browser screenshots, etc.)
        if isinstance(context, dict):
            # Check for screenshot data in context
            if 'screenshot' in context:
                screenshot_data = context.get('screenshot')
                if isinstance(screenshot_data, bytes):
                    images.append(screenshot_data)
                    self.logger.debug('Found screenshot in context')
                elif isinstance(screenshot_data, str):
                    # Try to decode base64 string
                    try:
                        import base64

                        decoded = base64.b64decode(screenshot_data)
                        images.append(decoded)
                        self.logger.debug('Found base64-encoded screenshot in context')
                    except Exception:
                        pass

            # Check for image data in nested structures
            for key, value in context.items():
                if key == 'screenshot' or 'image' in key.lower():
                    if isinstance(value, bytes):
                        images.append(value)
                    elif isinstance(value, str):
                        try:
                            import base64

                            decoded = base64.b64decode(value)
                            images.append(decoded)
                        except Exception:
                            pass

        return images

    def _analyze_search_results_with_llm(
        self,
        processing_result: Dict[str, Any],
        materials: List[Dict[str, Any]],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze processed search results and determine the answer based on subtask description.

        Args:
            processing_result: Result from SearchResultProcessor containing content_summary, web_pages, etc.
            materials: List of materials (web pages, files) extracted from search results.
            subtask_description: Description of the subtask being executed.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Dictionary with LLM analysis result as the primary content.
        """
        try:
            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                implicit_reqs = query_analysis.get('implicit_requirements', [])
                constraints = query_analysis.get('constraints', {})
                answer_format = query_analysis.get('answer_format', '')

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )
                if implicit_reqs:
                    requirements_context += (
                        f'\nImplicit Requirements: {", ".join(implicit_reqs)}'
                    )
                if constraints:
                    constraints_str = []
                    if constraints.get('temporal'):
                        constraints_str.append(
                            f'Temporal: {", ".join(constraints["temporal"])}'
                        )
                    if constraints.get('spatial'):
                        constraints_str.append(
                            f'Spatial: {", ".join(constraints["spatial"])}'
                        )
                    if constraints.get('categorical'):
                        constraints_str.append(
                            f'Categorical: {", ".join(constraints["categorical"])}'
                        )
                    if constraints_str:
                        requirements_context += (
                            f'\nConstraints: {"; ".join(constraints_str)}'
                        )
                if answer_format:
                    requirements_context += f'\nAnswer Format: {answer_format}'

            # Prepare content summary from processed results
            content_summary = processing_result.get('content_summary', '')

            # Build materials summary with full content (no truncation)
            materials_summary = []
            for material in materials:
                material_type = material.get('type', 'unknown')
                title = material.get('title', 'Untitled')
                url = material.get('url', '')
                content = material.get('content', '') or ''

                # Include full content for complete context
                if content:
                    materials_summary.append(
                        f'[{material_type.upper()}] {title}\nURL: {url}\nContent: {content}'
                    )
                else:
                    materials_summary.append(
                        f'[{material_type.upper()}] {title}\nURL: {url}'
                    )

            system_prompt = """You are an expert at analyzing search results and extracting information relevant to a specific task.

Given:
1. A subtask description that needs to be completed
2. Processed search results (content from web pages and files)
3. Problem requirements and constraints

Your task:
- Analyze the search results with respect to the subtask description
- Extract and determine the answer or information that addresses the subtask
- Provide a clear, focused response that directly answers what the subtask is asking for
- If the search results don't contain the needed information, clearly state that
- If multiple pieces of information are found, synthesize them appropriately

Return a JSON object with:
- answer: The answer or information determined from the search results (string)
- reasoning: Brief explanation of how you arrived at this answer (string)
- confidence: Confidence level from 0.0 to 1.0 (float)
- sources_used: List of source titles/URLs that were most relevant (array of strings)

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}

Subtask Description: {subtask_description}
{requirements_context}

Processed Search Results:
{content_summary if content_summary else 'No content extracted from search results.'}

Materials Found ({len(materials)} total):
{chr(10).join(materials_summary) if materials_summary else 'No materials found.'}

Analyze these search results with respect to the subtask description and determine the answer or information that addresses what the subtask is asking for."""

            self.logger.info(
                f'Analyzing search results with LLM for subtask: {subtask_description[:100]}...'
            )

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent analysis
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            analysis_data = json.loads(json_text)

            answer = analysis_data.get(
                'answer', 'No answer determined from search results.'
            )
            reasoning = analysis_data.get('reasoning', 'No reasoning provided.')
            confidence = analysis_data.get('confidence', 0.0)
            sources_used = analysis_data.get('sources_used', [])

            self.logger.info(
                f'LLM analysis complete: answer length={len(answer)}, confidence={confidence:.2f}'
            )

            # Return structured result with LLM analysis as primary content
            return {
                'content': answer,  # Primary result - LLM-determined answer
                'reasoning': reasoning,
                'confidence': confidence,
                'sources_used': sources_used,
                'analysis_type': 'search_results_analysis',
            }

        except Exception as e:
            self.logger.error(
                f'Failed to analyze search results with LLM: {e}', exc_info=True
            )
            # Fallback: return content summary as result
            return {
                'content': processing_result.get('content_summary', ''),
                'reasoning': f'LLM analysis failed: {str(e)}',
                'confidence': 0.0,
                'sources_used': [],
                'analysis_type': 'search_results_analysis_fallback',
            }

    def _is_file_url(self, url: str) -> bool:
        """
        Check if a URL points to a downloadable file.

        Args:
            url: URL to check.

        Returns:
            True if URL appears to point to a file, False otherwise.
        """
        if not url:
            return False

        url_lower = url.lower()

        # Check for file extensions
        file_extensions = {
            '.pdf',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.txt',
            '.csv',
            '.jpg',
            '.jpeg',
            '.png',
            '.gif',
            '.bmp',
            '.svg',
            '.webp',
            '.zip',
            '.tar',
            '.gz',
            '.rar',
            '.7z',
            '.mp3',
            '.mp4',
            '.avi',
            '.mov',
            '.wmv',
            '.ppt',
            '.pptx',
        }

        # Check if URL contains file extension
        for ext in file_extensions:
            if ext in url_lower:
                return True

        # Check if URL ends with common file patterns
        if any(url_lower.endswith(ext) for ext in file_extensions):
            return True

        # Check for patterns like /pdf/, /download/, etc. that often indicate files
        file_indicators = ['/pdf/', '/download/', '/file/', '/attachment/']
        if any(indicator in url_lower for indicator in file_indicators):
            return True

        return False

    def _is_result_relevant(
        self,
        search_result: Any,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Use LLM to determine if a search result is relevant to the problem.

        Args:
            search_result: SearchResult to evaluate.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            True if relevant, False otherwise.
        """
        try:
            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                implicit_reqs = query_analysis.get('implicit_requirements', [])

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )
                if implicit_reqs:
                    requirements_context += (
                        f'\nImplicit Requirements: {", ".join(implicit_reqs)}'
                    )

            system_prompt = """You are an expert at evaluating whether a search result is relevant to solving a problem.
Given a problem description and a search result (title, snippet, URL), determine if this result is likely to contain information useful for solving the problem.

Return a JSON object with:
- relevant: boolean indicating if the result is relevant
- reasoning: brief explanation of why it is or isn't relevant

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}
{requirements_context}

Search Result:
- Title: {search_result.title}
- Snippet: {search_result.snippet}
- URL: {search_result.url}

Is this search result relevant to solving the problem?"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent evaluation
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            relevant = result_data.get('relevant', False)
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'Relevance check for {search_result.url}: {relevant}. Reasoning: {reasoning}'
            )

            return relevant
        except Exception as e:
            self.logger.warning(
                f'Failed to determine relevance using LLM: {e}. Defaulting to relevant=True.'
            )
            # Default to relevant if LLM check fails to avoid skipping potentially useful pages
            return True

    def _process_search_results_for_downloads(
        self,
        search_results: List[Any],
        attachments: List[Attachment],
        subtask_id: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Process search results: navigate to relevant non-file pages, download file URLs.

        Args:
            search_results: List of SearchResult objects from search tool.
            attachments: Attachments list to append downloaded files to.
            subtask_id: ID of the subtask that produced these results.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
        """
        from ..models import SearchResult

        if not search_results:
            return

        self.logger.info(
            f'Processing {len(search_results)} search results: navigating relevant pages and downloading files...'
        )

        downloadable_extensions = {
            '.pdf',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.txt',
            '.csv',
        }

        downloaded_count = 0
        navigated_count = 0

        def extract_paper_id(url_or_text: str) -> Optional[tuple[str, str]]:
            """
            Extract paper/preprint ID from URL or text.

            Returns:
                Tuple of (paper_id, repository_type) if found, None otherwise.
                repository_type can be 'preprint' (for preprint repositories) or None.
            """
            # Pattern for preprint IDs: YYMM.NNNNN or archive-category/YYMMNNN format
            # This pattern works for preprint repositories using similar ID formats
            preprint_patterns = [
                # Direct URL patterns (domain-agnostic - works for any domain)
                r'(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # abs/2207.01510 or pdf/2207.01510
                r'(?:abs|pdf)/([a-z-]+/\d{7}(?:v\d+)?)',  # abs/cs/1234567 or pdf/math/1234567
                # Citation patterns (common formats across repositories)
                r'[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)',  # :2207.01510 or :2207.01510
                r'[:\s]+([a-z-]+/\d{7}(?:v\d+)?)',  # :cs/1234567
                r'\.(\d{4}\.\d{4,5}(?:v\d+)?)',  # .2207.01510
                r'\.([a-z-]+/\d{7}(?:v\d+)?)',  # .cs/1234567
                r'[:\.](\d{4}\.\d{4,5}(?:v\d+)?)',  # :2207.01510 or .2207.01510
            ]

            for pattern in preprint_patterns:
                match = re.search(pattern, url_or_text, re.IGNORECASE)
                if match:
                    paper_id = match.group(1)
                    # Remove version suffix if present (e.g., v1, v2)
                    paper_id = re.sub(r'v\d+$', '', paper_id)
                    # Detect repository type based on ID format
                    # YYMM.NNNNN or archive/category format suggests preprint repository
                    if re.match(r'^\d{4}\.\d{4,5}$', paper_id) or '/' in paper_id:
                        return (paper_id, 'preprint')
                    return (paper_id, None)
            return None

        def get_pdf_url_from_paper_id(
            paper_id: str, repository_type: str, original_url: str = ''
        ) -> Optional[str]:
            """
            Construct PDF URL from paper ID based on repository type.

            Args:
                paper_id: The extracted paper ID
                repository_type: Type of repository ('preprint' for preprint repositories, etc.)
                original_url: Original URL from search result (used to infer domain)

            Returns:
                PDF URL if repository type is known and domain can be inferred, None otherwise
            """
            if repository_type == 'preprint':
                # Try to infer base URL from original URL
                if original_url:
                    try:
                        from urllib.parse import urlparse

                        parsed = urlparse(original_url)
                        if parsed.netloc:
                            # Use the domain from original URL
                            base_url = f'{parsed.scheme}://{parsed.netloc}'
                            return f'{base_url}/pdf/{paper_id}.pdf'
                    except Exception:
                        pass

                # If we can't infer domain from URL, we cannot construct a valid download URL
                # without domain-specific knowledge
                return None
            return None

        for result in search_results:
            if not isinstance(result, SearchResult):
                continue

            url = result.url
            if not url:
                continue

            # Check if URL looks like a downloadable file
            url_lower = url.lower()
            is_downloadable = False
            download_url = url

            # First, check by file extension (direct downloadable files)
            for ext in downloadable_extensions:
                if ext in url_lower:
                    is_downloadable = True
                    break

            # If not directly downloadable, try to extract paper ID from URL or snippet
            if not is_downloadable:
                # Try extracting paper ID from URL first
                paper_info = extract_paper_id(url)

                # If not found in URL, try snippet
                if not paper_info:
                    paper_info = extract_paper_id(result.snippet or '')

                if paper_info:
                    paper_id, repository_type = paper_info
                    # Construct PDF URL from paper ID
                    download_url = get_pdf_url_from_paper_id(
                        paper_id, repository_type, url
                    )
                    if download_url:
                        is_downloadable = True
                        self.logger.info(
                            f'Extracted paper ID {paper_id} from URL/snippet, constructing PDF URL: {download_url}'
                        )

            # Determine if URL is a file (using new helper method)
            is_file = self._is_file_url(url) or is_downloadable

            if is_file:
                # It's a file - download it
                try:
                    self.logger.info(
                        f'Downloading file from search result: {download_url}'
                    )
                    attachment = self.tool_belt.download_file_from_url(download_url)
                    attachments.append(attachment)
                    downloaded_count += 1
                    self.logger.info(
                        f'Added attachment {attachment.filename} from search result '
                        f'(subtask {subtask_id}). Total attachments: {len(attachments)}'
                    )
                except Exception as e:
                    self.logger.warning(
                        f'Failed to download file from {download_url}: {e}. Continuing...'
                    )
                    continue
            else:
                # Not a file - check relevance and navigate if relevant
                try:
                    is_relevant = self._is_result_relevant(
                        result, problem, query_analysis
                    )
                    if is_relevant:
                        self.logger.info(f'Navigating to relevant non-file page: {url}')
                        # Navigate to the page (this will load content and add to state)
                        nav_result = self.tool_belt.browser_navigate(url)
                        navigated_count += 1
                        self.logger.info(
                            f'Successfully navigated to {url}. '
                            f'Page loaded: {nav_result.get("success", False)}'
                        )
                    else:
                        self.logger.info(f'Skipping non-relevant page: {url}')
                except Exception as e:
                    self.logger.warning(
                        f'Failed to process search result {url}: {e}. Continuing...'
                    )
                    continue

        if downloaded_count > 0 or navigated_count > 0:
            self.logger.info(
                f'Processed {len(search_results)} search results: '
                f'Downloaded {downloaded_count} file(s), navigated to {navigated_count} page(s). '
                f'Total attachments now: {len(attachments)}'
            )
