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
from .executor_utils import determine_tool_parameters
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

    def _get_search_queries_from_metadata(self, subtask: Subtask) -> List[str]:
        """Extract search_queries from subtask metadata with backward compatibility."""
        search_queries = subtask.metadata.get('search_queries', [])
        if not search_queries:
            old_search_query = subtask.metadata.get('search_query', '')
            if old_search_query:
                search_queries = [old_search_query]
        return search_queries

    def _log_subtask_start(self, subtask: Subtask) -> Dict[str, Any]:
        """Log subtask execution start and return state before."""
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
        return state_before

    def _determine_tool_and_parameters(
        self,
        subtask: Subtask,
        problem: str,
        attachments: Optional[List[Attachment]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Determine tool name and parameters, handling search_queries."""
        tool_name = subtask.metadata.get('tool', 'unknown')
        parameters = subtask.metadata.get('parameters', {})
        search_queries = self._get_search_queries_from_metadata(subtask)

        # Skip LLM call for 'search' tool when search_queries are already provided
        should_determine_params = (not parameters or tool_name == 'unknown') and not (
            tool_name == 'search' and search_queries
        )

        if should_determine_params:
            self.logger.debug(
                f'Determining parameters for subtask {subtask.id} (parameters not provided)'
            )
            parameters = determine_tool_parameters(
                subtask=subtask,
                problem=problem,
                llm_service=self.llm_service,
                state_manager=self.state_manager,
                attachments=attachments,
                logger=self.logger,
            )
            self.logger.debug(
                f'Determined parameters: {json.dumps(parameters, default=str)[:200]}...'
            )
        elif tool_name == 'search' and search_queries:
            if not parameters:
                parameters = {'num_results': 10}
            self.logger.debug(
                f'Skipping parameter determination for search tool - using search_queries from metadata: {search_queries}'
            )

        self.logger.info(
            f'  Parameters: {json.dumps(parameters, default=str)[:300]}...'
        )
        self.logger.info('-' * 80)
        return tool_name, parameters

    def _normalize_tool_for_search_first(
        self, tool_name: str, subtask: Subtask, parameters: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Convert browser_navigate/unknown tools to search-first workflow."""
        should_search_first = tool_name not in [
            'llm_reasoning',
            'code_interpreter',
            'read_attachment',
            'analyze_media',
        ] and not tool_name.endswith('_api')

        if should_search_first and tool_name != 'search':
            search_queries = self._get_search_queries_from_metadata(subtask)
            if not search_queries:
                self.logger.warning(
                    f'Subtask {subtask.id} missing search_queries in metadata. '
                    f'Using description as fallback.'
                )
                search_queries = [subtask.description]

            search_query = search_queries[0] if search_queries else subtask.description
            self.logger.info(
                f'Converting tool "{tool_name}" to search-first workflow. '
                f'Will search with {len(search_queries)} queries, first query: "{search_query}"'
            )
            tool_name = 'search'
            if 'query' not in parameters:
                parameters['query'] = search_query

        return tool_name, parameters

    def _build_llm_reasoning_context(
        self, subtask: Subtask
    ) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """Build context for LLM reasoning from dependencies and previous steps."""
        previous_results = {}
        missing_dependencies = []

        # Collect direct dependencies
        if subtask.dependencies:
            for dep_id in subtask.dependencies:
                if dep_id in self.state_manager.subtasks:
                    dep_subtask = self.state_manager.subtasks[dep_id]
                    if (
                        dep_subtask.status == 'completed'
                        and dep_subtask.result is not None
                    ):
                        previous_results[dep_id] = self._serialize_result_for_code(
                            dep_subtask.result
                        )
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

            # Fail if all dependencies are missing
            if missing_dependencies and not previous_results:
                error_msg = (
                    f'Cannot execute LLM reasoning - all dependencies are missing or incomplete: '
                    f'{", ".join(missing_dependencies)}. '
                    f'No dependencies available to proceed.'
                )
                self.logger.error(error_msg)
                self.state_manager.fail_subtask(subtask.id, error_msg)
                subtask.metadata['error'] = error_msg
                subtask.metadata['error_type'] = 'missing_dependencies'
                raise ValueError(error_msg)
            elif missing_dependencies:
                warning_msg = (
                    f'Some dependencies are missing or incomplete: {", ".join(missing_dependencies)}. '
                    f'Proceeding with available dependencies: {", ".join(previous_results.keys())}.'
                )
                self.logger.warning(warning_msg)

        # Add previous steps for context
        try:
            current_step_num = (
                int(subtask.id.split('_')[1]) if '_' in subtask.id else None
            )
        except (ValueError, IndexError):
            current_step_num = None

        if current_step_num is not None:
            for step_num in range(1, current_step_num):
                prev_step_id = f'step_{step_num}'
                if prev_step_id in self.state_manager.subtasks:
                    prev_subtask = self.state_manager.subtasks[prev_step_id]
                    if (
                        prev_subtask.status == 'completed'
                        and prev_subtask.result is not None
                        and prev_step_id not in previous_results
                    ):
                        previous_results[prev_step_id] = (
                            self._serialize_result_for_code(prev_subtask.result)
                        )

        # Build structured context with materials
        context = {}
        for step_id, step_result in previous_results.items():
            materials = self._extract_materials_from_result(step_id, step_result)
            summary = self._extract_summary_from_result(step_result)
            additional_info = (
                step_result.get('additional_information', '')
                if isinstance(step_result, dict)
                else ''
            )

            context[step_id] = {
                'materials': materials,
                'summary': summary,
                'full_result': step_result,
            }

            if additional_info:
                context[f'{step_id} additional_information'] = additional_info
                simplified_key = step_id.replace('step_', 'step')
                if simplified_key != step_id:
                    context[f'{simplified_key} additional_information'] = (
                        additional_info
                    )

        context['dependency_results'] = previous_results
        return context, previous_results, missing_dependencies

    def _extract_materials_from_result(
        self, step_id: str, step_result: Any
    ) -> List[Dict[str, Any]]:
        """Extract materials array from step result."""
        materials = []
        if isinstance(step_result, dict):
            if 'materials' in step_result:
                materials = step_result['materials']
            elif step_result.get('type') == 'pdf' and 'sections' in step_result:
                filename = step_result.get('filename', '')
                url = step_result.get('url', '')
                if not url and step_id in self.state_manager.subtasks:
                    subtask = self.state_manager.subtasks[step_id]
                    pdf_data = subtask.metadata.get('pdf_data', {})
                    url = pdf_data.get('url', '') or url

                materials.append(
                    {
                        'type': 'pdf',
                        'title': filename or 'PDF',
                        'url': url,
                        'sections': step_result.get('sections', []),
                        'image_analysis': step_result.get('image_analysis', ''),
                        'content': step_result.get('formatted_text', '')
                        or step_result.get('full_text', ''),
                    }
                )
            elif 'processing_summary' in step_result:
                processing = step_result.get('processing_summary', {})
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
                    if file_data.get('type') == 'pdf' and 'sections' in file_data:
                        materials.append(
                            {
                                'type': 'pdf',
                                'title': file_data.get('title', '')
                                or file_data.get('url', '').split('/')[-1]
                                or 'PDF',
                                'url': file_data.get('url', ''),
                                'sections': file_data.get('sections', []),
                                'image_analysis': file_data.get('image_analysis', ''),
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

        # Fallback: check subtask metadata
        if not materials and step_id in self.state_manager.subtasks:
            subtask = self.state_manager.subtasks[step_id]
            pdf_data = subtask.metadata.get('pdf_data', {})
            if pdf_data and pdf_data.get('sections'):
                materials.append(
                    {
                        'type': 'pdf',
                        'title': pdf_data.get('filename', 'PDF'),
                        'url': '',
                        'sections': pdf_data.get('sections', []),
                        'image_analysis': pdf_data.get('image_analysis', ''),
                        'content': pdf_data.get('full_text', ''),
                    }
                )

        return materials

    def _extract_summary_from_result(self, step_result: Any) -> str:
        """Extract summary text from step result."""
        if isinstance(step_result, dict):
            if step_result.get('type') == 'pdf' and 'formatted_text' in step_result:
                return step_result.get('formatted_text', '')
            return step_result.get('content', '') or str(step_result)
        return str(step_result)

    def _execute_llm_reasoning(
        self,
        subtask: Subtask,
        parameters: Dict[str, Any],
        attachments: Optional[List[Attachment]] = None,
    ) -> Any:
        """Execute LLM reasoning tool."""
        code = parameters.get('code', '')
        task_description = parameters.get('task_description', '')
        context = parameters.get('context', {})

        # Convert code to task description if needed
        if code and not task_description:
            task_description = f'Execute the following Python code logic: {code}'
        elif not task_description:
            task_description = subtask.description

        # Build context from dependencies and previous steps
        context, previous_results, _ = self._build_llm_reasoning_context(subtask)

        materials_count = sum(
            len(ctx.get('materials', []))
            for ctx in context.values()
            if isinstance(ctx, dict) and 'materials' in ctx
        )
        self.logger.info(
            f'Added {len(previous_results)} previous step result(s) to LLM reasoning context '
            f'({materials_count} materials total)'
        )

        # Check for images and use visual LLM if needed
        images = self._extract_images_from_context(context, attachments)
        if images:
            self.logger.info(
                f'Images detected in subtask {subtask.id}, using visual LLM for processing'
            )
            return self.tool_belt.llm_reasoning_with_images(
                task_description, context, images
            )
        else:
            return self.tool_belt.llm_reasoning(task_description, context)

    def _normalize_search_queries(
        self, subtask: Subtask, parameters: Dict[str, Any]
    ) -> List[str]:
        """Normalize search queries to exactly 3 queries."""
        search_queries = self._get_search_queries_from_metadata(subtask)

        if not search_queries:
            old_search_query = subtask.metadata.get('search_query', '')
            if old_search_query:
                search_queries = [old_search_query]

        if not search_queries:
            fallback_query = parameters.get('query', subtask.description)
            search_queries = [fallback_query]
            if fallback_query == subtask.description:
                self.logger.warning(
                    f'Subtask {subtask.id} missing search_queries in metadata. '
                    f'Using description as fallback query: "{fallback_query}"'
                )

        # Ensure exactly 3 queries
        if len(search_queries) < 3:
            while len(search_queries) < 3:
                search_queries.append(
                    search_queries[-1] if search_queries else subtask.description
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

        return search_queries

    def _perform_search_queries(
        self,
        search_queries: List[str],
        num_results: int,
        seen_urls: Optional[set] = None,
    ) -> List[Any]:
        """Execute multiple search queries and combine results."""
        if seen_urls is None:
            seen_urls = set()

        all_results = []
        for idx, query in enumerate(search_queries, 1):
            self.logger.info(
                f'Search query {idx}/{len(search_queries)}: "{query}" '
                f'(num_results={num_results})'
            )
            query_results = self.tool_belt.search(query, num_results)

            for result in query_results:
                url = (
                    result.url
                    if hasattr(result, 'url')
                    else result.get('url', '')
                    if isinstance(result, dict)
                    else str(result)
                )
                # Filter out Hugging Face GAIA-Subset-Benchmark dataset URL
                if (
                    url
                    and 'huggingface.co/datasets/Intelligent-Internet/GAIA-Subset-Benchmark'
                    in url
                ):
                    self.logger.debug(
                        f'Filtered out Hugging Face GAIA-Subset-Benchmark URL from search results: {url}'
                    )
                    continue
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)

            self.logger.debug(
                f'Query {idx} returned {len(query_results)} results, '
                f'{len(all_results)} total unique results so far'
            )

        return all_results

    def _execute_search(
        self,
        subtask: Subtask,
        problem: str,
        parameters: Dict[str, Any],
        query_analysis: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Attachment]] = None,
    ) -> Any:
        """Execute search tool with filtering, retry, and processing."""
        # Add problem to parameters for paper exploration detection
        if 'problem' not in parameters:
            parameters['problem'] = problem
        search_queries = self._normalize_search_queries(subtask, parameters)
        num_results = parameters.get('num_results', 5)

        self.logger.info(
            f'Executing {len(search_queries)} different search queries for subtask {subtask.id}'
        )

        # Perform initial search
        search_results = self._perform_search_queries(search_queries, num_results)
        self.logger.info(
            f'Combined search results: {len(search_results)} unique results from {len(search_queries)} queries'
        )

        # Try to identify downloadable resources if no results
        if not search_results:
            self._try_download_resources(
                problem,
                subtask.description,
                search_queries[0] if search_queries else '',
                attachments,
            )

        # Process search results
        if search_results:
            return self._process_search_results(
                search_results, subtask, problem, query_analysis, attachments
            )
        else:
            return self._handle_empty_search_results(subtask, problem, query_analysis)

    def _try_download_resources(
        self,
        problem: str,
        subtask_description: str,
        query: str,
        attachments: Optional[List[Attachment]],
    ) -> None:
        """Try to identify and download resources when search returns no results."""
        try:
            resources = self.search_handler.identify_downloadable_resources(
                problem, subtask_description, query
            )
            if resources and attachments is not None:
                for resource in resources:
                    if resource.get('url'):
                        try:
                            attachment = self.tool_belt.download_file_from_url(
                                resource['url']
                            )
                            attachments.append(attachment)
                            self.logger.info(
                                f'Downloaded resource: {resource.get("title", "Downloaded resource")} from {resource["url"]}'
                            )
                        except Exception as e:
                            self.logger.warning(
                                f'Failed to download resource from {resource["url"]}: {e}'
                            )
        except Exception as e:
            self.logger.debug(f'Resource identification failed: {e}')

    def _process_search_results(
        self,
        search_results: List[Any],
        subtask: Subtask,
        problem: str,
        query_analysis: Optional[Dict[str, Any]],
        attachments: Optional[List[Attachment]],
    ) -> Any:
        """Process search results and return analysis."""
        self.logger.info(
            f'Processing {len(search_results)} search results systematically...'
        )
        processing_result = self.search_processor.process_search_results(
            search_results=search_results,
            subtask_description=subtask.description,
            problem=problem,
            query_analysis=query_analysis,
            attachments=attachments,
            max_results_to_process=len(search_results),
        )

        # Build materials array
        materials = []
        for web_page in processing_result.get('web_pages', []):
            materials.append(
                {
                    'type': 'web_page',
                    'title': web_page.get('title', ''),
                    'url': web_page.get('url', ''),
                    'content': web_page.get('content', ''),
                    'image_analysis': web_page.get('image_analysis', ''),
                }
            )

        for file_data in processing_result.get('downloaded_files', []):
            if file_data.get('type') == 'pdf' and 'sections' in file_data:
                sections = file_data.get('sections', [])
                relevant_pages = sorted(
                    set(s.get('page') for s in sections if s.get('page'))
                )
                materials.append(
                    {
                        'type': 'pdf',
                        'title': file_data.get('title', '')
                        or file_data.get('url', '').split('/')[-1]
                        or 'PDF',
                        'url': file_data.get('url', ''),
                        'sections': sections,
                        'relevant_pages': relevant_pages,
                        'image_analysis': file_data.get('image_analysis', ''),
                        'content': file_data.get('content', ''),
                    }
                )
            else:
                materials.append(
                    {
                        'type': 'file' if file_data.get('type') != 'pdf' else 'pdf',
                        'title': file_data.get('title', '')
                        or file_data.get('url', '').split('/')[-1]
                        or 'File',
                        'url': file_data.get('url', ''),
                        'content': file_data.get('content', ''),
                    }
                )

        # Analyze with LLM
        llm_analysis = self.search_handler.analyze_search_results_with_llm(
            processing_result=processing_result,
            materials=materials,
            subtask_description=subtask.description,
            problem=problem,
            query_analysis=query_analysis,
            subtask_id=subtask.id,
        )

        # Store metadata
        subtask.metadata['search_analysis'] = llm_analysis
        subtask.metadata['search_metadata'] = {
            'search_results': search_results,
            'materials': materials,
            'processing_summary': processing_result,
            'relevant_count': processing_result.get('relevant_count', 0),
            'web_pages': processing_result.get('web_pages', []),
            'downloaded_files': processing_result.get('downloaded_files', []),
        }

        return llm_analysis

    def _handle_empty_search_results(
        self,
        subtask: Subtask,
        problem: str,
        query_analysis: Optional[Dict[str, Any]],
    ) -> Any:
        """Handle case when search returns no results."""
        llm_analysis = self.search_handler.analyze_search_results_with_llm(
            processing_result={
                'content_summary': '',
                'web_pages': [],
                'downloaded_files': [],
            },
            materials=[],
            subtask_description=subtask.description,
            problem=problem,
            query_analysis=query_analysis,
            subtask_id=subtask.id,
        )
        subtask.metadata['search_analysis'] = llm_analysis
        return llm_analysis.get('content', 'No search results found.')

    def _execute_browser_navigate(
        self, subtask: Subtask, parameters: Dict[str, Any]
    ) -> Any:
        """Execute browser_navigate tool."""
        url = parameters.get('url', '')
        action = parameters.get('action', None)
        link_text = parameters.get('link_text', None)
        selector = parameters.get('selector', None)
        extraction_query = parameters.get('extraction_query', None)

        if not url or not url.strip():
            error_msg = (
                f'Invalid parameters for browser_navigate: URL is required but was not provided. '
                f'Parameters received: {json.dumps(parameters, default=str)}'
            )
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'parameters_received': parameters,
            }

        if action in ('extract_count', 'extract_statistics') and not extraction_query:
            extraction_query = subtask.description

        return self.tool_belt.browser_navigate(
            url, action, link_text, selector, extraction_query
        )

    def _execute_read_attachment(
        self,
        subtask: Subtask,
        parameters: Dict[str, Any],
        problem: str,
        query_analysis: Optional[Dict[str, Any]],
        attachments: Optional[List[Attachment]],
    ) -> Any:
        """Execute read_attachment tool."""
        if not attachments:
            return 'Error: No attachments available'

        attachment = attachments[parameters.get('attachment_index', 0)]
        options = parameters.get('options', {})
        result = self.tool_belt.read_attachment(
            attachment, options, problem, query_analysis
        )

        # Store PDF data in metadata if structured
        if (
            isinstance(result, dict)
            and result.get('type') == 'pdf'
            and 'sections' in result
        ):
            url = ''
            if hasattr(attachment, 'metadata') and attachment.metadata:
                url = attachment.metadata.get('source_url', '') or url
            subtask.metadata['pdf_data'] = {
                'filename': result.get('filename', attachment.filename),
                'url': url,
                'sections': result.get('sections', []),
                'image_analysis': result.get('image_analysis', ''),
                'full_text': result.get('full_text', ''),
            }
            self.logger.info(
                f'Extracted {len(result.get("sections", []))} sections from PDF {attachment.filename}'
            )

        return result

    def _execute_analyze_media(
        self,
        parameters: Dict[str, Any],
        attachments: Optional[List[Attachment]],
    ) -> Any:
        """Execute analyze_media tool."""
        if not attachments:
            return 'Error: No attachments available'

        attachment = attachments[parameters.get('attachment_index', 0)]
        analysis_type = parameters.get('analysis_type', 'auto')
        return self.tool_belt.analyze_media(attachment, analysis_type)

    def _execute_api_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any] | List[Dict[str, Any]],
    ) -> Any:
        """
        Execute an API tool.

        Args:
            tool_name: Name of the API tool (e.g., 'github_api', 'wikipedia_api')
            parameters: Parameters for the API call. Can be:
                - A dict with 'method' and parameters (single call)
                - A list of dicts, each with 'function' (or 'method') and 'parameters' (or direct parameters) (chained calls)

        Returns:
            API response
        """
        # Extract API name from tool name (remove '_api' suffix)
        api_name = tool_name.replace('_api', '')

        # Check if parameters is a list (multiple API calls)
        if isinstance(parameters, list):
            return self._execute_chained_api_calls(api_name, parameters)

        # Single API call (backward compatible)
        method = parameters.get('method')
        if not method:
            error_msg = f'API tool {tool_name} requires "method" parameter. Received parameters: {json.dumps(parameters, default=str)}'
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'parameters_received': parameters,
            }

        # Remove method from parameters before passing to API
        api_params = {k: v for k, v in parameters.items() if k != 'method'}

        try:
            result = self.tool_belt.call_api(api_name, method, **api_params)
            if result is None:
                return {
                    'success': False,
                    'error': f'API call {api_name}.{method} returned None',
                    'api_name': api_name,
                    'method': method,
                }
            return {
                'success': True,
                'api_name': api_name,
                'method': method,
                'data': result,
            }
        except Exception as e:
            error_msg = f'API call {api_name}.{method} failed: {str(e)}'
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'api_name': api_name,
                'method': method,
                'exception_type': type(e).__name__,
            }

    def _execute_chained_api_calls(
        self,
        api_name: str,
        api_calls: List[Dict[str, Any]],
    ) -> Any:
        """
        Execute a chain of API calls, using results from previous calls.

        Args:
            api_name: Name of the API (e.g., 'wikipedia', 'github')
            api_calls: List of API call specifications. Each dict can have:
                - 'function' (method name) and 'parameters' (dict with method parameters)
                - OR directly contain 'method' and parameters

        Returns:
            Final API response or error dict
        """
        self.logger.info(f'Executing {len(api_calls)} chained API calls for {api_name}')
        previous_results = {}

        for idx, call_spec in enumerate(api_calls, 1):
            self.logger.info(f'Executing API call {idx}/{len(api_calls)}')

            # Extract method name and parameters
            if 'function' in call_spec:
                method = call_spec['function']
                call_params = call_spec.get('parameters', {})
            elif 'method' in call_spec:
                method = call_spec['method']
                call_params = {k: v for k, v in call_spec.items() if k != 'method'}
            else:
                error_msg = f'API call {idx} missing "function" or "method" field. Call spec: {json.dumps(call_spec, default=str)}'
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'api_name': api_name,
                    'call_index': idx,
                }

            # Replace placeholders in parameters with results from previous calls
            resolved_params = self._resolve_parameter_placeholders(
                call_params, previous_results, api_name, method
            )

            self.logger.debug(
                f'API call {idx}: {api_name}.{method} with params: {json.dumps(resolved_params, default=str)[:200]}...'
            )

            try:
                result = self.tool_belt.call_api(api_name, method, **resolved_params)
                if result is None:
                    error_msg = f'API call {api_name}.{method} (call {idx}/{len(api_calls)}) returned None'
                    self.logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'api_name': api_name,
                        'method': method,
                        'call_index': idx,
                    }

                # Store result for next call
                previous_results[f'call_{idx}'] = result
                previous_results['last_result'] = result
                previous_results['last_method'] = method

                # Extract common fields from result for use in subsequent calls
                self._extract_common_fields(result, previous_results)

                self.logger.info(f'API call {idx}/{len(api_calls)} succeeded')

            except Exception as e:
                error_msg = f'API call {api_name}.{method} (call {idx}/{len(api_calls)}) failed: {str(e)}'
                self.logger.error(error_msg, exc_info=True)
                return {
                    'success': False,
                    'error': error_msg,
                    'api_name': api_name,
                    'method': method,
                    'call_index': idx,
                    'exception_type': type(e).__name__,
                }

        # Return final result
        final_result = previous_results.get('last_result')
        if final_result is None:
            return {
                'success': False,
                'error': 'No results from chained API calls',
                'api_name': api_name,
            }

        return {
            'success': True,
            'api_name': api_name,
            'method': previous_results.get('last_method', 'chained'),
            'data': final_result,
            'all_results': previous_results,
        }

    def _extract_common_fields(
        self,
        result: Any,
        previous_results: Dict[str, Any],
    ) -> None:
        """
        Extract common fields from API result for use in subsequent calls.

        This method automatically extracts commonly used fields from API results
        and stores them in previous_results for use in chained API calls.

        Args:
            result: The API result to extract fields from
            previous_results: Dict to store extracted fields
        """
        if isinstance(result, list) and len(result) > 0:
            # If result is a list, extract from first item
            first_item = result[0]
            if isinstance(first_item, dict):
                # Extract common ID fields
                for field_name in [
                    'revid',
                    'revision_id',
                    'id',
                    'identifier',
                    'rev_id',
                ]:
                    if field_name in first_item:
                        value = first_item[field_name]
                        # Validate: must not be None, empty string, or empty value
                        if value is not None and value != '' and value != []:
                            # Store with both specific and generic names
                            previous_results[field_name] = value
                            # Also store as generic 'id' if it's a common identifier
                            if field_name in ['revid', 'revision_id', 'rev_id']:
                                previous_results['revision_id'] = value
                            self.logger.debug(
                                f'Extracted {field_name}={value} from result for use in subsequent calls'
                            )
        elif isinstance(result, dict):
            # Extract common fields directly from dict result
            for field_name in [
                'revid',
                'revision_id',
                'id',
                'identifier',
                'rev_id',
                'revision',
            ]:
                if field_name in result:
                    value = result[field_name]
                    # Validate: must not be None, empty string, or empty value
                    if value is not None and value != '' and value != []:
                        previous_results[field_name] = value
                        # Also store as generic 'id' if it's a common identifier
                        if field_name in ['revid', 'revision_id', 'rev_id', 'revision']:
                            previous_results['revision_id'] = value
                        self.logger.debug(
                            f'Extracted {field_name}={value} from result for use in subsequent calls'
                        )

    def _resolve_parameter_placeholders(
        self,
        params: Dict[str, Any],
        previous_results: Dict[str, Any],
        api_name: str,
        method: str,
    ) -> Dict[str, Any]:
        """
        Resolve parameter placeholders using results from previous API calls.

        Supports general placeholders:
        - "<from previous call>" or "<field_name>" - extracts field from previous results
        - "<last_result>" - uses the entire last result
        - "<last_result[0].field>" - extracts nested field from last result

        Args:
            params: Parameters dict that may contain placeholders
            previous_results: Results from previous API calls
            api_name: Name of the API
            method: Method name for this call

        Returns:
            Resolved parameters dict
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('<') and value.endswith('>'):
                # This is a placeholder
                placeholder = value[1:-1].strip()  # Remove < and >

                # Try to resolve the placeholder
                resolved_value = self._resolve_single_placeholder(
                    placeholder, previous_results, key
                )

                if resolved_value is not None:
                    resolved[key] = resolved_value
                    self.logger.debug(
                        f'Resolved {key} placeholder "{value}" to: {resolved_value}'
                    )
                else:
                    # If we can't resolve, skip this parameter to avoid passing invalid value
                    self.logger.warning(
                        f'Could not resolve placeholder {key}={value}, skipping parameter'
                    )
                    continue
            else:
                resolved[key] = value

        return resolved

    def _resolve_single_placeholder(
        self,
        placeholder: str,
        previous_results: Dict[str, Any],
        param_name: str,
    ) -> Any:
        """
        Resolve a single placeholder string to a value from previous results.

        Supports general placeholders:
        - Direct key lookup: "<field_name>" - looks for field_name in previous_results
        - Step-specific: "<field_name_from_step_N>" - looks for field_name_from_step_N in previous_results
        - Generic: "<from previous call>" - tries to infer field from parameter name
        - Nested: "<last_result[0].field>" - extracts nested field from last result

        Args:
            placeholder: The placeholder text (without < >)
            previous_results: Results from previous API calls and previous subtasks
            param_name: Name of the parameter (for context)

        Returns:
            Resolved value or None if cannot be resolved
        """
        # Direct key lookup (handles both generic and step-specific keys)
        if placeholder in previous_results:
            value = previous_results[placeholder]
            # Convert to appropriate type if needed
            return self._convert_value_for_parameter(value, param_name)

        # Handle step-specific placeholders (e.g., "revision_id_from_step_1")
        # This pattern is already handled by direct lookup above, but we can also
        # try to extract from the step result if the key doesn't exist
        if '_from_' in placeholder:
            # Try to extract step number and field name
            # Format: "field_name_from_step_N" or "field_name_from_stepN"
            parts = placeholder.split('_from_')
            if len(parts) == 2:
                field_name = parts[0]
                step_ref = parts[1]

                # Try to find the step result and extract the field
                if step_ref in previous_results:
                    step_result = previous_results[step_ref]
                    # Extract field from step result
                    extracted = self._extract_field_from_any_structure(
                        step_result, field_name
                    )
                    if extracted is not None:
                        return self._convert_value_for_parameter(extracted, param_name)

        # Special cases
        if placeholder == 'from previous call':
            # Try common field names based on parameter name
            if 'revision' in param_name.lower() or 'rev' in param_name.lower():
                # Try to find revision_id
                for field in ['revision_id', 'revid', 'rev_id']:
                    if field in previous_results:
                        return self._convert_value_for_parameter(
                            previous_results[field], param_name
                        )
                # Try to extract from last_result
                return self._extract_from_last_result(
                    previous_results, ['revid', 'revision_id', 'rev_id'], param_name
                )
            else:
                # Generic: try to extract from last_result
                last_result = previous_results.get('last_result')
                if last_result is not None:
                    return last_result

        # Try nested extraction: "last_result[0].field"
        if placeholder.startswith('last_result'):
            return self._extract_nested_value(placeholder, previous_results, param_name)

        # Try to extract from last_result with field name
        if placeholder in ['id', 'identifier', 'revision_id', 'revid']:
            return self._extract_from_last_result(
                previous_results, [placeholder], param_name
            )

        return None

    def _extract_field_from_any_structure(
        self,
        data: Any,
        field_name: str,
    ) -> Any:
        """
        Extract a field from any data structure (dict, list, API response, etc.).

        This is a general method that works for any field name without hardcoding
        specific field variations.

        Args:
            data: The data structure to extract from
            field_name: Name of the field to extract (supports dot notation for nested fields)

        Returns:
            Extracted value or None
        """
        # Handle API response structure: {'success': True, 'data': {...}}
        if isinstance(data, dict) and 'data' in data:
            data = data['data']

        # Handle dot notation for nested fields (e.g., "user.id")
        if '.' in field_name:
            parts = field_name.split('.')
            current = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and len(current) > 0:
                    # Try first item in list
                    if isinstance(current[0], dict) and part in current[0]:
                        current = current[0][part]
                    else:
                        return None
                else:
                    return None
            return current

        # If data is a list, check first item
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                # Try exact match first
                if field_name in first_item:
                    return first_item[field_name]
                # Try case-insensitive match
                for key in first_item.keys():
                    if key.lower() == field_name.lower():
                        return first_item[key]

        # If data is a dict, check directly
        if isinstance(data, dict):
            # Try exact match first
            if field_name in data:
                return data[field_name]
            # Try case-insensitive match
            for key in data.keys():
                if key.lower() == field_name.lower():
                    return data[key]

        return None

    def _extract_from_last_result(
        self,
        previous_results: Dict[str, Any],
        field_names: List[str],
        param_name: str,
    ) -> Any:
        """
        Extract a field from the last_result in previous_results.

        Args:
            previous_results: Results from previous API calls
            field_names: List of field names to try
            param_name: Name of the parameter (for context)

        Returns:
            Extracted value or None
        """
        last_result = previous_results.get('last_result')
        if last_result is None:
            return None

        # If last_result is a list, check first item
        if isinstance(last_result, list) and len(last_result) > 0:
            first_item = last_result[0]
            if isinstance(first_item, dict):
                for field_name in field_names:
                    if field_name in first_item:
                        value = first_item[field_name]
                        return self._convert_value_for_parameter(value, param_name)

        # If last_result is a dict, check directly
        if isinstance(last_result, dict):
            for field_name in field_names:
                if field_name in last_result:
                    value = last_result[field_name]
                    return self._convert_value_for_parameter(value, param_name)

        return None

    def _extract_nested_value(
        self,
        placeholder: str,
        previous_results: Dict[str, Any],
        param_name: str,
    ) -> Any:
        """
        Extract a nested value from previous results using dot notation.

        Args:
            placeholder: Placeholder like "last_result[0].revid"
            previous_results: Results from previous API calls
            param_name: Name of the parameter (for context)

        Returns:
            Extracted value or None
        """
        try:
            # Parse "last_result[0].field" or "last_result.field"
            parts = placeholder.replace('[', '.').replace(']', '').split('.')
            if parts[0] == 'last_result':
                last_result = previous_results.get('last_result')
                if last_result is None:
                    return None

                # Navigate through the structure
                current = last_result
                for part in parts[1:]:
                    if part.isdigit():
                        # Array index
                        idx = int(part)
                        if isinstance(current, list) and 0 <= idx < len(current):
                            current = current[idx]
                        else:
                            return None
                    else:
                        # Dict key
                        if isinstance(current, dict):
                            current = current.get(part)
                            if current is None:
                                return None
                        else:
                            return None

                return self._convert_value_for_parameter(current, param_name)
        except Exception as e:
            self.logger.debug(f'Error extracting nested value {placeholder}: {e}')
            return None

    def _convert_value_for_parameter(self, value: Any, param_name: str) -> Any:
        """
        Convert a value to the appropriate type for a parameter.

        Args:
            value: The value to convert
            param_name: Name of the parameter (for type hints)

        Returns:
            Converted value, or None if value is invalid
        """
        if value is None:
            return None

        # Reject empty strings and empty collections
        if value == '' or value == []:
            self.logger.warning(
                f'Empty value encountered for parameter {param_name}, returning None'
            )
            return None

        # If parameter name suggests it should be an integer, try to convert
        if any(
            keyword in param_name.lower()
            for keyword in ['id', 'number', 'count', 'index']
        ):
            try:
                converted = int(value)
                # Ensure it's a valid positive integer (IDs should be positive)
                if converted > 0:
                    return converted
                else:
                    self.logger.warning(
                        f'Invalid ID value {converted} for parameter {param_name}, returning None'
                    )
                    return None
            except (ValueError, TypeError):
                # If conversion fails, log and return None instead of original value
                self.logger.warning(
                    f'Could not convert value {value} to integer for parameter {param_name}, returning None'
                )
                return None

        return value

    def _format_and_store_result(
        self, subtask: Subtask, result: Any, state_before: Dict[str, Any]
    ) -> tuple[Any, str, str]:
        """Format result for storage and return result_to_store, result_str, result_type."""
        # Handle structured PDF results
        if (
            isinstance(result, dict)
            and result.get('type') == 'pdf'
            and 'sections' in result
        ):
            sections_text = []
            for section in result.get('sections', []):
                section_title = section.get('title', 'Untitled')
                section_content = section.get('content', '')
                section_page = section.get('page', '?')
                sections_text.append(
                    f'[Section: {section_title} (Page {section_page})]\n{section_content}'
                )

            formatted_result = '\n\n'.join(sections_text)
            if result.get('image_analysis'):
                formatted_result += '\n\nIMAGE ANALYSIS (from visual LLM):\n'
                formatted_result += result.get('image_analysis', '')

            url = ''
            if hasattr(subtask, 'metadata') and subtask.metadata:
                pdf_data = subtask.metadata.get('pdf_data', {})
                url = pdf_data.get('url', '') or url

            sections = result.get('sections', [])
            relevant_pages = sorted(
                set(s.get('page') for s in sections if s.get('page'))
            )

            result_to_store = {
                'type': 'pdf',
                'formatted_text': formatted_result,
                'sections': sections,
                'relevant_pages': relevant_pages,
                'image_analysis': result.get('image_analysis', ''),
                'filename': result.get('filename', ''),
                'url': url,
                'full_text': result.get('full_text', formatted_result),
            }
            result_str = formatted_result
            result_type = 'pdf_with_sections'
        else:
            result_str = str(result)
            result_type = type(result).__name__
            result_to_store = result

        # Store result
        if subtask.id in self.state_manager.subtasks:
            current_subtask = self.state_manager.subtasks[subtask.id]
            if current_subtask.status != 'failed':
                self.state_manager.complete_subtask(subtask.id, result_to_store)
        else:
            self.state_manager.complete_subtask(subtask.id, result_to_store)

        return result_to_store, result_str, result_type

    def _log_subtask_end(
        self,
        subtask: Subtask,
        result_to_store: Any,
        result_str: str,
        result_type: str,
        state_before: Dict[str, Any],
    ) -> None:
        """Log subtask execution end."""
        state_after = self.state_manager.get_state_summary()

        # Extract image analysis if present
        has_image_analysis = False
        image_analysis = None
        if isinstance(result_to_store, dict) and result_to_store.get('type') == 'pdf':
            has_image_analysis = bool(result_to_store.get('image_analysis'))
            image_analysis = result_to_store.get('image_analysis', '')
        elif isinstance(result_to_store, dict) and 'content' in result_to_store:
            has_image_analysis = 'IMAGE ANALYSIS (from visual LLM):' in str(
                result_to_store.get('content', '')
            )
        else:
            has_image_analysis = 'IMAGE ANALYSIS (from visual LLM):' in result_str

        # Create result summary
        if has_image_analysis and isinstance(result_to_store, dict):
            if result_to_store.get('type') == 'pdf':
                sections_count = len(result_to_store.get('sections', []))
                result_summary = (
                    f'PDF with {sections_count} section(s) '
                    f'[+ Image Analysis: {len(image_analysis)} chars - see below]'
                )
            else:
                marker_idx = result_str.find('IMAGE ANALYSIS (from visual LLM):')
                if marker_idx >= 0:
                    text_before = result_str[:marker_idx].strip()
                    result_summary = (
                        text_before[:200] + '...'
                        if len(text_before) > 200
                        else text_before
                    )
                    result_summary += f'\n[+ Image Analysis: {len(image_analysis or "")} chars - see below]'
                else:
                    result_summary = (
                        result_str[:200] + '...'
                        if len(result_str) > 200
                        else result_str
                    )
        elif isinstance(result_to_store, dict) and 'content' in result_to_store:
            content = result_to_store.get('content', '')
            result_summary = content[:200] + '...' if len(content) > 200 else content
            if result_to_store.get('additional_information'):
                result_summary += f'\n[+ Additional Information: {len(result_to_store.get("additional_information", ""))} chars]'
        else:
            result_summary = (
                result_str[:200] + '...' if len(result_str) > 200 else result_str
            )

        self.logger.info('-' * 80)
        self.logger.info(f'AFTER executing subtask: {subtask.id}')
        self.logger.info('  Status: completed')
        self.logger.info(f'  Result type: {result_type}')
        self.logger.info(f'  Result summary: {result_summary}')
        self.logger.info(f'  Result length: {len(result_str)} chars')

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
            query_analysis: Optional query analysis results containing requirements.

        Returns:
            Result of the execution.
        """
        state_before = self._log_subtask_start(subtask)

        # Determine tool and parameters
        tool_name, parameters = self._determine_tool_and_parameters(
            subtask, problem, attachments
        )

        # Normalize tool (convert browser_navigate to search if needed)
        tool_name, parameters = self._normalize_tool_for_search_first(
            tool_name, subtask, parameters
        )

        try:
            # Execute based on tool type
            if tool_name in ('code_interpreter', 'llm_reasoning'):
                result = self._execute_llm_reasoning(subtask, parameters, attachments)
            elif tool_name == 'search':
                result = self._execute_search(
                    subtask, problem, parameters, query_analysis, attachments
                )
            elif tool_name == 'browser_navigate':
                result = self._execute_browser_navigate(subtask, parameters)
            elif tool_name == 'read_attachment':
                result = self._execute_read_attachment(
                    subtask, parameters, problem, query_analysis, attachments
                )
            elif tool_name == 'analyze_media':
                result = self._execute_analyze_media(parameters, attachments)
            elif tool_name.endswith('_api'):
                result = self._execute_api_tool(tool_name, parameters)
            else:
                result = f'Error: Unknown tool {tool_name}'

            # Format and store result
            result_to_store, result_str, result_type = self._format_and_store_result(
                subtask, result, state_before
            )

            # Log completion
            self._log_subtask_end(
                subtask, result_to_store, result_str, result_type, state_before
            )

            return result_to_store
        except ValueError:
            # Re-raise ValueError (from missing dependencies) without logging as error
            raise
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
            query_analysis: Optional query analysis results containing requirements.

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
            is_file = self.search_handler.is_file_url(url) or is_downloadable

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
                    is_relevant = self.search_handler.is_result_relevant(
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
