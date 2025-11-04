"""Execution Engine for orchestrating tool operations."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .json_utils import extract_json_from_text
from .llm_service import LLMService
from .models import Attachment
from .search_result_processor import SearchResultProcessor
from .state_manager import InformationStateManager, Subtask
from .tool_belt import ToolBelt


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
        # Initialize search result processor for systematic search result handling
        self.search_processor = SearchResultProcessor(
            llm_service=llm_service,
            tool_belt=tool_belt,
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
            # Only skip search for: code_interpreter, read_attachment, analyze_media
            should_search_first = tool_name not in [
                'code_interpreter',
                'read_attachment',
                'analyze_media',
            ]

            # If tool is browser_navigate or unknown, convert to search
            if should_search_first and tool_name != 'search':
                # Get LLM-generated search_query from metadata
                search_query = subtask.metadata.get('search_query', '')
                if not search_query:
                    # Fallback to description if search_query not provided
                    self.logger.warning(
                        f'Subtask {subtask.id} missing search_query in metadata. '
                        f'Using description as fallback.'
                    )
                    search_query = subtask.description

                self.logger.info(
                    f'Converting tool "{tool_name}" to search-first workflow. '
                    f'Will search with LLM-generated query: "{search_query}"'
                )
                tool_name = 'search'
                # Use LLM-generated search_query from metadata
                if 'query' not in parameters:
                    parameters['query'] = search_query

            # Execute based on tool type
            if tool_name == 'code_interpreter':
                code = parameters.get('code', '')
                context = parameters.get('context', {})
                self.logger.debug(
                    f'Executing code_interpreter with code length: {len(code)} chars'
                )
                result = self.tool_belt.code_interpreter(code, context)

                # Check if result indicates PDF import error - convert to search if needed
                if isinstance(result, str) and (
                    'PyPDF2' in result
                    or 'pdfplumber' in result.lower()
                    or 'pdf processing' in result.lower()
                ):
                    self.logger.warning(
                        'Code execution attempted PDF parsing, but PDF libraries are not available. '
                        'Converting subtask to use search instead.'
                    )
                    # Get LLM-generated search_query from metadata, fallback to description
                    search_query = subtask.metadata.get(
                        'search_query', subtask.description
                    )
                    # Try to make it more specific for PDFs if not already
                    if 'pdf' not in search_query.lower():
                        search_query = f'{search_query} PDF'
                    self.logger.info(
                        f'Executing search instead using LLM-generated query: "{search_query}"'
                    )
                    search_results = self.tool_belt.search(
                        search_query, num_results=5, search_type='web'
                    )

                    if search_results:
                        processing_result = (
                            self.search_processor.process_search_results(
                                search_results=search_results,
                                subtask_description=subtask.description,
                                problem=problem,
                                query_analysis=query_analysis,
                                attachments=attachments,
                                max_results_to_process=5,
                            )
                        )
                        result = {
                            'search_results': search_results,
                            'processing_summary': processing_result,
                            'content': processing_result.get('content_summary', ''),
                            'relevant_count': processing_result.get(
                                'relevant_count', 0
                            ),
                            'web_pages': processing_result.get('web_pages', []),
                            'downloaded_files': processing_result.get(
                                'downloaded_files', []
                            ),
                            'note': 'Converted from code_interpreter to search due to PDF processing requirement',
                        }
                    else:
                        result = {
                            'error': 'PDF processing not available via code. Converted to search but no results found.',
                            'original_error': result,
                            'suggestion': 'Use search queries to find PDFs, then read_attachment to extract content.',
                        }
            elif tool_name == 'search':
                # Prioritize LLM-generated search_query from metadata
                search_query = subtask.metadata.get('search_query', '')
                if search_query:
                    # Use LLM-generated search query if available
                    query = parameters.get('query', search_query)
                    self.logger.info(
                        f'Using LLM-generated search_query from planner: "{search_query}"'
                    )
                else:
                    # Fallback to parameters or description
                    query = parameters.get('query', subtask.description)
                    if query == subtask.description:
                        self.logger.warning(
                            f'Subtask {subtask.id} missing search_query in metadata. '
                            f'Using description as fallback query: "{query}"'
                        )

                num_results = parameters.get('num_results', 5)
                search_type = parameters.get('search_type', 'web')
                self.logger.debug(
                    f'Executing search: query="{query}", type={search_type}, num_results={num_results}'
                )
                search_results = self.tool_belt.search(query, num_results, search_type)

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
                                            from .models import SearchResult

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
                        max_results_to_process=5,
                    )

                    # Return structured processing result along with original search results
                    result = {
                        'search_results': search_results,
                        'processing_summary': processing_result,
                        'content': processing_result.get('content_summary', ''),
                        'relevant_count': processing_result.get('relevant_count', 0),
                        'web_pages': processing_result.get('web_pages', []),
                        'downloaded_files': processing_result.get(
                            'downloaded_files', []
                        ),
                    }
                else:
                    result = {
                        'search_results': [],
                        'processing_summary': {},
                        'content': '',
                    }
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
                    result = self.tool_belt.read_attachment(attachment, options)
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

            # Summarize result for logging
            result_str = str(result)
            result_summary = (
                result_str[:200] + '...' if len(result_str) > 200 else result_str
            )
            result_type = type(result).__name__

            self.state_manager.complete_subtask(subtask.id, result)

            self.logger.info('-' * 80)
            self.logger.info(f'AFTER executing subtask: {subtask.id}')
            self.logger.info('  Status: completed')
            self.logger.info(f'  Result type: {result_type}')
            self.logger.info(f'  Result summary: {result_summary}')
            self.logger.info(f'  Result length: {len(result_str)} chars')
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

            return result
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
- **code_interpreter limitations**: Can only use built-in Python functions and whitelisted modules (math, json, datetime, re, etc.).
  - CANNOT import: PyPDF2, pdfplumber, os, sys, subprocess, or any file parsing libraries
  - For PDF processing: DO NOT generate code that imports PDF libraries. Instead, suggest using search + read_attachment tools
  - For file operations: Use search to find files, then read_attachment to extract content
- **Use search for information gathering**: When you need to find PDFs, documents, or information, generate a specific search query instead of code
- **PDF processing**: Use search to locate PDFs, then read_attachment to extract text. Never try to parse PDFs with code.
- **Search query priority**: If a search_query is provided in the subtask information, you MUST use that exact query. Do not modify or replace it.

Return a JSON object with parameters specific to the tool type.
For code_interpreter: {"code": "python code", "context": {...}}
  - IMPORTANT: Only generate code that uses built-in Python functions (math, json, datetime, re, etc.)
  - DO NOT include imports for PyPDF2, pdfplumber, or any file parsing libraries
  - If the task requires PDF processing, return an error message suggesting use of search + read_attachment instead
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

        # Get LLM-generated search_query from metadata if available
        search_query = subtask.metadata.get('search_query', '')
        search_query_info = ''
        if search_query and subtask.metadata.get('tool') == 'search':
            search_query_info = (
                f'\nLLM-generated search_query (MUST USE): {search_query}'
            )

        user_prompt = f"""Problem: {problem}

Subtask: {subtask.description}
Tool: {subtask.metadata.get('tool', 'unknown')}{search_query_info}{attachment_info}

Determine the appropriate tool parameters.
IMPORTANT: If the tool is 'search' and a search_query is provided above, you MUST use that exact search_query in the parameters."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
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
                temperature=1.0,
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
    ) -> List[Any]:
        """
        Use LLM to filter and rank search results by relevance to the query.

        Args:
            search_results: List of SearchResult objects from search tool.
            query: The search query that produced these results.
            problem: The original problem being solved.
            query_analysis: Optional query analysis results containing requirements and constraints.

        Returns:
            Filtered list of most relevant SearchResult objects.
        """
        from .models import SearchResult

        if not search_results:
            return search_results

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
Given a search query, query requirements analysis, and a list of search results, identify which results are most relevant to answering the query.

Consider the explicit and implicit requirements from the query analysis when determining relevance.
A result is relevant if it helps satisfy any of the stated requirements or constraints.

Return a JSON object with:
- selected_indices: list of integers representing the indices (0-based) of the most relevant results
- reasoning: brief explanation of why these results were selected, specifically referencing which requirements they address
- max_results: limit to at most 3-5 most relevant results, prioritizing quality over quantity

Focus on:
- Direct relevance to the query terms and intent
- Alignment with explicit and implicit requirements from query analysis
- How well each result addresses the specific requirements and constraints
- Information quality and usefulness for satisfying the problem requirements"""

        user_prompt = f"""Original Problem: {problem}

Search Query: {query}
{requirements_context}
Search Results:
{json.dumps(results_list, indent=2)}

Identify the most relevant search results for answering the query. Pay special attention to which results best satisfy the requirements identified in the query analysis."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,  # Lower temperature for more consistent relevance scoring
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            selected_indices = result_data.get('selected_indices', [])
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'LLM selected {len(selected_indices)} most relevant result(s) out of {len(search_results)}. Reasoning: {reasoning}'
            )

            # Filter results to only include selected indices
            filtered_results = []
            for idx in selected_indices:
                if 0 <= idx < len(search_results):
                    filtered_results.append(search_results[idx])
                else:
                    self.logger.warning(
                        f'Invalid index {idx} in selected_indices (valid range: 0-{len(search_results) - 1})'
                    )

            # If no valid results were selected, return original results as fallback
            if not filtered_results:
                self.logger.warning(
                    'No valid results selected by LLM, using all original results'
                )
                return search_results

            self.logger.info(
                f'Filtered search results: {len(filtered_results)} relevant result(s) selected from {len(search_results)} total'
            )
            return filtered_results

        except Exception as e:
            self.logger.error(
                f'Failed to filter search results by relevance: {e}', exc_info=True
            )
            # Fallback to original results if filtering fails
            self.logger.warning('Using all original search results as fallback')
            return search_results

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
- reasoning: brief explanation of why it is or isn't relevant"""

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
        from .models import SearchResult

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
