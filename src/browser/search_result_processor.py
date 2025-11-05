"""Search Result Processor for handling search results with LLM-based relevance checking."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..llm import LLMService
from ..models import Attachment, SearchResult
from ..utils import extract_json_from_text
from .browser import Browser


class SearchResultProcessor:
    """
    Processes search results by:
    1. Determining relevance using LLM
    2. Classifying as web page vs file
    3. Dispatching to appropriate handler (navigate or download)
    4. Extracting and structuring content for next steps
    """

    def __init__(
        self,
        llm_service: LLMService,
        browser: Browser,
        tool_belt: Any,  # ToolBelt instance for file operations
        logger: logging.Logger,
    ):
        """
        Initialize SearchResultProcessor.

        Args:
            llm_service: LLM service for relevance checking.
            browser: Unified Browser instance.
            tool_belt: ToolBelt instance for file operations.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.browser = browser
        self.tool_belt = tool_belt
        self.logger = logger

    def process_search_results(
        self,
        search_results: List[SearchResult],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Attachment]] = None,
        max_results_to_process: int = 5,
    ) -> Dict[str, Any]:
        """
        Process search results by checking relevance, classifying, and handling appropriately.

        For each search result:
        1. Check relevance using LLM
        2. Determine if it's a web page or file
        3. If web page: use selenium navigator
        4. If file: download and add to attachments
        5. Extract and structure content

        Args:
            search_results: List of SearchResult objects to process.
            subtask_description: Description of the subtask being executed.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
            attachments: Optional list to append downloaded files to.
            max_results_to_process: Maximum number of results to process (default: 3).

        Returns:
            Dictionary with:
            - processed_count: Number of results processed
            - relevant_count: Number of relevant results
            - web_pages: List of processed web page results
            - downloaded_files: List of downloaded file attachments
            - content_summary: Aggregated content from all sources
        """
        if not search_results:
            self.logger.info('No search results to process')
            return {
                'processed_count': 0,
                'relevant_count': 0,
                'web_pages': [],
                'downloaded_files': [],
                'content_summary': '',
            }

        if attachments is None:
            attachments = []

        self.logger.info(
            f'Processing {min(len(search_results), max_results_to_process)} search results '
            f'for subtask: {subtask_description}'
        )

        processed_count = 0
        relevant_count = 0
        web_pages = []
        downloaded_files = []
        content_parts = []

        for idx, result in enumerate(search_results[:max_results_to_process]):
            processed_count += 1

            processed_result = self._process_single_result(
                result,
                subtask_description,
                problem,
                query_analysis,
                attachments,
                idx,
                len(search_results),
            )

            if processed_result:
                relevant_count += 1
                if processed_result['type'] == 'web_page':
                    web_pages.append(processed_result['data'])
                    content_parts.append(
                        f'[Web Page: {processed_result["title"]}]\nURL: {processed_result["url"]}\n{processed_result["content"]}'
                    )
                elif processed_result['type'] == 'file':
                    downloaded_files.append(processed_result['data'])
                    content_parts.append(
                        f'[File: {processed_result["title"]}]\n{processed_result["content"]}'
                    )

        # Step 5: Extract and structure content - summarize using LLM
        content_summary = self._summarize_content_with_llm(
            content_parts, problem, subtask_description, query_analysis
        )

        result_summary = {
            'processed_count': processed_count,
            'relevant_count': relevant_count,
            'web_pages': web_pages,
            'downloaded_files': downloaded_files,
            'content_summary': content_summary,
        }

        self.logger.info(
            f'Search result processing complete: '
            f'{processed_count} processed, {relevant_count} relevant, '
            f'{len(web_pages)} web pages, {len(downloaded_files)} files downloaded'
        )

        return result_summary

    def _process_single_result(
        self,
        result: Union[SearchResult, Dict],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]],
        attachments: List[Attachment],
        idx: int,
        total_results: int,
    ) -> Optional[Dict[str, Any]]:
        """Processes a single search result."""
        if isinstance(result, dict):
            result = SearchResult(
                snippet=result.get('snippet', ''),
                url=result.get('url', ''),
                title=result.get('title', ''),
                relevance_score=result.get('relevance_score', 0.0),
            )
        elif not isinstance(result, SearchResult):
            return None

        result_info = f'[{idx + 1}/{total_results}] {result.title}'
        self.logger.info(f'Processing search result {result_info}')

        is_relevant, relevance_reasoning = self._check_relevance(
            result, subtask_description, problem, query_analysis
        )
        if not is_relevant:
            self.logger.info(
                f'Result {result_info} deemed not relevant. Reason: {relevance_reasoning}'
            )
            return None

        self.logger.info(
            f'Result {result_info} is RELEVANT. Reason: {relevance_reasoning}'
        )

        is_file, file_type = self._classify_result_type(result)

        if is_file:
            self.logger.info(
                f'Result {result_info} is a file (type: {file_type}). Downloading...'
            )
            file_content = self._handle_file_result(
                result, attachments, subtask_description, problem, query_analysis
            )
            if file_content:
                return {
                    'type': 'file',
                    'title': result.title,
                    'url': result.url,
                    'content': file_content,
                    'data': {
                        'url': result.url,
                        'type': file_type,
                        'content': file_content,
                    },
                }
        else:
            self.logger.info(f'Result {result_info} is a web page. Navigating...')
            page_content = self._handle_web_page_result(
                result, subtask_description, problem
            )
            if page_content:
                return {
                    'type': 'web_page',
                    'title': result.title,
                    'url': result.url,
                    'content': page_content,
                    'data': {
                        'url': result.url,
                        'title': result.title,
                        'content': page_content,
                    },
                }
        return None

    def _check_relevance(
        self,
        search_result: SearchResult,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        Use LLM to determine if a search result is relevant.

        Args:
            search_result: SearchResult to evaluate.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Tuple of (is_relevant: bool, reasoning: str)
        """
        try:
            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                implicit_reqs = query_analysis.get('implicit_requirements', [])
                constraints = query_analysis.get('constraints', {})

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

            system_prompt = """You are an expert at evaluating whether a search result is relevant to a subtask.
Given a problem description, subtask description, and a search result (title, snippet, URL), determine if this result is likely to contain information useful for completing the subtask.

Consider:
- Direct relevance to the subtask description
- Alignment with problem requirements and constraints
- Quality and credibility of the source
- Whether the result provides actionable information
- **CRITICAL for aggregate/statistical queries**: Archive pages, browse pages, and database pages are relevant even if the snippet doesn't show the exact number. URLs containing "archive", "browse", "articles?year=", "articles?date=", or similar patterns suggest aggregate data pages that should be navigated to extract the information.

Return a JSON object with:
- relevant: boolean indicating if the result is relevant
- reasoning: brief explanation (1-2 sentences) of why it is or isn't relevant
- confidence: float from 0.0 to 1.0 indicating confidence in the relevance assessment

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

Search Result:
- Title: {search_result.title}
- Snippet: {search_result.snippet}
- URL: {search_result.url}

IMPORTANT GUIDANCE FOR AGGREGATE/STATISTICAL QUERIES:
- If the subtask asks for counts, totals, or aggregate data (e.g., "how many articles", "total number"), consider archive/browse pages as RELEVANT even if the snippet doesn't show the number
- URLs containing "archive", "browse", "articles?year=", "articles?date=", or domain-specific archive patterns (e.g., nature.com/articles?year=) are likely archive pages that contain aggregate data
- Individual articles are NOT relevant for aggregate queries - you need archive/browse pages
- For aggregate queries, prioritize pages that can be navigated to extract the count, even if the snippet only shows article titles

Is this search result relevant to completing the subtask?"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent evaluation
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            is_relevant = result_data.get('relevant', False)
            reasoning = result_data.get('reasoning', 'No reasoning provided')
            confidence = result_data.get('confidence', 0.5)

            self.logger.debug(
                f'Relevance check: {is_relevant} (confidence: {confidence:.2f}). '
                f'Reasoning: {reasoning}'
            )

            return is_relevant, reasoning

        except Exception as e:
            self.logger.warning(
                f'Failed to determine relevance using LLM: {e}. Defaulting to relevant=True.'
            )
            # Default to relevant if LLM check fails to avoid skipping potentially useful results
            return True, 'LLM check failed, assuming relevant'

    def _classify_result_type(self, search_result: SearchResult) -> tuple[bool, str]:
        """
        Classify a search result as either a file or web page.

        Args:
            search_result: SearchResult to classify.

        Returns:
            Tuple of (is_file: bool, file_type: str)
            file_type is one of: 'pdf', 'doc', 'spreadsheet', 'image', 'archive', 'text', 'webpage'
        """
        url = search_result.url.lower() if search_result.url else ''
        title = search_result.title.lower() if search_result.title else ''

        # Define file type mappings
        file_type_patterns = {
            'pdf': ['.pdf'],
            'doc': ['.doc', '.docx', '.odt'],
            'spreadsheet': ['.xls', '.xlsx', '.csv', '.ods'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
            'archive': ['.zip', '.tar', '.gz', '.rar', '.7z'],
            'text': ['.txt'],
        }

        # Check URL and title for file extensions
        for file_type, patterns in file_type_patterns.items():
            for pattern in patterns:
                if pattern in url or pattern in title:
                    return True, file_type

        # Check for file download indicators in URL
        file_indicators = ['/pdf/', '/download/', '/file/', '/attachment/', '/doc/']
        if any(indicator in url for indicator in file_indicators):
            # Try to determine specific type
            for file_type, patterns in file_type_patterns.items():
                if any(pattern.strip('.') in url for pattern in patterns):
                    return True, file_type
            return True, 'unknown'

        # If no file indicators found, it's a web page
        return False, 'webpage'

    def _handle_file_result(
        self,
        search_result: SearchResult,
        attachments: List[Attachment],
        subtask_description: str,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Handle a search result that is a file by downloading it.

        Args:
            search_result: SearchResult pointing to a file.
            attachments: List to append downloaded attachment to.
            subtask_description: Description of the subtask.
            problem: Optional problem description for relevance filtering.
            query_analysis: Optional query analysis for relevance filtering.

        Returns:
            String summary of file content if successful, None otherwise.
        """
        try:
            self.logger.info(f'Downloading file from URL: {search_result.url}')
            attachment = self.tool_belt.download_file_from_url(search_result.url)
            attachments.append(attachment)

            # Try to extract text content from the file with relevance filtering
            file_content = self._extract_file_content(
                attachment, problem, query_analysis
            )

            self.logger.info(
                f'Successfully downloaded and processed file: {attachment.filename} '
                f'(size: {len(attachment.data)} bytes, content length: {len(file_content)} chars)'
            )

            return file_content

        except Exception as e:
            self.logger.warning(
                f'Failed to download or process file from {search_result.url}: {e}'
            )
            return None

    def _extract_file_content(
        self,
        attachment: Attachment,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Extract text content from an attachment with optional relevance filtering.

        Args:
            attachment: Attachment to extract content from.
            problem: Optional problem description for relevance filtering (PDFs only).
            query_analysis: Optional query analysis for relevance filtering (PDFs only).

        Returns:
            Extracted text content.
        """
        try:
            # Use tool_belt's read_attachment to extract content with relevance filtering
            content = self.tool_belt.read_attachment(
                attachment, problem=problem, query_analysis=query_analysis
            )
            return content if content else f'[Content of {attachment.filename}]'
        except Exception as e:
            self.logger.debug(
                f'Failed to extract content from {attachment.filename}: {e}'
            )
            return f'[File: {attachment.filename} - content extraction failed]'

    def _handle_web_page_result(
        self,
        search_result: SearchResult,
        subtask_description: str,
        problem: str,
    ) -> Optional[str]:
        """
        Handle a search result that is a web page by navigating with the unified browser.
        """
        try:
            # Simple heuristic to decide if selenium is needed
            use_selenium = True

            self.logger.info(
                f'Navigating to web page: {search_result.url} (using_selenium={use_selenium})'
            )

            nav_result = self.browser.navigate(
                url=search_result.url, use_selenium=use_selenium
            )

            if not nav_result.get('success'):
                error_msg = nav_result.get('error', 'Navigation failed')
                self.logger.warning(
                    f'Failed to navigate to {search_result.url}: {error_msg}'
                )
                # If requests failed, try with selenium
                if not use_selenium:
                    self.logger.info(
                        f'Retrying navigation with Selenium for {search_result.url}'
                    )
                    nav_result = self.browser.navigate(
                        url=search_result.url, use_selenium=True
                    )
                    if not nav_result.get('success'):
                        error_msg = nav_result.get('error', 'Navigation failed again')
                        self.logger.warning(
                            f'Failed to navigate to {search_result.url} with Selenium: {error_msg}'
                        )
                        return None
                else:
                    return None

            # Determine if we need specific extraction based on subtask description
            action = self._determine_extraction_action(subtask_description)

            if action == 'extract_count':
                llm_extract_result = self.browser.extract_with_llm(
                    nav_result, subtask_description, self.llm_service
                )
                return json.dumps(llm_extract_result)
            elif action == 'find_table':
                table_data = self.browser.find_table(nav_result)
                return json.dumps(table_data) if table_data else 'No table found.'
            else:  # default to text extraction
                return self.browser.extract_text(nav_result)

        except Exception as e:
            self.logger.warning(
                f'Failed to navigate or extract content from {search_result.url}: {e}'
            )
            return None

    def _determine_extraction_action(self, subtask_description: str) -> Optional[str]:
        """
        Determine the appropriate extraction action based on subtask description.

        Args:
            subtask_description: Description of the subtask.

        Returns:
            Action string or None for default navigation.
        """
        desc_lower = subtask_description.lower()

        # Map keywords to actions
        if any(
            keyword in desc_lower
            for keyword in ['count', 'number of', 'how many', 'total', 'statistic']
        ):
            return 'extract_count'
        elif any(keyword in desc_lower for keyword in ['table', 'data', 'list']):
            return 'find_table'

        # Default to text extraction
        return 'extract_text'

    def _extract_navigation_content(self, nav_result: Dict[str, Any]) -> str:
        """
        Extract meaningful content from browser navigation result.

        Args:
            nav_result: Result dictionary from browser_navigate.

        Returns:
            Formatted text content.
        """
        content_parts = []

        # Extract URL
        if nav_result.get('url'):
            content_parts.append(f'URL: {nav_result["url"]}')

        # Extract main text content
        text_content = self.browser.extract_text(nav_result)
        if text_content:
            max_text_length = 5000
            if len(text_content) > max_text_length:
                text_content = text_content[:max_text_length] + '... [truncated]'
            content_parts.append(f'Content:\n{text_content}')

        # Extract structured data if available
        table_data = self.browser.find_table(nav_result)
        if table_data:
            content_parts.append(
                f'Table Data: {json.dumps(table_data, indent=2)[:1000]}'
            )

        return '\n\n'.join(content_parts) if content_parts else '[No content extracted]'

    def _summarize_content_with_llm(
        self,
        content_parts: List[str],
        problem: str,
        subtask_description: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Summarize search result content using LLM while considering main problem and query analysis.
        Processes each content part individually for better focus and accuracy.

        Args:
            content_parts: List of content strings from processed search results.
            problem: Original problem description.
            subtask_description: Description of the subtask being executed.
            query_analysis: Optional query analysis results.

        Returns:
            Summarized content string.
        """
        if not content_parts:
            return ''

        # Build context from query analysis (once, reused for all parts)
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            implicit_reqs = query_analysis.get('implicit_requirements', [])
            constraints = query_analysis.get('constraints', {})
            answer_format = query_analysis.get('answer_format', '')
            dependencies = query_analysis.get('dependencies', [])

            if explicit_reqs:
                requirements_context += (
                    f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                )
            if implicit_reqs:
                requirements_context += (
                    f'\nImplicit Requirements: {", ".join(implicit_reqs)}'
                )
            if dependencies:
                requirements_context += (
                    f'\nInformation Dependencies: {", ".join(dependencies)}'
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

        system_prompt = """You are an expert at summarizing and structuring search result content.
Given a single search result from a web page or file, create a focused summary that extracts key information relevant to the problem and subtask.

Your task:
1. Extract and highlight information that directly addresses the problem requirements
2. Preserve important facts, numbers, dates, and specific details
3. Structure the summary logically and clearly
4. Focus on actionable information that helps solve the problem
5. Remove redundant or irrelevant information
6. Maintain context about where information came from (web pages vs files)

Return a well-structured summary that:
- Preserves critical details and facts
- Is organized and easy to understand
- Focuses on relevance to the problem and subtask
- Includes source indicators when relevant"""

        summarized_parts = []
        total_length = sum(len(part) for part in content_parts)

        self.logger.info(
            f'Summarizing {len(content_parts)} content parts individually '
            f'({total_length} chars total) using LLM'
        )

        # Process each content part individually
        for idx, content_part in enumerate(content_parts):
            try:
                # Skip LLM call for very short content parts
                if len(content_part) < 200:
                    self.logger.debug(
                        f'Part {idx + 1}/{len(content_parts)} is too short ({len(content_part)} chars), '
                        f'skipping LLM summarization'
                    )
                    summarized_parts.append(content_part)
                    continue

                # Limit individual content length to avoid token limits
                max_content_length = 8000
                content_to_summarize = content_part
                if len(content_part) > max_content_length:
                    content_to_summarize = (
                        content_part[:max_content_length]
                        + '\n\n[... Content truncated for summarization ...]'
                    )
                    self.logger.debug(
                        f'Part {idx + 1}/{len(content_parts)} truncated from {len(content_part)} '
                        f'to {max_content_length} chars'
                    )

                user_prompt = f"""Main Problem: {problem}

Current Subtask: {subtask_description}
{requirements_context}

Search Result Content (Part {idx + 1} of {len(content_parts)}):
{content_to_summarize}

Summarize this content, extracting key information relevant to solving the problem and completing the subtask. Focus on facts, numbers, dates, and specific details that are directly useful."""

                self.logger.debug(
                    f'Summarizing part {idx + 1}/{len(content_parts)} '
                    f'({len(content_part)} chars)'
                )

                summary = self.llm_service.call_with_system_prompt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,  # Lower temperature for consistent summarization
                    max_tokens=2000,  # Reasonable token limit per part
                )

                summarized_parts.append(summary)
                self.logger.debug(
                    f'Part {idx + 1}/{len(content_parts)} summarized: '
                    f'{len(summary)} chars (from {len(content_part)} chars)'
                )

            except Exception as e:
                self.logger.warning(
                    f'Failed to summarize content part {idx + 1}/{len(content_parts)}: {e}. '
                    f'Using original content for this part.'
                )
                # Fallback to original content if summarization fails for this part
                summarized_parts.append(content_part)

        # Combine all summarized parts
        final_summary = '\n\n---\n\n'.join(summarized_parts)
        final_length = len(final_summary)

        self.logger.info(
            f'Content summarization complete: '
            f'{final_length} chars (from {total_length} chars total)'
        )

        return final_summary
