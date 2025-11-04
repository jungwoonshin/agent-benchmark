"""Search Result Processor for handling search results with LLM-based relevance checking."""

import json
import logging
from typing import Any, Dict, List, Optional

from .json_utils import extract_json_from_text
from .llm_service import LLMService
from .models import Attachment, SearchResult


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
        tool_belt: Any,  # ToolBelt instance
        logger: logging.Logger,
    ):
        """
        Initialize SearchResultProcessor.

        Args:
            llm_service: LLM service for relevance checking.
            tool_belt: ToolBelt instance for browser navigation and downloads.
            logger: Logger instance.
        """
        self.llm_service = llm_service
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
            max_results_to_process: Maximum number of results to process (default: 5).

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

        # Process up to max_results_to_process
        for idx, result in enumerate(search_results[:max_results_to_process]):
            if not isinstance(result, SearchResult):
                continue

            processed_count += 1
            result_info = f'[{idx + 1}/{len(search_results)}] {result.title}'

            self.logger.info(f'Processing search result {result_info}')

            # Step 1: Check relevance using LLM
            is_relevant, relevance_reasoning = self._check_relevance(
                result, subtask_description, problem, query_analysis
            )

            if not is_relevant:
                self.logger.info(
                    f'Result {result_info} deemed not relevant. Reason: {relevance_reasoning}'
                )
                continue

            relevant_count += 1
            self.logger.info(
                f'Result {result_info} is RELEVANT. Reason: {relevance_reasoning}'
            )

            # Step 2: Classify as web page or file
            is_file, file_type = self._classify_result_type(result)

            # Step 3 & 4: Handle based on type
            if is_file:
                # It's a file - download it
                self.logger.info(
                    f'Result {result_info} is a file (type: {file_type}). Downloading...'
                )
                file_content = self._handle_file_result(
                    result, attachments, subtask_description
                )
                if file_content:
                    downloaded_files.append(
                        {'url': result.url, 'type': file_type, 'content': file_content}
                    )
                    content_parts.append(f'[File: {result.title}]\n{file_content}')
            else:
                # It's a web page - navigate and extract
                self.logger.info(
                    f'Result {result_info} is a web page. Navigating with Selenium...'
                )
                page_content = self._handle_web_page_result(
                    result, subtask_description, problem
                )
                if page_content:
                    web_pages.append(
                        {
                            'url': result.url,
                            'title': result.title,
                            'content': page_content,
                        }
                    )
                    content_parts.append(
                        f'[Web Page: {result.title}]\nURL: {result.url}\n{page_content}'
                    )

        # Step 5: Create content summary
        content_summary = '\n\n---\n\n'.join(content_parts) if content_parts else ''

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

Return a JSON object with:
- relevant: boolean indicating if the result is relevant
- reasoning: brief explanation (1-2 sentences) of why it is or isn't relevant
- confidence: float from 0.0 to 1.0 indicating confidence in the relevance assessment"""

            user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

Search Result:
- Title: {search_result.title}
- Snippet: {search_result.snippet}
- URL: {search_result.url}

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
    ) -> Optional[str]:
        """
        Handle a search result that is a file by downloading it.

        Args:
            search_result: SearchResult pointing to a file.
            attachments: List to append downloaded attachment to.
            subtask_description: Description of the subtask.

        Returns:
            String summary of file content if successful, None otherwise.
        """
        try:
            self.logger.info(f'Downloading file from URL: {search_result.url}')
            attachment = self.tool_belt.download_file_from_url(search_result.url)
            attachments.append(attachment)

            # Try to extract text content from the file
            file_content = self._extract_file_content(attachment)

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

    def _extract_file_content(self, attachment: Attachment) -> str:
        """
        Extract text content from an attachment.

        Args:
            attachment: Attachment to extract content from.

        Returns:
            Extracted text content.
        """
        try:
            # Use tool_belt's read_attachment to extract content
            content = self.tool_belt.read_attachment(attachment)
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
        Handle a search result that is a web page by navigating with Selenium.

        Args:
            search_result: SearchResult pointing to a web page.
            subtask_description: Description of the subtask.
            problem: Original problem description.

        Returns:
            Extracted text content from the page if successful, None otherwise.
        """
        try:
            self.logger.info(
                f'Navigating to web page: {search_result.url} using Selenium'
            )

            # Determine if we need specific extraction based on subtask description
            action = self._determine_extraction_action(subtask_description)

            # Navigate to the page
            nav_result = self.tool_belt.browser_navigate(
                url=search_result.url,
                action=action,
                extraction_query=subtask_description
                if action in ['extract_count', 'extract_statistics']
                else None,
            )

            if not nav_result.get('success'):
                error_msg = nav_result.get('error', 'Navigation failed')
                self.logger.warning(
                    f'Failed to navigate to {search_result.url}: {error_msg}'
                )
                return None

            # Extract content based on what was returned
            page_content = self._extract_navigation_content(nav_result)

            self.logger.info(
                f'Successfully extracted content from {search_result.url} '
                f'(length: {len(page_content)} chars)'
            )

            return page_content

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
        elif any(keyword in desc_lower for keyword in ['search for', 'find text']):
            return 'search_text'
        else:
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
        if nav_result.get('text'):
            text = nav_result['text']
            # Limit text length to avoid overwhelming downstream processing
            max_text_length = 5000
            if len(text) > max_text_length:
                text = text[:max_text_length] + '... [truncated]'
            content_parts.append(f'Content:\n{text}')

        # Extract structured data if available
        if nav_result.get('table_data'):
            table_data = nav_result['table_data']
            content_parts.append(
                f'Table Data: {json.dumps(table_data, indent=2)[:1000]}'
            )

        # Extract numeric data if available
        if nav_result.get('extracted_counts'):
            counts = nav_result['extracted_counts']
            content_parts.append(f'Extracted Counts: {json.dumps(counts, indent=2)}')

        # Extract search results if available
        if nav_result.get('search_results'):
            search_results = nav_result['search_results']
            content_parts.append(
                f'Search Results: {json.dumps(search_results, indent=2)[:1000]}'
            )

        # Extract LLM extraction if available
        if nav_result.get('llm_extraction'):
            llm_extraction = nav_result['llm_extraction']
            content_parts.append(
                f'LLM Extracted Value: {llm_extraction.get("extracted_value")}\n'
                f'Context: {llm_extraction.get("context", "")[:500]}\n'
                f'Reasoning: {llm_extraction.get("reasoning", "")}'
            )

        return '\n\n'.join(content_parts) if content_parts else '[No content extracted]'
