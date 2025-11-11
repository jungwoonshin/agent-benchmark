"""Search Result Processor for handling search results with LLM-based relevance checking."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from ..llm import LLMService
from ..models import Attachment, SearchResult
from ..utils import extract_json_from_text
from .api_formatter import APIFormatter
from .browser import Browser
from .content_formatter import ContentFormatter
from .content_summarizer import ContentSummarizer
from .content_type_classifier import ContentTypeClassifier
from .content_type_detector import ContentTypeDetector
from .file_type_navigator import FileTypeNavigator
from .relevance_checker import RelevanceChecker
from .relevance_ranker import RelevanceRanker

if TYPE_CHECKING:
    from src.tools import ToolBelt


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
        tool_belt: 'ToolBelt',  # ToolBelt instance for file operations
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
        self.relevance_checker = RelevanceChecker(llm_service, tool_belt, logger)
        self.content_type_classifier = ContentTypeClassifier(llm_service, logger)
        self.content_type_detector = ContentTypeDetector(llm_service, logger)
        self.file_type_navigator = FileTypeNavigator(browser, logger, llm_service)
        # Initialize helper modules
        self.api_formatter = APIFormatter(tool_belt, logger)
        self.content_formatter = ContentFormatter(logger)
        self.content_summarizer = ContentSummarizer(llm_service, logger)
        self.relevance_ranker = RelevanceRanker(llm_service, logger)

    def _format_text_for_logging(self, text: str) -> str:
        """Format text for logging (delegates to ContentFormatter)."""
        return self.content_formatter.format_text_for_logging(text)

    def _rank_search_results_by_relevance(
        self,
        search_results: List[SearchResult],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Rank search results by relevance (delegates to RelevanceRanker)."""
        return self.relevance_ranker.rank_by_relevance(
            search_results, subtask_description, problem, query_analysis
        )

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
        Process search results by first selecting relevant ones using LLM, then processing only the selected results.

        New workflow:
        1. Use LLM to select which search results are relevant (based on title, snippet, URL only)
           - LLM filters out irrelevant results entirely
           - Returns only relevant results, sorted by relevance (most relevant first)
        2. Process only the selected relevant results (up to max_results_to_process)
        3. For each result: download/navigate to get full content
        4. Check relevance using LLM with full content
        5. If relevant: extract and structure content, then STOP (break loop)
        6. If not relevant: continue to next result

        Stops processing as soon as the first relevant result is found.

        Args:
            search_results: List of SearchResult objects to process.
            subtask_description: Description of the subtask being executed.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
            attachments: Optional list to append downloaded files to.
            max_results_to_process: Maximum number of results to process (default: 5).

        Returns:
            Dictionary with:
            - processed_count: Number of results processed before finding a match
            - relevant_count: Number of relevant results (0 or 1)
            - web_pages: List of processed web page results (max 1)
            - downloaded_files: List of downloaded file attachments (max 1)
            - content_summary: Content summary from the first relevant result
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

        # Log full list of search results in a pretty format
        search_results_summary = [
            f'{"=" * 80}',
            f'Received {len(search_results)} search result(s) for subtask: {subtask_description}',
            f'{"=" * 80}',
        ]
        for idx, result in enumerate(search_results, 1):
            snippet_preview = (
                result.snippet[:200] + '...'
                if len(result.snippet) > 200
                else result.snippet
            )
            search_results_summary.append(f'\n[{idx}]')
            search_results_summary.append(f'    Title: {result.title}')
            search_results_summary.append(f'    URL: {result.url}')
            search_results_summary.append(f'    Snippet: {snippet_preview}')
        search_results_summary.append(f'\n{"=" * 80}\n')

        self.logger.info('\n'.join(search_results_summary))

        # Step 1: Use LLM to select relevant search results (before processing)
        selected_results = self.relevance_ranker.rank_by_relevance(
            search_results, subtask_description, problem, query_analysis
        )

        # Step 2: Process only the selected relevant results (up to max_results_to_process)
        results_to_process = selected_results[:max_results_to_process]

        self.logger.info(
            f'Processing {len(results_to_process)} selected relevant search result(s) '
            f'(selected from {len(search_results)} total) for subtask: {subtask_description}'
        )

        processed_count = 0
        relevant_count = 0
        web_pages = []
        downloaded_files = []
        content_parts = []

        # Process selected relevant results one by one, stopping at the first relevant match
        for idx, result in enumerate(results_to_process):
            processed_count += 1

            self.logger.info(
                f'Processing selected relevant result [{idx + 1}/{len(results_to_process)}]: {result.title}'
            )

            processed_result = self._process_single_result(
                result,
                subtask_description,
                problem,
                query_analysis,
                attachments,
                idx,
                len(results_to_process),
            )

            if processed_result:
                # Found a relevant result - terminate immediately and move to next step
                relevant_count = 1
                self.logger.info(
                    f'✓ Found relevant result: {result.title}. Terminating search result processing and moving to next step.'
                )

                if processed_result['type'] == 'web_page':
                    web_pages.append(processed_result['data'])
                    content_text = f'[Web Page: {processed_result["title"]}]\nURL: {processed_result["url"]}\n{processed_result["content"]}'
                    # Add image analysis if available
                    if processed_result.get('image_analysis'):
                        content_text += f'\n\nIMAGE ANALYSIS (from visual LLM):\n{processed_result["image_analysis"]}'
                    content_parts.append(content_text)
                elif processed_result['type'] == 'pdf':
                    # Structured PDF with sections
                    downloaded_files.append(processed_result['data'])
                    # Build content summary with sections and include relevant pages information
                    sections_text = '\n\n'.join(
                        f'[Section: {s.get("title", "Untitled")} (Page {s.get("page", "?")})]\n{s.get("content", "")}'
                        for s in processed_result.get('sections', [])
                    )
                    content_text = f'[PDF: {processed_result["title"]}]\nURL: {processed_result["url"]}\n\nRelevant Pages: {", ".join(str(s.get("page", "?")) for s in processed_result.get("sections", []) if s.get("page"))}\n\n{sections_text}'
                    # Add image analysis if available
                    if processed_result.get('image_analysis'):
                        content_text += f'\n\nIMAGE ANALYSIS (from visual LLM):\n{processed_result["image_analysis"]}'
                    content_parts.append(content_text)
                elif processed_result['type'] == 'file':
                    downloaded_files.append(processed_result['data'])
                    content_parts.append(
                        f'[File: {processed_result["title"]}]\n{processed_result["content"]}'
                    )

                # Terminate immediately after finding first relevant result - move to next step
                # Skip remaining search results
                self.logger.info(
                    f'Terminating early: Found relevant result after processing {processed_count} selected result(s). '
                    f'Skipping remaining {len(results_to_process) - processed_count} selected result(s).'
                )
                break
            else:
                self.logger.info(
                    f'Result [{idx + 1}] not relevant, continuing to next result...'
                )

        # Step 5: Extract and structure content - summarize using LLM
        if content_parts:
            content_summary = self.content_summarizer.summarize_multiple_results(
                content_parts, problem, subtask_description, query_analysis
            )
        else:
            content_summary = ''

        result_summary = {
            'processed_count': processed_count,
            'relevant_count': relevant_count,
            'web_pages': web_pages,
            'downloaded_files': downloaded_files,
            'content_summary': content_summary,
        }

        if relevant_count > 0:
            self.logger.info(
                f'Search result processing TERMINATED: Found relevant result after processing {processed_count} result(s). '
                f'{len(web_pages)} web pages, {len(downloaded_files)} files downloaded. '
                f'Ready to move to next step.'
            )
        else:
            self.logger.info(
                f'Search result processing complete: No relevant results found after processing {processed_count} result(s).'
            )

        return result_summary

    def _requires_visual_analysis(
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
            # Extract step number from subtask_description (look for "step_1", "step_2", etc.)
            step_num = None
            step_match = re.search(
                r'step[_\s]*(\d+)', subtask_description, re.IGNORECASE
            )
            if step_match:
                try:
                    step_num = int(step_match.group(1))
                except (ValueError, IndexError):
                    pass

            # Build context from query analysis, filtering by step number if available
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                if explicit_reqs:
                    # Filter requirements by step number if step number is available
                    if step_num is not None:
                        # Filter to only requirements matching this step number
                        # Requirements are tagged as "Step N: requirement" or "step N: requirement"
                        filtered_reqs = []
                        for req in explicit_reqs:
                            req_str = str(req)
                            # Check if requirement matches this step number
                            req_step_match = re.search(
                                r'step[_\s]*(\d+)[:\s]', req_str, re.IGNORECASE
                            )
                            if req_step_match:
                                try:
                                    req_step_num = int(req_step_match.group(1))
                                    if req_step_num == step_num:
                                        filtered_reqs.append(req)
                                except (ValueError, IndexError):
                                    # If we can't parse step number, include it (fallback)
                                    filtered_reqs.append(req)
                            else:
                                # If requirement doesn't have step tag, don't include it
                                # (only include step-tagged requirements)
                                pass
                        if filtered_reqs:
                            requirements_context += (
                                f'\nExplicit Requirements: {", ".join(filtered_reqs)}'
                            )
                    else:
                        # If no step number found, include all requirements (backward compatibility)
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

    def _summarize_single_result_content(
        self,
        content: str,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
        content_type: str = 'general',
    ) -> str:
        """Summarize single result content (delegates to ContentSummarizer)."""
        return self.content_summarizer.summarize_single_result(
            content, subtask_description, problem, query_analysis, content_type
        )

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
        """
        Process a single search result:
        1. Download/navigate FIRST to get full content (regardless of file or web page)
        2. Determine relevance using subtask_description, problem, query_analysis, and full content
        3. If relevant, summarize the full content with subtask_description and problem context
        4. Check if subtask_description requires visual analysis
        5. If visual is required, use visual LLM to analyze data
        6. Finally return relevant summarized data with respect to subtask_description and problem
        """
        # Step 1: Normalize result to SearchResult object
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

        # Step 1: Determine what content type is needed for this subtask
        required_content_type = (
            self.content_type_classifier.classify_required_content_type(
                subtask_description, problem, query_analysis
            )
        )
        self.logger.info(f'Required content type for subtask: {required_content_type}')

        # Step 2: Try API first before downloading/navigating
        api_result = None
        try:
            api_result = self.tool_belt.try_api_for_search_result(
                result.url, problem, subtask_description
            )
            if api_result:
                self.logger.info(
                    f'Successfully retrieved data from {api_result["api_name"]} API for {result.url}'
                )
        except Exception as e:
            self.logger.debug(
                f'API check failed for {result.url}: {e}. Proceeding with normal download/navigation.'
            )

        # Step 3: Download/navigate FIRST to get full content (before checking relevance)
        # Check and adjust URL parameters if needed
        adjusted_url = self._check_and_adjust_url_parameters(
            result.url, subtask_description, problem, query_analysis
        )
        if adjusted_url != result.url:
            self.logger.info(f'URL parameters adjusted: {result.url} -> {adjusted_url}')
            result.url = adjusted_url

        # Initialize variables
        is_file = False
        file_type = None
        full_content = None
        extracted_data = None
        section_titles = None
        content_type = None

        # If API was successful, format API data and use it instead of downloading/navigating
        if api_result:
            api_data = api_result.get('data')
            api_name = api_result.get('api_name')

            # Format API data into content format
            # Pass problem/query_analysis so PDFs can return structured data with image_analysis
            formatted_content = self.api_formatter.format(
                api_data, api_name, result.url, problem, query_analysis
            )
            if formatted_content:
                full_content = formatted_content
                content_type = 'api_data'
                extracted_data = {
                    'type': 'api_data',
                    'api_name': api_name,
                    'url': result.url,
                    'content': formatted_content,
                    'raw_data': api_data,
                    'image_analysis': api_data.get(
                        '_image_analysis', ''
                    ),  # Include image analysis if available
                }
                section_titles = None
                # Skip download/navigation since we have API data
                is_file = False
                file_type = None
            else:
                # API data formatting failed, fall back to normal processing
                self.logger.warning(
                    f'Failed to format API data from {api_name}. Falling back to normal processing.'
                )
                api_result = None

        # If API was not used or failed, proceed with normal download/navigation
        if not api_result:
            # Download/navigate to get full content
            is_file, file_type = self._classify_result_type(result)

        # Process file download/navigation only if API was not used
        if not api_result and is_file:
            self.logger.info(
                f'Result {result_info} is a file (type: {file_type}). Downloading to extract content...'
            )
            try:
                attachment = self.tool_belt.download_file_from_url(result.url)
                if attachment:
                    attachments.append(attachment)

                    # For PDFs, check if visual analysis is needed, then extract images and do visual analysis BEFORE relevance check
                    image_analysis_for_relevance = ''
                    if file_type == 'pdf':
                        # First check if visual analysis is actually needed
                        requires_visual = self._requires_visual_analysis(
                            subtask_description, problem, query_analysis
                        )

                        if requires_visual:
                            self.logger.info(
                                'PDF detected and visual analysis is required. Extracting images and performing visual analysis before relevance check...'
                            )
                            try:
                                # Extract file content with image processing enabled to get visual analysis
                                file_content_with_images = self._extract_file_content(
                                    attachment,
                                    problem,
                                    query_analysis,
                                    skip_image_processing=False,
                                )

                                if (
                                    isinstance(file_content_with_images, dict)
                                    and file_content_with_images.get('type') == 'pdf'
                                ):
                                    # Get image analysis from the extracted data
                                    image_analysis_for_relevance = (
                                        file_content_with_images.get(
                                            'image_analysis', ''
                                        )
                                    )
                                    if image_analysis_for_relevance:
                                        self.logger.info(
                                            f'Visual analysis completed: {len(image_analysis_for_relevance)} characters extracted'
                                        )
                                    # Use the extracted data
                                    file_content = file_content_with_images
                                else:
                                    # Fallback: extract without images if structured extraction failed
                                    file_content = self._extract_file_content(
                                        attachment,
                                        problem,
                                        query_analysis,
                                        skip_image_processing=True,
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f'Failed to extract images for visual analysis: {e}. Proceeding without visual analysis.'
                                )
                                # Fallback to extraction without images
                                file_content = self._extract_file_content(
                                    attachment,
                                    problem,
                                    query_analysis,
                                    skip_image_processing=True,
                                )
                        else:
                            self.logger.info(
                                'PDF detected but visual analysis not required. Extracting content without image processing...'
                            )
                            # Visual analysis not needed, extract without images
                            file_content = self._extract_file_content(
                                attachment,
                                problem,
                                query_analysis,
                                skip_image_processing=True,
                            )
                    else:
                        # For non-PDF files, extract normally
                        file_content = self._extract_file_content(
                            attachment,
                            problem,
                            query_analysis,
                            skip_image_processing=True,
                        )

                    # Determine content_type: prefer dict with type='pdf', fallback to file_type
                    if (
                        isinstance(file_content, dict)
                        and file_content.get('type') == 'pdf'
                    ):
                        content_type = 'pdf'
                        full_content = file_content.get('full_text', '')
                        sections = file_content.get('sections', [])
                        section_titles = [
                            s.get('title', '') for s in sections if s.get('title')
                        ]
                        extracted_data = file_content
                        # Store image analysis in extracted_data for later use
                        if image_analysis_for_relevance:
                            extracted_data['image_analysis'] = (
                                image_analysis_for_relevance
                            )
                    else:
                        # If file_type indicates PDF but extraction returned string,
                        # still set content_type to 'pdf' for proper metadata extraction
                        if file_type == 'pdf':
                            content_type = 'pdf'
                            self.logger.debug(
                                'PDF detected by file_type but extraction returned string. '
                                'Setting content_type to "pdf" for metadata extraction.'
                            )
                        else:
                            content_type = 'file'

                        full_content = (
                            file_content
                            if isinstance(file_content, str)
                            else str(file_content)
                        )
                        extracted_data = file_content
            except Exception as e:
                self.logger.warning(
                    f'Failed to download/extract content from {result.url}: {e}'
                )
                return None
        elif not api_result:
            # Only process web pages if API was not used
            self.logger.info(
                f'Result {result_info} is a web page. Navigating to extract content...'
            )
            try:
                # Navigate to get raw HTML first for content type detection
                nav_result = self.browser.navigate(url=result.url, use_selenium=True)
                raw_html_content = (
                    nav_result.get('content', '') if nav_result.get('success') else ''
                )

                page_result = self._handle_web_page_result(
                    result, subtask_description, problem, query_analysis
                )
                if page_result:
                    content_type = 'web_page'
                    if isinstance(page_result, dict):
                        # Get full content and ensure it's formatted with html2text
                        raw_content = page_result.get('content', '')
                        # Format with html2text to get full formatted content
                        full_content = self.content_formatter.format_web_page_content(
                            raw_content
                        )
                        # Update the content in page_result with formatted version
                        page_result['content'] = full_content
                    else:
                        # String content - format with html2text
                        full_content = self.content_formatter.format_web_page_content(
                            page_result
                        )
                        # Convert to dict format for consistency
                        page_result = {
                            'content': full_content,
                            'image_analysis': '',
                        }
                    # Extract section titles from web page
                    section_titles = self._extract_web_page_sections(result.url)
                    extracted_data = page_result

                    # Step 3: Detect actual content type and navigate to PDF if needed
                    if required_content_type == 'pdf' and content_type == 'web_page':
                        self.logger.info(
                            'PDF required but web page retrieved. Attempting to find and navigate to PDF...'
                        )
                        # Detect what we actually got using raw HTML
                        detected_type = self.content_type_detector.detect_content_type(
                            url=result.url,
                            page_content=raw_html_content,
                            page_title=result.title,
                            is_file_download=False,
                        )

                        if detected_type != 'pdf':
                            # Try to find and navigate to PDF using raw HTML
                            # Use the general method with context for better detection
                            nav_result = (
                                self.file_type_navigator.find_and_navigate_to_file(
                                    current_url=result.url,
                                    page_content=raw_html_content,
                                    desired_file_type='pdf',
                                    page_title=result.title,
                                    subtask_description=subtask_description,
                                    problem=problem,
                                )
                            )

                            if nav_result and nav_result.get('success'):
                                # Check if the result indicates a direct file download
                                if nav_result.get('is_file_download'):
                                    # LLM found a direct file URL - download it
                                    pdf_url = nav_result.get(
                                        'file_url'
                                    ) or nav_result.get('url')
                                    self.logger.info(
                                        f'LLM found direct PDF file URL: {pdf_url}. Downloading...'
                                    )
                                    try:
                                        attachment = (
                                            self.tool_belt.download_file_from_url(
                                                pdf_url
                                            )
                                        )
                                        if attachment:
                                            # Ensure metadata includes source_url for metadata extraction
                                            if (
                                                not hasattr(attachment, 'metadata')
                                                or not attachment.metadata
                                            ):
                                                attachment.metadata = {}
                                            attachment.metadata['source_url'] = pdf_url
                                            attachments.append(attachment)
                                            # Mark as file so attachment is used for relevance checking
                                            is_file = True
                                            # Extract PDF content
                                            file_content = self._extract_file_content(
                                                attachment,
                                                problem,
                                                query_analysis,
                                                skip_image_processing=True,
                                            )
                                            if (
                                                isinstance(file_content, dict)
                                                and file_content.get('type') == 'pdf'
                                            ):
                                                content_type = 'pdf'
                                                full_content = file_content.get(
                                                    'full_text', ''
                                                )
                                                sections = file_content.get(
                                                    'sections', []
                                                )
                                                section_titles = [
                                                    s.get('title', '')
                                                    for s in sections
                                                    if s.get('title')
                                                ]
                                                extracted_data = file_content
                                            else:
                                                self.logger.warning(
                                                    'Downloaded PDF but extraction failed. Using web page content.'
                                                )
                                        else:
                                            self.logger.warning(
                                                'Failed to download PDF from target URL. Using web page content.'
                                            )
                                    except Exception as e:
                                        self.logger.warning(
                                            f'Error downloading PDF from target URL: {e}. Using web page content.'
                                        )
                                else:
                                    # Successfully navigated to PDF page
                                    self.logger.info(
                                        f'Successfully navigated to PDF from web page: {nav_result.get("url", "unknown")}'
                                    )
                                    # Try to download the PDF
                                    pdf_url = nav_result.get('url', result.url)
                                    try:
                                        attachment = (
                                            self.tool_belt.download_file_from_url(
                                                pdf_url
                                            )
                                        )
                                        if attachment:
                                            # Ensure metadata includes source_url for metadata extraction
                                            if (
                                                not hasattr(attachment, 'metadata')
                                                or not attachment.metadata
                                            ):
                                                attachment.metadata = {}
                                            attachment.metadata['source_url'] = pdf_url
                                            attachments.append(attachment)
                                            # Mark as file so attachment is used for relevance checking
                                            is_file = True
                                            # Extract PDF content
                                            file_content = self._extract_file_content(
                                                attachment,
                                                problem,
                                                query_analysis,
                                                skip_image_processing=True,
                                            )
                                            if (
                                                isinstance(file_content, dict)
                                                and file_content.get('type') == 'pdf'
                                            ):
                                                content_type = 'pdf'
                                                full_content = file_content.get(
                                                    'full_text', ''
                                                )
                                                sections = file_content.get(
                                                    'sections', []
                                                )
                                                section_titles = [
                                                    s.get('title', '')
                                                    for s in sections
                                                    if s.get('title')
                                                ]
                                                extracted_data = file_content
                                            else:
                                                self.logger.warning(
                                                    'Navigated to PDF but extraction failed. Using web page content.'
                                                )
                                        else:
                                            self.logger.warning(
                                                'Navigated to PDF but download failed. Using web page content.'
                                            )
                                    except Exception as e:
                                        self.logger.warning(
                                            f'Error downloading PDF after navigation: {e}. Using web page content.'
                                        )
                            else:
                                self.logger.info(
                                    'Could not find PDF download link on page. Using web page content.'
                                )
                        else:
                            self.logger.info(
                                'Page appears to be a PDF viewer. Content type already correct.'
                            )

            except Exception as e:
                self.logger.warning(
                    f'Failed to navigate/extract content from {result.url}: {e}'
                )
                return None

        if not extracted_data or not full_content:
            self.logger.warning(f'No content extracted from result {result_info}')
            return None

        # Step 3: Determine relevance using full content
        # Get the attachment if it was downloaded
        attachment_for_relevance = None
        if is_file and attachments:
            # Get the most recently added attachment (should be the one we just downloaded)
            attachment_for_relevance = attachments[-1] if attachments else None
            self.logger.debug(
                f'Attachment for relevance check: {attachment_for_relevance is not None}, '
                f'attachments list length: {len(attachments) if attachments else 0}'
            )
        else:
            self.logger.debug(
                f'No attachment for relevance check: is_file={is_file}, '
                f'has_attachments={attachments is not None and len(attachments) > 0 if attachments else False}'
            )

        # Get image analysis if it was extracted before relevance check (for PDFs and API data)
        image_analysis_for_relevance = ''
        if isinstance(extracted_data, dict):
            # Check both PDF files and API data (which may contain PDF content with images)
            if extracted_data.get('type') in ('pdf', 'api_data'):
                image_analysis_for_relevance = extracted_data.get('image_analysis', '')
                if image_analysis_for_relevance:
                    self.logger.info(
                        f'Including image analysis in relevance check ({len(image_analysis_for_relevance)} chars)'
                    )

        self.logger.debug(
            f'Relevance check: content_type={content_type}, '
            f'extracted_data_type={type(extracted_data).__name__}, '
            f'is_pdf_dict={isinstance(extracted_data, dict) and extracted_data.get("type") == "pdf" if isinstance(extracted_data, dict) else False}'
        )

        is_relevant, relevance_reasoning = self._check_relevance(
            result,
            subtask_description,
            problem,
            query_analysis,
            full_content=full_content,
            section_titles=section_titles,
            content_type=content_type,
            attachment=attachment_for_relevance,
            image_analysis=image_analysis_for_relevance,
        )

        if not is_relevant:
            self.logger.info(
                f'Result {result_info} deemed not relevant after full content analysis. Reason: {relevance_reasoning}'
            )
            return None

        self.logger.info(
            f'Result {result_info} is RELEVANT. Reason: {relevance_reasoning}. '
            f'Proceeding with content summarization and visual analysis check.'
        )

        # Step 4: Summarize the full content with subtask_description and problem context
        summarized_content = self.content_summarizer.summarize_single_result(
            full_content,
            subtask_description,
            problem,
            query_analysis,
            content_type=content_type or 'general',
        )

        # Update extracted_data with summarized content
        if isinstance(extracted_data, dict):
            extracted_data['summarized_content'] = summarized_content
            # Keep original content for reference but use summarized for return
            extracted_data['content'] = summarized_content
        else:
            # Convert to dict format
            extracted_data = {
                'content': summarized_content,
                'image_analysis': '',
            }

        # Step 5: Check if subtask_description requires visual analysis
        requires_visual = self._requires_visual_analysis(
            subtask_description, problem, query_analysis
        )

        # Step 6: If visual is required, use visual LLM to analyze data
        if requires_visual:
            self.logger.info(
                'Visual analysis required for subtask. Processing with visual LLM...'
            )

            if (
                hasattr(self.tool_belt, 'image_recognition')
                and self.tool_belt.image_recognition
            ):
                try:
                    if (
                        is_file
                        and isinstance(extracted_data, dict)
                        and extracted_data.get('type') == 'pdf'
                    ):
                        # Check if image analysis was already done before relevance check
                        existing_image_analysis = extracted_data.get(
                            'image_analysis', ''
                        )
                        if existing_image_analysis:
                            self.logger.info(
                                'Image analysis already completed before relevance check. Skipping duplicate processing.'
                            )
                            # Image analysis is already in extracted_data, no need to process again
                        else:
                            # Process PDF images (fallback if not done before relevance check)
                            extracted_images = extracted_data.get(
                                'extracted_images', []
                            )
                            if extracted_images:
                                attachment = next(
                                    (
                                        a
                                        for a in attachments
                                        if a.filename
                                        == extracted_data.get('filename', '')
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
                                        extracted_data['image_analysis'] = (
                                            image_analysis
                                        )
                                        self.logger.info(
                                            f'Processed {len(extracted_images)} image(s) from PDF with visual LLM'
                                        )
                    elif not is_file:
                        # Process web page screenshot
                        screenshot = self.browser.take_screenshot(as_base64=False)
                        if screenshot:
                            task_desc = f'Analyze this webpage screenshot and extract relevant information for: {subtask_description}'
                            image_analysis = self.tool_belt.image_recognition.recognize_images_from_browser(
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
                            if image_analysis:
                                if isinstance(extracted_data, dict):
                                    extracted_data['image_analysis'] = image_analysis
                                else:
                                    # Convert to dict format
                                    extracted_data = {
                                        'content': extracted_data,
                                        'image_analysis': image_analysis,
                                    }
                                self.logger.info(
                                    'Processed webpage screenshot with visual LLM'
                                )
                except Exception as e:
                    self.logger.warning(f'Failed to process visual analysis: {e}')
            else:
                self.logger.warning(
                    'Visual analysis required but image_recognition tool not available'
                )
        else:
            self.logger.debug('Visual analysis not required for this subtask')

        # Step 7: Return relevant summarized data with respect to subtask_description and problem
        if is_file and extracted_data:
            if isinstance(extracted_data, dict) and extracted_data.get('type') == 'pdf':
                return {
                    'type': 'pdf',
                    'title': result.title,
                    'url': result.url,
                    'sections': extracted_data.get('sections', []),
                    'image_analysis': extracted_data.get('image_analysis', ''),
                    'content': summarized_content,  # Return summarized content
                    'data': {
                        'url': result.url,
                        'type': file_type,
                        'title': result.title,
                        'sections': extracted_data.get('sections', []),
                        'image_analysis': extracted_data.get('image_analysis', ''),
                        'content': summarized_content,  # Return summarized content
                    },
                }
            else:
                return {
                    'type': 'file',
                    'title': result.title,
                    'url': result.url,
                    'content': summarized_content,  # Return summarized content
                    'data': {
                        'url': result.url,
                        'type': file_type,
                        'title': result.title,
                        'content': summarized_content,  # Return summarized content
                    },
                }
        elif not is_file and extracted_data:
            if isinstance(extracted_data, dict):
                # Content is already summarized
                page_content = extracted_data.get('content', summarized_content)
                image_analysis = extracted_data.get('image_analysis', '')
                return {
                    'type': 'web_page',
                    'title': result.title,
                    'url': result.url,
                    'content': page_content,
                    'image_analysis': image_analysis,
                    'data': {
                        'url': result.url,
                        'title': result.title,
                        'content': page_content,
                        'image_analysis': image_analysis,
                    },
                }
            else:
                # Content is already summarized
                return {
                    'type': 'web_page',
                    'title': result.title,
                    'url': result.url,
                    'content': summarized_content,
                    'data': {
                        'url': result.url,
                        'title': result.title,
                        'content': summarized_content,
                    },
                }

        return None

    def _extract_web_page_sections(self, url: str) -> List[str]:
        """
        Extract section titles (headings) from a web page.

        Args:
            url: URL of the web page.

        Returns:
            List of section titles (headings).
        """
        try:
            # Navigate to the page
            nav_result = self.browser.navigate(url=url, use_selenium=False)

            if not nav_result.get('success'):
                # Try with selenium if requests failed
                nav_result = self.browser.navigate(url=url, use_selenium=True)
                if not nav_result.get('success'):
                    return []

            # Extract headings from the page
            soup = nav_result.get('soup')
            if not soup:
                return []

            # Extract all headings (h1-h6)
            headings = []
            for level in range(1, 7):
                for heading in soup.find_all(f'h{level}'):
                    text = heading.get_text(strip=True)
                    if text:
                        headings.append(text)

            return headings[:20]  # Limit to first 20 headings to avoid token bloat

        except Exception as e:
            self.logger.debug(f'Failed to extract web page sections from {url}: {e}')
            return []

    def _extract_pdf_sections(self, url: str) -> List[str]:
        """
        Extract section titles from a PDF file.

        Args:
            url: URL of the PDF file.

        Returns:
            List of section titles.
        """
        try:
            # Download the PDF
            attachment = self.tool_belt.download_file_from_url(url)
            if not attachment:
                return []

            # Extract content with sections
            content = self.tool_belt.read_attachment(attachment)

            if isinstance(content, dict) and content.get('type') == 'pdf':
                sections = content.get('sections', [])
                # Extract just the titles
                section_titles = [
                    s.get('title', '') for s in sections if s.get('title')
                ]
                return section_titles[:20]  # Limit to first 20 sections
            else:
                # If not structured, try to extract headings using image_recognition
                if (
                    hasattr(self.tool_belt, 'image_recognition')
                    and self.tool_belt.image_recognition
                ):
                    try:
                        import fitz  # PyMuPDF

                        pdf_doc = fitz.open(stream=attachment.data, filetype='pdf')
                        sections = (
                            self.tool_belt.image_recognition._extract_pdf_structure(
                                pdf_doc, range(len(pdf_doc))
                            )
                        )
                        section_titles = [
                            s.get('title', '') for s in sections if s.get('title')
                        ]
                        pdf_doc.close()
                        return section_titles[:20]
                    except Exception as e:
                        self.logger.debug(f'Failed to extract PDF structure: {e}')
                        return []

            return []

        except Exception as e:
            self.logger.debug(f'Failed to extract PDF sections from {url}: {e}')
            return []

    def _check_relevance(
        self,
        search_result: SearchResult,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
        full_content: Optional[str] = None,
        section_titles: Optional[List[str]] = None,
        content_type: Optional[str] = None,
        attachment: Optional[Attachment] = None,
        image_analysis: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Use LLM to determine if a search result is relevant.
        Includes full content and section titles (if available) to help determine relevance.
        For PDFs, also extracts and includes arXiv metadata (submission date, paper ID) in the prompt.
        For PDFs with images, includes visual LLM analysis to help determine relevance.

        Args:
            search_result: SearchResult to evaluate.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
            full_content: Optional full content of the search result (web page or file).
            section_titles: Optional list of section titles/headings.
            content_type: Optional type of content ('web_page', 'pdf', 'file').
            attachment: Optional Attachment object (used for PDF metadata extraction).
            image_analysis: Optional visual LLM analysis of images from PDF (extracted before relevance check).

        Returns:
            Tuple of (is_relevant: bool, reasoning: str)
        """
        return self.relevance_checker.check_relevance(
            search_result=search_result,
            subtask_description=subtask_description,
            problem=problem,
            query_analysis=query_analysis,
            full_content=full_content,
            section_titles=section_titles,
            content_type=content_type,
            attachment=attachment,
            image_analysis=image_analysis,
        )

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
    ) -> Union[None, str, Dict[str, Any]]:
        """
        Handle a search result that is a file by downloading it.

        Args:
            search_result: SearchResult pointing to a file.
            attachments: List to append downloaded attachment to.
            subtask_description: Description of the subtask.
            problem: Optional problem description for relevance filtering.
            query_analysis: Optional query analysis for relevance filtering.

        Returns:
            String summary of file content or structured dict (for PDFs) if successful, None otherwise.
        """
        try:
            self.logger.info(f'Downloading file from URL: {search_result.url}')
            attachment = self.tool_belt.download_file_from_url(search_result.url)
            attachments.append(attachment)

            # Try to extract text content from the file with relevance filtering
            file_content = self._extract_file_content(
                attachment, problem, query_analysis
            )

            if isinstance(file_content, dict):
                # Structured PDF data
                content_length = len(file_content.get('full_text', ''))
                self.logger.info(
                    f'Successfully downloaded and processed PDF: {attachment.filename} '
                    f'(size: {len(attachment.data)} bytes, {len(file_content.get("sections", []))} sections)'
                )
            else:
                content_length = (
                    len(file_content) if isinstance(file_content, str) else 0
                )
                self.logger.info(
                    f'Successfully downloaded and processed file: {attachment.filename} '
                    f'(size: {len(attachment.data)} bytes, content length: {content_length} chars)'
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
        skip_image_processing: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Extract text content from an attachment with optional relevance filtering.

        Args:
            attachment: Attachment to extract content from.
            problem: Optional problem description for relevance filtering (PDFs only).
            query_analysis: Optional query analysis for relevance filtering (PDFs only).
            skip_image_processing: If True, skip image processing (for relevance check).

        Returns:
            Extracted text content (string) or structured data (dict) for PDFs with sections.
        """
        try:
            # Use tool_belt's read_attachment to extract content with relevance filtering
            content = self.tool_belt.read_attachment(
                attachment,
                options={},  # Pass empty options dict
                problem=problem,
                query_analysis=query_analysis,
                skip_image_processing=skip_image_processing,
            )

            # Check if content is an error message
            if isinstance(content, str):
                content_lower = content.lower()
                if (
                    content.startswith('Error:')
                    or content.startswith('Error')
                    or 'error:'
                    in content_lower[:50]  # Check first 50 chars for error indicators
                ):
                    self.logger.warning(
                        f'PDF extraction returned error for {attachment.filename}: {content}'
                    )
                    # Try to extract basic text even if structured extraction failed
                    try:
                        import fitz  # PyMuPDF

                        pdf_doc = fitz.open(stream=attachment.data, filetype='pdf')
                        # Extract text from all pages as fallback
                        text_parts = []
                        for page_num in range(len(pdf_doc)):
                            page = pdf_doc[page_num]
                            page_text = page.get_text().strip()
                            if page_text:
                                text_parts.append(f'[Page {page_num + 1}]\n{page_text}')
                        pdf_doc.close()

                        if text_parts:
                            fallback_text = '\n\n'.join(text_parts)
                            self.logger.info(
                                f'Successfully extracted fallback text from {attachment.filename} '
                                f'({len(fallback_text)} chars)'
                            )
                            return fallback_text
                        else:
                            self.logger.warning(
                                f'No text content found in PDF {attachment.filename}'
                            )
                            return '[No text content found in PDF]'
                    except ImportError:
                        self.logger.error(
                            f'PyMuPDF not available. Cannot extract text from {attachment.filename}'
                        )
                        return content  # Return the original error
                    except Exception as fallback_error:
                        self.logger.warning(
                            f'Fallback extraction also failed for {attachment.filename}: {fallback_error}'
                        )
                        return content  # Return the original error

            if not content:
                self.logger.warning(f'No content returned for {attachment.filename}')
                return f'[Content of {attachment.filename}]'

            # Check if content is structured (dict) from PDF processing
            if isinstance(content, dict) and content.get('type') == 'pdf':
                return content

            # Return string content (backward compatibility)
            return content if isinstance(content, str) else str(content)
        except Exception as e:
            self.logger.error(
                f'Failed to extract content from {attachment.filename}: {e}',
                exc_info=True,
            )
            # Try fallback extraction
            try:
                import fitz  # PyMuPDF

                pdf_doc = fitz.open(stream=attachment.data, filetype='pdf')
                text_parts = []
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    page_text = page.get_text().strip()
                    if page_text:
                        text_parts.append(f'[Page {page_num + 1}]\n{page_text}')
                pdf_doc.close()

                if text_parts:
                    fallback_text = '\n\n'.join(text_parts)
                    self.logger.info(
                        f'Successfully extracted fallback text from {attachment.filename} '
                        f'after exception ({len(fallback_text)} chars)'
                    )
                    return fallback_text
            except Exception as fallback_error:
                self.logger.error(
                    f'Fallback extraction failed for {attachment.filename}: {fallback_error}',
                    exc_info=True,
                )

            return (
                f'[File: {attachment.filename} - content extraction failed: {str(e)}]'
            )

    def _handle_web_page_result(
        self,
        search_result: SearchResult,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Handle a search result that is a web page by navigating with the unified browser.
        Includes image extraction/analysis.

        Returns:
            String content (backward compatibility) or dict with 'content' and 'image_analysis' keys.
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

            # Step 1.5: Detect and toggle expandable elements if needed
            if use_selenium and self.llm_service:
                nav_result = self._toggle_relevant_expandable_elements(
                    nav_result, subtask_description, problem, query_analysis
                )

            # Step 2: Extract and process images from the page if visual LLM is available
            image_analysis = ''
            if (
                self.llm_service
                and hasattr(self.tool_belt, 'image_recognition')
                and self.tool_belt.image_recognition
            ):
                try:
                    # Take screenshot of the page
                    screenshot = self.browser.take_screenshot(as_base64=False)
                    if screenshot:
                        self.logger.info(
                            f'[Visual LLM Analysis] Captured screenshot from {search_result.url} '
                            f'({len(screenshot)} bytes). Processing with visual LLM...'
                        )
                        # Process screenshot with visual LLM
                        task_desc = f'Analyze this webpage screenshot and extract relevant information for: {subtask_description}'
                        self.logger.debug(
                            f'[Visual LLM Analysis] Task description: {task_desc[:200]}{"..." if len(task_desc) > 200 else ""}'
                        )
                        image_analysis = self.tool_belt.image_recognition.recognize_images_from_browser(
                            screenshot,
                            context={
                                'url': search_result.url,
                                'title': search_result.title,
                                'text': self.browser.extract_text(nav_result)[:1000]
                                if nav_result.get('success')
                                else '',
                            },
                            task_description=task_desc,
                        )
                        if image_analysis:
                            self.logger.info(
                                f'[Visual LLM Analysis] Analysis completed for {search_result.url}: '
                                f'{len(image_analysis)} chars returned'
                            )
                        else:
                            self.logger.warning(
                                f'[Visual LLM Analysis] No analysis result returned for {search_result.url}'
                            )
                except Exception as e:
                    self.logger.error(
                        f'[Visual LLM Analysis] Failed to extract/process images from {search_result.url}: {e}',
                        exc_info=True,
                    )

            # Determine if we need specific extraction based on subtask description
            action = self._determine_extraction_action(subtask_description)

            if action == 'extract_count':
                llm_extract_result = self.browser.extract_with_llm(
                    nav_result, subtask_description, self.llm_service
                )
                content = json.dumps(llm_extract_result)
            elif action == 'find_table':
                table_data = self.browser.find_table(nav_result)
                content = json.dumps(table_data) if table_data else 'No table found.'
            else:  # default to markdown extraction using html2text
                content = self.browser.extract_markdown(nav_result)

            # Return dict with content and image_analysis if image_analysis is available, otherwise return string for backward compatibility
            if image_analysis:
                return {
                    'content': content,
                    'image_analysis': image_analysis,
                }
            else:
                return content

        except Exception as e:
            self.logger.warning(
                f'Failed to navigate or extract content from {search_result.url}: {e}'
            )
            return None

    def _check_and_adjust_url_parameters(
        self,
        url: str,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Check if URL parameters are appropriate for the task using LLM.
        If parameters need adjustment, return a modified URL.

        Args:
            url: URL to check.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Original URL or adjusted URL with corrected parameters.
        """
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query, keep_blank_values=True)

            # If no query parameters, return original URL
            if not query_params:
                return url

            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )

            # Format query parameters for LLM
            params_str = ', '.join(
                f'{k}={v[0] if v else ""}' for k, v in query_params.items()
            )

            system_prompt = """You are an expert at evaluating URL parameters for web page navigation.
Given a problem description, subtask description, and URL parameters, determine if the parameters are appropriate for the task.

Consider:
- Whether parameter values align with the task requirements
- Whether parameter values might filter out relevant content
- Whether parameter values need to be adjusted to get the most appropriate page for finding correct answers

Return a JSON object with:
- parameters_appropriate: boolean indicating if current parameters are appropriate
- reasoning: brief explanation (1-2 sentences) of why parameters are or aren't appropriate
- adjusted_parameters: object with parameter names as keys and corrected values as values (only include parameters that need adjustment, use null to remove a parameter)
- confidence: float from 0.0 to 1.0 indicating confidence in the assessment

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

URL: {url}
Current Parameters: {params_str}

Are these URL parameters appropriate for completing the subtask? If not, what should they be adjusted to?"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            parameters_appropriate = result_data.get('parameters_appropriate', True)
            adjusted_params = result_data.get('adjusted_parameters', {})

            if parameters_appropriate and not adjusted_params:
                self.logger.debug(
                    'URL parameters are appropriate, no adjustment needed'
                )
                return url

            # Apply adjustments
            new_params = query_params.copy()
            for param_name, param_value in adjusted_params.items():
                if param_value is None:
                    # Remove parameter
                    new_params.pop(param_name, None)
                else:
                    # Update parameter (convert to list format for parse_qs compatibility)
                    new_params[param_name] = [str(param_value)]

            # Reconstruct URL with adjusted parameters
            new_query = urlencode(new_params, doseq=True)
            adjusted_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment,
                )
            )

            reasoning = result_data.get('reasoning', 'No reasoning provided')
            self.logger.info(
                f'URL parameters adjusted. Reasoning: {reasoning}. '
                f'New URL: {adjusted_url}'
            )

            return adjusted_url

        except Exception as e:
            self.logger.warning(
                f'Failed to check/adjust URL parameters: {e}. Using original URL.'
            )
            return url

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

    def _toggle_relevant_expandable_elements(
        self,
        nav_result: Dict[str, Any],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect expandable elements on the page and use LLM to determine which ones
        should be toggled to reveal information relevant to the subtask.

        Args:
            nav_result: Navigation result dictionary.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Updated navigation result after toggling relevant elements.
        """
        try:
            # Detect expandable elements
            expandable_elements = self.browser.detect_expandable_elements(nav_result)

            if not expandable_elements:
                self.logger.debug('No expandable elements detected on page')
                return nav_result

            self.logger.info(
                f'Detected {len(expandable_elements)} expandable element(s) on page'
            )

            # Use LLM to determine which elements to toggle
            elements_to_toggle = self._determine_elements_to_toggle(
                expandable_elements,
                subtask_description,
                problem,
                query_analysis,
            )

            if not elements_to_toggle:
                self.logger.debug(
                    'LLM determined no expandable elements need to be toggled'
                )
                return nav_result

            self.logger.info(
                f'Toggling {len(elements_to_toggle)} expandable element(s) based on LLM analysis'
            )

            # Toggle each element
            current_result = nav_result
            for element_info in elements_to_toggle:
                toggle_result = self.browser.toggle_expandable_element(element_info)
                if toggle_result.get('success'):
                    current_result = toggle_result
                    self.logger.info(
                        f'Successfully toggled element: {element_info.get("text", "unknown")}'
                    )
                else:
                    error = toggle_result.get('error', 'Unknown error')
                    self.logger.warning(
                        f'Failed to toggle element "{element_info.get("text", "unknown")}": {error}'
                    )

            return current_result

        except Exception as e:
            self.logger.warning(
                f'Failed to toggle expandable elements: {e}. Continuing with original page content.'
            )
            return nav_result

    def _determine_elements_to_toggle(
        self,
        expandable_elements: List[Dict[str, Any]],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to determine which expandable elements should be toggled
        to reveal information relevant to the subtask.

        Args:
            expandable_elements: List of expandable element dictionaries.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            List of element dictionaries that should be toggled.
        """
        if not expandable_elements:
            return []

        # Build context from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            if explicit_reqs:
                requirements_context += (
                    f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                )

        # Format elements for LLM
        elements_text = []
        for idx, elem in enumerate(expandable_elements, 1):
            elem_info = f"""Element {idx}:
- Text: {elem.get('text', 'N/A')}
- Type: {elem.get('type', 'N/A')}
- ID: {elem.get('id', 'N/A')}
- Classes: {elem.get('classes', 'N/A')}
- Aria Label: {elem.get('aria-label', 'N/A')}
- Aria Expanded: {elem.get('aria-expanded', 'N/A')}"""
            elements_text.append(elem_info)

        system_prompt = """You are an expert at analyzing web pages to determine which expandable/collapsible elements (buttons, toggles) need to be clicked to reveal information relevant to a specific task.

Given a subtask description and a list of expandable elements on a web page, determine which elements should be toggled (clicked) to reveal hidden information that is needed to complete the subtask.

Consider:
- The subtask description and what information is being sought
- The text, labels, and attributes of each expandable element
- Whether clicking an element would reveal information relevant to the subtask
- Elements that are currently collapsed (aria-expanded="false") and need to be expanded
- Elements whose text/labels suggest they contain relevant information

Return a JSON object with:
- elements_to_toggle: array of element indices (1-based) that should be toggled, sorted by priority (most relevant first)
- reasoning: brief explanation of which elements were selected and why (1-2 sentences)

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

Expandable Elements on Page:
{chr(10).join(elements_text)}

Which expandable elements should be toggled to reveal information needed for this subtask? Consider elements that might contain hidden information relevant to the subtask description."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            element_indices = result_data.get('elements_to_toggle', [])
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'LLM selected {len(element_indices)} element(s) to toggle. Reasoning: {reasoning}'
            )

            # Convert 1-based indices to 0-based and get elements
            elements_to_toggle = []
            for idx in element_indices:
                try:
                    idx_int = int(idx) - 1  # Convert to 0-based
                    if 0 <= idx_int < len(expandable_elements):
                        elements_to_toggle.append(expandable_elements[idx_int])
                except (ValueError, TypeError):
                    continue

            return elements_to_toggle

        except Exception as e:
            self.logger.warning(
                f'Failed to determine elements to toggle using LLM: {e}. '
                f'Not toggling any elements.'
            )
            return []

    def _format_api_data(
        self,
        api_data: Any,
        api_name: str,
        url: str,
    ) -> Optional[str]:
        """Format API data (delegates to APIFormatter)."""
        return self.api_formatter.format(api_data, api_name, url)

    def _format_web_page_content(self, content: str) -> str:
        """Format web page content (delegates to ContentFormatter)."""
        return self.content_formatter.format_web_page_content(content)

    def _extract_navigation_content(self, nav_result: Dict[str, Any]) -> str:
        """Extract meaningful content from browser navigation result (delegates to ContentFormatter)."""
        return self.content_formatter.extract_navigation_content(
            nav_result, self.browser
        )

    def _summarize_content_with_llm(
        self,
        content_parts: List[str],
        problem: str,
        subtask_description: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Summarize content with LLM (delegates to ContentSummarizer)."""
        return self.content_summarizer.summarize_multiple_results(
            content_parts, problem, subtask_description, query_analysis
        )
