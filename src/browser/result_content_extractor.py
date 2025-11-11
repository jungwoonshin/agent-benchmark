"""Content extraction module for search results.

Handles extraction of content from API results, files, and web pages.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..models import Attachment, SearchResult

if TYPE_CHECKING:
    from src.tools import ToolBelt


class ResultContentExtractor:
    """Extracts content from different result types (API, files, web pages)."""

    def __init__(
        self,
        tool_belt: 'ToolBelt',
        browser: Any,  # Browser instance
        api_formatter: Any,  # APIFormatter instance
        content_formatter: Any,  # ContentFormatter instance
        content_type_classifier: Any,  # ContentTypeClassifier instance
        content_type_detector: Any,  # ContentTypeDetector instance
        file_type_navigator: Any,  # FileTypeNavigator instance
        logger: logging.Logger,
    ):
        """Initialize ResultContentExtractor."""
        self.tool_belt = tool_belt
        self.browser = browser
        self.api_formatter = api_formatter
        self.content_formatter = content_formatter
        self.content_type_classifier = content_type_classifier
        self.content_type_detector = content_type_detector
        self.file_type_navigator = file_type_navigator
        self.logger = logger

    def try_api_extraction(
        self,
        url: str,
        problem: str,
        subtask_description: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to extract content using API first.

        Args:
            url: URL to try API extraction for.
            problem: Problem description.
            subtask_description: Subtask description.

        Returns:
            API result dict if successful, None otherwise.
        """
        try:
            api_result = self.tool_belt.try_api_for_search_result(
                url, problem, subtask_description
            )
            if api_result:
                self.logger.info(
                    f'Successfully retrieved data from {api_result["api_name"]} API for {url}'
                )
                return api_result
        except Exception as e:
            self.logger.debug(
                f'API check failed for {url}: {e}. Proceeding with normal download/navigation.'
            )
        return None

    def format_api_result(
        self,
        api_result: Dict[str, Any],
        url: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Format API result into content format.

        Args:
            api_result: API result dictionary.
            url: URL of the result.
            problem: Problem description.
            query_analysis: Optional query analysis.

        Returns:
            Formatted content dict if successful, None otherwise.
        """
        api_data = api_result.get('data')
        api_name = api_result.get('api_name')

        formatted_content = self.api_formatter.format(
            api_data, api_name, url, problem, query_analysis
        )
        if formatted_content:
            return {
                'type': 'api_data',
                'api_name': api_name,
                'url': url,
                'content': formatted_content,
                'raw_data': api_data,
                'image_analysis': api_data.get('_image_analysis', ''),
                'full_content': formatted_content,
                'extracted_data': {
                    'type': 'api_data',
                    'api_name': api_name,
                    'url': url,
                    'content': formatted_content,
                    'raw_data': api_data,
                    'image_analysis': api_data.get('_image_analysis', ''),
                },
                'section_titles': None,
                'is_file': False,
                'file_type': None,
                'content_type': 'api_data',
            }
        return None

    def extract_file_content(
        self,
        result: SearchResult,
        result_info: str,
        attachments: List[Attachment],
        problem: str,
        query_analysis: Optional[Dict[str, Any]],
        subtask_description: str,
        requires_visual: bool,
        extract_file_content_func: callable,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract content from a file result.

        Args:
            result: SearchResult object.
            result_info: Formatted result info string.
            attachments: List to append downloaded attachment to.
            problem: Problem description.
            query_analysis: Optional query analysis.
            subtask_description: Subtask description.
            requires_visual: Whether visual analysis is required.
            extract_file_content_func: Function to extract file content.

        Returns:
            Content extraction result dict if successful, None otherwise.
        """
        self.logger.info(
            f'Result {result_info} is a file. Downloading to extract content...'
        )
        try:
            attachment = self.tool_belt.download_file_from_url(result.url)
            if not attachment:
                return None

            attachments.append(attachment)

            # Determine file type
            is_file, file_type = self._classify_result_type(result)

            # For PDFs, handle visual analysis if needed
            image_analysis_for_relevance = ''
            if file_type == 'pdf':
                if requires_visual:
                    self.logger.info(
                        'PDF detected and visual analysis is required. Extracting images and performing visual analysis before relevance check...'
                    )
                    try:
                        file_content_with_images = extract_file_content_func(
                            attachment,
                            problem,
                            query_analysis,
                            skip_image_processing=False,
                        )

                        if (
                            isinstance(file_content_with_images, dict)
                            and file_content_with_images.get('type') == 'pdf'
                        ):
                            image_analysis_for_relevance = file_content_with_images.get(
                                'image_analysis', ''
                            )
                            if image_analysis_for_relevance:
                                self.logger.info(
                                    f'Visual analysis completed: {len(image_analysis_for_relevance)} characters extracted'
                                )
                            file_content = file_content_with_images
                        else:
                            file_content = extract_file_content_func(
                                attachment,
                                problem,
                                query_analysis,
                                skip_image_processing=True,
                            )
                    except Exception as e:
                        self.logger.warning(
                            f'Failed to extract images for visual analysis: {e}. Proceeding without visual analysis.'
                        )
                        file_content = extract_file_content_func(
                            attachment,
                            problem,
                            query_analysis,
                            skip_image_processing=True,
                        )
                else:
                    self.logger.info(
                        'PDF detected but visual analysis not required. Extracting content without image processing...'
                    )
                    file_content = extract_file_content_func(
                        attachment,
                        problem,
                        query_analysis,
                        skip_image_processing=True,
                    )
            else:
                file_content = extract_file_content_func(
                    attachment,
                    problem,
                    query_analysis,
                    skip_image_processing=True,
                )

            # Process extracted content
            if isinstance(file_content, dict) and file_content.get('type') == 'pdf':
                content_type = 'pdf'
                full_content = file_content.get('full_text', '')
                sections = file_content.get('sections', [])
                section_titles = [
                    s.get('title', '') for s in sections if s.get('title')
                ]
                extracted_data = file_content
                if image_analysis_for_relevance:
                    extracted_data['image_analysis'] = image_analysis_for_relevance
            else:
                if file_type == 'pdf':
                    content_type = 'pdf'
                    self.logger.debug(
                        'PDF detected by file_type but extraction returned string. '
                        'Setting content_type to "pdf" for metadata extraction.'
                    )
                else:
                    content_type = 'file'

                full_content = (
                    file_content if isinstance(file_content, str) else str(file_content)
                )
                extracted_data = file_content
                section_titles = None

            return {
                'is_file': True,
                'file_type': file_type,
                'full_content': full_content,
                'extracted_data': extracted_data,
                'section_titles': section_titles,
                'content_type': content_type,
            }

        except Exception as e:
            self.logger.warning(
                f'Failed to download/extract content from {result.url}: {e}'
            )
            return None

    def extract_web_page_content(
        self,
        result: SearchResult,
        result_info: str,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]],
        required_content_type: str,
        handle_web_page_result_func: callable,
        extract_web_page_sections_func: callable,
        extract_file_content_func: callable,
        attachments: List[Attachment],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract content from a web page result.

        Args:
            result: SearchResult object.
            result_info: Formatted result info string.
            subtask_description: Subtask description.
            problem: Problem description.
            query_analysis: Optional query analysis.
            required_content_type: Required content type for the subtask.
            handle_web_page_result_func: Function to handle web page results.
            extract_web_page_sections_func: Function to extract web page sections.
            extract_file_content_func: Function to extract file content.
            attachments: List to append downloaded attachments to.

        Returns:
            Content extraction result dict if successful, None otherwise.
        """
        self.logger.info(
            f'Result {result_info} is a web page. Navigating to extract content...'
        )
        try:
            # Navigate to get raw HTML first for content type detection
            nav_result = self.browser.navigate(url=result.url, use_selenium=True)
            raw_html_content = (
                nav_result.get('content', '') if nav_result.get('success') else ''
            )

            page_result = handle_web_page_result_func(
                result, subtask_description, problem, query_analysis
            )
            if not page_result:
                return None

            content_type = 'web_page'
            if isinstance(page_result, dict):
                raw_content = page_result.get('content', '')
                full_content = self.content_formatter.format_web_page_content(
                    raw_content
                )
                page_result['content'] = full_content
            else:
                full_content = self.content_formatter.format_web_page_content(
                    page_result
                )
                page_result = {
                    'content': full_content,
                    'image_analysis': '',
                }

            section_titles = extract_web_page_sections_func(result.url)
            extracted_data = page_result

            # Check if PDF navigation is needed
            if required_content_type == 'pdf' and content_type == 'web_page':
                pdf_result = self._navigate_to_pdf_from_web_page(
                    result,
                    raw_html_content,
                    subtask_description,
                    problem,
                    extract_file_content_func,
                    attachments,
                )
                if pdf_result:
                    return pdf_result

            return {
                'is_file': False,
                'file_type': None,
                'full_content': full_content,
                'extracted_data': extracted_data,
                'section_titles': section_titles,
                'content_type': content_type,
            }

        except Exception as e:
            self.logger.warning(
                f'Failed to navigate/extract content from {result.url}: {e}'
            )
            return None

    def _navigate_to_pdf_from_web_page(
        self,
        result: SearchResult,
        raw_html_content: str,
        subtask_description: str,
        problem: str,
        extract_file_content_func: callable,
        attachments: List[Attachment],
    ) -> Optional[Dict[str, Any]]:
        """
        Navigate to PDF from a web page if PDF is required.

        Args:
            result: SearchResult object.
            raw_html_content: Raw HTML content of the page.
            subtask_description: Subtask description.
            problem: Problem description.
            extract_file_content_func: Function to extract file content.
            attachments: List to append downloaded attachments to.

        Returns:
            PDF extraction result dict if successful, None otherwise.
        """
        self.logger.info(
            'PDF required but web page retrieved. Attempting to find and navigate to PDF...'
        )

        detected_type = self.content_type_detector.detect_content_type(
            url=result.url,
            page_content=raw_html_content,
            page_title=result.title,
            is_file_download=False,
        )

        if detected_type == 'pdf':
            self.logger.info(
                'Page appears to be a PDF viewer. Content type already correct.'
            )
            return None

        nav_result = self.file_type_navigator.find_and_navigate_to_file(
            current_url=result.url,
            page_content=raw_html_content,
            desired_file_type='pdf',
            page_title=result.title,
            subtask_description=subtask_description,
            problem=problem,
        )

        if not nav_result or not nav_result.get('success'):
            self.logger.info(
                'Could not find PDF download link on page. Using web page content.'
            )
            return None

        # Handle direct file download
        if nav_result.get('is_file_download'):
            pdf_url = nav_result.get('file_url') or nav_result.get('url')
            return self._download_pdf_from_url(
                pdf_url,
                result,
                extract_file_content_func,
                attachments,
            )

        # Handle navigation to PDF page
        pdf_url = nav_result.get('url', result.url)
        self.logger.info(f'Successfully navigated to PDF from web page: {pdf_url}')
        return self._download_pdf_from_url(
            pdf_url,
            result,
            extract_file_content_func,
            attachments,
        )

    def _download_pdf_from_url(
        self,
        pdf_url: str,
        result: SearchResult,
        extract_file_content_func: callable,
        attachments: List[Attachment],
    ) -> Optional[Dict[str, Any]]:
        """
        Download PDF from URL and extract content.

        Args:
            pdf_url: URL of the PDF.
            result: SearchResult object.
            extract_file_content_func: Function to extract file content.
            attachments: List to append downloaded attachments to.

        Returns:
            PDF extraction result dict if successful, None otherwise.
        """
        try:
            attachment = self.tool_belt.download_file_from_url(pdf_url)
            if not attachment:
                self.logger.warning(
                    'Failed to download PDF from target URL. Using web page content.'
                )
                return None

            # Ensure metadata includes source_url
            if not hasattr(attachment, 'metadata') or not attachment.metadata:
                attachment.metadata = {}
            attachment.metadata['source_url'] = pdf_url
            attachments.append(attachment)

            # Extract PDF content
            file_content = extract_file_content_func(
                attachment,
                None,  # problem
                None,  # query_analysis
                skip_image_processing=True,
            )

            if isinstance(file_content, dict) and file_content.get('type') == 'pdf':
                sections = file_content.get('sections', [])
                section_titles = [
                    s.get('title', '') for s in sections if s.get('title')
                ]
                return {
                    'is_file': True,
                    'file_type': 'pdf',
                    'full_content': file_content.get('full_text', ''),
                    'extracted_data': file_content,
                    'section_titles': section_titles,
                    'content_type': 'pdf',
                }
            else:
                self.logger.warning(
                    'Downloaded PDF but extraction failed. Using web page content.'
                )
                return None

        except Exception as e:
            self.logger.warning(
                f'Error downloading PDF from target URL: {e}. Using web page content.'
            )
            return None

    def _classify_result_type(self, search_result: SearchResult) -> Tuple[bool, str]:
        """
        Classify a search result as either a file or web page.

        Args:
            search_result: SearchResult to classify.

        Returns:
            Tuple of (is_file: bool, file_type: str)
        """
        url = search_result.url.lower() if search_result.url else ''
        title = search_result.title.lower() if search_result.title else ''

        file_type_patterns = {
            'pdf': ['.pdf'],
            'doc': ['.doc', '.docx', '.odt'],
            'spreadsheet': ['.xls', '.xlsx', '.csv', '.ods'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
            'archive': ['.zip', '.tar', '.gz', '.rar', '.7z'],
            'text': ['.txt'],
        }

        for file_type, patterns in file_type_patterns.items():
            for pattern in patterns:
                if pattern in url or pattern in title:
                    return True, file_type

        file_indicators = ['/pdf/', '/download/', '/file/', '/attachment/', '/doc/']
        if any(indicator in url for indicator in file_indicators):
            for file_type, patterns in file_type_patterns.items():
                if any(pattern.strip('.') in url for pattern in patterns):
                    return True, file_type
            return True, 'unknown'

        return False, 'webpage'
