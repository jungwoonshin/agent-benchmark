"""Content Type Detector for identifying what type of content was retrieved."""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..llm import LLMService


class ContentTypeDetector:
    """Detects what type of content was actually retrieved from a URL."""

    def __init__(self, llm_service: 'LLMService', logger: logging.Logger):
        """
        Initialize ContentTypeDetector.

        Args:
            llm_service: LLM service for detection.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger

    def detect_content_type(
        self,
        url: str,
        page_content: Optional[str] = None,
        page_title: Optional[str] = None,
        is_file_download: bool = False,
        file_extension: Optional[str] = None,
    ) -> str:
        """
        Detect what type of content was retrieved.

        Args:
            url: URL that was accessed.
            page_content: Optional HTML/text content of the page.
            page_title: Optional title of the page.
            is_file_download: Whether a file was downloaded.
            file_extension: Optional file extension if a file was downloaded.

        Returns:
            'pdf' if PDF was retrieved, 'web_page' if web page was retrieved, or 'unknown'.
        """
        # Quick check: if file was downloaded with PDF extension
        if is_file_download and file_extension and file_extension.lower() == 'pdf':
            self.logger.info(f'Detected PDF file from URL: {url}')
            return 'pdf'

        # Quick check: if URL ends with .pdf
        if url.lower().endswith('.pdf'):
            self.logger.info(f'Detected PDF from URL pattern: {url}')
            return 'pdf'

        # If we have page content, use LLM to analyze
        if page_content:
            return self._detect_from_content(url, page_content, page_title)

        # Default: assume web page if no file was downloaded
        if not is_file_download:
            self.logger.info(f'Assuming web page for URL: {url}')
            return 'web_page'

        return 'unknown'

    def _detect_from_content(
        self, url: str, page_content: str, page_title: Optional[str] = None
    ) -> str:
        """
        Use LLM to detect content type from page content.

        Args:
            url: URL that was accessed.
            page_content: HTML/text content of the page.
            page_title: Optional title of the page.

        Returns:
            'pdf' if PDF viewer/download page, 'web_page' if regular web page, or 'unknown'.
        """
        self.logger.info(f'Analyzing content type from page content for URL: {url}')

        # Truncate content for LLM analysis
        content_preview = (
            page_content[:2000] if len(page_content) > 2000 else page_content
        )

        system_prompt = """You are an expert at analyzing web page content to determine if it's a PDF viewer/download page or a regular web page.

Given a URL and page content, determine:
- "pdf" if the page is a PDF viewer, PDF download page, or directly displays PDF content
- "web_page" if it's a regular HTML web page with text, links, forms, etc.
- "unknown" if you cannot determine

Look for indicators:
- PDF viewer: embedded PDF viewers, PDF.js, PDF download buttons
- PDF download page: download links, "Download PDF" buttons, PDF file links
- Regular web page: HTML content, navigation, forms, interactive elements

Return JSON only with:
{"content_type": "pdf" | "web_page" | "unknown", "reasoning": "brief explanation"}"""

        title_info = f'\nPage Title: {page_title}' if page_title else ''
        user_prompt = f"""URL: {url}
{title_info}

Page Content (preview):
{content_preview}

What type of content is this?"""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            import json

            from ..utils import extract_json_from_text

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            content_type = result_data.get('content_type', 'unknown')
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'Content type detection: {content_type}. Reasoning: {reasoning}'
            )

            return content_type

        except Exception as e:
            self.logger.warning(
                f'Failed to detect content type using LLM: {e}. Defaulting to "web_page".'
            )
            return 'web_page'
