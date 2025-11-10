"""Content formatter for web pages and navigation results."""

import json
import logging
import re
from typing import Any, Dict, Optional

import html2text


class ContentFormatter:
    """Formats web page content and navigation results."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize content formatter.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)

    def format_web_page_content(self, content: str) -> str:
        """
        Format web page content using html2text for better readability.

        Args:
            content: Content string (may be HTML, markdown, or plain text).

        Returns:
            Formatted content string in markdown format.
        """
        if not content:
            return content

        # If content looks like HTML (contains HTML tags), convert it
        if re.search(r'<[a-z][\s\S]*>', content, re.IGNORECASE):
            try:
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = False
                h.body_width = 0
                h.unicode_snob = True
                h.skip_internal_links = False
                h.inline_links = True
                h.escape_snob = True

                formatted = h.handle(content)
                return formatted.strip()
            except Exception as e:
                self.logger.warning(
                    f'Failed to format content with html2text: {e}. Using original content.'
                )
                return content

        return content

    def format_text_for_logging(self, text: str) -> str:
        """
        Format text for logging by showing first 2 sentences and total length.

        Args:
            text: Text content to format.

        Returns:
            Formatted string with first 2 sentences and length info.
        """
        if not text:
            return '[Empty text]'

        sentences = re.split(r'([.!?]+)', text)
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                if sentence.strip():
                    reconstructed.append(sentence.strip())

        if len(reconstructed) >= 2:
            preview = ' '.join(reconstructed[:2])
        elif len(reconstructed) == 1:
            preview = reconstructed[0]
        else:
            preview = text[:200].strip()

        total_length = len(text)
        return f'{preview}... [Length: {total_length} chars]'

    def extract_navigation_content(
        self, nav_result: Dict[str, Any], browser: Any
    ) -> str:
        """
        Extract meaningful content from browser navigation result.

        Args:
            nav_result: Result dictionary from browser_navigate.
            browser: Browser instance for extracting text and tables.

        Returns:
            Formatted text content.
        """
        content_parts = []

        if nav_result.get('url'):
            content_parts.append(f'URL: {nav_result["url"]}')

        text_content = browser.extract_text(nav_result)
        if text_content:
            content_parts.append(
                f'Content: {self.format_text_for_logging(text_content)}'
            )

        table_data = browser.find_table(nav_result)
        if table_data:
            content_parts.append(
                f'Table Data: {json.dumps(table_data, indent=2)[:1000]}'
            )

        return '\n\n'.join(content_parts) if content_parts else '[No content extracted]'
