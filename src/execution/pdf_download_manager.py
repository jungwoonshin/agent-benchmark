"""PDF Download Manager - Handles PDF downloads and storage."""

import logging
import re
from typing import Any, Dict, List, Optional

from ..models import Attachment
from ..tools import ToolBelt


class PDFDownloadManager:
    """Manages PDF downloads from search results and ensures they're available for subsequent subtasks."""

    def __init__(
        self, tool_belt: ToolBelt, logger: logging.Logger
    ):
        """
        Initialize PDF Download Manager.

        Args:
            tool_belt: ToolBelt instance with download capabilities.
            logger: Logger instance.
        """
        self.tool_belt = tool_belt
        self.logger = logger

    def download_pdf_from_url(
        self, url: str, filename: Optional[str] = None
    ) -> Optional[Attachment]:
        """
        Download a PDF from a URL.

        Args:
            url: URL of the PDF to download.
            filename: Optional filename for the downloaded file.

        Returns:
            Attachment object if successful, None otherwise.
        """
        try:
            self.logger.info(f'Downloading PDF from URL: {url}')
            attachment = self.tool_belt.download_file_from_url(url, filename)
            self.logger.info(
                f'Successfully downloaded PDF: {attachment.filename} '
                f'({len(attachment.data)} bytes)'
            )
            return attachment
        except Exception as e:
            self.logger.error(
                f'Failed to download PDF from {url}: {e}', exc_info=True
            )
            return None

    def extract_pdf_urls_from_text(self, text: str) -> List[str]:
        """
        Extract PDF URLs from text content (e.g., from search result snippets or results).

        Args:
            text: Text content to search for PDF URLs.

        Returns:
            List of PDF URLs found in the text.
        """
        pdf_urls = []
        if not text:
            return pdf_urls

        # Pattern for arXiv PDF URLs
        arxiv_pattern = r'https?://arxiv\.org/pdf/[\d.]+\.pdf'
        arxiv_matches = re.findall(arxiv_pattern, text, re.IGNORECASE)
        pdf_urls.extend(arxiv_matches)

        # Pattern for general PDF URLs
        pdf_pattern = r'https?://[^\s]+\.pdf'
        pdf_matches = re.findall(pdf_pattern, text, re.IGNORECASE)
        pdf_urls.extend(pdf_matches)

        # Pattern for arXiv abstract pages (convert to PDF)
        arxiv_abs_pattern = r'https?://arxiv\.org/abs/([\d.]+)'
        arxiv_abs_matches = re.findall(arxiv_abs_pattern, text, re.IGNORECASE)
        for paper_id in arxiv_abs_matches:
            pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
            if pdf_url not in pdf_urls:
                pdf_urls.append(pdf_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in pdf_urls:
            if url.lower() not in seen:
                seen.add(url.lower())
                unique_urls.append(url)

        return unique_urls

    def download_pdfs_from_search_results(
        self,
        search_results: List[Any],
        subtask_description: str,
        problem: str,
        pdf_detector: Any,
        result_text: Optional[str] = None,
    ) -> List[Attachment]:
        """
        Download PDFs from search results based on subtask requirements.

        Args:
            search_results: List of SearchResult objects.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            pdf_detector: PDFDownloadDetector instance.
            result_text: Optional text content from search results (may contain PDF URLs).

        Returns:
            List of downloaded PDF attachments.
        """
        downloaded_pdfs = []
        downloaded_urls = set()  # Track downloaded URLs to avoid duplicates

        # First, extract PDF URLs from result text if provided
        if result_text:
            self.logger.info('Extracting PDF URLs from result text...')
            text_pdf_urls = self.extract_pdf_urls_from_text(result_text)
            for pdf_url in text_pdf_urls:
                if pdf_url not in downloaded_urls:
                    attachment = self.download_pdf_from_url(pdf_url)
                    if attachment:
                        downloaded_pdfs.append(attachment)
                        downloaded_urls.add(pdf_url)
                        self.logger.info(
                            f'Downloaded PDF from text: {attachment.filename} from {pdf_url}'
                        )

        if not search_results:
            self.logger.info(
                f'Downloaded {len(downloaded_pdfs)} PDF(s) from text content'
            )
            return downloaded_pdfs

        self.logger.info(
            f'Scanning {len(search_results)} search results for PDFs...'
        )

        for result in search_results:
            # Extract PDF URL from search result
            pdf_url = pdf_detector.extract_pdf_url_from_result(
                result, subtask_description
            )

            if pdf_url and pdf_url not in downloaded_urls:
                attachment = self.download_pdf_from_url(pdf_url)
                if attachment:
                    downloaded_pdfs.append(attachment)
                    downloaded_urls.add(pdf_url)
                    self.logger.info(
                        f'Downloaded PDF: {attachment.filename} from {pdf_url}'
                    )
            else:
                # Check if result URL is a direct PDF link
                if hasattr(result, 'url'):
                    url = result.url
                elif isinstance(result, dict):
                    url = result.get('url', '')
                else:
                    continue

                if not url or url in downloaded_urls:
                    continue

                url_lower = url.lower()
                if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
                    attachment = self.download_pdf_from_url(url)
                    if attachment:
                        downloaded_pdfs.append(attachment)
                        downloaded_urls.add(url)
                        self.logger.info(
                            f'Downloaded PDF: {attachment.filename} from {url}'
                        )
                else:
                    # Check snippet/content for PDF URLs
                    snippet = ''
                    if hasattr(result, 'snippet'):
                        snippet = result.snippet
                    elif isinstance(result, dict):
                        snippet = result.get('snippet', '')
                    
                    if snippet:
                        snippet_urls = self.extract_pdf_urls_from_text(snippet)
                        for pdf_url in snippet_urls:
                            if pdf_url not in downloaded_urls:
                                attachment = self.download_pdf_from_url(pdf_url)
                                if attachment:
                                    downloaded_pdfs.append(attachment)
                                    downloaded_urls.add(pdf_url)
                                    self.logger.info(
                                        f'Downloaded PDF from snippet: {attachment.filename} from {pdf_url}'
                                    )

        self.logger.info(
            f'Downloaded {len(downloaded_pdfs)} PDF(s) total from search results and text'
        )
        return downloaded_pdfs

    def ensure_pdf_in_attachments(
        self,
        attachments: List[Attachment],
        pdf_url: Optional[str] = None,
        search_results: Optional[List[Any]] = None,
        subtask_description: Optional[str] = None,
        pdf_detector: Optional[Any] = None,
    ) -> List[Attachment]:
        """
        Ensure a PDF is in the attachments list, downloading if necessary.

        Args:
            attachments: Current attachments list.
            pdf_url: Optional direct PDF URL to download.
            search_results: Optional search results to scan for PDFs.
            subtask_description: Optional subtask description for context.
            pdf_detector: Optional PDFDownloadDetector instance.

        Returns:
            Updated attachments list with PDFs added.
        """
        # Check if we already have PDFs in attachments
        existing_pdfs = [
            att for att in attachments if att.filename.lower().endswith('.pdf')
        ]

        if existing_pdfs:
            self.logger.debug(
                f'Found {len(existing_pdfs)} existing PDF(s) in attachments'
            )

        # Download from direct URL if provided
        if pdf_url:
            attachment = self.download_pdf_from_url(pdf_url)
            if attachment:
                # Check if we already have this PDF (by URL or filename)
                url_in_attachments = any(
                    hasattr(att, 'metadata')
                    and att.metadata.get('source_url') == pdf_url
                    for att in attachments
                )
                if not url_in_attachments:
                    attachments.append(attachment)
                    self.logger.info(
                        f'Added PDF to attachments: {attachment.filename}'
                    )

        # Download from search results if provided
        if search_results and pdf_detector and subtask_description:
            downloaded_pdfs = self.download_pdfs_from_search_results(
                search_results, subtask_description, '', pdf_detector
            )
            for pdf in downloaded_pdfs:
                # Check if we already have this PDF
                url_in_attachments = any(
                    hasattr(att, 'metadata')
                    and att.metadata.get('source_url')
                    == pdf.metadata.get('source_url')
                    for att in attachments
                )
                if not url_in_attachments:
                    attachments.append(pdf)
                    self.logger.info(
                        f'Added PDF to attachments: {pdf.filename}'
                    )

        return attachments

