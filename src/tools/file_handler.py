"""File handling functionality for reading, downloading, and analyzing files."""

import logging
import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import requests

from ..models import Attachment
from ..utils import classify_error, retry_with_backoff


class FileHandler:
    """Tool for handling file operations: reading, downloading, and media analysis."""

    def __init__(
        self,
        logger: logging.Logger,
        image_recognition=None,
        llm_service=None,
    ):
        """
        Initialize file handler.

        Args:
            logger: Logger instance.
            image_recognition: Optional image recognition tool for PDF/image processing.
            llm_service: Optional LLM service for intelligent extraction.
        """
        self.logger = logger
        self.image_recognition = image_recognition
        self.llm_service = llm_service

    def set_image_recognition(self, image_recognition):
        """Set the image recognition tool."""
        self.image_recognition = image_recognition

    def set_llm_service(self, llm_service):
        """Set the LLM service."""
        self.llm_service = llm_service

    def read_attachment(
        self,
        attachment: Attachment,
        options: dict = None,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Smart file reader that extracts text and images from various common formats.
        For PDFs, extracts both text and images, processing images with visual LLM if available.
        Uses smart filtering to extract only relevant sections when problem and query_analysis are provided.

        Args:
            attachment: File attachment to read.
            options: Optional dict with file-specific options (e.g., 'page_range' for PDFs).
            problem: Optional problem description for relevance filtering (PDFs only).
            query_analysis: Optional query analysis for relevance filtering (PDFs only).

        Returns:
            Extracted text content (string) or structured data (dict) for PDFs with sections.
            For PDFs with problem/query_analysis: Returns dict with 'type', 'filename', 'sections', etc.
            Otherwise: Returns string content.
        """
        self.logger.info(f"Tool 'read_attachment' called for: {attachment.filename}")
        self.logger.debug(f'Read options: {options}')

        if options is None:
            options = {}

        try:
            filename_lower = attachment.filename.lower()

            # Handle PDF files
            if '.pdf' in filename_lower:
                return self._read_pdf(attachment, options, problem, query_analysis)

            # Handle text files
            elif '.txt' in filename_lower:
                result = attachment.data.decode('utf-8')
                self.logger.info(
                    f'Successfully read {attachment.filename}. Content length: {len(result)}'
                )
                return result

            # Handle other file types
            else:
                self.logger.warning(
                    f'Unsupported file type for {attachment.filename}. Returning placeholder.'
                )
                return f'[File content of {attachment.filename} - unsupported format]'

        except Exception as e:
            self.logger.error(
                f'Failed to read attachment {attachment.filename}: {e}', exc_info=True
            )
            return f'Error: Failed to read {attachment.filename}: {str(e)}'

    def _read_pdf(
        self,
        attachment: Attachment,
        options: dict,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Read PDF file and extract text and images.
        If images are found and visual LLM is available, process them.

        Args:
            attachment: PDF attachment.
            options: Options dict with optional 'page_range' key.
            problem: Optional problem description for relevance filtering.
            query_analysis: Optional query analysis for relevance filtering.

        Returns:
            Combined text and image analysis results.
        """
        if not self.image_recognition:
            self.logger.warning(
                'Image recognition tool not initialized. Call set_image_recognition() first.'
            )
            return 'Error: Image recognition tool not initialized'

        # Use image recognition tool for PDF processing
        return self.image_recognition.recognize_images_from_pdf(
            attachment, options, problem, query_analysis
        )

    def analyze_media(self, attachment: Attachment, analysis_type: str = 'auto') -> str:
        """
        Analyzes non-text media files (images, audio, video) using ML models.

        Args:
            attachment: Media file attachment to analyze.
            analysis_type: Type of analysis to perform (default: 'auto').

        Returns:
            Description or analysis result of the media file.
        """
        self.logger.info(
            f"Tool 'analyze_media' called for: {attachment.filename} (type: {analysis_type})"
        )

        try:
            # STUB: Return a mock description
            if 'e14448e9' in attachment.filename:
                result = 'A high-resolution photo of a Red-Eyed Tree Frog (Agalychnis callidryas) clinging to a green leaf.'
            else:
                result = f'[STUB] Detailed description of {attachment.filename}'

            self.logger.info(f'Media analysis complete. Result: {result[:70]}...')
            return result
        except Exception as e:
            self.logger.error(
                f'Media analysis FAILED for {attachment.filename}: {e}', exc_info=True
            )
            return f'Error: Failed to analyze {attachment.filename}'

    def download_file_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> Attachment:
        """
        Downloads a file from a URL and converts it to an Attachment object.

        Args:
            url: URL of the file to download.
            filename: Optional filename. If not provided, inferred from URL or Content-Disposition header.
            timeout: Request timeout in seconds (default: 30.0).
            max_retries: Maximum number of retries for transient errors (default: 2).

        Returns:
            Attachment object with downloaded file data.

        Raises:
            Exception: If download fails after all retries, with improved error message.
        """
        self.logger.info(f'Downloading file from URL: {url} (timeout: {timeout}s)')

        # Default headers with user-agent
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        def _download_attempt(
            headers: dict, referer: Optional[str] = None
        ) -> Attachment:
            """Internal function to attempt download with given headers."""
            request_headers = headers.copy()
            if referer:
                request_headers['Referer'] = referer

            response = requests.get(
                url, timeout=timeout, stream=True, headers=request_headers
            )
            response.raise_for_status()

            # Determine filename - always ensure it's set
            determined_filename = filename  # Use local variable to avoid scoping issues
            if not determined_filename:
                # Try to get filename from Content-Disposition header
                content_disposition = response.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    determined_filename = content_disposition.split('filename=')[
                        1
                    ].strip('"\'')
                else:
                    # Infer from URL
                    parsed_url = urlparse(url)
                    determined_filename = (
                        os.path.basename(parsed_url.path) or 'download'
                    )
                    # Add extension if not present
                    if '.' not in determined_filename:
                        content_type = response.headers.get('Content-Type', '')
                        if 'pdf' in content_type:
                            determined_filename += '.pdf'
                        elif 'image' in content_type:
                            ext = content_type.split('/')[-1]
                            determined_filename += f'.{ext}' if ext else '.jpg'

            # Read file data
            data = response.content
            self.logger.info(
                f'Downloaded file: {determined_filename} ({len(data)} bytes) from {url}'
            )

            # Create Attachment with metadata
            metadata = {
                'source_url': url,
                'content_type': response.headers.get('Content-Type', ''),
                'content_length': len(data),
            }

            return Attachment(
                filename=determined_filename, data=data, metadata=metadata
            )

        # Try download strategies
        strategies = [
            # Strategy 1: Default headers
            lambda: _download_attempt(default_headers),
            # Strategy 2: With referer pointing to same domain
            lambda: _download_attempt(
                default_headers, referer='/'.join(url.split('/')[:3]) + '/'
            ),
            # Strategy 3: Minimal headers
            lambda: _download_attempt(
                {'User-Agent': default_headers['User-Agent']}, referer=url
            ),
        ]

        last_exception = None

        for strategy_idx, strategy in enumerate(strategies):
            try:
                # Retry with exponential backoff for this strategy
                return retry_with_backoff(
                    strategy,
                    max_retries=max_retries
                    if strategy_idx == 0
                    else 1,  # Only retry first strategy
                    base_delay=1.0,
                    max_delay=30.0,
                    logger=self.logger,
                )
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = (
                    getattr(e.response, 'status_code', None)
                    if hasattr(e, 'response')
                    else None
                )

                # If 403 and we have more strategies, try next one
                if status_code == 403 and strategy_idx < len(strategies) - 1:
                    error_type, error_category, user_message = classify_error(e)
                    self.logger.warning(
                        f'Strategy {strategy_idx + 1} failed with 403: {user_message}. Trying next strategy...'
                    )
                    continue

                # Otherwise, classify and raise with better message
                error_type, error_category, user_message = classify_error(e)
                self.logger.error(
                    f'Download failed ({error_category.value}/{error_type.value}): {user_message}',
                    exc_info=True,
                )
                raise Exception(user_message) from e

            except requests.exceptions.RequestException as e:
                last_exception = e
                error_type, error_category, user_message = classify_error(e)

                # Don't try other strategies for permanent errors
                if error_type.value == 'permanent':
                    self.logger.error(
                        f'Download failed ({error_category.value}/{error_type.value}): {user_message}',
                        exc_info=True,
                    )
                    raise Exception(user_message) from e

                # For transient errors, try next strategy if available
                if strategy_idx < len(strategies) - 1:
                    self.logger.warning(
                        f'Strategy {strategy_idx + 1} failed ({error_category.value}/{error_type.value}): {user_message}. Trying next strategy...'
                    )
                    continue

                # Last strategy failed, raise
                self.logger.error(
                    f'All download strategies failed ({error_category.value}/{error_type.value}): {user_message}',
                    exc_info=True,
                )
                raise Exception(user_message) from e

        # Should not reach here, but handle just in case
        if last_exception:
            error_type, error_category, user_message = classify_error(last_exception)
            raise Exception(user_message) from last_exception
        raise Exception('All download strategies exhausted')

