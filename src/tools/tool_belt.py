"""ToolBelt class providing various tools for the agent."""

import logging
from typing import Any, Dict, List, Optional

from ..models import Attachment, SearchResult
from .browser_tool import BrowserTool
from .context_extractor import ContextExtractor
from .file_handler import FileHandler
from .image_recognition import ImageRecognition
from .llm_reasoning import LLMReasoningTool
from .search_tool import SearchTool


class ToolBelt:
    """Collection of tools for data retrieval and processing."""

    def __init__(self):
        """Initializes the ToolBelt. A logger will be set via set_logger()."""
        # Default to a null logger to prevent errors if not set
        self.logger = logging.getLogger('null')
        self.llm_service = None  # Optional LLM service for intelligent extraction

        # Tool instances (initialized in set_logger)
        self.llm_reasoning_tool = None
        self.context_extractor = None
        self.search_tool = None
        self.file_handler = None
        self.browser_tool = None
        self.image_recognition = None

    def set_logger(self, logger: logging.Logger):
        """Receives and sets the logger from the Agent."""
        self.logger = logger
        self.logger.info('ToolBelt logger initialized.')

        # Initialize tool instances
        self.llm_reasoning_tool = LLMReasoningTool(
            self.llm_service, logger, image_recognition=None
        )
        self.context_extractor = ContextExtractor(logger)
        self.search_tool = SearchTool(logger)
        self.image_recognition = ImageRecognition(logger, self.llm_service)
        self.file_handler = FileHandler(
            logger,
            image_recognition=self.image_recognition,
            llm_service=self.llm_service,
        )
        self.browser_tool = BrowserTool(
            logger,
            llm_service=self.llm_service,
            image_recognition=self.image_recognition,
        )

        # Set image recognition in LLM reasoning tool
        self.llm_reasoning_tool.set_image_recognition(self.image_recognition)

    def set_llm_service(self, llm_service):
        """Set the LLM service for intelligent extraction."""
        self.llm_service = llm_service

        # Update all tools that need LLM service
        if self.llm_reasoning_tool:
            self.llm_reasoning_tool.set_llm_service(llm_service)
        if self.image_recognition:
            self.image_recognition.set_llm_service(llm_service)
        if self.file_handler:
            self.file_handler.set_llm_service(llm_service)
        if self.browser_tool:
            self.browser_tool.set_llm_service(llm_service)

    # LLM Reasoning methods
    def llm_reasoning(self, task_description: str, context: dict = None) -> str:
        """
        Performs calculations, data processing, and analysis using LLM reasoning.

        Args:
            task_description: Description of what needs to be calculated/analyzed
            context: Optional context dictionary with variables and data available

        Returns:
            String representation of the reasoning result or answer
        """
        if not self.llm_reasoning_tool:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.llm_reasoning_tool.llm_reasoning(task_description, context)

    def llm_reasoning_with_images(
        self, task_description: str, context: dict = None, images: List[bytes] = None
    ) -> str:
        """
        Performs calculations, data processing, and analysis using visual LLM with image inputs.

        Args:
            task_description: Description of what needs to be calculated/analyzed
            context: Optional context dictionary with variables and data available
            images: List of image data as bytes for visual processing

        Returns:
            String representation of the reasoning result or answer
        """
        if not self.llm_reasoning_tool:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.llm_reasoning_tool.llm_reasoning_with_images(
            task_description, context, images
        )

    def code_interpreter(self, python_code: str, context: dict = None) -> str:
        """
        DEPRECATED: This method is disabled. Use llm_reasoning instead.
        Kept for backward compatibility but redirects to LLM reasoning.
        """
        if not self.llm_reasoning_tool:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.llm_reasoning_tool.code_interpreter(python_code, context)

    # Context extraction methods (kept for backward compatibility)
    def _extract_zip_codes_from_context(self, context: dict, code: str) -> List[str]:
        """Extract zip codes from context (search results, URLs, text)."""
        if not self.context_extractor:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.context_extractor.extract_zip_codes_from_context(context, code)

    def _extract_dates_from_context(self, context: dict, code: str) -> List[str]:
        """Extract dates from context."""
        if not self.context_extractor:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.context_extractor.extract_dates_from_context(context, code)

    def _extract_numbers_from_context(self, context: dict, code: str) -> List[float]:
        """Extract numbers from context."""
        if not self.context_extractor:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.context_extractor.extract_numbers_from_context(context, code)

    def _extract_urls_from_context(self, context: dict, code: str) -> List[str]:
        """Extract URLs from context."""
        if not self.context_extractor:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.context_extractor.extract_urls_from_context(context, code)

    def _extract_structured_data_from_context(
        self, context: dict, code: str
    ) -> Optional[str]:
        """Try to extract any structured data from context."""
        if not self.context_extractor:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.context_extractor.extract_structured_data_from_context(
            context, code
        )

    # Search method
    def search(
        self, query: str, num_results: int = 5, search_type: str = 'web'
    ) -> List[SearchResult]:
        """
        Performs a web or specialized search using Google Custom Search API.

        Args:
            query: Search query string.
            num_results: Number of results to return (default: 5).
            search_type: Type of search (default: 'web').

        Returns:
            List of SearchResult objects.
        """
        if not self.search_tool:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.search_tool.search(query, num_results, search_type)

    # File handling methods
    def read_attachment(
        self,
        attachment: Attachment,
        options: dict = None,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Smart file reader that extracts text and images from various common formats.

        Args:
            attachment: File attachment to read.
            options: Optional dict with file-specific options.
            problem: Optional problem description for relevance filtering.
            query_analysis: Optional query analysis for relevance filtering.

        Returns:
            Extracted text content from the file.
        """
        if not self.file_handler:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.file_handler.read_attachment(
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
        if not self.file_handler:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.file_handler.analyze_media(attachment, analysis_type)

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
            filename: Optional filename.
            timeout: Request timeout in seconds (default: 30.0).
            max_retries: Maximum number of retries (default: 2).

        Returns:
            Attachment object with downloaded file data.
        """
        if not self.file_handler:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.file_handler.download_file_from_url(
            url, filename, timeout, max_retries
        )

    # Browser navigation method
    def browser_navigate(
        self,
        url: str,
        action: Optional[str] = None,
        link_text: Optional[str] = None,
        selector: Optional[str] = None,
        extraction_query: Optional[str] = None,
        max_retries: int = 3,
        capture_screenshot: bool = False,
    ) -> Dict[str, Any]:
        """
        Navigate to a web page and optionally interact with it.

        Args:
            url: URL to navigate to.
            action: Optional action to perform.
            link_text: Optional link text to click.
            selector: Optional CSS selector for extraction.
            extraction_query: Optional query for LLM-based extraction.
            max_retries: Maximum number of retry attempts (default: 3).
            capture_screenshot: If True, capture screenshot (default: False).

        Returns:
            Dictionary with page data or extracted content.
        """
        if not self.browser_tool:
            raise ValueError('ToolBelt not initialized. Call set_logger() first.')
        return self.browser_tool.browser_navigate(
            url,
            action,
            link_text,
            selector,
            extraction_query,
            max_retries,
            capture_screenshot,
        )
