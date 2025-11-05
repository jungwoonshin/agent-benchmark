"""Browser navigation functionality for web page interaction."""

import logging
from typing import Any, Dict, Optional

import requests

from ..browser import Browser
from ..utils import retry_with_backoff


class BrowserTool:
    """Tool for navigating web pages and extracting content."""

    def __init__(
        self,
        logger: logging.Logger,
        llm_service=None,
        image_recognition=None,
    ):
        """
        Initialize browser tool.

        Args:
            logger: Logger instance.
            llm_service: Optional LLM service for intelligent extraction.
            image_recognition: Optional image recognition tool for screenshot analysis.
        """
        self.logger = logger
        self.llm_service = llm_service
        self.image_recognition = image_recognition
        self.browser = None  # Unified browser for both navigation and extraction

    def initialize_navigators(self):
        """Initialize browser if not already initialized."""
        if not self.browser:
            self.browser = Browser(logger=self.logger, headless=True)

    def set_llm_service(self, llm_service):
        """Set the LLM service."""
        self.llm_service = llm_service

    def set_image_recognition(self, image_recognition):
        """Set the image recognition tool."""
        self.image_recognition = image_recognition

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
            action: Optional action to perform ('click_link', 'extract_text', 'find_table', 'search_text').
            link_text: Optional link text to click (for 'click_link' action).
            selector: Optional CSS selector for extraction (for 'extract_text' action).
            extraction_query: Optional query for LLM-based extraction.
            max_retries: Maximum number of retry attempts for transient errors (default: 3).
            capture_screenshot: If True, capture screenshot of the page for visual processing (default: False).

        Returns:
            Dictionary with page data or extracted content. May include 'screenshot' key if capture_screenshot=True.
        """
        self.initialize_navigators()

        self.logger.info(f"Tool 'browser_navigate' called: {url}, action={action}")

        def _navigate_with_retry():
            """Inner function that raises exceptions for retry logic."""
            # Use Selenium for navigation (handles JS)
            page_data = self.browser.navigate(url, use_selenium=True)

            if not page_data.get('success'):
                status_code = page_data.get('status_code', 0)
                error_msg = page_data.get('error', 'Failed to load page')

                # Create appropriate exception type based on status code
                # This allows retry_with_backoff to classify and handle it correctly
                if status_code in (500, 502, 503, 504):
                    # These are transient server errors - should be retried
                    exc = requests.exceptions.HTTPError(error_msg)
                    exc.response = type(
                        'obj', (object,), {'status_code': status_code}
                    )()
                    raise exc
                elif status_code == 403:
                    # Forbidden - permanent error
                    exc = requests.exceptions.HTTPError(error_msg)
                    exc.response = type(
                        'obj', (object,), {'status_code': status_code}
                    )()
                    raise exc
                elif status_code == 404:
                    # Not found - permanent error
                    exc = requests.exceptions.HTTPError(error_msg)
                    exc.response = type(
                        'obj', (object,), {'status_code': status_code}
                    )()
                    raise exc
                else:
                    # Other errors - return as-is for now
                    result = {
                        'success': False,
                        'error': error_msg,
                        'status_code': status_code,
                    }
                    if 'diagnostics' in page_data:
                        result['diagnostics'] = page_data['diagnostics']
                    return result

            return page_data

        try:
            # Use retry logic for transient errors
            page_data = retry_with_backoff(
                _navigate_with_retry,
                max_retries=max_retries,
                base_delay=1.0,
                max_delay=30.0,
                logger=self.logger,
            )

            # If we got a dict with success=False, return it
            if isinstance(page_data, dict) and not page_data.get('success'):
                return page_data

            # Perform action if specified
            if action == 'click_link' and link_text:
                # Use Selenium's click_element with link_text
                page_data = self.browser.click_element(link_text=link_text)
                if not page_data.get('success'):
                    error_msg = page_data.get('error', 'Failed to click link')
                    result = {
                        'success': False,
                        'error': error_msg,
                        'url': page_data.get('url', url),
                    }
                    # Include diagnostics if available
                    if 'diagnostics' in page_data:
                        result['diagnostics'] = page_data['diagnostics']
                    # Include original error details
                    if 'original_error' in page_data.get('diagnostics', {}):
                        result['original_error'] = page_data['diagnostics'][
                            'original_error'
                        ]
                    return result
            elif action == 'extract_text':
                # Use Browser's extraction (handles both static and JS pages)
                text = self.browser.extract_text(page_data, selector=selector)

                # Check if text extraction was successful
                if not text or len(text.strip()) < 10:
                    # Text extraction may have failed
                    diagnostics = {
                        'text_extraction_empty': True,
                        'text_length': len(text) if text else 0,
                        'content_length': len(page_data.get('content', '')),
                        'has_soup': page_data.get('soup') is not None,
                    }

                    # Check if this might be a JavaScript-rendered page
                    content = page_data.get('content', '')
                    if content and len(content) > 1000 and len(text.strip()) < 50:
                        diagnostics['likely_javascript_rendered'] = True
                        diagnostics['suggestion'] = (
                            'Page may be JavaScript-rendered. Content requires browser execution. '
                            'Consider using a different approach or browser automation tool.'
                        )

                    if diagnostics.get('likely_javascript_rendered'):
                        self.logger.warning(
                            f'Text extraction returned minimal content. '
                            f'This may be a JavaScript-rendered page. '
                            f'URL: {page_data["url"]}'
                        )

                    return {
                        'success': True,  # Page loaded, but extraction had issues
                        'url': page_data['url'],
                        'text': text,
                        'warning': 'Text extraction returned minimal or empty content',
                        'diagnostics': diagnostics,
                    }

                return {
                    'success': True,
                    'url': page_data['url'],
                    'text': text,
                }
            elif action == 'find_table':
                table_data = self.browser.find_table(page_data)
                # Also try to extract numeric data from tables for better count extraction
                context_keywords = None
                if selector:
                    context_keywords = [
                        kw.strip() for kw in selector.split(',') if kw.strip()
                    ]
                numeric_data = self.browser.extract_numeric_data(
                    page_data, context_keywords=context_keywords
                )
                return {
                    'success': True,
                    'url': page_data['url'],
                    'table_data': table_data,
                    'numeric_data': numeric_data,
                    'extracted_counts': numeric_data.get('table_numbers', []),
                }
            elif action == 'search_text' and selector:
                # Use selector as search terms (comma-separated)
                search_terms = [term.strip() for term in selector.split(',')]
                search_results = self.browser.search_text(page_data, search_terms)
                return {
                    'success': True,
                    'url': page_data['url'],
                    'search_results': search_results,
                }
            elif action == 'extract_count' or action == 'extract_statistics':
                # Use LLM extraction if available and extraction_query is provided
                # This provides more accurate extraction by avoiding false matches from dates, etc.
                if not self.llm_service:
                    self.logger.warning(
                        'LLM extraction not available (llm_service is None). '
                        'Falling back to regex-based extraction.'
                    )
                elif not extraction_query:
                    self.logger.warning(
                        'LLM extraction not available (extraction_query is missing). '
                        'Falling back to regex-based extraction.'
                    )

                if self.llm_service and extraction_query:
                    self.logger.info(
                        f'Using LLM extraction with query: {extraction_query}'
                    )
                    context_keywords = None
                    if selector:
                        context_keywords = [
                            kw.strip() for kw in selector.split(',') if kw.strip()
                        ]

                    llm_result = self.browser.extract_with_llm(
                        page_data=page_data,
                        extraction_query=extraction_query,
                        llm_service=self.llm_service,
                        context_keywords=context_keywords,
                    )

                    extracted_value = llm_result.get('extracted_value')
                    if extracted_value is not None:
                        # Convert to format compatible with existing code
                        return {
                            'success': True,
                            'url': page_data['url'],
                            'llm_extraction': llm_result,
                            'numeric_data': {
                                'numeric_values': [
                                    {
                                        'value': extracted_value,
                                        'context': llm_result.get('context', ''),
                                        'pattern': 'LLM extraction',
                                    }
                                ],
                                'counts': [
                                    {
                                        'value': extracted_value,
                                        'context': llm_result.get('context', ''),
                                        'pattern': 'LLM extraction',
                                    }
                                ],
                            },
                            'extracted_counts': [
                                {
                                    'value': extracted_value,
                                    'context': llm_result.get('context', ''),
                                    'pattern': 'LLM extraction',
                                    'confidence': llm_result.get('confidence', 0.0),
                                    'reasoning': llm_result.get('reasoning', ''),
                                }
                            ],
                            # For backward compatibility
                            'search_results': {
                                'found': [str(extracted_value)],
                                'not_found': [],
                                'contexts': {
                                    str(extracted_value): [
                                        llm_result.get('context', '')[:100]
                                    ]
                                },
                            },
                        }
                    else:
                        self.logger.warning(
                            f'LLM extraction returned None. '
                            f'Reasoning: {llm_result.get("reasoning", "unknown")}. '
                            f'Falling back to regex-based extraction.'
                        )

                # Fallback to regex-based extraction if LLM not available or failed
                # selector can contain comma-separated context keywords (e.g., "article,count,total")
                self.logger.debug(
                    'Using regex-based extraction (fallback method). '
                    'Note: This method works on plain text (HTML tags are stripped).'
                )
                context_keywords = None
                if selector:
                    context_keywords = [
                        kw.strip() for kw in selector.split(',') if kw.strip()
                    ]

                numeric_data = self.browser.extract_numeric_data(
                    page_data, context_keywords=context_keywords
                )

                # Return the most relevant counts/numbers
                all_numbers = (
                    numeric_data.get('numeric_values', [])
                    + numeric_data.get('counts', [])
                    + numeric_data.get('table_numbers', [])
                    + numeric_data.get('list_numbers', [])
                )

                # Extract unique values with their contexts
                unique_values = {}
                for item in all_numbers:
                    value = item['value']
                    if value not in unique_values:
                        unique_values[value] = item

                return {
                    'success': True,
                    'url': page_data['url'],
                    'numeric_data': numeric_data,
                    'extracted_counts': list(unique_values.values()),
                    # For backward compatibility, also include search_results format
                    'search_results': {
                        'found': [str(v['value']) for v in unique_values.values()],
                        'not_found': [],
                        'contexts': {
                            str(v['value']): [v.get('context', '')[:100]]
                            for v in unique_values.values()
                        },
                    },
                }

            # Return basic page data
            extracted_text = self.browser.extract_text(page_data)
            result = {
                'success': True,
                'url': page_data['url'],
                'text': extracted_text[:5000],  # Limit text length
                'links': self.browser.extract_links_list(page_data)[:20],  # Limit links
            }

            # Capture screenshot if requested
            if capture_screenshot and self.browser:
                screenshot = self.browser.take_screenshot(as_base64=False)
                if screenshot:
                    result['screenshot'] = screenshot
                    self.logger.debug(
                        f'Screenshot captured for {url}: {len(screenshot)} bytes'
                    )
                    # Optionally process screenshot with image recognition if available
                    if self.image_recognition and self.image_recognition.llm_service:
                        try:
                            screenshot_analysis = (
                                self.image_recognition.recognize_images_from_browser(
                                    screenshot,
                                    context={
                                        'url': result.get('url'),
                                        'text': result.get('text', '')[:500],
                                    },
                                )
                            )
                            result['screenshot_analysis'] = screenshot_analysis
                        except Exception as e:
                            self.logger.warning(f'Failed to analyze screenshot: {e}')

            return result

        except Exception as e:
            self.logger.error(f'Browser navigation error: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'url': url,
            }
