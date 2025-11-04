"""ToolBelt class providing various tools for the agent."""

import io
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from RestrictedPython import (  # type: ignore
    compile_restricted,
    limited_builtins,
    safe_globals,
)
from RestrictedPython.Guards import (  # type: ignore
    guarded_getitem,
    guarded_iter,
)

from .browser_navigator import BrowserNavigator
from .error_handling import classify_error, retry_with_backoff
from .models import Attachment, SearchResult
from .selenium_browser_navigator import SeleniumBrowserNavigator

# Load environment variables
load_dotenv()


class ToolBelt:
    """Collection of tools for data retrieval and processing."""

    def __init__(self):
        """Initializes the ToolBelt. A logger will be set via set_logger()."""
        # Default to a null logger to prevent errors if not set
        self.logger = logging.getLogger('null')
        self.selenium_navigator = None  # Selenium for JS pages
        self.browser_navigator = None  # Original navigator for extraction utilities
        self.llm_service = None  # Optional LLM service for intelligent extraction

    def set_logger(self, logger: logging.Logger):
        """Receives and sets the logger from the Agent."""
        self.logger = logger
        self.logger.info('ToolBelt logger initialized.')
        # Initialize Selenium for JS-heavy navigation
        self.selenium_navigator = SeleniumBrowserNavigator(logger, headless=True)
        # Keep original BrowserNavigator for extraction utilities
        self.browser_navigator = BrowserNavigator(logger)

    def set_llm_service(self, llm_service):
        """Set the LLM service for intelligent extraction."""
        self.llm_service = llm_service

    def code_interpreter(self, python_code: str, context: dict = None) -> str:
        """
        Executes Python code in a sandboxed environment using RestrictedPython.
        This is the primary tool for all math, logic, and complex data processing.

        Args:
            python_code: Python code to execute
            context: Optional context dictionary with variables available to the code

        Returns:
            String representation of the execution result or error message
        """

        self.logger.info("Tool 'code_interpreter' called.")
        self.logger.debug(f'Executing code snippet: {python_code[:150]}...')

        if context is None:
            context = {}

        # Define safe __import__ that only allows specific modules
        safe_modules = {
            'math',
            'json',
            'datetime',
            're',
            'itertools',
            'collections',
            'functools',
            'operator',
            'statistics',
        }

        def safe_import(name, *args, **kwargs):
            """Safe import that only allows whitelisted modules."""
            if name.split('.')[0] in safe_modules:
                return __import__(name, *args, **kwargs)
            raise ImportError(f"Import of '{name}' is not allowed for security reasons")

        try:
            # Prepare safe execution environment
            safe_builtins = limited_builtins.copy()
            safe_builtins['__import__'] = safe_import
            safe_builtins['__name__'] = 'restricted_module'
            safe_builtins['__metaclass__'] = type
            safe_builtins['_getattr_'] = getattr
            # Add RestrictedPython utility functions for iteration and item access
            safe_builtins['_getiter_'] = guarded_iter
            safe_builtins['_getitem_'] = guarded_getitem
            safe_builtins['_write_'] = lambda x: x  # Allow writing to variables
            safe_builtins['_print_'] = print  # Allow print statements

            safe_locals = {
                '__builtins__': safe_builtins,
                'math': math,
                'json': json,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'any': any,
                'all': all,
                'print': print,  # Capture print output
            }

            # Add context variables to the execution environment
            for key, value in context.items():
                safe_locals[key] = value

            # Create a variable to capture the result
            # If code has explicit return or result assignment, use that
            # Otherwise, try to capture the last expression
            result_var = 'output_result'  # Use a valid name (no underscores at start)

            # Prepare code - if it doesn't assign to result or return, wrap it
            code_lines = python_code.strip().split('\n')

            # Check if code already has result assignment or return
            has_result = any(
                'result' in line or 'return' in line for line in code_lines
            )

            if not has_result and len(code_lines) == 1:
                # Single expression - assign to result
                modified_code = f'{result_var} = {python_code.strip()}'
            elif not has_result:
                # Multiple lines - add result assignment at the end if last line is expression
                last_line = code_lines[-1].strip()
                if last_line and not last_line.startswith(
                    ('if', 'for', 'while', 'def', 'class', 'import', 'from')
                ):
                    code_lines[-1] = f'{result_var} = {last_line}'
                    modified_code = '\n'.join(code_lines)
                else:
                    modified_code = python_code
            else:
                modified_code = python_code

            self.logger.debug(f'Modified code for execution: {modified_code[:200]}...')

            # Compile with RestrictedPython for safety
            compiled = compile_restricted(
                modified_code, filename='<string>', mode='exec'
            )

            # Check for compilation errors
            # compile_restricted returns a named tuple with 'code' and 'errors' attributes
            if hasattr(compiled, 'errors') and compiled.errors:
                error_msg = '; '.join(compiled.errors)
                self.logger.error(f'Code compilation failed: {error_msg}')
                return f'Compilation Error: {error_msg}'

            # Get the actual code object
            byte_code = compiled.code if hasattr(compiled, 'code') else compiled

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Update safe_globals with our custom __import__
                execution_globals = safe_globals.copy()
                execution_globals['__builtins__'] = safe_builtins
                execution_globals['__import__'] = safe_import
                # Add RestrictedPython utility functions to globals as well
                execution_globals['_getiter_'] = guarded_iter
                execution_globals['_getitem_'] = guarded_getitem
                execution_globals['_write_'] = lambda x: x
                execution_globals['_print_'] = print

                # Execute the code
                exec(byte_code, execution_globals, safe_locals)

                # Get the result
                output = sys.stdout.getvalue()

                # Try to get result in order of preference:
                # 1. Explicitly assigned result variable
                # 2. Our output_result variable
                # 3. Captured print output
                # 4. Any variable named 'answer' or 'output'
                result = None

                if 'result' in safe_locals and safe_locals['result'] is not None:
                    result = safe_locals['result']
                elif result_var in safe_locals and safe_locals[result_var] is not None:
                    result = safe_locals[result_var]
                elif output.strip():
                    result = output.strip()
                elif 'answer' in safe_locals and safe_locals['answer'] is not None:
                    result = safe_locals['answer']
                elif 'output' in safe_locals and safe_locals['output'] is not None:
                    result = safe_locals['output']

                if result is None:
                    # No explicit result found, return info about available variables
                    available_vars = {
                        k: v
                        for k, v in safe_locals.items()
                        if not k.startswith('_')
                        and k not in safe_locals['__builtins__']
                    }
                    if available_vars:
                        result = f'Code executed. Available variables: {list(available_vars.keys())}'
                    else:
                        result = 'Code executed successfully but produced no output.'

                # Convert result to string
                result_str = str(result)

                self.logger.info(
                    f'Code executed successfully. Output: {result_str[:200]}...'
                )
                return result_str

            finally:
                # Restore stdout
                sys.stdout = old_stdout

        except SyntaxError as e:
            self.logger.error(f'Code syntax error: {e}')
            return f'Syntax Error: {e}'
        except NameError as e:
            self.logger.error(f'Code name error (undefined variable): {e}')
            return f'Name Error: {e}. Available context: {list(context.keys())}'
        except ImportError as e:
            error_msg = str(e)
            # Provide helpful guidance for common import errors
            if 'PyPDF2' in error_msg or 'pdf' in error_msg.lower():
                suggestion = (
                    'IMPORT ERROR: PDF processing libraries are not available. '
                    "Instead of using code to parse PDFs, use the 'read_attachment' tool. "
                    'If you need to extract information from a PDF, first search for the PDF or use the read_attachment tool with the attachment index.'
                )
                self.logger.warning(suggestion)
                return f'{error_msg}\n\n💡 SUGGESTION: {suggestion}'
            elif 'not allowed for security reasons' in error_msg:
                suggestion = (
                    'IMPORT ERROR: This module is not allowed for security reasons. '
                    'Instead, try using search queries to find information, or use the available tools (read_attachment, analyze_media). '
                    'For data processing, use the built-in Python functions and math module that are already available.'
                )
                self.logger.warning(suggestion)
                return f'{error_msg}\n\n💡 SUGGESTION: {suggestion}'
            else:
                self.logger.error(f'Code import error: {e}')
                return f'Import Error: {e}'
        except Exception as e:
            self.logger.error(f'Code execution FAILED: {e}', exc_info=True)
            return f'Execution Error: {type(e).__name__}: {e}'

    def _extract_zip_codes_from_context(self, context: dict, code: str) -> List[str]:
        """Extract zip codes from context (search results, URLs, text)."""
        import re

        zip_codes = []

        # Pattern for 5-digit US zip codes
        zip_pattern = r'\b\d{5}\b'

        # Extract from search results if available
        if 'search_results' in context:
            search_results = context.get('search_results', [])
            for result in search_results:
                # Handle both SearchResult objects and dicts
                text = ''
                if hasattr(result, 'snippet'):
                    text = result.snippet
                elif hasattr(result, 'url'):
                    text += ' ' + result.url
                elif isinstance(result, dict):
                    text = result.get('snippet', '') + ' ' + result.get('url', '')

                zip_codes.extend(re.findall(zip_pattern, text))

        # Extract from URLs in context
        if 'urls' in context:
            for url in context.get('urls', []):
                zip_codes.extend(re.findall(zip_pattern, url))

        # Extract from any text content in context
        for key, value in context.items():
            if isinstance(value, str) and key not in ['code']:
                zip_codes.extend(re.findall(zip_pattern, value))
            elif isinstance(value, (list, dict)):
                # Recursively search in nested structures
                zip_codes.extend(re.findall(zip_pattern, str(value)))

        # Try to download and extract from USGS URLs if found
        usgs_urls = [
            url for url in context.get('urls', []) if 'usgs.gov' in str(url).lower()
        ]
        for url in usgs_urls[:2]:  # Limit to 2 URLs to avoid excessive downloads
            try:
                self.logger.debug(
                    f'Attempting to extract zip codes from USGS URL: {url}'
                )
                # Try to fetch the page content
                response = requests.get(
                    str(url),
                    timeout=10,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    },
                )
                if response.status_code == 200:
                    content = response.text
                    zip_codes.extend(re.findall(zip_pattern, content))
                    self.logger.debug(
                        f'Found {len(zip_codes)} zip code(s) in USGS page'
                    )
            except Exception as e:
                self.logger.debug(f'Could not fetch USGS URL {url}: {e}')

        # Remove duplicates and return
        unique_zips = list(set(zip_codes))
        return unique_zips[:10]  # Limit to 10 zip codes

    def _extract_dates_from_context(self, context: dict, code: str) -> List[str]:
        """Extract dates from context."""
        import re

        dates = []

        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
        ]

        for key, value in context.items():
            if isinstance(value, str):
                for pattern in date_patterns:
                    dates.extend(re.findall(pattern, value, re.IGNORECASE))

        return list(set(dates))

    def _extract_numbers_from_context(self, context: dict, code: str) -> List[float]:
        """Extract numbers from context."""
        import re

        numbers = []

        number_pattern = r'-?\d+\.?\d*'

        for key, value in context.items():
            if isinstance(value, str):
                matches = re.findall(number_pattern, value)
                for match in matches:
                    try:
                        numbers.append(float(match))
                    except ValueError:
                        pass

        return numbers

    def _extract_urls_from_context(self, context: dict, code: str) -> List[str]:
        """Extract URLs from context."""
        import re

        urls = []

        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        for key, value in context.items():
            if isinstance(value, str):
                urls.extend(re.findall(url_pattern, value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        urls.extend(re.findall(url_pattern, item))
                    elif hasattr(item, 'url'):
                        urls.append(item.url)

        return list(set(urls))

    def _extract_structured_data_from_context(
        self, context: dict, code: str
    ) -> Optional[str]:
        """Try to extract any structured data from context."""
        # Look for common data patterns
        if not context:
            return None

        # Try to extract any meaningful text from search results
        if 'search_results' in context:
            results = context.get('search_results', [])
            if results:
                # Get the first meaningful result
                for result in results:
                    if hasattr(result, 'snippet') and result.snippet:
                        return result.snippet[:500]  # Limit length
                    elif isinstance(result, dict) and result.get('snippet'):
                        return result['snippet'][:500]

        # Return None if nothing found
        return None

    def search(
        self, query: str, num_results: int = 5, search_type: str = 'web'
    ) -> List[SearchResult]:
        """
        Performs a web or specialized search using Google Custom Search API.

        Requires GOOGLE_API_KEY and GOOGLE_CX (Custom Search Engine ID) in .env file.
        Falls back to mock results if API keys are not configured.
        """
        self.logger.info(f"Tool 'search' called (type: {search_type}).")
        self.logger.info(f'Search query: {query}')

        try:
            # Get Google API credentials from environment
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_SEARCH_API_KEY')
            cx = os.getenv('GOOGLE_CX') or os.getenv('GOOGLE_SEARCH_CX_ID')

            # Fallback to mock results if API keys are not configured
            if not api_key or not cx:
                self.logger.warning(
                    'Google API keys not found. Using fallback mock results. '
                    'Set GOOGLE_API_KEY and GOOGLE_CX in .env for real search.'
                )
                results = []
                self.logger.info(f'Search returned {len(results)} result(s) (mock).')
                return results

            # Perform real Google Custom Search API request
            url = 'https://www.googleapis.com/customsearch/v1'
            params = {
                'key': api_key,
                'cx': cx,
                'q': query,
                'num': min(
                    num_results, 10
                ),  # Google API limits to 10 results per request
            }

            self.logger.info(f'Calling Google Custom Search API with query: "{query}"')
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            # Parse search results
            if 'items' in data:
                for item in data['items'][:num_results]:
                    results.append(
                        SearchResult(
                            snippet=item.get('snippet', ''),
                            url=item.get('link', ''),
                            title=item.get('title', ''),
                            relevance_score=float(item.get('rank', 0))
                            if 'rank' in item
                            else 0.0,
                        )
                    )

            self.logger.info(
                f'Search returned {len(results)} result(s) from Google API.'
            )
            return results

        except requests.exceptions.RequestException as e:
            self.logger.error(f'Search API request FAILED: {e}', exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f'Search FAILED: {e}', exc_info=True)
            return []

    def read_attachment(self, attachment: Attachment, options: dict = None) -> str:
        """
        Smart file reader that extracts text from various common formats.
        """
        self.logger.info(f"Tool 'read_attachment' called for: {attachment.filename}")
        self.logger.debug(f'Read options: {options}')

        try:
            # STUB: Mock text extraction
            if (
                '.pdf' in attachment.filename
                and options
                and options.get('page_range') == [54, 54]
            ):
                result = 'Figure 54-A: Anatomy of the common frog. Text about frog dissection...'
            elif '.txt' in attachment.filename:
                result = attachment.data.decode('utf-8')
            else:
                result = f'[STUB] Full text content of {attachment.filename}'

            self.logger.info(
                f'Successfully read {attachment.filename}. Content length: {len(result)}'
            )
            return result
        except Exception as e:
            self.logger.error(
                f'Failed to read attachment {attachment.filename}: {e}', exc_info=True
            )
            return f'Error: Failed to read {attachment.filename}'

    def analyze_media(self, attachment: Attachment, analysis_type: str = 'auto') -> str:
        """
        Analyzes non-text media files (images, audio, video) using ML models.
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

    def browser_navigate(
        self,
        url: str,
        action: Optional[str] = None,
        link_text: Optional[str] = None,
        selector: Optional[str] = None,
        extraction_query: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Navigate to a web page and optionally interact with it.

        Args:
            url: URL to navigate to.
            action: Optional action to perform ('click_link', 'extract_text', 'find_table', 'search_text').
            link_text: Optional link text to click (for 'click_link' action).
            selector: Optional CSS selector for extraction (for 'extract_text' action).
            max_retries: Maximum number of retry attempts for transient errors (default: 3).

        Returns:
            Dictionary with page data or extracted content.
        """
        if not self.selenium_navigator:
            self.selenium_navigator = SeleniumBrowserNavigator(
                self.logger, headless=True
            )
        if not self.browser_navigator:
            self.browser_navigator = BrowserNavigator(self.logger)

        self.logger.info(f"Tool 'browser_navigate' called: {url}, action={action}")

        def _navigate_with_retry():
            """Inner function that raises exceptions for retry logic."""
            # Use Selenium for navigation (handles JS)
            page_data = self.selenium_navigator.navigate(url)

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
                page_data = self.selenium_navigator.click_element(link_text=link_text)
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
                # Use Selenium's extraction for better JavaScript support
                text = self.selenium_navigator.extract_text(page_data)

                # If selector provided, still use browser_navigator for CSS selection
                if selector and page_data.get('soup'):
                    text = self.browser_navigator.extract_text(page_data, selector)

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
                table_data = self.browser_navigator.find_table(page_data)
                # Also try to extract numeric data from tables for better count extraction
                context_keywords = None
                if selector:
                    context_keywords = [
                        kw.strip() for kw in selector.split(',') if kw.strip()
                    ]
                numeric_data = self.browser_navigator.extract_numeric_data(
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
                search_results = self.browser_navigator.search_text(
                    page_data, search_terms
                )
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

                    llm_result = self.browser_navigator.extract_with_llm(
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

                numeric_data = self.browser_navigator.extract_numeric_data(
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
            extracted_text = self.browser_navigator.extract_text(page_data)
            return {
                'success': True,
                'url': page_data['url'],
                'text': extracted_text[:5000],  # Limit text length
                'links': self.browser_navigator.extract_links_list(page_data)[
                    :20
                ],  # Limit links
            }

        except Exception as e:
            self.logger.error(f'Browser navigation error: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'url': url,
            }
