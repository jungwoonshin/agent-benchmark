"""
Unified Browser Navigation Module for web page interaction and data extraction.
This module provides a single Browser class that can handle both static HTML pages
(using requests) and dynamic, JavaScript-rendered pages (using Selenium).
"""

import json
import logging
import os
import re
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin

import html2text
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

if TYPE_CHECKING:
    from ..llm import LLMService


class Browser:
    """
    A unified browser for navigating and interacting with web pages.
    It can use either 'requests' for static pages or 'selenium' for dynamic pages.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        headless: bool = True,
        timeout: int = 30,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.headless = headless
        self.timeout = timeout

        # Requests session
        self.session = requests.Session()
        self.session.headers.update(
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        )

        # Selenium driver (initialized on demand)
        self.driver = None

    def _resolve_chromedriver_path(self) -> str:
        """
        Resolve the correct chromedriver executable path.
        Handles cases where ChromeDriverManager returns incorrect paths.
        """
        try:
            driver_path = ChromeDriverManager().install()
            driver_path = Path(driver_path)

            # Helper function to check if a file is a binary executable
            def is_binary_executable(file_path: Path) -> bool:
                """Check if a file is a binary executable (not a text file)."""
                try:
                    if not file_path.exists() or not file_path.is_file():
                        return False
                    # Check file size (text files are usually smaller, but not foolproof)
                    if (
                        file_path.stat().st_size < 1000
                    ):  # Very small files are likely not the driver
                        return False
                    # Check if it's a binary file by reading first bytes
                    with open(file_path, 'rb') as f:
                        first_bytes = f.read(16)
                        # Check for common binary formats:
                        # ELF (Linux): \x7fELF
                        # Mach-O (macOS): \xcf\xfa\xed\xfe (64-bit) or \xce\xfa\xed\xfe (32-bit)
                        # Universal binary: \xca\xfe\xba\xbe
                        binary_magic = (
                            first_bytes.startswith(b'\x7fELF')
                            or first_bytes.startswith(b'\xcf\xfa\xed\xfe')
                            or first_bytes.startswith(b'\xce\xfa\xed\xfe')
                            or first_bytes.startswith(b'\xca\xfe\xba\xbe')
                        )
                        return binary_magic
                except Exception:
                    return False

            # If the path exists, check if it's the executable
            if driver_path.exists():
                if driver_path.is_file() and driver_path.name == 'chromedriver':
                    if is_binary_executable(driver_path):
                        os.chmod(
                            str(driver_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                        )
                        return str(driver_path)
                elif driver_path.is_dir():
                    # If it's a directory, look for chromedriver inside
                    chromedriver_candidate = driver_path / 'chromedriver'
                    if chromedriver_candidate.exists() and is_binary_executable(
                        chromedriver_candidate
                    ):
                        os.chmod(
                            str(chromedriver_candidate),
                            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
                        )
                        return str(chromedriver_candidate)

                # Search in parent directories for the actual chromedriver
                search_paths = (
                    [driver_path.parent, driver_path.parent.parent]
                    if driver_path.is_file()
                    else [driver_path]
                )
                for search_dir in search_paths:
                    if search_dir and search_dir.exists():
                        # Look for chromedriver files in this directory
                        for item in search_dir.iterdir():
                            if item.is_file() and item.name == 'chromedriver':
                                if is_binary_executable(item):
                                    os.chmod(
                                        str(item),
                                        stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
                                    )
                                    return str(item)

            # If we still can't find it, search in the webdriver-manager cache
            wdm_cache = Path.home() / '.wdm' / 'drivers' / 'chromedriver'
            if wdm_cache.exists():
                for chromedriver_file in wdm_cache.rglob('chromedriver'):
                    if chromedriver_file.is_file() and is_binary_executable(
                        chromedriver_file
                    ):
                        os.chmod(
                            str(chromedriver_file),
                            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
                        )
                        self.logger.info(
                            f'Found chromedriver in cache: {chromedriver_file}'
                        )
                        return str(chromedriver_file)

            # Fallback: return the original path and let it fail with a clearer error
            self.logger.warning(
                f'Could not find chromedriver executable, using: {driver_path}'
            )
            return str(driver_path)
        except Exception as e:
            self.logger.error(f'Error resolving chromedriver path: {e}')
            raise

    def _initialize_driver(self):
        """Initialize the Selenium WebDriver with Chrome."""
        if self.driver:
            return

        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(
                '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            chrome_options.add_argument('--log-level=3')
            # Anti-detection options to avoid being blocked
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option(
                'excludeSwitches', ['enable-automation', 'enable-logging']
            )
            chrome_options.add_experimental_option('useAutomationExtension', False)
            # Set page load strategy to 'eager' to avoid waiting for all resources
            # This helps avoid timeouts on slow-loading pages
            try:
                from selenium.webdriver.common.page_load_strategy import (
                    PageLoadStrategy,
                )

                chrome_options.page_load_strategy = PageLoadStrategy.EAGER
            except ImportError:
                # Fallback for older Selenium versions
                chrome_options.page_load_strategy = 'eager'
            chromedriver_path = self._resolve_chromedriver_path()
            self.logger.debug(f'Using chromedriver at: {chromedriver_path}')
            service = Service(chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            # Execute script to remove webdriver property
            self.driver.execute_cdp_cmd(
                'Page.addScriptToEvaluateOnNewDocument',
                {
                    'source': """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
                },
            )
            self.driver.set_page_load_timeout(self.timeout)
            self.logger.info(
                f'Selenium WebDriver initialized (headless={self.headless})'
            )
        except Exception as e:
            self.logger.error(
                f'Failed to initialize Selenium WebDriver: {e}', exc_info=True
            )
            raise

    def navigate(
        self,
        url: str,
        use_selenium: bool = False,
        wait_for_js: float = 2.0,
        fallback_to_requests: bool = True,
    ) -> Dict[str, Any]:
        """
        Navigate to a URL and return page content.

        Args:
            url: URL to navigate to.
            use_selenium: If True, use Selenium for navigation. Otherwise, use requests.
            wait_for_js: Seconds to wait for JavaScript execution (Selenium only).
            fallback_to_requests: If True, fallback to requests if Selenium fails.

        Returns:
            Dictionary with page data.
        """
        if use_selenium:
            result = self._navigate_selenium(url, wait_for_js)
            # If Selenium fails and fallback is enabled, try with requests
            if not result.get('success') and fallback_to_requests:
                error = result.get('error', '')
                # Check if it's an empty response or connection error
                if (
                    'ERR_EMPTY_RESPONSE' in error
                    or 'empty' in error.lower()
                    or result.get('content', '') == ''
                ):
                    self.logger.info(
                        f'Selenium failed with empty response, falling back to requests for {url}'
                    )
                    requests_result = self._navigate_requests(url)
                    if requests_result.get('success'):
                        return requests_result
            return result
        else:
            return self._navigate_requests(url)

    def _navigate_requests(self, url: str) -> Dict[str, Any]:
        """Navigate using requests."""
        try:
            self.logger.info(f'Navigating to: {url} using requests')
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            result = {
                'url': response.url,
                'status_code': response.status_code,
                'content': response.text,
                'success': response.ok,
            }
            if result['success']:
                try:
                    result['soup'] = BeautifulSoup(response.text, 'html.parser')
                    self.logger.info(
                        f'Successfully loaded page: {response.url} ({len(response.text)} chars)'
                    )
                except Exception as e:
                    self.logger.warning(f'Failed to parse HTML: {e}')
                    result['soup'] = None
            else:
                error_msg = f'Failed to load page: HTTP {response.status_code}'
                self.logger.warning(error_msg)
                result['soup'] = None
                result['error'] = error_msg
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f'Error navigating to {url} with requests: {e}')
            return {
                'url': url,
                'status_code': 0,
                'content': '',
                'soup': None,
                'success': False,
                'error': str(e),
            }

    def _navigate_selenium(self, url: str, wait_for_js: float = 2.0) -> Dict[str, Any]:
        """Navigate using Selenium."""
        try:
            self._initialize_driver()
            self.logger.info(f'Navigating to: {url} using Selenium')

            try:
                self.driver.get(url)
            except TimeoutException:
                # Even if page load times out, try to get what we have
                self.logger.warning(
                    f'Page load timeout for {url}, attempting to extract partial content'
                )
            except WebDriverException as e:
                error_str = str(e)
                # Check for empty response errors
                if 'ERR_EMPTY_RESPONSE' in error_str or 'empty' in error_str.lower():
                    error_msg = f'Empty response from server: {error_str}'
                    self.logger.warning(
                        f'Failed to load page with Selenium: {error_msg}'
                    )
                    return {
                        'url': url,
                        'status_code': 502,
                        'content': '',
                        'success': False,
                        'error': error_msg,
                        'soup': None,
                    }
                raise

            # Wait for page to be interactive (DOM ready, not necessarily all resources)
            try:
                WebDriverWait(self.driver, min(wait_for_js, 5)).until(
                    lambda d: d.execute_script('return document.readyState')
                    in ['interactive', 'complete']
                )
            except TimeoutException:
                # If readyState doesn't become interactive, continue anyway
                self.logger.debug(
                    'Page readyState check timed out, continuing with current state'
                )

            content = self.driver.page_source
            final_url = self.driver.current_url

            # Check for empty or error responses
            if not content or len(content.strip()) < 100:
                # Check if page contains error messages
                page_text = content.lower() if content else ''
                error_indicators = [
                    'err_empty_response',
                    '페이지가 작동하지 않습니다',
                    'page not working',
                    'no data was sent',
                    '전송한 데이터가 없습니다',
                ]
                if any(indicator in page_text for indicator in error_indicators):
                    error_msg = 'Page returned empty response or error message'
                    self.logger.warning(f'Empty or error response detected for {url}')
                    return {
                        'url': final_url,
                        'status_code': 502,
                        'content': content,
                        'success': False,
                        'error': error_msg,
                        'soup': None,
                    }

            result = {
                'url': final_url,
                'status_code': 200,  # Selenium doesn't provide status codes directly
                'content': content,
                'success': True,
            }
            try:
                result['soup'] = BeautifulSoup(content, 'html.parser')
                self.logger.info(
                    f'Successfully loaded page: {final_url} ({len(content)} chars)'
                )
            except Exception as e:
                self.logger.warning(f'Failed to parse HTML with Selenium: {e}')
                result['soup'] = None
            return result
        except TimeoutException as e:
            error_msg = f'Page load timeout after {self.timeout}s: {str(e)}'
            self.logger.warning(f'Failed to load page with Selenium: {error_msg}')
            return {
                'url': url,
                'status_code': 504,
                'content': '',
                'success': False,
                'error': error_msg,
                'soup': None,
            }
        except WebDriverException as e:
            error_str = str(e)
            error_msg = f'WebDriver error: {error_str}'
            self.logger.warning(f'Failed to load page with Selenium: {error_msg}')
            # Check for empty response in the error
            if 'ERR_EMPTY_RESPONSE' in error_str or 'empty' in error_str.lower():
                return {
                    'url': url,
                    'status_code': 502,
                    'content': '',
                    'success': False,
                    'error': error_msg,
                    'soup': None,
                }
            return {
                'url': url,
                'status_code': 500,
                'content': '',
                'success': False,
                'error': error_msg,
                'soup': None,
            }
        except Exception as e:
            error_msg = f'Unexpected error with Selenium: {str(e)}'
            self.logger.error(f'Selenium navigation failed: {error_msg}', exc_info=True)
            return {
                'url': url,
                'status_code': 500,
                'content': '',
                'success': False,
                'error': error_msg,
                'soup': None,
            }

    def close(self):
        """Close the browser and clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info('Selenium WebDriver closed')
            except Exception as e:
                self.logger.warning(f'Error closing WebDriver: {e}')
            finally:
                self.driver = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def click_element(
        self,
        selector: Optional[str] = None,
        link_text: Optional[str] = None,
        wait_for_js: float = 2.0,
    ) -> Dict[str, Any]:
        """Click an element on the current page using Selenium."""
        self._initialize_driver()
        try:
            if selector:
                element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()
                self.logger.info(f'Clicked element with selector: {selector}')
            elif link_text:
                element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, link_text))
                )
                element.click()
                self.logger.info(f'Clicked link with text: {link_text}')
            else:
                return {
                    'success': False,
                    'error': 'Either selector or link_text must be provided',
                }

            WebDriverWait(self.driver, wait_for_js).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            return self._navigate_selenium(self.driver.current_url, wait_for_js=0)
        except (TimeoutException, NoSuchElementException) as e:
            return {
                'success': False,
                'error': f'Element not found or not clickable: {e}',
            }
        except Exception as e:
            self.logger.error(f'Click failed: {e}', exc_info=True)
            return {'success': False, 'error': str(e)}

    def find_link(
        self,
        page_data: Dict[str, Any],
        link_text: Optional[str] = None,
        url_pattern: Optional[str] = None,
        partial_match: bool = True,
    ) -> Optional[str]:
        if not page_data.get('soup'):
            return None
        soup = page_data['soup']
        base_url = page_data['url']
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            if href.lower().startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue
            if link_text:
                if (partial_match and link_text.lower() in text.lower()) or (
                    not partial_match and text.strip() == link_text.strip()
                ):
                    return urljoin(base_url, href)
            if url_pattern and re.search(url_pattern, href, re.IGNORECASE):
                return urljoin(base_url, href)
        self.logger.warning(
            f'Link not found: text="{link_text}", pattern="{url_pattern}"'
        )
        return None

    def extract_text(
        self, page_data: Dict[str, Any], selector: Optional[str] = None
    ) -> str:
        if not page_data.get('soup'):
            return page_data.get('content', '')
        soup = page_data['soup']
        if selector:
            elements = soup.select(selector)
            return '\n'.join([elem.get_text(strip=True) for elem in elements])
        for script in soup(['script', 'style', 'noscript']):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)

    def extract_markdown(
        self, page_data: Dict[str, Any], selector: Optional[str] = None
    ) -> str:
        """
        Extract content from HTML and convert to markdown format.
        This preserves structure (headings, links, lists) better than plain text.

        Args:
            page_data: Dictionary containing 'soup' (BeautifulSoup) or 'content' (HTML string).
            selector: Optional CSS selector to extract specific elements.

        Returns:
            Markdown-formatted string.
        """
        if not page_data.get('soup'):
            html_content = page_data.get('content', '')
            if not html_content:
                return ''
        else:
            soup = page_data['soup']
            if selector:
                elements = soup.select(selector)
                if not elements:
                    return ''
                html_content = '\n'.join(str(elem) for elem in elements)
            else:
                # Remove script and style tags before conversion
                soup_copy = BeautifulSoup(str(soup), 'html.parser')
                for script in soup_copy(['script', 'style', 'noscript']):
                    script.decompose()
                html_content = str(soup_copy)

        # Configure html2text converter
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True  # Use unicode characters
        h.skip_internal_links = False
        h.inline_links = True  # Put links inline

        # Convert HTML to markdown
        markdown = h.handle(html_content)
        return markdown.strip()

    def extract_html(
        self,
        page_data: Dict[str, Any],
        selector: Optional[str] = None,
        prettify: bool = True,
    ) -> str:
        if not page_data.get('soup'):
            return page_data.get('content', '')
        soup = page_data['soup']
        if selector:
            elements = soup.select(selector)
            if not elements:
                return ''
            html_content = '\n'.join(str(elem) for elem in elements)
        else:
            html_content = str(soup)
        if prettify:
            temp_soup = BeautifulSoup(html_content, 'html.parser')
            return temp_soup.prettify()
        return html_content

    def find_table(
        self, page_data: Dict[str, Any], headers: Optional[List[str]] = None
    ) -> Optional[List[Dict[str, str]]]:
        if not page_data.get('soup'):
            return None
        soup = page_data['soup']
        tables = soup.find_all('table')
        for table in tables:
            table_headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if headers and not all(
                h.lower() in ' '.join(table_headers).lower() for h in headers
            ):
                continue
            rows = []
            for tr in table.find_all('tr')[1:]:
                cells = tr.find_all(['td', 'th'])
                row_data = {
                    (
                        table_headers[i] if i < len(table_headers) else f'Column_{i}'
                    ): cell.get_text(strip=True)
                    for i, cell in enumerate(cells)
                }
                if row_data:
                    rows.append(row_data)
            if rows:
                return rows
        return None

    def extract_numeric_data(
        self,
        page_data: Dict[str, Any],
        context_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts numeric data. Note: This method uses regex and may be brittle.
        Prefer extract_with_llm for more robust extraction.
        """
        full_text = self.extract_text(page_data)
        number_patterns = [
            r'(\d{1,7})\s+(?:articles?|items?|papers?|publications?|records?|entries?|count|total|number)',
            r'(?:total|count|number|articles?|items?):\s*(\d{1,7})',
        ]
        if context_keywords:
            for keyword in context_keywords:
                keyword_lower = keyword.lower()
                number_patterns.extend(
                    [
                        rf'(\d{{1,7}})\s+{re.escape(keyword_lower)}s?\b',
                        rf'{re.escape(keyword_lower)}s?[:\s]+(\d{{1,7}})',
                    ]
                )

        numeric_matches = []
        for pattern in number_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                try:
                    value = int(match.group(1))
                    context = full_text[
                        max(0, match.start() - 50) : min(
                            len(full_text), match.end() + 50
                        )
                    ].strip()
                    numeric_matches.append({'value': value, 'context': context})
                except (ValueError, IndexError):
                    continue

        # Return in the expected format with all keys
        return {
            'numeric_values': numeric_matches,
            'counts': numeric_matches,  # Same as numeric_values for compatibility
            'table_numbers': [],  # Can be populated from find_table if needed
            'list_numbers': [],  # Can be populated from list extraction if needed
        }

    def extract_with_llm(
        self,
        page_data: Dict[str, Any],
        extraction_query: str,
        llm_service: 'LLMService',
        context_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        page_text = self.extract_markdown(page_data)
        max_text_length = 45000
        if len(page_text) > max_text_length:
            page_text = page_text[:max_text_length] + '\n[Text content truncated...]'

        # Add context keywords to the query if provided
        query_with_context = extraction_query
        if context_keywords:
            query_with_context += f' (Focus on: {", ".join(context_keywords)})'

        system_prompt = """You are an expert at extracting specific information from web page content.
        Return a JSON object with:
        - extracted_value: The actual value found (number, text, or null if not found)
        - confidence: A value between 0 and 1
        - reasoning: A brief explanation.
        - context: The surrounding text where the value was found.
        Return your response as valid JSON only."""

        user_prompt = f"""Extract the following information from the web page content:
        Query: {query_with_context}
        Web page content:
        ---
        {page_text}
        ---
        Return the result as a JSON object."""

        try:
            response = llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                response_format={'type': 'json_object'},
            )
            return json.loads(response)
        except Exception as e:
            self.logger.error(f'LLM extraction failed: {e}')
            return {'extracted_value': None, 'confidence': 0.0, 'reasoning': str(e)}

    def extract_links_list(self, page_data: Dict[str, Any]) -> List[str]:
        """Extract all links from the page."""
        if not page_data.get('soup'):
            return []
        soup = page_data['soup']
        base_url = page_data['url']
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if href.lower().startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue
            full_url = urljoin(base_url, href)
            links.append(full_url)
        return links

    def search_text(
        self, page_data: Dict[str, Any], search_terms: List[str]
    ) -> Dict[str, Any]:
        """Search for text terms in the page content."""
        full_text = self.extract_text(page_data)
        found = []
        not_found = []
        contexts = {}

        for term in search_terms:
            term_lower = term.lower()
            if term_lower in full_text.lower():
                found.append(term)
                # Find context around the term
                term_index = full_text.lower().find(term_lower)
                if term_index != -1:
                    context_start = max(0, term_index - 100)
                    context_end = min(len(full_text), term_index + len(term) + 100)
                    context = full_text[context_start:context_end].strip()
                    if term not in contexts:
                        contexts[term] = []
                    contexts[term].append(context)
            else:
                not_found.append(term)

        return {'found': found, 'not_found': not_found, 'contexts': contexts}

    def take_screenshot(self, as_base64: bool = False) -> Optional[bytes]:
        """Take a screenshot of the current page.

        Args:
            as_base64: If True, return base64-encoded string. If False, return PNG bytes.

        Returns:
            Screenshot as bytes (PNG) or base64-encoded bytes, or None on error.
        """
        if not self.driver:
            self.logger.warning('Cannot take screenshot: WebDriver not initialized')
            return None
        try:
            screenshot = self.driver.get_screenshot_as_png()
            if as_base64:
                import base64

                return base64.b64encode(screenshot)
            return screenshot
        except Exception as e:
            self.logger.error(f'Failed to take screenshot: {e}')
            return None

    def detect_expandable_elements(
        self, page_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect expandable/collapsible elements (buttons, toggles) on the page.

        Args:
            page_data: Dictionary containing 'soup' (BeautifulSoup) or 'content' (HTML string).

        Returns:
            List of dictionaries with element information (text, selector, type, etc.).
        """
        expandable_elements = []
        if not page_data.get('soup'):
            return expandable_elements

        soup = page_data['soup']

        # Common patterns for expandable elements:
        # 1. Buttons with aria-expanded attribute
        # 2. Buttons with "toggle", "expand", "show", "hide" in text/class/id
        # 3. Elements with collapse/accordion classes
        # 4. Wikipedia-style toggle buttons

        # Find buttons with aria-expanded
        for button in soup.find_all(
            ['button', 'a', 'span', 'div'],
            attrs={'aria-expanded': lambda x: x is not None},
        ):
            text = button.get_text(strip=True)
            element_id = button.get('id', '')
            classes = ' '.join(button.get('class', []))
            aria_label = button.get('aria-label', '')
            aria_expanded = button.get('aria-expanded', '')

            if text or aria_label or element_id:
                # Try to find a CSS selector
                selector = None
                if element_id:
                    selector = f'#{element_id}'
                elif classes:
                    # Use first class as selector
                    first_class = classes.split()[0] if classes else None
                    if first_class:
                        selector = f'.{first_class}'

                expandable_elements.append(
                    {
                        'type': 'aria-expanded',
                        'text': text or aria_label or element_id,
                        'selector': selector,
                        'id': element_id,
                        'classes': classes,
                        'aria-label': aria_label,
                        'aria-expanded': aria_expanded,
                        'tag': button.name,
                    }
                )

        # Find buttons/links with toggle/expand/collapse keywords
        toggle_keywords = [
            'toggle',
            'expand',
            'collapse',
            'show',
            'hide',
            'more',
            'less',
            'view',
        ]

        # Track elements we've already added to avoid duplicates
        added_element_ids = set()
        added_element_texts = set()

        for keyword in toggle_keywords:
            # Search in button text, aria-label, id, class
            for element in soup.find_all(['button', 'a', 'span', 'div']):
                text = element.get_text(strip=True).lower()
                element_id = element.get('id', '').lower()
                classes = ' '.join(element.get('class', [])).lower()
                aria_label = element.get('aria-label', '').lower()

                if (
                    keyword in text
                    or keyword in element_id
                    or keyword in classes
                    or keyword in aria_label
                ):
                    # Check if we already added this element
                    elem_id = element.get('id', '')
                    text_display = element.get_text(strip=True)
                    text_key = text_display.lower()[:50]  # Use first 50 chars as key

                    # Skip if we've already added this element
                    if elem_id and elem_id in added_element_ids:
                        continue
                    if text_key and text_key in added_element_texts:
                        continue

                    selector = None
                    if elem_id:
                        selector = f'#{elem_id}'
                    elif element.get('class'):
                        first_class = element.get('class')[0]
                        if first_class:
                            selector = f'.{first_class}'

                    expandable_elements.append(
                        {
                            'type': 'keyword-match',
                            'text': text_display,
                            'selector': selector,
                            'id': elem_id,
                            'classes': ' '.join(element.get('class', [])),
                            'aria-label': element.get('aria-label', ''),
                            'keyword': keyword,
                            'tag': element.name,
                        }
                    )

                    # Track added elements
                    if elem_id:
                        added_element_ids.add(elem_id)
                    if text_key:
                        added_element_texts.add(text_key)

        # Limit to avoid too many elements
        return expandable_elements[:30]

    def _dismiss_popups_and_banners(self) -> bool:
        """
        Attempt to dismiss cookie consent banners and other popup dialogs.

        Returns:
            True if any popup was dismissed, False otherwise.
        """
        if not self.driver:
            return False

        dismissed = False

        # Common selectors for cookie consent banners and popups
        popup_selectors = [
            # Cookie consent banners
            'dialog[data-cc-banner]',
            'dialog.cc-banner',
            '[data-cc-banner]',
            '.cc-banner',
            # Generic popup/dialog patterns
            'dialog[open]',
            '.modal[style*="display: block"]',
            '.popup[style*="display: block"]',
            # Common accept/close buttons
            'button[data-cc-action="accept"]',
            'button[data-cc-action="dismiss"]',
            'button.cc-banner__button--accept',
            'button.cc-banner__button--dismiss',
            '[aria-label*="accept" i]',
            '[aria-label*="dismiss" i]',
            '[aria-label*="close" i]',
        ]

        # XPath selectors for text-based matching (used separately)
        xpath_selectors = [
            '//button[contains(text(), "Accept")]',
            '//button[contains(text(), "Dismiss")]',
            '//button[contains(text(), "Close")]',
            '//button[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "accept")]',
        ]

        try:
            # Try to find and close cookie consent banners
            for selector in popup_selectors[:4]:  # Focus on dialog elements first
                try:
                    popup = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if popup.is_displayed():
                        # Try to find accept/dismiss button within the popup
                        accept_buttons = popup.find_elements(
                            By.CSS_SELECTOR,
                            'button[data-cc-action="accept"], '
                            'button[data-cc-action="dismiss"], '
                            'button.cc-banner__button--accept, '
                            'button.cc-banner__button--dismiss, '
                            '[aria-label*="accept" i], '
                            '[aria-label*="dismiss" i]',
                        )

                        if accept_buttons:
                            # Click the first accept/dismiss button
                            self.driver.execute_script(
                                'arguments[0].click();', accept_buttons[0]
                            )
                            dismissed = True
                            self.logger.info('Dismissed cookie consent banner')
                            time.sleep(0.5)  # Wait for popup to close
                            break
                        else:
                            # Try to close the dialog directly via JavaScript
                            self.driver.execute_script(
                                'arguments[0].removeAttribute("open"); '
                                'arguments[0].style.display = "none";',
                                popup,
                            )
                            dismissed = True
                            self.logger.info('Closed popup dialog via JavaScript')
                            time.sleep(0.5)
                            break
                except (NoSuchElementException, ElementNotInteractableException):
                    continue

            # Also try to find standalone accept buttons using CSS selectors
            if not dismissed:
                for selector in popup_selectors[4:]:
                    try:
                        button = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if button.is_displayed():
                            self.driver.execute_script('arguments[0].click();', button)
                            dismissed = True
                            self.logger.info('Clicked popup dismiss button')
                            time.sleep(0.5)
                            break
                    except (NoSuchElementException, ElementNotInteractableException):
                        continue

            # Try XPath selectors for text-based button matching
            if not dismissed:
                for xpath in xpath_selectors:
                    try:
                        button = self.driver.find_element(By.XPATH, xpath)
                        if button.is_displayed():
                            self.driver.execute_script('arguments[0].click();', button)
                            dismissed = True
                            self.logger.info('Clicked popup dismiss button via XPath')
                            time.sleep(0.5)
                            break
                    except (NoSuchElementException, ElementNotInteractableException):
                        continue

        except Exception as e:
            self.logger.debug(f'Error dismissing popups: {e}')

        return dismissed

    def toggle_expandable_element(
        self,
        element_info: Dict[str, Any],
        wait_for_js: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Toggle an expandable element using Selenium.

        Args:
            element_info: Dictionary with element information (from detect_expandable_elements).
            wait_for_js: Seconds to wait after clicking.

        Returns:
            Dictionary with success status and updated page data.
        """
        self._initialize_driver()
        try:
            element = None

            # Try to find element by selector first
            if element_info.get('selector'):
                try:
                    element = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, element_info['selector'])
                        )
                    )
                except (TimeoutException, NoSuchElementException):
                    pass

            # Try by ID if selector didn't work
            if not element and element_info.get('id'):
                try:
                    element = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.ID, element_info['id']))
                    )
                except (TimeoutException, NoSuchElementException):
                    pass

            # Try by link text as fallback
            if not element and element_info.get('text'):
                try:
                    element = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located(
                            (By.PARTIAL_LINK_TEXT, element_info['text'][:50])
                        )
                    )
                except (TimeoutException, NoSuchElementException):
                    pass

            if not element:
                return {
                    'success': False,
                    'error': f'Element not found: {element_info.get("text", "unknown")}',
                }

            # Dismiss any popups/banners that might intercept clicks
            self._dismiss_popups_and_banners()

            # Scroll element into view
            self.driver.execute_script(
                'arguments[0].scrollIntoView({behavior: "smooth", block: "center"});',
                element,
            )

            # Wait a bit for scroll
            time.sleep(0.5)

            # Try to click the element with error handling
            try:
                element.click()
            except (
                ElementClickInterceptedException,
                ElementNotInteractableException,
            ) as e:
                # If click is intercepted, try dismissing popups again
                self.logger.warning(
                    f'Click intercepted, attempting to dismiss popups: {e}'
                )
                self._dismiss_popups_and_banners()
                time.sleep(0.5)

                # Try JavaScript click as fallback
                try:
                    self.driver.execute_script('arguments[0].click();', element)
                    self.logger.info('Used JavaScript click as fallback')
                except Exception:
                    # If JavaScript click also fails, try scrolling more and retry
                    self.driver.execute_script(
                        'window.scrollTo(0, arguments[0].offsetTop - 200);', element
                    )
                    time.sleep(0.5)
                    try:
                        self.driver.execute_script('arguments[0].click();', element)
                        self.logger.info(
                            'Used JavaScript click after scroll adjustment'
                        )
                    except Exception as final_error:
                        raise ElementClickInterceptedException(
                            f'Failed to click element after multiple attempts: {final_error}'
                        )
            else:
                self.logger.info(
                    f'Toggled expandable element: {element_info.get("text", "unknown")}'
                )

            # Wait for page to update
            WebDriverWait(self.driver, wait_for_js).until(
                lambda d: d.execute_script('return document.readyState')
                in ['interactive', 'complete']
            )

            # Get updated page content
            return self._navigate_selenium(self.driver.current_url, wait_for_js=0)

        except (TimeoutException, NoSuchElementException) as e:
            return {
                'success': False,
                'error': f'Element not found or not clickable: {e}',
            }
        except (ElementClickInterceptedException, ElementNotInteractableException) as e:
            return {
                'success': False,
                'error': f'Element click intercepted or not interactable: {e}',
            }
        except Exception as e:
            self.logger.error(f'Toggle failed: {e}', exc_info=True)
            return {'success': False, 'error': str(e)}
