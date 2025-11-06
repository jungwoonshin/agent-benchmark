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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
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
            chrome_options.add_experimental_option(
                'excludeSwitches', ['enable-logging']
            )
            chromedriver_path = self._resolve_chromedriver_path()
            self.logger.debug(f'Using chromedriver at: {chromedriver_path}')
            service = Service(chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
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
        self, url: str, use_selenium: bool = False, wait_for_js: float = 2.0
    ) -> Dict[str, Any]:
        """
        Navigate to a URL and return page content.

        Args:
            url: URL to navigate to.
            use_selenium: If True, use Selenium for navigation. Otherwise, use requests.
            wait_for_js: Seconds to wait for JavaScript execution (Selenium only).

        Returns:
            Dictionary with page data.
        """
        if use_selenium:
            return self._navigate_selenium(url, wait_for_js)
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
            self.driver.get(url)

            # Replace time.sleep with WebDriverWait
            WebDriverWait(self.driver, wait_for_js).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            content = self.driver.page_source
            final_url = self.driver.current_url
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
        except TimeoutException:
            error_msg = f'Page load timeout after {self.timeout}s'
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
            error_msg = f'WebDriver error: {str(e)}'
            self.logger.warning(f'Failed to load page with Selenium: {error_msg}')
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
        page_text = self.extract_text(page_data)
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
