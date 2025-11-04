"""Selenium-based Browser Navigation for JavaScript-rendered pages."""

import logging
import time
from typing import Any, Dict, Optional

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


class SeleniumBrowserNavigator:
    """
    Selenium-based browser navigation for JavaScript-rendered pages.

    Provides functionality for:
    - Navigating to JavaScript-heavy websites
    - Handling dynamic content
    - Extracting data after JS execution
    - Clicking links and interacting with elements
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        headless: bool = True,
        timeout: int = 30,
    ):
        """
        Initialize SeleniumBrowserNavigator.

        Args:
            logger: Optional logger instance.
            headless: Whether to run browser in headless mode (default: True).
            timeout: Default timeout for page loads and waits in seconds.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self._initialize_driver()

    def _initialize_driver(self):
        """Initialize the Selenium WebDriver with Chrome."""
        try:
            chrome_options = Options()

            if self.headless:
                chrome_options.add_argument('--headless=new')

            # Additional options for stability
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(
                '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            # Disable logging
            chrome_options.add_argument('--log-level=3')
            chrome_options.add_experimental_option(
                'excludeSwitches', ['enable-logging']
            )

            # Use webdriver-manager to automatically handle driver installation
            service = Service(ChromeDriverManager().install())
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

    def navigate(self, url: str, wait_for_js: float = 2.0) -> Dict[str, Any]:
        """
        Navigate to a URL and wait for JavaScript to execute.

        Args:
            url: URL to navigate to.
            wait_for_js: Seconds to wait for JavaScript execution (default: 2.0).

        Returns:
            Dictionary with:
            - url: The final URL (may differ if redirected)
            - content: Rendered HTML after JS execution
            - soup: BeautifulSoup parsed object
            - success: Boolean indicating if navigation succeeded
            - status_code: HTTP status (200 if successful, error code otherwise)
        """
        try:
            self.logger.info(f'Navigating to: {url}')

            # Navigate to URL
            self.driver.get(url)

            # Wait for JavaScript to execute
            time.sleep(wait_for_js)

            # Get the rendered HTML
            content = self.driver.page_source
            final_url = self.driver.current_url

            result = {
                'url': final_url,
                'status_code': 200,  # Selenium doesn't provide status codes directly
                'content': content,
                'success': True,
            }

            # Parse with BeautifulSoup
            try:
                result['soup'] = BeautifulSoup(content, 'html.parser')
                self.logger.info(
                    f'Successfully loaded page: {final_url} ({len(content)} chars)'
                )
            except Exception as e:
                self.logger.warning(f'Failed to parse HTML: {e}')
                result['soup'] = None

            return result

        except TimeoutException:
            error_msg = f'Page load timeout after {self.timeout}s'
            self.logger.warning(f'Failed to load page: {error_msg}')
            return {
                'url': url,
                'status_code': 504,  # Gateway timeout
                'content': '',
                'success': False,
                'error': error_msg,
                'soup': None,
            }

        except WebDriverException as e:
            error_msg = f'WebDriver error: {str(e)}'
            self.logger.warning(f'Failed to load page: {error_msg}')
            return {
                'url': url,
                'status_code': 500,
                'content': '',
                'success': False,
                'error': error_msg,
                'soup': None,
            }

        except Exception as e:
            error_msg = f'Unexpected error: {str(e)}'
            self.logger.error(f'Navigation failed: {error_msg}', exc_info=True)
            return {
                'url': url,
                'status_code': 500,
                'content': '',
                'success': False,
                'error': error_msg,
                'soup': None,
            }

    def click_element(
        self,
        selector: Optional[str] = None,
        link_text: Optional[str] = None,
        wait_for_js: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Click an element on the current page.

        Args:
            selector: CSS selector for the element to click.
            link_text: Alternative - text content of link to click.
            wait_for_js: Seconds to wait after clicking.

        Returns:
            Dictionary with navigation result after clicking.
        """
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

            # Wait for page to load/update
            time.sleep(wait_for_js)

            # Return updated page data
            return self.navigate(self.driver.current_url, wait_for_js=0)

        except TimeoutException:
            return {
                'success': False,
                'error': 'Element not found or not clickable (timeout)',
            }
        except NoSuchElementException:
            return {
                'success': False,
                'error': 'Element not found',
            }
        except Exception as e:
            self.logger.error(f'Click failed: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
            }

    def fill_form(
        self,
        form_data: Dict[str, str],
        submit_selector: Optional[str] = None,
        wait_for_js: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Fill a form with data and optionally submit it.

        Args:
            form_data: Dictionary mapping CSS selectors to values.
            submit_selector: Optional CSS selector for submit button.
            wait_for_js: Seconds to wait after submission.

        Returns:
            Dictionary with result after form submission.
        """
        try:
            # Fill each field
            for selector, value in form_data.items():
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                element.clear()
                element.send_keys(value)
                self.logger.debug(f'Filled field {selector} with value: {value}')

            # Submit if requested
            if submit_selector:
                submit_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, submit_selector))
                )
                submit_button.click()
                self.logger.info('Form submitted')
                time.sleep(wait_for_js)

                return self.navigate(self.driver.current_url, wait_for_js=0)

            return {'success': True}

        except Exception as e:
            self.logger.error(f'Form filling failed: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
            }

    def extract_text(self, page_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract visible text from current page or page_data.

        Args:
            page_data: Optional page data dict with 'soup' key.

        Returns:
            Extracted text content.
        """
        if page_data and 'soup' in page_data and page_data['soup']:
            soup = page_data['soup']
        else:
            # Get current page
            content = self.driver.page_source
            soup = BeautifulSoup(content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return text

    def get_current_url(self) -> str:
        """Get the current URL."""
        return self.driver.current_url if self.driver else ''

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
        """Ensure browser is closed on object destruction."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes browser."""
        self.close()
