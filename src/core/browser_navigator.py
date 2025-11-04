"""Browser Navigation Module for web page interaction and data extraction."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from .llm_service import LLMService


class BrowserNavigator:
    """
    Browser navigation and web page interaction utilities.

    Provides functionality for:
    - Navigating to web pages
    - Extracting structured data from HTML
    - Finding and clicking links/buttons
    - Extracting text content from pages
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize BrowserNavigator.

        Args:
            logger: Optional logger instance. If not provided, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update(
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        )

    def navigate(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Navigate to a URL and return page content.

        Args:
            url: URL to navigate to.
            timeout: Request timeout in seconds.

        Returns:
            Dictionary with:
            - url: The URL that was loaded (may differ if redirected)
            - content: Raw HTML content
            - soup: BeautifulSoup parsed object
            - status_code: HTTP status code
            - success: Boolean indicating if request succeeded
        """
        try:
            self.logger.info(f'Navigating to: {url}')
            response = self.session.get(url, timeout=timeout, allow_redirects=True)

            result = {
                'url': response.url,  # Final URL after redirects
                'status_code': response.status_code,
                'content': response.text,
                'success': response.status_code == 200,
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

                # Add diagnostics for common error codes
                if response.status_code == 400:
                    result['diagnostics'] = {
                        'status_code': 400,
                        'error_type': 'bad_request',
                        'suggestions': [
                            'URL may be malformed or contain invalid parameters',
                            'Server may be rejecting the request format',
                            'Check if URL encoding is correct',
                            'Try simplifying the URL or removing query parameters',
                            'For arXiv: Consider using search tool instead of direct navigation',
                        ],
                    }
                    # Try to get more info from response
                    if hasattr(response, 'text') and response.text:
                        result['diagnostics']['response_preview'] = response.text[:500]
                elif response.status_code == 403:
                    result['diagnostics'] = {
                        'status_code': 403,
                        'error_type': 'forbidden',
                        'suggestions': [
                            'Access may require authentication',
                            'Server may be blocking requests based on User-Agent or IP',
                            'Try using different headers or authentication',
                        ],
                    }
                elif response.status_code == 404:
                    result['diagnostics'] = {
                        'status_code': 404,
                        'error_type': 'not_found',
                        'suggestions': [
                            'URL may be incorrect or resource no longer exists',
                            'Check if URL has moved or been renamed',
                            'Try searching for the resource instead',
                        ],
                    }

            return result

        except requests.exceptions.RequestException as e:
            self.logger.error(f'Error navigating to {url}: {e}')
            return {
                'url': url,
                'status_code': 0,
                'content': '',
                'soup': None,
                'success': False,
                'error': str(e),
            }

    def find_link(
        self,
        page_data: Dict[str, Any],
        link_text: Optional[str] = None,
        url_pattern: Optional[str] = None,
        partial_match: bool = True,
    ) -> Optional[str]:
        """
        Find a link on the page by text or URL pattern.

        Args:
            page_data: Page data dictionary from navigate().
            link_text: Text content of the link to find.
            url_pattern: URL pattern to match (regex).
            partial_match: If True, link_text matches partially; if False, exact match.

        Returns:
            URL of the found link, or None if not found.
        """
        if not page_data.get('soup'):
            return None

        soup = page_data['soup']
        base_url = page_data['url']

        # Find all links
        links = soup.find_all('a', href=True)

        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            # Skip JavaScript handlers and invalid URLs
            if href.lower().startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue

            # Check text match
            if link_text:
                if partial_match:
                    if link_text.lower() in text.lower():
                        full_url = urljoin(base_url, href)
                        # Validate URL before returning
                        if self._is_valid_url(full_url):
                            self.logger.info(
                                f'Found link by text "{link_text}": {full_url}'
                            )
                            return full_url
                        else:
                            self.logger.warning(
                                f'Found link but URL is invalid: {full_url}'
                            )
                else:
                    if text.strip() == link_text.strip():
                        full_url = urljoin(base_url, href)
                        # Validate URL before returning
                        if self._is_valid_url(full_url):
                            self.logger.info(
                                f'Found link by exact text "{link_text}": {full_url}'
                            )
                            return full_url
                        else:
                            self.logger.warning(
                                f'Found link but URL is invalid: {full_url}'
                            )

            # Check URL pattern match
            if url_pattern:
                if re.search(url_pattern, href, re.IGNORECASE):
                    full_url = urljoin(base_url, href)
                    # Validate URL before returning
                    if self._is_valid_url(full_url):
                        self.logger.info(
                            f'Found link by URL pattern "{url_pattern}": {full_url}'
                        )
                        return full_url
                    else:
                        self.logger.warning(
                            f'Found link but URL is invalid: {full_url}'
                        )

        self.logger.warning(
            f'Link not found: text="{link_text}", pattern="{url_pattern}"'
        )
        return None

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate if a URL is properly formatted and navigable.

        Args:
            url: URL to validate.

        Returns:
            True if URL is valid, False otherwise.
        """
        if not url or not url.strip():
            return False

        # Check for basic URL format
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            # Must have scheme (http/https) or be a relative path
            if parsed.scheme and parsed.scheme not in ('http', 'https'):
                return False
            # If no scheme, check if it's a valid relative path
            if not parsed.scheme:
                # Relative paths starting with / are usually okay
                if not url.startswith('/'):
                    return False
            return True
        except Exception:
            return False

    def click_link(
        self,
        page_data: Dict[str, Any],
        link_text: Optional[str] = None,
        url_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find and navigate to a link on the current page.

        Args:
            page_data: Page data dictionary from current page.
            link_text: Text content of the link to click.
            url_pattern: URL pattern to match.

        Returns:
            Page data dictionary from the new page.
        """
        # Check if page was loaded successfully first
        if not page_data.get('success'):
            error_msg = (
                f'Cannot click link: current page failed to load. '
                f'Error: {page_data.get("error", "Unknown error")}, '
                f'Status: {page_data.get("status_code", 0)}'
            )
            self.logger.error(error_msg)
            return {
                'url': page_data.get('url', ''),
                'status_code': page_data.get('status_code', 0),
                'content': '',
                'soup': None,
                'success': False,
                'error': error_msg,
                'diagnostics': {
                    'page_load_failed': True,
                    'original_error': page_data.get('error'),
                    'status_code': page_data.get('status_code'),
                },
            }

        # Check if page is likely JavaScript-rendered (might have content but empty text)
        # This is important because links on JS-rendered pages may not be in the HTML
        content = page_data.get('content', '')
        soup = page_data.get('soup')
        likely_js_rendered = False
        if soup and content:
            # Check if page has content but likely needs JS to render links
            extracted_text = soup.get_text(strip=True) if soup else ''
            if len(content) > 1000 and len(extracted_text.strip()) < 100:
                likely_js_rendered = True
                current_url = page_data.get('url', '')
                if 'arxiv.org' in current_url.lower():
                    self.logger.warning(
                        '⚠️ ArXiv pages are JavaScript-rendered. Links may not be available in HTML. '
                        'Consider using search tool or API instead of browser navigation.'
                    )

        link_url = self.find_link(page_data, link_text, url_pattern)
        if link_url:
            # Validate URL one more time before navigation
            if not self._is_valid_url(link_url):
                error_msg = (
                    f'Found link but URL is invalid or malformed: {link_url}. '
                    f'This may indicate the page is JavaScript-rendered and links are not in static HTML.'
                )
                self.logger.error(error_msg)
                return {
                    'url': page_data.get('url', ''),
                    'status_code': 0,
                    'content': '',
                    'soup': None,
                    'success': False,
                    'error': error_msg,
                    'diagnostics': {
                        'link_found': True,
                        'target_url': link_url,
                        'url_invalid': True,
                        'likely_js_rendered': likely_js_rendered,
                        'suggestions': [
                            'Page may be JavaScript-rendered - links require JS to generate valid URLs',
                            'Try using search tool instead of browser navigation',
                            'Consider using API endpoints if available',
                        ],
                    },
                }

            nav_result = self.navigate(link_url)
            if not nav_result.get('success'):
                # Provide detailed error information
                error_msg = (
                    f'Link found but navigation failed: {nav_result.get("error", "Unknown error")}. '
                    f'Target URL: {link_url}, Status: {nav_result.get("status_code", 0)}'
                )
                self.logger.error(error_msg)
                nav_result['error'] = error_msg
                diagnostics = {
                    'link_found': True,
                    'target_url': link_url,
                    'navigation_failed': True,
                    'status_code': nav_result.get('status_code'),
                }

                # Add specific diagnostics for HTTP 400 errors
                if nav_result.get('status_code') == 400:
                    diagnostics['error_type'] = 'bad_request'
                    diagnostics['likely_causes'] = [
                        'URL may be malformed or constructed incorrectly',
                        'Server may be rejecting the request format',
                        'URL may contain invalid parameters',
                        'Page may be JavaScript-rendered - URL requires JS to be valid',
                    ]
                    diagnostics['suggestions'] = [
                        'Try using search tool instead of browser navigation',
                        'Check if URL encoding is correct',
                        'For JavaScript-rendered pages, consider alternative approaches',
                    ]
                    if likely_js_rendered:
                        diagnostics['js_rendered_detected'] = True
                        diagnostics['suggestions'].insert(
                            0,
                            '⚠️ Page detected as JavaScript-rendered - links may not work with static HTML parsing',
                        )

                # Include existing diagnostics if available
                if 'diagnostics' in nav_result:
                    diagnostics.update(nav_result['diagnostics'])

                nav_result['diagnostics'] = diagnostics
            return nav_result
        else:
            # Provide helpful diagnostics about why link wasn't found
            diagnostics = {
                'link_found': False,
                'searched_text': link_text,
                'searched_pattern': url_pattern,
            }

            # Try to find similar links or provide suggestions
            if page_data.get('soup'):
                soup = page_data['soup']
                all_links = soup.find_all('a', href=True)
                link_texts = [
                    link.get_text(strip=True) for link in all_links[:20]
                ]  # Sample first 20
                diagnostics['sample_link_texts'] = link_texts

                # Try to find partial matches
                if link_text:
                    similar_links = [
                        text
                        for text in link_texts
                        if link_text.lower() in text.lower()
                        or text.lower() in link_text.lower()
                    ]
                    if similar_links:
                        diagnostics['similar_links_found'] = similar_links[:5]

            error_msg = (
                f'Link not found on page. Searched for text="{link_text}", pattern="{url_pattern}". '
                f'Current URL: {page_data.get("url", "unknown")}'
            )
            if diagnostics.get('similar_links_found'):
                error_msg += f'\nSimilar links found: {", ".join(diagnostics["similar_links_found"])}'

            self.logger.error(error_msg)
            return {
                'url': page_data.get('url', ''),
                'status_code': 0,
                'content': '',
                'soup': None,
                'success': False,
                'error': error_msg,
                'diagnostics': diagnostics,
            }

    def extract_text(
        self, page_data: Dict[str, Any], selector: Optional[str] = None
    ) -> str:
        """
        Extract text content from a page.

        Args:
            page_data: Page data dictionary from navigate().
            selector: Optional CSS selector to extract specific element.

        Returns:
            Extracted text content.
        """
        if not page_data.get('soup'):
            # Fallback to raw content if soup not available
            content = page_data.get('content', '')
            if not content:
                self.logger.warning(
                    'No soup and no content available for text extraction. '
                    'Page may not have loaded properly.'
                )
            return content

        soup = page_data['soup']

        if selector:
            elements = soup.select(selector)
            if not elements:
                self.logger.warning(
                    f'No elements found for selector "{selector}". '
                    'Page structure may be different than expected.'
                )
            texts = [elem.get_text(strip=True) for elem in elements]
            result = '\n'.join(texts)
            if not result:
                self.logger.warning(
                    f'Selector "{selector}" matched elements but extracted text is empty. '
                    'Elements may contain only whitespace or be JavaScript-rendered.'
                )
            return result
        else:
            # Extract all text, removing script and style elements
            for script in soup(['script', 'style', 'noscript']):
                script.decompose()

            # Get text content
            text_content = soup.get_text(separator='\n', strip=True)

            # Check if text extraction returned meaningful content
            if not text_content or len(text_content.strip()) < 50:
                # Page might be JavaScript-rendered - check content length
                raw_content = page_data.get('content', '')
                content_length = len(raw_content) if raw_content else 0

                self.logger.warning(
                    f'Text extraction returned minimal/empty content ({len(text_content)} chars). '
                    f'Raw HTML length: {content_length} chars. '
                    'This may indicate a JavaScript-rendered page that requires browser execution, '
                    'or the page content is loaded dynamically.'
                )

                # Try to extract at least some basic info from meta tags or title
                title = soup.find('title')
                if title:
                    title_text = title.get_text(strip=True)
                    if title_text:
                        text_content = f'Page Title: {title_text}\n\n'

                # Check for meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    text_content += f'Description: {meta_desc.get("content")}\n'

                # If still empty, return a helpful message
                if not text_content or len(text_content.strip()) < 10:
                    text_content = (
                        f'⚠️ WARNING: Page loaded successfully but text extraction failed. '
                        f'This page may require JavaScript to render content. '
                        f'Raw HTML size: {content_length} characters. '
                        f'Try using a different extraction method or action.'
                    )

            return text_content

    def extract_html(
        self,
        page_data: Dict[str, Any],
        selector: Optional[str] = None,
        prettify: bool = True,
        remove_scripts: bool = True,
    ) -> str:
        """
        Extract HTML content from a page.

        Args:
            page_data: Page data dictionary from navigate().
            selector: Optional CSS selector to extract specific element(s).
            prettify: If True, format HTML with indentation. If False, return compact HTML.
            remove_scripts: If True, remove script and style tags before extraction.

        Returns:
            HTML content as string.
        """
        if not page_data.get('soup'):
            # If no soup available, return raw content
            return page_data.get('content', '')

        soup = page_data['soup']

        # Remove script and style elements if requested
        if remove_scripts:
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()

        if selector:
            elements = soup.select(selector)
            if not elements:
                return ''
            # If multiple elements, combine them
            if len(elements) == 1:
                html_content = str(elements[0])
            else:
                # Combine multiple elements
                html_content = '\n'.join(str(elem) for elem in elements)
        else:
            # Extract entire document
            html_content = str(soup)

        # Prettify if requested
        if prettify:
            # Parse and prettify (BeautifulSoup's prettify method)
            temp_soup = BeautifulSoup(html_content, 'html.parser')
            return temp_soup.prettify()
        else:
            return html_content

    def extract_html_clean(
        self,
        page_data: Dict[str, Any],
        selector: Optional[str] = None,
        remove_attributes: bool = False,
        keep_links: bool = True,
    ) -> str:
        """
        Extract clean, formatted HTML content from a page.

        Args:
            page_data: Page data dictionary from navigate().
            selector: Optional CSS selector to extract specific element(s).
            remove_attributes: If True, remove all attributes except href/src/alt.
            keep_links: If True, preserve links (href attributes).

        Returns:
            Clean HTML content as string.
        """
        if not page_data.get('soup'):
            return page_data.get('content', '')

        soup = page_data['soup']

        # Remove script, style, and other unwanted elements
        for element in soup(['script', 'style', 'noscript', 'meta', 'link']):
            element.decompose()

        if selector:
            elements = soup.select(selector)
            if not elements:
                return ''
            soup = BeautifulSoup('', 'html.parser')
            for elem in elements:
                soup.append(elem.extract())

        # Clean attributes if requested
        if remove_attributes:
            for tag in soup.find_all(True):
                # Keep only essential attributes
                attrs_to_keep = []
                if keep_links and tag.name == 'a' and tag.get('href'):
                    attrs_to_keep.append('href')
                if tag.name == 'img' and tag.get('src'):
                    attrs_to_keep.append('src')
                if tag.name == 'img' and tag.get('alt'):
                    attrs_to_keep.append('alt')

                # Remove all attributes except those to keep
                new_attrs = {
                    key: tag.attrs[key] for key in attrs_to_keep if key in tag.attrs
                }
                tag.attrs = new_attrs

        return soup.prettify()

    def find_table(
        self, page_data: Dict[str, Any], headers: Optional[List[str]] = None
    ) -> Optional[List[Dict[str, str]]]:
        """
        Find and extract data from a table on the page.

        Args:
            page_data: Page data dictionary from navigate().
            headers: Optional list of header names to match.

        Returns:
            List of dictionaries with table data, or None if not found.
        """
        if not page_data.get('soup'):
            return None

        soup = page_data['soup']
        tables = soup.find_all('table')

        for table in tables:
            # Extract headers
            table_headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    table_headers.append(th.get_text(strip=True))

            # If headers specified, check if they match
            if headers:
                if not all(
                    h.lower() in ' '.join(table_headers).lower() for h in headers
                ):
                    continue

            # Extract rows
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header row
                cells = tr.find_all(['td', 'th'])
                row_data = {}
                for i, cell in enumerate(cells):
                    header = (
                        table_headers[i] if i < len(table_headers) else f'Column_{i}'
                    )
                    row_data[header] = cell.get_text(strip=True)
                if row_data:
                    rows.append(row_data)

            if rows:
                self.logger.info(f'Extracted table with {len(rows)} rows')
                return rows

        return None

    def search_text(
        self, page_data: Dict[str, Any], search_terms: List[str]
    ) -> Dict[str, Any]:
        """
        Search for specific terms in the page content.

        Args:
            page_data: Page data dictionary from navigate().
            search_terms: List of terms to search for.

        Returns:
            Dictionary with:
            - found: List of terms that were found
            - not_found: List of terms that were not found
            - contexts: Dictionary mapping terms to their context snippets
        """
        text = self.extract_text(page_data).lower()

        found = []
        not_found = []
        contexts = {}

        for term in search_terms:
            term_lower = term.lower()
            if term_lower in text:
                found.append(term)
                # Extract context around the term
                pattern = f'.{{0,100}}{re.escape(term_lower)}.{{0,100}}'
                matches = re.findall(pattern, text, re.IGNORECASE)
                contexts[term] = matches[:3] if matches else []
            else:
                not_found.append(term)

        return {
            'found': found,
            'not_found': not_found,
            'contexts': contexts,
        }

    def extract_links_list(self, page_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract all links from the page with their text and URLs.

        Args:
            page_data: Page data dictionary from navigate().

        Returns:
            List of dictionaries with 'text' and 'url' keys.
        """
        if not page_data.get('soup'):
            return []

        soup = page_data['soup']
        base_url = page_data['url']
        links = []

        for link in soup.find_all('a', href=True):
            text = link.get_text(strip=True)
            href = link.get('href', '')
            full_url = urljoin(base_url, href)

            if text or href:
                links.append(
                    {
                        'text': text,
                        'url': full_url,
                    }
                )

        self.logger.info(f'Extracted {len(links)} links from page')
        return links

    def extract_numeric_data(
        self,
        page_data: Dict[str, Any],
        context_keywords: Optional[List[str]] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract numeric counts and statistics from a web page with context awareness.

        This method looks for numeric patterns associated with keywords like "article",
        "count", "total", etc. It searches in text, tables, and structured elements.

        Args:
            page_data: Page data dictionary from navigate().
            context_keywords: Optional list of keywords to look for near numbers
                           (e.g., ["article", "count", "total"]).
            min_value: Optional minimum value threshold to filter results.
            max_value: Optional maximum value threshold to filter results.

        Returns:
            Dictionary with:
            - numeric_values: List of extracted numeric values with context
            - counts: List of potential count values
            - statistics: List of statistics found
            - table_numbers: Numbers extracted from tables
        """
        if not page_data.get('soup'):
            return {
                'numeric_values': [],
                'counts': [],
                'statistics': [],
                'table_numbers': [],
            }

        soup = page_data['soup']
        results = {
            'numeric_values': [],
            'counts': [],
            'statistics': [],
            'table_numbers': [],
        }

        # Extract full page text for pattern matching
        full_text = self.extract_text(page_data)

        # Pattern to match numbers with context
        # Matches: "1002 articles", "Total: 1002", "1002 items", etc.
        number_patterns = [
            r'(\d{1,7})\s+(?:articles?|items?|papers?|publications?|records?|entries?|count|total|number)',
            r'(?:total|count|number|articles?|items?):\s*(\d{1,7})',
            r'(\d{1,7})\s+(?:found|published|listed|shown|available)',
            r'(?:show|display|found|published)\s+(?:all\s+)?(\d{1,7})',
            r'(\d{1,7})\s+(?:results?|matches?|hits?)',
        ]

        # If context keywords provided, create more specific patterns
        if context_keywords:
            for keyword in context_keywords:
                keyword_lower = keyword.lower()
                # Pattern: "1002 articles" or "articles: 1002"
                patterns = [
                    rf'(\d{{1,7}})\s+{re.escape(keyword_lower)}s?\b',
                    rf'{re.escape(keyword_lower)}s?[:\s]+(\d{{1,7}})',
                    rf'(\d{{1,7}})\s+{re.escape(keyword_lower)}',
                ]
                number_patterns.extend(patterns)

        # Find all numeric matches with context
        numeric_matches = []
        seen_values = set()

        for pattern in number_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1)
                try:
                    value = int(value_str)
                    # Apply filters
                    if min_value is not None and value < min_value:
                        continue
                    if max_value is not None and value > max_value:
                        continue

                    # Get context (surrounding text)
                    start = max(0, match.start() - 50)
                    end = min(len(full_text), match.end() + 50)
                    context = full_text[start:end].strip()

                    # Avoid duplicates
                    value_key = (value, context[:100])
                    if value_key not in seen_values:
                        seen_values.add(value_key)
                        numeric_matches.append(
                            {
                                'value': value,
                                'context': context,
                                'pattern': match.group(0),
                            }
                        )
                except ValueError:
                    continue

        # Extract numbers from tables
        table_numbers = []
        tables = soup.find_all('table')
        for table in tables:
            cells = table.find_all(['td', 'th'])
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                # Look for numeric cells
                numeric_match = re.search(r'^\s*(\d{1,7})\s*$', cell_text)
                if numeric_match:
                    try:
                        value = int(numeric_match.group(1))
                        if min_value is not None and value < min_value:
                            continue
                        if max_value is not None and value > max_value:
                            continue

                        # Get row/header context
                        row = cell.find_parent('tr')
                        row_text = row.get_text(sep=' ', strip=True) if row else ''
                        table_numbers.append(
                            {
                                'value': value,
                                'context': row_text,
                                'source': 'table',
                            }
                        )
                    except ValueError:
                        continue

        # Extract numbers from list items and structured elements
        list_numbers = []
        for list_elem in soup.find_all(['ul', 'ol', 'dl']):
            list_items = list_elem.find_all(
                'li' if list_elem.name in ['ul', 'ol'] else ['dt', 'dd']
            )
            for item in list_items:
                item_text = item.get_text(strip=True)
                # Look for patterns like "X articles" or "Total: X"
                for pattern in number_patterns[:3]:  # Use simpler patterns for lists
                    match = re.search(pattern, item_text, re.IGNORECASE)
                    if match:
                        try:
                            value = int(match.group(1))
                            if min_value is not None and value < min_value:
                                continue
                            if max_value is not None and value > max_value:
                                continue
                            list_numbers.append(
                                {
                                    'value': value,
                                    'context': item_text,
                                    'source': 'list',
                                }
                            )
                            break
                        except ValueError:
                            continue

        # Combine all results
        results['numeric_values'] = numeric_matches
        results['counts'] = [
            m
            for m in numeric_matches
            if 'count' in m['context'].lower() or 'total' in m['context'].lower()
        ]
        results['statistics'] = numeric_matches  # All numeric values can be statistics
        results['table_numbers'] = table_numbers
        results['list_numbers'] = list_numbers

        # Log findings
        all_numbers = (
            [m['value'] for m in numeric_matches]
            + [n['value'] for n in table_numbers]
            + [n['value'] for n in list_numbers]
        )
        if all_numbers:
            self.logger.info(
                f'Extracted {len(all_numbers)} numeric values: {sorted(set(all_numbers))[:10]}'
            )
            # Log sample contexts for debugging
            if numeric_matches:
                sample_context = numeric_matches[0].get('context', '')[:200]
                self.logger.debug(f'Sample extraction context: {sample_context}')
        else:
            self.logger.warning(
                'No numeric values extracted. This might indicate the page structure '
                "doesn't match expected patterns. Consider using LLM extraction instead."
            )
            # Log a sample of the page text for debugging
            sample_text = full_text[:500] if full_text else 'No text extracted'
            self.logger.debug(f'Sample page text (first 500 chars): {sample_text}')
            # Also log HTML sample to help debug why extraction failed
            try:
                html_sample = self.extract_html(
                    page_data, prettify=False, remove_scripts=True
                )
                self.logger.debug(
                    f'HTML content sample (first 1000 chars) for debugging:\n'
                    f'{html_sample[:1000]}'
                )
            except Exception as html_error:
                self.logger.debug(f'Could not extract HTML for debugging: {html_error}')

        return results

    def extract_with_llm(
        self,
        page_data: Dict[str, Any],
        extraction_query: str,
        llm_service: 'LLMService',
        context_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to extract specific information from a web page.

        This method extracts the page text and uses an LLM to intelligently
        extract the specific information requested, avoiding false matches
        from dates, page numbers, etc.

        Args:
            page_data: Page data dictionary from navigate().
            extraction_query: Description of what to extract (e.g., "total number of articles published in 2020").
            llm_service: LLM service instance for extraction.
            context_keywords: Optional list of keywords to help focus extraction.

        Returns:
            Dictionary with:
            - extracted_value: The extracted value (can be number, text, etc.)
            - extracted_data: Structured data extracted
            - confidence: Confidence level (if available)
            - reasoning: Brief explanation of what was found
        """
        if not page_data.get('soup'):
            return {
                'extracted_value': None,
                'extracted_data': {},
                'confidence': 0.0,
                'reasoning': 'Page not loaded successfully',
            }

        # Extract HTML content - always use HTML for better structure understanding
        # HTML preserves semantic structure that helps LLM identify correct information
        page_html = self.extract_html(page_data, prettify=False, remove_scripts=True)

        # Truncate HTML to avoid token limits while preserving structure
        # Keep first part that likely contains main content
        max_html_length = 45000  # Leave room for prompt text
        if len(page_html) > max_html_length:
            # Truncate but try to preserve HTML structure by finding a good break point
            truncated = page_html[:max_html_length]
            # Try to find the last closing tag before truncation to avoid broken HTML
            last_tag_end = truncated.rfind('>')
            if last_tag_end > max_html_length - 1000:  # If we're close to a tag
                truncated = truncated[: last_tag_end + 1]
            final_page_content = truncated + '\n[HTML content truncated for length...]'
        else:
            final_page_content = page_html

        # Build system prompt for extraction
        system_prompt = """You are an expert at extracting specific information from web page content.

The content is provided in HTML format (with tags). Use the HTML structure to better understand the page layout and identify the correct information.

Your task is to extract the exact information requested from the provided web page content.
Be precise and avoid extracting:
- Dates (e.g., "25 Mar 2020" is not "25 articles")
- Page numbers or volume numbers
- Small numbers that are clearly dates or indices
- Unrelated numeric values

Focus on:
- Actual counts, totals, or statistics that match the query
- Numbers that clearly represent the requested quantity
- Explicit statements like "X articles", "Total: X", "X results found"
- HTML elements that contain count information (e.g., elements with classes like "count", "total", "number")

Return a JSON object with:
- extracted_value: The actual value found (number, text, or null if not found)
- extracted_data: Additional structured data if available
- confidence: A value between 0 and 1 indicating confidence in the extraction
- reasoning: Brief explanation of where/how you found the value, or why it wasn't found
- context: The surrounding text where the value was found (if applicable)"""

        # Build user prompt
        context_hint = ''
        if context_keywords:
            context_hint = (
                f'\nContext keywords to focus on: {", ".join(context_keywords)}'
            )

        user_prompt = f"""Extract the following information from the web page content:

Query: {extraction_query}{context_hint}

Web page content:
---
{final_page_content}
---

Extract the exact information requested. The content is in HTML format - use the HTML structure to locate the information (look for semantic HTML elements, classes, or IDs that might indicate counts or totals). Pay attention to HTML elements that clearly contain numeric data relevant to the query. If you find multiple potential matches, select the one that most clearly matches the query and has the strongest context indicating it's the correct value.

Return the result as a JSON object."""

        try:
            response = llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for more precise extraction
                response_format={'type': 'json_object'},
            )

            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON if response is wrapped
                json_match = re.search(
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL
                )
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    # Fallback: try to extract just the value
                    number_match = re.search(r'\d{3,}', response)
                    if number_match:
                        result = {
                            'extracted_value': int(number_match.group(0)),
                            'confidence': 0.7,
                            'reasoning': 'Extracted from LLM response (fallback)',
                        }
                    else:
                        result = {
                            'extracted_value': None,
                            'confidence': 0.0,
                            'reasoning': 'Could not parse LLM response',
                        }

            self.logger.info(
                f'LLM extraction result: value={result.get("extracted_value")}, '
                f'confidence={result.get("confidence", 0.0):.2f}'
            )

            return result

        except Exception as e:
            self.logger.error(f'LLM extraction failed: {e}')
            # Log HTML sample for debugging
            try:
                html_sample = self.extract_html(
                    page_data, prettify=False, remove_scripts=True
                )
                self.logger.debug(
                    f'HTML content sample (first 1000 chars for debugging):\n'
                    f'{html_sample[:1000]}'
                )
            except Exception as html_error:
                self.logger.debug(f'Could not extract HTML for debugging: {html_error}')
            return {
                'extracted_value': None,
                'extracted_data': {},
                'confidence': 0.0,
                'reasoning': f'Extraction failed: {str(e)}',
            }
