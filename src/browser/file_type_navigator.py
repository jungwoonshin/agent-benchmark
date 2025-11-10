"""File Type Navigator for finding and clicking file download buttons/links."""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from ..llm import LLMService
    from .browser import Browser

from ..utils import extract_json_from_text


class FileTypeNavigator:
    """Navigates web pages to find and access file download buttons/links using LLM-based detection."""

    def __init__(
        self,
        browser: 'Browser',
        logger: logging.Logger,
        llm_service: Optional['LLMService'] = None,
    ):
        """
        Initialize FileTypeNavigator.

        Args:
            browser: Browser instance for navigation.
            logger: Logger instance.
            llm_service: Optional LLM service for intelligent link finding.
        """
        self.browser = browser
        self.logger = logger
        self.llm_service = llm_service

    def _is_file_url(self, url: str) -> bool:
        """
        Check if a URL points to a downloadable file.

        Args:
            url: URL to check.

        Returns:
            True if URL appears to point to a file, False otherwise.
        """
        if not url:
            return False

        url_lower = url.lower()

        # Check for file extensions
        file_extensions = {
            '.pdf',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.txt',
            '.csv',
            '.jpg',
            '.jpeg',
            '.png',
            '.gif',
            '.bmp',
            '.svg',
            '.webp',
            '.zip',
            '.tar',
            '.gz',
            '.rar',
            '.7z',
            '.mp3',
            '.mp4',
            '.avi',
            '.mov',
            '.wmv',
            '.ppt',
            '.pptx',
        }

        # Check if URL contains file extension
        for ext in file_extensions:
            if ext in url_lower:
                return True

        # Check if URL ends with common file patterns
        if any(url_lower.endswith(ext) for ext in file_extensions):
            return True

        # Check for patterns like /pdf/, /download/, etc. that often indicate files
        file_indicators = ['/pdf/', '/download/', '/file/', '/attachment/']
        if any(indicator in url_lower for indicator in file_indicators):
            return True

        return False

    def find_and_navigate_to_file(
        self,
        current_url: str,
        page_content: str,
        desired_file_type: str = 'pdf',
        page_title: Optional[str] = None,
        subtask_description: Optional[str] = None,
        problem: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find and navigate to file download links/buttons on a web page using LLM-based detection.
        This is a general solution that works for any file type (PDF, text, images, etc.).

        Args:
            current_url: Current page URL.
            page_content: HTML content of the current page.
            desired_file_type: Type of file to find ('pdf', 'text', 'image', 'document', etc.).
            page_title: Optional title of the current page.
            subtask_description: Optional subtask description for context.
            problem: Optional problem description for context.

        Returns:
            Dictionary with navigation result if file link found and clicked, None otherwise.
            If target is a direct file URL, returns dict with 'is_file_download': True and 'file_url': target.
        """
        self.logger.info(
            f'Searching for {desired_file_type} download links on page: {current_url}'
        )

        try:
            soup = BeautifulSoup(page_content, 'html.parser')

            # Strategy 1: Use LLM to intelligently find the right button/link
            # This is the primary strategy as it can handle various page structures
            llm_result = self._find_file_navigation_with_llm(
                soup,
                current_url,
                desired_file_type,
                page_title,
                subtask_description,
                problem,
            )

            if llm_result:
                action_type = llm_result.get('action_type', 'navigate')
                target = llm_result.get('target')

                if action_type == 'navigate' and target:
                    # Check if target is a direct file URL - if so, return download info instead of navigating
                    if self._is_file_url(target):
                        self.logger.info(
                            f'LLM found {desired_file_type} file URL: {target}. Returning download info...'
                        )
                        return {
                            'success': True,
                            'is_file_download': True,
                            'file_url': target,
                            'url': target,
                            'action': 'download',
                        }
                    # Direct URL navigation (not a file)
                    self.logger.info(
                        f'LLM found {desired_file_type} link: {target}. Navigating...'
                    )
                    return self.browser.navigate(url=target, use_selenium=True)
                elif action_type == 'click' and target:
                    # Check if target is a direct file URL - if so, return download info instead of clicking
                    if self._is_file_url(target):
                        self.logger.info(
                            f'LLM found {desired_file_type} file URL via click: {target}. Returning download info...'
                        )
                        return {
                            'success': True,
                            'is_file_download': True,
                            'file_url': target,
                            'url': target,
                            'action': 'download',
                        }
                    # Click element (button or link)
                    selector = llm_result.get('selector')
                    link_text = llm_result.get('link_text')
                    self.logger.info(
                        f'LLM found {desired_file_type} button/link. Clicking...'
                    )
                    click_result = self.browser.click_element(
                        selector=selector, link_text=link_text
                    )
                    if click_result.get('success'):
                        return click_result
                    else:
                        # If click failed, try navigating to the URL if available
                        if target and target.startswith('http'):
                            # Check if target is a file URL before navigating
                            if self._is_file_url(target):
                                self.logger.info(
                                    f'Click failed, but target is a file URL: {target}. Returning download info...'
                                )
                                return {
                                    'success': True,
                                    'is_file_download': True,
                                    'file_url': target,
                                    'url': target,
                                    'action': 'download',
                                }
                            self.logger.info(
                                f'Click failed, trying direct navigation to: {target}'
                            )
                            return self.browser.navigate(url=target, use_selenium=True)

            # Strategy 2: Fallback to pattern-based detection for common file types
            if desired_file_type.lower() == 'pdf':
                # Use pattern-based PDF detection directly to avoid recursion
                pdf_links = self._find_direct_pdf_links(soup, current_url)
                if pdf_links:
                    self.logger.info(f'Found {len(pdf_links)} direct PDF link(s)')
                    file_url = pdf_links[0]
                    self.logger.info(f'Navigating to PDF: {file_url}')
                    return self.browser.navigate(url=file_url, use_selenium=True)
                
                # Try finding PDF download buttons
                download_links = self._find_pdf_download_buttons(soup, current_url)
                if download_links:
                    self.logger.info(f'Found {len(download_links)} PDF download button(s)')
                    file_url = download_links[0]
                    self.logger.info(f'Navigating to PDF: {file_url}')
                    return self.browser.navigate(url=file_url, use_selenium=True)

            # Strategy 3: Try direct file extension matching
            direct_links = self._find_direct_file_links(
                soup, current_url, desired_file_type
            )
            if direct_links:
                self.logger.info(
                    f'Found {len(direct_links)} direct {desired_file_type} link(s)'
                )
                file_url = direct_links[0]
                self.logger.info(f'Navigating to {desired_file_type}: {file_url}')
                return self.browser.navigate(url=file_url, use_selenium=True)

            self.logger.info(f'No {desired_file_type} download links found on page')
            return None

        except Exception as e:
            self.logger.warning(
                f'Error searching for {desired_file_type} links on {current_url}: {e}',
                exc_info=True,
            )
            return None

    def find_and_navigate_to_pdf(
        self, current_url: str, page_content: str, page_title: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find PDF download links/buttons on a web page and navigate to them.
        This is a convenience method that calls find_and_navigate_to_file with file_type='pdf'.

        Args:
            current_url: Current page URL.
            page_content: HTML content of the current page.
            page_title: Optional title of the current page.

        Returns:
            Dictionary with navigation result if PDF link found, None otherwise.
        """
        return self.find_and_navigate_to_file(
            current_url, page_content, desired_file_type='pdf', page_title=page_title
        )

    def _find_direct_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find direct links to PDF files."""
        pdf_links = []

        # Find all links with .pdf extension
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                full_url = urljoin(base_url, href)
                pdf_links.append(full_url)

        # Also check for embedded PDFs or iframes
        for iframe in soup.find_all('iframe', src=True):
            src = iframe['src']
            if src.lower().endswith('.pdf') or 'pdf' in src.lower():
                full_url = urljoin(base_url, src)
                pdf_links.append(full_url)

        return list(set(pdf_links))  # Remove duplicates

    def _find_pdf_download_buttons(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[str]:
        """Find download buttons/links with PDF-related text."""
        download_links = []

        # Keywords that indicate PDF download
        pdf_keywords = [
            'download pdf',
            'download',
            'pdf',
            'view pdf',
            'get pdf',
            'read pdf',
            'full text',
            'fulltext',
            'article pdf',
            'paper pdf',
        ]

        # Search in link text and button text
        for element in soup.find_all(['a', 'button']):
            text = element.get_text(strip=True).lower()
            href = element.get('href', '')

            # Check if text contains PDF keywords
            if any(keyword in text for keyword in pdf_keywords):
                if href:
                    full_url = urljoin(base_url, href)
                    download_links.append(full_url)
                elif element.name == 'button':
                    # Button might trigger JavaScript - look for data attributes
                    data_url = element.get('data-url', '') or element.get(
                        'data-href', ''
                    )
                    if data_url:
                        full_url = urljoin(base_url, data_url)
                        download_links.append(full_url)

        return list(set(download_links))  # Remove duplicates

    def _find_file_navigation_with_llm(
        self,
        soup: BeautifulSoup,
        base_url: str,
        desired_file_type: str,
        page_title: Optional[str] = None,
        subtask_description: Optional[str] = None,
        problem: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to intelligently find the right button/link to navigate to a desired file type.
        This is a general solution that works for any file type and page structure.

        Args:
            soup: BeautifulSoup object of the page.
            base_url: Base URL of the page.
            desired_file_type: Type of file to find ('pdf', 'text', 'image', etc.).
            page_title: Optional page title.
            subtask_description: Optional subtask description for context.
            problem: Optional problem description for context.

        Returns:
            Dictionary with navigation instructions, or None if not found.
        """
        if not self.llm_service:
            return None

        # Extract all clickable elements (links, buttons, etc.)
        clickable_elements = []

        # Extract links
        for link in soup.find_all('a', href=True, limit=100):
            text = link.get_text(strip=True)
            href = link.get('href', '')
            element_id = link.get('id', '')
            classes = ' '.join(link.get('class', []))
            onclick = link.get('onclick', '')

            if text or href:
                clickable_elements.append(
                    {
                        'type': 'link',
                        'text': text,
                        'href': href,
                        'id': element_id,
                        'classes': classes,
                        'onclick': onclick[:200]
                        if onclick
                        else '',  # Limit onclick length
                    }
                )

        # Extract buttons
        for button in soup.find_all(['button', 'input'], limit=50):
            button_type = button.get('type', '').lower()
            if button_type in ['submit', 'button'] or button.name == 'button':
                text = button.get_text(strip=True) or button.get('value', '')
                element_id = button.get('id', '')
                classes = ' '.join(button.get('class', []))
                onclick = button.get('onclick', '')
                data_url = (
                    button.get('data-url', '')
                    or button.get('data-href', '')
                    or button.get('data-pdf-url', '')
                )

                if text or onclick or data_url:
                    clickable_elements.append(
                        {
                            'type': 'button',
                            'text': text,
                            'href': data_url,  # Use data-url as href for buttons
                            'id': element_id,
                            'classes': classes,
                            'onclick': onclick[:200] if onclick else '',
                        }
                    )

        if not clickable_elements:
            self.logger.debug('No clickable elements found on page')
            return None

        # Limit to top 50 elements to avoid token limits
        clickable_elements = clickable_elements[:50]

        # Build context for LLM
        context_parts = []
        if problem:
            context_parts.append(f'Problem: {problem}')
        if subtask_description:
            context_parts.append(f'Subtask: {subtask_description}')

        context = '\n'.join(context_parts) if context_parts else ''

        # Format elements for LLM
        elements_text = []
        for idx, elem in enumerate(clickable_elements, 1):
            elem_str = f'[{idx}] Type: {elem["type"]}'
            if elem['text']:
                elem_str += f", Text: '{elem['text']}'"
            if elem['href']:
                elem_str += f", URL/href: '{elem['href']}'"
            if elem['id']:
                elem_str += f", ID: '{elem['id']}'"
            if elem['classes']:
                elem_str += f", Classes: '{elem['classes']}'"
            if elem['onclick']:
                elem_str += f", OnClick: '{elem['onclick'][:100]}...'"
            elements_text.append(elem_str)

        elements_list = '\n'.join(elements_text)

        system_prompt = """You are an expert at analyzing web pages to find the correct button or link that leads to a specific file type.

Given a list of clickable elements (links, buttons) from a web page, identify which element should be clicked to access the desired file type.

Your task:
1. Analyze each clickable element (links, buttons) on the page
2. Determine which element is most likely to lead to the desired file type
3. Consider:
   - Element text (e.g., "Download PDF", "View HTML", "Full Text", "Get Document")
   - URL patterns (e.g., links ending in .pdf, .txt, etc.)
   - Element context (download sections, article sections, etc.)
   - Data attributes (data-url, data-href) for JavaScript-triggered downloads
   - OnClick handlers that might trigger file downloads

Return a JSON object with:
- action_type: "navigate" (for direct URL links) or "click" (for buttons/JavaScript links)
- target: The URL to navigate to (for "navigate") or the element identifier (for "click")
- selector: CSS selector for the element (if action_type is "click" and you can construct a selector)
- link_text: Link text to match (if action_type is "click" and matching by text)
- confidence: Confidence score from 0.0 to 1.0
- reasoning: Brief explanation of why this element was chosen

If no suitable element is found, return:
{"action_type": null, "target": null, "confidence": 0.0, "reasoning": "explanation"}

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        file_type_description = self._get_file_type_description(desired_file_type)
        title_info = f'\nPage Title: {page_title}' if page_title else ''

        user_prompt = f"""Page URL: {base_url}
{title_info}
{context}

Desired File Type: {desired_file_type} ({file_type_description})

Clickable elements on the page:
{elements_list}

Which element should be clicked to access the {desired_file_type} file? Provide the element number and how to interact with it."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent detection
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            action_type = result_data.get('action_type')
            if not action_type:
                self.logger.debug(
                    f'LLM did not find a suitable element for {desired_file_type}'
                )
                return None

            target = result_data.get('target')
            if not target:
                return None

            # If target is a relative URL, make it absolute
            if action_type == 'navigate' and not target.startswith('http'):
                target = urljoin(base_url, target)

            # If action is click but we have a URL, we can try navigating directly
            if action_type == 'click' and target.startswith('http'):
                self.logger.info(
                    f'Button has direct URL, will try navigation: {target}'
                )
                # Return as navigate action if we have a direct URL
                return {
                    'action_type': 'navigate',
                    'target': target,
                    'selector': result_data.get('selector'),
                    'link_text': result_data.get('link_text'),
                    'confidence': result_data.get('confidence', 0.5),
                    'reasoning': result_data.get('reasoning', ''),
                }

            return {
                'action_type': action_type,
                'target': target,
                'selector': result_data.get('selector'),
                'link_text': result_data.get('link_text'),
                'confidence': result_data.get('confidence', 0.5),
                'reasoning': result_data.get('reasoning', ''),
            }

        except Exception as e:
            self.logger.debug(
                f'LLM-based file navigation finding failed: {e}', exc_info=True
            )
            return None

    def _get_file_type_description(self, file_type: str) -> str:
        """Get a description of the file type for LLM prompts."""
        descriptions = {
            'pdf': 'PDF document file',
            'text': 'text file or plain text document',
            'txt': 'text file',
            'image': 'image file',
            'img': 'image file',
            'document': 'document file (PDF, DOC, etc.)',
            'doc': 'Word document',
            'docx': 'Word document',
            'html': 'HTML file or web page source',
            'xml': 'XML file',
            'json': 'JSON file',
            'csv': 'CSV file',
            'xls': 'Excel spreadsheet',
            'xlsx': 'Excel spreadsheet',
        }
        return descriptions.get(file_type.lower(), f'{file_type} file')

    def _find_direct_file_links(
        self, soup: BeautifulSoup, base_url: str, file_type: str
    ) -> List[str]:
        """Find direct links to files with specific extension."""
        file_links = []
        extension = f'.{file_type.lower()}'

        # Find all links with the file extension
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith(extension):
                full_url = urljoin(base_url, href)
                file_links.append(full_url)

        # Also check for embedded files or iframes
        for iframe in soup.find_all('iframe', src=True):
            src = iframe['src']
            if src.lower().endswith(extension) or file_type.lower() in src.lower():
                full_url = urljoin(base_url, src)
                file_links.append(full_url)

        return list(set(file_links))  # Remove duplicates

    def _find_pdf_with_llm(
        self, soup: BeautifulSoup, base_url: str, page_title: Optional[str] = None
    ) -> Optional[str]:
        """Use LLM to intelligently find PDF download links (legacy method)."""
        result = self._find_file_navigation_with_llm(soup, base_url, 'pdf', page_title)
        if result and result.get('target'):
            return result['target']
        return None
