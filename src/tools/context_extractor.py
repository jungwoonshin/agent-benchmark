"""Context extraction utilities for extracting data from context dictionaries."""

import logging
import re
from typing import Any, Dict, List, Optional

import requests


class ContextExtractor:
    """Utility class for extracting structured data from context dictionaries."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize context extractor.

        Args:
            logger: Logger instance.
        """
        self.logger = logger

    def extract_zip_codes_from_context(self, context: dict, code: str) -> List[str]:
        """Extract zip codes from context (search results, URLs, text)."""
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

    def extract_dates_from_context(self, context: dict, code: str) -> List[str]:
        """Extract dates from context."""
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

    def extract_numbers_from_context(self, context: dict, code: str) -> List[float]:
        """Extract numbers from context."""
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

    def extract_urls_from_context(self, context: dict, code: str) -> List[str]:
        """Extract URLs from context."""
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

    def extract_structured_data_from_context(
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

