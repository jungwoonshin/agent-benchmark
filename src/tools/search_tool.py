"""Search functionality using Serper API."""

import logging
import os
from typing import List

import requests

from ..models import SearchResult


class SearchTool:
    """Tool for performing web searches using Serper API."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize search tool.

        Args:
            logger: Logger instance.
        """
        self.logger = logger
        # Default API key, can be overridden by environment variable
        self.default_api_key = 'cbf9fdb489521d2528d60175f2e5665e71ee6904'

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Performs a web search using Serper API.

        Uses SERPER_API_KEY from environment if available, otherwise uses default key.
        Falls back to empty results if API request fails.

        Args:
            query: Search query string.
            num_results: Number of results to return (default: 5).

        Returns:
            List of SearchResult objects.
        """
        self.logger.info("Tool 'search' called.")
        self.logger.info(f'Search query: {query}')

        try:
            # Get Serper API key from environment or use default
            api_key = os.getenv('SERPER_API_KEY') or self.default_api_key

            if not api_key:
                self.logger.warning(
                    'Serper API key not found. Using fallback empty results. '
                    'Set SERPER_API_KEY in .env for real search.'
                )
                results = []
                self.logger.info(f'Search returned {len(results)} result(s) (mock).')
                return results

            # Perform Serper API request using google.serper.dev endpoint
            url = 'https://google.serper.dev/search'
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json',
            }
            payload = {
                'q': query,
                'num': min(num_results, 100),  # Serper API supports up to 100 results
                'gl': 'us',  # Fixed geographic location to disable location-based personalization
                'hl': 'en',  # Fixed language to disable language-based personalization
            }

            self.logger.info(f'Calling Serper API with query: "{query}"')
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            # Parse search results from Serper API response
            # Serper returns results in 'organic' field
            if 'organic' in data:
                for idx, item in enumerate(data['organic'][:num_results]):
                    results.append(
                        SearchResult(
                            snippet=item.get('snippet', ''),
                            url=item.get('link', ''),
                            title=item.get('title', ''),
                            relevance_score=float(
                                idx + 1
                            ),  # Use position as relevance score
                        )
                    )

            self.logger.info(
                f'Search returned {len(results)} result(s) from Serper API.'
            )
            return results

        except requests.exceptions.RequestException as e:
            self.logger.error(f'Search API request FAILED: {e}', exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f'Search FAILED: {e}', exc_info=True)
            return []
