"""Search functionality using Google Custom Search API."""

import logging
import os
from typing import List

import requests

from ..models import SearchResult


class SearchTool:
    """Tool for performing web searches using Google Custom Search API."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize search tool.

        Args:
            logger: Logger instance.
        """
        self.logger = logger

    def search(
        self, query: str, num_results: int = 5, search_type: str = 'web'
    ) -> List[SearchResult]:
        """
        Performs a web or specialized search using Google Custom Search API.

        Requires GOOGLE_API_KEY and GOOGLE_CX (Custom Search Engine ID) in .env file.
        Falls back to mock results if API keys are not configured.

        Args:
            query: Search query string.
            num_results: Number of results to return (default: 5).
            search_type: Type of search (default: 'web').

        Returns:
            List of SearchResult objects.
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

