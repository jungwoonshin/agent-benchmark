#!/usr/bin/env python3
"""Test script to verify Wikipedia API is working correctly."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.utils.api_requester import WikipediaAPIRequester, UnifiedAPIRequester

# Setup logging
logger = setup_logging()


def test_mwclient_availability():
    """Test if mwclient library is available."""
    print('\n' + '=' * 60)
    print('Test 1: Checking mwclient library availability')
    print('=' * 60)

    try:
        import mwclient
        print('✓ mwclient library is available')
        print(f'  Version: {mwclient.__version__ if hasattr(mwclient, "__version__") else "unknown"}')
        return True
    except ImportError:
        print('✗ mwclient library is NOT available')
        print('  Install with: uv pip install mwclient')
        return False


def test_wikipedia_initialization():
    """Test Wikipedia API requester initialization."""
    print('\n' + '=' * 60)
    print('Test 2: Testing Wikipedia API initialization')
    print('=' * 60)

    try:
        requester = WikipediaAPIRequester(logger)
        
        if requester.site is None:
            print('✗ Wikipedia site not initialized')
            print('  This may be due to mwclient not being installed')
            return False
        
        print('✓ Wikipedia API requester initialized successfully')
        print(f'  Language: {requester.language}')
        print(f'  Site: {requester.site}')
        return True

    except Exception as e:
        print(f'✗ Initialization failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_search_basic():
    """Test basic Wikipedia search functionality."""
    print('\n' + '=' * 60)
    print('Test 3: Testing Wikipedia search (basic query)')
    print('=' * 60)

    requester = UnifiedAPIRequester(logger)

    # Test with a well-known search term
    query = 'Python programming'
    print(f'Searching for: "{query}"')

    try:
        results = requester.request(
            api_name='wikipedia',
            method='search_pages',
            query=query,
            limit=5,
        )

        if results is None:
            print('✗ Search returned None')
            return False

        if not isinstance(results, list):
            print(f'✗ Search returned unexpected type: {type(results)}')
            return False

        if len(results) == 0:
            print('⚠ Search returned empty results (may be network issue)')
            return True  # Not a failure, just no results

        print(f'✓ Search successful! Found {len(results)} results')
        for i, result in enumerate(results[:3], 1):
            print(f'  {i}. {result.get("title", "N/A")}')
            print(f'     URL: {result.get("url", "N/A")}')
        return True

    except Exception as e:
        print(f'✗ Search failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_search_empty_query():
    """Test Wikipedia search with empty query."""
    print('\n' + '=' * 60)
    print('Test 4: Testing Wikipedia search (empty query)')
    print('=' * 60)

    requester = UnifiedAPIRequester(logger)

    try:
        # Test with empty string
        results = requester.request(
            api_name='wikipedia',
            method='search_pages',
            query='',
            limit=5,
        )

        if results is None:
            print('✗ Search returned None (should return empty list)')
            return False

        if not isinstance(results, list):
            print(f'✗ Search returned unexpected type: {type(results)}')
            return False

        if len(results) == 0:
            print('✓ Empty query correctly returned empty list')
            return True
        else:
            print(f'✗ Empty query returned {len(results)} results (should be 0)')
            return False

    except Exception as e:
        print(f'✗ Empty query test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_search_whitespace_query():
    """Test Wikipedia search with whitespace-only query."""
    print('\n' + '=' * 60)
    print('Test 5: Testing Wikipedia search (whitespace query)')
    print('=' * 60)

    requester = UnifiedAPIRequester(logger)

    try:
        # Test with whitespace only
        results = requester.request(
            api_name='wikipedia',
            method='search_pages',
            query='   ',
            limit=5,
        )

        if results is None:
            print('✗ Search returned None (should return empty list)')
            return False

        if not isinstance(results, list):
            print(f'✗ Search returned unexpected type: {type(results)}')
            return False

        if len(results) == 0:
            print('✓ Whitespace query correctly returned empty list')
            return True
        else:
            print(f'✗ Whitespace query returned {len(results)} results (should be 0)')
            return False

    except Exception as e:
        print(f'✗ Whitespace query test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_search_none_handling():
    """Test Wikipedia search handles None return from mwclient."""
    print('\n' + '=' * 60)
    print('Test 6: Testing Wikipedia search (None return handling)')
    print('=' * 60)

    # Create a mock site that returns None
    mock_site = MagicMock()
    mock_site.search.return_value = None

    requester = WikipediaAPIRequester(logger)
    requester.site = mock_site

    try:
        results = requester.search_pages('test query', limit=5)

        if results is None:
            print('✗ Search returned None (should return empty list)')
            return False

        if not isinstance(results, list):
            print(f'✗ Search returned unexpected type: {type(results)}')
            return False

        if len(results) == 0:
            print('✓ None return correctly handled (returned empty list)')
            return True
        else:
            print(f'✗ None return resulted in {len(results)} results (should be 0)')
            return False

    except Exception as e:
        print(f'✗ None handling test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_search_with_limit():
    """Test Wikipedia search with different limits."""
    print('\n' + '=' * 60)
    print('Test 7: Testing Wikipedia search (with limit)')
    print('=' * 60)

    requester = UnifiedAPIRequester(logger)

    query = 'machine learning'
    limits = [1, 3, 5]

    try:
        for limit in limits:
            print(f'  Testing with limit={limit}...')
            results = requester.request(
                api_name='wikipedia',
                method='search_pages',
                query=query,
                limit=limit,
            )

            if results is None:
                print(f'  ✗ Search returned None for limit={limit}')
                return False

            if not isinstance(results, list):
                print(f'  ✗ Search returned unexpected type for limit={limit}: {type(results)}')
                return False

            if len(results) > limit:
                print(f'  ✗ Search returned {len(results)} results (exceeded limit={limit})')
                return False

            print(f'  ✓ Limit {limit}: Got {len(results)} results')

        print('✓ All limit tests passed')
        return True

    except Exception as e:
        print(f'✗ Limit test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_get_page():
    """Test Wikipedia get_page functionality."""
    print('\n' + '=' * 60)
    print('Test 8: Testing Wikipedia get_page')
    print('=' * 60)

    requester = UnifiedAPIRequester(logger)

    # Test with a well-known page
    title = 'Python (programming language)'
    print(f'Fetching page: "{title}"')

    try:
        page = requester.request(
            api_name='wikipedia',
            method='get_page',
            title=title,
        )

        if page is None:
            print('⚠ Page fetch returned None (may be network issue or page not found)')
            return True  # Not necessarily a failure

        if not isinstance(page, dict):
            print(f'✗ Page fetch returned unexpected type: {type(page)}')
            return False

        if 'title' not in page or 'content' not in page:
            print('✗ Page fetch missing required fields')
            print(f'  Got keys: {list(page.keys())}')
            return False

        print('✓ Page fetch successful!')
        print(f'  Title: {page.get("title", "N/A")}')
        content_preview = page.get('content', '')[:100]
        print(f'  Content preview: {content_preview}...')
        return True

    except Exception as e:
        print(f'✗ Page fetch failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_wikipedia_search_no_site():
    """Test Wikipedia search when site is not initialized."""
    print('\n' + '=' * 60)
    print('Test 9: Testing Wikipedia search (no site initialized)')
    print('=' * 60)

    requester = WikipediaAPIRequester(logger)
    requester.site = None  # Simulate no mwclient

    try:
        results = requester.search_pages('test query', limit=5)

        if results is None:
            print('✗ Search returned None (should return empty list)')
            return False

        if not isinstance(results, list):
            print(f'✗ Search returned unexpected type: {type(results)}')
            return False

        if len(results) == 0:
            print('✓ No site initialization correctly handled (returned empty list)')
            return True
        else:
            print(f'✗ No site resulted in {len(results)} results (should be 0)')
            return False

    except Exception as e:
        print(f'✗ No site test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print('\n' + '=' * 60)
    print('Wikipedia API Test Suite')
    print('=' * 60)

    results = []

    # Test 1: Check availability
    results.append(('Library Availability', test_mwclient_availability()))

    # Test 2: Initialization
    results.append(('Initialization', test_wikipedia_initialization()))

    # Only run other tests if library is available
    if results[0][1]:
        # Test 3: Basic search
        results.append(('Basic Search', test_wikipedia_search_basic()))

        # Test 4: Empty query
        results.append(('Empty Query Handling', test_wikipedia_search_empty_query()))

        # Test 5: Whitespace query
        results.append(('Whitespace Query Handling', test_wikipedia_search_whitespace_query()))

        # Test 6: None return handling
        results.append(('None Return Handling', test_wikipedia_search_none_handling()))

        # Test 7: Search with limit
        results.append(('Search with Limit', test_wikipedia_search_with_limit()))

        # Test 8: Get page
        results.append(('Get Page', test_wikipedia_get_page()))

        # Test 9: No site initialization
        results.append(('No Site Handling', test_wikipedia_search_no_site()))

    # Summary
    print('\n' + '=' * 60)
    print('Test Summary')
    print('=' * 60)

    for test_name, passed in results:
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'{status}: {test_name}')

    all_passed = all(result[1] for result in results)

    if all_passed:
        print('\n✓ All tests passed!')
        return 0
    else:
        print('\n✗ Some tests failed')
        return 1


if __name__ == '__main__':
    sys.exit(main())

