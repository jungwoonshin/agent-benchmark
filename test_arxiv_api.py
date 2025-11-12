#!/usr/bin/env python3
"""Test script to verify arXiv API is working correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.utils.api_requester import UnifiedAPIRequester
from src.utils.arxiv_api_client import fetch_paper_from_api, is_arxiv_available

# Setup logging
logger = setup_logging()


def test_arxiv_availability():
    """Test if arxiv library is available."""
    print('\n' + '=' * 60)
    print('Test 1: Checking arXiv library availability')
    print('=' * 60)

    available = is_arxiv_available()
    if available:
        print('✓ arXiv library is available')
        return True
    else:
        print('✗ arXiv library is NOT available')
        print('  Install with: uv pip install arxiv')
        return False


def test_arxiv_extract_id_from_url():
    """Test arXiv extract_id_from_url functionality."""
    print('\n' + '=' * 60)
    print('Test 2: Testing arXiv extract_id_from_url')
    print('=' * 60)

    requester = UnifiedAPIRequester(logger)

    # Test with a valid arXiv URL
    url = 'https://arxiv.org/abs/1706.03762'

    print(f"Extracting ID from URL: '{url}'")

    try:
        paper_id = requester.request(
            api_name='arxiv',
            method='extract_id_from_url',
            url=url,
        )

        if paper_id is None:
            print('✗ extract_id_from_url returned None')
            return False

        print(f'✓ extract_id_from_url successful! Extracted ID: {paper_id}')
        return True

    except Exception as e:
        print(f'✗ extract_id_from_url failed with error: {e}')
        import traceback

        traceback.print_exc()
        return False


def test_arxiv_paper_fetch():
    """Test fetching a specific paper by ID."""
    print('\n' + '=' * 60)
    print('Test 3: Testing arXiv paper fetch by ID')
    print('=' * 60)

    # Use a well-known paper ID
    paper_id = '1706.03762'  # "Attention Is All You Need"

    print(f'Fetching paper: {paper_id}')

    try:
        paper = fetch_paper_from_api(paper_id, logger)

        if paper is None:
            print('✗ Paper fetch returned None')
            return False

        print('✓ Paper fetch successful!')
        print(f'  Title: {paper.title[:80]}...')
        print(f'  Authors: {", ".join([str(a) for a in paper.authors[:3]])}...')
        print(f'  Published: {paper.published}')
        print(f'  PDF URL: {paper.pdf_url}')

        return True

    except Exception as e:
        print(f'✗ Paper fetch failed with error: {e}')
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print('\n' + '=' * 60)
    print('arXiv API Test Suite')
    print('=' * 60)

    results = []

    # Test 1: Check availability
    results.append(('Library Availability', test_arxiv_availability()))

    # Only run other tests if library is available
    if results[0][1]:
        # Test 2: Extract ID from URL
        results.append(('Extract ID from URL', test_arxiv_extract_id_from_url()))

        # Test 3: Fetch paper
        results.append(('Paper Fetch', test_arxiv_paper_fetch()))

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
