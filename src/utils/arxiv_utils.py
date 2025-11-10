"""Utility functions for arXiv paper metadata extraction."""

import logging
import re
from typing import Any, Dict, Optional

try:
    import arxiv
except ImportError:
    arxiv = None


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """
    Extract arXiv paper ID from a URL.

    Args:
        url: URL that may contain arXiv paper ID (e.g., arxiv.org/abs/2207.01510)

    Returns:
        arXiv paper ID (e.g., "2207.01510") without version suffix, or None if not found
    """
    if not url:
        return None

    arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/([\d.]+)', url, re.IGNORECASE)
    if not arxiv_match:
        return None

    paper_id = arxiv_match.group(1)
    # Remove version suffix if present (e.g., 2207.01510v1 -> 2207.01510)
    paper_id = re.sub(r'v\d+$', '', paper_id)
    return paper_id


def extract_arxiv_id_from_text(text: str) -> Optional[str]:
    """
    Extract arXiv paper ID from text content.

    Args:
        text: Text that may contain arXiv paper ID (e.g., "arXiv:2207.01510v1")

    Returns:
        arXiv paper ID (e.g., "2207.01510") without version suffix, or None if not found
    """
    if not text:
        return None

    # Look for arXiv ID in text (format: arXiv:YYMM.XXXXX or YYMM.XXXXX)
    arxiv_id_match = re.search(r'arxiv[:\s]+([\d]{4}\.[\d]{5})', text, re.IGNORECASE)
    if arxiv_id_match:
        paper_id = arxiv_id_match.group(1)
        # Remove version suffix if present
        paper_id = re.sub(r'v\d+$', '', paper_id)
        return paper_id

    return None


def get_arxiv_metadata(
    paper_id: str, logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch arXiv paper metadata from arXiv API using paper ID.

    Args:
        paper_id: arXiv paper ID (e.g., "2207.01510")
        logger: Optional logger for logging

    Returns:
        Dictionary with keys:
        - paper_id: arXiv paper ID
        - submission_date: Submission date in YYYY-MM-DD format
        - submission_date_text: Human-readable submission date
        - submission_month: Month in YYYY-MM format
        - title: Paper title
        - authors: List of author names
        - categories: List of arXiv categories
        - summary: Paper abstract
        - confidence: Confidence score (1.0 for API results)
        None if fetch fails or arxiv library not available
    """
    if not arxiv:
        if logger:
            logger.warning(
                'arxiv library not available. Install with: uv pip install arxiv'
            )
        return None

    if not paper_id:
        return None

    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results(), None)

        if not paper:
            if logger:
                logger.debug(f'No paper found for arXiv ID: {paper_id}')
            return None

        submission_date = None
        submission_month = None
        submission_date_text = None

        if paper.published:
            submission_date = paper.published.strftime('%Y-%m-%d')
            submission_month = paper.published.strftime('%Y-%m')
            submission_date_text = (
                f'Submitted on {paper.published.strftime("%B %d, %Y")}'
            )

        result = {
            'paper_id': paper_id,
            'submission_date': submission_date,
            'submission_date_text': submission_date_text,
            'submission_month': submission_month,
            'title': paper.title,
            'authors': [str(author) for author in paper.authors],
            'categories': [str(cat) for cat in paper.categories],
            'summary': paper.summary,
            'confidence': 1.0,  # High confidence from API
        }

        if logger:
            logger.debug(
                f'Fetched arXiv metadata from API for {paper_id}: '
                f'submission_date={submission_date}'
            )

        return result

    except Exception as e:
        if logger:
            logger.warning(
                f'Failed to fetch metadata from arXiv API for {paper_id}: {e}'
            )
        return None


def get_arxiv_submission_date(
    url: str, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Extract arXiv paper ID from URL and get submission date using arXiv API.

    Args:
        url: URL that may contain arXiv paper ID
        logger: Optional logger for logging

    Returns:
        Submission date in YYYY-MM-DD format, or None if not found/not arXiv
    """
    paper_id = extract_arxiv_id_from_url(url)
    if not paper_id:
        return None

    metadata = get_arxiv_metadata(paper_id, logger)
    if metadata:
        return metadata.get('submission_date')

    return None
