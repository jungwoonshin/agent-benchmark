"""arXiv API client with dynamic import handling."""

import logging
import re
from typing import Any, Dict, Optional

# Module-level cache for arxiv module
_arxiv_module = None
_import_attempted = False


def _get_arxiv_module():
    """
    Dynamically import arxiv module, retrying if it becomes available later.

    Returns:
        arxiv module if available, None otherwise
    """
    global _arxiv_module, _import_attempted

    # If we've already successfully imported, return cached module
    if _arxiv_module is not None:
        return _arxiv_module

    # If import failed before, try again (library might have been installed)
    if _import_attempted:
        try:
            import arxiv

            _arxiv_module = arxiv
            return _arxiv_module
        except ImportError:
            return None

    # First attempt
    _import_attempted = True
    try:
        import arxiv

        _arxiv_module = arxiv
        return _arxiv_module
    except ImportError:
        return None


def is_arxiv_available() -> bool:
    """
    Check if arxiv library is available.

    Returns:
        True if arxiv library can be imported, False otherwise
    """
    return _get_arxiv_module() is not None


def fetch_paper_from_api(
    paper_id: str, logger: Optional[logging.Logger] = None
) -> Optional[Any]:
    """
    Fetch arXiv paper object from API.

    Args:
        paper_id: arXiv paper ID (e.g., "2207.01510")
        logger: Optional logger for logging

    Returns:
        arxiv.Result object if found, None otherwise
    """
    arxiv = _get_arxiv_module()
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

        return paper
    except Exception as e:
        if logger:
            logger.warning(f'Failed to fetch paper from arXiv API for {paper_id}: {e}')
        return None


def extract_dates_from_paper(paper: Any) -> Dict[str, Optional[str]]:
    """
    Extract submission and updated dates from arxiv paper object.

    Args:
        paper: arxiv.Result object

    Returns:
        Dictionary with keys:
        - submission_date: YYYY-MM-DD format
        - submission_month: YYYY-MM format
        - submission_date_text: Human-readable submission date
        - updated_date: YYYY-MM-DD format
        - updated_date_text: Human-readable updated date
    """
    submission_date = None
    submission_month = None
    submission_date_text = None

    if paper.published:
        submission_date = paper.published.strftime('%Y-%m-%d')
        submission_month = paper.published.strftime('%Y-%m')
        submission_date_text = f'Submitted on {paper.published.strftime("%B %d, %Y")}'

    updated_date = None
    updated_date_text = None
    if paper.updated:
        updated_date = paper.updated.strftime('%Y-%m-%d')
        updated_date_text = f'Updated on {paper.updated.strftime("%B %d, %Y")}'

    return {
        'submission_date': submission_date,
        'submission_month': submission_month,
        'submission_date_text': submission_date_text,
        'updated_date': updated_date,
        'updated_date_text': updated_date_text,
    }


def extract_metadata_from_paper(
    paper: Any, paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract all metadata from arxiv paper object.

    Args:
        paper: arxiv.Result object
        paper_id: Optional paper ID (if not provided, will be extracted from entry_id)

    Returns:
        Dictionary with paper metadata
    """
    dates = extract_dates_from_paper(paper)

    # Get PDF URL
    pdf_url = paper.pdf_url if hasattr(paper, 'pdf_url') else None
    if not pdf_url:
        # Extract paper ID from entry_id if not provided
        if not paper_id and hasattr(paper, 'entry_id') and paper.entry_id:
            # entry_id format: http://arxiv.org/abs/2207.01510v1
            match = re.search(r'/([\d.]+)v?\d*$', paper.entry_id)
            if match:
                paper_id = match.group(1)

        if paper_id:
            pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'

    return {
        'entry_id': paper.entry_id if hasattr(paper, 'entry_id') else None,
        'title': paper.title,
        'authors': [str(author) for author in paper.authors],
        'categories': [str(cat) for cat in paper.categories],
        'primary_category': (
            str(paper.primary_category)
            if hasattr(paper, 'primary_category') and paper.primary_category
            else None
        ),
        'summary': paper.summary,
        'comment': (
            paper.comment if hasattr(paper, 'comment') and paper.comment else None
        ),
        'journal_ref': (
            paper.journal_ref
            if hasattr(paper, 'journal_ref') and paper.journal_ref
            else None
        ),
        'doi': (paper.doi if hasattr(paper, 'doi') and paper.doi else None),
        'pdf_url': pdf_url,
        **dates,
    }
