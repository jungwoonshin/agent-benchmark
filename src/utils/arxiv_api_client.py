"""arXiv API client with dynamic import handling."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

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


def fetch_papers_from_api_batch(
    paper_ids: List[str], logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Fetch multiple arXiv paper objects from API in a single batch request.

    Args:
        paper_ids: List of arXiv paper IDs (e.g., ["2207.01510", "2502.12430"])
        logger: Optional logger for logging

    Returns:
        Dictionary mapping paper_id to arxiv.Result object.
        Only includes successfully fetched papers.
        Example: {"2207.01510": <Result>, "2502.12430": <Result>}
    """
    arxiv = _get_arxiv_module()
    if not arxiv:
        if logger:
            logger.warning(
                'arxiv library not available. Install with: uv pip install arxiv'
            )
        return {}

    if not paper_ids:
        return {}

    # Filter out empty IDs
    valid_ids = [pid for pid in paper_ids if pid]
    if not valid_ids:
        return {}

    try:
        if logger:
            logger.info(
                f'Starting arXiv API batch request for {len(valid_ids)} paper(s): {valid_ids}'
            )
        
        # Use the new Client API with batch id_list
        # Configure with longer delays to prevent rate limiting
        client = arxiv.Client(
            page_size=100,
            delay_seconds=5.0,  # Increased from default 3.0
            num_retries=5,  # Increased from default 3
        )
        search = arxiv.Search(id_list=valid_ids)
        
        try:
            results = client.results(search)
        except Exception as e:
            if logger:
                logger.error(
                    f'✗ FAILED: Error creating results iterator for batch request ({len(valid_ids)} papers). '
                    f'Error type: {type(e).__name__}, Error: {e}'
                )
            return {}
        
        # Create a dictionary mapping paper_id to Result object
        papers_dict = {}
        try:
            for paper in results:
                # Extract paper ID from entry_id
                if hasattr(paper, 'entry_id') and paper.entry_id:
                    # entry_id format: http://arxiv.org/abs/2207.01510v1
                    match = re.search(r'/([\d.]+)v?\d*$', paper.entry_id)
                    if match:
                        paper_id = match.group(1)
                        papers_dict[paper_id] = paper
        except Exception as e:
            # Error while iterating results (could be network error, rate limit, etc.)
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            if logger:
                if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                    logger.error(
                        f'✗ FAILED: Rate limited while iterating results ({len(papers_dict)}/{len(valid_ids)} fetched so far). '
                        f'Error: {e}'
                    )
                elif error_type == 'UnexpectedEmptyPage':
                    logger.error(
                        f'✗ FAILED: Unexpected empty page while iterating results ({len(papers_dict)}/{len(valid_ids)} fetched so far). '
                        f'This may indicate rate limiting. Error: {e}'
                    )
                else:
                    logger.error(
                        f'✗ FAILED: Error while iterating results ({len(papers_dict)}/{len(valid_ids)} fetched so far). '
                        f'Error type: {error_type}, Error: {e}'
                    )
            
            # Return partial results if we got some papers before the error
            if papers_dict:
                return papers_dict
            return {}

        if logger:
            fetched_count = len(papers_dict)
            if fetched_count == len(valid_ids):
                logger.info(
                    f'✓ SUCCESS: Fetched all {fetched_count}/{len(valid_ids)} papers in batch'
                )
            elif fetched_count > 0:
                missing = set(valid_ids) - set(papers_dict.keys())
                logger.warning(
                    f'⚠ PARTIAL SUCCESS: Fetched {fetched_count}/{len(valid_ids)} papers. '
                    f'Missing: {missing}'
                )
            else:
                logger.error(
                    f'✗ FAILED: No papers fetched from batch request for {len(valid_ids)} paper(s). '
                    f'Iterator completed but returned no results. This may indicate: '
                    f'1) Invalid paper IDs, 2) Rate limiting, 3) Network issues, or 4) API unavailability.'
                )

        return papers_dict
    except Exception as e:
        # Check if it's a rate limit error (HTTP 429)
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        # Handle rate limiting (HTTP 429) more gracefully
        if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            if logger:
                logger.error(
                    f'✗ FAILED: Rate limited by arXiv API for batch request ({len(valid_ids)} papers). '
                    f'Please wait before making more requests. Error: {e}'
                )
        # Handle UnexpectedEmptyPage (which can occur with rate limits)
        elif error_type == 'UnexpectedEmptyPage':
            if logger:
                logger.error(
                    f'✗ FAILED: Unexpected empty response from arXiv API for batch request ({len(valid_ids)} papers). '
                    f'This may indicate rate limiting or temporary unavailability. Error: {e}'
                )
        else:
            if logger:
                logger.error(
                    f'✗ FAILED: Error fetching papers from arXiv API (batch of {len(valid_ids)} papers). '
                    f'Error type: {error_type}, Error: {e}'
                )
        return {}


def fetch_paper_from_api(
    paper_id: Union[str, List[str]], logger: Optional[logging.Logger] = None
) -> Union[Optional[Any], Dict[str, Any]]:
    """
    Fetch arXiv paper object(s) from API.
    
    Supports both single paper ID (string) and batch requests (list of IDs).
    For batch requests, use fetch_papers_from_api_batch() for better control.

    Args:
        paper_id: arXiv paper ID (e.g., "2207.01510") or list of IDs for batch request
        logger: Optional logger for logging

    Returns:
        - If paper_id is a string: arxiv.Result object if found, None otherwise
        - If paper_id is a list: Dictionary mapping paper_id to arxiv.Result object
    """
    arxiv = _get_arxiv_module()
    if not arxiv:
        if logger:
            logger.warning(
                'arxiv library not available. Install with: uv pip install arxiv'
            )
        return None if isinstance(paper_id, str) else {}

    if not paper_id:
        return None if isinstance(paper_id, str) else {}

    # Handle batch request
    if isinstance(paper_id, list):
        return fetch_papers_from_api_batch(paper_id, logger)

    # Handle single paper request
    try:
        # Use the new Client API instead of deprecated Search.results()
        # Configure with longer delays to prevent rate limiting
        client = arxiv.Client(
            page_size=100,
            delay_seconds=5.0,  # Increased from default 3.0
            num_retries=5,  # Increased from default 3
        )
        search = arxiv.Search(id_list=[paper_id])
        results = client.results(search)
        paper = next(results, None)

        if not paper:
            if logger:
                logger.debug(f'No paper found for arXiv ID: {paper_id}')
            return None

        return paper
    except Exception as e:
        # Check if it's a rate limit error (HTTP 429)
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        # Handle rate limiting (HTTP 429) more gracefully
        if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            if logger:
                logger.warning(
                    f'Rate limited by arXiv API for {paper_id}. '
                    f'Please wait before making more requests. Error: {e}'
                )
        # Handle UnexpectedEmptyPage (which can occur with rate limits)
        elif error_type == 'UnexpectedEmptyPage':
            if logger:
                logger.warning(
                    f'Unexpected empty response from arXiv API for {paper_id}. '
                    f'This may indicate rate limiting or temporary unavailability. Error: {e}'
                )
        else:
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
