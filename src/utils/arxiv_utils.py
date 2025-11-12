"""High-level arXiv utility functions combining ID extraction and API access."""

import logging
from typing import Any, Dict, List, Optional

from .arxiv_api_client import (
    extract_metadata_from_paper,
    fetch_paper_from_api,
    fetch_papers_from_api_batch,
    is_arxiv_available,
)
from .arxiv_id_extractor import (
    extract_arxiv_id_from_text,
    extract_arxiv_id_from_url,
)


def get_arxiv_metadata(
    paper_id: str,
    logger: Optional[logging.Logger] = None,
    download_pdf: bool = False,
    tool_belt: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch arXiv paper metadata from arXiv API using paper ID.
    Downloads full content (PDF) if requested.

    Args:
        paper_id: arXiv paper ID (e.g., "2207.01510")
        logger: Optional logger for logging
        download_pdf: If True, download the PDF content. Requires tool_belt.
        tool_belt: Optional ToolBelt instance for downloading PDFs.

    Returns:
        Dictionary with keys:
        - paper_id: arXiv paper ID
        - entry_id: Full arXiv entry ID
        - submission_date: Submission date (published) in YYYY-MM-DD format
        - submission_date_text: Human-readable submission date
        - submission_month: Month in YYYY-MM format
        - updated_date: Last updated date in YYYY-MM-DD format
        - updated_date_text: Human-readable updated date
        - title: Paper title
        - authors: List of author names
        - categories: List of arXiv categories
        - primary_category: Primary arXiv category
        - summary: Paper abstract
        - comment: Comments (if available)
        - journal_ref: Journal reference (if available)
        - doi: DOI (if available)
        - pdf_url: URL to PDF
        - pdf_attachment: Attachment object if PDF was downloaded (only if download_pdf=True)
        - confidence: Confidence score (1.0 for API results)
        None if fetch fails or arxiv library not available
    """
    if not is_arxiv_available():
        if logger:
            logger.warning(
                'arxiv library not available. Install with: uv pip install arxiv'
            )
        return None

    if not paper_id:
        return None

    # Fetch paper from API
    paper = fetch_paper_from_api(paper_id, logger)
    if not paper:
        return None

    # Extract metadata (pass paper_id for PDF URL construction)
    metadata = extract_metadata_from_paper(paper, paper_id=paper_id)
    metadata['paper_id'] = paper_id
    metadata['confidence'] = 1.0  # High confidence from API

    # Download PDF if requested
    pdf_attachment = None
    if download_pdf and metadata.get('pdf_url'):
        if tool_belt:
            try:
                if logger:
                    logger.info(
                        f'Downloading PDF from arXiv for paper {paper_id}: {metadata["pdf_url"]}'
                    )
                pdf_attachment = tool_belt.download_file_from_url(metadata['pdf_url'])
                if logger:
                    logger.info(
                        f'Successfully downloaded PDF: {pdf_attachment.filename} '
                        f'({len(pdf_attachment.data)} bytes)'
                    )
            except Exception as e:
                if logger:
                    logger.warning(f'Failed to download PDF for paper {paper_id}: {e}')
        else:
            if logger:
                logger.warning(
                    'download_pdf=True but tool_belt not provided. '
                    'Skipping PDF download.'
                )

    # Include PDF attachment if downloaded
    if pdf_attachment:
        metadata['pdf_attachment'] = pdf_attachment

    if logger:
        logger.debug(
            f'Fetched arXiv metadata from API for {paper_id}: '
            f'submission_date={metadata.get("submission_date")}, '
            f'updated_date={metadata.get("updated_date")}, '
            f'pdf_url={metadata.get("pdf_url")}'
        )

    return metadata


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


def get_arxiv_submission_dates_batch(
    urls: List[str], logger: Optional[logging.Logger] = None, batch_size: int = 5
) -> Dict[str, Optional[str]]:
    """
    Extract arXiv paper IDs from multiple URLs and get submission dates using batch API requests.
    Makes requests in batches of specified size (default: 5) to avoid rate limiting.

    Args:
        urls: List of URLs that may contain arXiv paper IDs
        logger: Optional logger for logging
        batch_size: Number of papers to fetch per batch request (default: 5)

    Returns:
        Dictionary mapping URL to submission date (YYYY-MM-DD format).
        Only includes URLs that contain valid arXiv IDs and were successfully fetched.
        Example: {"https://arxiv.org/abs/2207.01510": "2022-07-04", ...}
    """
    if not urls:
        return {}

    # Extract all arXiv IDs from URLs
    url_to_paper_id = {}
    paper_ids = []
    for url in urls:
        paper_id = extract_arxiv_id_from_url(url)
        if paper_id:
            url_to_paper_id[url] = paper_id
            if paper_id not in paper_ids:
                paper_ids.append(paper_id)

    if not paper_ids:
        return {}

    # Split paper_ids into chunks of batch_size
    papers_dict = {}
    total_batches = (len(paper_ids) + batch_size - 1) // batch_size

    if logger:
        logger.debug(
            f'Fetching {len(paper_ids)} arXiv papers in {total_batches} batch(es) of {batch_size}'
        )

    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        if logger:
            logger.info(
                f'[Batch {batch_num}/{total_batches}] Starting request for {len(batch)} paper(s): {batch}'
            )

        # Add delay between batches to prevent rate limiting
        if i > 0:  # Don't delay before first batch
            import time

            delay = 3.0  # 3 seconds between batches
            if logger:
                logger.debug(
                    f'Waiting {delay} seconds before next batch to avoid rate limiting...'
                )
            time.sleep(delay)

        # Fetch this batch
        batch_results = fetch_papers_from_api_batch(batch, logger)
        papers_dict.update(batch_results)

        if logger:
            success_count = len(batch_results)
            if success_count == len(batch):
                logger.info(
                    f'[Batch {batch_num}/{total_batches}] ✓ SUCCESS: {success_count}/{len(batch)} papers fetched'
                )
            elif success_count > 0:
                logger.warning(
                    f'[Batch {batch_num}/{total_batches}] ⚠ PARTIAL: {success_count}/{len(batch)} papers fetched'
                )
            else:
                logger.error(
                    f'[Batch {batch_num}/{total_batches}] ✗ FAILED: 0/{len(batch)} papers fetched'
                )

    # Build result dictionary mapping URL to submission date
    result = {}
    for url, paper_id in url_to_paper_id.items():
        if paper_id in papers_dict:
            paper = papers_dict[paper_id]
            metadata = extract_metadata_from_paper(paper, paper_id=paper_id)
            submission_date = metadata.get('submission_date')
            if submission_date:
                result[url] = submission_date

    if logger:
        success_rate = (
            (len(result) / len(url_to_paper_id) * 100) if url_to_paper_id else 0
        )
        if len(result) == len(url_to_paper_id):
            logger.info(
                f'✓ OVERALL SUCCESS: Fetched submission dates for {len(result)}/{len(url_to_paper_id)} arXiv URLs '
                f'({success_rate:.1f}% success rate, {total_batches} batch request(s))'
            )
        elif len(result) > 0:
            logger.warning(
                f'⚠ OVERALL PARTIAL: Fetched submission dates for {len(result)}/{len(url_to_paper_id)} arXiv URLs '
                f'({success_rate:.1f}% success rate, {total_batches} batch request(s))'
            )
        else:
            logger.error(
                f'✗ OVERALL FAILED: No submission dates fetched for {len(url_to_paper_id)} arXiv URLs '
                f'({total_batches} batch request(s))'
            )

    return result


# Re-export ID extraction functions for backward compatibility
__all__ = [
    'extract_arxiv_id_from_url',
    'extract_arxiv_id_from_text',
    'fetch_papers_from_api_batch',
    'get_arxiv_metadata',
    'get_arxiv_submission_date',
    'get_arxiv_submission_dates_batch',
    'is_arxiv_available',
]
