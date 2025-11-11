"""High-level arXiv utility functions combining ID extraction and API access."""

import logging
from typing import Any, Dict, Optional

from .arxiv_api_client import (
    extract_metadata_from_paper,
    fetch_paper_from_api,
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


# Re-export ID extraction functions for backward compatibility
__all__ = [
    'extract_arxiv_id_from_url',
    'extract_arxiv_id_from_text',
    'get_arxiv_metadata',
    'get_arxiv_submission_date',
    'is_arxiv_available',
]
