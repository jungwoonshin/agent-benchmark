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

        # Extract submission date (published date)
        submission_date = None
        submission_month = None
        submission_date_text = None

        if paper.published:
            submission_date = paper.published.strftime('%Y-%m-%d')
            submission_month = paper.published.strftime('%Y-%m')
            submission_date_text = (
                f'Submitted on {paper.published.strftime("%B %d, %Y")}'
            )

        # Extract updated date
        updated_date = None
        updated_date_text = None
        if paper.updated:
            updated_date = paper.updated.strftime('%Y-%m-%d')
            updated_date_text = (
                f'Updated on {paper.updated.strftime("%B %d, %Y")}'
            )

        # Get PDF URL
        pdf_url = paper.pdf_url if hasattr(paper, 'pdf_url') else None
        if not pdf_url:
            # Construct PDF URL from paper ID
            pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'

        # Download PDF if requested
        pdf_attachment = None
        if download_pdf and pdf_url:
            if tool_belt:
                try:
                    if logger:
                        logger.info(f'Downloading PDF from arXiv for paper {paper_id}: {pdf_url}')
                    pdf_attachment = tool_belt.download_file_from_url(pdf_url)
                    if logger:
                        logger.info(
                            f'Successfully downloaded PDF: {pdf_attachment.filename} '
                            f'({len(pdf_attachment.data)} bytes)'
                        )
                except Exception as e:
                    if logger:
                        logger.warning(
                            f'Failed to download PDF for paper {paper_id}: {e}'
                        )
            else:
                if logger:
                    logger.warning(
                        'download_pdf=True but tool_belt not provided. '
                        'Skipping PDF download.'
                    )

        result = {
            'paper_id': paper_id,
            'entry_id': paper.entry_id if hasattr(paper, 'entry_id') else None,
            'submission_date': submission_date,
            'submission_date_text': submission_date_text,
            'submission_month': submission_month,
            'updated_date': updated_date,
            'updated_date_text': updated_date_text,
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
                paper.comment 
                if hasattr(paper, 'comment') and paper.comment 
                else None
            ),
            'journal_ref': (
                paper.journal_ref 
                if hasattr(paper, 'journal_ref') and paper.journal_ref 
                else None
            ),
            'doi': (
                paper.doi 
                if hasattr(paper, 'doi') and paper.doi 
                else None
            ),
            'pdf_url': pdf_url,
            'confidence': 1.0,  # High confidence from API
        }

        # Include PDF attachment if downloaded
        if pdf_attachment:
            result['pdf_attachment'] = pdf_attachment

        if logger:
            logger.debug(
                f'Fetched arXiv metadata from API for {paper_id}: '
                f'submission_date={submission_date}, '
                f'updated_date={updated_date}, '
                f'pdf_url={pdf_url}'
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
