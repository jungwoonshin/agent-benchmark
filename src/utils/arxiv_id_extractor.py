"""Pure regex-based arXiv paper ID extraction (no external dependencies)."""

import re
from typing import Optional


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

    # Match arxiv.org/abs/ID, arxiv.org/pdf/ID, arxiv.org/html/ID, etc.
    # Supports version suffixes (v1, v2) and .pdf extension
    arxiv_match = re.search(
        r'arxiv\.org/(?:abs|pdf|html)/([\d.]+?)(?:v\d+)?(?:\.pdf)?(?:\?|$|#|/)',
        url,
        re.IGNORECASE,
    )
    if arxiv_match:
        return arxiv_match.group(1)

    # Also handle external sites that reference arXiv papers
    # e.g., ui.adsabs.harvard.edu/abs/arXiv:2206.07506
    external_match = re.search(r'arXiv[:\s]+([\d]{4}\.[\d]{5})', url, re.IGNORECASE)
    if external_match:
        paper_id = external_match.group(1)
        # Remove version suffix if present
        paper_id = re.sub(r'v\d+$', '', paper_id)
        return paper_id

    return None


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


def normalize_arxiv_id(paper_id: str) -> Optional[str]:
    """
    Normalize arXiv paper ID by removing version suffix.

    Args:
        paper_id: arXiv paper ID that may include version suffix (e.g., "2207.01510v1")

    Returns:
        Normalized arXiv paper ID without version suffix, or None if invalid format
    """
    if not paper_id:
        return None

    # Remove version suffix if present
    normalized = re.sub(r'v\d+$', '', paper_id)

    # Validate format (should be YYMM.XXXXX)
    if re.match(r'^\d{4}\.\d{5}$', normalized):
        return normalized

    return None
