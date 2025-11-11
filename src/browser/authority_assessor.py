"""Authority assessment for search results without hardcoding sources."""

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..models import SearchResult


class AuthorityAssessor:
    """Assesses source authority using multiple signals without hardcoding."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize AuthorityAssessor."""
        self.logger = logger or logging.getLogger(__name__)

    def extract_authority_signals(self, result: SearchResult) -> Dict[str, any]:
        """
        Extract authority signals from a search result.

        Returns dict with:
        - content_type_quality: 'full_document', 'abstract', 'snippet', 'unknown'
        - domain_indicators: list of authority indicators from domain
        - publication_indicators: list of publication-type indicators
        - metadata_quality: 'high', 'medium', 'low'
        - authority_score: 0.0-1.0 composite score
        """
        signals = {
            'content_type_quality': self._assess_content_type_quality(result),
            'domain_indicators': self._extract_domain_indicators(result.url),
            'publication_indicators': self._extract_publication_indicators(result),
            'metadata_quality': self._assess_metadata_quality(result),
            'url_structure_quality': self._assess_url_structure(result.url),
        }

        signals['authority_score'] = self._calculate_authority_score(signals)

        return signals

    def _assess_content_type_quality(self, result: SearchResult) -> str:
        """Assess content type quality from URL and title."""
        url_lower = result.url.lower()
        title_lower = result.title.lower() if result.title else ''

        # Full documents are highest quality
        if any(ext in url_lower for ext in ['.pdf', '.doc', '.docx']):
            return 'full_document'

        # Journal/article pages often have full content
        if any(
            indicator in url_lower
            for indicator in [
                '/article/',
                '/paper/',
                '/publication/',
                '/pub/',
                'doi.org',
                'pubmed',
                'pmc',
                'arxiv.org/abs/',
            ]
        ):
            return 'full_document'

        # Abstracts are medium quality
        if 'abstract' in title_lower or 'abstract' in url_lower:
            return 'abstract'

        # Snippets are lowest quality
        return 'snippet'

    def _extract_domain_indicators(self, url: str) -> List[str]:
        """Extract authority indicators from domain without hardcoding specific domains."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            indicators = []

            # TLD-based indicators
            if domain.endswith('.edu'):
                indicators.append('academic_institution')
            elif domain.endswith('.gov'):
                indicators.append('government')
            elif domain.endswith('.org'):
                indicators.append('organization')

            # Academic/research patterns
            academic_patterns = [
                r'university',
                r'\.edu',
                r'ac\.uk',
                r'\.ac\.',
                r'research',
                r'scholar',
                r'academic',
            ]
            if any(re.search(pattern, domain) for pattern in academic_patterns):
                indicators.append('academic_pattern')

            # Publication platform patterns
            pub_patterns = [
                r'journal',
                r'publisher',
                r'press',
                r'publication',
                r'doi',
                r'pubmed',
                r'pmc',
                r'arxiv',
                r'science',
            ]
            if any(re.search(pattern, domain) for pattern in pub_patterns):
                indicators.append('publication_platform')

            # Official/primary source patterns
            official_patterns = [r'official', r'primary', r'source', r'original']
            if any(re.search(pattern, domain) for pattern in official_patterns):
                indicators.append('official_source')

            return indicators
        except Exception:
            return []

    def _extract_publication_indicators(self, result: SearchResult) -> List[str]:
        """Extract publication-type indicators from title and snippet."""
        indicators = []
        text = f'{result.title} {result.snippet}'.lower()

        # Peer-reviewed indicators
        if any(
            term in text
            for term in [
                'peer-reviewed',
                'peer reviewed',
                'journal article',
                'research article',
                'scientific paper',
                'published in',
            ]
        ):
            indicators.append('peer_reviewed')

        # Conference/workshop indicators
        if any(
            term in text
            for term in ['conference', 'proceedings', 'workshop', 'symposium']
        ):
            indicators.append('conference_publication')

        # Preprint indicators
        if any(term in text for term in ['preprint', 'arxiv', 'biorxiv', 'medrxiv']):
            indicators.append('preprint')

        # DOI presence indicates formal publication
        if 'doi:' in text or 'doi.org' in result.url.lower():
            indicators.append('has_doi')

        return indicators

    def _assess_metadata_quality(self, result: SearchResult) -> str:
        """Assess metadata quality from available information."""
        snippet = result.snippet or ''
        has_date = bool(re.search(r'\d{4}', snippet))
        has_author = bool(re.search(r'author|by\s+[A-Z]', snippet, re.I))
        has_journal = any(
            term in snippet.lower()
            for term in ['journal', 'published in', 'volume', 'issue']
        )

        if has_date and (has_author or has_journal):
            return 'high'
        elif has_date or has_author:
            return 'medium'
        else:
            return 'low'

    def _assess_url_structure(self, url: str) -> str:
        """Assess URL structure quality."""
        url_lower = url.lower()

        # Direct document links are better
        if url_lower.endswith(('.pdf', '.doc', '.docx')):
            return 'direct_document'

        # Stable/permanent URLs are better
        if any(
            indicator in url_lower
            for indicator in ['doi.org', 'handle.net', 'persistent', 'stable']
        ):
            return 'stable_url'

        # Article/paper URLs are good
        if any(
            indicator in url_lower
            for indicator in ['/article/', '/paper/', '/pub/', '/publication/']
        ):
            return 'structured_url'

        return 'generic_url'

    def _calculate_authority_score(self, signals: Dict) -> float:
        """Calculate composite authority score from signals."""
        score = 0.0

        # Content type quality (40% weight)
        content_scores = {
            'full_document': 1.0,
            'abstract': 0.6,
            'snippet': 0.3,
            'unknown': 0.2,
        }
        score += content_scores.get(signals['content_type_quality'], 0.2) * 0.4

        # Domain indicators (25% weight)
        domain_score = 0.0
        if 'academic_institution' in signals['domain_indicators']:
            domain_score = max(domain_score, 0.9)
        if 'government' in signals['domain_indicators']:
            domain_score = max(domain_score, 0.85)
        if 'publication_platform' in signals['domain_indicators']:
            domain_score = max(domain_score, 0.8)
        if 'academic_pattern' in signals['domain_indicators']:
            domain_score = max(domain_score, 0.75)
        if 'official_source' in signals['domain_indicators']:
            domain_score = max(domain_score, 0.7)
        if not domain_score:  # No indicators found
            domain_score = 0.5
        score += domain_score * 0.25

        # Publication indicators (20% weight)
        pub_score = 0.5  # Base score
        if 'peer_reviewed' in signals['publication_indicators']:
            pub_score = 0.95
        elif 'has_doi' in signals['publication_indicators']:
            pub_score = 0.85
        elif 'conference_publication' in signals['publication_indicators']:
            pub_score = 0.75
        elif 'preprint' in signals['publication_indicators']:
            pub_score = 0.65
        score += pub_score * 0.2

        # Metadata quality (10% weight)
        metadata_scores = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        score += metadata_scores.get(signals['metadata_quality'], 0.3) * 0.1

        # URL structure (5% weight)
        url_scores = {
            'direct_document': 1.0,
            'stable_url': 0.9,
            'structured_url': 0.7,
            'generic_url': 0.5,
        }
        score += url_scores.get(signals['url_structure_quality'], 0.5) * 0.05

        return min(1.0, max(0.0, score))
