"""PDF Download Detector - Identifies when a subtask should download a PDF."""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple


class PDFDownloadDetector:
    """Detects when a subtask description indicates a PDF should be downloaded."""

    def __init__(self, llm_service: Any, logger: logging.Logger):
        """
        Initialize PDF Download Detector.

        Args:
            llm_service: LLM service for intelligent detection.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger

        # Keywords that indicate PDF download intent
        self.pdf_keywords = [
            'download pdf',
            'retrieve pdf',
            'get pdf',
            'fetch pdf',
            'obtain pdf',
            'pdf download',
            'pdf file',
            'full pdf',
            'complete pdf',
            'paper pdf',
            'article pdf',
            'document pdf',
            'download the pdf',
            'retrieve the pdf',
            'get the pdf',
            'fetch the pdf',
            'obtain the pdf',
        ]

        # Patterns for arXiv and other repositories
        self.repository_patterns = [
            r'arxiv\.org',
            r'pubmed',
            r'doi\.org',
            r'researchgate',
            r'academia\.edu',
            r'scholar\.google',
        ]

    def should_download_pdf(
        self,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a subtask should download a PDF.

        Args:
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Tuple of (should_download: bool, reasoning: Optional[str]).
        """
        # First, check for explicit PDF download keywords
        description_lower = subtask_description.lower()
        for keyword in self.pdf_keywords:
            if keyword in description_lower:
                self.logger.info(
                    f'PDF download keyword detected: "{keyword}" in subtask description'
                )
                return True, f'Subtask explicitly mentions "{keyword}"'

        # Check if subtask mentions retrieving/downloading a document/paper/article
        retrieval_keywords = [
            'retrieve',
            'download',
            'get',
            'fetch',
            'obtain',
            'access',
            'locate',
        ]
        document_keywords = [
            'paper',
            'article',
            'document',
            'publication',
            'preprint',
            'manuscript',
        ]

        has_retrieval = any(
            keyword in description_lower for keyword in retrieval_keywords
        )
        has_document = any(
            keyword in description_lower for keyword in document_keywords
        )

        if has_retrieval and has_document:
            self.logger.info(
                'PDF download likely needed: subtask mentions retrieving a document/paper/article'
            )
            return True, 'Subtask mentions retrieving a document/paper/article'

        # Check for repository patterns in problem or subtask
        combined_text = f'{problem} {subtask_description}'
        for pattern in self.repository_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                self.logger.info(
                    f'Repository pattern detected: {pattern} - PDF download likely needed'
                )
                return (
                    True,
                    f'Repository pattern "{pattern}" detected in problem/subtask',
                )

        # Use LLM for intelligent detection if available
        if self.llm_service:
            return self._llm_detect_pdf_download(
                subtask_description, problem, query_analysis
            )

        return False, None

    def _llm_detect_pdf_download(
        self,
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Use LLM to intelligently detect if PDF download is needed.

        Args:
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Tuple of (should_download: bool, reasoning: Optional[str]).
        """
        try:
            system_prompt = """You are an expert at analyzing subtasks to determine if they require downloading a PDF file.

A subtask requires PDF download if it:
- Explicitly mentions downloading, retrieving, getting, or fetching a PDF
- Mentions retrieving a paper, article, document, or publication (which are typically PDFs)
- References a specific paper/article by ID, title, or citation that needs to be accessed
- Mentions accessing content from repositories like arXiv, PubMed, or academic databases
- Requires reading the full text of a document (not just a summary or abstract)

A subtask does NOT require PDF download if it:
- Only asks to search for information (without retrieving the document)
- Only asks to find a URL or link
- Only asks to summarize or analyze already-available content
- Is about web browsing or navigation without document retrieval

Return a JSON object with:
- should_download: boolean indicating if PDF download is needed
- reasoning: brief explanation (1-2 sentences) of why or why not

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )

            user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

Does this subtask require downloading a PDF file? Analyze the subtask description and determine if it explicitly or implicitly requires retrieving a PDF document."""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent detection
                response_format={'type': 'json_object'},
            )

            from ..utils.json_utils import extract_json_from_text

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            should_download = result_data.get('should_download', False)
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            if should_download:
                self.logger.info(f'LLM detected PDF download needed: {reasoning}')
            else:
                self.logger.debug(
                    f'LLM determined PDF download not needed: {reasoning}'
                )

            return should_download, reasoning

        except Exception as e:
            self.logger.warning(
                f'LLM PDF detection failed: {e}. Falling back to keyword-based detection.'
            )
            return False, None

    def extract_pdf_url_from_result(
        self, search_result: Any, subtask_description: str
    ) -> Optional[str]:
        """
        Extract PDF URL from a search result.

        Args:
            search_result: SearchResult object or dict.
            subtask_description: Description of the subtask.

        Returns:
            PDF URL if found, None otherwise.
        """
        # Get URL from search result
        if hasattr(search_result, 'url'):
            url = search_result.url
        elif isinstance(search_result, dict):
            url = search_result.get('url', '')
        else:
            return None

        if not url:
            return None

        url_lower = url.lower()

        # Check if URL is already a PDF
        if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
            return url

        # Check for arXiv patterns
        arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/([\d.]+)', url, re.IGNORECASE)
        if arxiv_match:
            paper_id = arxiv_match.group(1)
            # Convert to PDF URL
            pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
            self.logger.info(f'Converted arXiv URL to PDF: {url} -> {pdf_url}')
            return pdf_url

        # Check for DOI patterns
        doi_match = re.search(r'doi\.org/([^\s]+)', url, re.IGNORECASE)
        if doi_match:
            # DOI URLs often need special handling, but we can try
            self.logger.debug(f'DOI detected: {url} - may need special handling')

        return None
