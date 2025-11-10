"""Relevance checking utilities for search results."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..utils import extract_json_from_text

if TYPE_CHECKING:
    from src.tools import ToolBelt

    from ..llm import LLMService
    from ..models import Attachment, SearchResult


def get_content_type_label(content_type: Optional[str]) -> str:
    """Get human-readable label for content type."""
    if content_type == 'pdf':
        return 'PDF'
    elif content_type == 'web_page':
        return 'Web Page'
    else:
        return 'File'


def build_requirements_context(query_analysis: Optional[Dict[str, Any]]) -> str:
    """Build requirements context string from query analysis."""
    if not query_analysis:
        return ''

    explicit_reqs = query_analysis.get('explicit_requirements', [])
    if explicit_reqs:
        return f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
    return ''


def build_section_titles_info(
    section_titles: Optional[List[str]], content_type: Optional[str]
) -> str:
    """Build section titles information string."""
    if not section_titles:
        return ''

    content_type_label = get_content_type_label(content_type)
    titles_str = '\n'.join(f'- {title}' for title in section_titles[:20])
    return f'\n\n{content_type_label} Section Titles:\n{titles_str}'


def build_arxiv_metadata_info(
    metadata: Optional[Dict[str, Any]],
    content_type: Optional[str],
    logger: logging.Logger,
) -> str:
    """Build arXiv metadata information string.

    Always returns a metadata section for PDFs (even if metadata is None) to ensure
    submission dates appear before content_info in the prompt.
    For non-PDFs, returns empty string.
    """
    # Only show metadata section for PDFs
    if content_type != 'pdf':
        return ''

    info = '\n\narXiv Metadata (extracted from PDF):\n'

    if not metadata:
        # Still show the section even if metadata extraction failed or wasn't attempted
        # This ensures the section appears before content_info for PDFs
        info += '- Submission Date: Not extracted (metadata extraction not available or failed)\n'
        logger.debug(
            'No arXiv metadata available for PDF, but showing metadata section in prompt'
        )
        return info

    logger.info(
        f'Extracted arXiv metadata: paper_id={metadata.get("paper_id")}, '
        f'submission_date={metadata.get("submission_date")}, '
        f'updated_date={metadata.get("updated_date")}, '
        f'submission_month={metadata.get("submission_month")}'
    )

    # Paper ID
    if metadata.get('paper_id'):
        info += f'- Paper ID: {metadata["paper_id"]}\n'
    
    # Entry ID
    if metadata.get('entry_id'):
        info += f'- Entry ID: {metadata["entry_id"]}\n'

    # Title
    if metadata.get('title'):
        info += f'- Title: {metadata["title"]}\n'

    # Authors
    authors = metadata.get('authors')
    if authors and isinstance(authors, list) and len(authors) > 0:
        authors_str = ', '.join(authors[:5])  # Show first 5 authors
        if len(authors) > 5:
            authors_str += f' (and {len(authors) - 5} more)'
        info += f'- Authors: {authors_str}\n'

    # Submission date (published date)
    submission_date = metadata.get('submission_date')
    submission_month = metadata.get('submission_month')
    submission_date_text = metadata.get('submission_date_text')

    if submission_date:
        info += f'- Submission Date: {submission_date}'
        if submission_date_text:
            info += f' ({submission_date_text})'
        info += '\n'
    elif submission_month:
        info += f'- Submission Month: {submission_month}\n'
    elif submission_date_text:
        info += f'- Submission Date Text: {submission_date_text}\n'
    else:
        # Still show the section even if no date found, so it appears before content_info
        info += '- Submission Date: Not found in PDF metadata\n'
        logger.warning(
            'arXiv metadata extracted but no submission date information found'
        )

    # Updated date
    updated_date = metadata.get('updated_date')
    updated_date_text = metadata.get('updated_date_text')
    if updated_date:
        info += f'- Updated Date: {updated_date}'
        if updated_date_text:
            info += f' ({updated_date_text})'
        info += '\n'

    # Categories
    categories = metadata.get('categories')
    if categories and isinstance(categories, list) and len(categories) > 0:
        info += f'- Categories: {", ".join(categories)}\n'
    
    # Primary category
    if metadata.get('primary_category'):
        info += f'- Primary Category: {metadata["primary_category"]}\n'

    # Summary/Abstract
    if metadata.get('summary'):
        summary = metadata['summary']
        # Truncate if too long
        if len(summary) > 500:
            summary = summary[:500] + '...'
        info += f'- Abstract: {summary}\n'

    # Journal reference
    if metadata.get('journal_ref'):
        info += f'- Journal Reference: {metadata["journal_ref"]}\n'

    # DOI
    if metadata.get('doi'):
        info += f'- DOI: {metadata["doi"]}\n'

    # Comment
    if metadata.get('comment'):
        info += f'- Comment: {metadata["comment"]}\n'

    # PDF URL
    if metadata.get('pdf_url'):
        info += f'- PDF URL: {metadata["pdf_url"]}\n'

    return info


def build_image_analysis_info(
    image_analysis: Optional[str], logger: logging.Logger
) -> str:
    """Build image analysis information string."""
    if not image_analysis:
        return ''

    logger.info(
        f'Including visual LLM analysis ({len(image_analysis)} chars) in relevance check'
    )
    return (
        '\n\nIMAGE ANALYSIS (from visual LLM - extracted before relevance check):\n'
        + image_analysis
    )


def build_full_content_info(
    full_content: Optional[str], content_type: Optional[str], max_length: int = 10000
) -> str:
    """Build full content information string with truncation."""
    if not full_content:
        return ''

    content_type_label = get_content_type_label(content_type)
    content_preview = full_content

    if len(full_content) > max_length:
        content_preview = (
            full_content[:max_length]
            + f'\n\n[... Content truncated: {len(full_content)} total characters ...]'
        )

    return f'\n\n{content_type_label} Full Content:\n{content_preview}'


def build_content_info(
    section_titles: Optional[List[str]],
    image_analysis: Optional[str],
    full_content: Optional[str],
    content_type: Optional[str],
    logger: logging.Logger,
) -> str:
    """Build complete content information string."""
    content_info = ''

    if section_titles:
        content_info += build_section_titles_info(section_titles, content_type)

    if image_analysis:
        content_info += build_image_analysis_info(image_analysis, logger)

    if full_content:
        content_info += build_full_content_info(full_content, content_type)

    return content_info


def build_explicit_requirements_check(
    query_analysis: Optional[Dict[str, Any]], step_num: Optional[int] = None
) -> str:
    """Build explicit requirements check section.

    Args:
        query_analysis: Optional query analysis results.
        step_num: Optional step number to filter requirements. If provided, only
                  requirements matching this step number will be included.

    Returns:
        String with explicit requirements check section, or empty string if no
        matching requirements found.
    """
    explicit_reqs = (
        query_analysis.get('explicit_requirements', []) if query_analysis else []
    )

    if not explicit_reqs:
        return ''

    # Filter requirements by step number if step number is provided
    if step_num is not None:
        filtered_reqs = []
        for req in explicit_reqs:
            req_str = str(req)
            # Check if requirement matches this step number
            req_step_match = re.search(r'step[_\s]*(\d+)[:\s]', req_str, re.IGNORECASE)
            if req_step_match:
                try:
                    req_step_num = int(req_step_match.group(1))
                    if req_step_num == step_num:
                        filtered_reqs.append(req)
                except (ValueError, IndexError):
                    # If we can't parse step number, don't include it
                    pass
            # If requirement doesn't have step tag, don't include it
            # (only include step-tagged requirements when filtering by step)

        explicit_reqs = filtered_reqs
        if not explicit_reqs:
            return ''

    check = '\n\nCRITICAL: Explicit Requirements Check:\n'
    check += 'The following explicit requirements are provided:\n'
    for idx, req in enumerate(explicit_reqs, 1):
        check += f'{idx}. {req}\n'
    check += (
        '\nIMPORTANT: Requirements may refer to different entities, documents, or sources. '
        'A search result is relevant if it satisfies ALL requirements that refer to the SAME entity/source. '
        'You do NOT need to satisfy requirements that refer to different entities - only the requirements for the entity this result is about.'
    )
    return check


def get_system_prompt() -> str:
    """Get the system prompt for relevance checking."""
    return """You evaluate whether a search result is useful for completing a subtask.
Given the problem, subtask, decide if the page likely helps complete the subtask.
Full content of the search result is provided to help you make an accurate determination.
For PDFs with images, visual LLM analysis of the images is also provided to help determine relevance.

CRITICAL RULE: If explicit requirements are provided, identify which entity/source/document this search result is about, then check if ALL requirements for that specific entity are satisfied.

A result is relevant if it satisfies ALL requirements for ONE complete entity group. You do NOT need to satisfy requirements for other entities.

Criteria:
- Match the subtask intent and problem requirements
- Source appears trustworthy and offers actionable information
- Use the full content provided to verify if the result actually contains the information needed
- For PDFs: If image analysis from visual LLM is provided, use it to understand figures, charts, diagrams, and visual information that may be critical for relevance
- For date checks: use dates only from the content_info provided, and treat dates within ±1 month of the required date as acceptable
- Do not use dates from the title/snippet/url/id to determine relevance
- Section titles help indicate what topics are covered in the page/document
- For explicit requirements: Identify which entity/source this result is about, then verify that ALL requirements for that entity are satisfied based on the full content and image analysis (if provided). Requirements for other entities do not need to be satisfied.

Return JSON only with:
{"relevant": boolean, "reasoning": "1-2 sentences explaining why it is/isn't relevant, including which explicit requirements are/aren't satisfied", "confidence": 0.0-1.0, "explicit_requirements_satisfied": [list of requirement indices (1-based) that are satisfied, or empty array if no explicit requirements]}

If explicit requirements exist, your reasoning MUST explicitly state which requirements are satisfied and which are not."""


def build_user_prompt(
    subtask_description: str,
    requirements_context: str,
    explicit_reqs_check: str,
    arxiv_metadata_info: str,
    content_info: str,
) -> str:
    """Build the user prompt for relevance checking."""
    # Ensure submission dates appear before content_info
    # The arxiv_metadata_info already includes submission dates if available
    # We ensure it's always placed before content_info
    metadata_section = arxiv_metadata_info if arxiv_metadata_info else ''

    return f"""Subtask: {subtask_description}
{requirements_context}{explicit_reqs_check}

Search Result:
{metadata_section}
- content_info: {content_info}

Is this search result relevant to completing the subtask?"""


def parse_relevance_response(
    response: str, explicit_reqs: List[str], logger: logging.Logger
) -> Tuple[bool, str]:
    """Parse LLM response and return relevance result."""
    json_text = extract_json_from_text(response)
    result_data = json.loads(json_text)

    is_relevant = result_data.get('relevant', False)
    reasoning = result_data.get('reasoning', 'No reasoning provided')
    confidence = result_data.get('confidence', 0.5)
    explicit_reqs_satisfied = result_data.get('explicit_requirements_satisfied', [])

    # Log explicit requirements satisfaction
    if explicit_reqs:
        if not explicit_reqs_satisfied:
            logger.debug(
                f'No explicit requirements satisfied, but trusting LLM relevance judgment: {is_relevant}'
            )
        else:
            logger.debug(f'Explicit requirements satisfied: {explicit_reqs_satisfied}')

    logger.debug(
        f'Relevance check: {is_relevant} (confidence: {confidence:.2f}). '
        f'Reasoning: {reasoning}'
    )

    return is_relevant, reasoning


def extract_arxiv_metadata_safely(
    tool_belt: 'ToolBelt',
    llm_service: 'LLMService',
    attachment: Optional['Attachment'],
    content_type: Optional[str],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Safely extract arXiv metadata from PDF attachment."""
    # Log why metadata extraction might be skipped
    if content_type != 'pdf':
        logger.debug(
            f'Skipping arXiv metadata extraction: content_type is "{content_type}", not "pdf"'
        )
        return None

    if not attachment:
        logger.debug('Skipping arXiv metadata extraction: no attachment provided')
        return None

    try:
        if not hasattr(tool_belt, 'image_recognition'):
            logger.debug(
                'Skipping arXiv metadata extraction: tool_belt has no image_recognition attribute'
            )
            return None

        if not tool_belt.image_recognition:
            logger.debug(
                'Skipping arXiv metadata extraction: image_recognition is None'
            )
            return None

        # Note: extract_arxiv_metadata_from_pdf now uses arxiv library, not LLM
        # LLM service is no longer required for arXiv metadata extraction
        # (but may still be needed for other image_recognition features)

        logger.debug(
            'Attempting to extract arXiv metadata from PDF attachment using arxiv library'
        )
        # Download PDF and fetch full metadata when checking relevance
        metadata = tool_belt.image_recognition.extract_arxiv_metadata_from_pdf(
            attachment,
            tool_belt=tool_belt,
            download_pdf=True,  # Download full PDF content for relevance checking
        )

        if metadata:
            logger.debug(f'Successfully extracted arXiv metadata: {metadata}')
        else:
            logger.debug('arXiv metadata extraction returned None or empty result')

        return metadata
    except Exception as e:
        logger.warning(
            f'Failed to extract arXiv metadata for relevance check: {e}', exc_info=True
        )
        return None


class RelevanceChecker:
    """Handles relevance checking for search results."""

    def __init__(
        self, llm_service: 'LLMService', tool_belt: 'ToolBelt', logger: logging.Logger
    ):
        """Initialize RelevanceChecker."""
        self.llm_service = llm_service
        self.tool_belt = tool_belt
        self.logger = logger

    def check_relevance(
        self,
        search_result: 'SearchResult',
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
        full_content: Optional[str] = None,
        section_titles: Optional[List[str]] = None,
        content_type: Optional[str] = None,
        attachment: Optional['Attachment'] = None,
        image_analysis: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a search result is relevant using LLM.

        Args:
            search_result: SearchResult to evaluate.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
            full_content: Optional full content of the search result.
            section_titles: Optional list of section titles/headings.
            content_type: Optional type of content ('web_page', 'pdf', 'file').
            attachment: Optional Attachment object (used for PDF metadata extraction).
            image_analysis: Optional visual LLM analysis of images from PDF.

        Returns:
            Tuple of (is_relevant: bool, reasoning: str)
        """
        self.logger.info(
            f'Search result: Title: {search_result.title},\n '
            f'Snippet: {search_result.snippet},\n URL: {search_result.url}'
        )

        try:
            # Log parameters for debugging
            self.logger.info(
                f'Relevance check parameters: content_type={content_type}, '
                f'has_attachment={attachment is not None}, '
                f'has_full_content={full_content is not None}, '
                f'has_section_titles={section_titles is not None and len(section_titles) > 0 if section_titles else False}'
            )

            # Extract step number from subtask_description (look for "step_1", "step_2", etc.)
            step_num = None
            step_match = re.search(
                r'step[_\s]*(\d+)', subtask_description, re.IGNORECASE
            )
            if step_match:
                try:
                    step_num = int(step_match.group(1))
                except (ValueError, IndexError):
                    pass

            # Extract arXiv metadata safely
            metadata = extract_arxiv_metadata_safely(
                self.tool_belt, self.llm_service, attachment, content_type, self.logger
            )

            # Build context components
            requirements_context = build_requirements_context(query_analysis)
            content_info = build_content_info(
                section_titles, image_analysis, full_content, content_type, self.logger
            )
            arxiv_metadata_info = build_arxiv_metadata_info(
                metadata, content_type, self.logger
            )
            explicit_reqs_check = build_explicit_requirements_check(
                query_analysis, step_num
            )

            # Log if arxiv metadata will be included
            if arxiv_metadata_info:
                self.logger.info(
                    f'Including arXiv metadata in relevance check prompt: '
                    f'{len(arxiv_metadata_info)} characters'
                )

            # Build prompts
            system_prompt = get_system_prompt()
            user_prompt = build_user_prompt(
                subtask_description,
                requirements_context,
                explicit_reqs_check,
                arxiv_metadata_info,
                content_info,
            )

            # Call LLM
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            # Parse response
            explicit_reqs = (
                query_analysis.get('explicit_requirements', [])
                if query_analysis
                else []
            )
            return parse_relevance_response(response, explicit_reqs, self.logger)

        except Exception as e:
            self.logger.warning(
                f'Failed to determine relevance using LLM: {e}. '
                f'Defaulting to relevant=True.'
            )
            return True, 'LLM check failed, assuming relevant'

    def check_relevance_batch(
        self,
        search_results: List['SearchResult'],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[bool, str]]:
        """
        Check relevance for multiple search results in a single batch using LLM.
        Uses only title, URL, and snippet (abstract) for each result.

        Args:
            search_results: List of SearchResult objects to evaluate.
            subtask_description: Description of the subtask.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            List of tuples (is_relevant: bool, reasoning: str) for each result, in same order.
        """
        if not search_results:
            return []

        self.logger.info(
            f'Batch relevance check: Evaluating {len(search_results)} search results'
        )

        try:
            # Build requirements context
            requirements_context = build_requirements_context(query_analysis)
            explicit_reqs_check = build_explicit_requirements_check(query_analysis)

            # Format search results for batch evaluation
            results_text = []
            for idx, result in enumerate(search_results, 1):
                result_info = f"""Result {idx}:
- Title: {result.title}
- URL: {result.url}
- Snippet/Abstract: {result.snippet}
"""
                results_text.append(result_info)

            search_results_text = '\n\n'.join(results_text)

            # Build batch system prompt
            batch_system_prompt = """You evaluate whether multiple search results are useful for completing a subtask.
Given the problem and subtask, evaluate each search result based on its title, URL, and snippet/abstract.
You will receive multiple search results and should evaluate each one.

CRITICAL RULE: If explicit requirements are provided, identify which entity/source/document each search result is about, then check if ALL requirements for that specific entity are satisfied.
Requirements may be grouped by different stages (e.g., "step 1:", "step 2:", "step 3:").
A result is relevant if it satisfies ALL requirements for ONE complete entity group. You do NOT need to satisfy requirements for other entities.

Criteria:
- Match the subtask intent and problem requirements
- Source appears trustworthy and offers actionable information
- Use the title, URL, and snippet/abstract to assess relevance
- For aggregate/statistics: index/browse/archive/database pages are relevant even if the snippet lacks the exact number
- For explicit requirements: Identify which entity/source each result is about, then verify that ALL requirements for that entity are satisfied based on the available information

IMPORTANT DATE HANDLING: When evaluating dates (especially for arXiv papers):
- DO NOT use arXiv ID formats (e.g., 2207.01510) to infer submission dates - the arXiv ID format does not reliably indicate the actual submission date
- ONLY use explicit date information from snippets/abstracts, metadata, or explicitly stated submission dates
- If a snippet mentions "submitted on [date]" or "originally submitted [date]", use that date
- If no explicit date is mentioned in the snippet, do not infer dates from arXiv IDs or URLs

Return JSON only with:
{"results": [{"index": 1, "relevant": boolean, "reasoning": "1-2 sentences explaining why it is/isn't relevant", "confidence": 0.0-1.0, "explicit_requirements_satisfied": [list of requirement indices (1-based) that are satisfied, or empty array if no explicit requirements]}, ...]}

The "index" field should match the result number (1-based). Return results in the same order as provided.
If explicit requirements exist, your reasoning MUST explicitly state which requirements are satisfied and which are not."""

            # Build batch user prompt
            batch_user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}{explicit_reqs_check}

Search Results:
{search_results_text}

Evaluate each search result and determine if it is relevant to completing the subtask.
Return a JSON object with a "results" array containing one evaluation per result."""

            # Call LLM
            response = self.llm_service.call_with_system_prompt(
                system_prompt=batch_system_prompt,
                user_prompt=batch_user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            # Parse response
            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            results_list = result_data.get('results', [])
            explicit_reqs = (
                query_analysis.get('explicit_requirements', [])
                if query_analysis
                else []
            )

            # Map results by index
            results_by_index = {r.get('index'): r for r in results_list}

            # Build return list in same order as input
            batch_results = []
            for idx, result in enumerate(search_results, 1):
                result_eval = results_by_index.get(idx, {})
                is_relevant = result_eval.get('relevant', False)
                reasoning = result_eval.get(
                    'reasoning', 'No reasoning provided for this result'
                )
                confidence = result_eval.get('confidence', 0.5)
                explicit_reqs_satisfied = result_eval.get(
                    'explicit_requirements_satisfied', []
                )

                # Log explicit requirements satisfaction
                if explicit_reqs:
                    if not explicit_reqs_satisfied:
                        self.logger.debug(
                            f'Result {idx} ({result.title[:50]}...): No explicit requirements satisfied, '
                            f'but trusting LLM relevance judgment: {is_relevant}'
                        )
                    else:
                        self.logger.debug(
                            f'Result {idx} ({result.title[:50]}...): Explicit requirements satisfied: {explicit_reqs_satisfied}'
                        )

                self.logger.debug(
                    f'Result {idx} ({result.title[:50]}...): Relevance={is_relevant} '
                    f'(confidence: {confidence:.2f}). Reasoning: {reasoning}'
                )

                batch_results.append((is_relevant, reasoning))

            self.logger.info(
                f'Batch relevance check complete: {sum(1 for r, _ in batch_results if r)}/{len(batch_results)} results marked as relevant'
            )

            return batch_results

        except Exception as e:
            self.logger.warning(
                f'Failed to determine batch relevance using LLM: {e}. '
                f'Defaulting all results to relevant=True.'
            )
            # Return all as relevant if batch check fails
            return [(True, 'LLM batch check failed, assuming relevant')] * len(
                search_results
            )
