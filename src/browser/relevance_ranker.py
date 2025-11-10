"""Relevance ranker for search results using LLM."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..llm import LLMService
from ..models import SearchResult
from ..utils import extract_json_from_text
from ..utils.arxiv_utils import get_arxiv_submission_date


class RelevanceRanker:
    """Ranks search results by relevance using LLM."""

    def __init__(
        self, llm_service: LLMService, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize relevance ranker.

        Args:
            llm_service: LLM service for ranking.
            logger: Optional logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)

    def rank_by_relevance(
        self,
        search_results: List[SearchResult],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Use LLM to select and rank relevant search results.

        Args:
            search_results: List of SearchResult objects to filter and rank.
            subtask_description: Description of the subtask being executed.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            List of SearchResult objects that are relevant, sorted by relevance.
        """
        if not search_results:
            return []

        if len(search_results) == 1:
            return search_results

        self.logger.info(
            f'Using LLM to select relevant search results from {len(search_results)} total results...'
        )

        # Build context from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            if explicit_reqs:
                requirements_context += (
                    f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                )

        # Format search results for LLM with submission_date for arXiv papers
        results_text = []
        for idx, result in enumerate(search_results, 1):
            snippet_preview = (
                result.snippet[:300] + '...'
                if len(result.snippet) > 300
                else result.snippet
            )

            # Get submission_date for arXiv papers using arxiv library
            submission_date = get_arxiv_submission_date(result.url, self.logger)

            # Build result text with submission_date on top if available
            result_lines = []
            if submission_date:
                result_lines.append(f'Submission Date: {submission_date}')
            result_lines.append(f'Title: {result.title}')
            result_lines.append(f'URL: {result.url}')
            result_lines.append(f'Snippet: {snippet_preview}')

            results_text.append(f'[{idx}]\n' + '\n'.join(result_lines))

        system_prompt = """You are an expert at identifying search result candidates for a specific task.
Given a subtask description and a list of search results (with title, URL, and snippet), determine which results are promising enough to be considered as candidates for further processing.

Your task:
1. Analyze each search result based on its title, URL, and snippet
2. Determine if each result is a good CANDIDATE - meaning it appears relevant enough to potentially contain useful information for the subtask
3. Include results that show promise or potential relevance, even if not perfectly aligned
4. Exclude only results that are clearly irrelevant, completely off-topic, or obviously unrelated
5. Rank the selected candidates from most promising to least promising
6. Assign a candidate score from 0.0 to 1.0 for each selected result (indicating how promising it is as a candidate)

CRITICAL: Focus on candidate selection, not strict relevance. Include results that are promising enough to warrant further investigation. It's better to include a reasonable set of candidates than to be overly restrictive.

IMPORTANT DATE HANDLING: When evaluating dates (especially for arXiv papers):
- Use the Submission Date field if provided (this is the actual submission date from arXiv API)
- DO NOT use arXiv ID formats (e.g., 2207.01510) to infer submission dates - the arXiv ID format does not reliably indicate the actual submission date
- ONLY use explicit date information from snippets, metadata, or the Submission Date field
- If a snippet mentions "submitted on [date]" or "originally submitted [date]", use that date
- If no explicit date is mentioned in the snippet and no Submission Date field is provided, do not infer dates from arXiv IDs or URLs

Return a JSON object with:
- selected_indices: array of result indices (1-based) that are good candidates, sorted by promise (most promising first)
- scores: object mapping result index (as string, 1-based) to candidate score (0.0 to 1.0) for selected results
- reasoning: brief explanation of which results were selected as candidates and why (1-2 sentences)

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}

Search Results:
{chr(10).join(results_text)}

Identify which search results are promising enough to be considered as candidates for further processing. Select results that appear relevant enough to potentially contain useful information for this subtask. Return only the indices of candidate results, sorted from most promising to least promising. Exclude only results that are clearly irrelevant, completely off-topic, or obviously unrelated."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            selected_indices = result_data.get('selected_indices', [])
            scores = result_data.get('scores', {})
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'LLM selection complete. Selected {len(selected_indices)} relevant result(s) out of {len(search_results)} total. Reasoning: {reasoning}'
            )

            # Validate selected_indices
            if not selected_indices:
                self.logger.warning(
                    'LLM did not select any relevant results. Using first result as fallback.'
                )
                return [search_results[0]] if search_results else []

            # Convert 1-based indices to 0-based and validate
            valid_indices = []
            for idx in selected_indices:
                try:
                    idx_int = int(idx) - 1  # Convert to 0-based
                    if 0 <= idx_int < len(search_results):
                        valid_indices.append(idx_int)
                except (ValueError, TypeError):
                    continue

            if not valid_indices:
                self.logger.warning(
                    'No valid indices found in LLM response. Using first result as fallback.'
                )
                return [search_results[0]] if search_results else []

            # Get selected results in order (already sorted by relevance)
            selected_results = [search_results[i] for i in valid_indices]

            # Update relevance scores if provided
            for idx, result in enumerate(selected_results):
                original_idx = valid_indices[idx]
                score_key = str(original_idx + 1)  # 1-based for score lookup
                if score_key in scores:
                    result.relevance_score = float(scores[score_key])

            self.logger.info(
                f'Selected {len(selected_results)} relevant search result(s). Top result: {selected_results[0].title[:50]}...'
            )

            return selected_results

        except Exception as e:
            self.logger.warning(
                f'Failed to select relevant search results using LLM: {e}. Using first {min(3, len(search_results))} results as fallback.'
            )
            return search_results[: min(3, len(search_results))]
