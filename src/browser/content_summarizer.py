"""Content summarizer using LLM for search results."""

import logging
from typing import Any, Dict, List, Optional

from ..llm import LLMService


class ContentSummarizer:
    """Summarizes search result content using LLM."""

    def __init__(
        self, llm_service: LLMService, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize content summarizer.

        Args:
            llm_service: LLM service for summarization.
            logger: Optional logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)

    def summarize_multiple_results(
        self,
        content_parts: List[str],
        problem: str,
        subtask_description: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Summarize multiple search result content parts using LLM.

        Args:
            content_parts: List of content strings from processed search results.
            problem: Original problem description.
            subtask_description: Description of the subtask being executed.
            query_analysis: Optional query analysis results.

        Returns:
            Summarized content string.
        """
        if not content_parts:
            return ''

        # Build context from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            answer_format = query_analysis.get('answer_format', '')
            dependencies = query_analysis.get('dependencies', [])

            if explicit_reqs:
                requirements_context += (
                    f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                )
            if dependencies:
                requirements_context += (
                    f'\nInformation Dependencies: {", ".join(dependencies)}'
                )
            if answer_format:
                requirements_context += f'\nAnswer Format: {answer_format}'

        system_prompt = """You are an expert at summarizing and structuring search result content.
Given a single search result from a web page or file, create a focused summary that extracts key information relevant to the problem and subtask.

Your task:
1. Extract and highlight information that directly addresses the problem requirements
2. Preserve important facts, numbers, dates, and specific details
3. Structure the summary logically and clearly
4. Focus on actionable information that helps solve the problem
5. Remove redundant or irrelevant information
6. Maintain context about where information came from (web pages vs files)

Return a well-structured summary that:
- Preserves critical details and facts
- Is organized and easy to understand
- Focuses on relevance to the problem and subtask
- Includes source indicators when relevant"""

        summarized_parts = []
        total_length = sum(len(part) for part in content_parts)

        self.logger.info(
            f'Summarizing {len(content_parts)} content parts individually '
            f'({total_length} chars total) using LLM'
        )

        # Process each content part individually
        for idx, content_part in enumerate(content_parts):
            try:
                # Skip LLM call for very short content parts
                if len(content_part) < 200:
                    self.logger.debug(
                        f'Part {idx + 1}/{len(content_parts)} is too short ({len(content_part)} chars), '
                        f'skipping LLM summarization'
                    )
                    summarized_parts.append(content_part)
                    continue

                # Limit individual content length to avoid token limits
                max_content_length = 8000
                content_to_summarize = content_part
                if len(content_part) > max_content_length:
                    content_to_summarize = (
                        content_part[:max_content_length]
                        + '\n\n[... Content truncated for summarization ...]'
                    )
                    self.logger.debug(
                        f'Part {idx + 1}/{len(content_parts)} truncated from {len(content_part)} '
                        f'to {max_content_length} chars'
                    )

                user_prompt = f"""Main Problem: {problem}

Current Subtask: {subtask_description}
{requirements_context}

Search Result Content (Part {idx + 1} of {len(content_parts)}):
{content_to_summarize}

Summarize this content, extracting key information relevant to solving the problem and completing the subtask. Focus on facts, numbers, dates, and specific details that are directly useful."""

                self.logger.debug(
                    f'Summarizing part {idx + 1}/{len(content_parts)} '
                    f'({len(content_part)} chars)'
                )

                summary = self.llm_service.call_with_system_prompt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8192,
                )

                summarized_parts.append(summary)
                self.logger.debug(
                    f'Part {idx + 1}/{len(content_parts)} summarized: '
                    f'{len(summary)} chars (from {len(content_part)} chars)'
                )

            except Exception as e:
                self.logger.warning(
                    f'Failed to summarize content part {idx + 1}/{len(content_parts)}: {e}. '
                    f'Using original content for this part.'
                )
                summarized_parts.append(content_part)

        # Combine all summarized parts
        final_summary = '\n\n---\n\n'.join(summarized_parts)
        final_length = len(final_summary)

        self.logger.info(
            f'Content summarization complete: '
            f'{final_length} chars (from {total_length} chars total)'
        )

        return final_summary
