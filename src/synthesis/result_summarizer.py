"""Result summarization utilities for extracting key information from subtask results."""

import json
from typing import Any, Dict, List, Optional


class ResultSummarizer:
    """Utility class for summarizing subtask results for synthesis prompts."""

    def __init__(
        self,
        max_context_per_subtask: int = 800,
        max_field_snippet: int = 160,
    ):
        """
        Initialize Result Summarizer.

        Args:
            max_context_per_subtask: Maximum characters per subtask context.
            max_field_snippet: Maximum characters for field snippets.
        """
        self.max_context_per_subtask = max_context_per_subtask
        self.max_field_snippet = max_field_snippet

    def truncate(self, text: str, limit: int) -> str:
        """Truncate text to limit, adding ellipsis if needed."""
        if not isinstance(text, str):
            text = str(text)
        return text if len(text) <= limit else text[: limit - 3] + '...'

    def format_confidence(self, confidence: Any) -> str:
        """Format confidence value as string."""
        if isinstance(confidence, (int, float)):
            return f' (confidence: {confidence:.2f})'
        return ''

    def extract_counts_summary(
        self, extracted_counts: List[Dict[str, Any]], max_items: int = 3
    ) -> List[str]:
        """
        Extract summary lines from extracted_counts field.

        Args:
            extracted_counts: List of count items with value, context, confidence.
            max_items: Maximum number of items to include.

        Returns:
            List of formatted summary lines.
        """
        lines = []
        if not isinstance(extracted_counts, list) or not extracted_counts:
            return lines

        for item in extracted_counts[:max_items]:
            value = item.get('value')
            if value is None:
                continue

            ctx = self.truncate(item.get('context', ''), self.max_field_snippet)
            conf_str = self.format_confidence(item.get('confidence'))
            lines.append(f'COUNT: {value}{conf_str} - {ctx}'.strip())

        return lines

    def extract_llm_extraction_summary(
        self, llm_extraction: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract summary line from llm_extraction field.

        Args:
            llm_extraction: Dict with extracted_value, reasoning, context, confidence.

        Returns:
            Formatted summary line or None if no valid extraction.
        """
        if not isinstance(llm_extraction, dict):
            return None

        extracted_value = llm_extraction.get('extracted_value')
        if extracted_value is None:
            return None

        conf_str = self.format_confidence(llm_extraction.get('confidence'))
        reas = self.truncate(
            llm_extraction.get('reasoning', ''), self.max_field_snippet
        )
        ctx = self.truncate(llm_extraction.get('context', ''), self.max_field_snippet)

        # Only include context if it's different from reasoning and adds value
        if ctx and ctx != reas and len(ctx) > 10:
            return f'LLM_EXTRACT: {extracted_value}{conf_str} - {reas} [context: {ctx}]'.strip()
        else:
            return f'LLM_EXTRACT: {extracted_value}{conf_str} - {reas}'.strip()

    def extract_numeric_data_summary(
        self, numeric_data: Dict[str, Any], max_items: int = 2
    ) -> List[str]:
        """
        Extract summary lines from numeric_data.counts field.

        Args:
            numeric_data: Dict with counts list.
            max_items: Maximum number of items to include.

        Returns:
            List of formatted summary lines.
        """
        lines = []
        if not isinstance(numeric_data, dict):
            return lines

        counts = numeric_data.get('counts') or []
        if not isinstance(counts, list) or not counts:
            return lines

        for item in counts[:max_items]:
            value = item.get('value')
            if value is None:
                continue

            ctx = self.truncate(item.get('context', ''), self.max_field_snippet)
            lines.append(f'REGEX_COUNT: {value} - {ctx}'.strip())

        return lines

    def extract_image_analysis_summary(self, image_analysis: Any) -> Optional[str]:
        """
        Extract summary line from image_analysis field.

        Args:
            image_analysis: Image analysis string or None.

        Returns:
            Formatted summary line or None if no image analysis.
        """
        if not image_analysis or not isinstance(image_analysis, str):
            return None

        summary = self.truncate(image_analysis, self.max_context_per_subtask)
        return f'IMAGE_ANALYSIS: {summary}'

    def summarize_dict_result(self, result: Dict[str, Any], max_lines: int = 6) -> str:
        """
        Summarize a dictionary result by extracting key fields.

        Args:
            result: Dictionary result to summarize.
            max_lines: Maximum number of summary lines to return.

        Returns:
            Formatted summary string.
        """
        lines: List[str] = []

        # Priority 1: extracted_counts
        extracted_counts = result.get('extracted_counts') or []
        lines.extend(self.extract_counts_summary(extracted_counts))

        # Priority 2: llm_extraction
        llm_extraction = result.get('llm_extraction') or {}
        llm_line = self.extract_llm_extraction_summary(llm_extraction)
        if llm_line:
            lines.append(llm_line)

        # Priority 3: numeric_data.counts
        numeric_data = result.get('numeric_data') or {}
        lines.extend(self.extract_numeric_data_summary(numeric_data))

        # Priority 4: image_analysis
        image_analysis = result.get('image_analysis')
        image_line = self.extract_image_analysis_summary(image_analysis)
        if image_line:
            lines.append(image_line)

        if lines:
            return '\n'.join(lines[:max_lines])

        # Fallback: compact JSON dump
        try:
            compact = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        except Exception:
            compact = str(result)
        return self.truncate(compact, self.max_context_per_subtask)

    def extract_image_analysis_from_string(
        self, result_str: str
    ) -> Optional[Dict[str, str]]:
        """
        Extract image analysis section from string result.

        Args:
            result_str: String result that may contain image analysis.

        Returns:
            Dict with 'text_before' and 'image_analysis' keys, or None if not found.
        """
        marker = 'IMAGE ANALYSIS (from visual LLM):'
        if marker not in result_str:
            return None

        marker_idx = result_str.find(marker)
        text_before = result_str[:marker_idx].strip()
        image_analysis = result_str[marker_idx:].strip()

        return {
            'text_before': text_before,
            'image_analysis': image_analysis,
        }

    def summarize_string_result(self, result: Any) -> str:
        """
        Summarize a string result, preserving image analysis if present.

        Args:
            result: String result to summarize.

        Returns:
            Formatted summary string.
        """
        result_str = str(result)

        # Check for image analysis marker
        image_data = self.extract_image_analysis_from_string(result_str)
        if image_data:
            text_before_summary = self.truncate(
                image_data['text_before'], self.max_context_per_subtask // 2
            )
            image_analysis_summary = self.truncate(
                image_data['image_analysis'], self.max_context_per_subtask
            )

            if text_before_summary:
                return f'{text_before_summary}\n\n{image_analysis_summary}'
            else:
                return image_analysis_summary

        # No image analysis, just truncate
        return self.truncate(result_str, self.max_context_per_subtask)

    def summarize(self, result: Any, max_lines: int = 6) -> str:
        """
        Summarize a result (dict or string) for use in synthesis prompts.

        Args:
            result: Result to summarize (dict or string).
            max_lines: Maximum number of summary lines for dict results.

        Returns:
            Compact summary string optimized for token usage.
        """
        if isinstance(result, dict):
            return self.summarize_dict_result(result, max_lines)
        else:
            return self.summarize_string_result(result)
