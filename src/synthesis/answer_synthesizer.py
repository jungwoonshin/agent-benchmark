"""Answer Synthesis Module for constructing final answers."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..llm import LLMService
from ..models import Attachment
from ..state import InformationStateManager
from ..utils import extract_json_from_text
from .result_summarizer import ResultSummarizer


class AnswerSynthesizer:
    """Synthesizes final answers from partial results."""

    def __init__(
        self,
        llm_service: LLMService,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize Answer Synthesizer.

        Args:
            llm_service: LLM service instance.
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.state_manager = state_manager
        self.logger = logger

        # Prompt size/length controls
        self.MAX_CONTEXT_PER_SUBTASK = 800
        self.MAX_REASONING_SECTION = 1200
        self.MAX_KNOWLEDGE_FACTS = 20
        self.MAX_FIELD_SNIPPET = 160

        self.result_summarizer = ResultSummarizer(
            max_context_per_subtask=self.MAX_CONTEXT_PER_SUBTASK,
            max_field_snippet=self.MAX_FIELD_SNIPPET,
        )

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """
        Extract numeric values from text using regex patterns.

        Args:
            text: Text to extract numbers from.

        Returns:
            First numeric value found, or None if none found.
        """
        if not text:
            return None

        # Pattern to match integers and decimals
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        if matches:
            try:
                # Return the first valid number
                return float(matches[0])
            except ValueError:
                pass
        return None

    def _extract_dates(self, text: str, format_hint: Optional[str] = None) -> List[str]:
        """
        Extract dates from text using various date patterns.

        Args:
            text: Text to extract dates from.
            format_hint: Optional hint about expected date format (e.g., "MM/DD/YY").

        Returns:
            List of date strings found in the text.
        """
        if not text:
            return []

        dates = []

        # MM/DD/YY or MM/DD/YYYY pattern
        mmddyy_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}'
        dates.extend(re.findall(mmddyy_pattern, text))

        # YYYY-MM-DD pattern
        iso_pattern = r'\d{4}-\d{2}-\d{2}'
        dates.extend(re.findall(iso_pattern, text))

        # Month DD, YYYY pattern
        month_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        dates.extend(re.findall(month_pattern, text, re.IGNORECASE))

        return dates

    def _extract_zip_codes(self, text: str) -> List[str]:
        """
        Extract five-digit US zip codes from text.

        Args:
            text: Text to extract zip codes from.

        Returns:
            List of zip code strings found (five-digit format).
        """
        if not text:
            return []

        # Five-digit zip code pattern
        zip_pattern = r'\b\d{5}\b'
        zip_codes = re.findall(zip_pattern, text)

        # Remove duplicates while preserving order
        seen = set()
        unique_zips = []
        for zip_code in zip_codes:
            if zip_code not in seen:
                seen.add(zip_code)
                unique_zips.append(zip_code)

        return unique_zips

    def _extract_exact_matches(
        self, text: str, requirements: List[str], case_sensitive: bool = False
    ) -> List[str]:
        """
        Extract exact term matches from text based on requirements.

        Args:
            text: Text to search in.
            requirements: List of requirement strings to look for.
            case_sensitive: Whether matching should be case-sensitive.

        Returns:
            List of matching terms found in the text.
        """
        if not text or not requirements:
            return []

        matches = []

        # Extract potential answer terms from requirements
        # Look for quoted strings, single words, or specific terms
        for req in requirements:
            # Look for quoted terms in requirements
            quoted_terms = re.findall(r'"([^"]+)"', req)
            for term in quoted_terms:
                if term.lower() in text.lower() if not case_sensitive else term in text:
                    matches.append(term)

            # Look for single-word answers (word, single word, etc.)
            if any(kw in req.lower() for kw in ['word', 'term', 'name']):
                # Try to find single-word answers that might be mentioned
                words = re.findall(r'\b\w+\b', text)
                for word in words:
                    if len(word) > 3:  # Filter out very short words
                        matches.append(word)

        return list(set(matches))

    # ===== Prompt Building Utilities (concise, structured, token-efficient) =====

    def _is_error_result(self, result: Any) -> bool:
        if isinstance(result, dict) and result.get('status') == 'failed':
            return True
        if isinstance(result, str) and (
            result.startswith('Error:')
            or result.startswith('Name Error:')
            or result.startswith('Execution Error:')
            or result.startswith('Import Error:')
        ):
            return True
        return False

    def _truncate(self, text: str, limit: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text if len(text) <= limit else text[: limit - 3] + '...'

    def _summarize_result_for_prompt(self, result: Any) -> str:
        """
        Produce a compact, high-signal summary of a subtask result prioritizing
        extracted fields used by synthesis. This greatly reduces token usage.

        Args:
            result: Result to summarize (dict or string).

        Returns:
            Compact summary string.
        """
        return self.result_summarizer.summarize(result, max_lines=6)

    def _build_requirements_str(self, explicit_requirements: List[str]) -> str:
        all_requirements = explicit_requirements or []
        if not all_requirements:
            return ''
        lines = ['\nCRITICAL ANSWER REQUIREMENTS:']
        lines.extend([f'- {r}' for r in all_requirements])

        # Add compact format constraint notice if present
        format_keywords = [
            'word',
            'single word',
            'one word',
            'number',
            'phrase',
            'sentence',
            'date',
            'zip code',
            'zipcode',
        ]
        if any(
            any(kw in (r or '').lower() for kw in format_keywords)
            for r in all_requirements
        ):
            lines.append(
                '\nFORMAT CONSTRAINT ACTIVE: Answer MUST strictly follow the requirements above.'
            )
        return '\n'.join(lines)

    def _build_query_analysis_str(self, query_analysis: Dict[str, Any]) -> str:
        """Build a formatted string from query analysis for the prompt."""
        lines = ['Query Analysis:']

        # Add answer format if present
        answer_format = query_analysis.get('answer_format', '')
        if answer_format:
            if isinstance(answer_format, dict):
                # If answer_format is a dict, extract relevant instruction
                format_str = answer_format.get(
                    'for_final_factual_answer',
                    answer_format.get('for_this_analysis_step', str(answer_format)),
                )
            else:
                format_str = str(answer_format)
            if format_str:
                lines.append(f'Answer Format: {format_str}')

        return '\n'.join(lines) if len(lines) > 1 else ''

    def _build_attachment_info(self, attachments: Optional[List['Attachment']]) -> str:
        if not attachments:
            return ''
        info = [f'\nAvailable Attachments ({len(attachments)}):']
        for i, att in enumerate(attachments):
            size_kb = (len(att.data) / 1024) if getattr(att, 'data', None) else 0
            info.append(f'- [{i}] {att.filename} ({size_kb:.1f} KB)')
        return '\n'.join(info)

    def _build_execution_summary_str(
        self, execution_summary: Dict[str, Any], query_analysis: Dict[str, Any]
    ) -> str:
        if not execution_summary:
            return ''

        lines: List[str] = ['## Execution Results (condensed)']

        # Add per-subtask compact summaries
        for subtask_id, result in execution_summary.items():
            if self._is_error_result(result):
                continue
            summary = self._summarize_result_for_prompt(result)
            if summary:
                lines.append(f'\n### {subtask_id}\n{summary}')

        # Key values for calculations
        calculation_values: List[str] = []
        for subtask_id, result in execution_summary.items():
            if self._is_error_result(result) or not isinstance(result, dict):
                continue

            if result.get('extracted_counts'):
                for item in result['extracted_counts']:
                    val = item.get('value')
                    if val is not None:
                        calculation_values.append(f'- {subtask_id}: {val}')
            elif result.get('llm_extraction') and isinstance(
                result['llm_extraction'], dict
            ):
                ev = result['llm_extraction'].get('extracted_value')
                if ev is not None:
                    calculation_values.append(f'- {subtask_id}: {ev}')

        if calculation_values:
            lines.append('\n## Key Values for Calculations (use exactly)')
            lines.extend(calculation_values[:10])

        return '\n'.join(lines)

    def _build_knowledge_facts_str(self) -> str:
        facts = [
            {'entity': f.entity, 'relationship': f.relationship, 'value': f.value}
            for f in self.state_manager.knowledge_graph
        ]
        if not facts:
            return ''
        limited = facts[: self.MAX_KNOWLEDGE_FACTS]
        try:
            facts_str = json.dumps(limited, ensure_ascii=False, default=str)
        except Exception:
            facts_str = str(limited)
        return '## Knowledge Graph Facts\n' + facts_str

    def _build_system_prompt(self, query_analysis: Dict[str, Any] = None) -> str:
        """
        Concise, strict system prompt to minimize tokens while enforcing rules.
        """
        base_prompt = (
            'CRITICAL: Respond with a single valid JSON object only. No markdown, no extra text.\n'
            'Role: Expert answer synthesizer. Combine evidence to produce a precise final answer.\n\n'
            'Output schema: {"final_answer": string, "description": string}.\n\n'
            'Rules:\n'
            '- final_answer MUST strictly follow all format constraints (e.g., single word, number only, date, zip).\n'
            "- Use EXACT extracted values found in execution results: sections 'Extracted Structured Data' or 'Key Values for Calculations', fields 'extracted_counts' or 'llm_extraction' are authoritative.\n"
            '- When execution results contain reasoning or analysis, identify the specific conclusion or answer that directly solves the problem, not intermediate steps or discussed concepts.\n'
            '- Distinguish between what is analyzed or discussed in reasoning versus what is determined to be the actual answer to the problem.\n'
            '- If the problem asks for a character name, extract the NAME (the word used to refer to the character), not the character symbol or code that uses it.\n'
            '- Do NOT add units unless explicitly required.\n'
            '- No explanations or meta-text in final_answer.\n\n'
        )

        # Add general terminology guidance when ambiguity is possible
        if query_analysis and query_analysis.get('has_terminology_ambiguity'):
            terminology_context = query_analysis.get('terminology_context', '').lower()

            terminology_guidance = '\nCRITICAL: TERMINOLOGY SELECTION GUIDANCE\n'
            terminology_guidance += (
                '- When multiple valid terms/names exist for the same concept, prioritize the most commonly recognized and widely used term\n'
                '- Prefer the term that is most standard and frequently used in the relevant domain or context\n'
                '- Select the term that is most unambiguous and clearly understood by the broadest audience\n'
                '- When domain-specific terminology exists, use the term that is most standard within that domain\n'
                '- Avoid less common or alternative names when a more standard term is available\n'
                '- Consider the term that would be most immediately recognizable and unambiguous to experts in the field\n'
                '- Follow any explicit requirements about term length or form while maintaining clarity and standard usage\n'
            )

            if terminology_context:
                terminology_guidance += f'- The context is: {terminology_context}\n'
                terminology_guidance += (
                    f'- Prefer terminology standard in {terminology_context} contexts\n'
                )

            base_prompt += terminology_guidance + '\n'

        base_prompt += 'Return only the JSON object.'
        return base_prompt

    def _build_user_prompt(
        self,
        problem: str,
        query_analysis: Dict[str, Any],
        requirements_str: str,
        attachment_info: str,
        execution_summary_str: str,
        knowledge_facts_str: str,
    ) -> str:
        # Build query analysis section
        query_analysis_str = self._build_query_analysis_str(query_analysis)

        # Add terminology context if relevant
        terminology_info = ''
        if query_analysis.get('has_terminology_ambiguity'):
            context = query_analysis.get('terminology_context', '')
            if context:
                terminology_info = f'\nTerminology Context: {context}\n'
                terminology_info += 'When multiple valid terms exist, prioritize the most commonly recognized and widely used term in this context. Prefer the most standard and frequently used terminology while following any explicit requirements.\n'

        return (
            f'Problem:\n{problem}\n\n'
            f'{query_analysis_str}\n'
            f'{terminology_info}'
            f'{requirements_str}\n\n'
            f'Available Sources:{attachment_info}\n\n'
            f'{execution_summary_str}\n\n'
            f'{knowledge_facts_str}\n\n'
            'Instructions:\n'
            '- Extract the exact answer that matches the format.\n'
            "- For calculations, use values shown under 'Key Values for Calculations' exactly.\n"
            '- When execution results contain step-by-step reasoning or analysis, identify the final conclusion or answer that the reasoning leads to.\n'
            '- If the reasoning discusses multiple concepts or mentions various terms, extract the specific answer that directly addresses the problem question.\n'
            '- When the problem asks for a character name, ensure you extract the NAME of the character (the word used to refer to it), not the character symbol itself.\n'
            '- Distinguish between what the reasoning discusses or analyzes versus what it concludes is the actual answer.\n'
            '- Show brief work in description, but keep final_answer strictly formatted.\n'
        )

    def _validate_and_correct_format(
        self,
        final_answer: str,
        query_analysis: Dict[str, Any],
        execution_summary: Dict[str, Any],
    ) -> str:
        """
        Validate final answer format and correct if needed.

        Args:
            final_answer: The synthesized answer.
            query_analysis: Query analysis with format requirements.
            execution_summary: Execution results for re-extraction if needed.

        Returns:
            Corrected answer that matches format requirements.
        """
        all_requirements = query_analysis.get('explicit_requirements', [])
        requirements_text = ' '.join(all_requirements).lower()

        # Check if format requires a single word
        if any(
            kw in requirements_text for kw in ['single word', 'one word', 'word only']
        ):
            # Extract only the first word
            words = final_answer.split()
            if len(words) > 1:
                # Multiple words found, extract just the first meaningful word
                for word in words:
                    word_clean = re.sub(r'[^\w]', '', word)
                    if len(word_clean) > 2:  # Skip very short words
                        self.logger.debug(
                            f'Format correction: extracted single word "{word_clean}" from "{final_answer}"'
                        )
                        return word_clean
            elif len(words) == 1:
                # Clean up the single word
                word_clean = re.sub(r'[^\w]', '', words[0])
                return word_clean

        # Check if format requires a number only
        if any(kw in requirements_text for kw in ['number only', 'numeric', 'integer']):
            # Extract only numeric value
            numeric_value = self._extract_numeric_value(final_answer)
            if numeric_value is not None:
                # Return as integer if it's a whole number, otherwise as string
                if numeric_value.is_integer():
                    return str(int(numeric_value))
                return str(numeric_value)

        # Check if format requires zip codes
        if 'zip code' in requirements_text or 'zipcode' in requirements_text:
            zip_codes = self._extract_zip_codes(final_answer)
            if zip_codes:
                return ','.join(zip_codes)
            # If zip codes not in answer, try to extract from execution results
            all_results_text = ' '.join([str(r) for r in execution_summary.values()])
            zip_codes_from_results = self._extract_zip_codes(all_results_text)
            if zip_codes_from_results:
                self.logger.debug(
                    f'Format correction: extracted zip codes "{zip_codes_from_results}" from execution results'
                )
                return ','.join(zip_codes_from_results)

        # Check if format requires a date
        if 'date' in requirements_text:
            dates = self._extract_dates(final_answer)
            if dates:
                return dates[0]
            # If date not in answer, try to extract from execution results
            all_results_text = ' '.join([str(r) for r in execution_summary.values()])
            dates_from_results = self._extract_dates(all_results_text)
            if dates_from_results:
                self.logger.debug(
                    f'Format correction: extracted date "{dates_from_results[0]}" from execution results'
                )
                return dates_from_results[0]

        return final_answer

    def synthesize(
        self,
        problem: str,
        execution_results: Dict[str, Any],
        query_analysis: Dict[str, Any],
        attachments: Optional[List['Attachment']] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize final report answer from execution results.

        Generates a comprehensive final report answer that integrates all execution results,
        knowledge graph facts, and attachments into a well-structured response.

        Args:
            problem: Original problem statement.
            execution_results: Results from plan execution, containing execution_summary.
            query_analysis: Query analysis with answer format requirements.
            attachments: Optional list of file attachments (including downloaded files).

        Returns:
            Dictionary with:
            - final_answer: The comprehensive synthesized report answer
            - validation_info: Empty dictionary (validation is handled by validation.py)
        """
        self.logger.info('Synthesizing final report answer')

        system_prompt = self._build_system_prompt(query_analysis)

        # Extract explicit requirements for format constraints
        explicit_requirements = query_analysis.get('explicit_requirements', [])

        # Build requirements section for prompt (condensed)
        requirements_str = self._build_requirements_str(explicit_requirements)

        # Build attachment information for the prompt (compact)
        attachment_info = self._build_attachment_info(attachments)

        # Build comprehensive but condensed execution summary
        execution_summary = execution_results.get('execution_summary', {})

        # Format knowledge graph facts (limited)
        knowledge_facts_str = self._build_knowledge_facts_str()

        user_prompt = self._build_user_prompt(
            problem=problem,
            query_analysis=query_analysis,
            requirements_str=requirements_str,
            attachment_info=attachment_info,
            execution_summary_str=execution_summary,
            knowledge_facts_str=knowledge_facts_str,
        )

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Maximum determinism for consistent answer formatting
                response_format={'type': 'json_object'},
                max_tokens=4096,
            )

            # Validate response exists
            if not response:
                self.logger.warning('LLM service returned empty response')
                raise ValueError('Empty response from LLM service')

            # Log the raw response for debugging
            self.logger.debug(
                f'Raw synthesis response (first 500 chars): {response[:500]}'
            )

            # Extract JSON and parse it - return as-is
            json_text = extract_json_from_text(response)
            response_data = json.loads(json_text)

            # Extract final_answer from the JSON response
            final_answer = response_data.get('final_answer', '')
            description = response_data.get('description', '')

            # Log what was extracted
            self.logger.debug(
                f'Extracted final_answer: {final_answer}, description: {description[:100] if description else "None"}'
            )

            # Enforce final format compliance post-hoc for robustness
            try:
                corrected = self._validate_and_correct_format(
                    str(final_answer),
                    query_analysis=query_analysis,
                    execution_summary=execution_results.get('execution_summary', {}),
                )
                if corrected != final_answer:
                    self.logger.debug(
                        f'Post-validated final_answer corrected from "{final_answer}" to "{corrected}"'
                    )
                    response_data['final_answer'] = corrected
                    final_answer = corrected
            except Exception:
                # If anything goes wrong, keep the original final_answer
                pass

            # Log summary
            answer_preview = (
                final_answer[:100] + '...'
                if len(str(final_answer)) > 100
                else final_answer
            )
            self.logger.info(
                f'Final report answer synthesized: {answer_preview} '
                f'(length: {len(str(final_answer))})'
            )

            # Return final_answer with empty validation_info (validation handled by validation.py)
            # return {'final_answer': final_answer, 'validation_info': {}}
            return response_data

        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse synthesis response: {e}')
            self.logger.debug(
                f'Response was: {response[:1000] if "response" in locals() else "N/A"}'
            )
            # Try to extract any answer-like text from the response
            fallback_answer = 'Unable to determine answer'
            if 'response' in locals():
                # Look for answer-like patterns in the raw response
                answer_match = re.search(
                    r'(?:answer|result)[:\s]+([^\n\.]{1,200})', response, re.IGNORECASE
                )
                if answer_match:
                    fallback_answer = answer_match.group(1).strip()
            return {'final_answer': fallback_answer, 'validation_info': {}}
        except Exception as e:
            self.logger.error(f'Answer synthesis failed: {e}', exc_info=True)
            # Always return a valid format even on error
            fallback_answer = 'Unable to synthesize answer due to an error. Please check logs for details.'
            # Try to extract any useful information from execution results as fallback
            try:
                execution_data = execution_results.get('execution_summary', {})
                if isinstance(execution_data, dict) and execution_data:
                    # Get the first non-empty result as fallback
                    for result_value in execution_data.values():
                        if result_value and isinstance(result_value, str):
                            if (
                                len(result_value) > 20
                                and 'error' not in result_value.lower()
                            ):
                                fallback_answer = f'Partial answer (synthesis failed): {result_value[:500]}'
                                break
            except Exception:
                pass  # Use default fallback if extraction fails
            return {
                'final_answer': fallback_answer,
                'description': 'Exception occurred during answer synthesis',
            }
