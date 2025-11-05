"""Answer Synthesis Module for constructing final answers."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..llm import LLMService
from ..models import Attachment
from ..state import InformationStateManager
from ..utils import extract_json_from_text


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
        all_requirements = query_analysis.get(
            'explicit_requirements', []
        ) + query_analysis.get('implicit_requirements', [])
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

    def _extract_structured_data_hints(
        self, execution_summary: Dict[str, Any], query_analysis: Dict[str, Any]
    ) -> str:
        """
        Extract structured data (numbers, dates, zip codes) from execution results
        and format as hints for the synthesis prompt.

        Args:
            execution_summary: Dictionary of execution results.
            query_analysis: Query analysis with requirements.

        Returns:
            Formatted string with extracted structured data.
        """
        hints = []

        # PRIORITY 1: Extract from browser_navigate extracted_counts (most reliable)
        extracted_counts_found = []
        for result in execution_summary.values():
            # Skip error results
            if isinstance(result, dict) and result.get('status') == 'failed':
                continue
            if isinstance(result, str) and (
                result.startswith('Error:')
                or result.startswith('Name Error:')
                or result.startswith('Execution Error:')
            ):
                continue

            if isinstance(result, dict):
                # Check for extracted_counts (from extract_count/extract_statistics)
                if 'extracted_counts' in result and result.get('extracted_counts'):
                    for count_item in result['extracted_counts']:
                        value = count_item.get('value')
                        if value is not None:
                            context = count_item.get('context', '')[:100]
                            confidence = count_item.get('confidence', 'N/A')
                            extracted_counts_found.append(
                                {
                                    'value': value,
                                    'context': context,
                                    'confidence': confidence,
                                }
                            )

                # Check for llm_extraction (from LLM-based extraction)
                if 'llm_extraction' in result and result.get('llm_extraction'):
                    llm_extraction = result['llm_extraction']
                    extracted_value = llm_extraction.get('extracted_value')
                    if extracted_value is not None:
                        confidence = llm_extraction.get('confidence', 0.0)
                        reasoning = llm_extraction.get('reasoning', '')[:100]
                        extracted_counts_found.append(
                            {
                                'value': extracted_value,
                                'context': reasoning,
                                'confidence': confidence,
                            }
                        )

                # Check for numeric_data.counts (fallback from regex extraction)
                if 'numeric_data' in result and result.get('numeric_data'):
                    numeric_data = result['numeric_data']
                    counts = numeric_data.get('counts', [])
                    if counts:
                        for count_item in counts[:3]:  # Top 3 counts
                            value = count_item.get('value')
                            if value is not None:
                                context = count_item.get('context', '')[:100]
                                extracted_counts_found.append(
                                    {
                                        'value': value,
                                        'context': context,
                                        'confidence': 'regex_extraction',
                                    }
                                )

        # Add extracted counts as primary hints
        if extracted_counts_found:
            for item in extracted_counts_found:
                conf_str = (
                    f' (confidence: {item["confidence"]:.2f})'
                    if isinstance(item['confidence'], (int, float))
                    else ''
                )
                context_str = f' - {item["context"]}' if item.get('context') else ''
                hints.append(
                    f'EXTRACTED COUNT VALUE: {item["value"]}{conf_str}{context_str}'
                )

        # Fallback: Extract numeric values from text (lower priority)
        all_results_text = ' '.join(
            [str(result) for result in execution_summary.values()]
        )

        # Only extract from text if we didn't find explicit extracted counts
        if not extracted_counts_found:
            numeric_value = self._extract_numeric_value(all_results_text)
            if numeric_value is not None:
                hints.append(f'Numeric value found (from text): {numeric_value}')

        # Extract zip codes
        zip_codes = self._extract_zip_codes(all_results_text)
        if zip_codes:
            hints.append(f'Zip codes found: {", ".join(zip_codes)}')

        # Extract dates
        dates = self._extract_dates(all_results_text)
        if dates:
            hints.append(f'Dates found: {", ".join(dates[:5])}')  # Limit to first 5

        # Extract exact matches from requirements
        all_requirements = query_analysis.get(
            'explicit_requirements', []
        ) + query_analysis.get('implicit_requirements', [])
        exact_matches = self._extract_exact_matches(all_results_text, all_requirements)
        if exact_matches:
            hints.append(
                f'Potential answer terms found: {", ".join(exact_matches[:5])}'
            )

        return '\n'.join(hints) if hints else ''

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
            execution_results: Results from plan execution, containing execution_summary and reasoning_summary.
            query_analysis: Query analysis with answer format requirements.
            attachments: Optional list of file attachments (including downloaded files).

        Returns:
            Dictionary with:
            - final_answer: The comprehensive synthesized report answer
            - validation_info: Empty dictionary (validation is handled by validation.py)
        """
        self.logger.info('Synthesizing final report answer')

        system_prompt = """You are an expert at synthesizing final answers from multiple sources.

Your task is to generate a precise answer that:
1. Directly addresses the user's problem/question
2. Integrates all available information from execution results, knowledge graph, and attachments
3. STRICTLY follows the format requirements specified (e.g., "word" = single word only, "number" = number only)
4. Provides ONLY the factual answer without extra explanations
5. **CRITICAL: ALWAY return the NAME in words using the shortest common name rather than the symbol**

CRITICAL: USE EXACT EXTRACTED VALUES FOR CALCULATIONS
- When execution results show extracted values (e.g., "EXTRACTED VALUE: XXX"), you MUST use that EXACT number for calculations
- DO NOT assume, estimate, or guess different numbers when an extracted value is available
- If a calculation is needed, use the exact extracted value from the results
- Look for sections marked "=== EXTRACTED VALUES (PRIMARY DATA) ===" or "EXTRACTED COUNT VALUE:" - these are the actual extracted values to use
- DO NOT use phrases like "assumed", "approximately", or "estimated" when an exact extracted value exists

CRITICAL FORMAT COMPLIANCE RULES:
- If format requirement says "word" or "single word": return ONLY a single word (no explanations, no sentences)
- If format requirement says "number": return ONLY a number (no text, no units unless specified)
- If format requirement says "phrase": return a short phrase only
- The final_answer MUST strictly match the format requirements - no deviations
- DO NOT add explanations, summaries, or meta-commentary to the answer
- DO NOT expand beyond the format constraint even if you think more context would help

GENERATION GUIDELINES:
- Extract the exact answer from the execution results that matches the format requirement
- SHOW YOUR WORK: Before providing the final answer, identify and extract all intermediate values from execution results (numbers, dates, strings, etc.)
- For calculations: Use the EXACT extracted values shown in the execution results (look for "EXTRACTED VALUE" markers)
- If format requires a word, find the exact word in the results and return ONLY that word
- If format requires a number, extract the number and return ONLY that number (no text, no units, no explanations)
- If format requires a date, extract the date in the exact format specified
- If format requires zip codes, extract the five-digit zip codes found in results
- Prioritize format compliance over completeness - better to return a correctly formatted partial answer than a well-formatted but incorrectly formatted complete answer
- Pay special attention to structured data extraction hints provided in the execution results
- If information is missing, return the best available answer that still meets format requirements

CRITICAL: If the `reasoning_summary` explicitly identifies a specific number as the 'best_ball' or optimal choice to maximize winning probability, you MUST use that number as the `final_answer`. Ensure the `final_answer` is consistent with the `reasoning_summary`.

ALWAYS return a JSON object with EXACTLY these two fields:
- "final_answer": The answer in the exact format specified (e.g., if "word" required, return just the word, not a sentence explaining the word). This is the ONLY field that should be used for answer extraction.
- "description": A brief description of how the final_answer was derived (optional explanation, but final_answer is the authoritative answer)

CRITICAL: The "final_answer" field is the ONLY answer that should be extracted and used. Do NOT include dates, intermediate values, or other information in final_answer unless they are the actual answer to the question.

The final_answer MUST be formatted exactly as required - no exceptions."""

        # Extract and simplify answer_format to avoid confusion
        answer_format_raw = query_analysis.get('answer_format', 'text')
        if isinstance(answer_format_raw, dict):
            # If answer_format is a dict, extract the relevant instruction
            answer_format_str = answer_format_raw.get(
                'for_final_factual_answer',
                answer_format_raw.get('for_this_analysis_step', str(answer_format_raw)),
            )
        else:
            answer_format_str = str(answer_format_raw)

        # Extract explicit and implicit requirements for format constraints
        explicit_requirements = query_analysis.get('explicit_requirements', [])
        implicit_requirements = query_analysis.get('implicit_requirements', [])

        # Build requirements section for prompt
        requirements_str = ''
        all_requirements = explicit_requirements + implicit_requirements
        if all_requirements:
            requirements_str = '\n\nCRITICAL ANSWER REQUIREMENTS (MUST BE FOLLOWED):\n'
            for req in all_requirements:
                requirements_str += f'- {req}\n'
            # Check for format constraints in requirements (e.g., "word", "single word", "number")
            format_keywords = [
                'word',
                'single word',
                'one word',
                'number',
                'phrase',
                'sentence',
            ]
            format_requirements = [
                r
                for r in all_requirements
                if any(kw in r.lower() for kw in format_keywords)
            ]
            if format_requirements:
                requirements_str += '\n⚠️ FORMAT CONSTRAINT DETECTED - Answer MUST strictly adhere to format requirements above!'

        # Build attachment information for the prompt
        attachment_info = ''
        if attachments:
            attachment_info = (
                f'\n\nAvailable Attachments ({len(attachments)} file(s)):\n'
            )
            for i, att in enumerate(attachments):
                file_size = len(att.data) if att.data else 0
                file_size_kb = file_size / 1024 if file_size > 0 else 0
                attachment_info += (
                    f'  - Attachment {i}: {att.filename} ({file_size_kb:.1f} KB)\n'
                )
            attachment_info += '\nNOTE: These attachments (including any files downloaded during execution) are available for analysis. '
            attachment_info += 'If the execution results reference these files or if you need to extract information from them, '
            attachment_info += 'you may reference them in your answer. However, note that the actual file contents have already been '
            attachment_info += (
                'processed and should be reflected in the execution results above.'
            )

        # Build comprehensive execution summary
        execution_summary = execution_results.get('execution_summary', {})
        reasoning_summary = execution_results.get('reasoning_summary', {})

        # Format execution results for better readability
        execution_summary_str = ''
        if execution_summary:
            execution_summary_str = '## Execution Results Summary\n'
            for subtask_id, result in execution_summary.items():
                # Filter out error results - skip failed subtasks and error strings
                if isinstance(result, dict) and result.get('status') == 'failed':
                    self.logger.debug(
                        f'Skipping failed subtask {subtask_id} from synthesis'
                    )
                    continue

                # Check if result is an error string
                if isinstance(result, str) and (
                    result.startswith('Error:')
                    or result.startswith('Name Error:')
                    or result.startswith('Execution Error:')
                    or result.startswith('Import Error:')
                ):
                    self.logger.debug(
                        f'Skipping error result from subtask {subtask_id}: {result[:100]}'
                    )
                    continue

                if isinstance(result, dict):
                    result_str = json.dumps(
                        result, indent=2, ensure_ascii=False, default=str
                    )
                else:
                    result_str = str(result)
                execution_summary_str += f'\n### Subtask {subtask_id}:\n{result_str}\n'

            # Add structured data extraction hints
            structured_data_hints = self._extract_structured_data_hints(
                execution_summary, query_analysis
            )
            if structured_data_hints:
                execution_summary_str += f'\n## Extracted Structured Data (USE THESE VALUES FOR CALCULATIONS):\n{structured_data_hints}\n'

            # Extract and highlight key values for calculations
            calculation_values = []
            for subtask_id, result in execution_summary.items():
                # Skip error results
                if isinstance(result, dict) and result.get('status') == 'failed':
                    continue
                if isinstance(result, str) and (
                    result.startswith('Error:')
                    or result.startswith('Name Error:')
                    or result.startswith('Execution Error:')
                    or result.startswith('Import Error:')
                ):
                    continue

                if isinstance(result, dict):
                    # Check for extracted_counts
                    if 'extracted_counts' in result and result.get('extracted_counts'):
                        for count_item in result['extracted_counts']:
                            value = count_item.get('value')
                            if value is not None:
                                context = count_item.get('context', '')[:200]
                                calculation_values.append(
                                    {
                                        'subtask': subtask_id,
                                        'value': value,
                                        'context': context,
                                        'type': 'extracted_count',
                                    }
                                )
                    # Check for llm_extraction
                    elif 'llm_extraction' in result and result.get('llm_extraction'):
                        llm_extraction = result['llm_extraction']
                        extracted_value = llm_extraction.get('extracted_value')
                        if extracted_value is not None:
                            reasoning = llm_extraction.get('reasoning', '')[:200]
                            calculation_values.append(
                                {
                                    'subtask': subtask_id,
                                    'value': extracted_value,
                                    'context': reasoning,
                                    'type': 'llm_extraction',
                                }
                            )

            if calculation_values:
                calculation_hints = (
                    '\n## Key Values for Calculations (USE THESE EXACT VALUES):\n'
                )
                calculation_hints += 'CRITICAL: These are actual extracted values. Use them EXACTLY in your calculations.\n\n'
                for calc_val in calculation_values:
                    calculation_hints += (
                        f'- From {calc_val["subtask"]}: {calc_val["value"]}\n'
                    )
                    if calc_val.get('context'):
                        calculation_hints += f'  Context: {calc_val["context"]}\n'
                calculation_hints += '\n⚠️ When performing calculations, use these exact values. Do not assume or estimate different numbers.\n'
                execution_summary_str += calculation_hints

        # Format reasoning summary
        reasoning_summary_str = ''
        if reasoning_summary:
            reasoning_summary_str = '## Reasoning Summary\n'
            for key, value in reasoning_summary.items():
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + '... (truncated)'
                reasoning_summary_str += f'\n### {key}:\n{value_str}\n'

        # Format knowledge graph facts
        knowledge_facts = [
            {'entity': f.entity, 'relationship': f.relationship, 'value': f.value}
            for f in self.state_manager.knowledge_graph
        ]
        knowledge_facts_str = ''
        if knowledge_facts:
            knowledge_facts_str = '## Knowledge Graph Facts\n'
            knowledge_facts_str += json.dumps(knowledge_facts, indent=2, default=str)

        user_prompt = f"""Problem Statement:
{problem}

Answer Format Requirements:
{answer_format_str}
{requirements_str}

Your Task:
Generate a final answer that directly addresses the problem above.
Extract and synthesize information from all available sources below, then construct the answer.

CRITICAL: Your final_answer MUST strictly comply with ALL format requirements and constraints listed above.

Available Sources:{attachment_info}

{execution_summary_str}

{reasoning_summary_str}

{knowledge_facts_str}

---

Instructions:
1. Review all execution results and extract the key factual information relevant to the problem
2. Pay attention to the "Extracted Structured Data" and "Key Values for Calculations" sections above - these show the ACTUAL extracted values that MUST be used
3. For calculations: Use the EXACT values shown in "Key Values for Calculations" section. Do NOT assume, estimate, or guess different numbers
4. SHOW YOUR WORK: Identify the specific values from execution results that answer the question:
   - If question asks for a calculated number, use the extracted values shown above in your calculation
   - If question asks for a number, find and extract the exact number from results
   - If question asks for a date, find and extract the exact date matching the format requirement
   - If question asks for a word/term, find the exact word in results
   - If question asks for zip codes, extract all five-digit zip codes from results
   - **If question asks for "character name", convert any symbol (`, ., ,) to its NAME ("backtick", "dot", "comma")**
5. When performing calculations:
   - Find the extracted value in the "Key Values for Calculations" section
   - Use that EXACT value in your calculation
   - Show the calculation: extracted_value × multiplier = result
6. Synthesize this information into a coherent answer
7. STRICTLY follow the format requirements - if it says "word", return ONLY a single word; if it says "number", return ONLY a number; if it says "character name", return the NAME not the symbol
8. Ensure the final_answer field contains ONLY the ACTUAL ANSWER (no explanations, no meta-commentary, no summaries)
9. If format requires a word, extract the exact word from the results - do not add explanations
10. Before finalizing, validate that your answer matches the expected format exactly and uses the extracted values correctly

Return a JSON object with EXACTLY these two fields:
- "final_answer": The actual answer to the question (extract and use this field only)
- "description": Brief explanation of how the answer was derived

CRITICAL:
- The "final_answer" field is the ONLY answer value. Extract and use this field exclusively.
- Do NOT include dates, intermediate calculations, or context in final_answer unless they ARE the answer.
- If the question asks for a word, final_answer should contain ONLY that word.
- If the question asks for a number, final_answer should contain ONLY that number.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,  # Maximum determinism for consistent answer formatting
                response_format={'type': 'json_object'},
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

    def build_monologue(
        self,
        problem: str,
        query_analysis: Dict[str, Any],
        problem_classification: Dict[str, Any],
        plan: List[Any],
        execution_results: Dict[str, Any],
        synthesis: Dict[str, Any],
    ) -> str:
        """
        Build human-readable reasoning monologue.

        Args:
            problem: Original problem.
            query_analysis: Query analysis.
            problem_classification: Problem classification.
            plan: Execution plan.
            execution_results: Execution results.
            synthesis: Final synthesis.

        Returns:
            Formatted monologue string.
        """
        self.logger.info('Building reasoning monologue')

        system_prompt = """You are creating a human-readable reasoning monologue.
Convert the problem-solving process into a clear, step-by-step narrative in first person.

CRITICAL: You MUST use the EXACT values extracted from execution results. DO NOT assume, estimate, or guess numbers.
- If execution results show an extracted count (e.g., "1002 articles"), you MUST use that exact number
- DO NOT use phrases like "assumed", "estimated", "approximately" for values that were explicitly extracted
- When showing calculations, use the actual extracted values, not assumptions

The monologue should:
- Be written in first person ("I", "my")
- Follow markdown formatting with headers (##)
- Explain each step clearly
- Show the reasoning process
- Use ACTUAL extracted values from execution results, not assumptions
- Be professional but accessible

Format the monologue with sections like:
## Initiating Breakdown
## Step 1: [Description]
## Step 2: [Description]
...
## Final Answer"""

        # Extract key values from execution results for the monologue
        execution_summary = execution_results.get('execution_summary', {})
        extracted_values_summary = []

        for subtask_id, result in execution_summary.items():
            # Skip error results
            if isinstance(result, dict) and result.get('status') == 'failed':
                continue
            if isinstance(result, str) and (
                result.startswith('Error:')
                or result.startswith('Name Error:')
                or result.startswith('Execution Error:')
            ):
                continue

            if isinstance(result, dict):
                # Check for extracted_counts
                if 'extracted_counts' in result and result.get('extracted_counts'):
                    for count_item in result['extracted_counts']:
                        value = count_item.get('value')
                        if value is not None:
                            context = count_item.get('context', '')[:150]
                            extracted_values_summary.append(
                                f'  - Subtask {subtask_id}: Extracted value = {value}'
                                f' (Context: {context})'
                            )
                # Check for llm_extraction
                elif 'llm_extraction' in result and result.get('llm_extraction'):
                    llm_extraction = result['llm_extraction']
                    extracted_value = llm_extraction.get('extracted_value')
                    if extracted_value is not None:
                        reasoning = llm_extraction.get('reasoning', '')[:150]
                        extracted_values_summary.append(
                            f'  - Subtask {subtask_id}: Extracted value = {extracted_value}'
                            f' (Reasoning: {reasoning})'
                        )

        extracted_values_str = ''
        if extracted_values_summary:
            extracted_values_str = (
                '\n\n=== EXTRACTED VALUES FROM EXECUTION (USE THESE EXACT VALUES) ===\n'
            )
            extracted_values_str += '\n'.join(extracted_values_summary)
            extracted_values_str += '\n\nCRITICAL: These are ACTUAL extracted values. Use them exactly, do not assume or estimate different values.\n'

        # Format execution summary for monologue
        execution_summary_str = ''
        if execution_summary:
            execution_summary_str = '\n\n=== Execution Results ===\n'
            for subtask_id, result in execution_summary.items():
                # Skip error results from monologue
                if isinstance(result, dict) and result.get('status') == 'failed':
                    continue
                if isinstance(result, str) and (
                    result.startswith('Error:')
                    or result.startswith('Name Error:')
                    or result.startswith('Execution Error:')
                    or result.startswith('Import Error:')
                ):
                    continue

                if isinstance(result, dict):
                    result_str = json.dumps(
                        result, indent=2, ensure_ascii=False, default=str
                    )
                else:
                    result_str = str(result)
                # Truncate long results
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + '... (truncated)'
                execution_summary_str += f'\nSubtask {subtask_id}:\n{result_str}\n'

        user_prompt = f"""Problem: {problem}

Query Analysis: {json.dumps(query_analysis, indent=2)}

Problem Type: {problem_classification.get('primary_type', 'Unknown')}

Execution Plan: {len(plan)} steps{extracted_values_str}{execution_summary_str}

Final Synthesis: {synthesis.get('final_answer', 'N/A')}

Create a comprehensive reasoning monologue that:
1. Uses the EXACT extracted values shown above (do not assume different numbers)
2. Shows the actual calculation using the extracted values
3. Explains each step clearly
4. Does not use phrases like "assumed" or "estimated" for values that were explicitly extracted"""

        try:
            monologue = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Consistent but readable monologue generation
            )
            return monologue
        except Exception as e:
            self.logger.error(f'Monologue generation failed: {e}')
            return f"""## Initiating Breakdown
I analyzed the problem: {problem}

## Problem Classification
Problem type: {problem_classification.get('primary_type', 'Unknown')}

## Execution
Executed {len(plan)} steps with {len(execution_results)} results.

## Final Answer
{synthesis.get('final_answer', 'Unable to determine answer')}"""
