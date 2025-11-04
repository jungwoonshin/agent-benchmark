"""Answer Synthesis Module for constructing final answers."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .json_utils import extract_json_from_text
from .llm_service import LLMService
from .models import SearchResult
from .state_manager import InformationStateManager

if TYPE_CHECKING:
    from .models import Attachment


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

    def _validate_execution_results(
        self,
        execution_summary: Dict[str, Any],
        query_analysis: Dict[str, Any],
        problem: str,
    ) -> Dict[str, Any]:
        """
        Validate execution results for completeness and data quality using LLM.

        Args:
            execution_summary: Dictionary of execution results.
            query_analysis: Query analysis with requirements.
            problem: Original problem statement.

        Returns:
            Dictionary with validation information including:
            - is_complete: Whether critical data is present
            - missing_requirements: List of requirements not satisfied
            - data_quality: Assessment of data quality
        """
        validation = {
            'is_complete': True,
            'missing_requirements': [],
            'data_quality': 'good',
            'warnings': [],
        }

        # Check if execution results are empty
        if not execution_summary:
            validation['is_complete'] = False
            validation['missing_requirements'] = ['No execution results available']
            validation['data_quality'] = 'poor'
            validation['warnings'].append('Execution results are empty')
            return validation

        # Use LLM to validate execution results
        try:
            # Format execution results for LLM analysis
            execution_summary_str = ''
            for subtask_id, result in execution_summary.items():
                result_str = self._format_result_content(result)
                # Truncate very long results to avoid token limits
                if len(result_str) > 500:
                    result_str = result_str[:500] + '... (truncated)'
                execution_summary_str += f'\n### Subtask {subtask_id}:\n{result_str}\n'

            # Extract requirements
            explicit_requirements = query_analysis.get('explicit_requirements', [])
            implicit_requirements = query_analysis.get('implicit_requirements', [])
            all_requirements = explicit_requirements + implicit_requirements
            answer_format = query_analysis.get('answer_format', 'text')

            system_prompt = """You are an expert at validating execution results for completeness and data quality.
Analyze whether the execution results contain sufficient information to answer the problem question.

Your task is to:
1. Check if execution results are empty or contain only placeholder/stub responses
2. Determine if all critical requirements from the query analysis are addressed by the results
3. Assess whether the data quality is sufficient to extract the required answer format
4. Identify any missing information that prevents answering the question

Return a JSON object with:
- is_complete: boolean indicating if sufficient data is present to answer the question
- missing_requirements: list of requirement strings that are not satisfied by the results
- data_quality: string indicating quality level ("good", "fair", "poor")
- warnings: list of warning messages about data quality issues (e.g., stub responses, incomplete data)"""

            user_prompt = f"""Problem: {problem}

Answer Format Requirement: {answer_format}

Requirements that must be satisfied:
{chr(10).join(['- ' + req for req in all_requirements])}

Execution Results:
{execution_summary_str}

Analyze whether these execution results contain sufficient information to answer the problem.
Pay special attention to:
- Whether the results contain actual data (not just placeholders or stubs)
- Whether all explicit and implicit requirements have corresponding data in the results
- Whether the data is in a format that can be used to construct the required answer format
- Whether any critical information is missing that would prevent answering the question"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for more consistent validation
                response_format={'type': 'json_object'},
            )

            # Parse LLM response
            json_text = extract_json_from_text(response)
            llm_validation = json.loads(json_text)

            # Update validation with LLM results
            validation['is_complete'] = llm_validation.get('is_complete', True)
            validation['missing_requirements'] = llm_validation.get(
                'missing_requirements', []
            )
            validation['data_quality'] = llm_validation.get('data_quality', 'good')
            validation['warnings'] = llm_validation.get('warnings', [])

            self.logger.debug(
                f'LLM validation: is_complete={validation["is_complete"]}, '
                f'data_quality={validation["data_quality"]}, '
                f'warnings={len(validation["warnings"])}'
            )

        except Exception as e:
            self.logger.warning(f'LLM validation failed, using default validation: {e}')
            # Fallback to basic validation if LLM call fails
            stub_keywords = ['[STUB]', 'stub', 'placeholder', 'example']
            has_stubs = any(
                any(keyword in str(result).lower() for keyword in stub_keywords)
                for result in execution_summary.values()
            )
            if has_stubs:
                validation['warnings'].append(
                    'Some execution results contain placeholder/stub responses'
                )
                validation['data_quality'] = 'fair'

        return validation

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

    def _format_result_content(self, result: Any) -> str:
        """
        Format execution result content for better readability.
        Extracts actual content from objects instead of showing object representations.

        Args:
            result: Execution result (can be various types).

        Returns:
            Formatted string with actual content.
        """
        if result is None:
            return 'No result'

        # Handle lists of SearchResult objects
        if isinstance(result, list) and len(result) > 0:
            formatted_items = []
            for item in result:
                if isinstance(item, SearchResult):
                    # Extract actual content from SearchResult
                    item_str = f'Title: {item.title}\n'
                    item_str += f'URL: {item.url}\n'
                    item_str += f'Snippet: {item.snippet}\n'
                    if item.relevance_score > 0:
                        item_str += f'Relevance: {item.relevance_score:.2f}'
                    formatted_items.append(item_str)
                elif isinstance(item, dict):
                    # Format dictionaries nicely
                    formatted_items.append(
                        json.dumps(item, indent=2, ensure_ascii=False, default=str)
                    )
                else:
                    # For other types, convert to string
                    formatted_items.append(str(item))
            return '\n\n---\n\n'.join(formatted_items)

        # Handle SearchResult directly (not in a list)
        if isinstance(result, SearchResult):
            formatted = f'Title: {result.title}\n'
            formatted += f'URL: {result.url}\n'
            formatted += f'Snippet: {result.snippet}\n'
            if result.relevance_score > 0:
                formatted += f'Relevance: {result.relevance_score:.2f}'
            return formatted

        # Handle dictionaries
        if isinstance(result, dict):
            # Special handling for search results with processed content (from SearchResultProcessor)
            if 'content' in result and 'processing_summary' in result:
                highlighted = '=== PROCESSED SEARCH RESULTS (WEB PAGES + FILES) ===\n'
                
                # Show summary stats
                summary = result.get('processing_summary', {})
                highlighted += f"Processed: {summary.get('processed_count', 0)} results\n"
                highlighted += f"Relevant: {summary.get('relevant_count', 0)} results\n"
                highlighted += f"Web Pages: {len(result.get('web_pages', []))} navigated\n"
                highlighted += f"Files: {len(result.get('downloaded_files', []))} downloaded\n\n"
                
                # Show the aggregated content (this is the key extracted content!)
                content = result.get('content', '')
                if content:
                    highlighted += '=== EXTRACTED CONTENT FROM WEB PAGES AND FILES ===\n'
                    # Limit length to avoid token limits
                    max_content_length = 3000
                    if len(content) > max_content_length:
                        highlighted += content[:max_content_length] + '\n\n...[content truncated for length]...\n'
                    else:
                        highlighted += content + '\n'
                else:
                    highlighted += '[No content extracted]\n'
                
                highlighted += '\n=== FULL RESULT DETAILS ===\n'
                highlighted += json.dumps(
                    result, indent=2, ensure_ascii=False, default=str
                )[:2000]  # Limit JSON dump to avoid overwhelming output
                return highlighted
            

            # Special handling for browser_navigate errors - show diagnostics clearly
            if not result.get('success', True) and 'diagnostics' in result:
                diagnostics = result['diagnostics']
                error_details = f'ERROR: {result.get("error", "Unknown error")}\n'
                error_details += f'Status Code: {result.get("status_code", "N/A")}\n'

                if diagnostics.get('page_load_failed'):
                    error_details += (
                        f'🔴 Page Load Failed\n'
                        f'  Original Error: {diagnostics.get("original_error", "Unknown")}\n'
                        f'  Status: {diagnostics.get("status_code", "N/A")}\n'
                    )
                elif diagnostics.get('navigation_failed'):
                    error_details += (
                        f'🔴 Navigation Failed After Link Click\n'
                        f'  Target URL: {diagnostics.get("target_url", "Unknown")}\n'
                        f'  Status: {diagnostics.get("status_code", "N/A")}\n'
                    )
                elif diagnostics.get('link_found') is False:
                    error_details += (
                        f'🔴 Link Not Found on Page\n'
                        f'  Searched Text: "{diagnostics.get("searched_text", "N/A")}"\n'
                        f'  Searched Pattern: "{diagnostics.get("searched_pattern", "N/A")}"\n'
                    )
                    if diagnostics.get('similar_links_found'):
                        error_details += f'  💡 Similar Links Found: {", ".join(diagnostics["similar_links_found"][:3])}\n'
                    if diagnostics.get('sample_link_texts'):
                        sample = diagnostics['sample_link_texts'][:5]
                        error_details += (
                            f'  Sample Links on Page: {", ".join(sample)}\n'
                        )

                error_details += '\n💡 SUGGESTION: Check if:\n'
                error_details += '  - The URL is correct and accessible\n'
                error_details += '  - The link text/pattern matches exactly\n'
                error_details += '  - Authentication or special access is required\n'
                error_details += '  - Try alternative search terms or URL patterns\n'

                return (
                    error_details
                    + '\n=== FULL RESULT DETAILS ===\n'
                    + json.dumps(result, indent=2, ensure_ascii=False, default=str)
                )

            # Special handling for browser_navigate results with extracted counts
            # Highlight extracted values prominently so LLM can easily identify them
            if 'extracted_counts' in result and result.get('extracted_counts'):
                extracted_counts = result['extracted_counts']
                highlighted = '=== EXTRACTED VALUES (PRIMARY DATA) ===\n'
                for count_item in extracted_counts:
                    value = count_item.get('value', 'N/A')
                    context = count_item.get('context', '')
                    confidence = count_item.get('confidence', 'N/A')
                    reasoning = count_item.get('reasoning', '')
                    highlighted += f'EXTRACTED VALUE: {value}\n'
                    if context:
                        highlighted += f'  Context: {context[:200]}\n'
                    if confidence != 'N/A':
                        highlighted += f'  Confidence: {confidence:.2f}\n'
                    if reasoning:
                        highlighted += f'  Reasoning: {reasoning[:200]}\n'
                    highlighted += '\n'
                highlighted += '=== FULL RESULT DETAILS ===\n'
                highlighted += json.dumps(
                    result, indent=2, ensure_ascii=False, default=str
                )
                return highlighted

            # Special handling for LLM extraction results
            if 'llm_extraction' in result and result.get('llm_extraction'):
                llm_extraction = result['llm_extraction']
                extracted_value = llm_extraction.get('extracted_value')
                if extracted_value is not None:
                    highlighted = (
                        f'=== EXTRACTED VALUE (LLM EXTRACTION): {extracted_value} ===\n'
                    )
                    highlighted += (
                        f'Confidence: {llm_extraction.get("confidence", 0.0):.2f}\n'
                    )
                    highlighted += f'Reasoning: {llm_extraction.get("reasoning", "")}\n'
                    highlighted += '\n=== FULL RESULT DETAILS ===\n'
                    highlighted += json.dumps(
                        result, indent=2, ensure_ascii=False, default=str
                    )
                    return highlighted

            # Special handling for numeric_data
            if 'numeric_data' in result and result.get('numeric_data'):
                numeric_data = result['numeric_data']
                counts = numeric_data.get('counts', [])
                if counts:
                    highlighted = '=== EXTRACTED COUNTS (PRIMARY DATA) ===\n'
                    for count_item in counts[:5]:  # Limit to top 5
                        value = count_item.get('value', 'N/A')
                        context = count_item.get('context', '')
                        highlighted += f'COUNT VALUE: {value}\n'
                        if context:
                            highlighted += f'  Context: {context[:200]}\n'
                        highlighted += '\n'
                    highlighted += '=== FULL RESULT DETAILS ===\n'
                    highlighted += json.dumps(
                        result, indent=2, ensure_ascii=False, default=str
                    )
                    return highlighted

            return json.dumps(result, indent=2, ensure_ascii=False, default=str)

        # Handle strings
        if isinstance(result, str):
            return result

        # For other types, convert to string
        return str(result)

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
            - validation_info: Dictionary with validation results (is_complete, data_quality, missing_requirements, warnings)
        """
        self.logger.info('Synthesizing final report answer')

        system_prompt = """You are an expert at synthesizing final answers from multiple sources.

Your task is to generate a precise answer that:
1. Directly addresses the user's problem/question
2. Integrates all available information from execution results, knowledge graph, and attachments
3. STRICTLY follows the format requirements specified (e.g., "word" = single word only, "number" = number only)
4. Provides ONLY the factual answer without extra explanations

CRITICAL: USE EXACT EXTRACTED VALUES FOR CALCULATIONS
- When execution results show extracted values (e.g., "EXTRACTED VALUE: 1002"), you MUST use that EXACT number for calculations
- DO NOT assume, estimate, or guess different numbers when an extracted value is available
- If a calculation is needed (e.g., "1002 × 0.04"), use the exact extracted value from the results
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

ALWAYS return a JSON object with ONLY:
- "final_answer": The answer in the exact format specified (e.g., if "word" required, return just the word, not a sentence explaining the word)

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

        # Validate execution results for completeness
        validation_info = self._validate_execution_results(
            execution_summary, query_analysis, problem
        )

        # Check for failed tasks and refuse to answer if critical tasks failed
        failed_tasks = []
        all_failed = True
        for subtask_id, result in execution_summary.items():
            if isinstance(result, dict):
                # Check for errors or failures
                if result.get('success') is False or 'error' in result:
                    error_msg = result.get('error', 'Unknown error')
                    failed_tasks.append(f"{subtask_id}: {error_msg}")
                else:
                    all_failed = False
            elif isinstance(result, str) and ('error' in result.lower() or '[stub]' in result.lower()):
                failed_tasks.append(f"{subtask_id}: {result[:100]}")
            else:
                all_failed = False
        
        # If all tasks failed or validation indicates poor data quality with no usable data
        if (all_failed and failed_tasks) or (
            validation_info.get('data_quality') == 'poor' 
            and not validation_info.get('is_complete')
            and len(failed_tasks) > 0
        ):
            self.logger.error(f"Cannot answer question - {len(failed_tasks)} task(s) failed")
            for task_failure in failed_tasks:
                self.logger.error(f"  - {task_failure}")
            
            failure_details = "\n".join([f"  - {task}" for task in failed_tasks])
            refusal_message = (
                f"Unable to answer the question. The following task(s) failed:\n"
                f"{failure_details}\n\n"
                f"These failures prevented gathering the necessary information to answer your question."
            )
            return {
                'final_answer': refusal_message,
                'validation_info': validation_info
            }


        # Format execution results for better readability
        execution_summary_str = ''
        if execution_summary:
            execution_summary_str = '## Execution Results Summary\n'
            for subtask_id, result in execution_summary.items():
                result_str = self._format_result_content(result)
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

            # Add validation warnings if data quality is poor
            if validation_info.get('warnings'):
                warnings_str = '\n'.join(
                    [f'- {w}' for w in validation_info['warnings']]
                )
                execution_summary_str += (
                    f'\n## Data Quality Warnings:\n{warnings_str}\n'
                )

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
5. When performing calculations:
   - Find the extracted value in the "Key Values for Calculations" section
   - Use that EXACT value in your calculation (e.g., if extracted value is 1002, use 1002, not 500 or 800)
   - Show the calculation: extracted_value × multiplier = result
6. Synthesize this information into a coherent answer
7. STRICTLY follow the format requirements - if it says "word", return ONLY a single word; if it says "number", return ONLY a number
8. Ensure the final_answer field contains ONLY the ACTUAL ANSWER (no explanations, no meta-commentary, no summaries)
9. If format requires a word, extract the exact word from the results - do not add explanations
10. Before finalizing, validate that your answer matches the expected format exactly and uses the extracted values correctly

Return ONLY a JSON object with a "final_answer" field containing your synthesized answer."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
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
            # Extract JSON and parse it to get final_answer field
            json_text = extract_json_from_text(response)
            response_data = json.loads(json_text)
            final_answer = response_data.get('final_answer', response)

            # Format final_answer properly
            if isinstance(final_answer, (dict, list)):
                # Convert structured data to JSON string
                final_answer = json.dumps(final_answer, indent=2, ensure_ascii=False)
                self.logger.debug(
                    'Converted final_answer from structured data to JSON string'
                )
            elif isinstance(final_answer, str):
                # Remove any leading/trailing whitespace
                final_answer = final_answer.strip()
            else:
                # Convert non-string types to string
                final_answer = str(final_answer)

            # Validate and correct format compliance
            final_answer = self._validate_and_correct_format(
                final_answer, query_analysis, execution_summary
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

            # Return final_answer and validation_info
            return {'final_answer': final_answer, 'validation_info': validation_info}
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
            # Include validation info even on error
            validation_info = self._validate_execution_results(
                execution_results.get('execution_summary', {}), query_analysis, problem
            )
            return {'final_answer': fallback_answer, 'validation_info': validation_info}
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
            # Include validation info even on error
            validation_info = self._validate_execution_results(
                execution_results.get('execution_summary', {}), query_analysis, problem
            )
            return {'final_answer': fallback_answer, 'validation_info': validation_info}

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
                result_str = self._format_result_content(result)
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
                temperature=1.0,
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
