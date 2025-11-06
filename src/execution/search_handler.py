"""Search result handling and processing."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..models import Attachment, SearchResult
from ..tools import ToolBelt
from ..utils import extract_json_from_text


class SearchHandler:
    """Handles search result filtering, query generation, and processing."""

    def __init__(
        self,
        tool_belt: ToolBelt,
        llm_service: Any,
        logger: logging.Logger,
    ):
        """
        Initialize SearchHandler.

        Args:
            tool_belt: ToolBelt instance with available tools.
            llm_service: LLM service for search result analysis.
            logger: Logger instance.
        """
        self.tool_belt = tool_belt
        self.llm_service = llm_service
        self.logger = logger

    def filter_search_results_by_relevance(
        self,
        search_results: List[Any],
        query: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], bool]:
        """
        Use LLM to filter and rank search results by relevance to the query.

        Args:
            search_results: List of SearchResult objects from search tool.
            query: The search query that produced these results.
            problem: The original problem being solved.
            query_analysis: Optional query analysis results containing requirements and constraints.

        Returns:
            Tuple of (filtered list of most relevant SearchResult objects, was_selected_indices_empty).
            was_selected_indices_empty is True if the LLM returned 0 selected indices.
        """
        if not search_results:
            return search_results, False

        self.logger.info(
            f'Filtering {len(search_results)} search results by relevance using LLM'
        )

        # Build a formatted list of search results for the LLM
        results_list = []
        for i, result in enumerate(search_results):
            if isinstance(result, SearchResult):
                results_list.append(
                    {
                        'index': i,
                        'title': result.title,
                        'snippet': result.snippet,
                        'url': result.url,
                    }
                )
            elif isinstance(result, dict):
                # Handle already-serialized SearchResult dictionaries
                results_list.append(
                    {
                        'index': i,
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'url': result.get('url', ''),
                    }
                )

        # Extract requirement information from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            implicit_reqs = query_analysis.get('implicit_requirements', [])
            constraints = query_analysis.get('constraints', {})
            answer_format = query_analysis.get('answer_format', '')

            requirements_context = '\n\nKey Requirements from Query Analysis:\n'
            if explicit_reqs:
                requirements_context += (
                    f'- Explicit Requirements: {", ".join(explicit_reqs)}\n'
                )
            if implicit_reqs:
                requirements_context += (
                    f'- Implicit Requirements: {", ".join(implicit_reqs)}\n'
                )
            if constraints:
                constraints_str = []
                if constraints.get('temporal'):
                    constraints_str.append(
                        f'Temporal: {", ".join(constraints["temporal"])}'
                    )
                if constraints.get('spatial'):
                    constraints_str.append(
                        f'Spatial: {", ".join(constraints["spatial"])}'
                    )
                if constraints.get('categorical'):
                    constraints_str.append(
                        f'Categorical: {", ".join(constraints["categorical"])}'
                    )
                if constraints_str:
                    requirements_context += (
                        f'- Constraints: {"; ".join(constraints_str)}\n'
                    )
            if answer_format:
                requirements_context += f'- Expected Answer Format: {answer_format}\n'

        system_prompt = """You are an expert at evaluating search result relevance.
Given a search query/subtask description, query requirements analysis, and a list of search results, identify which results are relevant to answering the query.

Consider the explicit and implicit requirements from the query analysis when determining relevance.
A result is relevant if it helps satisfy any of the stated requirements or constraints.

Return a JSON object with:
- selected_indices: list of integers representing the indices (0-based) of ALL relevant results, ordered by relevance (most relevant first)
- reasoning: brief explanation of why these results were selected, specifically referencing which requirements they address

IMPORTANT:
- Select ALL results that are relevant to the query, not just a fixed number
- Include any result that could help answer the query or satisfy the requirements
- Order results by relevance (most relevant first)
- Return your response as valid JSON only, without any markdown formatting or additional text

Focus on:
- Direct relevance to the query/subtask terms and intent
- Alignment with explicit and implicit requirements from query analysis
- How well each result addresses the specific requirements and constraints
- Information quality and usefulness for satisfying the problem requirements
- For aggregate/statistical queries (e.g., "how many articles"), prioritize archive/browse pages over individual articles"""

        user_prompt = f"""Original Problem: {problem}

Subtask/Query: {query}
{requirements_context}

Search Results (combined from multiple search queries):
{json.dumps(results_list, indent=2)}

Identify ALL relevant search results for answering the query/subtask. Pay special attention to which results satisfy the requirements identified in the query analysis. Return the indices of all relevant results, ordered by relevance (most relevant first). Include any result that could help answer the query or satisfy the requirements."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative but focused for search result filtering
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            selected_indices = result_data.get('selected_indices', [])
            reasoning = result_data.get('reasoning', 'No reasoning provided')
            was_selected_indices_empty = len(selected_indices) == 0

            self.logger.info(
                f'LLM selected {len(selected_indices)} most relevant result(s) out of {len(search_results)}. Reasoning: {reasoning}'
            )

            # Filter results to only include selected indices, preserving order
            filtered_results = []
            seen_indices = set()  # Avoid duplicates
            for idx in selected_indices:
                if 0 <= idx < len(search_results) and idx not in seen_indices:
                    filtered_results.append(search_results[idx])
                    seen_indices.add(idx)
                elif idx not in seen_indices:
                    self.logger.warning(
                        f'Invalid index {idx} in selected_indices (valid range: 0-{len(search_results) - 1})'
                    )

            # If no valid results were selected, return all original results as fallback
            if not filtered_results:
                self.logger.warning(
                    'No valid results selected by LLM, using all original results as fallback'
                )
                return search_results, was_selected_indices_empty

            self.logger.info(
                f'LLM filtered {len(search_results)} results down to {len(filtered_results)} most relevant'
            )
            return filtered_results, was_selected_indices_empty

        except Exception as e:
            self.logger.error(
                f'Failed to filter search results by relevance: {e}', exc_info=True
            )
            # Fallback to original results if filtering fails
            self.logger.warning('Using all original search results as fallback')
            return search_results, False

    def generate_new_search_queries(
        self,
        subtask_description: str,
        problem: str,
        previous_queries: List[str],
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate new search queries when initial search returned 0 selected indices.

        Args:
            subtask_description: Description of the subtask.
            problem: Original problem being solved.
            previous_queries: List of previous search queries that didn't yield results.
            query_analysis: Optional query analysis results.

        Returns:
            List of 3 new search queries.
        """
        self.logger.info(
            f'Generating new search queries for subtask. Previous queries: {previous_queries}'
        )

        system_prompt = """You are an expert at creating effective search queries.
Given a subtask description, problem context, and previous search queries that didn't yield relevant results, create 3 NEW and DIFFERENT search queries.

CRITICAL REQUIREMENTS:
- Create exactly 3 different search queries
- Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
- Keep each query SHORT: 3-8 keywords maximum (typically 5-6 words)
- Remove filler words like "article", "submitted", "descriptors", "about", "related to"
- Use dates in format: "August 11 2016" or "2016-08-11" or "August 2016"
- Separate keywords with spaces, NOT commas or special formatting
- Make queries DIFFERENT from the previous ones - try alternative keyword combinations, synonyms, or different phrasings
- Focus on the core information needed to complete the subtask

Return a JSON object with:
- search_queries: array of exactly 3 different search queries in keyword-only format

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        # Build context about previous queries
        previous_queries_context = ''
        if previous_queries:
            previous_queries_context = (
                '\n\nPrevious search queries that did not yield relevant results:\n'
            )
            for i, query in enumerate(previous_queries, 1):
                previous_queries_context += f'{i}. {query}\n'
            previous_queries_context += '\nCreate NEW queries with different keyword combinations or phrasings.\n'

        # Extract requirement information from query analysis
        requirements_context = ''
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])
            implicit_reqs = query_analysis.get('implicit_requirements', [])
            constraints = query_analysis.get('constraints', {})

            if explicit_reqs or implicit_reqs or constraints:
                requirements_context = '\n\nKey Requirements:\n'
                if explicit_reqs:
                    requirements_context += (
                        f'- Explicit Requirements: {", ".join(explicit_reqs)}\n'
                    )
                if implicit_reqs:
                    requirements_context += (
                        f'- Implicit Requirements: {", ".join(implicit_reqs)}\n'
                    )
                if constraints:
                    constraints_str = []
                    if constraints.get('temporal'):
                        constraints_str.append(
                            f'Temporal: {", ".join(constraints["temporal"])}'
                        )
                    if constraints.get('spatial'):
                        constraints_str.append(
                            f'Spatial: {", ".join(constraints["spatial"])}'
                        )
                    if constraints.get('categorical'):
                        constraints_str.append(
                            f'Categorical: {", ".join(constraints["categorical"])}'
                        )
                    if constraints_str:
                        requirements_context += (
                            f'- Constraints: {"; ".join(constraints_str)}\n'
                        )

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}
{requirements_context}
{previous_queries_context}

Create 3 NEW search queries (different from previous ones) that will help find information to complete this subtask. Use keyword-only format (3-8 keywords each)."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative for generating alternative queries
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            new_queries = result_data.get('search_queries', [])
            if not new_queries or len(new_queries) < 3:
                self.logger.warning(
                    f'LLM generated only {len(new_queries)} queries, padding to 3'
                )
                # Pad with variations of the subtask description
                while len(new_queries) < 3:
                    new_queries.append(subtask_description)
            elif len(new_queries) > 3:
                new_queries = new_queries[:3]
                self.logger.debug(
                    f'LLM generated {len(new_queries)} queries, using first 3'
                )

            self.logger.info(
                'Generated %d new search queries: %s', len(new_queries), new_queries
            )
            return new_queries

        except Exception as e:
            self.logger.error(
                f'Failed to generate new search queries: {e}', exc_info=True
            )
            # Fallback: create variations from subtask description
            self.logger.warning(
                'Using fallback: creating query variations from subtask description'
            )
            # Simple fallback: use subtask description with slight variations
            base_query = subtask_description
            return [base_query, base_query, base_query]

    def identify_downloadable_resources(
        self, problem: str, subtask_description: str, query: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to identify downloadable resources (PDFs, documents, etc.) from problem description.
        This is a general-purpose method that works for any type of resource.

        Args:
            problem: Original problem description.
            subtask_description: Description of the subtask.
            query: The search query.

        Returns:
            List of dictionaries with resource information:
            - url: Direct URL to the resource
            - title: Descriptive title
            - description: Description of the resource
            - relevance: Relevance score (0-1)
        """
        system_prompt = """You are an expert at identifying downloadable resources from problem descriptions.
Given a problem description, identify any resources that need to be downloaded to solve the problem.

Resources could include:
- Academic papers (preprint repositories, PubMed, etc.)
- Government documents
- Reports or datasets
- Documents mentioned by URL, ID, or citation
- Any downloadable files referenced in the problem

For each resource you identify, provide:
- url: The direct download URL if you can construct it (e.g., for preprint repositories: https://domain.org/pdf/PAPER_ID.pdf)
- title: A descriptive title
- description: Brief description of the resource
- relevance: Relevance score from 0.0 to 1.0

Return a JSON object with key "resources" containing an array of resource objects.
If you cannot construct a direct URL but can identify a resource, include it with url as null and provide as much info as possible.
If no resources are found, return {"resources": []}.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.

Examples:
- "Preprint paper 2207.01510" -> {"url": "https://repository.org/pdf/2207.01510.pdf", "title": "Preprint paper 2207.01510", "description": "Preprint paper with ID 2207.01510", "relevance": 1.0}
- "Paper submitted to preprint repository in June 2022" -> {"url": null, "title": "June 2022 preprint submission", "description": "Paper submitted to preprint repository in June 2022", "relevance": 0.8}
- "Document at https://example.com/doc.pdf" -> {"url": "https://example.com/doc.pdf", "title": "Document from example.com", "description": "PDF document", "relevance": 1.0}"""

        user_prompt = f"""Problem: {problem}

Subtask: {subtask_description}

Search Query: {query}

Identify any downloadable resources (papers, documents, PDFs, etc.) mentioned in the above text.
Return as JSON with "resources" key containing an array of resource objects."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Creative but focused for resource identification
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            data = json.loads(json_text)

            resources = data.get('resources', [])
            if not isinstance(resources, list):
                resources = []

            # Filter to only include resources with URLs (can be downloaded)
            downloadable_resources = [
                r for r in resources if isinstance(r, dict) and r.get('url')
            ]

            if downloadable_resources:
                self.logger.info(
                    f'Identified {len(downloadable_resources)} downloadable resource(s)'
                )
            return downloadable_resources
        except Exception as e:
            self.logger.debug(f'Resource identification failed: {e}')
            return []

    def analyze_search_results_with_llm(
        self,
        processing_result: Dict[str, Any],
        materials: List[Dict[str, Any]],
        subtask_description: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze processed search results and determine the answer based on subtask description.

        Args:
            processing_result: Result from SearchResultProcessor containing content_summary, web_pages, etc.
            materials: List of materials (web pages, files) extracted from search results.
            subtask_description: Description of the subtask being executed.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            Dictionary with LLM analysis result as the primary content.
        """
        try:
            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                implicit_reqs = query_analysis.get('implicit_requirements', [])
                constraints = query_analysis.get('constraints', {})
                answer_format = query_analysis.get('answer_format', '')

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )
                if implicit_reqs:
                    requirements_context += (
                        f'\nImplicit Requirements: {", ".join(implicit_reqs)}'
                    )
                if constraints:
                    constraints_str = []
                    if constraints.get('temporal'):
                        constraints_str.append(
                            f'Temporal: {", ".join(constraints["temporal"])}'
                        )
                    if constraints.get('spatial'):
                        constraints_str.append(
                            f'Spatial: {", ".join(constraints["spatial"])}'
                        )
                    if constraints.get('categorical'):
                        constraints_str.append(
                            f'Categorical: {", ".join(constraints["categorical"])}'
                        )
                    if constraints_str:
                        requirements_context += (
                            f'\nConstraints: {"; ".join(constraints_str)}'
                        )
                if answer_format:
                    requirements_context += f'\nAnswer Format: {answer_format}'

            # Prepare content summary from processed results
            content_summary = processing_result.get('content_summary', '')

            # Build materials summary with full content (no truncation)
            materials_summary = []
            for material in materials:
                material_type = material.get('type', 'unknown')
                title = material.get('title', 'Untitled')
                url = material.get('url', '')
                content = material.get('content', '') or ''

                # Include full content for complete context
                if content:
                    materials_summary.append(
                        f'[{material_type.upper()}] {title}\nURL: {url}\nContent: {content}'
                    )
                else:
                    materials_summary.append(
                        f'[{material_type.upper()}] {title}\nURL: {url}'
                    )

            system_prompt = """You are an expert at analyzing search results and extracting information relevant to a specific task.

Given:
1. A subtask description that needs to be completed
2. Processed search results (content from web pages and files)
3. Problem requirements and constraints

Your task:
- Analyze the search results with respect to the subtask description
- Extract and determine the answer or information that addresses the subtask
- Provide a clear, focused response that directly answers what the subtask is asking for
- If the search results don't contain the needed information, clearly state that
- If multiple pieces of information are found, synthesize them appropriately

Return a JSON object with:
- answer: The answer or information determined from the search results (string)
- reasoning: Brief explanation of how you arrived at this answer (string)
- confidence: Confidence level from 0.0 to 1.0 (float)
- sources_used: List of source titles/URLs that were most relevant (array of strings)

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}

Subtask Description: {subtask_description}
{requirements_context}

Processed Search Results:
{content_summary if content_summary else 'No content extracted from search results.'}

Materials Found ({len(materials)} total):
{chr(10).join(materials_summary) if materials_summary else 'No materials found.'}

Analyze these search results with respect to the subtask description and determine the answer or information that addresses what the subtask is asking for."""

            self.logger.info(
                f'Analyzing search results with LLM for subtask: {subtask_description[:100]}...'
            )

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent analysis
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            analysis_data = json.loads(json_text)

            answer = analysis_data.get(
                'answer', 'No answer determined from search results.'
            )
            reasoning = analysis_data.get('reasoning', 'No reasoning provided.')
            confidence = analysis_data.get('confidence', 0.0)
            sources_used = analysis_data.get('sources_used', [])

            self.logger.info(
                f'LLM analysis complete: answer length={len(answer)}, confidence={confidence:.2f}'
            )

            # Return structured result with LLM analysis as primary content
            return {
                'content': answer,  # Primary result - LLM-determined answer
                'reasoning': reasoning,
                'confidence': confidence,
                'sources_used': sources_used,
                'analysis_type': 'search_results_analysis',
            }

        except Exception as e:
            self.logger.error(
                f'Failed to analyze search results with LLM: {e}', exc_info=True
            )
            # Fallback: return content summary as result
            return {
                'content': processing_result.get('content_summary', ''),
                'reasoning': f'LLM analysis failed: {str(e)}',
                'confidence': 0.0,
                'sources_used': [],
                'analysis_type': 'search_results_analysis_fallback',
            }

    def is_file_url(self, url: str) -> bool:
        """
        Check if a URL points to a downloadable file.

        Args:
            url: URL to check.

        Returns:
            True if URL appears to point to a file, False otherwise.
        """
        if not url:
            return False

        url_lower = url.lower()

        # Check for file extensions
        file_extensions = {
            '.pdf',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.txt',
            '.csv',
            '.jpg',
            '.jpeg',
            '.png',
            '.gif',
            '.bmp',
            '.svg',
            '.webp',
            '.zip',
            '.tar',
            '.gz',
            '.rar',
            '.7z',
            '.mp3',
            '.mp4',
            '.avi',
            '.mov',
            '.wmv',
            '.ppt',
            '.pptx',
        }

        # Check if URL contains file extension
        for ext in file_extensions:
            if ext in url_lower:
                return True

        # Check if URL ends with common file patterns
        if any(url_lower.endswith(ext) for ext in file_extensions):
            return True

        # Check for patterns like /pdf/, /download/, etc. that often indicate files
        file_indicators = ['/pdf/', '/download/', '/file/', '/attachment/']
        if any(indicator in url_lower for indicator in file_indicators):
            return True

        return False

    def is_result_relevant(
        self,
        search_result: Any,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Use LLM to determine if a search result is relevant to the problem.

        Args:
            search_result: SearchResult to evaluate.
            problem: Original problem description.
            query_analysis: Optional query analysis results.

        Returns:
            True if relevant, False otherwise.
        """
        try:
            # Build context from query analysis
            requirements_context = ''
            if query_analysis:
                explicit_reqs = query_analysis.get('explicit_requirements', [])
                implicit_reqs = query_analysis.get('implicit_requirements', [])

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )
                if implicit_reqs:
                    requirements_context += (
                        f'\nImplicit Requirements: {", ".join(implicit_reqs)}'
                    )

            system_prompt = """You are an expert at evaluating whether a search result is relevant to solving a problem.
Given a problem description and a search result (title, snippet, URL), determine if this result is likely to contain information useful for solving the problem.

Return a JSON object with:
- relevant: boolean indicating if the result is relevant
- reasoning: brief explanation of why it is or isn't relevant

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}
{requirements_context}

Search Result:
- Title: {search_result.title}
- Snippet: {search_result.snippet}
- URL: {search_result.url}

Is this search result relevant to solving the problem?"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for consistent evaluation
                response_format={'type': 'json_object'},
            )

            json_text = extract_json_from_text(response)
            result_data = json.loads(json_text)

            relevant = result_data.get('relevant', False)
            reasoning = result_data.get('reasoning', 'No reasoning provided')

            self.logger.info(
                f'Relevance check for {search_result.url}: {relevant}. Reasoning: {reasoning}'
            )

            return relevant
        except Exception as e:
            self.logger.warning(
                f'Failed to determine relevance using LLM: {e}. Defaulting to relevant=True.'
            )
            # Default to relevant if LLM check fails to avoid skipping potentially useful pages
            return True

    def process_search_results_for_downloads(
        self,
        search_results: List[Any],
        attachments: List[Attachment],
        subtask_id: str,
        problem: str,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Process search results: navigate to relevant non-file pages, download file URLs.

        Args:
            search_results: List of SearchResult objects from search tool.
            attachments: Attachments list to append downloaded files to.
            subtask_id: ID of the subtask that produced these results.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
        """
        if not search_results:
            return

        self.logger.info(
            f'Processing {len(search_results)} search results: navigating relevant pages and downloading files...'
        )

        downloadable_extensions = {
            '.pdf',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.txt',
            '.csv',
        }

        downloaded_count = 0
        navigated_count = 0

        def extract_paper_id(url_or_text: str) -> Optional[Tuple[str, str]]:
            """
            Extract paper/preprint ID from URL or text.

            Returns:
                Tuple of (paper_id, repository_type) if found, None otherwise.
                repository_type can be 'preprint' (for preprint repositories) or None.
            """
            # Pattern for preprint IDs: YYMM.NNNNN or archive-category/YYMMNNN format
            # This pattern works for preprint repositories using similar ID formats
            preprint_patterns = [
                # Direct URL patterns (domain-agnostic - works for any domain)
                r'(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # abs/2207.01510 or pdf/2207.01510
                r'(?:abs|pdf)/([a-z-]+/\d{7}(?:v\d+)?)',  # abs/cs/1234567 or pdf/math/1234567
                # Citation patterns (common formats across repositories)
                r'[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)',  # :2207.01510 or :2207.01510
                r'[:\s]+([a-z-]+/\d{7}(?:v\d+)?)',  # :cs/1234567
                r'\.(\d{4}\.\d{4,5}(?:v\d+)?)',  # .2207.01510
                r'\.([a-z-]+/\d{7}(?:v\d+)?)',  # .cs/1234567
                r'[:\.](\d{4}\.\d{4,5}(?:v\d+)?)',  # :2207.01510 or .2207.01510
            ]

            for pattern in preprint_patterns:
                match = re.search(pattern, url_or_text, re.IGNORECASE)
                if match:
                    paper_id = match.group(1)
                    # Remove version suffix if present (e.g., v1, v2)
                    paper_id = re.sub(r'v\d+$', '', paper_id)
                    # Detect repository type based on ID format
                    # YYMM.NNNNN or archive/category format suggests preprint repository
                    if re.match(r'^\d{4}\.\d{4,5}$', paper_id) or '/' in paper_id:
                        return (paper_id, 'preprint')
                    return (paper_id, None)
            return None

        def get_pdf_url_from_paper_id(
            paper_id: str, repository_type: str, original_url: str = ''
        ) -> Optional[str]:
            """
            Construct PDF URL from paper ID based on repository type.

            Args:
                paper_id: The extracted paper ID
                repository_type: Type of repository ('preprint' for preprint repositories, etc.)
                original_url: Original URL from search result (used to infer domain)

            Returns:
                PDF URL if repository type is known and domain can be inferred, None otherwise
            """
            if repository_type == 'preprint':
                # Try to infer base URL from original URL
                if original_url:
                    try:
                        parsed = urlparse(original_url)
                        if parsed.netloc:
                            # Use the domain from original URL
                            base_url = f'{parsed.scheme}://{parsed.netloc}'
                            return f'{base_url}/pdf/{paper_id}.pdf'
                    except Exception:
                        pass

                # If we can't infer domain from URL, we cannot construct a valid download URL
                # without domain-specific knowledge
                return None
            return None

        for result in search_results:
            if not isinstance(result, SearchResult):
                continue

            url = result.url
            if not url:
                continue

            # Check if URL looks like a downloadable file
            url_lower = url.lower()
            is_downloadable = False
            download_url = url

            # First, check by file extension (direct downloadable files)
            for ext in downloadable_extensions:
                if ext in url_lower:
                    is_downloadable = True
                    break

            # If not directly downloadable, try to extract paper ID from URL or snippet
            if not is_downloadable:
                # Try extracting paper ID from URL first
                paper_info = extract_paper_id(url)

                # If not found in URL, try snippet
                if not paper_info:
                    paper_info = extract_paper_id(result.snippet or '')

                if paper_info:
                    paper_id, repository_type = paper_info
                    # Construct PDF URL from paper ID
                    download_url = get_pdf_url_from_paper_id(
                        paper_id, repository_type, url
                    )
                    if download_url:
                        is_downloadable = True
                        self.logger.info(
                            f'Extracted paper ID {paper_id} from URL/snippet, constructing PDF URL: {download_url}'
                        )

            # Determine if URL is a file (using new helper method)
            is_file = self.is_file_url(url) or is_downloadable

            if is_file:
                # It's a file - download it
                try:
                    self.logger.info(
                        f'Downloading file from search result: {download_url}'
                    )
                    attachment = self.tool_belt.download_file_from_url(download_url)
                    attachments.append(attachment)
                    downloaded_count += 1
                    self.logger.info(
                        f'Added attachment {attachment.filename} from search result '
                        f'(subtask {subtask_id}). Total attachments: {len(attachments)}'
                    )
                except Exception as e:
                    self.logger.warning(
                        f'Failed to download file from {download_url}: {e}. Continuing...'
                    )
                    continue
            else:
                # Not a file - check relevance and navigate if relevant
                try:
                    is_relevant = self.is_result_relevant(
                        result, problem, query_analysis
                    )
                    if is_relevant:
                        self.logger.info(f'Navigating to relevant non-file page: {url}')
                        # Navigate to the page (this will load content and add to state)
                        nav_result = self.tool_belt.browser_navigate(url)
                        navigated_count += 1
                        self.logger.info(
                            f'Successfully navigated to {url}. '
                            f'Page loaded: {nav_result.get("success", False)}'
                        )
                    else:
                        self.logger.info(f'Skipping non-relevant page: {url}')
                except Exception as e:
                    self.logger.warning(
                        f'Failed to process search result {url}: {e}. Continuing...'
                    )
                    continue

        if downloaded_count > 0 or navigated_count > 0:
            self.logger.info(
                f'Processed {len(search_results)} search results: '
                f'Downloaded {downloaded_count} file(s), navigated to {navigated_count} page(s). '
                f'Total attachments now: {len(attachments)}'
            )
