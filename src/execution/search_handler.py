"""Search result handling and processing."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..models import Attachment
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
- Create exactly 3 different search queries with varying complexity levels:
  * First query: Simple - use minimal essential keywords only
  * Second query: Normal - include standard descriptive terms and context
  * Third query: Complex - incorporate additional qualifiers, specific attributes, or detailed context
- Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
- Keep each query SHORT: 3-8 keywords maximum (typically 5-6 words)
- Remove filler words like "article", "submitted", "descriptors", "about", "related to"
- Use dates in format: "August 11 2016" or "2016-08-11" or "August 2016"
- Separate keywords with spaces, NOT commas or special formatting
- Make queries DIFFERENT from the previous ones - try alternative keyword combinations, synonyms, or different phrasings
- Focus on the core information needed to complete the subtask

Return a JSON object with:
- search_queries: array of exactly 3 different search queries in keyword-only format, ordered from simple to complex

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

            if explicit_reqs:
                requirements_context = '\n\nKey Requirements:\n'
                if explicit_reqs:
                    requirements_context += (
                        f'- Explicit Requirements: {", ".join(explicit_reqs)}\n'
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

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

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
                answer_format = query_analysis.get('answer_format', '')

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
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
2. An overall problem/question that the subtask is part of
3. Processed search results (content from web pages and files)
4. Problem requirements

Your task:
- Analyze the search results and extract TWO types of information:
  1. Information directly related to the SUBTASK DESCRIPTION - this should go in the "answer" field
  2. Information related to the OVERALL PROBLEM but not directly answering the subtask - this should go in "additional_information"

CRITICAL DISTINCTION:
- "answer" (content): Must contain information that DIRECTLY addresses what the subtask_description is asking for. This is the primary answer to the specific subtask.
- "additional_information": Should contain information that is relevant to the overall problem/question but does NOT directly answer the subtask. This includes:
  * Contextual information about the problem domain
  * Related concepts or background information
  * Image analysis results
  * Information that might be useful for other subtasks or the final answer synthesis
  * Any findings that relate to the problem but not specifically to this subtask

Return a JSON object with:
- answer: The answer or information that DIRECTLY addresses the subtask description (string). This should focus ONLY on what the subtask is asking for.
- reasoning: Brief explanation of how you arrived at this answer (string)
- confidence: Confidence level from 0.0 to 1.0 (float)
- sources_used: List of source titles/URLs that were most relevant for the answer (array of strings)
- additional_information: Information relevant to the overall problem/question but NOT directly answering the subtask. If no such information is available, use an empty string. (string)

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Overall Problem/Question: {problem}

Subtask Description: {subtask_description}
{requirements_context}

Processed Search Results:
{content_summary if content_summary else 'No content extracted from search results.'}

Materials Found ({len(materials)} total):
{chr(10).join(materials_summary) if materials_summary else 'No materials found.'}

Analyze these search results and extract:
1. Answer to the SUBTASK DESCRIPTION (put in "answer" field) - information that directly addresses what the subtask is asking for
2. Information related to the OVERALL PROBLEM but not directly answering the subtask (put in "additional_information" field) - contextual information, related concepts, or findings relevant to the problem but not this specific subtask"""

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
            additional_information = analysis_data.get('additional_information', '')

            self.logger.info(
                f'LLM analysis complete: answer length={len(answer)}, confidence={confidence:.2f}, '
                f'additional_info length={len(additional_information)}'
            )

            # Return structured result with LLM analysis as primary content
            return {
                'content': answer,  # Primary result - LLM-determined answer
                'reasoning': reasoning,
                'confidence': confidence,
                'sources_used': sources_used,
                'additional_information': additional_information,  # Extra relevant information
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

                if explicit_reqs:
                    requirements_context += (
                        f'\nExplicit Requirements: {", ".join(explicit_reqs)}'
                    )

            system_prompt = """You are an expert at evaluating whether a search result is relevant to solving a problem.
Given a problem description and a search result (title, snippet, URL), determine if this result is likely to contain information useful for solving the problem.

Return a JSON object with:
- relevant: boolean indicating if the result is relevant
- reasoning: brief explanation of why it is or isn't relevant

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

            user_prompt = f"""Problem: {problem}

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
        subtask_description: Optional[str] = None,
    ) -> None:
        """
        Process search results to find the most relevant web page or download files.

        Args:
            search_results: List of SearchResult objects from selected_indices.
            attachments: Attachments list to append downloaded files or relevant page content to.
            subtask_id: ID of the subtask that produced these results.
            problem: Original problem description.
            query_analysis: Optional query analysis results.
            subtask_description: Description of the subtask (used to construct subtask_goal).
        """
        if not self.tool_belt.browser_navigator:
            self.logger.error('BrowserNavigator not initialized in ToolBelt.')
            return

        if not subtask_description:
            self.logger.warning(
                'No subtask_description provided, using problem as fallback.'
            )
            subtask_description = problem

        self.logger.info(
            f'Starting intelligent search and navigation for subtask {subtask_id}'
        )

        # Construct subtask goal for BrowserNavigator
        subtask_goal = subtask_description
        if query_analysis:
            explicit_reqs = query_analysis.get('explicit_requirements', [])

            if explicit_reqs:
                goal_details = []
                if explicit_reqs:
                    goal_details.append(f'Explicit: {", ".join(explicit_reqs)}')
                subtask_goal += f' (Requirements: {"; ".join(goal_details)})'

        # First, try the standard search and navigation approach
        relevant_page_info = self.tool_belt.browser_navigator.find_relevant_page(
            subtask_goal
        )

        if relevant_page_info:
            self.logger.info(
                f'BrowserNavigator found a relevant page: {relevant_page_info["url"]}'
            )
            # Create an Attachment from the relevant page content and add to attachments
            page_attachment = Attachment(
                filename=f'page_content_{len(attachments) + 1}.md',
                content=relevant_page_info['content'].encode('utf-8'),  # Store as bytes
                url=relevant_page_info['url'],
                content_type='text/markdown',
                metadata={
                    'title': f'Web Page: {relevant_page_info["url"]}',
                    'summary': relevant_page_info['summary'],
                    'relevance': relevant_page_info['relevance'],
                },
            )
            attachments.append(page_attachment)
            self.logger.info(
                f'Added relevant page as attachment. Total attachments: {len(attachments)}'
            )
        else:
            self.logger.info(
                'BrowserNavigator did not find a relevant page via standard search.'
            )

            # If no relevant page found, try navigating from the most relevant homepage
            # of the selected search results
            if search_results:
                self.logger.info(
                    f'Attempting homepage navigation from {len(search_results)} selected search results.'
                )

                # Convert to SearchResult objects if needed
                from ..models import SearchResult

                selected_results = []
                for result in search_results:
                    if isinstance(result, SearchResult):
                        selected_results.append(result)
                    elif isinstance(result, dict):
                        # Convert dict to SearchResult if needed
                        selected_results.append(
                            SearchResult(
                                title=result.get('title', ''),
                                url=result.get('url', ''),
                                snippet=result.get('snippet', ''),
                                relevance_score=result.get('relevance_score', 0.0),
                            )
                        )

                if selected_results:
                    homepage_page_info = (
                        self.tool_belt.browser_navigator.navigate_from_homepage(
                            selected_results, subtask_goal
                        )
                    )

                    if homepage_page_info:
                        self.logger.info(
                            f'BrowserNavigator found a relevant page via homepage navigation: {homepage_page_info["url"]}'
                        )
                        # Create an Attachment from the homepage navigation result
                        page_attachment = Attachment(
                            filename=f'page_content_{len(attachments) + 1}.md',
                            content=homepage_page_info['content'].encode('utf-8'),
                            url=homepage_page_info['url'],
                            content_type='text/markdown',
                            metadata={
                                'title': f'Web Page (via Homepage): {homepage_page_info["url"]}',
                                'summary': homepage_page_info['summary'],
                                'relevance': homepage_page_info['relevance'],
                            },
                        )
                        attachments.append(page_attachment)
                        self.logger.info(
                            f'Added relevant page from homepage navigation as attachment. '
                            f'Total attachments: {len(attachments)}'
                        )
                    else:
                        self.logger.info(
                            'BrowserNavigator did not find a relevant page via homepage navigation.'
                        )
                else:
                    self.logger.warning(
                        'No valid SearchResult objects found for homepage navigation.'
                    )

        # Old logic for downloading files (if still needed, can be integrated or replaced)
        # For now, keeping the old download logic in case BrowserNavigator doesn't handle all file types.
        # This part could be refactored to be called by BrowserNavigator or removed if BrowserNavigator covers all cases.
        downloadable_extensions = {
            '.pdf',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.txt',
            '.csv',
        }

        # The old iteration over search_results is no longer needed since BrowserNavigator handles it.
        # The following commented out code is the original logic for reference or future integration.
        # if not search_results:
        #     return

        # self.logger.info(
        #     f'Processing {len(search_results)} search results: navigating relevant pages and downloading files...'
        # )

        # downloaded_count = 0
        # navigated_count = 0

        # def extract_paper_id(url_or_text: str) -> Optional[Tuple[str, str]]:
        # ... (same as before)
        # def get_pdf_url_from_paper_id(
        # ... (same as before)

        # for result in search_results:
        # ... (rest of old logic)

        # This part of the function will now only handle direct file downloads if `search_results` were used externally
        # and `BrowserNavigator` didn't cover it. Given the new approach, it's likely redundant or needs explicit calling.
        # I'll leave a simplified version for explicit downloads in case some non-web-navigable files are still in search_results.
        for result in search_results:
            if not isinstance(result, SearchResult):
                continue

            url = result.url
            if not url:
                continue

            url_lower = url.lower()
            is_downloadable_file = False
            for ext in downloadable_extensions:
                if ext in url_lower:
                    is_downloadable_file = True
                    break

            if is_downloadable_file:
                try:
                    self.logger.info(f'Attempting to download direct file: {url}')
                    attachment = self.tool_belt.download_file_from_url(url)
                    attachments.append(attachment)
                    self.logger.info(
                        f'Downloaded file {attachment.filename}. Total attachments: {len(attachments)}'
                    )
                except Exception as e:
                    self.logger.warning(f'Failed to download direct file {url}: {e}')
