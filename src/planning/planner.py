"""Planning Module for generating execution strategies."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..llm import LLMService
from ..state import InformationStateManager, Subtask
from ..utils import extract_json_from_text


class Planner:
    """Generates execution plans and strategies."""

    def __init__(
        self,
        llm_service: LLMService,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize Planner.

        Args:
            llm_service: LLM service instance.
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.state_manager = state_manager
        self.logger = logger

    def create_plan(
        self,
        problem: str,
        query_analysis: Dict[str, Any],
        problem_classification: Dict[str, Any],
        previous_plan: Optional[List[Subtask]] = None,
        missing_requirements: Optional[List[str]] = None,
        validation_warnings: Optional[List[str]] = None,
    ) -> List[Subtask]:
        """
        Create an execution plan from problem analysis.

        Args:
            problem: The problem description.
            query_analysis: Query analysis from QueryUnderstanding.
            problem_classification: Problem classification.

        Returns:
            List of Subtask objects representing the execution plan.
        """
        self.logger.info('Creating execution plan')

        system_prompt = """You are an expert planner. Produce the smallest correct plan (3–7 essential subtasks). Output JSON only.

Core rules:
- Minimal steps; order by dependencies. No redundant or fallback steps.
- Clear data flow: retrieval steps state what they fetch and from where; dependent steps state which prior step they use and how. No forward references.
- Preserve any required units and formats in the final output.

Tooling:
- search: Use for information retrieval (web, archives, databases, files, PDFs). The system handles relevance, navigation, downloads, and PDF text extraction.
- read_attachment: Use for extracting information from already-downloaded files.
- llm_reasoning: Use for computation/analysis after retrieval.
- Prefer API tools when applicable (github_api, wikipedia_api, youtube_api, twitter_api, reddit_api, arxiv_api, wayback_api, google_maps_api). Do NOT use browser_navigate.

Search-first rule:
- If specialized knowledge is involved (programming syntax, standards, domain-specific concepts, historical/factual/current info), create step_1: search to retrieve authoritative info, then step_2: llm_reasoning to apply it.

Subtasks schema:
- id: sequential step_1, step_2, ...
- description: self-contained instruction including action, purpose, specific data to find/process (entities/dates), constraints, expected output.
- tool: one of llm_reasoning, search, read_attachment, analyze_media, github_api, wikipedia_api, youtube_api, twitter_api, reddit_api, arxiv_api, wayback_api, google_maps_api.
- parameters: REQUIRED for API tools. MUST be a list of dicts, each with "function" (API method name) and "parameters" (dict with method parameters).
  - Single API call: [{"function": "method_name", "parameters": {...}}]
  - Multiple chained API calls: [{"function": "method1", "parameters": {...}}, {"function": "method2", "parameters": {...}}]
  - For Wikipedia with year requirements: [{"function": "get_page_revisions", "parameters": {"title": "...", "start_date": "YYYY-01-01", "end_date": "YYYY-12-31", "limit": 1}}, {"function": "get_page", "parameters": {"title": "...", "revision_id": "<from previous call>"}}]
  - CRITICAL: Always use list format, even for single API calls. Each dict must have "function" and "parameters" keys.
- search_queries: ONLY for 'search'. Exactly 3 queries:
  - Complexity: simple, normal, complex (in this order)
  - Format: keyword-only, 3–5 words, no verbs/action words/descriptions/unnecessary words; separate with spaces; for research add "pdf"
  - Keyword selection: include source/domain, topic, dates; use broad terms; do not add action words (count/find/get/retrieve/search/list) or measurement terms (number/total/amount) unless intrinsic to the topic; focus on what to search for (entities/topics/sources), not what to do with results
- dependencies: list of prior step IDs; [] if none
- parallelizable: boolean

API methods (reference):
- github_api: search_issues, get_issue, get_repository_commit, get_repository_contents
- wikipedia_api: get_page, search_pages, get_page_revisions
- youtube_api: get_video_info, search_videos
- twitter_api: get_user_tweets
- reddit_api: get_user_posts, search_posts
- arxiv_api: get_metadata
- wayback_api: get_archived_url
- google_maps_api: get_place_details, get_street_view_image

Return: JSON object with 'subtasks' only."""

        # Build user prompt with context about previous attempts if retrying
        retry_context = ''
        if previous_plan and (missing_requirements or validation_warnings):
            retry_context = '\n\n⚠️ RETRY MODE: Previous execution failed validation.\n'
            if missing_requirements:
                retry_context += f'Missing requirements that must be addressed: {", ".join(missing_requirements[:5])}\n'
            if validation_warnings:
                retry_context += (
                    f'Validation warnings: {"; ".join(validation_warnings[:3])}\n'
                )
            retry_context += 'Create an IMPROVED plan that addresses these issues. CRITICAL REQUIREMENTS:\n'
            retry_context += "1. Each subtask description must be COMPLETE and SELF-CONTAINED, including what to do, why it's needed, what specific information/data to find/process, constraints, and expected output.\n"
            retry_context += "2. Each subtask with tool='search' MUST include a search_queries array with exactly 3 different search queries in KEYWORD-ONLY format (3-8 keywords each, no verbs, no action words, no descriptive phrases), ordered by complexity: simple (minimal keywords), normal (standard terms), complex (additional qualifiers). Use general, broad keywords rather than overly specific terms. CRITICAL: Do NOT add action words (count, find, get, retrieve, search, list, etc.) or measurement terms (number, total, amount, etc.) unless they are core to the topic itself. Focus on WHAT to search for (entities, topics, sources), not WHAT to do with the results. The search tool will automatically navigate and extract from archives.\n"
            retry_context += '3. Each subtask with an API tool (github_api, wikipedia_api, youtube_api, twitter_api, reddit_api, arxiv_api, wayback_api, google_maps_api) MUST include a \'parameters\' field as a LIST of dicts. Each dict must have "function" (method name) and "parameters" (method parameters). For single API calls: [{"function": "method_name", "parameters": {...}}]. For chained calls: [{"function": "method1", "parameters": {...}}, {"function": "method2", "parameters": {...}}]. For Wikipedia with year requirement: [{"function": "get_page_revisions", "parameters": {"title": "Page Title", "start_date": "2022-01-01", "end_date": "2022-12-31", "limit": 1}}, {"function": "get_page", "parameters": {"title": "Page Title", "revision_id": "<from previous call>"}}].\n'

        # Extract step classifications if available
        step_classifications_info = ''
        step_classifications = problem_classification.get('step_classifications', [])
        if step_classifications:
            step_classifications_info = (
                '\n\nStep-Level Classification (for reference):\n'
            )
            for i, step in enumerate(step_classifications, 1):
                search_indicator = (
                    '🔍 REQUIRES SEARCH'
                    if step.get('requires_search', False)
                    else '🧠 LLM-ONLY (no search)'
                )
                step_classifications_info += (
                    f'  Step {i}: {step.get("step_description", "N/A")}\n'
                    f'    Type: {step.get("step_type", "N/A")}\n'
                    f'    {search_indicator}\n'
                    f'    Reasoning: {step.get("reasoning", "N/A")}\n\n'
                )
            step_classifications_info += (
                'IMPORTANT: Use this step breakdown to guide your plan. '
                'Steps marked "LLM-ONLY" should use llm_reasoning, NOT search.\n'
            )

        # Note: Step-tagged requirements are not available at planning time
        # Requirements will be assigned to steps after subtasks are generated
        # Include general requirements if any exist
        general_requirements_info = ''
        explicit_requirements = query_analysis.get('explicit_requirements', [])
        if explicit_requirements:
            # Only show general requirements (non-step-tagged) at planning time
            general_reqs = [
                req for req in explicit_requirements if not str(req).startswith('Step ')
            ]
            if general_reqs:
                general_requirements_info = (
                    '\n\nGeneral Requirements (apply to all steps):\n'
                )
                for req in general_reqs:
                    general_requirements_info += f'  - {req}\n'
                general_requirements_info += '\n'

        user_prompt = f"""Create a minimal plan (2–7 subtasks) for this problem.
{retry_context}
Problem: {problem}

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Problem Classification:
{json.dumps(problem_classification, indent=2)}
{step_classifications_info}
{general_requirements_info}

Before planning, check if specialized knowledge is required (programming syntax/standards, domain concepts, historical/factual/current info). If yes, make step_1 a 'search' subtask to retrieve authoritative info, then add an 'llm_reasoning' subtask to apply it.

Each subtask description must be self-contained and include: action, purpose, specific data to find/process (entities/dates), constraints, expected output. Maintain data flow: retrieval steps state what they fetch and from where; dependent steps cite which prior step they use and how; list dependencies; no forward references. Incorporate relevant details from the problem, analysis, and classification.

Generate the smallest correct plan."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,  # Balanced creativity for flexible planning
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            plan_data = json.loads(json_text)
            subtasks = []

            for i, task_data in enumerate(plan_data.get('subtasks', []), 1):
                subtask = Subtask(
                    id=task_data.get('id', f'step_{i}'),
                    description=task_data.get('description', ''),
                    dependencies=task_data.get('dependencies', []),
                )
                # Extract search_queries from LLM response (prefer new format, fallback to old)
                search_queries = task_data.get('search_queries', [])
                tool_type = task_data.get('tool', 'unknown')

                # Handle backward compatibility: if search_query (singular) exists, convert to array
                if not search_queries:
                    old_search_query = task_data.get('search_query', '')
                    if old_search_query:
                        search_queries = [old_search_query]
                        self.logger.debug(
                            f'Subtask {task_data.get("id", f"step_{i}")} uses old search_query format. '
                            f'Converted to search_queries array with 1 query.'
                        )

                # Only warn if it's a search tool without search queries
                if tool_type == 'search' and not search_queries:
                    self.logger.warning(
                        f'Subtask {task_data.get("id", f"step_{i}")} uses search tool but missing search_queries. '
                        f'Using description as fallback for single query.'
                    )
                    search_queries = [task_data.get('description', '')]

                # Ensure we have exactly 3 queries for search tools
                if tool_type == 'search' and len(search_queries) < 3:
                    # If we have fewer than 3, duplicate the last one to reach 3
                    while len(search_queries) < 3:
                        search_queries.append(
                            search_queries[-1]
                            if search_queries
                            else task_data.get('description', '')
                        )
                    self.logger.warning(
                        f'Subtask {task_data.get("id", f"step_{i}")} has only {len([q for q in search_queries if q])} unique search queries. '
                        f'Padded to 3 queries.'
                    )
                elif tool_type == 'search' and len(search_queries) > 3:
                    # If we have more than 3, take the first 3
                    search_queries = search_queries[:3]
                    self.logger.debug(
                        f'Subtask {task_data.get("id", f"step_{i}")} has {len(search_queries)} search queries. '
                        f'Using first 3.'
                    )

                # Extract parameters
                parameters = task_data.get('parameters', {})

                # Normalize API tool parameters to list format
                if tool_type.endswith('_api'):
                    if not parameters:
                        self.logger.warning(
                            f'Subtask {task_data.get("id", f"step_{i}")} uses API tool {tool_type} but missing parameters. '
                            f'Parameters will need to be determined during execution.'
                        )
                    elif isinstance(parameters, dict):
                        # Convert single API call dict to list format
                        if 'method' in parameters:
                            # Old format: {"method": "...", ...params}
                            method = parameters.pop('method')
                            params = parameters
                            parameters = [{'function': method, 'parameters': params}]
                            self.logger.debug(
                                f'Converted single API call to list format: {method}'
                            )
                        elif 'function' in parameters:
                            # Already has function, wrap in list
                            parameters = [parameters]
                            self.logger.debug(
                                'Wrapped API call with function in list format'
                            )
                        else:
                            # No method/function, try to infer or keep as-is (will be handled later)
                            self.logger.warning(
                                f'Subtask {task_data.get("id", f"step_{i}")} API tool parameters missing "function" or "method". '
                                f'Will attempt to infer during execution. Parameters: {parameters}'
                            )
                            # Wrap in list format anyway
                            parameters = [{'function': None, 'parameters': parameters}]
                    elif isinstance(parameters, list):
                        # List format - validate and normalize each call
                        normalized_list = []
                        for call_idx, call_spec in enumerate(parameters, 1):
                            if not isinstance(call_spec, dict):
                                self.logger.warning(
                                    f'Subtask {task_data.get("id", f"step_{i}")} API call {call_idx} is not a dict: {call_spec}'
                                )
                                normalized_list.append(call_spec)
                                continue

                            # Normalize to {function, parameters} format
                            if 'function' in call_spec:
                                # Already in correct format
                                if 'parameters' not in call_spec:
                                    # Move all other keys to parameters
                                    func = call_spec['function']
                                    params = {
                                        k: v
                                        for k, v in call_spec.items()
                                        if k != 'function'
                                    }
                                    normalized_list.append(
                                        {'function': func, 'parameters': params}
                                    )
                                else:
                                    normalized_list.append(call_spec)
                            elif 'method' in call_spec:
                                # Convert method to function
                                method = call_spec['method']
                                if 'parameters' in call_spec:
                                    normalized_list.append(
                                        {
                                            'function': method,
                                            'parameters': call_spec['parameters'],
                                        }
                                    )
                                else:
                                    # Move all other keys to parameters
                                    params = {
                                        k: v
                                        for k, v in call_spec.items()
                                        if k != 'method'
                                    }
                                    normalized_list.append(
                                        {'function': method, 'parameters': params}
                                    )
                            else:
                                # No function/method - will need to infer
                                self.logger.warning(
                                    f'Subtask {task_data.get("id", f"step_{i}")} API call {call_idx} missing "function" or "method". '
                                    f'Will attempt to infer during execution.'
                                )
                                normalized_list.append(
                                    {'function': None, 'parameters': call_spec}
                                )
                        parameters = normalized_list

                subtask.metadata = {
                    'tool': task_data.get('tool', 'unknown'),
                    'parallelizable': task_data.get('parallelizable', False),
                    'parameters': parameters,
                    'search_queries': search_queries,  # Store LLM-generated search queries (array of 3)
                }
                subtasks.append(subtask)
                self.state_manager.add_subtask(subtask)
                queries_preview = (
                    ', '.join([f'"{q[:30]}..."' for q in search_queries[:3]])
                    if search_queries
                    else 'none'
                )
                self.logger.debug(
                    f'Created subtask {subtask.id}: tool={subtask.metadata.get("tool")}, '
                    f'search_queries=[{queries_preview}]'
                )

            # Note: Step-tagged requirements validation is skipped at planning time
            # Requirements will be assigned to steps after subtasks are generated in query_understanding
            # Validation of step alignment will happen later when requirements are used

            self.logger.info(f'Created execution plan with {len(subtasks)} subtasks')
            return subtasks
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse plan response: {e}')
            # Fallback to simple plan
            fallback_subtask = Subtask(
                id='step_1',
                description='Analyze problem and determine approach',
                dependencies=[],
            )
            fallback_subtask.metadata = {'tool': 'unknown', 'parallelizable': False}
            return [fallback_subtask]
        except Exception as e:
            self.logger.error(f'Plan creation failed: {e}', exc_info=True)
            raise
