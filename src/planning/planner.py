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

IMPORTANT: Do NOT include tool calls, parameters, or search_queries in your response. Only generate subtask descriptions and dependencies. Tool calls will be added later based on requirements.

Subtasks schema:
- id: sequential step_1, step_2, ...
- description: self-contained instruction including action, purpose, specific data to find/process (entities/dates), constraints, expected output. The description should clearly indicate what type of operation is needed (search, API call, computation, file reading, etc.) without explicitly naming tools.
- dependencies: list of prior step IDs; [] if none
- parallelizable: boolean

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
            retry_context += '2. Do NOT include tool calls, parameters, or search_queries. Only provide descriptions and dependencies. Tool calls will be added later based on requirements.\n'

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

Each subtask description must be self-contained and include: action, purpose, specific data to find/process (entities/dates), constraints, expected output. Maintain data flow: retrieval steps state what they fetch and from where; dependent steps cite which prior step they use and how; list dependencies; no forward references. Incorporate relevant details from the problem, analysis, and classification.

IMPORTANT: Do NOT include tool names, parameters, or search_queries in your response. Only provide subtask descriptions and dependencies. Tool calls will be determined later based on requirements.

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
                # Initialize metadata without tool calls - these will be added later
                subtask.metadata = {
                    'parallelizable': task_data.get('parallelizable', False),
                }
                subtasks.append(subtask)
                self.state_manager.add_subtask(subtask)
                self.logger.debug(
                    f'Created subtask {subtask.id}: {subtask.description[:100]}...'
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
            fallback_subtask.metadata = {'parallelizable': False}
            return [fallback_subtask]
        except Exception as e:
            self.logger.error(f'Plan creation failed: {e}', exc_info=True)
            raise

    def append_tool_calls_to_subtasks(
        self,
        subtasks: List[Subtask],
        problem: str,
        query_analysis: Dict[str, Any],
    ) -> List[Subtask]:
        """
        Append tool calls, parameters, and search_queries to subtasks based on
        external requirements and subtask descriptions.

        Args:
            subtasks: List of subtasks without tool calls.
            problem: The problem description.
            query_analysis: Query analysis with external requirements.

        Returns:
            List of subtasks with tool calls appended.
        """
        self.logger.info('Appending tool calls to subtasks based on requirements')

        # Get requirements for each step
        explicit_requirements = query_analysis.get('explicit_requirements', [])
        step_requirements = {}
        general_requirements = []

        for req in explicit_requirements:
            req_str = str(req)
            if req_str.startswith('Step '):
                # Extract step number
                parts = req_str.split(':', 1)
                if len(parts) == 2:
                    step_part = parts[0].strip()
                    req_text = parts[1].strip()
                    # Extract step number (e.g., "Step 1" -> "step_1")
                    step_num = step_part.replace('Step', '').strip()
                    step_id = f'step_{step_num}'
                    if step_id not in step_requirements:
                        step_requirements[step_id] = []
                    step_requirements[step_id].append(req_text)
            else:
                general_requirements.append(req_str)

        system_prompt = """You are an expert at determining appropriate tools and parameters for subtasks.

Given a subtask description and requirements, determine:
1. The appropriate tool to use
2. Tool parameters (if needed)
3. Search queries (if using search tool)

Available tools:
- search: For information retrieval (web, archives, databases, files, PDFs)
- read_attachment: For extracting information from already-downloaded files
- llm_reasoning: For computation/analysis after retrieval
- github_api, wikipedia_api, youtube_api, twitter_api, reddit_api, arxiv_api, wayback_api, google_maps_api: For API-based information retrieval

For search tool:
- Generate exactly 3 search queries in KEYWORD-ONLY format (3-8 keywords each)
- No verbs, action words, or descriptive phrases
- Ordered by complexity: simple, normal, complex
- Include source/domain, topic, dates when relevant
- For research add "pdf"

For API tools:
- Provide parameters as a LIST of dicts
- Each dict: {"function": "method_name", "parameters": {...}}
- For chained calls: [{"function": "method1", "parameters": {...}}, {"function": "method2", "parameters": {...}}]
- Use "<from previous call>" placeholder for values from previous calls in chain

Return JSON with:
- tool: tool name
- parameters: dict or list (for API tools)
- search_queries: list of 3 queries (only for search tool)"""

        for subtask in subtasks:
            # Get requirements for this specific step
            step_reqs = step_requirements.get(subtask.id, [])
            all_reqs = step_reqs + general_requirements

            requirements_context = ''
            if all_reqs:
                requirements_context = '\n\nRequirements for this subtask:\n'
                for req in all_reqs:
                    requirements_context += f'  - {req}\n'

            user_prompt = f"""Problem: {problem}

Subtask ID: {subtask.id}
Subtask Description: {subtask.description}
Dependencies: {subtask.dependencies}
{requirements_context}

Determine the appropriate tool, parameters, and search queries (if applicable) for this subtask."""

            try:
                response = self.llm_service.call_with_system_prompt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,  # Lower temperature for consistent tool selection
                    response_format={'type': 'json_object'},
                )
                json_text = extract_json_from_text(response)
                tool_data = json.loads(json_text)

                tool_name = tool_data.get('tool', 'unknown')
                parameters = tool_data.get('parameters', {})
                search_queries = tool_data.get('search_queries', [])

                # Normalize search_queries
                if tool_name == 'search':
                    if not search_queries:
                        # Fallback: use subtask description
                        search_queries = [subtask.description]
                    # Ensure exactly 3 queries
                    while len(search_queries) < 3:
                        search_queries.append(
                            search_queries[-1]
                            if search_queries
                            else subtask.description
                        )
                    if len(search_queries) > 3:
                        search_queries = search_queries[:3]

                # Normalize API tool parameters to list format
                if tool_name.endswith('_api'):
                    if not parameters:
                        self.logger.warning(
                            f'Subtask {subtask.id} uses API tool {tool_name} but missing parameters.'
                        )
                        parameters = []
                    elif isinstance(parameters, dict):
                        # Convert single API call dict to list format
                        if 'function' in parameters:
                            parameters = [parameters]
                        elif 'method' in parameters:
                            method = parameters.pop('method')
                            params = parameters
                            parameters = [{'function': method, 'parameters': params}]
                        else:
                            # No function/method, wrap as-is
                            parameters = [{'function': None, 'parameters': parameters}]
                    elif isinstance(parameters, list):
                        # Validate list format
                        normalized_list = []
                        for call_spec in parameters:
                            if isinstance(call_spec, dict):
                                if (
                                    'function' not in call_spec
                                    and 'method' in call_spec
                                ):
                                    # Convert method to function
                                    method = call_spec.pop('method')
                                    params = call_spec
                                    normalized_list.append(
                                        {'function': method, 'parameters': params}
                                    )
                                else:
                                    normalized_list.append(call_spec)
                            else:
                                normalized_list.append(call_spec)
                        parameters = normalized_list

                # Append to subtask metadata
                subtask.metadata['tool'] = tool_name
                if parameters:
                    subtask.metadata['parameters'] = parameters
                if search_queries:
                    subtask.metadata['search_queries'] = search_queries

                self.logger.info(
                    f'Appended tool calls to {subtask.id}: tool={tool_name}, '
                    f'has_parameters={bool(parameters)}, '
                    f'has_search_queries={bool(search_queries)}'
                )

            except Exception as e:
                self.logger.error(
                    f'Failed to append tool calls to {subtask.id}: {e}', exc_info=True
                )
                # Fallback: set unknown tool
                subtask.metadata['tool'] = 'unknown'

        self.logger.info('Finished appending tool calls to all subtasks')
        return subtasks
