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

        system_prompt = """You are an expert at creating efficient, minimal number of execution plans for complex problems.
Each plan should be as detailed as possible.
Given a problem analysis and classification, create a concise execution plan with only the ESSENTIAL steps needed.

CRITICAL: Create the MINIMUM number of subtasks necessary to solve the problem. Aim for 3-7 essential subtasks maximum.

**CRITICAL: DATA FLOW AND DEPENDENCY CLARITY**

When creating subtask descriptions, you MUST follow these universal principles to ensure logical information flow:

1. **Separation of Retrieval vs. Usage**:
   - Each subtask description must clearly distinguish between what information this subtask WILL RETRIEVE/FIND (its own output) versus what information it WILL USE (from dependencies)
   - DO NOT mix these in a way that suggests information is available before it's retrieved
   - For retrieval steps: Focus on what information will be retrieved from the source
   - For dependent steps: Clearly state which step provides each piece of information used

2. **Logical Information Flow**:
   - A subtask description should ONLY reference information that will be available from its dependencies (explicitly listed in dependencies array) OR is part of the original problem statement
   - DO NOT reference information that will be produced by other subtasks unless those subtasks are listed as dependencies
   - If information from another step is needed, that step MUST be in the dependencies array

3. **Clear Source Attribution**:
   - When a subtask retrieves information from a source, clearly state what source provides it and what specific information to retrieve
   - When a subtask uses information from another step, clearly state which step provides the information and what will be done with it
   - Ensure each piece of information is attributed to its correct source or step

4. **Dependency-Driven Descriptions**:
   - If a subtask depends on other steps, its description must explicitly reference those steps and state what information will be used from each
   - If a subtask has no dependencies, its description should focus on what it retrieves, not what others will use with it
   - DO NOT create descriptions that reference information from steps that aren't listed as dependencies

5. **Forward-Reference Prevention**:
   - DO NOT describe what information will be used from future steps in a way that suggests it's available now
   - DO NOT mention "to later determine" or "for later use" in ways that reference information not yet available
   - Instead, describe what THIS step retrieves, and let dependent steps describe how they use it

6. **Unit and Format Preservation**:
   - When the problem specifies answer units or format requirements, the final subtask description MUST preserve those requirements
   - DO NOT convert to base units and then round - maintain the unit requirement from the question
   - Pay attention to unit modifiers in the question and ensure the final answer maintains those units
   - The description should explicitly state the expected unit format for the final answer

**AVOIDING REDUNDANCY - UNIVERSAL PRINCIPLES**:

Before creating your plan, analyze the problem to identify:
- What are the UNIQUE data sources needed? (each source = potentially ONE subtask)
- What are the UNIQUE processing operations? (group operations on same data)
- What is the FINAL output? (work backwards - what's the last meaningful step that produces it?)

Then apply these rules:

1. **Combine operations with the same tool and target**:
   - If multiple steps use the SAME tool (search, read_attachment, etc.) on the SAME source/target, combine them into ONE subtask
   - Examples: Multiple searches of same website/database/archive → ONE search subtask
   - Examples: Multiple reads from same document → ONE read_attachment subtask
   - Examples: Multiple calculations using same data → ONE llm_reasoning subtask
   - **Test**: Can I describe both operations in a single comprehensive instruction? If yes, combine them.

2. **Eliminate redundant processing steps**:
   - If step N already performs comparison/filtering/selection on data from step M, DO NOT add step N+1 to "compare step M with step N"
   - The step that performs the analysis IS the final result - don't add another step to restate it
   - Only create synthesis steps when truly combining INDEPENDENT results from parallel branches
   - **Test**: Does this step just reword or restate the output of a previous step? If yes, remove it.

3. **Group sequential operations on the same data**:
   - If you need to: fetch data → extract field A → extract field B, combine into ONE step "fetch data and extract fields A and B"
   - Avoid: step_1 "download PDF", step_2 "extract text from PDF" → Instead: step_1 "download PDF and extract text"
   - **Test**: Does step N+1 operate on ONLY the output of step N, using the same tool? If yes, merge them.

4. **Parallel vs Sequential - choose wisely**:
   - Create parallel subtasks ONLY when they are truly independent (no shared data dependencies)
   - If two subtasks process the SAME source/document, combine them rather than making them parallel
   - **Test**: Can these steps share a tool invocation? If yes, combine them instead of parallelizing.

- DO NOT create separate subtasks for each requirement or dependency
- DO NOT create fallback strategy subtasks upfront - handle failures during execution if needed
- Focus on the core workflow: what information to gather → how to process it → how to synthesize the answer

TOOL SELECTION GUIDELINES:
- **search**: ALWAYS PREFERRED as the primary tool for information gathering:
  * Use search to find information sources, URLs, and initial results
  * After search, the system will automatically:
    - Check relevance of each result using LLM
    - Classify results as web pages or files
    - Navigate to web pages using browser automation
    - Download files and extract content
  * Search handles ALL scenarios: archives, databases, websites, files, documents
  * **For archives with date requirements**: Still use search - the SearchResultProcessor will navigate to the archive and use advanced search features automatically
  * **The search tool now includes intelligent result processing** - it's not just finding URLs, it processes them too
  * **CRITICAL**: When you need to find PDFs or extract information from PDFs, use search with specific queries. The system will automatically download PDFs and extract text content.
  
- **read_attachment**: Use to read files that have already been downloaded or provided:
  * Use this when you have an attachment (PDF, text file, etc.) and need to extract specific information from it
  * The system will automatically extract text content from PDFs - no code needed
  * Specify page ranges or extraction options if needed

**MANDATORY: CHECK FOR SPECIALIZED DOMAIN KNOWLEDGE FIRST**
Before creating any subtask, you MUST check if the problem requires specialized domain knowledge:
- Does the problem involve programming languages, syntax, or technical specifications?
- Does it require domain-specific terminology, concepts, or knowledge?
- Does it need historical facts, technical standards, or specialized information?

If YES to any of the above, you MUST create a search subtask FIRST to retrieve that knowledge, then use llm_reasoning to apply it.

**WHEN TO SEARCH - MANDATORY RULES:**
**YOU MUST use search** when the task requires specialized domain knowledge, syntax, or specifications:
- **Programming language syntax or semantics**: ALWAYS search first when problems involve specific programming languages, esoteric languages, domain-specific languages, or specialized syntax rules - even if some syntax is mentioned in the problem, search for complete and authoritative documentation
- **Technical specifications or standards**: When problems require knowledge of specific technical standards, protocols, formats, or specifications
- **Domain-specific terminology or concepts**: When problems involve specialized terminology, concepts, or knowledge from specific fields
- **Historical or factual information**: When problems require specific historical facts, dates, events, or factual information that needs verification
- **Current or time-sensitive information**: When problems require up-to-date information, recent events, or current data

**CRITICAL**: If a problem mentions specialized domain knowledge (like programming language syntax), you MUST create a search subtask to retrieve authoritative documentation, even if partial information is provided in the problem statement. Do NOT assume the LLM has complete or accurate knowledge of specialized domains.

WHEN NOT TO SEARCH - CRITICAL GUIDELINES:
**DO NOT use search** when the task can be solved using only LLM reasoning or available information

**WHEN TO SEARCH - CRITICAL GUIDELINES:**
**ALWAYS use search** when the task requires specialized domain knowledge, syntax, or specifications that may not be in general training data:
- **Programming language syntax or semantics**: When problems involve specific programming languages, esoteric languages, domain-specific languages, or specialized syntax rules that may not be in general training data
- **Technical specifications or standards**: When problems require knowledge of specific technical standards, protocols, formats, or specifications
- **Domain-specific terminology or concepts**: When problems involve specialized terminology, concepts, or knowledge from specific fields that may not be widely known
- **Historical or factual information**: When problems require specific historical facts, dates, events, or factual information that needs verification
- **Current or time-sensitive information**: When problems require up-to-date information, recent events, or current data

**KEY PRINCIPLE**: Use search when you need to RETRIEVE information that is NOT already available OR when the problem requires specialized domain knowledge that may not be reliably available through general reasoning. If all information is present and the task uses only general knowledge, use LLM reasoning instead.
  
PRIORITIZATION RULES:
1. **Always use search first** for any information gathering task OR when specialized domain knowledge is required
2. **For specialized domain knowledge** (programming languages, technical specifications, domain-specific concepts):
   - Create a search subtask FIRST to retrieve the necessary knowledge, syntax, or specifications
   - Then use llm_reasoning to apply that knowledge to solve the problem
   - The search subtask should retrieve the domain knowledge, and the reasoning subtask should apply it
3. Search will automatically process results by:
   - Checking relevance with LLM (filters out non-relevant results)
   - Navigating to web pages to extract content
   - Downloading files and extracting text
   - Handling archives, databases, and complex websites
4. **For archives and databases**: Search query should be keyword-only format, and the system will navigate to the archive and extract automatically
5. **For PDF processing**: Use search to find and download PDFs, then use read_attachment to extract information.
6. **Use LLM reasoning** for computation, data processing, and analysis tasks AFTER the necessary information has been retrieved.

IMPORTANT: Do NOT use browser_navigate as a tool in your plan. Use 'search' instead - it handles everything automatically.

CRITICAL: SUBTASK ID FORMAT
- Use sequential step IDs: step_1, step_2, step_3, etc.
- The first subtask you create should be step_1, second should be step_2, etc.
- Do NOT skip step numbers - create step_1, step_2, step_3 in order
- Requirements will be assigned to these steps after generation

Return a JSON object with:
- subtasks: list of objects, each with:
  - id: unique identifier (e.g., "step_1", "step_2") - use sequential numbering starting from step_1
  - description: COMPLETE, SELF-CONTAINED instruction that includes:
    * What needs to be done (clear action verb and objective)
    * What specific information/data to find, process, or analyze (include key terms, dates, entities, requirements)
    * Any constraints, formats, or requirements (e.g., date ranges, specific sources, output format)
    * CRITICAL: The description must be complete enough that an LLM can process it WITHOUT needing the full problem context
    * Include relevant details from the problem: specific dates, entities, requirements, formats mentioned in the problem
    * For search tasks: specify what information to find
    * For llm_reasoning tasks: specify what calculation/analysis to perform and what data to use
    * For read_attachment tasks: specify what information to extract from which file
  - tool: which tool to use (llm_reasoning, search, read_attachment, analyze_media)
    * search: Use for ALL information gathering (web pages, archives, databases, files, PDFs). The system will automatically download and extract content from PDFs.
    * llm_reasoning: Use for computation, data processing, analysis, and reasoning tasks. This replaces code_interpreter with LLM-based problem solving.
    * read_attachment: Use to read files that were already provided or downloaded. This automatically extracts text from PDFs - no code needed.
  - search_queries: (ONLY for 'search' tool) - Array of exactly 3 different search queries with varying complexity levels. OMIT this field entirely for non-search tools.
    * COMPLEXITY LEVELS (apply to the 3 queries in order):
      - First query: Simple - use minimal essential keywords only
      - Second query: Normal - include standard descriptive terms and context
      - Third query: Complex - incorporate additional qualifiers, specific attributes, or detailed context
    * FORMAT RULES (apply to each of the 3 queries):
      - Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
      - Keep it SHORT: 3-8 keywords maximum (typically 5-6 words)
      - Remove filler words
      - Separate keywords with spaces, NOT commas or special formatting
      - **For paper/research exploration**: If the subtask involves finding academic papers, articles, preprints, or research documents, add "pdf" to prioritize PDF files
    * KEYWORD SELECTION:
      - Include: Domain/source (arXiv, Nature, etc.), topic keywords, dates, location (if relevant)
      - Use different keyword combinations or phrasings for each query
      - Use general, broad keywords rather than overly specific terms
      - Avoid specific measurement or quantification terms that narrow the search too much
      - Focus on core concepts, entities, and topics rather than specific attributes or metrics
      - The search tool will navigate to the source and extract the information automatically, so queries should be general enough to find relevant sources
  - dependencies: list of subtask IDs that must complete first (empty array [] if none)
  - parallelizable: boolean indicating if this can run in parallel with others


Order subtasks logically based on dependencies. Keep it minimal and essential. YOU MUST AVOID REDUNDANT SUBTASKS.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

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
            retry_context += "2. Each subtask with tool='search' MUST include a search_queries array with exactly 3 different search queries in KEYWORD-ONLY format (3-8 keywords each, no verbs or descriptive phrases), ordered by complexity: simple (minimal keywords), normal (standard terms), complex (additional qualifiers). Use general, broad keywords rather than overly specific terms. Avoid specific measurement or quantification terms that narrow the search too much. Focus on core concepts, entities, and topics. Remove words like 'article', 'submitted', 'descriptors', 'about'. The search tool will automatically navigate and extract from archives.\n"

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

        user_prompt = f"""Create a MINIMAL execution plan for this problem. Include only the essential steps (3-7 subtasks maximum).
{retry_context}
Problem: {problem}

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Problem Classification:
{json.dumps(problem_classification, indent=2)}
{step_classifications_info}
{general_requirements_info}

⚠️ MANDATORY CHECK BEFORE CREATING SUBTASKS:
Before creating any subtask, you MUST determine if this problem requires specialized domain knowledge:
- Does it involve programming languages, syntax, or technical specifications?
- Does it require domain-specific terminology, concepts, or knowledge?
- Does it need historical facts, technical standards, or specialized information?

If YES, you MUST create a search subtask FIRST (step_1) to retrieve that knowledge, then create a reasoning subtask (step_2) to apply it.
Do NOT skip the search step even if some information is mentioned in the problem - always search for authoritative documentation.

CRITICAL REQUIREMENT FOR SUBTASK DESCRIPTIONS:
Each subtask description must be COMPLETE and SELF-CONTAINED. It must include:
1. What to do (clear action)
2. Why it's needed (context from the problem)
3. What specific information/data to find or process (include key terms, dates, entities from the problem)
4. Any constraints or requirements (date ranges, formats, sources mentioned in the problem)
5. Expected output or criteria
6. Incorporate any general requirements listed above (requirements will be assigned to specific steps after generation)

**CRITICAL: DATA FLOW CLARITY IN DESCRIPTIONS**
- For subtasks with NO dependencies: Focus on what information THIS step retrieves from its source. State the source clearly and what specific information will be retrieved. Do NOT reference information from other steps that haven't been retrieved yet.
- For subtasks WITH dependencies: Clearly state which step provides each piece of information used, and what action will be performed with it. Each referenced step MUST be listed in the dependencies array.
- DO NOT create descriptions that reference information from steps that aren't listed as dependencies.
- DO NOT use forward references that suggest information is available before it's retrieved.
- Ensure logical flow: retrieval steps describe what they retrieve; dependent steps describe what they use from dependencies.

Incorporate relevant details from the problem, query analysis, and problem classification into EACH subtask description. The description should be detailed enough that an LLM can execute it without needing to see the full problem context.

Generate a concise, essential execution plan with the minimum number of steps needed.

FINAL REMINDER:
- If the problem requires specialized domain knowledge (programming languages, syntax, technical specifications, domain-specific concepts), you MUST create a search subtask FIRST
- Use search for information retrieval and specialized domain knowledge
- Use llm_reasoning for computation, analysis, and applying retrieved knowledge AFTER search has retrieved it"""

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

                subtask.metadata = {
                    'tool': task_data.get('tool', 'unknown'),
                    'parallelizable': task_data.get('parallelizable', False),
                    'parameters': task_data.get('parameters', {}),
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
