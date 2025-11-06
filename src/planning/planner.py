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
  * **For archives with date requirements** (e.g., "arXiv papers from June 2022"): Still use search - the SearchResultProcessor will navigate to the archive and use advanced search features automatically
  * **The search tool now includes intelligent result processing** - it's not just finding URLs, it processes them too
  * **CRITICAL**: When you need to find PDFs or extract information from PDFs, use search with specific queries. The system will automatically download PDFs and extract text content.
  
- **read_attachment**: Use to read files that have already been downloaded or provided:
  * Use this when you have an attachment (PDF, text file, etc.) and need to extract specific information from it
  * The system will automatically extract text content from PDFs - no code needed
  * Specify page ranges or extraction options if needed
  
- **browser_navigate**: DEPRECATED - Do not use directly
  * The search tool now handles web navigation automatically
  * Browser navigation is performed internally after search
  * No need to explicitly call browser_navigate

WHEN NOT TO SEARCH - CRITICAL GUIDELINES:
**DO NOT use search** when the task can be solved using only LLM reasoning or available information:
- **Pure logical deduction**: Problems that can be solved through reasoning alone (e.g., "If A implies B and B implies C, what does A imply?")
- **Mathematical calculations**: When all required numbers/formulas are provided or can be derived mathematically (e.g., "Calculate 15% of 200")
- **Text analysis of provided content**: When you already have all the text/data needed (e.g., analyzing text from attachments, comparing provided documents)
- **Format conversions**: When converting data formats using LLM reasoning (e.g., date format conversions, unit conversions)
- **Data filtering/processing**: When filtering or processing data that's already available (use LLM reasoning)
- **Answer synthesis**: When combining already-retrieved information to form an answer (use LLM reasoning)
- **Judgment/evaluation tasks**: When the task requires only LLM judgment on provided information (e.g., "Is this text positive or negative?" when text is already available)

**KEY PRINCIPLE**: Only use search when you need to RETRIEVE information that is NOT already available. If all information is present or can be derived through reasoning/computation, use LLM reasoning instead.

**Examples of tasks that DO NOT need search**:
- "Calculate the average of these numbers: [1, 2, 3, 4, 5]" → Use LLM reasoning
- "What is 25% of 80?" → Use LLM reasoning
- "Analyze the sentiment of this text: [text provided]" → Use LLM reasoning
- "Compare these two documents: [both provided]" → Use read_attachment + LLM reasoning
- "Extract dates from this PDF: [already downloaded]" → Use read_attachment
- "If it rains on Monday and Tuesday, what days will it rain?" → Use LLM reasoning
  
PRIORITIZATION RULES:
1. **Always use search first** for any information gathering task
2. Search will automatically process results by:
   - Checking relevance with LLM (filters out non-relevant results)
   - Navigating to web pages to extract content
   - Downloading files and extracting text
   - Handling archives, databases, and complex websites
3. **For archives and databases**: Search query should be keyword-only format (e.g., "arXiv AI regulation June 2022"), and the system will navigate to the archive and extract automatically
4. **For PDF processing**: Use search to find and download PDFs, then use read_attachment to extract information.
5. **Use LLM reasoning** for computation, data processing, and analysis tasks.

IMPORTANT: Do NOT use browser_navigate as a tool in your plan. Use 'search' instead - it handles everything automatically.

Return a JSON object with:
- subtasks: list of objects, each with:
  - id: unique identifier (e.g., "step_1", "step_2")
  - description: COMPLETE, SELF-CONTAINED instruction that includes:
    * What needs to be done (clear action verb and objective)
    * What specific information/data to find, process, or analyze (include key terms, dates, entities, requirements)
    * Any constraints, formats, or requirements (e.g., date ranges, specific sources, output format)
    * CRITICAL: The description must be complete enough that an LLM can process it WITHOUT needing the full problem context
    * Include relevant details from the problem: specific dates, entities, requirements, formats mentioned in the problem
    * For search tasks: specify what information to find (e.g., "Find arXiv papers about X submitted in Y month")
    * For llm_reasoning tasks: specify what calculation/analysis to perform and what data to use
    * For read_attachment tasks: specify what information to extract from which file
  - tool: which tool to use (llm_reasoning, search, read_attachment, analyze_media)
    * search: Use for ALL information gathering (web pages, archives, databases, files, PDFs). The system will automatically download and extract content from PDFs.
    * llm_reasoning: Use for computation, data processing, analysis, and reasoning tasks. This replaces code_interpreter with LLM-based problem solving.
    * read_attachment: Use to read files that were already provided or downloaded. This automatically extracts text from PDFs - no code needed.
    * analyze_media: Use to analyze images, audio, or video files
  - search_queries: (ONLY for 'search' tool) - Array of exactly 3 different search queries. OMIT this field entirely for non-search tools.
    * FORMAT RULES (apply to each of the 3 queries):
      - Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
      - Keep it SHORT: 3-8 keywords maximum (typically 5-6 words)
      - Remove filler words like "article", "submitted", "descriptors", "about", "related to"
      - Use dates in format: "August 11 2016" or "2016-08-11" or "August 2016"
      - Separate keywords with spaces, NOT commas or special formatting
    * KEYWORD SELECTION:
      - Include: Domain/source (arXiv, Nature, etc.), topic keywords, dates, location (if relevant)
      - Use different keyword combinations or phrasings for each query
      - DO NOT BE TOO SPECIFIC - THE SEARCH TOOL WILL NAVIGATE TO THE SOURCE AND EXTRACT THE INFORMATION AUTOMATICALLY
  - dependencies: list of subtask IDs that must complete first (empty array [] if none)
  - parallelizable: boolean indicating if this can run in parallel with others

CRITICAL RULES TO AVOID REDUNDANCY:
1. **search_queries field** (JSON structure requirement):
   - For tool='search': MUST include search_queries array with exactly 3 different queries
   - For tool='llm_reasoning', 'read_attachment', or 'analyze_media': OMIT search_queries entirely (do not include empty array)
   
2. **Identify and eliminate duplicate patterns** (applies to ALL tools):
   Before creating multiple subtasks, ask: "Do these steps use the same tool on the same target?"
   
   Pattern A - Same tool, same source:
   - BAD: step_1 "Search [source] for X" + step_N "Search [source] for Y"
   - GOOD: step_1 "Search [source] for X and Y" (list all requirements in one description)
   
   Pattern B - Same document, multiple extractions:
   - BAD: step_1 "Read doc and extract A" + step_N "Read doc and extract B"
   - GOOD: step_1 "Read doc and extract A and B"
   
   Pattern C - Same data, multiple calculations:
   - BAD: step_1 "Calculate X from data" + step_N "Calculate Y from data"
   - GOOD: step_1 "Calculate X and Y from data"
   
3. **Eliminate redundant comparison/output steps**:
   Before creating a "compare" or "output" step, ask: "Does the previous step already produce this result?"
   
   - BAD: step_N "find item from list A that matches criteria in B" + step_N+1 "compare list A with result from step_N"
   - GOOD: step_N "find item from list A that matches criteria in B" (this IS the final answer)
   
   - BAD: step_N "analyze and select the answer" + step_N+1 "output the result from step_N"
   - GOOD: step_N "analyze and select the answer" (already outputs it)
   
   Only create synthesis steps when combining results from TRULY INDEPENDENT parallel branches, not when restating a previous result.

Order subtasks logically based on dependencies. Keep it minimal and essential.

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
            retry_context += "2. Each subtask with tool='search' MUST include a search_queries array with exactly 3 different search queries in KEYWORD-ONLY format (3-8 keywords each, no verbs or descriptive phrases). Example: ['arXiv Physics Society August 11 2016', 'arXiv Physics Society 2016', 'Physics Society arXiv August'] NOT ['arXiv Physics and Society article submitted August 11 2016']. Remove words like 'article', 'submitted', 'descriptors', 'about'. The search tool will automatically navigate and extract from archives.\n"

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

        user_prompt = f"""Create a MINIMAL execution plan for this problem. Include only the essential steps (3-7 subtasks maximum).
{retry_context}
Problem: {problem}

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Problem Classification:
{json.dumps(problem_classification, indent=2)}
{step_classifications_info}

CRITICAL REQUIREMENT FOR SUBTASK DESCRIPTIONS:
Each subtask description must be COMPLETE and SELF-CONTAINED. It must include:
1. What to do (clear action)
2. Why it's needed (context from the problem)
3. What specific information/data to find or process (include key terms, dates, entities from the problem)
4. Any constraints or requirements (date ranges, formats, sources mentioned in the problem)
5. Expected output or criteria

Incorporate relevant details from the problem, query analysis, and problem classification into EACH subtask description. The description should be detailed enough that an LLM can execute it without needing to see the full problem context.

Generate a concise, essential execution plan with the minimum number of steps needed.
Remember: Only use search for steps that require information retrieval. Use llm_reasoning for computation, analysis, and LLM-only judgment tasks."""

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
