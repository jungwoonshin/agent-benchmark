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
- Group related operations into single subtasks when possible
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
  - description: what needs to be done (be concise)
  - tool: which tool to use (llm_reasoning, search, read_attachment, analyze_media)
    * search: Use for ALL information gathering (web pages, archives, databases, files, PDFs). The system will automatically download and extract content from PDFs.
    * llm_reasoning: Use for computation, data processing, analysis, and reasoning tasks. This replaces code_interpreter with LLM-based problem solving.
    * read_attachment: Use to read files that were already provided or downloaded. This automatically extracts text from PDFs - no code needed.
    * analyze_media: Use to analyze images, audio, or video files
  - search_queries: REQUIRED ARRAY OF 3 STRINGS (only for subtasks using 'search' tool) - Generate exactly 3 different search queries to try for this subtask. Each query should approach the information need from a different angle to maximize coverage.
    * REQUIRED ONLY when tool is 'search' - for llm_reasoning or LLM-only tasks, use empty array [] or omit
    * CRITICAL: Generate exactly 3 different search queries that:
      - Use different keyword combinations or phrasings
      - Approach the information need from different angles (e.g., one focusing on domain, one on topic, one on specific details)
      - Each query should be concise and keyword-optimized
    * FORMAT RULES (apply to each of the 3 queries):
      - Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
      - Keep it SHORT: 3-8 keywords maximum (typically 5-6 words)
      - Remove filler words like "article", "submitted", "descriptors", "about", "related to"
      - Use dates in format: "August 11 2016" or "2016-08-11" or "August 2016"
      - Separate keywords with spaces, NOT commas or special formatting
    * KEYWORD SELECTION:
      - Include: Domain/source (arXiv, Nature, etc.), topic keywords, dates, location (if relevant)
      - DO NOT BE TOO SPECIFIC - THE SEARCH TOOL WILL NAVIGATE TO THE SOURCE AND EXTRACT THE INFORMATION AUTOMATICALLY
  - dependencies: list of subtask IDs that must complete first (empty if none)
  - parallelizable: boolean indicating if this can run in parallel with others

CRITICAL: The search_queries field is REQUIRED ONLY for subtasks using the 'search' tool:
- For subtasks with tool='search': MUST include a search_queries array with exactly 3 different search queries in concise keyword-only format (3-8 keywords each, no verbs or descriptive phrases). Example: ['arXiv Physics Society August 11 2016', 'arXiv Physics Society 2016', 'Physics Society arXiv August'] NOT ['arXiv Physics and Society article submitted August 11 2016 society descriptors']. Remove words like 'article', 'submitted', 'descriptors', 'about'. The search tool will automatically navigate and extract from archives.
- For subtasks with tool='llm_reasoning' or other non-search tools: Use empty array [] for search_queries or omit it
- Each search query in the array MUST be:
  - NO verbs, NO descriptive phrases, NO filler words
  - Optimized for search engines (like Google search bar queries)
  - Different from the other queries (different angles/approaches)
- The executor will try all 3 search queries and combine results - it will NOT use the description field

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
            retry_context += "Create an IMPROVED plan that addresses these issues. CRITICAL: Each subtask with tool='search' MUST include a search_queries array with exactly 3 different search queries in KEYWORD-ONLY format (3-8 keywords each, no verbs or descriptive phrases). Example: ['arXiv Physics Society August 11 2016', 'arXiv Physics Society 2016', 'Physics Society arXiv August'] NOT ['arXiv Physics and Society article submitted August 11 2016']. Remove words like 'article', 'submitted', 'descriptors', 'about'. The search tool will automatically navigate and extract from archives.\n"

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
