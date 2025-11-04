"""Planning Module for generating execution strategies."""

import json
import logging
from typing import Any, Dict, List, Optional

from .json_utils import extract_json_from_text
from .llm_service import LLMService
from .state_manager import InformationStateManager, Subtask


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
  * **CRITICAL**: When you need to find PDFs or extract information from PDFs, use search with specific queries. The system will automatically download PDFs and extract text content. DO NOT try to parse PDFs with code_interpreter.
  
- **read_attachment**: Use to read files that have already been downloaded or provided:
  * Use this when you have an attachment (PDF, text file, etc.) and need to extract specific information from it
  * The system will automatically extract text content from PDFs - no code needed
  * Specify page ranges or extraction options if needed
  
- **browser_navigate**: DEPRECATED - Do not use directly
  * The search tool now handles web navigation automatically
  * Browser navigation is performed internally after search
  * No need to explicitly call browser_navigate
  
PRIORITIZATION RULES:
1. **Always use search first** for any information gathering task
2. Search will automatically process results by:
   - Checking relevance with LLM (filters out non-relevant results)
   - Navigating to web pages to extract content
   - Downloading files and extracting text
   - Handling archives, databases, and complex websites
3. **For archives and databases**: Search query should be keyword-only format (e.g., "arXiv AI regulation June 2022"), and the system will navigate to the archive and extract automatically
4. **For PDF processing**: Use search to find and download PDFs, then use read_attachment to extract information. DO NOT use code_interpreter to parse PDFs - PDF parsing libraries are not available.
5. **Only use code_interpreter** for computation, data processing, and analysis tasks (using built-in Python functions and math module - no external libraries for file parsing)

IMPORTANT: Do NOT use browser_navigate as a tool in your plan. Use 'search' instead - it handles everything automatically.

Return a JSON object with:
- subtasks: list of objects, each with:
  - id: unique identifier (e.g., "step_1", "step_2")
  - description: what needs to be done (be concise)
  - tool: which tool to use (code_interpreter, search, read_attachment, analyze_media)
    * search: Use for ALL information gathering (web pages, archives, databases, files, PDFs). The system will automatically download and extract content from PDFs.
    * code_interpreter: Use ONLY for computation, data processing, and analysis (using built-in Python functions). DO NOT use for PDF parsing or file operations - use search and read_attachment instead.
    * read_attachment: Use to read files that were already provided or downloaded. This automatically extracts text from PDFs - no code needed.
    * analyze_media: Use to analyze images, audio, or video files
  - search_query: REQUIRED STRING - The exact search query to use when searching the web for this subtask. This MUST be a concise, keyword-optimized search query (not a sentence or description).
    * FORMAT RULES:
      - Use ONLY keywords and essential terms - NO verbs, NO descriptive phrases, NO unnecessary words
      - Keep it SHORT: 3-8 keywords maximum (typically 5-6 words)
      - Remove filler words like "article", "submitted", "descriptors", "about", "related to"
      - Use dates in format: "August 11 2016" or "2016-08-11" or "August 2016"
      - Separate keywords with spaces, NOT commas or special formatting
    * EXAMPLES:
      - Good: "arXiv Physics Society August 11 2016"
      - Good: "arXiv AI regulation June 2022"
      - Good: "Python datetime documentation"
      - Good: "Tokyo population 2023"
      - Bad: "arXiv Physics and Society article submitted August 11 2016 society descriptors" (too verbose, includes unnecessary words)
      - Bad: "arXiv AI regulation papers submitted June 2022" (includes "papers submitted" - unnecessary)
      - Bad: "AI papers" (too vague, missing context)
      - Bad: "information about" (not keywords, too vague)
    * KEYWORD SELECTION:
      - Include: Domain/source (arXiv, Nature, etc.), topic keywords, dates, location (if relevant)
      - Exclude: "article", "paper", "submitted", "published", "descriptors", "about", "related", "information", "find", "search for"
  - dependencies: list of subtask IDs that must complete first (empty if none)
  - parallelizable: boolean indicating if this can run in parallel with others

CRITICAL: Every subtask MUST include a search_query field. The search_query MUST be:
- Concise keyword-only format (3-8 keywords, typically 5-6 words)
- NO verbs, NO descriptive phrases, NO filler words
- Optimized for search engines (like Google search bar queries)
- The executor will ALWAYS use this exact search_query when performing web searches - it will NOT use the description field

Order subtasks logically based on dependencies. Keep it minimal and essential."""

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
            retry_context += "Create an IMPROVED plan that addresses these issues. CRITICAL: Each subtask MUST include a search_query field in KEYWORD-ONLY format (3-8 keywords, no verbs or descriptive phrases). Example: 'arXiv Physics Society August 11 2016' NOT 'arXiv Physics and Society article submitted August 11 2016'. Remove words like 'article', 'submitted', 'descriptors', 'about'. The search tool will automatically navigate and extract from archives.\n"

        user_prompt = f"""Create a MINIMAL execution plan for this problem. Include only the essential steps (3-7 subtasks maximum).
{retry_context}
Problem: {problem}

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Problem Classification:
{json.dumps(problem_classification, indent=2)}

Generate a concise, essential execution plan with the minimum number of steps needed."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
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
                # Extract search_query from LLM response
                search_query = task_data.get('search_query', '')
                if not search_query:
                    # If search_query not provided, log warning and use description as fallback
                    self.logger.warning(
                        f'Subtask {task_data.get("id", f"step_{i}")} missing search_query. '
                        f'Using description as fallback.'
                    )
                    search_query = task_data.get('description', '')

                subtask.metadata = {
                    'tool': task_data.get('tool', 'unknown'),
                    'parallelizable': task_data.get('parallelizable', False),
                    'parameters': task_data.get('parameters', {}),
                    'search_query': search_query,  # Store LLM-generated search query
                }
                subtasks.append(subtask)
                self.state_manager.add_subtask(subtask)
                self.logger.debug(
                    f'Created subtask {subtask.id}: tool={subtask.metadata.get("tool")}, '
                    f'search_query="{search_query[:100]}..."'
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
