"""Query Understanding Module for parsing and analyzing problem queries."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..llm import LLMService
from ..state import Subtask
from ..utils import extract_json_from_text


class QueryUnderstanding:
    """Module for understanding and parsing problem queries."""

    def __init__(self, llm_service: LLMService, logger: logging.Logger):
        """
        Initialize Query Understanding module.

        Args:
            llm_service: LLM service instance.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.logger = logger

    def analyze(
        self,
        problem: str,
        attachments: Optional[List[Any]] = None,
        subtasks: Optional[List[Subtask]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a problem query to extract requirements and dependencies.
        Assigns step numbers to requirements based on previously generated subtasks.

        Args:
            problem: The problem description.
            attachments: Optional list of attachments.
            subtasks: Optional list of subtasks (if provided, requirements will be assigned to matching steps).

        Returns:
            Dictionary containing:
            - explicit_requirements: List of explicit requirements with step numbers
            - dependencies: List of information dependencies
            - answer_format: Expected answer format
            - cross_references: List of cross-references between data sources
        """
        self.logger.info('Analyzing query for requirements and dependencies')

        attachment_info = ''
        if attachments:
            attachment_info = f'\nAttachments: {[a.filename for a in attachments]}'

        # Build subtasks context if provided
        subtasks_context = ''
        if subtasks:
            subtasks_context = (
                '\n\nGenerated Subtasks (assign requirements to these steps):\n'
            )
            for i, subtask in enumerate(subtasks, 1):
                # Extract step number from subtask ID (e.g., "step_1" -> 1)
                step_num_match = re.search(r'step_(\d+)', subtask.id)
                step_num = step_num_match.group(1) if step_num_match else str(i)
                subtasks_context += (
                    f'  Step {step_num} (id: {subtask.id}): {subtask.description}\n'
                )
            subtasks_context += (
                '\nIMPORTANT: Assign each requirement to the step number that matches the subtask ID AND matches what that subtask is actually doing/verifying. '
                'Read each subtask description carefully to understand what it does, then assign requirements that must be satisfied BY THAT SPECIFIC STEP. '
                'Do NOT assign requirements about extracting data to a step that is about verifying/searching in a different source.\n'
            )

        system_prompt = """You are an expert at analyzing complex problem queries.
Analyze the given problem and extract:
1. Explicit requirements (stated directly)
2. Information dependencies (what information is needed and in what order)
3. Answer format requirements
4. Cross-references between different data sources
5. **Terminology context** (when terminology choice matters)

CRITICAL: TERMINOLOGY AMBIGUITY DETECTION
- Detect if the problem asks for names, terms, or identifiers that might have multiple valid forms
- Identify cases where a single concept, character, symbol, or entity may be referred to by different names or terms
- When terminology ambiguity is possible, identify the domain/context where terminology matters
- Consider that some concepts may have both common names and alternative names, or both technical and general terms
- This helps ensure answers use context-appropriate, unambiguous terminology
- The domain context helps select the most appropriate term when multiple valid options exist
- Set has_terminology_ambiguity to true whenever the answer could legitimately be expressed using different but equally valid terminology

CRITICAL RULES FOR EXPLICIT REQUIREMENTS:
- Explicit requirements are CONSTRAINTS/CONDITIONS that must be satisfied, NOT action items or tasks
- DO NOT include action verbs like "Identify", "Locate", "Determine", "Find", "Search", "Provide"
- Extract the constraints/conditions from the problem, not the action verbs
- Each requirement should describe WHAT must be true/satisfied, not WHAT to do
- Requirements should be verifiable conditions that can be checked against results
- Carefully read and parse all quantitative details in the problem statement
- Understand compound descriptions and relationships - if the problem describes multiple items where each has multiple parts, calculate the total correctly
- Preserve exact quantities, counts, and structural relationships as stated in the problem
- Do not simplify or reduce quantities when extracting requirements - maintain the full detail from the problem statement

CRITICAL: ANSWER FORMAT AND UNIT REQUIREMENTS
- The answer_format field must clearly specify the expected format and units for the final answer
- If the problem asks for a number in specific units, include this in the answer_format using general language
- Use natural, general language to describe unit requirements - avoid technical notation or abbreviations
- If the problem shows an expected answer format, infer that the answer should be in thousands and state this clearly

CRITICAL: STEP NUMBER ASSIGNMENT
- If subtasks are provided, you MUST assign each requirement to the correct step number
- Match requirements to subtasks based on WHAT EACH SUBTASK ACTUALLY DOES, not just the step number
- Read each subtask description carefully to understand its purpose and what it verifies/processes
- Assign requirements to the step that must satisfy/verify that requirement during its execution
- Use the step number from the subtask ID (e.g., subtask id="step_1" → assign as "Step 1: requirement")
- Each requirement should be tagged with "Step N:" where N matches the subtask step number
- Requirements about extracting data from a source should go to the step that extracts from that source
- Requirements about verifying/checking content in a different source should go to the step that checks that source
- Requirements that apply to multiple steps should be assigned to the most relevant step based on what it verifies
- If no subtasks are provided, analyze the problem and assign step numbers based on logical flow

Return your analysis as a JSON object with these keys:
- explicit_requirements: list of strings (constraints/conditions with step context, e.g., "Step 1: requirement text", "Step 2: requirement text")
- dependencies: list of strings describing information needs
- answer_format: string describing expected format with correct units
- cross_references: list of strings describing cross-references
- has_terminology_ambiguity: boolean (true if the answer might have multiple valid forms due to terminology)
- terminology_context: string describing the domain/context where terminology matters

Be thorough and precise. Assign requirements to the correct step numbers based on the provided subtasks.
IMPORTANT: When specifying answer_format, use clear, general language to describe units
IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Problem: {problem}{attachment_info}{subtasks_context}

Analyze this problem and assign requirements to the correct step numbers based on the generated subtasks.

CRITICAL: Read the problem statement carefully and extract all quantitative details accurately. If the problem describes multiple items where each has multiple parts, ensure requirements reflect the correct total count. Preserve all structural relationships and quantities exactly as stated in the problem.

CRITICAL: When assigning requirements to steps, carefully read each subtask description to understand what it does. Assign each requirement to the step that must satisfy/verify that requirement. For example:
- If Step 1 extracts data from Source A, assign requirements about Source A data extraction to Step 1
- If Step 2 verifies/checks content in Source B, assign requirements about verifying Source B content to Step 2
- Do NOT assign requirements about Source B to Step 1 if Step 1 only works with Source A"""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
                response_format={'type': 'json_object'},
            )
            # Extract JSON from response (might be wrapped in markdown)
            json_text = extract_json_from_text(response)
            analysis = json.loads(json_text)
            self.logger.info(
                f'Query analysis complete: {len(analysis.get("explicit_requirements", []))} '
                f'explicit requirements'
            )
            return analysis
        except json.JSONDecodeError as e:
            self.logger.warning(f'Failed to parse LLM response as JSON: {e}')
            self.logger.debug(f'Response was: {response[:500]}')
            # Fallback to basic structure
            return {
                'explicit_requirements': [problem],
                'dependencies': [],
                'answer_format': 'text',
                'cross_references': [],
            }
        except Exception as e:
            self.logger.error(f'Query analysis failed: {e}', exc_info=True)
            raise
