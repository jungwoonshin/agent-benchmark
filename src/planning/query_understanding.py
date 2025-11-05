"""Query Understanding Module for parsing and analyzing problem queries."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..llm import LLMService
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
        self, problem: str, attachments: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a problem query to extract requirements and dependencies.

        Args:
            problem: The problem description.
            attachments: Optional list of attachments.

        Returns:
            Dictionary containing:
            - explicit_requirements: List of explicit requirements
            - implicit_requirements: List of implicit requirements
            - dependencies: List of information dependencies
            - constraints: Dictionary of constraints (temporal, spatial, categorical)
            - answer_format: Expected answer format
            - cross_references: List of cross-references between data sources
        """
        self.logger.info('Analyzing query for requirements and dependencies')

        attachment_info = ''
        if attachments:
            attachment_info = f'\nAttachments: {[a.filename for a in attachments]}'

        system_prompt = """You are an expert at analyzing complex problem queries.
Analyze the given problem and extract:
1. Explicit requirements (stated directly)
2. Implicit requirements (inferred from context)
3. Information dependencies (what information is needed and in what order)
4. Constraints (temporal, spatial, categorical)
5. Answer format requirements
6. Cross-references between different data sources

Return your analysis as a JSON object with these keys:
- explicit_requirements: list of strings
- implicit_requirements: list of strings
- dependencies: list of strings describing information needs
- constraints: object with keys: temporal, spatial, categorical (each a list)
- answer_format: string describing expected format
- cross_references: list of strings describing cross-references

Be thorough and precise.

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Problem: {problem}{attachment_info}

Analyze this problem and provide a structured breakdown of requirements and dependencies."""

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
                f'explicit, {len(analysis.get("implicit_requirements", []))} implicit requirements'
            )
            return analysis
        except json.JSONDecodeError as e:
            self.logger.warning(f'Failed to parse LLM response as JSON: {e}')
            self.logger.debug(f'Response was: {response[:500]}')
            # Fallback to basic structure
            return {
                'explicit_requirements': [problem],
                'implicit_requirements': [],
                'dependencies': [],
                'constraints': {'temporal': [], 'spatial': [], 'categorical': []},
                'answer_format': 'text',
                'cross_references': [],
            }
        except Exception as e:
            self.logger.error(f'Query analysis failed: {e}', exc_info=True)
            raise
