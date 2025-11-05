"""Reasoning Engine for inference and hypothesis generation."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..state import InformationStateManager
from ..utils import extract_json_from_text
from .llm_service import LLMService


class ReasoningEngine:
    """Handles pattern matching, constraint propagation, and hypothesis generation."""

    def __init__(
        self,
        llm_service: LLMService,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize Reasoning Engine.

        Args:
            llm_service: LLM service instance.
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.llm_service = llm_service
        self.state_manager = state_manager
        self.logger = logger

    def analyze_patterns(
        self, data: List[Any], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify patterns and similarities across data.

        Args:
            data: List of data items to analyze.
            context: Optional context description.

        Returns:
            Dictionary with pattern analysis results.
        """
        self.logger.info('Analyzing patterns in data')

        system_prompt = """You are an expert at pattern recognition and entity matching.
Analyze the given data to identify:
- Similar structures across different contexts
- Entity equivalences (same entity mentioned differently)
- Recurring patterns or relationships
- Inconsistencies or contradictions

Return a JSON object with:
- patterns: list of identified patterns
- entity_mappings: dictionary mapping equivalent entities
- inconsistencies: list of detected contradictions
- relationships: list of discovered relationships

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Data to analyze:
{json.dumps(data, indent=2, default=str)}

{context if context else ''}

Identify patterns and relationships."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Consistent logical reasoning
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            return json.loads(json_text)
        except Exception as e:
            self.logger.error(f'Pattern analysis failed: {e}', exc_info=True)
            return {
                'patterns': [],
                'entity_mappings': {},
                'inconsistencies': [],
                'relationships': [],
            }

    def propagate_constraints(
        self, constraints: Dict[str, Any], current_knowledge: List[Any]
    ) -> Dict[str, Any]:
        """
        Apply constraints to narrow solution space.

        Args:
            constraints: Dictionary of constraints.
            current_knowledge: Current knowledge facts.

        Returns:
            Dictionary with narrowed solution space and detected contradictions.
        """
        self.logger.info('Propagating constraints')

        system_prompt = """You are an expert at constraint propagation and logical reasoning.
Given constraints and current knowledge, determine:
- How constraints narrow the solution space
- Whether contradictions exist
- What new constraints can be inferred
- What possibilities are eliminated

Return a JSON object with:
- narrowed_space: description of narrowed solution space
- contradictions: list of detected contradictions
- inferred_constraints: list of newly inferred constraints
- eliminated_possibilities: list of eliminated options

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Constraints:
{json.dumps(constraints, indent=2, default=str)}

Current Knowledge:
{json.dumps(current_knowledge, indent=2, default=str)}

Propagate constraints and check for contradictions."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Consistent space narrowing
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result = json.loads(json_text)
            # Log contradictions if found
            if result.get('contradictions'):
                self.logger.warning(
                    f'Detected contradictions: {result["contradictions"]}'
                )
            return result
        except Exception as e:
            self.logger.error(f'Constraint propagation failed: {e}', exc_info=True)
            return {
                'narrowed_space': '',
                'contradictions': [],
                'inferred_constraints': [],
                'eliminated_possibilities': [],
            }

    def generate_hypotheses(
        self, problem: str, evidence: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate solutions/hypotheses.

        Args:
            problem: The problem description.
            evidence: List of evidence items.

        Returns:
            List of hypothesis dictionaries with likelihood scores.
        """
        self.logger.info('Generating hypotheses')

        system_prompt = """You are an expert at hypothesis generation.
Given a problem and evidence, generate candidate solutions ranked by likelihood.

Return a JSON object with:
- hypotheses: list of objects, each with:
  - solution: description of the candidate solution
  - likelihood: float between 0 and 1
  - reasoning: explanation of why this is plausible
  - evidence_support: which evidence supports this hypothesis

IMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text."""

        user_prompt = f"""Problem: {problem}

Evidence:
{json.dumps(evidence, indent=2, default=str)}

Generate ranked candidate solutions."""

        try:
            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Consistent hypothesis generation
                response_format={'type': 'json_object'},
            )
            json_text = extract_json_from_text(response)
            result = json.loads(json_text)
            hypotheses = result.get('hypotheses', [])
            # Sort by likelihood
            hypotheses.sort(key=lambda x: x.get('likelihood', 0), reverse=True)
            self.logger.info(f'Generated {len(hypotheses)} hypotheses')
            return hypotheses
        except Exception as e:
            self.logger.error(f'Hypothesis generation failed: {e}', exc_info=True)
            return []
