"""GAIASolver - Wrapper around Agent for GAIA benchmark compatibility."""

import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from langfuse import observe, propagate_attributes

from .core import Agent
from .models import Attachment
from .tools import ToolBelt

# Load environment variables
load_dotenv()


@dataclass
class SolverAnswer:
    """Answer object returned by GAIASolver.solve()"""

    answer: str
    confidence: float
    sources: List[str]


class ToolRegistry:
    """Tool registry wrapper for compatibility."""

    def __init__(self, tool_belt: ToolBelt):
        """Initialize with a ToolBelt instance."""
        self.tool_belt = tool_belt

    def list_tool_names(self) -> List[str]:
        """List available tool names."""
        # Extract method names from ToolBelt that are tools
        tool_names = []
        skip_methods = ['set_logger', 'set_llm_service']
        for attr_name in dir(self.tool_belt):
            if not attr_name.startswith('_') and callable(
                getattr(self.tool_belt, attr_name)
            ):
                # Skip non-tool methods
                if attr_name not in skip_methods:
                    tool_names.append(attr_name)
        return sorted(tool_names)


class GAIASolver:
    """GAIA Solver - Wrapper around Agent for GAIA benchmark compatibility."""

    def __init__(self, llm_model: str = None):
        """
        Initialize GAIASolver.

        Args:
            llm_model: OpenAI model to use (default: from LLM_MODEL env var, or 'openai/gpt-oss-120b').
        """
        self.logger = logging.getLogger('src.solver')
        self.logger.info('Initializing GAIASolver')

        # Create ToolBelt
        tool_belt = ToolBelt()

        # Load model from environment variable if not provided
        model = llm_model or os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')
        # Create Agent
        try:
            self.agent = Agent(tool_belt=tool_belt, logger=self.logger, llm_model=model)
        except ValueError as e:
            self.logger.error(f'Failed to initialize Agent: {e}')
            raise

        # Create tool registry
        self.tool_registry = ToolRegistry(tool_belt)

        # Session-based cache: maps question to session_id
        # Same question in same session will reuse the same session_id
        self._question_to_session_id: dict[str, str] = {}

        tool_count = len(self.tool_registry.list_tool_names())
        self.logger.info(f'GAIASolver initialized with {tool_count} tools')

    @observe()
    def solve(
        self, question: str, attachments: Optional[List[Attachment]] = None
    ) -> SolverAnswer:
        """
        Solve a question and return an answer object.

        Args:
            question: The question to solve.
            attachments: Optional list of attachments.

        Returns:
            SolverAnswer object with answer, confidence, and sources.
        """
        # Check if we've seen this question before in this session
        # Same question in same session will reuse the same session_id
        question_key = question.strip()
        is_reusing = question_key in self._question_to_session_id

        if is_reusing:
            # Reuse existing session_id
            session_id = self._question_to_session_id[question_key]
            self.logger.info(
                f'Reusing session_id for same question: {session_id} '
                f'[question: {question[:100]}...]'
            )
        else:
            # Create new session_id for this question
            session_id = str(uuid.uuid4())
            self._question_to_session_id[question_key] = session_id
            self.logger.info(
                f'Created new session_id for question: {session_id} '
                f'[question: {question[:100]}...]'
            )

        try:
            # Propagate session_id to all child observations
            # All nested @observe() decorated functions will automatically inherit session_id
            with propagate_attributes(session_id=session_id):
                # Call agent.solve() - all nested observations will inherit session_id
                # The session_id will be automatically propagated to all @observe() decorated methods
                final_answer = self.agent.solve(question, attachments)

            # Get confidence - default based on answer quality
            confidence = (
                0.9
                if final_answer and final_answer != 'Error: Unable to solve problem'
                else 0.0
            )

            # Extract sources from knowledge graph facts
            sources = []
            knowledge_graph = self.agent.state_manager.knowledge_graph

            # Collect sources from knowledge facts
            for fact in knowledge_graph:
                # Add sources from fact.sources list
                if fact.sources:
                    sources.extend(fact.sources)
                # Also check if value is a URL
                if fact.value and isinstance(fact.value, str):
                    if fact.value.startswith('http://') or fact.value.startswith(
                        'https://'
                    ):
                        sources.append(fact.value)

            # Remove duplicates while preserving order
            seen = set()
            unique_sources = []
            for source in sources:
                if source not in seen:
                    seen.add(source)
                    unique_sources.append(source)
            sources = unique_sources

            self.logger.info(
                f'Answer: {final_answer if final_answer else "None"}... [session_id: {session_id}]'
            )
            self.logger.info(f'Confidence: {confidence:.2f} [session_id: {session_id}]')
            self.logger.info(f'Sources: {len(sources)} [session_id: {session_id}]')

            return SolverAnswer(
                answer=final_answer or '', confidence=confidence, sources=sources
            )

        except Exception as e:
            self.logger.error(
                f'Error solving question: {e} [session_id: {session_id}]', exc_info=True
            )

            return SolverAnswer(answer=f'Error: {str(e)}', confidence=0.0, sources=[])
