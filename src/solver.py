"""GAIASolver - Wrapper around Agent for GAIA benchmark compatibility."""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

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
            llm_model: OpenAI model to use (default: from LLM_MODEL env var, or 'gpt-5').
        """
        self.logger = logging.getLogger('src.solver')
        self.logger.info('Initializing GAIASolver')

        # Create ToolBelt
        tool_belt = ToolBelt()

        # Load model from environment variable if not provided
        model = llm_model or os.getenv('LLM_MODEL', 'gpt-5')
        # Create Agent
        try:
            self.agent = Agent(tool_belt=tool_belt, logger=self.logger, llm_model=model)
        except ValueError as e:
            self.logger.error(f'Failed to initialize Agent: {e}')
            raise

        # Create tool registry
        self.tool_registry = ToolRegistry(tool_belt)

        tool_count = len(self.tool_registry.list_tool_names())
        self.logger.info(f'GAIASolver initialized with {tool_count} tools')

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
        self.logger.info(f'Starting to solve question: {question[:100]}...')

        try:
            # Call agent.solve()
            final_answer, monologue = self.agent.solve(question, attachments)

            # Extract confidence and sources from state manager
            state_summary = self.agent.state_manager.get_state_summary()

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

            # Fallback: extract URLs from monologue if no sources found
            if not sources:
                import re

                url_pattern = r'https?://[^\s\)]+'
                urls = re.findall(url_pattern, monologue)
                sources = list(set(urls))  # Remove duplicates

            self.logger.info(f'Answer: {final_answer if final_answer else "None"}...')
            self.logger.info(f'Confidence: {confidence:.2f}')
            self.logger.info(f'Sources: {len(sources)}')

            return SolverAnswer(
                answer=final_answer or '', confidence=confidence, sources=sources
            )

        except Exception as e:
            self.logger.error(f'Error solving question: {e}', exc_info=True)
            return SolverAnswer(answer=f'Error: {str(e)}', confidence=0.0, sources=[])
