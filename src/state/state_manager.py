"""Information State Manager for tracking knowledge and progress."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeFact:
    """Represents a single fact in the knowledge graph."""

    entity: str
    relationship: str
    value: Any
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subtask:
    """Represents a subtask in the execution plan."""

    id: str
    description: str
    status: str = 'pending'  # pending, in_progress, completed, failed
    result: Any = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InformationStateManager:
    """Manages knowledge graph, progress tracking, and context."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize Information State Manager.

        Args:
            logger: Logger instance.
        """
        self.logger = logger
        self.knowledge_graph: List[KnowledgeFact] = []
        self.subtasks: Dict[str, Subtask] = {}
        self.completed_subtasks: List[str] = []
        self.pending_requirements: List[str] = []
        self.dead_ends: List[str] = []
        self.active_constraints: Dict[str, Any] = {}
        self.partial_solutions: List[Dict[str, Any]] = []
        self.assumptions: List[str] = []

    def add_fact(
        self,
        entity: str,
        relationship: str,
        value: Any,
        confidence: float = 1.0,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a fact to the knowledge graph.

        Args:
            entity: Entity name.
            relationship: Relationship type.
            value: Relationship value.
            confidence: Confidence score (0-1).
            source: Source of the fact.
            metadata: Additional metadata.
        """
        fact = KnowledgeFact(
            entity=entity,
            relationship=relationship,
            value=value,
            confidence=confidence,
            sources=[source] if source else [],
            metadata=metadata or {},
        )
        self.knowledge_graph.append(fact)
        self.logger.debug(
            f'Added fact: {entity} -[{relationship}]-> {value} '
            f'(confidence: {confidence:.2f})'
        )

    def get_facts(
        self,
        entity: Optional[str] = None,
        relationship: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeFact]:
        """
        Query facts from knowledge graph.

        Args:
            entity: Filter by entity name.
            relationship: Filter by relationship type.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching facts.
        """
        facts = self.knowledge_graph
        if entity:
            facts = [f for f in facts if f.entity == entity]
        if relationship:
            facts = [f for f in facts if f.relationship == relationship]
        facts = [f for f in facts if f.confidence >= min_confidence]
        return facts

    def add_subtask(self, subtask: Subtask):
        """Add a subtask to track."""
        self.subtasks[subtask.id] = subtask
        self.logger.debug(f'Added subtask: {subtask.id} - {subtask.description}')

    def complete_subtask(self, subtask_id: str, result: Any = None):
        """Mark a subtask as completed."""
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = 'completed'
            self.subtasks[subtask_id].result = result
            self.completed_subtasks.append(subtask_id)
            self.logger.info(f'Completed subtask: {subtask_id}')

    def fail_subtask(self, subtask_id: str, reason: str = ''):
        """Mark a subtask as failed."""
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = 'failed'
            self.dead_ends.append(f'{subtask_id}: {reason}')
            self.logger.warning(f'Failed subtask: {subtask_id}: {reason}')

    def add_constraint(self, name: str, constraint: Any):
        """Add an active constraint."""
        self.active_constraints[name] = constraint
        self.logger.debug(f'Added constraint: {name}')

    def add_assumption(self, assumption: str):
        """Track an assumption made during reasoning."""
        self.assumptions.append(assumption)
        self.logger.debug(f'Added assumption: {assumption}')

    def get_failed_subtasks(self) -> List[Subtask]:
        """Get list of failed subtasks."""
        return [s for s in self.subtasks.values() if s.status == 'failed']

    def retry_subtask(self, subtask_id: str):
        """Reset a failed subtask to pending status for retry."""
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = 'pending'
            self.subtasks[subtask_id].result = None
            # Remove from completed list if it was there
            if subtask_id in self.completed_subtasks:
                self.completed_subtasks.remove(subtask_id)
            self.logger.info(f'Reset subtask for retry: {subtask_id}')

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            'knowledge_facts': len(self.knowledge_graph),
            'subtasks_total': len(self.subtasks),
            'subtasks_completed': len(self.completed_subtasks),
            'subtasks_pending': len(
                [
                    s
                    for s in self.subtasks.values()
                    if s.status == 'pending' or s.status == 'in_progress'
                ]
            ),
            'subtasks_failed': len(
                [s for s in self.subtasks.values() if s.status == 'failed']
            ),
            'dead_ends': len(self.dead_ends),
            'active_constraints': len(self.active_constraints),
            'assumptions': len(self.assumptions),
        }
