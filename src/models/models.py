"""Data models for the agent system."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Attachment:
    """Represents a file attachment provided by the user."""

    filename: str  # e.g., "document.pdf", "image.png"
    data: bytes  # Raw file content
    metadata: dict = field(default_factory=dict)  # Optional metadata


@dataclass
class SearchResult:
    """Represents a single web search result."""

    snippet: str  # Text snippet from the result
    url: str  # The source URL
    title: str  # The page title
    relevance_score: float = 0.0  # Optional relevance score


@dataclass
class RevisionData:
    """Represents metadata for a single Wikipedia revision."""

    timestamp: datetime
    editor: str
    tags: list[str]
    comment: str
    revision_id: str = ''  # Optional revision ID
