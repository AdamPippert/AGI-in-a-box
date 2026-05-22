"""
Memory Document Model

Documents are stored as Markdown with YAML frontmatter,
making them human-readable and machine-parseable.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import yaml


class DocumentType(Enum):
    """Types of memory documents."""

    NOTE = "note"
    CONVERSATION = "conversation"
    FACT = "fact"
    TASK = "task"
    DECISION = "decision"
    ARTIFACT = "artifact"
    SUMMARY = "summary"


@dataclass
class DocumentMetadata:
    """YAML frontmatter metadata for documents."""

    id: str
    doc_type: DocumentType
    created_at: datetime
    updated_at: datetime
    agent_id: str = ""
    session_id: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    importance: float = 0.5
    ttl_days: int | None = None
    parent_id: str | None = None
    related_ids: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.doc_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "source": self.source,
            "importance": self.importance,
            "ttl_days": self.ttl_days,
            "parent_id": self.parent_id,
            "related_ids": self.related_ids,
            **self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        known_keys = {
            "id", "type", "created_at", "updated_at", "agent_id",
            "session_id", "tags", "source", "importance", "ttl_days",
            "parent_id", "related_ids"
        }
        custom = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            doc_type=DocumentType(data.get("type", "note")),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", datetime.now()),
            agent_id=data.get("agent_id", ""),
            session_id=data.get("session_id", ""),
            tags=data.get("tags", []),
            source=data.get("source", ""),
            importance=data.get("importance", 0.5),
            ttl_days=data.get("ttl_days"),
            parent_id=data.get("parent_id"),
            related_ids=data.get("related_ids", []),
            custom=custom,
        )


@dataclass
class MemoryDocument:
    """
    A memory document with YAML frontmatter and Markdown content.

    Format:
    ```
    ---
    id: doc_abc123
    type: note
    created_at: 2026-05-22T10:00:00
    tags: [research, claude]
    ---

    # Document Title

    Document content in Markdown...
    ```
    """

    metadata: DocumentMetadata
    content: str
    embedding: list[float] | None = None

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def doc_type(self) -> DocumentType:
        return self.metadata.doc_type

    @classmethod
    def create(
        cls,
        content: str,
        doc_type: DocumentType = DocumentType.NOTE,
        agent_id: str = "",
        session_id: str = "",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> MemoryDocument:
        """Factory method to create a new document."""
        now = datetime.now()
        metadata = DocumentMetadata(
            id=kwargs.get("id", f"doc_{uuid.uuid4().hex[:12]}"),
            doc_type=doc_type,
            created_at=now,
            updated_at=now,
            agent_id=agent_id,
            session_id=session_id,
            tags=tags or [],
            **{k: v for k, v in kwargs.items() if k != "id"},
        )
        return cls(metadata=metadata, content=content)

    def to_markdown(self) -> str:
        """Serialize to Markdown with YAML frontmatter."""
        frontmatter = yaml.dump(
            self.metadata.to_dict(),
            default_flow_style=False,
            sort_keys=False,
        )
        return f"---\n{frontmatter}---\n\n{self.content}"

    @classmethod
    def from_markdown(cls, text: str) -> MemoryDocument:
        """Parse from Markdown with YAML frontmatter."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, text, re.DOTALL)

        if not match:
            return cls.create(content=text)

        frontmatter_str, content = match.groups()
        frontmatter = yaml.safe_load(frontmatter_str)

        metadata = DocumentMetadata.from_dict(frontmatter)
        return cls(metadata=metadata, content=content.strip())

    def update_content(self, new_content: str) -> None:
        """Update document content and timestamp."""
        self.content = new_content
        self.metadata.updated_at = datetime.now()
        self.embedding = None  # Clear embedding for re-computation

    def add_tags(self, *tags: str) -> None:
        """Add tags to the document."""
        for tag in tags:
            if tag not in self.metadata.tags:
                self.metadata.tags.append(tag)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "content": self.content,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryDocument:
        """Deserialize from dictionary."""
        return cls(
            metadata=DocumentMetadata.from_dict(data["metadata"]),
            content=data["content"],
            embedding=data.get("embedding"),
        )

    def get_text_for_embedding(self) -> str:
        """Get text representation for embedding generation."""
        parts = [self.content]

        if self.metadata.tags:
            parts.append(f"Tags: {', '.join(self.metadata.tags)}")

        if self.metadata.source:
            parts.append(f"Source: {self.metadata.source}")

        return "\n".join(parts)
