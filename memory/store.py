"""
Memory Store

Tiered storage for memory documents with automatic retention management.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from .document import MemoryDocument, DocumentType


class MemoryTier(Enum):
    """Memory storage tiers."""

    SESSION = "session"   # Ephemeral, cleared on session end
    WORKING = "working"   # Short-term, 7-day default retention
    ARCHIVE = "archive"   # Long-term, indefinite retention


@dataclass
class TierConfig:
    """Configuration for a memory tier."""

    tier: MemoryTier
    retention_days: int | None
    max_documents: int | None = None
    max_size_mb: float | None = None
    auto_promote: bool = False
    promote_to: MemoryTier | None = None
    compress: bool = False


DEFAULT_TIER_CONFIGS = {
    MemoryTier.SESSION: TierConfig(
        tier=MemoryTier.SESSION,
        retention_days=None,  # Cleared explicitly
        max_documents=1000,
    ),
    MemoryTier.WORKING: TierConfig(
        tier=MemoryTier.WORKING,
        retention_days=7,
        max_documents=10000,
        auto_promote=True,
        promote_to=MemoryTier.ARCHIVE,
    ),
    MemoryTier.ARCHIVE: TierConfig(
        tier=MemoryTier.ARCHIVE,
        retention_days=None,  # Indefinite
        compress=True,
    ),
}


class MemoryStore:
    """
    Tiered memory storage with file-based persistence.

    Documents are stored as Markdown files with YAML frontmatter,
    organized by tier and agent.

    Structure:
        {base_path}/
            session/{agent_id}/
                doc_abc123.md
            working/{agent_id}/
                doc_def456.md
            archive/{agent_id}/
                doc_ghi789.md
    """

    def __init__(
        self,
        base_path: Path | str,
        tier_configs: dict[MemoryTier, TierConfig] | None = None,
    ):
        self.base_path = Path(base_path)
        self.tier_configs = tier_configs or DEFAULT_TIER_CONFIGS

        self._ensure_directories()
        self._index: dict[str, tuple[MemoryTier, str]] = {}  # doc_id -> (tier, agent_id)
        self._load_index()

    def _ensure_directories(self) -> None:
        """Create tier directories if they don't exist."""
        for tier in MemoryTier:
            (self.base_path / tier.value).mkdir(parents=True, exist_ok=True)

    def _get_doc_path(self, tier: MemoryTier, agent_id: str, doc_id: str) -> Path:
        """Get the file path for a document."""
        agent_dir = self.base_path / tier.value / (agent_id or "_global")
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir / f"{doc_id}.md"

    def _load_index(self) -> None:
        """Load document index from disk."""
        self._index.clear()

        for tier in MemoryTier:
            tier_path = self.base_path / tier.value
            if not tier_path.exists():
                continue

            for agent_dir in tier_path.iterdir():
                if not agent_dir.is_dir():
                    continue

                agent_id = agent_dir.name if agent_dir.name != "_global" else ""

                for doc_file in agent_dir.glob("*.md"):
                    doc_id = doc_file.stem
                    self._index[doc_id] = (tier, agent_id)

    def store(
        self,
        document: MemoryDocument,
        tier: MemoryTier = MemoryTier.WORKING,
    ) -> None:
        """Store a document in the specified tier."""
        agent_id = document.metadata.agent_id
        doc_path = self._get_doc_path(tier, agent_id, document.id)

        doc_path.write_text(document.to_markdown())
        self._index[document.id] = (tier, agent_id)

        config = self.tier_configs.get(tier)
        if config and config.max_documents:
            self._enforce_limits(tier, agent_id, config)

    def get(self, doc_id: str) -> MemoryDocument | None:
        """Retrieve a document by ID."""
        if doc_id not in self._index:
            return None

        tier, agent_id = self._index[doc_id]
        doc_path = self._get_doc_path(tier, agent_id, doc_id)

        if not doc_path.exists():
            del self._index[doc_id]
            return None

        return MemoryDocument.from_markdown(doc_path.read_text())

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id not in self._index:
            return False

        tier, agent_id = self._index[doc_id]
        doc_path = self._get_doc_path(tier, agent_id, doc_id)

        if doc_path.exists():
            doc_path.unlink()

        del self._index[doc_id]
        return True

    def update(self, document: MemoryDocument) -> bool:
        """Update an existing document."""
        if document.id not in self._index:
            return False

        tier, agent_id = self._index[document.id]
        doc_path = self._get_doc_path(tier, agent_id, document.id)

        document.metadata.updated_at = datetime.now()
        doc_path.write_text(document.to_markdown())
        return True

    def move(self, doc_id: str, to_tier: MemoryTier) -> bool:
        """Move a document to a different tier."""
        if doc_id not in self._index:
            return False

        from_tier, agent_id = self._index[doc_id]
        if from_tier == to_tier:
            return True

        from_path = self._get_doc_path(from_tier, agent_id, doc_id)
        to_path = self._get_doc_path(to_tier, agent_id, doc_id)

        if not from_path.exists():
            del self._index[doc_id]
            return False

        shutil.move(str(from_path), str(to_path))
        self._index[doc_id] = (to_tier, agent_id)
        return True

    def list_documents(
        self,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
        doc_type: DocumentType | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[MemoryDocument]:
        """List documents with optional filtering."""
        documents = []

        for doc_id, (doc_tier, doc_agent) in self._index.items():
            if tier and doc_tier != tier:
                continue
            if agent_id is not None and doc_agent != agent_id:
                continue

            doc = self.get(doc_id)
            if doc is None:
                continue

            if doc_type and doc.doc_type != doc_type:
                continue
            if tags and not any(t in doc.metadata.tags for t in tags):
                continue
            if since and doc.metadata.created_at < since:
                continue

            documents.append(doc)

        documents.sort(key=lambda d: d.metadata.updated_at, reverse=True)

        if limit:
            documents = documents[:limit]

        return documents

    def iter_documents(
        self,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
    ) -> Iterator[MemoryDocument]:
        """Iterate over documents without loading all into memory."""
        for doc_id, (doc_tier, doc_agent) in list(self._index.items()):
            if tier and doc_tier != tier:
                continue
            if agent_id is not None and doc_agent != agent_id:
                continue

            doc = self.get(doc_id)
            if doc:
                yield doc

    def clear_session(self, agent_id: str | None = None) -> int:
        """Clear all session-tier documents for an agent."""
        count = 0
        to_delete = []

        for doc_id, (tier, doc_agent) in self._index.items():
            if tier != MemoryTier.SESSION:
                continue
            if agent_id is not None and doc_agent != agent_id:
                continue
            to_delete.append(doc_id)

        for doc_id in to_delete:
            if self.delete(doc_id):
                count += 1

        return count

    def cleanup_expired(self) -> int:
        """Remove documents past their retention period."""
        count = 0
        now = datetime.now()
        to_delete = []

        for doc_id, (tier, agent_id) in self._index.items():
            config = self.tier_configs.get(tier)
            if not config or config.retention_days is None:
                continue

            doc = self.get(doc_id)
            if doc is None:
                continue

            ttl = doc.metadata.ttl_days or config.retention_days
            expiry = doc.metadata.created_at + timedelta(days=ttl)

            if now > expiry:
                if config.auto_promote and config.promote_to:
                    self.move(doc_id, config.promote_to)
                else:
                    to_delete.append(doc_id)

        for doc_id in to_delete:
            if self.delete(doc_id):
                count += 1

        return count

    def _enforce_limits(
        self,
        tier: MemoryTier,
        agent_id: str,
        config: TierConfig,
    ) -> None:
        """Enforce document count limits for a tier."""
        if not config.max_documents:
            return

        docs = self.list_documents(tier=tier, agent_id=agent_id)
        if len(docs) <= config.max_documents:
            return

        docs.sort(key=lambda d: d.metadata.importance)
        excess = len(docs) - config.max_documents

        for doc in docs[:excess]:
            if config.auto_promote and config.promote_to:
                self.move(doc.id, config.promote_to)
            else:
                self.delete(doc.id)

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_documents": len(self._index),
            "tiers": {},
        }

        for tier in MemoryTier:
            tier_docs = [
                doc_id for doc_id, (t, _) in self._index.items() if t == tier
            ]
            tier_path = self.base_path / tier.value

            size_bytes = 0
            if tier_path.exists():
                for f in tier_path.rglob("*.md"):
                    size_bytes += f.stat().st_size

            stats["tiers"][tier.value] = {
                "document_count": len(tier_docs),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            }

        return stats
