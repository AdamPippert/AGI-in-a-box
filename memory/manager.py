"""
Memory Manager

High-level interface for memory operations, coordinating storage,
retrieval, and lifecycle management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .document import MemoryDocument, DocumentType
from .store import MemoryStore, MemoryTier, TierConfig
from .retrieval import HybridRetriever, RetrievalResult, EmbeddingFunction


@dataclass
class MemoryConfig:
    """Configuration for the memory manager."""

    base_path: Path | str
    default_agent_id: str = ""
    auto_index: bool = True
    index_on_store: bool = False
    cleanup_interval_hours: float = 24.0
    embedding_model: str = "all-MiniLM-L6-v2"


class MemoryManager:
    """
    Unified interface for memory operations.

    Coordinates document storage, retrieval, and lifecycle management
    across all memory tiers.

    Example:
        ```python
        manager = MemoryManager(MemoryConfig(base_path="./memory"))

        # Store a memory
        doc = manager.remember(
            "Claude prefers concise responses",
            doc_type=DocumentType.FACT,
            tags=["preferences", "claude"],
        )

        # Retrieve relevant memories
        results = manager.recall("What are Claude's preferences?")
        for result in results:
            print(result.document.content)

        # Clean up old memories
        manager.cleanup()
        ```
    """

    def __init__(
        self,
        config: MemoryConfig,
        embedding_fn: EmbeddingFunction | None = None,
        tier_configs: dict[MemoryTier, TierConfig] | None = None,
    ):
        self.config = config
        self.store = MemoryStore(
            base_path=config.base_path,
            tier_configs=tier_configs,
        )
        self.retriever = HybridRetriever(
            store=self.store,
            embedding_fn=embedding_fn,
        )

        self._current_session_id: str = ""
        self._cleanup_task: asyncio.Task | None = None

    @property
    def session_id(self) -> str:
        return self._current_session_id

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new memory session."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._current_session_id = session_id

        if self.config.auto_index:
            self.retriever.build_index()

        return session_id

    def end_session(self, clear_session_memory: bool = True) -> None:
        """End the current session."""
        if clear_session_memory and self._current_session_id:
            self.store.clear_session()

        self._current_session_id = ""

    def remember(
        self,
        content: str,
        doc_type: DocumentType = DocumentType.NOTE,
        tier: MemoryTier = MemoryTier.WORKING,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        **kwargs: Any,
    ) -> MemoryDocument:
        """
        Store a new memory.

        Args:
            content: The memory content
            doc_type: Type of document
            tier: Storage tier
            agent_id: Agent owning this memory (uses default if not specified)
            tags: Searchable tags
            importance: Importance score (0.0 to 1.0)
            **kwargs: Additional metadata

        Returns:
            The stored document
        """
        doc = MemoryDocument.create(
            content=content,
            doc_type=doc_type,
            agent_id=agent_id or self.config.default_agent_id,
            session_id=self._current_session_id,
            tags=tags,
            importance=importance,
            **kwargs,
        )

        if self.retriever.embedding_fn:
            doc.embedding = self.retriever.embedding_fn(doc.get_text_for_embedding())

        self.store.store(doc, tier=tier)

        if self.config.index_on_store:
            self.retriever.build_index()

        return doc

    def recall(
        self,
        query: str,
        top_k: int = 5,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
        doc_type: DocumentType | None = None,
        tags: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant memories.

        Args:
            query: Natural language query
            top_k: Maximum results to return
            tier: Filter by tier
            agent_id: Filter by agent
            doc_type: Filter by document type
            tags: Filter by tags (any match)
            min_score: Minimum relevance score

        Returns:
            List of retrieval results ordered by relevance
        """
        results = self.retriever.search(
            query=query,
            top_k=top_k * 2,  # Get more for filtering
            tier=tier,
            agent_id=agent_id or self.config.default_agent_id if agent_id is None else agent_id,
        )

        filtered = []
        for result in results:
            if result.score < min_score:
                continue
            if doc_type and result.document.doc_type != doc_type:
                continue
            if tags and not any(t in result.document.metadata.tags for t in tags):
                continue

            filtered.append(result)

            if len(filtered) >= top_k:
                break

        return filtered

    def recall_recent(
        self,
        limit: int = 10,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> list[MemoryDocument]:
        """Retrieve most recent memories."""
        return self.store.list_documents(
            tier=tier,
            agent_id=agent_id,
            doc_type=doc_type,
            limit=limit,
        )

    def forget(self, doc_id: str) -> bool:
        """Delete a memory by ID."""
        return self.store.delete(doc_id)

    def update(self, doc_id: str, content: str) -> MemoryDocument | None:
        """Update a memory's content."""
        doc = self.store.get(doc_id)
        if doc is None:
            return None

        doc.update_content(content)

        if self.retriever.embedding_fn:
            doc.embedding = self.retriever.embedding_fn(doc.get_text_for_embedding())

        self.store.update(doc)
        return doc

    def promote(self, doc_id: str, to_tier: MemoryTier) -> bool:
        """Move a memory to a higher tier."""
        return self.store.move(doc_id, to_tier)

    def find_similar(
        self,
        doc_id: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Find memories similar to a given memory."""
        doc = self.store.get(doc_id)
        if doc is None:
            return []

        return self.retriever.search_similar(doc, top_k=top_k)

    def summarize_session(self) -> str:
        """Generate a summary of current session memories."""
        session_docs = self.store.list_documents(
            tier=MemoryTier.SESSION,
            agent_id=self.config.default_agent_id,
        )

        if not session_docs:
            return "No memories in current session."

        lines = [f"Session: {self._current_session_id}", f"Memories: {len(session_docs)}", ""]

        by_type: dict[DocumentType, list[MemoryDocument]] = {}
        for doc in session_docs:
            by_type.setdefault(doc.doc_type, []).append(doc)

        for doc_type, docs in by_type.items():
            lines.append(f"## {doc_type.value.title()}s ({len(docs)})")
            for doc in docs[:5]:
                preview = doc.content[:100].replace("\n", " ")
                lines.append(f"- {preview}...")
            if len(docs) > 5:
                lines.append(f"  ... and {len(docs) - 5} more")
            lines.append("")

        return "\n".join(lines)

    def cleanup(self) -> dict[str, int]:
        """Run cleanup operations."""
        expired = self.store.cleanup_expired()

        return {
            "expired_removed": expired,
        }

    def reindex(self, tier: MemoryTier | None = None) -> int:
        """Rebuild search indices."""
        return self.retriever.build_index(tier=tier)

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        store_stats = self.store.get_stats()

        return {
            "session_id": self._current_session_id,
            "default_agent_id": self.config.default_agent_id,
            **store_stats,
        }

    def export_memories(
        self,
        output_path: Path | str,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Export memories to a directory as Markdown files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        count = 0
        for doc in self.store.iter_documents(tier=tier, agent_id=agent_id):
            doc_path = output_path / f"{doc.id}.md"
            doc_path.write_text(doc.to_markdown())
            count += 1

        return count

    def import_memories(
        self,
        input_path: Path | str,
        tier: MemoryTier = MemoryTier.ARCHIVE,
    ) -> int:
        """Import memories from Markdown files."""
        input_path = Path(input_path)
        count = 0

        for md_file in input_path.glob("*.md"):
            content = md_file.read_text()
            doc = MemoryDocument.from_markdown(content)

            if self.retriever.embedding_fn and doc.embedding is None:
                doc.embedding = self.retriever.embedding_fn(
                    doc.get_text_for_embedding()
                )

            self.store.store(doc, tier=tier)
            count += 1

        return count

    async def start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                self.cleanup()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_background_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
