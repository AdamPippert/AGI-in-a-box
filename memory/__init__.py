"""
AGI-in-a-Box Memory System

Hierarchical memory with hybrid vector + lexical retrieval.

Tiers:
- Session: Ephemeral scratch space (cleared on session end)
- Working: Short-term agent memory (7-day default retention)
- Archive: Long-term durable storage (indefinite retention)

Documents are stored as Markdown with YAML frontmatter for
human-readable, machine-parseable memory.
"""

from .document import MemoryDocument, DocumentMetadata
from .store import MemoryStore, MemoryTier
from .retrieval import HybridRetriever, RetrievalResult
from .manager import MemoryManager

__all__ = [
    "MemoryDocument",
    "DocumentMetadata",
    "MemoryStore",
    "MemoryTier",
    "HybridRetriever",
    "RetrievalResult",
    "MemoryManager",
]
