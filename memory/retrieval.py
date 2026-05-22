"""
Hybrid Retrieval System

Combines vector similarity search with BM25 lexical search
for robust memory retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .document import MemoryDocument
from .store import MemoryStore, MemoryTier


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    document: MemoryDocument
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25:
    """
    BM25 lexical search implementation.

    Okapi BM25 ranking function for term-based retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._doc_lens: list[int] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def index(self, documents: list[tuple[str, str]]) -> None:
        """Index documents for BM25 search."""
        self._documents = []
        self._doc_ids = []
        self._doc_freqs = Counter()

        for doc_id, content in documents:
            tokens = self.tokenize(content)
            self._documents.append(tokens)
            self._doc_ids.append(doc_id)
            self._doc_freqs.update(set(tokens))

        self._doc_lens = [len(d) for d in self._documents]
        self._avg_doc_len = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0

        n_docs = len(self._documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            self._idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents matching the query."""
        query_tokens = self.tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self._documents):
            score = 0.0
            doc_len = self._doc_lens[i]
            term_freqs = Counter(doc_tokens)

            for term in query_tokens:
                if term not in self._idf:
                    continue

                tf = term_freqs.get(term, 0)
                idf = self._idf[term]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avg_doc_len
                )

                score += idf * numerator / denominator

            scores.append((self._doc_ids[i], score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class VectorIndex:
    """
    Simple in-memory vector index using numpy.

    For production, consider using Qdrant, Pinecone, or pgvector.
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._doc_ids: list[str] = []

    def index(self, documents: list[tuple[str, list[float]]]) -> None:
        """Index document vectors."""
        self._doc_ids = [doc_id for doc_id, _ in documents]
        vectors = [vec for _, vec in documents]
        self._vectors = np.array(vectors, dtype=np.float32)

        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._vectors = self._vectors / norms

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for similar documents."""
        if self._vectors is None or len(self._vectors) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1)

        similarities = np.dot(self._vectors, query)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (self._doc_ids[i], float(similarities[i]))
            for i in top_indices
        ]


EmbeddingFunction = Callable[[str], list[float]]


class HybridRetriever:
    """
    Hybrid retrieval combining vector and lexical search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    both retrieval methods.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_fn: EmbeddingFunction | None = None,
        vector_weight: float = 0.6,
        lexical_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedding_fn = embedding_fn
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k

        self._bm25 = BM25()
        self._vector_index = VectorIndex()
        self._doc_cache: dict[str, MemoryDocument] = {}

    def _get_default_embedding_fn(self) -> EmbeddingFunction:
        """Get default embedding function using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            def embed(text: str) -> list[float]:
                return model.encode(text).tolist()

            return embed
        except ImportError:
            raise ImportError(
                "sentence-transformers required for embeddings: "
                "pip install sentence-transformers"
            )

    def build_index(
        self,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Build search indices from stored documents."""
        documents = list(self.store.iter_documents(tier=tier, agent_id=agent_id))

        if not documents:
            return 0

        bm25_docs = [
            (doc.id, doc.get_text_for_embedding())
            for doc in documents
        ]
        self._bm25.index(bm25_docs)

        if self.embedding_fn is None:
            self.embedding_fn = self._get_default_embedding_fn()

        vector_docs = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_fn(doc.get_text_for_embedding())
                self.store.update(doc)

            vector_docs.append((doc.id, doc.embedding))
            self._doc_cache[doc.id] = doc

        self._vector_index.index(vector_docs)

        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        tier: MemoryTier | None = None,
        agent_id: str | None = None,
        use_vector: bool = True,
        use_lexical: bool = True,
    ) -> list[RetrievalResult]:
        """
        Search for relevant documents.

        Uses Reciprocal Rank Fusion to combine vector and lexical results.
        """
        results: dict[str, RetrievalResult] = {}

        if use_lexical:
            lexical_results = self._bm25.search(query, top_k=top_k * 2)

            for rank, (doc_id, score) in enumerate(lexical_results):
                if doc_id not in results:
                    doc = self._doc_cache.get(doc_id) or self.store.get(doc_id)
                    if doc is None:
                        continue

                    if tier and self.store._index.get(doc_id, (None,))[0] != tier:
                        continue
                    if agent_id is not None and doc.metadata.agent_id != agent_id:
                        continue

                    results[doc_id] = RetrievalResult(
                        document=doc,
                        score=0.0,
                        lexical_score=score,
                    )

                results[doc_id].lexical_score = 1.0 / (self.rrf_k + rank + 1)

        if use_vector and self.embedding_fn:
            query_embedding = self.embedding_fn(query)
            vector_results = self._vector_index.search(query_embedding, top_k=top_k * 2)

            for rank, (doc_id, score) in enumerate(vector_results):
                if doc_id not in results:
                    doc = self._doc_cache.get(doc_id) or self.store.get(doc_id)
                    if doc is None:
                        continue

                    if tier and self.store._index.get(doc_id, (None,))[0] != tier:
                        continue
                    if agent_id is not None and doc.metadata.agent_id != agent_id:
                        continue

                    results[doc_id] = RetrievalResult(
                        document=doc,
                        score=0.0,
                        vector_score=score,
                    )

                results[doc_id].vector_score = 1.0 / (self.rrf_k + rank + 1)

        for result in results.values():
            result.score = (
                self.vector_weight * result.vector_score
                + self.lexical_weight * result.lexical_score
            )

        sorted_results = sorted(results.values(), key=lambda r: -r.score)
        return sorted_results[:top_k]

    def search_similar(
        self,
        document: MemoryDocument,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Find documents similar to a given document."""
        if document.embedding is None and self.embedding_fn:
            document.embedding = self.embedding_fn(document.get_text_for_embedding())

        if document.embedding is None:
            return []

        vector_results = self._vector_index.search(document.embedding, top_k=top_k + 1)

        results = []
        for doc_id, score in vector_results:
            if doc_id == document.id:
                continue

            doc = self._doc_cache.get(doc_id) or self.store.get(doc_id)
            if doc:
                results.append(
                    RetrievalResult(
                        document=doc,
                        score=score,
                        vector_score=score,
                    )
                )

        return results[:top_k]
