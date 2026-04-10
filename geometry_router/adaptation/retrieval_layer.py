from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Protocol

from .models import RetrievalContext, RetrievalSnippet


class RetrievalAugmentor(Protocol):
    def retrieve(self, context: RetrievalContext) -> List[RetrievalSnippet]:
        ...


@dataclass
class SimpleRetrievalAugmentor:
    provider: Callable[[RetrievalContext], List[RetrievalSnippet]]

    def retrieve(self, context: RetrievalContext) -> List[RetrievalSnippet]:
        return self.provider(context)


@dataclass
class FileBackedKeywordRetrievalProvider:
    corpus_path: Path
    max_results: int = 3

    def __call__(self, context: RetrievalContext) -> List[RetrievalSnippet]:
        if not self.corpus_path.exists():
            return []

        try:
            entries = json.loads(self.corpus_path.read_text())
        except json.JSONDecodeError:
            return []

        query_tokens = set(context.query.lower().split())
        scored: List[RetrievalSnippet] = []
        for item in entries:
            content = str(item.get("content", ""))
            source = str(item.get("source", "unknown"))
            tokens = set(content.lower().split())
            overlap = len(query_tokens.intersection(tokens))
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_tokens))
            scored.append(RetrievalSnippet(source=source, content=content, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: self.max_results]
