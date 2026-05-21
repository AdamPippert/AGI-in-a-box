from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .models import MemoryRecord


class MemorySignatureStore(Protocol):
    def add(self, record: MemoryRecord) -> None:
        ...

    def list_records(self) -> List[MemoryRecord]:
        ...


class MemoryMatcher(Protocol):
    def best_match(self, signature: Dict[str, float], records: List[MemoryRecord]) -> Optional[MemoryRecord]:
        ...


class CorrectionInjector(Protocol):
    def inject(self, record: MemoryRecord) -> Dict[str, str]:
        ...


@dataclass
class FileMemorySignatureStore:
    path: Path

    def add(self, record: MemoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")

    def list_records(self) -> List[MemoryRecord]:
        if not self.path.exists():
            return []

        records: List[MemoryRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(MemoryRecord(**payload))
        return records


@dataclass
class DotProductMemoryMatcher:
    similarity_threshold: float = 0.75

    def best_match(
        self,
        signature: Dict[str, float],
        records: List[MemoryRecord],
    ) -> Optional[MemoryRecord]:
        best: Optional[MemoryRecord] = None
        best_score = -1.0

        for record in records:
            score = self._similarity(signature, record.mismatch_signature)
            if score > best_score:
                best = record
                best_score = score

        if best is None or best_score < self.similarity_threshold:
            return None
        return best

    def _similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a).intersection(b)
        if not keys:
            return 0.0
        num = sum(a[k] * b[k] for k in keys)
        den_a = sum(a[k] ** 2 for k in keys) ** 0.5
        den_b = sum(b[k] ** 2 for k in keys) ** 0.5
        if den_a == 0 or den_b == 0:
            return 0.0
        return num / (den_a * den_b)


class DictCorrectionInjector:
    def inject(self, record: MemoryRecord) -> Dict[str, str]:
        return {
            "correction": json.dumps(record.correction),
            "provenance": record.provenance,
            "quality_score": str(record.quality_score),
        }
