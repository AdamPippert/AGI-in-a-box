from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .models import AdaptationCandidate, MemoryRecord, utc_now_iso


class AdaptationCandidateCollector(Protocol):
    def add_event(self, mismatch_key: str, quality_score: float, instruction: str, preferred_response: str, metadata: Optional[Dict[str, str]] = None) -> AdaptationCandidate:
        ...


class AdapterTrainingExportService(Protocol):
    def export(self, candidate: AdaptationCandidate) -> Path:
        ...


@dataclass
class InMemoryAdaptationCollector:
    state: Dict[str, Dict[str, object]]

    def add_event(
        self,
        mismatch_key: str,
        quality_score: float,
        instruction: str,
        preferred_response: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> AdaptationCandidate:
        bucket = self.state.setdefault(
            mismatch_key,
            {"occurrences": 0, "quality_total": 0.0, "instruction": instruction, "response": preferred_response, "metadata": metadata or {}},
        )
        bucket["occurrences"] = int(bucket["occurrences"]) + 1
        bucket["quality_total"] = float(bucket["quality_total"]) + quality_score

        occurrences = int(bucket["occurrences"])
        avg_quality = float(bucket["quality_total"]) / occurrences
        return AdaptationCandidate(
            timestamp=utc_now_iso(),
            mismatch_key=mismatch_key,
            occurrences=occurrences,
            avg_quality=avg_quality,
            instruction=str(bucket["instruction"]),
            preferred_response=str(bucket["response"]),
            metadata=dict(bucket["metadata"]),
        )


@dataclass
class JsonlAdapterTrainingExportService:
    output_dir: Path

    def export(self, candidate: AdaptationCandidate) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"adaptation_{candidate.mismatch_key}_{candidate.timestamp.replace(':', '-')}.jsonl"
        path = self.output_dir / filename
        payload = {
            "format": "adapter_training_example",
            "candidate": asdict(candidate),
            "safety": {
                "live_weight_mutation": False,
                "offline_review_required": True,
            },
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path


def mismatch_key_from_record(record: MemoryRecord) -> str:
    workflow = record.workflow_type or "unknown"
    reasons = "-".join(sorted(str(k) for k in record.mismatch_signature.keys()))
    return f"{workflow}-{reasons}".replace(" ", "_")
