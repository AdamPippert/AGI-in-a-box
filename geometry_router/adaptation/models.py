from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResidualSignal:
    timestamp: str
    route_confidence: float
    top2_margin: float
    overlap_resolution_used: bool
    stability_score: Optional[float] = None
    residual_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    query: str
    workflow_type: str
    route_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalSnippet:
    source: str
    content: str
    score: float = 1.0


@dataclass
class MemoryRecord:
    timestamp: str
    workflow_type: str
    query_summary: str
    mismatch_signature: Dict[str, Any]
    routing_context: Dict[str, Any]
    retrieved_context: List[Dict[str, Any]]
    correction: Dict[str, Any]
    quality_score: float
    provenance: str
    expires_at: Optional[str] = None
    review_state: str = "unreviewed"


@dataclass
class AdaptationCandidate:
    timestamp: str
    mismatch_key: str
    occurrences: int
    avg_quality: float
    instruction: str
    preferred_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptivePipelineResult:
    retrieval_triggered: bool = False
    retrieval_snippets: List[RetrievalSnippet] = field(default_factory=list)
    memory_hit: bool = False
    injected_correction: Optional[Dict[str, Any]] = None
    adaptation_candidate_created: bool = False
    adaptation_export_path: Optional[str] = None
    observability: Dict[str, Any] = field(default_factory=dict)
