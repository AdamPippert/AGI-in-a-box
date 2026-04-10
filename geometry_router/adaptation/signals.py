from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from .models import ResidualSignal, utc_now_iso


class ResidualSignalProvider(Protocol):
    def build_signal(self, *, confidence: float, routing_scores: Dict[str, float], overlap_resolution_used: bool, stability_score: float | None = None) -> ResidualSignal:
        ...


@dataclass
class DefaultResidualSignalProvider:
    low_confidence_threshold: float = 0.45
    low_margin_threshold: float = 0.08
    low_stability_threshold: float = 0.35

    def build_signal(
        self,
        *,
        confidence: float,
        routing_scores: Dict[str, float],
        overlap_resolution_used: bool,
        stability_score: float | None = None,
    ) -> ResidualSignal:
        ranked = sorted(routing_scores.values(), reverse=True)
        top2_margin = ranked[0] - ranked[1] if len(ranked) > 1 else confidence

        reasons: List[str] = []
        residual = 0.0

        if confidence < self.low_confidence_threshold:
            reasons.append("low_confidence")
            residual += min(1.0, (self.low_confidence_threshold - confidence) / self.low_confidence_threshold)

        if top2_margin < self.low_margin_threshold:
            reasons.append("low_margin")
            residual += min(1.0, (self.low_margin_threshold - top2_margin) / max(self.low_margin_threshold, 1e-6))

        if overlap_resolution_used:
            reasons.append("overlap_resolution")
            residual += 0.3

        if stability_score is not None and stability_score < self.low_stability_threshold:
            reasons.append("low_stability")
            residual += min(1.0, (self.low_stability_threshold - stability_score) / self.low_stability_threshold)

        residual_score = max(0.0, min(1.0, residual / 3.0))

        return ResidualSignal(
            timestamp=utc_now_iso(),
            route_confidence=confidence,
            top2_margin=top2_margin,
            overlap_resolution_used=overlap_resolution_used,
            stability_score=stability_score,
            residual_score=residual_score,
            reasons=reasons,
        )
