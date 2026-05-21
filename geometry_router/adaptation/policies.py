from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import AdaptationCandidate, ResidualSignal


class RetrievalTriggerPolicy(Protocol):
    def should_trigger(self, signal: ResidualSignal) -> bool:
        ...


class UpdateEligibilityPolicy(Protocol):
    def eligible(self, candidate: AdaptationCandidate) -> bool:
        ...


@dataclass
class ThresholdRetrievalTriggerPolicy:
    residual_threshold: float = 0.35
    require_reason: bool = True

    def should_trigger(self, signal: ResidualSignal) -> bool:
        if self.require_reason and not signal.reasons:
            return False
        return signal.residual_score >= self.residual_threshold


@dataclass
class FrequencyQualityEligibilityPolicy:
    min_occurrences: int = 3
    min_avg_quality: float = 0.7

    def eligible(self, candidate: AdaptationCandidate) -> bool:
        return (
            candidate.occurrences >= self.min_occurrences
            and candidate.avg_quality >= self.min_avg_quality
        )
