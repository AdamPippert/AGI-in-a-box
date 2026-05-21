from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .adaptation_exports import (
    AdaptationCandidateCollector,
    AdapterTrainingExportService,
    mismatch_key_from_record,
)
from .memory_layer import CorrectionInjector, MemoryMatcher, MemorySignatureStore
from .models import AdaptivePipelineResult, MemoryRecord, RetrievalContext, RetrievalSnippet
from .policies import RetrievalTriggerPolicy, UpdateEligibilityPolicy
from .retrieval_layer import RetrievalAugmentor
from .signals import ResidualSignalProvider

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLayerConfig:
    layer1_enabled: bool = False
    layer2_enabled: bool = False
    layer3_enabled: bool = False


@dataclass
class AdaptiveLayerManager:
    config: AdaptiveLayerConfig
    signal_provider: ResidualSignalProvider
    retrieval_policy: RetrievalTriggerPolicy
    retrieval_augmentor: RetrievalAugmentor
    memory_store: MemorySignatureStore
    memory_matcher: MemoryMatcher
    correction_injector: CorrectionInjector
    adaptation_collector: AdaptationCandidateCollector
    adaptation_policy: UpdateEligibilityPolicy
    export_service: AdapterTrainingExportService

    def process(
        self,
        *,
        query: str,
        workflow_type: str,
        route_confidence: float,
        routing_scores: Dict[str, float],
        overlap_resolution_used: bool,
        stability_score: Optional[float],
        route_metadata: Optional[Dict[str, str]] = None,
    ) -> AdaptivePipelineResult:
        result = AdaptivePipelineResult(observability={"events": []})

        signal = self.signal_provider.build_signal(
            confidence=route_confidence,
            routing_scores=routing_scores,
            overlap_resolution_used=overlap_resolution_used,
            stability_score=stability_score,
        )
        result.observability["signal"] = {
            "residual_score": signal.residual_score,
            "reasons": signal.reasons,
            "route_confidence": signal.route_confidence,
            "top2_margin": signal.top2_margin,
        }
        logger.info("adaptive.signal_detected residual_score=%.4f reasons=%s", signal.residual_score, signal.reasons)

        snippets: List[RetrievalSnippet] = []
        if self.config.layer1_enabled and self.retrieval_policy.should_trigger(signal):
            snippets = self.retrieval_augmentor.retrieve(
                RetrievalContext(query=query, workflow_type=workflow_type, route_metadata=route_metadata or {})
            )
            result.retrieval_triggered = True
            result.retrieval_snippets = snippets
            result.observability["events"].append("retrieval_triggered")
            logger.info("adaptive.retrieval_triggered snippets=%d", len(snippets))
        else:
            logger.info("adaptive.retrieval_not_triggered")

        signature = {
            "residual_score": signal.residual_score,
            "route_confidence": signal.route_confidence,
            "top2_margin": signal.top2_margin,
        }

        if self.config.layer2_enabled:
            hit = self.memory_matcher.best_match(signature, self.memory_store.list_records())
            if hit is not None:
                result.memory_hit = True
                result.injected_correction = self.correction_injector.inject(hit)
                result.observability["events"].append("memory_hit")
                logger.info("adaptive.memory_hit provenance=%s", hit.provenance)
            else:
                logger.info("adaptive.memory_miss")

        correction_payload = result.injected_correction or {}
        correction_success = bool(correction_payload)
        quality = 0.9 if correction_success else 0.5

        record = MemoryRecord(
            timestamp=signal.timestamp,
            workflow_type=workflow_type,
            query_summary=query[:240],
            mismatch_signature=signature,
            routing_context=route_metadata or {},
            retrieved_context=[{"source": s.source, "content": s.content, "score": s.score} for s in snippets],
            correction=correction_payload,
            quality_score=quality,
            provenance="adaptive-layer",
        )

        if self.config.layer2_enabled and correction_success:
            self.memory_store.add(record)
            logger.info("adaptive.memory_store_add quality=%.2f", quality)

        if self.config.layer3_enabled and correction_success:
            key = mismatch_key_from_record(record)
            candidate = self.adaptation_collector.add_event(
                mismatch_key=key,
                quality_score=quality,
                instruction=query,
                preferred_response=str(correction_payload),
                metadata={"workflow_type": workflow_type},
            )
            if self.adaptation_policy.eligible(candidate):
                artifact = self.export_service.export(candidate)
                result.adaptation_candidate_created = True
                result.adaptation_export_path = str(artifact)
                result.observability["events"].append("adaptation_exported")
                logger.info("adaptive.adaptation_exported path=%s", artifact)
            else:
                result.observability["events"].append("adaptation_suppressed")
                logger.info("adaptive.adaptation_suppressed occurrences=%d avg_quality=%.3f", candidate.occurrences, candidate.avg_quality)

        return result
