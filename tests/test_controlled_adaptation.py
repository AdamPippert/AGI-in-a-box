from pathlib import Path

from geometry_router.adaptation import (
    AdaptiveLayerConfig,
    AdaptiveLayerManager,
    DefaultResidualSignalProvider,
    DotProductMemoryMatcher,
    FileMemorySignatureStore,
    FrequencyQualityEligibilityPolicy,
    InMemoryAdaptationCollector,
    JsonlAdapterTrainingExportService,
    ThresholdRetrievalTriggerPolicy,
)
from geometry_router.adaptation.memory_layer import CorrectionInjector
from geometry_router.adaptation.retrieval_layer import RetrievalAugmentor
from geometry_router.adaptation.models import MemoryRecord, RetrievalContext, RetrievalSnippet


class AlwaysCorrectionInjector(CorrectionInjector):
    def inject(self, record: MemoryRecord):
        return {"fix": "apply-known-good-route"}


class EmptyRetrieval(RetrievalAugmentor):
    def retrieve(self, context: RetrievalContext):
        return [RetrievalSnippet(source="synthetic", content="hint", score=1.0)]


def test_adaptation_exports_only_after_recurrence(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    store = FileMemorySignatureStore(path=memory_path)
    store.add(
        MemoryRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            workflow_type="routing",
            query_summary="known mismatch",
            mismatch_signature={"residual_score": 0.9, "route_confidence": 0.1, "top2_margin": 0.01},
            routing_context={},
            retrieved_context=[],
            correction={"fix": "apply-known-good-route"},
            quality_score=1.0,
            provenance="test",
        )
    )

    manager = AdaptiveLayerManager(
        config=AdaptiveLayerConfig(layer1_enabled=True, layer2_enabled=True, layer3_enabled=True),
        signal_provider=DefaultResidualSignalProvider(low_confidence_threshold=0.9, low_margin_threshold=0.2),
        retrieval_policy=ThresholdRetrievalTriggerPolicy(residual_threshold=0.1),
        retrieval_augmentor=EmptyRetrieval(),
        memory_store=store,
        memory_matcher=DotProductMemoryMatcher(similarity_threshold=0.7),
        correction_injector=AlwaysCorrectionInjector(),
        adaptation_collector=InMemoryAdaptationCollector(state={}),
        adaptation_policy=FrequencyQualityEligibilityPolicy(min_occurrences=2, min_avg_quality=0.7),
        export_service=JsonlAdapterTrainingExportService(output_dir=tmp_path / "exports"),
    )

    first = manager.process(
        query="route this safely",
        workflow_type="routing",
        route_confidence=0.1,
        routing_scores={"a": 0.5, "b": 0.49},
        overlap_resolution_used=True,
        stability_score=0.2,
    )
    second = manager.process(
        query="route this safely",
        workflow_type="routing",
        route_confidence=0.1,
        routing_scores={"a": 0.5, "b": 0.49},
        overlap_resolution_used=True,
        stability_score=0.2,
    )

    assert first.adaptation_candidate_created is False
    assert second.adaptation_candidate_created is True
    assert second.adaptation_export_path is not None

    exported = Path(second.adaptation_export_path)
    data = exported.read_text(encoding="utf-8")
    assert "live_weight_mutation" in data
    assert "false" in data.lower()
