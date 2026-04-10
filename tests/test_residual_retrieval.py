from pathlib import Path

from geometry_router.adaptation import (
    AdaptiveLayerConfig,
    AdaptiveLayerManager,
    DefaultResidualSignalProvider,
    DictCorrectionInjector,
    DotProductMemoryMatcher,
    FileBackedKeywordRetrievalProvider,
    FileMemorySignatureStore,
    FrequencyQualityEligibilityPolicy,
    InMemoryAdaptationCollector,
    JsonlAdapterTrainingExportService,
    SimpleRetrievalAugmentor,
    ThresholdRetrievalTriggerPolicy,
)


def make_manager(tmp_path: Path, *, layer1: bool, residual_threshold: float = 0.2) -> AdaptiveLayerManager:
    corpus = tmp_path / "corpus.json"
    corpus.write_text('[{"source":"runbook","content":"routing confidence low fallback policy"}]', encoding="utf-8")

    return AdaptiveLayerManager(
        config=AdaptiveLayerConfig(layer1_enabled=layer1, layer2_enabled=False, layer3_enabled=False),
        signal_provider=DefaultResidualSignalProvider(low_confidence_threshold=0.8, low_margin_threshold=0.2),
        retrieval_policy=ThresholdRetrievalTriggerPolicy(residual_threshold=residual_threshold),
        retrieval_augmentor=SimpleRetrievalAugmentor(FileBackedKeywordRetrievalProvider(corpus_path=corpus)),
        memory_store=FileMemorySignatureStore(path=tmp_path / "mem.jsonl"),
        memory_matcher=DotProductMemoryMatcher(similarity_threshold=0.8),
        correction_injector=DictCorrectionInjector(),
        adaptation_collector=InMemoryAdaptationCollector(state={}),
        adaptation_policy=FrequencyQualityEligibilityPolicy(min_occurrences=2, min_avg_quality=0.7),
        export_service=JsonlAdapterTrainingExportService(output_dir=tmp_path / "exports"),
    )


def test_retrieval_disabled_keeps_behavior(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, layer1=False)

    result = manager.process(
        query="low confidence routing",
        workflow_type="routing",
        route_confidence=0.1,
        routing_scores={"a": 0.4, "b": 0.39},
        overlap_resolution_used=True,
        stability_score=0.1,
    )

    assert result.retrieval_triggered is False
    assert result.retrieval_snippets == []


def test_retrieval_triggers_on_residual_threshold(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, layer1=True, residual_threshold=0.1)

    result = manager.process(
        query="routing confidence low",
        workflow_type="routing",
        route_confidence=0.1,
        routing_scores={"a": 0.4, "b": 0.39},
        overlap_resolution_used=True,
        stability_score=0.2,
    )

    assert result.retrieval_triggered is True
    assert len(result.retrieval_snippets) >= 1
