from pathlib import Path

from geometry_router.adaptation.memory_layer import (
    DictCorrectionInjector,
    DotProductMemoryMatcher,
    FileMemorySignatureStore,
)
from geometry_router.adaptation.models import MemoryRecord


def test_memory_store_and_reuse(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = FileMemorySignatureStore(path=path)
    matcher = DotProductMemoryMatcher(similarity_threshold=0.7)
    injector = DictCorrectionInjector()

    record = MemoryRecord(
        timestamp="2026-01-01T00:00:00+00:00",
        workflow_type="routing",
        query_summary="bad route corrected",
        mismatch_signature={"residual_score": 0.9, "route_confidence": 0.1, "top2_margin": 0.01},
        routing_context={"primary_model": "x"},
        retrieved_context=[],
        correction={"new_route": "specialist-y"},
        quality_score=0.95,
        provenance="unit-test",
    )
    store.add(record)

    loaded = store.list_records()
    match = matcher.best_match(
        {"residual_score": 0.89, "route_confidence": 0.12, "top2_margin": 0.02},
        loaded,
    )

    assert match is not None
    injected = injector.inject(match)
    assert "specialist-y" in injected["correction"]
