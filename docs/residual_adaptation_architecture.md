# Residual Adaptation Architecture

This repository now supports an optional three-layer adaptive architecture integrated into the geometry-aware routing server.

## Plain-language behavior

- **Layer 1:** “When the system seems unsure, it looks things up.”
- **Layer 2:** “When the system gets corrected, it remembers the fix for similar situations.”
- **Layer 3:** “Only after the same issue happens enough times does the system prepare a deeper learning update.”

## Important safety properties

- This is **not** naive always-on online learning.
- The runtime does **not** rewrite model weights on each interaction.
- Fast adaptation is done through retrieval + memory first.
- Durable learning is generated as isolated, reviewable, offline-ready artifacts.

## Where each layer lives

- Layer 1 (residual-aware retrieval):
  - `geometry_router/adaptation/signals.py`
  - `geometry_router/adaptation/policies.py`
  - `geometry_router/adaptation/retrieval_layer.py`
- Layer 2 (residual memory cache):
  - `geometry_router/adaptation/memory_layer.py`
- Layer 3 (controlled update export path):
  - `geometry_router/adaptation/adaptation_exports.py`

Orchestration entrypoint:
- `geometry_router/adaptation/manager.py`

Runtime wiring:
- `geometry_router/server.py`

## Technical flow in AGI-in-a-box

1. The router produces confidence/topology/routing-score diagnostics.
2. `DefaultResidualSignalProvider` converts those signals into a residual score.
3. `ThresholdRetrievalTriggerPolicy` decides if retrieval should fire.
4. `SimpleRetrievalAugmentor` invokes a pluggable provider (default file-backed keyword provider).
5. `DotProductMemoryMatcher` checks for matching prior correction signatures.
6. If a correction exists, it is injected and optionally persisted in JSONL memory.
7. Repeated high-quality correction events are collected.
8. `FrequencyQualityEligibilityPolicy` gates export eligibility.
9. Eligible candidates are exported as JSONL adaptation artifacts.

## Configuration

Environment variables (all default-disabled behavior preserving prior behavior):

- `ADAPT_LAYER1_ENABLED`
- `ADAPT_LAYER2_ENABLED`
- `ADAPT_LAYER3_ENABLED`
- `ADAPT_RESIDUAL_THRESHOLD`
- `ADAPT_LOW_CONFIDENCE_THRESHOLD`
- `ADAPT_LOW_MARGIN_THRESHOLD`
- `ADAPT_MEMORY_SIMILARITY_THRESHOLD`
- `ADAPT_MIN_OCCURRENCES`
- `ADAPT_MIN_AVG_QUALITY`
- `ADAPT_MEMORY_PATH`
- `ADAPT_EXPORT_DIR`
- `ADAPT_RETRIEVAL_CORPUS_PATH`

## Observability

The system logs activation paths with `adaptive.*` log events including:

- mismatch signal detection
- retrieval trigger / no-trigger
- memory hit / miss
- memory persistence writes
- adaptation suppress / export

The `/route` response also includes `adaptive.observability` metadata.

## Testing

See tests:
- `tests/test_residual_retrieval.py`
- `tests/test_residual_memory.py`
- `tests/test_controlled_adaptation.py`

These validate:
- disabled-path no-op behavior
- threshold-triggered retrieval
- memory reuse by signature similarity
- adaptation export only after recurrence thresholds
- no direct live-weight mutation semantics in exported artifacts metadata
