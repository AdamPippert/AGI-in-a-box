# Residual Adaptation Integration Plan

## Repository inspection summary

### Inference / workflow entry points
- `geometry_router/server.py` exposes HTTP `/route` and is the clearest runtime inference entry point.
- `geometry_router/examples/demo_routing.py` demonstrates local invocation of `GeometryAwareRouter.route(...)`.

### Agent execution wrappers
- The repository currently centers around routing workflows and deployment wrappers; no separate agent runtime package is present in tree.
- `docker-compose.yml` and `k8s/` define service wrappers for `router` and `crewai` containers.

### Retrieval / RAG hooks
- README documents dynamic data-source ingestion / RAG conceptually, but there is no concrete in-repo retrieval module yet.
- Practical extension point: add a pluggable retrieval callback at router server level, triggered by residual mismatch policy.

### Prompt library access patterns
- README references prompt library reuse, but no prompt-store module appears in current code tree.
- Integration approach: retrieval augmentor should remain interface-driven and accept external adapters.

### Persistence / history storage
- Existing stack includes Redis/Postgres in `docker-compose.yml`, but no Python persistence layer is wired into router logic.
- Minimal repo-native option: JSONL file-backed store (pluggable interface to swap later).

### Config loading approach
- `geometry_router/server.py` uses environment variables extensively.
- Compose already propagates env vars to router service; easiest safe path is extending env config in server and `.env.example`.

### Logging / observability hooks
- Logging already configured via `logging.basicConfig(...)` in server and module-level loggers in routing.
- Add structured log events/counters for each layer activation path.

### Test layout and conventions
- No existing `tests/` directory was found.
- Add pytest-based tests under `tests/` with small deterministic fixtures.

### geometry_router usage
- `GeometryAwareRouter.route(...)` returns confidence, overlap usage, hierarchy and topological distances — good residual proxy signals.
- Existing `update_model_profile(...)` performs online profile updates; new controlled adaptation layer must explicitly avoid base-weight mutation and remain export-only.

## Architecture mapping plan

### Layer 1: Residual-Aware Retrieval
- Add interfaces:
  - `ResidualSignalProvider`
  - `RetrievalTriggerPolicy`
  - `RetrievalAugmentor`
- Default signal provider will derive residual score from:
  - low routing confidence
  - overlap resolution usage
  - low margin between top-2 routing scores
  - optional topology instability (`stability_score`)
- If policy fires, augment context via a pluggable retrieval provider function and attach retrieved snippets to response metadata.

### Layer 2: Residual Memory Cache
- Add interfaces:
  - `MemorySignatureStore`
  - `MemoryMatcher`
  - `CorrectionInjector`
- Implement JSONL-backed `FileMemorySignatureStore`.
- Record mismatch events + successful correction payload and reuse when future signatures exceed similarity threshold.

### Layer 3: Controlled Weight Update Path
- Add interfaces:
  - `AdaptationCandidateCollector`
  - `UpdateEligibilityPolicy`
  - `AdapterTrainingExportService`
- Aggregate recurring mismatch signatures and emit reviewable JSONL artifacts only after thresholds are met.
- Explicitly no live base-model weight mutation.

## Planned files to modify

1. `geometry_router/server.py`
   - Wire adaptive manager into `/route` flow and expose observability metadata.
2. `docker-compose.yml`
   - Add environment toggles for three layers and thresholds.
3. `README.md`
   - Document feature flags and safety model at high level.
4. `pyproject.toml`
   - Add `pytest` test dependency.

## Planned files to add

1. `geometry_router/adaptation/models.py`
   - Typed dataclasses for residual signals, memory records, adaptation candidates.
2. `geometry_router/adaptation/signals.py`
   - Residual signal provider implementation using routing outputs.
3. `geometry_router/adaptation/policies.py`
   - Trigger policy and eligibility policy implementations.
4. `geometry_router/adaptation/retrieval_layer.py`
   - Retrieval augmentor with pluggable provider.
5. `geometry_router/adaptation/memory_layer.py`
   - Signature store, matcher, injector; JSONL persistence backend.
6. `geometry_router/adaptation/adaptation_exports.py`
   - Candidate collector and JSONL export service.
7. `geometry_router/adaptation/manager.py`
   - Orchestration facade for all three layers.
8. `geometry_router/adaptation/__init__.py`
   - Public exports.
9. `tests/test_residual_retrieval.py`
10. `tests/test_residual_memory.py`
11. `tests/test_controlled_adaptation.py`
12. `docs/residual_adaptation_architecture.md`
13. `.env.example`
   - New env defaults for toggles and thresholds.

## Implementation sequencing
1. Add plan doc (this file).
2. Create adaptation package with interfaces + default implementations.
3. Integrate manager into server route path behind flags (default off).
4. Add tests for each layer and disabled behavior.
5. Add architecture documentation and README updates.
6. Run tests and finalize.
