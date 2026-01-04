# Geometry-Aware Hierarchical Routing Framework

## Overview

This framework implements a topology-based multi-model routing system that uses persistent homology to characterize the "shape" of queries and route them to appropriate specialist models.

## Research Foundations

The framework synthesizes three cutting-edge research threads:

### 1. DeepSeek mHC (arXiv 2512.24880)
**Manifold-Constrained Hyper-Connections** - A mathematical framework that constrains information flow through neural networks using the Birkhoff polytope.

Key mechanism: Projects routing matrices onto doubly stochastic matrices via **Sinkhorn-Knopp algorithm**, achieving:
- Spectral norm bounded by 1 (preventing gradient explosion)
- Closure under composition (stability at arbitrary depth)
- Geometric interpretability

### 2. MIT Recursive Language Models (arXiv 2512.24601)
**Recursive decomposition for reasoning** - Treating prompts as programmable objects where the root LM can peek, grep, partition, and recursively invoke sub-LMs on context snippets.

Key insight: **Self-routing** - the orchestrating model makes all routing decisions, enabling processing of 10M+ token inputs through dynamic, inference-time decomposition.

### 3. Topological Data Analysis
**Persistent homology for problem characterization** - Mathematical tools to describe the intrinsic structure of problems, creating natural routing signals.

- **Betti numbers**: beta_0 = connected components, beta_1 = loops, beta_2 = voids
- **Persistence diagrams**: Track topological features across scales
- **Bottleneck/Wasserstein distances**: Measure similarity between topological signatures

## Core Algorithm

### Routing Pipeline

```
1. EXTRACT topological signature from query embeddings
   - Compute persistence diagram via Ripser (or approximate)
   - Extract Betti numbers (beta_0, beta_1, beta_2)
   - Classify complexity (TRIVIAL -> CHAOTIC)
   - Compute stability score

2. COMPUTE compatibility scores for each model
   - Match query topology to model's optimal profile
   - Apply capability bonuses for matching transformations
   - Scale by model priority

3. BUILD routing matrix and PROJECT via Sinkhorn-Knopp
   - Row 0: Query -> Model affinities
   - Rows 1+: Model -> Model affinities (hierarchy-based)
   - Project to doubly stochastic matrix

4. RESOLVE overlaps using combined distance
   overlap_score = (1 - alpha) * topo_distance + alpha * hier_distance
   Select model with MINIMUM overlap_score

5. RETURN ranked model selection with confidence
```

### Overlap Resolution Formula

When multiple models have similar routing scores (within `overlap_threshold`):

```
For each overlapping model m:
    topo_dist[m] = bottleneck_distance(query.persistence, model[m].centroid)
    hier_dist[m] = tree_distance(source_model, m)

Normalize distances:
    topo_dist = topo_dist / max(topo_dist)
    hier_dist = hier_dist / max(hier_dist)

Combined score (minimize both):
    combined[m] = topology_weight * topo_dist[m] + hierarchy_weight * hier_dist[m]

Select: argmin(combined)
```

Default weights: `topology_weight = 0.7`, `hierarchy_weight = 0.3`

## Topological Capabilities

Models are characterized by their ability to perform topological transformations:

| Capability | Description | Use Case |
|------------|-------------|----------|
| `B0_REDUCTION` | Clustering disparate info | High beta_0 (fragmented) queries |
| `B1_REDUCTION` | Untangling circular deps | High beta_1 (entangled) queries |
| `B2_REDUCTION` | Collapsing void structures | Complex 3D structures |
| `STRUCTURE_PRESERVE` | Maintaining topology | Creative tasks |
| `UNIVERSAL` | General purpose | Default fallback |

## Model Hierarchy

```
ORCHESTRATOR (Tier 0)
    |
    +-- Claude Opus 4 [UNIVERSAL]
            |
            +-- SPECIALIST (Tier 1)
            |       |
            |       +-- Claude Sonnet Code [B1_REDUCTION]
            |       |       |
            |       |       +-- EXECUTOR (Tier 2)
            |       |               +-- Claude Haiku 4 [UNIVERSAL]
            |       |               +-- GPT-4o Mini [UNIVERSAL]
            |       |
            |       +-- DeepSeek R1 [B0_REDUCTION]
            |       +-- Claude Sonnet Creative [STRUCTURE_PRESERVE]
            |
            +-- VERIFIER (Tier 3)
                    +-- Verification Ensemble [B0_REDUCTION]
```

## Recursive Decomposition

The `RecursiveRoutingController` decides when to decompose queries based on topology:

| Condition | Strategy |
|-----------|----------|
| beta_0 > 5 (many components) | `parallel` - decompose into parallel chunks |
| beta_1 > 3 (many loops) | `sequential` - chain through reasoning |
| Complexity = COMPLEX/CHAOTIC | `sequential` |
| stability_score < 0.3 | `parallel` |
| Otherwise | `none` - handle directly |

Termination conditions:
- Topology becomes TRIVIAL or SIMPLE
- beta_0 <= 2 and beta_1 <= 1
- 50% reduction from parent complexity
- Stability improved by 0.2+

## Usage

```python
from geometry_router import (
    create_default_registry,
    TopologicalFeatureExtractor,
    GeometryAwareRouter,
    RoutingContext
)

# Initialize
registry = create_default_registry()
extractor = TopologicalFeatureExtractor()
router = GeometryAwareRouter(registry=registry, feature_extractor=extractor)

# Route a query
embeddings = get_embeddings(query)  # Your embedding function
context = RoutingContext(
    source_model_id="previous-model-id",
    source_chunk_id="chunk-123"
)
decision = router.route(embeddings, context)

print(f"Route to: {decision.primary_model_id}")
print(f"Confidence: {decision.confidence}")
print(f"Overlap resolved: {decision.overlap_resolution_used}")
```

## Dependencies

**Required:**
- `numpy` - Core numerical operations
- `scipy` - Distance computations

**Optional (enhanced topology):**
- `ripser` - Exact persistent homology
- `persim` - Persistence images and distances

Without optional dependencies, the framework uses approximate topology based on spectral graph analysis.

## Configuration

Key parameters in `GeometryAwareRouter`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sinkhorn_iterations` | 20 | Iterations for doubly stochastic projection |
| `overlap_threshold` | 0.1 | Score difference for overlap detection |
| `hierarchy_weight` | 0.3 | Weight for hierarchical distance |
| `topology_weight` | 0.7 | Weight for topological distance |

## Online Learning

The router supports online learning of model specializations:

```python
# After successful routing, update model profile
router.update_model_profile(
    model_id="claude-sonnet-code",
    successful_signature=query_signature,
    learning_rate=0.1
)
```

This updates the model's persistence centroid using exponential moving average, improving future routing decisions.
