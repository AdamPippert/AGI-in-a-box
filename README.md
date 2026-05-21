AGI-in-a-box

Exactly what it sounds like: an LLM based collection of agent workflows that performs some of the tasks an AGI 
might need.  The title is pretentious, but the work is unglamorous.

Those tasks include things like:
  - self-configuration (change Ansible variable values on the fly and reconfigured the install)
  - determine if exterior cloud-based agents need to run to accomplish a task, and execute those agents
  - commit existing request history to a git repository automatically to provide a complete history of use
  - backup existing repository history past a specific threshold to low cost cloud backup target
  - backup existing model history, if desired, to a low cost cloud backup target for reuse
  - determine if new models should be downloaded and used in place of existing models, and obtain them
  - write LLM chat output to specific files for reuse, as needed, to reconfigure the main application
  - maintain a dynamic list of data sources and files to ingest via RAG for given tasks
  - maintain a prompt library for easy retrieval and reuse via Ansible variables and version control
  - methods for transferring the application to new hardware for upgrade or expansion

Why build this??
  - why not?
  - hardware configuration and orchestration keeps AGI practical and yours
  - as you use AI, the knowledge grows but so does the data and history
  - This can act as a model for other large-scale open source AI apps

Prerequisites:
  - install poetry to build an isolated environment for dependencies
  - run "poetry init" to install those dependencies
  - to run any of the agent workflows, type "poetry run python {name of script}"

Right now, these workflows in here are just tests so I can understand how agentic frameworks work.

## Geometry-Aware Hierarchical Routing

The `geometry_router` package implements a topology-based multi-model routing framework that uses persistent homology to characterize the "shape" of queries and route them to appropriate specialist models.

### Research Foundations

This framework synthesizes three cutting-edge research threads:

1. **DeepSeek mHC** (arXiv 2512.24880) - Manifold-Constrained Hyper-Connections using Sinkhorn-Knopp projection onto the Birkhoff polytope
2. **MIT Recursive Language Models** (arXiv 2512.24601) - Recursive decomposition for processing 10M+ token inputs
3. **Topological Data Analysis** - Persistent homology for problem characterization via Betti numbers

### Key Features

- **Topological query characterization** using persistence diagrams and Betti numbers
- **Manifold-constrained routing** via Sinkhorn-Knopp doubly stochastic matrix projection
- **Hierarchical overlap resolution** using combined topological + tree distance
- **Recursive decomposition control** based on topological complexity signals
- **Online learning** of model specializations from successful routings

### Quick Start

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
decision = router.route(embeddings)

print(f"Route to: {decision.primary_model_id}")
print(f"Confidence: {decision.confidence}")
```

### Dependencies

Add to your `pyproject.toml`:
```toml
numpy = "^1.24"
scipy = "^1.10"
ripser = "^0.6"      # Optional: exact persistent homology
persim = "^0.3"      # Optional: persistence images
```

See `geometry_router/docs/ARCHITECTURE.md` for detailed documentation.

## Adaptive Residual Architecture (Optional)

AGI-in-a-box now includes an optional three-layer adaptation extension integrated with `geometry_router`.

- Layer 1: **When the system seems unsure, it looks things up.**
- Layer 2: **When the system gets corrected, it remembers the fix for similar situations.**
- Layer 3: **Only after the same issue happens enough times does the system prepare a deeper learning update.**

### Safety model

- Layers are configuration-gated and disabled by default.
- Runtime does not mutate live base model weights.
- Durable learning outputs are isolated JSONL artifacts for offline review/training.

### Configuration toggles

See `.env.example` for:

- `ADAPT_LAYER1_ENABLED`, `ADAPT_LAYER2_ENABLED`, `ADAPT_LAYER3_ENABLED`
- threshold controls for residual trigger, memory matching, and adaptation eligibility
- persistence/export paths for residual memory and adaptation artifacts

Detailed design: `docs/residual_adaptation_architecture.md`.
