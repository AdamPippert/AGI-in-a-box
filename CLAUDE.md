# AGI-in-a-Box

## Goals

Multi-model orchestration framework with geometry-aware routing and prompt-to-VM execution:

- **geometry_router**: Topology-based query routing using persistent homology and Sinkhorn-Knopp projection
- **openprose**: Prompt collection to sandboxed VM translation layer for research workflows
- **CrewAI agents**: Self-configuration and task orchestration workflows

## Environment

```bash
# Setup
poetry install

# Run scripts
poetry run python {script.py}

# Lint
poetry run flake8

# Start services (Docker)
docker-compose up -d

# Kubernetes
kubectl apply -k k8s/overlays/dev/
```

**Python**: ^3.11  
**Key deps**: crewai, grpcio, pydantic, numpy, scipy

**Config**: Copy `.env.example` to `.env` and configure API keys/ports.

## Patterns

- **Type hints**: Use throughout; modules include `py.typed`
- **Dataclasses**: Prefer `@dataclass` for data structures with `to_dict()`/`from_dict()` methods
- **Async**: Use `async`/`await` for I/O-bound operations; VM execution is async
- **Enums**: Use for fixed sets of values (states, categories, types)
- **Module structure**: Each package has `__init__.py` exporting public API

**geometry_router architecture**:
- `topology/`: Persistent homology feature extraction (Betti numbers)
- `routing/`: Sinkhorn-Knopp constrained routing matrices
- `models/`: 4-tier model registry (Orchestrator → Specialist → Executor → Verifier)

**openprose architecture**:
- `prompts/`: YAML collections with schemas, loaded via `load_collection()`
- `vm/`: ProseVM with Sandbox (resource limits, policies, checkpointing)
- `pipelines/`: DAG-based orchestration with parallel branches

## Avoided

- Mutable default arguments in function signatures
- Bare `except:` clauses - catch specific exceptions
- `print()` for logging - use structured logging
- Hardcoded credentials - use environment variables
- Blocking calls in async contexts - use `asyncio.to_thread()` for sync functions
- Comments explaining what code does - names should be self-documenting
