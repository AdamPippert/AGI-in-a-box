# AGI-in-a-Box Deployment Guide

This guide covers deploying AGI-in-a-Box as containerized services using Docker/Podman or Kubernetes.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Load Balancer / Ingress                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐          ┌────▼─────┐         ┌────▼─────┐
    │ Geometry │          │ CrewAI   │         │  Model   │
    │  Router  │          │Orchestr. │         │   Pool   │
    │ (Stateless)│        │(Stateful)│         │(Ollama)  │
    └────┬────┘          └────┬─────┘         └────┬─────┘
         │                    │                    │
    ┌────┴────────────────────┴────────────────────┴────┐
    │              Internal Service Network              │
    ├───────────────────────────────────────────────────┤
    │  Redis    │  PostgreSQL  │  RabbitMQ  │  Qdrant  │
    │  (Cache)  │  (State)     │  (Queue)   │  (Vector)│
    └───────────────────────────────────────────────────┘
```

## Components

| Component | Description | Replicas | State |
|-----------|-------------|----------|-------|
| **geometry-router** | Core routing service using topological analysis | 2+ (HPA) | Stateless |
| **crewai-orchestrator** | Agent workflow execution | 1 | Stateful (memory) |
| **model-pool (Ollama)** | Local LLM inference | 1+ | Stateful (models) |
| **redis** | Model profile cache & session state | 1 | Stateful |
| **postgres** | Routing history & agent memory | 1 | Stateful |
| **qdrant** | Vector embeddings for memory | 1 | Stateful |
| **rabbitmq** | Async task queue | 1 | Stateful |

---

## Quick Start with Docker Compose / Podman

### Prerequisites

- Docker 20.10+ or Podman 4.0+
- Docker Compose v2 or podman-compose
- 8GB+ RAM (16GB+ recommended for local LLM)
- GPU with CUDA support (optional, for faster inference)

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/AdamPippert/AGI-in-a-box.git
cd AGI-in-a-box

# Copy environment template
cp .env.example .env

# Edit .env with your API keys and configuration
vim .env
```

### 2. Start Core Services

```bash
# Start router + redis (minimal deployment)
docker-compose up -d router redis

# Or with Podman
podman-compose up -d router redis
```

### 3. Start Full Stack (with local LLM)

```bash
# Start all services including Ollama
docker-compose --profile full up -d

# Pull a model into Ollama
docker exec -it agi-ollama ollama pull mixtral
```

### 4. Start with Monitoring

```bash
# Include Prometheus and Grafana
docker-compose --profile full --profile monitoring up -d
```

### Available Profiles

| Profile | Services Included |
|---------|-------------------|
| (default) | router, crewai, redis |
| `local-llm` | + ollama |
| `persistence` | + postgres |
| `memory` | + qdrant, rabbitmq |
| `monitoring` | + prometheus, grafana |
| `full` | All services |

### Verify Deployment

```bash
# Check service health
curl http://localhost:8080/health

# List available models
curl http://localhost:8080/models

# Test routing
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]]}'
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.25+
- kubectl configured
- Kustomize (built into kubectl 1.14+)
- Persistent storage provisioner
- (Optional) NVIDIA GPU Operator for GPU nodes

### 1. Deploy to Development

```bash
# Preview the manifests
kubectl kustomize k8s/overlays/dev

# Apply to cluster
kubectl apply -k k8s/overlays/dev

# Watch deployment progress
kubectl -n agi-dev get pods -w
```

### 2. Deploy to Production

```bash
# Create secrets first (don't commit real secrets!)
kubectl -n agi-prod create secret generic crewai-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...

# Deploy
kubectl apply -k k8s/overlays/prod

# Verify
kubectl -n agi-prod get all
```

### 3. Configure Ingress

Edit `k8s/base/ingress.yaml` with your domain:

```yaml
spec:
  rules:
    - host: agi.yourdomain.com
```

### 4. Scale the Router

```bash
# Manual scaling
kubectl -n agi-prod scale deployment geometry-router --replicas=5

# HPA will auto-scale based on CPU/memory
kubectl -n agi-prod get hpa
```

### Storage Classes

Update `storageClassName` in `k8s/base/storage.yaml` for your cluster:

| Cloud Provider | Storage Class |
|----------------|---------------|
| AWS EKS | `gp3` |
| GKE | `standard-rwo` |
| AKS | `managed-premium` |
| Local/Bare Metal | `local-path` |

---

## GPU Support

### Docker/Podman with NVIDIA GPU

```bash
# Ensure nvidia-container-toolkit is installed
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Start with GPU support (already configured in docker-compose.yml)
docker-compose --profile local-llm up -d

# Verify GPU access
docker exec agi-ollama nvidia-smi
```

### Kubernetes with GPU

1. Install NVIDIA GPU Operator:
```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace
```

2. The `model-pool-statefulset.yaml` already requests GPUs:
```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

---

## Model Management

### Pre-loading Models

Create an init container or Job to pre-pull models:

```bash
# Pull models into Ollama
docker exec agi-ollama ollama pull mixtral
docker exec agi-ollama ollama pull llama2:70b
docker exec agi-ollama ollama pull codellama

# List available models
docker exec agi-ollama ollama list
```

### Using External APIs

Configure in `.env` or ConfigMap:

```bash
# For Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
```

---

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus at `http://localhost:9090` (or your ingress URL).

Key metrics:
- `router_requests_total` - Total routing requests
- `router_request_duration_seconds` - Routing latency
- `model_selection_count` - Per-model selection frequency
- `topology_complexity_histogram` - Query complexity distribution

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin).

Import dashboards from `deploy/grafana/dashboards/`.

### Logging

```bash
# View router logs
docker-compose logs -f router

# Kubernetes
kubectl -n agi-prod logs -f deployment/geometry-router
```

---

## Scaling Considerations

### Horizontal Scaling

| Component | Scaling Strategy |
|-----------|------------------|
| geometry-router | Stateless, scale freely via HPA |
| crewai-orchestrator | Single instance (agent memory consistency) |
| model-pool | Scale based on inference demand, GPU availability |
| redis | Single instance or Redis Cluster for HA |

### Resource Recommendations

| Environment | Router CPU | Router RAM | Model Pool RAM | Model Pool GPU |
|-------------|------------|------------|----------------|----------------|
| Dev | 0.5 core | 1GB | 8GB | Optional |
| Prod (small) | 2 cores | 2GB | 16GB | 1x RTX 3090 |
| Prod (large) | 4 cores | 4GB | 32GB | 2x A100 |

---

## Troubleshooting

### Common Issues

**Router fails to start:**
```bash
# Check dependencies
docker-compose logs redis
kubectl -n agi-prod get pods -l app=redis
```

**Model routing returns low confidence:**
```bash
# Check if models are loaded
curl http://localhost:11434/api/tags

# Verify topology extraction
curl -X POST http://localhost:8080/route \
  -d '{"embeddings": [[...]], "debug": true}'
```

**GPU not detected:**
```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Kubernetes
kubectl -n agi-prod describe pod -l app=model-pool
```

### Health Checks

```bash
# All services health
curl http://localhost:8080/health  # Router
curl http://localhost:11434/       # Ollama
redis-cli -h localhost ping        # Redis
```

---

## Security Considerations

1. **Secrets Management**: Use Kubernetes Secrets, HashiCorp Vault, or cloud-native secret managers
2. **Network Policies**: The included NetworkPolicy restricts traffic to necessary paths
3. **Non-root containers**: All containers run as non-root users
4. **API Authentication**: Add authentication middleware for production (not included)
5. **TLS**: Configure TLS termination at ingress level

---

## Backup & Recovery

### Redis State
```bash
# Backup
docker exec agi-redis redis-cli BGSAVE
docker cp agi-redis:/data/dump.rdb ./backup/

# Restore
docker cp ./backup/dump.rdb agi-redis:/data/
docker restart agi-redis
```

### PostgreSQL
```bash
# Backup
docker exec agi-postgres pg_dump -U agi agi_routing > backup.sql

# Restore
docker exec -i agi-postgres psql -U agi agi_routing < backup.sql
```

### Model Profiles (Ollama)
```bash
# Models are stored in the ollama-models volume
docker run --rm -v ollama-models:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama-models.tar.gz /data
```

---

## Next Steps

- Set up CI/CD pipeline for automated deployments
- Configure alerting rules in Prometheus
- Implement API authentication/authorization
- Add custom Grafana dashboards for your use case
- Consider service mesh (Istio/Linkerd) for advanced traffic management
