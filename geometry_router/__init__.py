"""
Geometry-Aware Hierarchical Router

A topology-based multi-model routing framework that uses:
- Persistent homology for query characterization
- Manifold-constrained (Sinkhorn-Knopp) routing matrices
- Hierarchical overlap resolution
"""

from .topology.feature_extractor import (
    TopologicalFeatureExtractor,
    TopologicalSignature,
    PersistenceDiagram,
    TopologyComplexity,
    compute_bottleneck_distance,
    compute_wasserstein_distance
)

from .models.registry import (
    ModelRegistry,
    ModelNode,
    ModelTier,
    TopologicalCapability,
    TopologicalProfile,
    create_default_registry
)

from .routing.router import (
    GeometryAwareRouter,
    RoutingDecision,
    RoutingContext,
    RecursiveRoutingController,
    sinkhorn_knopp
)

__version__ = "0.1.0"
__all__ = [
    # Topology
    "TopologicalFeatureExtractor",
    "TopologicalSignature",
    "PersistenceDiagram",
    "TopologyComplexity",
    "compute_bottleneck_distance",
    "compute_wasserstein_distance",

    # Models
    "ModelRegistry",
    "ModelNode",
    "ModelTier",
    "TopologicalCapability",
    "TopologicalProfile",
    "create_default_registry",

    # Routing
    "GeometryAwareRouter",
    "RoutingDecision",
    "RoutingContext",
    "RecursiveRoutingController",
    "sinkhorn_knopp"
]
