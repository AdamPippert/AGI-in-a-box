"""
Geometry-Aware Hierarchical Router package.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

try:
    from .topology.feature_extractor import (
        TopologicalFeatureExtractor,
        TopologicalSignature,
        PersistenceDiagram,
        TopologyComplexity,
        compute_bottleneck_distance,
        compute_wasserstein_distance,
    )
    from .models.registry import (
        ModelRegistry,
        ModelNode,
        ModelTier,
        TopologicalCapability,
        TopologicalProfile,
        create_default_registry,
    )
    from .routing.router import (
        GeometryAwareRouter,
        RoutingDecision,
        RoutingContext,
        RecursiveRoutingController,
        sinkhorn_knopp,
    )

    __all__.extend([
        "TopologicalFeatureExtractor",
        "TopologicalSignature",
        "PersistenceDiagram",
        "TopologyComplexity",
        "compute_bottleneck_distance",
        "compute_wasserstein_distance",
        "ModelRegistry",
        "ModelNode",
        "ModelTier",
        "TopologicalCapability",
        "TopologicalProfile",
        "create_default_registry",
        "GeometryAwareRouter",
        "RoutingDecision",
        "RoutingContext",
        "RecursiveRoutingController",
        "sinkhorn_knopp",
    ])
except Exception:
    # Allow partial import in minimal environments used for adaptation-only tests.
    pass
