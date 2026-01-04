#!/usr/bin/env python3
"""
Demo: Geometry-Aware Hierarchical Routing

This example demonstrates the core routing functionality using
synthetic embeddings to simulate different query topologies.
"""

import numpy as np
from geometry_router import (
    create_default_registry,
    TopologicalFeatureExtractor,
    GeometryAwareRouter,
    RoutingContext,
    RecursiveRoutingController
)


def generate_simple_embeddings(n_points: int = 50, dim: int = 64) -> np.ndarray:
    """Generate embeddings with simple topology (single cluster)."""
    center = np.random.randn(dim)
    return center + 0.1 * np.random.randn(n_points, dim)


def generate_fragmented_embeddings(
    n_clusters: int = 8,
    points_per_cluster: int = 20,
    dim: int = 64
) -> np.ndarray:
    """Generate embeddings with high beta_0 (many disconnected components)."""
    embeddings = []
    for _ in range(n_clusters):
        center = np.random.randn(dim) * 5  # Spread clusters apart
        cluster = center + 0.1 * np.random.randn(points_per_cluster, dim)
        embeddings.append(cluster)
    return np.vstack(embeddings)


def generate_entangled_embeddings(n_points: int = 100, dim: int = 64) -> np.ndarray:
    """Generate embeddings with high beta_1 (loops/cycles)."""
    # Create points on multiple overlapping circles in high-dim space
    embeddings = []
    n_circles = 5

    for i in range(n_circles):
        t = np.linspace(0, 2 * np.pi, n_points // n_circles)
        # Random plane for each circle
        v1 = np.random.randn(dim)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.random.randn(dim)
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)

        circle = np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2)
        circle = circle + np.random.randn(dim) * 0.5  # Shift center
        circle = circle + 0.05 * np.random.randn(*circle.shape)  # Add noise
        embeddings.append(circle)

    return np.vstack(embeddings)


def main():
    print("=" * 60)
    print("Geometry-Aware Hierarchical Routing Demo")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing components...")
    registry = create_default_registry()
    extractor = TopologicalFeatureExtractor(use_approximate=True)
    router = GeometryAwareRouter(
        registry=registry,
        feature_extractor=extractor
    )
    recursive_controller = RecursiveRoutingController(router)

    print(f"   Registered {len(registry.all_models)} models")
    print(registry.summary())

    # Test 1: Simple query (should route to fast executor)
    print("\n" + "-" * 60)
    print("2. Testing SIMPLE query (single cluster)...")
    simple_embeddings = generate_simple_embeddings()
    decision = router.route(simple_embeddings)

    print(f"   Primary model: {decision.primary_model_id}")
    print(f"   Confidence: {decision.confidence:.4f}")
    print(f"   Fallbacks: {decision.fallback_model_ids}")
    print(f"   Overlap resolution used: {decision.overlap_resolution_used}")

    # Check decomposition decision
    sig = extractor.extract(simple_embeddings)
    should_decompose, strategy = recursive_controller.should_decompose(sig, 0)
    print(f"   Betti profile: {sig.betti_profile}")
    print(f"   Complexity: {sig.complexity.name}")
    print(f"   Should decompose: {should_decompose} (strategy: {strategy})")

    # Test 2: Fragmented query (should route to B0_REDUCTION specialist)
    print("\n" + "-" * 60)
    print("3. Testing FRAGMENTED query (many clusters)...")
    fragmented_embeddings = generate_fragmented_embeddings()
    decision = router.route(fragmented_embeddings)

    print(f"   Primary model: {decision.primary_model_id}")
    print(f"   Confidence: {decision.confidence:.4f}")
    print(f"   Fallbacks: {decision.fallback_model_ids}")
    print(f"   Overlap resolution used: {decision.overlap_resolution_used}")

    sig = extractor.extract(fragmented_embeddings)
    should_decompose, strategy = recursive_controller.should_decompose(sig, 0)
    print(f"   Betti profile: {sig.betti_profile}")
    print(f"   Complexity: {sig.complexity.name}")
    print(f"   Should decompose: {should_decompose} (strategy: {strategy})")

    # Test 3: Entangled query (should route to B1_REDUCTION specialist)
    print("\n" + "-" * 60)
    print("4. Testing ENTANGLED query (many loops)...")
    entangled_embeddings = generate_entangled_embeddings()
    decision = router.route(entangled_embeddings)

    print(f"   Primary model: {decision.primary_model_id}")
    print(f"   Confidence: {decision.confidence:.4f}")
    print(f"   Fallbacks: {decision.fallback_model_ids}")
    print(f"   Overlap resolution used: {decision.overlap_resolution_used}")

    sig = extractor.extract(entangled_embeddings)
    should_decompose, strategy = recursive_controller.should_decompose(sig, 0)
    print(f"   Betti profile: {sig.betti_profile}")
    print(f"   Complexity: {sig.complexity.name}")
    print(f"   Should decompose: {should_decompose} (strategy: {strategy})")

    # Test 4: Constrained routing
    print("\n" + "-" * 60)
    print("5. Testing CONSTRAINED routing (max latency 500ms)...")
    context = RoutingContext(
        max_latency_ms=500,
        required_capabilities={"coding"}
    )
    decision = router.route(simple_embeddings, context)

    print(f"   Primary model: {decision.primary_model_id}")
    print(f"   Confidence: {decision.confidence:.4f}")
    print(f"   (Constrained to models with latency < 500ms and 'coding' capability)")

    # Test 5: Overlap resolution with source context
    print("\n" + "-" * 60)
    print("6. Testing OVERLAP RESOLUTION with source context...")
    context = RoutingContext(
        source_model_id="claude-opus-4",
        source_chunk_id="chunk-001"
    )
    decision = router.route(fragmented_embeddings, context)

    print(f"   Primary model: {decision.primary_model_id}")
    print(f"   Confidence: {decision.confidence:.4f}")
    print(f"   Overlap resolution used: {decision.overlap_resolution_used}")
    if decision.hierarchy_distances:
        print(f"   Hierarchy distances: {decision.hierarchy_distances}")
    if decision.topological_distances:
        print(f"   Topo distances (sample): ", end="")
        sample = dict(list(decision.topological_distances.items())[:3])
        print({k: f"{v:.4f}" for k, v in sample.items()})

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
