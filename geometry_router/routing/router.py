"""
Geometry-Aware Hierarchical Router

Implements manifold-constrained routing with:
1. Sinkhorn-Knopp projection for doubly stochastic routing matrices
2. Topological similarity scoring
3. Overlap resolution via minimal hierarchical distance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple, Set
from dataclasses import dataclass
import logging

from ..topology.feature_extractor import (
    TopologicalSignature,
    TopologicalFeatureExtractor,
    compute_bottleneck_distance,
    compute_wasserstein_distance,
    TopologyComplexity
)
from ..models.registry import (
    ModelRegistry,
    ModelNode,
    ModelTier,
    TopologicalCapability
)

logger = logging.getLogger(__name__)


class RoutingDecision(NamedTuple):
    """Result of a routing decision."""
    primary_model_id: str
    fallback_model_ids: List[str]
    confidence: float
    routing_scores: Dict[str, float]
    overlap_resolution_used: bool
    hierarchy_distances: Dict[str, int]
    topological_distances: Dict[str, float]


@dataclass
class RoutingContext:
    """Context for a routing decision."""
    query_signature: TopologicalSignature = None
    source_chunk_id: Optional[str] = None
    source_model_id: Optional[str] = None  # Model that produced the chunk
    required_capabilities: Set[str] = None
    max_latency_ms: Optional[float] = None
    max_cost_per_token: Optional[float] = None
    prefer_tier: Optional[ModelTier] = None


def sinkhorn_knopp(
    matrix: np.ndarray,
    n_iterations: int = 20,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Project matrix onto Birkhoff polytope (doubly stochastic matrices).

    The Birkhoff polytope B_n = {X in R^{n x n} : X >= 0, X1 = 1, X^T 1 = 1}
    is the set of all doubly stochastic matrices.

    This ensures:
    - Spectral norm <= 1 (gradient stability)
    - Closure under composition (depth stability)
    - Bidirectional load balancing

    Args:
        matrix: Non-negative input matrix
        n_iterations: Number of alternating normalization steps
        epsilon: Small constant for numerical stability

    Returns:
        Doubly stochastic matrix (rows and columns sum to 1)
    """
    # Ensure non-negative
    M = np.maximum(matrix, epsilon)

    for _ in range(n_iterations):
        # Row normalization
        M = M / (M.sum(axis=1, keepdims=True) + epsilon)
        # Column normalization
        M = M / (M.sum(axis=0, keepdims=True) + epsilon)

    return M


def compute_topological_distance(
    sig1: TopologicalSignature,
    sig2: TopologicalSignature,
    method: str = "combined"
) -> float:
    """
    Compute topological distance between two signatures.

    Args:
        sig1, sig2: Topological signatures to compare
        method: "bottleneck", "wasserstein", or "combined"

    Returns:
        Distance value (lower = more similar)
    """
    if method == "bottleneck":
        return compute_bottleneck_distance(sig1, sig2)
    elif method == "wasserstein":
        return compute_wasserstein_distance(sig1, sig2)
    else:  # combined
        bn = compute_bottleneck_distance(sig1, sig2)
        ws = compute_wasserstein_distance(sig1, sig2)
        # Weighted combination: bottleneck for global, wasserstein for local
        return 0.6 * bn + 0.4 * ws


class GeometryAwareRouter:
    """
    Routes queries/chunks to models based on topological characteristics.

    Core algorithm:
    1. Extract topological signature of input
    2. Compute compatibility with each candidate model's profile
    3. Build routing matrix and project via Sinkhorn-Knopp
    4. Resolve overlaps using minimal hierarchical distance
    5. Return ranked model selection with confidence
    """

    def __init__(
        self,
        registry: ModelRegistry,
        feature_extractor: TopologicalFeatureExtractor = None,
        sinkhorn_iterations: int = 20,
        overlap_threshold: float = 0.1,
        hierarchy_weight: float = 0.3,
        topology_weight: float = 0.7
    ):
        """
        Args:
            registry: Model registry with hierarchy and profiles
            feature_extractor: Topology feature extractor (created if None)
            sinkhorn_iterations: Iterations for doubly stochastic projection
            overlap_threshold: Score difference threshold for overlap detection
            hierarchy_weight: Weight for hierarchical distance in overlap resolution
            topology_weight: Weight for topological distance in overlap resolution
        """
        self.registry = registry
        self.extractor = feature_extractor or TopologicalFeatureExtractor()
        self.sinkhorn_iters = sinkhorn_iterations
        self.overlap_threshold = overlap_threshold
        self.hierarchy_weight = hierarchy_weight
        self.topology_weight = topology_weight

        # Cache for model topological signatures (computed from profile centroids)
        self._model_signatures: Dict[str, TopologicalSignature] = {}

    def route(
        self,
        embeddings: np.ndarray,
        context: RoutingContext = None
    ) -> RoutingDecision:
        """
        Route input to optimal model based on topology.

        Args:
            embeddings: Input embedding matrix (n_tokens x embed_dim)
            context: Additional routing context

        Returns:
            RoutingDecision with selected model and metadata
        """
        # Extract topological signature
        signature = self.extractor.extract(embeddings)

        if context is None:
            context = RoutingContext(query_signature=signature)
        else:
            context.query_signature = signature

        # Get candidate models
        candidates = self._get_candidates(context)

        if not candidates:
            # Fallback to root
            root = self.registry.root
            return RoutingDecision(
                primary_model_id=root.model_id if root else "unknown",
                fallback_model_ids=[],
                confidence=0.0,
                routing_scores={},
                overlap_resolution_used=False,
                hierarchy_distances={},
                topological_distances={}
            )

        # Compute raw routing scores
        raw_scores = self._compute_routing_scores(signature, candidates, context)

        # Build and project routing matrix
        routing_matrix = self._build_routing_matrix(raw_scores, candidates)
        projected_matrix = sinkhorn_knopp(routing_matrix, self.sinkhorn_iters)

        # Extract model scores from projected matrix (first row = query routing)
        projected_scores = {
            candidates[i].model_id: float(projected_matrix[0, i])
            for i in range(len(candidates))
        }

        # Detect and resolve overlaps
        ranked = sorted(projected_scores.items(), key=lambda x: x[1], reverse=True)

        overlap_used = False
        topo_distances = {}
        hier_distances = {}

        # Check for overlap between top candidates
        if len(ranked) >= 2:
            top_score = ranked[0][1]
            overlapping = [
                (mid, score) for mid, score in ranked
                if abs(score - top_score) < self.overlap_threshold
            ]

            if len(overlapping) > 1:
                overlap_used = True
                # Resolve overlap using minimal topological distance to hierarchy
                primary_id = self._resolve_overlap(
                    overlapping,
                    signature,
                    context,
                    topo_distances,
                    hier_distances
                )

                # Reorder ranked list
                ranked = [(primary_id, projected_scores[primary_id])] + [
                    (mid, s) for mid, s in ranked if mid != primary_id
                ]

        # Build final decision
        primary = ranked[0][0]
        fallbacks = [mid for mid, _ in ranked[1:4]]  # Top 3 fallbacks
        confidence = ranked[0][1]

        # Compute remaining distances for diagnostics
        if not topo_distances:
            for model in candidates:
                topo_distances[model.model_id] = self._compute_model_topo_distance(
                    signature, model
                )

        if not hier_distances and context.source_model_id:
            for model in candidates:
                hier_distances[model.model_id] = self.registry.compute_hierarchy_distance(
                    context.source_model_id, model.model_id
                )

        return RoutingDecision(
            primary_model_id=primary,
            fallback_model_ids=fallbacks,
            confidence=confidence,
            routing_scores=projected_scores,
            overlap_resolution_used=overlap_used,
            hierarchy_distances=hier_distances,
            topological_distances=topo_distances
        )

    def _get_candidates(self, context: RoutingContext) -> List[ModelNode]:
        """Get candidate models based on context constraints."""
        candidates = []

        for model in self.registry.all_models:
            # Filter by tier preference
            if context.prefer_tier and model.tier != context.prefer_tier:
                continue

            # Filter by latency constraint
            if context.max_latency_ms and model.latency_ms > context.max_latency_ms:
                continue

            # Filter by cost constraint
            if context.max_cost_per_token and model.cost_per_token > context.max_cost_per_token:
                continue

            # Filter by required capabilities
            if context.required_capabilities:
                if not context.required_capabilities.intersection(model.capability_tags):
                    continue

            candidates.append(model)

        # If no candidates after filtering, return all models
        if not candidates:
            candidates = self.registry.all_models

        return candidates

    def _compute_routing_scores(
        self,
        signature: TopologicalSignature,
        candidates: List[ModelNode],
        context: RoutingContext
    ) -> Dict[str, float]:
        """Compute raw routing scores for each candidate."""
        scores = {}

        for model in candidates:
            # Base compatibility from topological profile
            topo_score = model.topo_profile.compatibility_score(
                betti_profile=signature.betti_profile,
                complexity=signature.complexity.name,
                persistence_image=signature.persistence_image
            )

            # Capability match bonus
            capability_bonus = 0.0
            required_transformation = self._infer_required_transformation(signature)
            if model.topo_profile.primary_capability == required_transformation:
                capability_bonus = 0.15
            elif model.topo_profile.primary_capability == TopologicalCapability.UNIVERSAL:
                capability_bonus = 0.05

            # Stability bonus for high-stability queries going to robust models
            stability_bonus = 0.0
            if signature.stability_score > 0.7 and model.base_priority > 0.7:
                stability_bonus = 0.05

            # Priority scaling
            priority_factor = model.base_priority

            # Combine scores
            total = (topo_score + capability_bonus + stability_bonus) * priority_factor
            scores[model.model_id] = float(np.clip(total, 0, 1))

        return scores

    def _infer_required_transformation(
        self,
        signature: TopologicalSignature
    ) -> TopologicalCapability:
        """Infer what topological transformation the query needs."""
        b0, b1, _ = signature.betti_profile

        # High beta_0 = fragmented -> needs clustering (B0_REDUCTION)
        if b0 > 5:
            return TopologicalCapability.B0_REDUCTION

        # High beta_1 = entangled -> needs untangling (B1_REDUCTION)
        if b1 > 3:
            return TopologicalCapability.B1_REDUCTION

        # Simple topology -> preserve structure (creative, simple tasks)
        if signature.complexity in {TopologyComplexity.TRIVIAL, TopologyComplexity.SIMPLE}:
            return TopologicalCapability.STRUCTURE_PRESERVE

        # Default to universal
        return TopologicalCapability.UNIVERSAL

    def _build_routing_matrix(
        self,
        scores: Dict[str, float],
        candidates: List[ModelNode]
    ) -> np.ndarray:
        """
        Build routing matrix for Sinkhorn-Knopp projection.

        Matrix structure:
        - Row 0: Query -> Model affinities (from scores)
        - Rows 1+: Model -> Model affinities (from hierarchy)
        """
        n = len(candidates) + 1  # +1 for query
        matrix = np.zeros((n, n))

        # Query -> Model affinities (row 0)
        for i, model in enumerate(candidates):
            matrix[0, i + 1] = scores.get(model.model_id, 0.1)

        # Model -> Model affinities (hierarchy-based)
        for i, model_i in enumerate(candidates):
            for j, model_j in enumerate(candidates):
                if i == j:
                    matrix[i + 1, j + 1] = 0.5  # Self-affinity
                else:
                    # Affinity inversely proportional to hierarchy distance
                    dist = self.registry.compute_hierarchy_distance(
                        model_i.model_id, model_j.model_id
                    )
                    matrix[i + 1, j + 1] = 1.0 / (1.0 + dist)

        # Ensure non-negative and positive
        matrix = np.maximum(matrix, 1e-6)

        return matrix

    def _resolve_overlap(
        self,
        overlapping: List[Tuple[str, float]],
        signature: TopologicalSignature,
        context: RoutingContext,
        topo_distances: Dict[str, float],
        hier_distances: Dict[str, int]
    ) -> str:
        """
        Resolve overlap between models with similar scores.

        Key insight: When models overlap their topologies, use the model
        that most closely matches a minimal topological distance to other
        parts of the hierarchy from which the chunk is derived.

        Combined metric:
        overlap_score = (1 - alpha) * topo_distance + alpha * hier_distance

        Select model with MINIMUM overlap_score.
        """
        candidates_ids = [mid for mid, _ in overlapping]

        # Compute topological distance to query for each
        for mid in candidates_ids:
            model = self.registry.get(mid)
            topo_distances[mid] = self._compute_model_topo_distance(signature, model)

        # Compute hierarchical distance from source (if known)
        source_id = context.source_model_id
        if source_id:
            for mid in candidates_ids:
                hier_distances[mid] = self.registry.compute_hierarchy_distance(
                    source_id, mid
                )
        else:
            # No source context: use distance from root
            root_id = self.registry.root.model_id if self.registry.root else None
            if root_id:
                for mid in candidates_ids:
                    hier_distances[mid] = self.registry.compute_hierarchy_distance(
                        root_id, mid
                    )
            else:
                # No hierarchy info: rely solely on topology
                for mid in candidates_ids:
                    hier_distances[mid] = 0

        # Normalize distances
        max_topo = max(topo_distances.values()) or 1.0
        max_hier = max(hier_distances.values()) or 1.0

        # Compute combined overlap scores
        overlap_scores = {}
        for mid in candidates_ids:
            norm_topo = topo_distances[mid] / max_topo
            norm_hier = hier_distances[mid] / max_hier

            # Combined: minimize both topological and hierarchical distance
            overlap_scores[mid] = (
                self.topology_weight * norm_topo +
                self.hierarchy_weight * norm_hier
            )

        # Select model with minimum overlap score
        best_model = min(overlap_scores.items(), key=lambda x: x[1])[0]

        logger.info(
            f"Overlap resolution: {len(overlapping)} candidates, "
            f"selected {best_model} with score {overlap_scores[best_model]:.4f}"
        )

        return best_model

    def _compute_model_topo_distance(
        self,
        query_signature: TopologicalSignature,
        model: ModelNode
    ) -> float:
        """
        Compute topological distance between query and model's profile.

        Uses model's persistence centroid if available, otherwise
        synthesizes from profile parameters.
        """
        # Check cache
        if model.model_id in self._model_signatures:
            model_sig = self._model_signatures[model.model_id]
        else:
            # Synthesize signature from profile
            model_sig = self._synthesize_model_signature(model)
            self._model_signatures[model.model_id] = model_sig

        return compute_topological_distance(query_signature, model_sig)

    def _synthesize_model_signature(self, model: ModelNode) -> TopologicalSignature:
        """
        Synthesize a topological signature from model profile.

        This is used when we don't have learned centroids from
        successful routing history.
        """
        profile = model.topo_profile

        # Create synthetic persistence image from optimal ranges
        pi_size = self.extractor.pi_size
        pi = np.zeros(pi_size)

        # Place mass at optimal Betti regions
        b0_mid = sum(profile.optimal_betti_0_range) / 2
        b1_mid = sum(profile.optimal_betti_1_range) / 2

        # Gaussian blobs at optimal points
        for i in range(pi_size[0]):
            for j in range(pi_size[1]):
                # Map to Betti space
                b0_val = i * 20 / pi_size[0]  # Assuming max beta_0 ~ 20
                b1_val = j * 10 / pi_size[1]  # Assuming max beta_1 ~ 10

                dist_b0 = abs(b0_val - b0_mid)
                dist_b1 = abs(b1_val - b1_mid)

                pi[i, j] = np.exp(-(dist_b0**2 + dist_b1**2) / 10)

        # Normalize
        if pi.max() > 0:
            pi = pi / pi.max()

        # Create synthetic persistence diagram
        from ..topology.feature_extractor import PersistenceDiagram

        h0 = np.array([[0.0, 0.3], [0.1, 0.4]])
        h1 = np.array([[0.2, 0.5]]) if b1_mid > 0 else np.array([]).reshape(0, 2)

        pd = PersistenceDiagram(h0=h0, h1=h1)

        # Infer complexity from profile
        complexity_map = {
            "TRIVIAL": TopologyComplexity.TRIVIAL,
            "SIMPLE": TopologyComplexity.SIMPLE,
            "MODERATE": TopologyComplexity.MODERATE,
            "COMPLEX": TopologyComplexity.COMPLEX,
            "CHAOTIC": TopologyComplexity.CHAOTIC
        }

        # Use most common complexity from profile
        if profile.optimal_complexity:
            complexity_str = list(profile.optimal_complexity)[0]
            complexity = complexity_map.get(complexity_str, TopologyComplexity.MODERATE)
        else:
            complexity = TopologyComplexity.MODERATE

        return TopologicalSignature(
            persistence_diagram=pd,
            persistence_image=pi,
            betti_profile=(int(b0_mid), int(b1_mid), 0),
            complexity=complexity,
            stability_score=0.7  # Neutral stability
        )

    def update_model_profile(
        self,
        model_id: str,
        successful_signature: TopologicalSignature,
        learning_rate: float = 0.1
    ) -> None:
        """
        Update model's topological profile based on successful routing.

        This enables online learning of model specializations.
        """
        model = self.registry.get(model_id)
        if not model:
            return

        profile = model.topo_profile

        # Update centroid with exponential moving average
        if profile.persistence_centroid is None:
            profile.persistence_centroid = successful_signature.persistence_image.copy()
        else:
            profile.persistence_centroid = (
                (1 - learning_rate) * profile.persistence_centroid +
                learning_rate * successful_signature.persistence_image
            )

        # Update covariance estimate
        diff = (
            successful_signature.persistence_image.flatten() -
            profile.persistence_centroid.flatten()
        )
        outer = np.outer(diff, diff)

        if profile.persistence_covariance is None:
            profile.persistence_covariance = outer
        else:
            profile.persistence_covariance = (
                (1 - learning_rate) * profile.persistence_covariance +
                learning_rate * outer
            )

        # Clear cached signature
        if model_id in self._model_signatures:
            del self._model_signatures[model_id]


class RecursiveRoutingController:
    """
    Controller for recursive routing following RLM patterns.

    Manages decomposition decisions and recursion depth based on
    topological complexity signals.
    """

    def __init__(
        self,
        router: GeometryAwareRouter,
        max_recursion_depth: int = 5,
        complexity_threshold: float = 0.3
    ):
        self.router = router
        self.max_depth = max_recursion_depth
        self.complexity_threshold = complexity_threshold

    def should_decompose(
        self,
        signature: TopologicalSignature,
        current_depth: int
    ) -> Tuple[bool, str]:
        """
        Decide if query should be decomposed into sub-queries.

        Returns:
            (should_decompose, strategy) where strategy is one of:
            - "parallel": High beta_0, decompose into parallel chunks
            - "sequential": High beta_1, chain through reasoning
            - "none": Simple enough to handle directly
        """
        if current_depth >= self.max_depth:
            return False, "max_depth"

        b0, b1, _ = signature.betti_profile

        # High beta_0 (many components) -> parallel decomposition
        if b0 > 5:
            return True, "parallel"

        # High beta_1 (many loops) -> sequential reasoning
        if b1 > 3:
            return True, "sequential"

        # Complex or chaotic -> needs decomposition
        if signature.complexity in {TopologyComplexity.COMPLEX, TopologyComplexity.CHAOTIC}:
            return True, "sequential"

        # Low stability -> might benefit from decomposition
        if signature.stability_score < self.complexity_threshold:
            return True, "parallel"

        return False, "none"

    def should_terminate(
        self,
        signature: TopologicalSignature,
        parent_signature: Optional[TopologicalSignature] = None
    ) -> bool:
        """
        Decide if recursion should terminate based on topological simplification.

        Terminate when topology has been sufficiently reduced.
        """
        # Simple topology = terminate
        if signature.complexity in {TopologyComplexity.TRIVIAL, TopologyComplexity.SIMPLE}:
            return True

        # Low Betti numbers = terminate
        b0, b1, _ = signature.betti_profile
        if b0 <= 2 and b1 <= 1:
            return True

        # If we have parent, check for sufficient reduction
        if parent_signature:
            pb0, pb1, _ = parent_signature.betti_profile

            # At least 50% reduction in complexity
            if b0 <= pb0 * 0.5 and b1 <= pb1 * 0.5:
                return True

            # Stability improved significantly
            if signature.stability_score > parent_signature.stability_score + 0.2:
                return True

        return False
