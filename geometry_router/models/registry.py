"""
Model Registry with Topological Profiles

Defines the model hierarchy and their topological specializations.
Each model has a profile describing what topological transformations
it excels at (beta_0 reduction, beta_1 reduction, structure preservation).
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class TopologicalCapability(Enum):
    """What topological transformations a model excels at."""
    B0_REDUCTION = "b0_reduction"      # Clustering disparate info
    B1_REDUCTION = "b1_reduction"      # Untangling circular dependencies
    B2_REDUCTION = "b2_reduction"      # Collapsing void structures
    STRUCTURE_PRESERVE = "preserve"     # Maintaining topology (creative)
    UNIVERSAL = "universal"             # General purpose


class ModelTier(Enum):
    """Hierarchical tier in the model architecture."""
    ORCHESTRATOR = 0   # Root routing/decomposition (e.g., Claude, GPT-4)
    SPECIALIST = 1     # Domain experts (code, math, creative)
    EXECUTOR = 2       # Fast execution models (Haiku, GPT-4-mini)
    VERIFIER = 3       # Validation/checking models


@dataclass
class TopologicalProfile:
    """
    Topological fingerprint of a model's capabilities.

    Describes what "shapes" of problems the model handles well,
    based on empirical performance on different topological signatures.
    """
    # Primary capability
    primary_capability: TopologicalCapability

    # Capability scores (0-1) for each transformation type
    capability_scores: Dict[TopologicalCapability, float] = field(default_factory=dict)

    # Optimal Betti number ranges this model handles
    optimal_betti_0_range: Tuple[int, int] = (0, 10)
    optimal_betti_1_range: Tuple[int, int] = (0, 5)

    # Complexity levels this model excels at
    optimal_complexity: Set[str] = field(default_factory=lambda: {"SIMPLE", "MODERATE"})

    # Centroid in persistence image space (learned from successful routings)
    persistence_centroid: Optional[np.ndarray] = None

    # Covariance of successful routing embeddings
    persistence_covariance: Optional[np.ndarray] = None

    def compatibility_score(
        self,
        betti_profile: Tuple[int, int, int],
        complexity: str,
        persistence_image: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute compatibility score for a given query topology.
        Higher = better match for this model.
        """
        score = 0.0
        b0, b1, b2 = betti_profile

        # Betti range compatibility
        b0_in_range = self.optimal_betti_0_range[0] <= b0 <= self.optimal_betti_0_range[1]
        b1_in_range = self.optimal_betti_1_range[0] <= b1 <= self.optimal_betti_1_range[1]

        if b0_in_range:
            score += 0.3
        else:
            # Penalty proportional to distance from range
            dist = min(abs(b0 - self.optimal_betti_0_range[0]),
                       abs(b0 - self.optimal_betti_0_range[1]))
            score -= 0.1 * dist

        if b1_in_range:
            score += 0.3
        else:
            dist = min(abs(b1 - self.optimal_betti_1_range[0]),
                       abs(b1 - self.optimal_betti_1_range[1]))
            score -= 0.1 * dist

        # Complexity compatibility
        if complexity in self.optimal_complexity:
            score += 0.2

        # Persistence image similarity (if centroid available)
        if persistence_image is not None and self.persistence_centroid is not None:
            pi_flat = persistence_image.flatten()
            centroid_flat = self.persistence_centroid.flatten()

            if self.persistence_covariance is not None:
                # Mahalanobis distance
                try:
                    cov_inv = np.linalg.pinv(self.persistence_covariance)
                    diff = pi_flat - centroid_flat
                    mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                    score += 0.2 * np.exp(-mahal_dist / 10)
                except Exception:
                    # Fallback to Euclidean
                    eucl_dist = np.linalg.norm(pi_flat - centroid_flat)
                    score += 0.2 * np.exp(-eucl_dist)
            else:
                eucl_dist = np.linalg.norm(pi_flat - centroid_flat)
                score += 0.2 * np.exp(-eucl_dist)

        return float(np.clip(score, 0, 1))


@dataclass
class ModelNode:
    """
    A model in the routing hierarchy.

    Contains identity, capabilities, and hierarchical relationships.
    """
    # Identity
    model_id: str
    display_name: str
    tier: ModelTier

    # Topological profile
    topo_profile: TopologicalProfile

    # Hierarchy relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Routing metadata
    base_priority: float = 0.5  # Default selection priority
    latency_ms: float = 100.0   # Typical response latency
    cost_per_token: float = 0.001
    max_context_length: int = 128000

    # Capability tags for non-topological routing
    capability_tags: Set[str] = field(default_factory=set)

    # API configuration
    api_endpoint: Optional[str] = None
    api_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "tier": self.tier.name,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "base_priority": self.base_priority,
            "latency_ms": self.latency_ms,
            "cost_per_token": self.cost_per_token,
            "capability_tags": list(self.capability_tags),
            "topo_profile": {
                "primary_capability": self.topo_profile.primary_capability.value,
                "optimal_betti_0_range": self.topo_profile.optimal_betti_0_range,
                "optimal_betti_1_range": self.topo_profile.optimal_betti_1_range,
                "optimal_complexity": list(self.topo_profile.optimal_complexity)
            }
        }


class ModelRegistry:
    """
    Registry of all models in the routing hierarchy.

    Maintains the hierarchical structure, topological profiles,
    and provides efficient lookup for routing decisions.
    """

    def __init__(self):
        self._models: Dict[str, ModelNode] = {}
        self._hierarchy_root: Optional[str] = None
        self._tier_index: Dict[ModelTier, List[str]] = {tier: [] for tier in ModelTier}
        self._capability_index: Dict[TopologicalCapability, List[str]] = {
            cap: [] for cap in TopologicalCapability
        }

    def register(self, model: ModelNode) -> None:
        """Register a model in the hierarchy."""
        self._models[model.model_id] = model
        self._tier_index[model.tier].append(model.model_id)
        self._capability_index[model.topo_profile.primary_capability].append(model.model_id)

        # Track root
        if model.tier == ModelTier.ORCHESTRATOR and model.parent_id is None:
            self._hierarchy_root = model.model_id

        # Update parent's children list
        if model.parent_id and model.parent_id in self._models:
            parent = self._models[model.parent_id]
            if model.model_id not in parent.children_ids:
                parent.children_ids.append(model.model_id)

    def get(self, model_id: str) -> Optional[ModelNode]:
        """Get model by ID."""
        return self._models.get(model_id)

    def get_by_tier(self, tier: ModelTier) -> List[ModelNode]:
        """Get all models at a tier."""
        return [self._models[mid] for mid in self._tier_index[tier]]

    def get_by_capability(self, capability: TopologicalCapability) -> List[ModelNode]:
        """Get models with a specific primary capability."""
        return [self._models[mid] for mid in self._capability_index[capability]]

    def get_children(self, model_id: str) -> List[ModelNode]:
        """Get direct children of a model."""
        model = self._models.get(model_id)
        if not model:
            return []
        return [self._models[cid] for cid in model.children_ids if cid in self._models]

    def get_siblings(self, model_id: str) -> List[ModelNode]:
        """Get sibling models (same parent)."""
        model = self._models.get(model_id)
        if not model or not model.parent_id:
            return []
        parent = self._models[model.parent_id]
        return [
            self._models[cid]
            for cid in parent.children_ids
            if cid != model_id and cid in self._models
        ]

    def get_ancestors(self, model_id: str) -> List[ModelNode]:
        """Get all ancestors up to root."""
        ancestors = []
        current = self._models.get(model_id)
        while current and current.parent_id:
            parent = self._models.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def compute_hierarchy_distance(self, model_a: str, model_b: str) -> int:
        """
        Compute hierarchical distance between two models.
        Distance = steps to common ancestor * 2 (up + down).
        """
        if model_a == model_b:
            return 0

        # Get ancestor chains
        ancestors_a = {model_a: 0}
        current = self._models.get(model_a)
        depth = 1
        while current and current.parent_id:
            ancestors_a[current.parent_id] = depth
            current = self._models.get(current.parent_id)
            depth += 1

        # Find first common ancestor from B
        current = self._models.get(model_b)
        depth_b = 0
        while current:
            if current.model_id in ancestors_a:
                return ancestors_a[current.model_id] + depth_b
            if current.parent_id:
                current = self._models.get(current.parent_id)
                depth_b += 1
            else:
                break

        # No common ancestor (shouldn't happen in well-formed hierarchy)
        return float('inf')

    @property
    def root(self) -> Optional[ModelNode]:
        """Get root orchestrator model."""
        return self._models.get(self._hierarchy_root) if self._hierarchy_root else None

    @property
    def all_models(self) -> List[ModelNode]:
        """Get all registered models."""
        return list(self._models.values())

    def to_dict(self) -> Dict:
        """Serialize entire registry."""
        return {
            "models": {mid: m.to_dict() for mid, m in self._models.items()},
            "root": self._hierarchy_root
        }

    def summary(self) -> str:
        """Human-readable summary of registry."""
        lines = ["Model Registry Summary:", "=" * 40]

        for tier in ModelTier:
            models = self.get_by_tier(tier)
            if models:
                lines.append(f"\n{tier.name} ({len(models)} models):")
                for m in models:
                    parent_info = f" (parent: {m.parent_id})" if m.parent_id else " (ROOT)"
                    lines.append(f"  - {m.display_name} [{m.model_id}]{parent_info}")
                    lines.append(f"    Primary: {m.topo_profile.primary_capability.value}")
                    lines.append(f"    Optimal beta_0: {m.topo_profile.optimal_betti_0_range}, "
                                 f"beta_1: {m.topo_profile.optimal_betti_1_range}")

        return "\n".join(lines)


def create_default_registry() -> ModelRegistry:
    """
    Create a default model registry with common models.
    This is a reference implementation - customize for your setup.
    """
    registry = ModelRegistry()

    # Orchestrator tier
    registry.register(ModelNode(
        model_id="claude-opus-4",
        display_name="Claude Opus 4",
        tier=ModelTier.ORCHESTRATOR,
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.UNIVERSAL,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.9,
                TopologicalCapability.B1_REDUCTION: 0.85,
                TopologicalCapability.STRUCTURE_PRESERVE: 0.9,
                TopologicalCapability.UNIVERSAL: 1.0
            },
            optimal_betti_0_range=(0, 20),
            optimal_betti_1_range=(0, 15),
            optimal_complexity={"TRIVIAL", "SIMPLE", "MODERATE", "COMPLEX"}
        ),
        base_priority=0.9,
        latency_ms=2000,
        cost_per_token=0.015,
        max_context_length=200000,
        capability_tags={"reasoning", "coding", "creative", "analysis", "orchestration"}
    ))

    # Specialist tier - Code
    registry.register(ModelNode(
        model_id="claude-sonnet-code",
        display_name="Claude Sonnet (Code Specialist)",
        tier=ModelTier.SPECIALIST,
        parent_id="claude-opus-4",
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.B1_REDUCTION,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.7,
                TopologicalCapability.B1_REDUCTION: 0.95,
                TopologicalCapability.STRUCTURE_PRESERVE: 0.6,
            },
            optimal_betti_0_range=(1, 8),
            optimal_betti_1_range=(0, 10),
            optimal_complexity={"MODERATE", "COMPLEX"}
        ),
        base_priority=0.8,
        latency_ms=1000,
        cost_per_token=0.003,
        max_context_length=200000,
        capability_tags={"coding", "debugging", "refactoring", "algorithms"}
    ))

    # Specialist tier - Math/Reasoning
    registry.register(ModelNode(
        model_id="deepseek-r1",
        display_name="DeepSeek R1 (Reasoning)",
        tier=ModelTier.SPECIALIST,
        parent_id="claude-opus-4",
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.B0_REDUCTION,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.95,
                TopologicalCapability.B1_REDUCTION: 0.8,
                TopologicalCapability.STRUCTURE_PRESERVE: 0.5,
            },
            optimal_betti_0_range=(2, 15),
            optimal_betti_1_range=(0, 5),
            optimal_complexity={"MODERATE", "COMPLEX", "CHAOTIC"}
        ),
        base_priority=0.85,
        latency_ms=3000,
        cost_per_token=0.002,
        max_context_length=128000,
        capability_tags={"math", "reasoning", "proof", "logic", "chain-of-thought"}
    ))

    # Specialist tier - Creative
    registry.register(ModelNode(
        model_id="claude-sonnet-creative",
        display_name="Claude Sonnet (Creative)",
        tier=ModelTier.SPECIALIST,
        parent_id="claude-opus-4",
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.STRUCTURE_PRESERVE,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.5,
                TopologicalCapability.B1_REDUCTION: 0.5,
                TopologicalCapability.STRUCTURE_PRESERVE: 0.95,
            },
            optimal_betti_0_range=(1, 5),
            optimal_betti_1_range=(0, 3),
            optimal_complexity={"TRIVIAL", "SIMPLE"}
        ),
        base_priority=0.75,
        latency_ms=800,
        cost_per_token=0.003,
        max_context_length=200000,
        capability_tags={"creative", "writing", "storytelling", "poetry"}
    ))

    # Executor tier - Fast execution
    registry.register(ModelNode(
        model_id="claude-haiku-4",
        display_name="Claude Haiku 4",
        tier=ModelTier.EXECUTOR,
        parent_id="claude-sonnet-code",
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.UNIVERSAL,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.6,
                TopologicalCapability.B1_REDUCTION: 0.6,
                TopologicalCapability.STRUCTURE_PRESERVE: 0.7,
                TopologicalCapability.UNIVERSAL: 0.65
            },
            optimal_betti_0_range=(1, 3),
            optimal_betti_1_range=(0, 2),
            optimal_complexity={"TRIVIAL", "SIMPLE"}
        ),
        base_priority=0.6,
        latency_ms=200,
        cost_per_token=0.00025,
        max_context_length=200000,
        capability_tags={"fast", "simple-tasks", "formatting", "extraction"}
    ))

    registry.register(ModelNode(
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        tier=ModelTier.EXECUTOR,
        parent_id="claude-sonnet-code",
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.UNIVERSAL,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.55,
                TopologicalCapability.B1_REDUCTION: 0.55,
                TopologicalCapability.STRUCTURE_PRESERVE: 0.6,
            },
            optimal_betti_0_range=(1, 3),
            optimal_betti_1_range=(0, 1),
            optimal_complexity={"TRIVIAL", "SIMPLE"}
        ),
        base_priority=0.55,
        latency_ms=150,
        cost_per_token=0.00015,
        max_context_length=128000,
        capability_tags={"fast", "simple-tasks", "classification"}
    ))

    # Verifier tier
    registry.register(ModelNode(
        model_id="verification-ensemble",
        display_name="Verification Ensemble",
        tier=ModelTier.VERIFIER,
        parent_id="claude-opus-4",
        topo_profile=TopologicalProfile(
            primary_capability=TopologicalCapability.B0_REDUCTION,
            capability_scores={
                TopologicalCapability.B0_REDUCTION: 0.8,
                TopologicalCapability.B1_REDUCTION: 0.7,
            },
            optimal_betti_0_range=(1, 5),
            optimal_betti_1_range=(0, 2),
            optimal_complexity={"SIMPLE", "MODERATE"}
        ),
        base_priority=0.7,
        latency_ms=500,
        cost_per_token=0.001,
        capability_tags={"verification", "fact-check", "consistency"}
    ))

    return registry
