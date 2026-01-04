"""
Topological Feature Extractor for Geometry-Aware Routing

Extracts persistent homology features from embeddings to characterize
the "shape" of queries/chunks for routing decisions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser not installed. Using approximate topology.")

try:
    from persim import PersistenceImager
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False


class TopologyComplexity(Enum):
    """Categorical complexity levels based on Betti numbers."""
    TRIVIAL = 0      # beta_0=1, beta_1=0 - single connected, no loops
    SIMPLE = 1       # beta_0<=2, beta_1<=1 - few components, minimal entanglement
    MODERATE = 2     # beta_0<=4, beta_1<=3 - multiple clusters, some loops
    COMPLEX = 3      # beta_0>4 or beta_1>3 - highly fragmented or entangled
    CHAOTIC = 4      # High persistence variance - unstable structure


@dataclass
class PersistenceDiagram:
    """Container for persistence diagram data."""
    h0: np.ndarray  # 0-dimensional features (connected components)
    h1: np.ndarray  # 1-dimensional features (loops/cycles)
    h2: Optional[np.ndarray] = None  # 2-dimensional features (voids)

    @property
    def betti_0(self) -> int:
        """Number of connected components (finite death times)."""
        if len(self.h0) == 0:
            return 0
        # Count features with finite death (exclude infinite persistence)
        finite = self.h0[self.h0[:, 1] != np.inf]
        return len(finite) + 1  # +1 for the essential component

    @property
    def betti_1(self) -> int:
        """Number of loops/cycles."""
        return len(self.h1) if len(self.h1) > 0 else 0

    @property
    def betti_2(self) -> int:
        """Number of voids (if computed)."""
        if self.h2 is None or len(self.h2) == 0:
            return 0
        return len(self.h2)

    @property
    def total_persistence(self) -> float:
        """Sum of all persistence values (death - birth)."""
        total = 0.0
        for dgm in [self.h0, self.h1]:
            if len(dgm) > 0:
                finite = dgm[dgm[:, 1] != np.inf]
                if len(finite) > 0:
                    total += np.sum(finite[:, 1] - finite[:, 0])
        return total

    @property
    def max_persistence(self) -> float:
        """Maximum persistence value across all features."""
        max_p = 0.0
        for dgm in [self.h0, self.h1]:
            if len(dgm) > 0:
                finite = dgm[dgm[:, 1] != np.inf]
                if len(finite) > 0:
                    max_p = max(max_p, np.max(finite[:, 1] - finite[:, 0]))
        return max_p


@dataclass
class TopologicalSignature:
    """Complete topological signature for routing decisions."""
    persistence_diagram: PersistenceDiagram
    persistence_image: np.ndarray  # Vectorized representation
    betti_profile: Tuple[int, int, int]  # (beta_0, beta_1, beta_2)
    complexity: TopologyComplexity
    stability_score: float  # 0-1, higher = more stable structure

    def to_vector(self) -> np.ndarray:
        """Flatten signature to routing-compatible vector."""
        betti_vec = np.array(self.betti_profile, dtype=np.float32)
        meta_vec = np.array([
            self.complexity.value / 4.0,  # Normalized complexity
            self.stability_score,
            self.persistence_diagram.total_persistence,
            self.persistence_diagram.max_persistence
        ], dtype=np.float32)
        return np.concatenate([
            self.persistence_image.flatten(),
            betti_vec,
            meta_vec
        ])


class TopologicalFeatureExtractor:
    """
    Extracts topological features from embeddings using persistent homology.

    The core insight: problems with similar topological signatures should
    route to similar experts. This extractor characterizes the "shape"
    of query embeddings to enable geometry-aware routing.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        persistence_image_size: Tuple[int, int] = (20, 20),
        subsample_size: Optional[int] = 500,
        use_approximate: bool = False
    ):
        """
        Args:
            max_dimension: Maximum homology dimension (0=components, 1=loops, 2=voids)
            persistence_image_size: Resolution of vectorized persistence image
            subsample_size: Max points for efficiency (None = use all)
            use_approximate: Force approximate methods even if ripser available
        """
        self.max_dimension = max_dimension
        self.pi_size = persistence_image_size
        self.subsample_size = subsample_size
        self.use_approximate = use_approximate or not RIPSER_AVAILABLE

        # Initialize persistence imager if available
        self._imager = None
        if PERSIM_AVAILABLE:
            self._imager = PersistenceImager(
                pixels=persistence_image_size,
                spread=0.1
            )

    def extract(self, embeddings: np.ndarray) -> TopologicalSignature:
        """
        Extract topological signature from embedding matrix.

        Args:
            embeddings: (n_points, embedding_dim) array

        Returns:
            TopologicalSignature with all routing-relevant features
        """
        # Subsample if needed for efficiency
        if self.subsample_size and len(embeddings) > self.subsample_size:
            indices = np.random.choice(
                len(embeddings),
                self.subsample_size,
                replace=False
            )
            embeddings = embeddings[indices]

        # Compute persistence diagram
        if self.use_approximate:
            pd = self._compute_approximate_topology(embeddings)
        else:
            pd = self._compute_ripser_topology(embeddings)

        # Vectorize to persistence image
        pi = self._compute_persistence_image(pd)

        # Compute derived features
        betti = (pd.betti_0, pd.betti_1, pd.betti_2)
        complexity = self._classify_complexity(pd)
        stability = self._compute_stability(pd)

        return TopologicalSignature(
            persistence_diagram=pd,
            persistence_image=pi,
            betti_profile=betti,
            complexity=complexity,
            stability_score=stability
        )

    def _compute_ripser_topology(self, embeddings: np.ndarray) -> PersistenceDiagram:
        """Compute exact persistent homology using ripser."""
        result = ripser(embeddings, maxdim=self.max_dimension)
        dgms = result['dgms']

        return PersistenceDiagram(
            h0=dgms[0],
            h1=dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2),
            h2=dgms[2] if len(dgms) > 2 else None
        )

    def _compute_approximate_topology(self, embeddings: np.ndarray) -> PersistenceDiagram:
        """
        Approximate topological features without ripser.
        Uses spectral clustering + neighborhood analysis.
        """
        n_points = len(embeddings)

        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings))

        # Approximate beta_0 via connected components at various thresholds
        # Use binary search to find threshold where components stabilize
        thresholds = np.percentile(distances[distances > 0], [10, 25, 50, 75, 90])
        component_counts = []

        for thresh in thresholds:
            adj = (distances < thresh).astype(int)
            np.fill_diagonal(adj, 0)
            n_components = self._count_components(adj)
            component_counts.append(n_components)

        # Approximate beta_1 via cycles in k-NN graph
        k = min(10, n_points - 1)
        knn_adj = np.zeros((n_points, n_points))
        for i in range(n_points):
            nearest = np.argsort(distances[i])[1:k+1]
            knn_adj[i, nearest] = 1

        # Simple cycle detection via eigenvalue analysis of graph Laplacian
        degree = np.sum(knn_adj, axis=1)
        laplacian = np.diag(degree) - knn_adj
        eigenvalues = np.linalg.eigvalsh(laplacian)

        # Count near-zero eigenvalues (components) and small eigenvalues (cycles)
        approx_b0 = np.sum(eigenvalues < 1e-6)
        approx_b1 = np.sum((eigenvalues > 1e-6) & (eigenvalues < 0.1))

        # Create synthetic persistence diagram
        h0_births = np.zeros(max(1, approx_b0))
        h0_deaths = np.random.uniform(0.1, 0.5, max(1, approx_b0))
        h0 = np.column_stack([h0_births, h0_deaths])

        h1_births = np.random.uniform(0.1, 0.3, max(0, approx_b1))
        h1_deaths = np.random.uniform(0.4, 0.8, max(0, approx_b1)) if approx_b1 > 0 else np.array([])
        h1 = np.column_stack([h1_births, h1_deaths]) if approx_b1 > 0 else np.array([]).reshape(0, 2)

        return PersistenceDiagram(h0=h0, h1=h1, h2=None)

    def _count_components(self, adj: np.ndarray) -> int:
        """Count connected components via BFS."""
        n = len(adj)
        visited = np.zeros(n, dtype=bool)
        components = 0

        for start in range(n):
            if not visited[start]:
                components += 1
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if not visited[node]:
                        visited[node] = True
                        neighbors = np.where(adj[node] > 0)[0]
                        queue.extend(neighbors[~visited[neighbors]])

        return components

    def _compute_persistence_image(self, pd: PersistenceDiagram) -> np.ndarray:
        """Convert persistence diagram to fixed-size image vector."""
        if self._imager is not None and len(pd.h1) > 0:
            try:
                return self._imager.transform([pd.h1])[0]
            except Exception:
                pass

        # Fallback: simple histogram-based vectorization
        pi = np.zeros(self.pi_size)

        for dgm in [pd.h0, pd.h1]:
            if len(dgm) == 0:
                continue
            finite = dgm[dgm[:, 1] != np.inf]
            if len(finite) == 0:
                continue

            births = finite[:, 0]
            deaths = finite[:, 1]
            persistence = deaths - births

            # 2D histogram: birth vs persistence
            b_bins = np.linspace(0, np.max(deaths) + 0.1, self.pi_size[0] + 1)
            p_bins = np.linspace(0, np.max(persistence) + 0.1, self.pi_size[1] + 1)

            hist, _, _ = np.histogram2d(births, persistence, bins=[b_bins, p_bins])
            pi += hist

        # Normalize
        if pi.max() > 0:
            pi = pi / pi.max()

        return pi

    def _classify_complexity(self, pd: PersistenceDiagram) -> TopologyComplexity:
        """Classify topological complexity from Betti numbers."""
        b0, b1 = pd.betti_0, pd.betti_1

        if b0 == 1 and b1 == 0:
            return TopologyComplexity.TRIVIAL
        elif b0 <= 2 and b1 <= 1:
            return TopologyComplexity.SIMPLE
        elif b0 <= 4 and b1 <= 3:
            return TopologyComplexity.MODERATE
        else:
            # Check for chaotic structure via persistence variance
            if len(pd.h1) > 0:
                finite = pd.h1[pd.h1[:, 1] != np.inf]
                if len(finite) > 1:
                    persistence = finite[:, 1] - finite[:, 0]
                    if np.std(persistence) > 0.3 * np.mean(persistence):
                        return TopologyComplexity.CHAOTIC
            return TopologyComplexity.COMPLEX

    def _compute_stability(self, pd: PersistenceDiagram) -> float:
        """
        Compute stability score based on persistence distribution.
        High stability = features persist consistently (robust structure).
        Low stability = features die quickly (fragile/noisy structure).
        """
        all_persistence = []

        for dgm in [pd.h0, pd.h1]:
            if len(dgm) > 0:
                finite = dgm[dgm[:, 1] != np.inf]
                if len(finite) > 0:
                    all_persistence.extend(finite[:, 1] - finite[:, 0])

        if not all_persistence:
            return 0.5  # Neutral stability for trivial topology

        persistence = np.array(all_persistence)

        # Stability = ratio of long-lived features to total features
        # Using median persistence as threshold
        median_p = np.median(persistence)
        long_lived = np.sum(persistence > median_p)
        stability = long_lived / len(persistence)

        # Also penalize high variance
        cv = np.std(persistence) / (np.mean(persistence) + 1e-8)
        variance_penalty = np.exp(-cv)

        return float(np.clip(stability * variance_penalty, 0, 1))


def compute_bottleneck_distance(
    sig1: TopologicalSignature,
    sig2: TopologicalSignature
) -> float:
    """
    Compute bottleneck distance between two topological signatures.
    This is the key metric for determining topological similarity.
    """
    if PERSIM_AVAILABLE:
        from persim import bottleneck
        try:
            d0 = bottleneck(sig1.persistence_diagram.h0, sig2.persistence_diagram.h0)
            d1 = bottleneck(sig1.persistence_diagram.h1, sig2.persistence_diagram.h1)
            return max(d0, d1)
        except Exception:
            pass

    # Fallback: L2 distance on persistence images
    return float(np.linalg.norm(
        sig1.persistence_image.flatten() - sig2.persistence_image.flatten()
    ))


def compute_wasserstein_distance(
    sig1: TopologicalSignature,
    sig2: TopologicalSignature,
    p: int = 2
) -> float:
    """
    Compute Wasserstein distance between persistence diagrams.
    More sensitive to small differences than bottleneck distance.
    """
    if PERSIM_AVAILABLE:
        from persim import wasserstein
        try:
            d0 = wasserstein(sig1.persistence_diagram.h0, sig2.persistence_diagram.h0, p=p)
            d1 = wasserstein(sig1.persistence_diagram.h1, sig2.persistence_diagram.h1, p=p)
            return d0 + d1
        except Exception:
            pass

    # Fallback: Weighted L2 on persistence images
    return float(np.sqrt(np.sum(
        (sig1.persistence_image - sig2.persistence_image) ** 2
    )))
