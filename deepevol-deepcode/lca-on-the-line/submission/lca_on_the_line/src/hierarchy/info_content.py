"""Information-content and tree-depth scores for a class hierarchy.

Paper references
----------------
Section 2 ("LCA Distance Measures Misprediction Severity")

    D_LCA(y', y) := f(y) - f(N_LCA(y, y'))

where ``f(.)`` is a function of a node, "such as the tree depth or entropy".
"We use the information content as described in (Valmadre, 2022)."

Section D.2.1 ("LCA DISTANCE")

    D_LCA^P(y', y) := (P(y)  - P(N_LCA(y', y)))
                     + (P(y') - P(N_LCA(y', y)))

      "where we also append (P(y') - P(N_LCA(y', y))) to counter tree imbalance."

    D_LCA^I(y', y) := I(y) - I(N_LCA(y', y))

      "we apply a uniform distribution p to all leaf nodes in the tree that
       indicate a class in the classification task.  The probability of each
       intermediate node in the tree is calculated by recursively summing the
       scores of its descendants.  Then, the information of each node is
       calculated as I(node) := -log2(p)."

With a uniform distribution over the ``|L|`` class leaves this is equivalent to

    I(node) = log2(|L|) - log2(|L(node)|)

where ``|L(node)|`` is the number of class leaves in the subtree rooted at the
node (compare the addendum: ``I(y) = log |L| - log |L(y)|``).

The module exposes plain score functions (``DepthScore``, ``InformationScore``)
as well as a convenience wrapper ``HierarchyScorer`` that caches the scores of
every node of a :class:`~src.hierarchy.wordnet.WordNetHierarchy`.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

from .wordnet import IMAGENET_NUM_CLASSES, WordNetHierarchy

logger = logging.getLogger(__name__)

# The paper writes log2 (D.2.1) while the referenced Valmadre (2022) notes use
# the natural log.  Both are monotone transforms of each other, so they produce
# identical rankings, correlations and distances up to a constant factor.  We
# follow the paper's text and use base 2.
_LOG_BASE = 2.0


def _log(x: float, base: float = _LOG_BASE) -> float:
    return math.log(x) / math.log(base)


class NodeScorer:
    """Base class: assigns a real-valued score to every node of a hierarchy."""

    name = "node-score"

    def __init__(self, hierarchy: WordNetHierarchy):
        self.hierarchy = hierarchy

    def score(self, node: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class DepthScore(NodeScorer):
    """``P(x)`` from D.2.1: the depth of node ``x`` inside tree ``T``.

    The virtual root that :class:`WordNetHierarchy` adds when a forest has
    several top-level synsets sits at depth 0; the ImageNet/WordNet root then
    gets depth 1.
    """

    name = "depth"

    def __init__(self, hierarchy: WordNetHierarchy):
        super().__init__(hierarchy)
        self._cache: Dict[str, float] = {}

    def score(self, node: str) -> float:
        if node not in self._cache:
            self._cache[node] = float(self.hierarchy.depth(node))
        return self._cache[node]

    def max_score(self) -> float:
        return max(self.score(n) for n in self.hierarchy.parents)

    def min_score(self) -> float:
        return min(self.score(n) for n in self.hierarchy.parents)


class InformationScore(NodeScorer):
    """``I(x) = -log2 p(x)`` with a uniform distribution over class leaves.

    ``p`` is uniform over the leaf nodes that indicate a class of the
    classification task (the 1000 ImageNet classes); intermediate nodes get the
    recursive sum of their descendants' probabilities, which for a uniform leaf
    distribution equals ``|L(node)| / |L|``.
    """

    name = "information"

    def __init__(
        self,
        hierarchy: WordNetHierarchy,
        num_leaves: Optional[int] = None,
        base: float = _LOG_BASE,
    ):
        super().__init__(hierarchy)
        self.base = base
        # ``leaf_count`` counts the class leaves contained in a subtree, which
        # is exactly |L(y)|.  The total probability mass of the tree is 1.
        self.num_leaves = int(num_leaves or hierarchy.leaf_count(hierarchy.root))
        self._cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # probability / information
    # ------------------------------------------------------------------
    def probability(self, node: str) -> float:
        """``p(node)`` = |L(node)| / |L| (recursive sum of descendant scores)."""
        if self.num_leaves <= 0:
            return 0.0
        return self.hierarchy.leaf_count(node) / float(self.num_leaves)

    def score(self, node: str) -> float:
        """``I(node) = -log p(node) = log|L| - log|L(node)|``."""
        if node not in self._cache:
            p = self.probability(node)
            self._cache[node] = -_log(max(p, 1e-12), self.base)
        return self._cache[node]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def max_score(self) -> float:
        """Information of the root (the largest score in the tree)."""
        return self.score(self.hierarchy.root)

    def min_score(self) -> float:
        """Information of a class leaf: ``p = 1/|L|`` -> ``log|L|``."""
        if self.num_leaves <= 0:
            return 0.0
        return _log(float(self.num_leaves), self.base)

    def normalized(self, node: str) -> float:
        """Map ``I`` onto ``[0, 1]`` (0 = root, 1 = class leaf)."""
        hi = self.max_score()
        if hi <= 0:
            return 0.0
        return self.score(node) / hi

    def class_information(self, class_index: int) -> float:
        return self.score(self.hierarchy.leaf_for_class(class_index))


class HierarchyScorer:
    """Caches depth and information scores for every node of a hierarchy.

    Usage
    -----
    >>> from src.hierarchy.wordnet import build_two_pair_hierarchy
    >>> scorer = HierarchyScorer(build_two_pair_hierarchy())
    >>> scorer.lca_distance_information(1, 0) > 0     # doctest: +SKIP
    True
    """

    def __init__(
        self,
        hierarchy: WordNetHierarchy,
        num_leaves: Optional[int] = None,
        base: float = _LOG_BASE,
    ):
        self.hierarchy = hierarchy
        self.depth = DepthScore(hierarchy)
        self.information = InformationScore(hierarchy, num_leaves=num_leaves, base=base)

        self._depth_cache: Dict[str, float] = {}
        self._info_cache: Dict[str, float] = {}
        self._class_depths: Optional[List[float]] = None
        self._class_infos: Optional[List[float]] = None

        self._prefill()

    # ------------------------------------------------------------------
    def _prefill(self) -> None:
        for node in self.hierarchy.parents:
            self._depth_cache[node] = self.depth.score(node)
            self._info_cache[node] = self.information.score(node)
        self._class_depths = [
            self._depth_cache[self.hierarchy.leaf_for_class(c)]
            for c in range(self.hierarchy.num_classes)
        ]
        self._class_infos = [
            self._info_cache[self.hierarchy.leaf_for_class(c)]
            for c in range(self.hierarchy.num_classes)
        ]

    # ------------------------------------------------------------------
    # node accessors
    # ------------------------------------------------------------------
    def depth_of_node(self, node: str) -> float:
        return self._depth_cache.get(node, self.depth.score(node))

    def info_of_node(self, node: str) -> float:
        return self._info_cache.get(node, self.information.score(node))

    def depth_of_class(self, class_index: int) -> float:
        return self._class_depths[int(class_index)]

    def info_of_class(self, class_index: int) -> float:
        return self._class_infos[int(class_index)]

    def lca_node(self, y: int, y_prime: int) -> str:
        return self.hierarchy.lca_of_classes(int(y), int(y_prime))

    # ------------------------------------------------------------------
    # pairwise distances (D.2.1)
    # ------------------------------------------------------------------
    def lca_distance_depth(self, y_prime: int, y: int) -> float:
        """``D_LCA^P(y', y)`` -- used for the linear-probing experiments."""
        node = self.lca_node(y, y_prime)
        p_lca = self.depth_of_node(node)
        return (self.depth_of_class(y) - p_lca) + (self.depth_of_class(y_prime) - p_lca)

    def lca_distance_information(self, y_prime: int, y: int) -> float:
        """``D_LCA^I(y', y) = I(y) - I(N_LCA(y, y'))`` -- used for LCA metrics."""
        node = self.lca_node(y, y_prime)
        return self.info_of_class(y) - self.info_of_node(node)

    # Short alias matching the metric used throughout the paper.
    def lca_distance(self, y_prime: int, y: int) -> float:
        return self.lca_distance_information(y_prime, y)

    # ------------------------------------------------------------------
    def pairwise_depth_matrix(self) -> List[List[float]]:
        n = self.hierarchy.num_classes
        return [[self.lca_distance_depth(j, i) for j in range(n)] for i in range(n)]

    def pairwise_information_matrix(self) -> List[List[float]]:
        n = self.hierarchy.num_classes
        return [
            [self.lca_distance_information(j, i) for j in range(n)] for i in range(n)
        ]


# ----------------------------------------------------------------------
# Functional helpers
# ----------------------------------------------------------------------
def compute_information_content(
    hierarchy: WordNetHierarchy,
    num_leaves: Optional[int] = None,
    base: float = _LOG_BASE,
) -> Dict[str, float]:
    """Return ``{node: I(node)}`` for every node of ``hierarchy``."""
    scorer = InformationScore(hierarchy, num_leaves=num_leaves, base=base)
    return {node: scorer.score(node) for node in hierarchy.parents}


def compute_depths(hierarchy: WordNetHierarchy) -> Dict[str, float]:
    """Return ``{node: P(node)}`` for every node of ``hierarchy``."""
    scorer = DepthScore(hierarchy)
    return {node: scorer.score(node) for node in hierarchy.parents}


def class_information_content(
    hierarchy: WordNetHierarchy,
    num_leaves: Optional[int] = None,
    class_indices: Optional[Sequence[int]] = None,
) -> List[float]:
    """Information content of the *class leaves* (used as ``f(y)`` in Eq. 1)."""
    scorer = InformationScore(hierarchy, num_leaves=num_leaves)
    indices = range(hierarchy.num_classes) if class_indices is None else class_indices
    return [scorer.class_information(int(c)) for c in indices]


def class_depth_score(
    hierarchy: WordNetHierarchy, class_indices: Optional[Sequence[int]] = None
) -> List[float]:
    """Depth ``P(y)`` of the class leaves."""
    scorer = DepthScore(hierarchy)
    indices = range(hierarchy.num_classes) if class_indices is None else class_indices
    return [scorer.score(hierarchy.leaf_for_class(int(c))) for c in indices]


def lca_distance_information(
    hierarchy: WordNetHierarchy,
    y_prime: int,
    y: int,
    scorer: Optional[HierarchyScorer] = None,
) -> float:
    scorer = scorer or HierarchyScorer(hierarchy)
    return scorer.lca_distance_information(int(y_prime), int(y))


def lca_distance_depth(
    hierarchy: WordNetHierarchy,
    y_prime: int,
    y: int,
    scorer: Optional[HierarchyScorer] = None,
) -> float:
    scorer = scorer or HierarchyScorer(hierarchy)
    return scorer.lca_distance_depth(int(y_prime), int(y))


__all__ = [
    "NodeScorer",
    "DepthScore",
    "InformationScore",
    "HierarchyScorer",
    "compute_information_content",
    "compute_depths",
    "class_information_content",
    "class_depth_score",
    "lca_distance_information",
    "lca_distance_depth",
    "IMAGENET_NUM_CLASSES",
]
