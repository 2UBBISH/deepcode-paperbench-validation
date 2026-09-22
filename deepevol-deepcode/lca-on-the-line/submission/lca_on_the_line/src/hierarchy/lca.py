"""Pairwise LCA (Lowest Common Ancestor) distance definitions.

Paper reference
---------------
Section 2 / Section D.2.1::

    D_LCA(y', y) := f(y) - f(N_LCA(y, y'))

with two choices of the node score ``f(.)``:

* tree depth ``P(x)`` (Section D.2.1, Eq. 2)::

      D_LCA^P(y', y) := (P(y) - P(N_LCA(y', y))) + (P(y') - P(N_LCA(y', y)))

  The second term is appended "to counter tree imbalance".

* information content ``I(x)`` (Section D.2.1, Eq. 3)::

      D_LCA^I(y', y) := I(y) - I(N_LCA(y', y))

  where the information of a node is computed following (Valmadre, 2022) by
  applying a uniform distribution ``p`` to all leaf nodes and recursively
  summing the scores of the descendants, and ``I(node) := -log2(p)``.

The paper adopts ``D_LCA^I`` for LCA measurements and ``D_LCA^P`` for the
linear probing experiments.

Sanity requirement (Addendum, "Sanity checking the LCA distance matrix"): the
pairwise matrix must store *distances*, not similarities, hence a diagonal of
zeros.  ``D_LCA(c, c) = 0`` holds for both variants because
``N_LCA(c, c) = c``.

This module exposes thin functional wrappers around
:class:`src.hierarchy.info_content.HierarchyScorer`, plus helpers that operate
directly on the pre-computed per-class scores.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from .info_content import (
    DepthScore,
    HierarchyScorer,
    InformationScore,
    class_depth_score,
    class_information_content,
    compute_depths,
    compute_information_content,
    lca_distance_depth,
    lca_distance_information,
)
from .wordnet import WordNetHierarchy

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DISTANCE_MODE",
    "lca_distance",
    "lca_distance_from_scores",
    "pairwise_lca_distance",
    "pairwise_lca_matrix",
    "lca_distance_matrix_from_scores",
    "reverse_lca_matrix",
    "LcaDistance",
]

#: The distance used for "LCA measurements" in the main paper (Section D.2.1).
DEFAULT_DISTANCE_MODE = "information"


def _resolve_scorer(
    hierarchy: WordNetHierarchy, scorer: Optional[HierarchyScorer] = None
) -> HierarchyScorer:
    """Return the provided scorer or build one for ``hierarchy``."""
    if scorer is None:
        return HierarchyScorer(hierarchy)
    return scorer


def lca_distance(
    y_prime: int,
    y: int,
    hierarchy: WordNetHierarchy,
    mode: str = DEFAULT_DISTANCE_MODE,
    scorer: Optional[HierarchyScorer] = None,
) -> float:
    """LCA distance ``D_LCA(y', y)`` between a prediction and a target class.

    Args:
        y_prime: prediction class index.
        y: ground-truth class index.
        hierarchy: the class hierarchy.
        mode: ``"information"`` (Eq. 3, default, used for LCA measurements) or
            ``"depth"`` (Eq. 2, used for linear probing).
        scorer: optional pre-built :class:`HierarchyScorer` for speed.

    Returns:
        A non-negative distance; ``0.0`` when ``y == y_prime``.
    """
    scorer = _resolve_scorer(hierarchy, scorer)
    if mode in ("information", "info", "information_content", "I"):
        return scorer.lca_distance_information(y_prime, y)
    if mode in ("depth", "P"):
        return scorer.lca_distance_depth(y_prime, y)
    raise ValueError(
        f"Unknown LCA distance mode {mode!r}; expected 'information' or 'depth'."
    )


def lca_distance_from_scores(
    y_prime: int,
    y: int,
    lca_class: int,
    class_scores: Sequence[float],
    mode: str = DEFAULT_DISTANCE_MODE,
) -> float:
    """Compute the LCA distance given already-computed class scores.

    ``D_LCA^I(y', y) = f(y) - f(N_LCA)`` and
    ``D_LCA^P(y', y) = (f(y) - f(N_LCA)) + (f(y') - f(N_LCA))``.

    Args:
        y_prime: prediction class index (used by the depth variant only).
        y: ground-truth class index.
        lca_class: class index of the lowest common ancestor node.  When no
            class node is an ancestor (the virtual root case) pass the index of
            an arbitrary class so that both terms become ``f(y) - f(root)``.
        class_scores: per-class node scores ``f(.)`` indexed by class id.
        mode: ``"information"`` or ``"depth"``.
    """
    f_y = float(class_scores[y])
    f_lca = float(class_scores[lca_class])
    if mode in ("information", "info", "information_content", "I"):
        return max(0.0, f_y - f_lca)
    if mode in ("depth", "P"):
        f_yp = float(class_scores[y_prime])
        return max(0.0, (f_y - f_lca) + (f_yp - f_lca))
    raise ValueError(
        f"Unknown LCA distance mode {mode!r}; expected 'information' or 'depth'."
    )


def pairwise_lca_distance(
    y_prime: int,
    y: int,
    hierarchy: WordNetHierarchy,
    mode: str = DEFAULT_DISTANCE_MODE,
) -> float:
    """Convenience wrapper building a scorer on the fly (no caching)."""
    return lca_distance(y_prime, y, hierarchy, mode=mode)


class LcaDistance:
    """Stateful pairwise LCA distance calculator over a class hierarchy.

    Pre-computes the per-class node scores once, so computing the full
    ``n x n`` matrix is a sequence of O(1) lookups.

    Example
    -------
    >>> dist = LcaDistance(hierarchy)               # information content
    >>> dist(120, 120)                              # 0.0
    >>> dist(120, 281)                              # some positive distance
    >>> M = dist.matrix()                           # 1000 x 1000 list of lists
    """

    def __init__(
        self,
        hierarchy: WordNetHierarchy,
        mode: str = DEFAULT_DISTANCE_MODE,
        num_leaves: Optional[int] = None,
        base: float = 2.0,
    ) -> None:
        if mode in ("information", "info", "information_content", "I"):
            self.mode = "information"
        elif mode in ("depth", "P"):
            self.mode = "depth"
        else:
            raise ValueError(
                f"Unknown LCA distance mode {mode!r}; expected 'information' or 'depth'."
            )
        self.hierarchy = hierarchy
        self.num_classes = hierarchy.num_classes
        self.scorer = HierarchyScorer(
            hierarchy, num_leaves=num_leaves, base=base
        )
        if self.mode == "information":
            self.class_scores: List[float] = [
                self.scorer.info_of_class(c) for c in range(self.num_classes)
            ]
        else:
            self.class_scores = [
                self.scorer.depth_of_class(c) for c in range(self.num_classes)
            ]
        # Per-class LCA class index cache (lazily filled); this is the dominant
        # cost when building the full matrix because the LCA lookup walks the
        # ancestor lists.
        self._lca_class_cache: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------ #
    # core distance
    # ------------------------------------------------------------------ #
    def lca_class_index(self, y: int, y_prime: int) -> int:
        """Index of the class node used as ``N_LCA`` for the pair."""
        key = (y, y_prime) if y <= y_prime else (y_prime, y)
        cached = self._lca_class_cache.get(key)
        if cached is not None:
            return cached
        node = self.scorer.lca_node(y, y_prime)
        # ``lca_of_class_node`` style resolution: the hierarchy returns the
        # synset node; map it back to a class index when possible, else fall
        # back to the ground-truth class (giving distance 0 for the information
        # variant against the root).
        lca_class = self.hierarchy.class_for_node(node) if hasattr(
            self.hierarchy, "class_for_node"
        ) else None
        if lca_class is None:
            lca_class = y
        self._lca_class_cache[key] = int(lca_class)
        return int(lca_class)

    def __call__(self, y_prime: int, y: int) -> float:
        """``D_LCA(y', y)``."""
        return lca_distance_from_scores(
            y_prime,
            y,
            self.lca_class_index(y, y_prime),
            self.class_scores,
            mode=self.mode,
        )

    def distance(self, y_prime: int, y: int) -> float:
        """Alias of :meth:`__call__`."""
        return self(y_prime, y)

    def distance_by_node(
        self,
        y_prime: int,
        y: int,
        root_fallback: bool = True,
    ) -> float:
        """Distance computed directly from node scores (no class-index mapping).

        Uses ``f`` evaluated at the *node* returned by the hierarchy, which is
        slightly more faithful when the LCA is an internal node that is not
        itself a class.  ``root_fallback`` clamps the result to be non-negative.
        """
        node_lca = self.scorer.lca_node(y, y_prime)
        if self.mode == "information":
            f_y = self.scorer.info_of_class(y)
            f_lca = self.scorer.info_of_node(node_lca)
            d = f_y - f_lca
        else:
            f_y = self.scorer.depth_of_class(y)
            f_yp = self.scorer.depth_of_class(y_prime)
            f_lca = self.scorer.depth_of_node(node_lca)
            d = (f_y - f_lca) + (f_yp - f_lca)
        return max(0.0, d) if root_fallback else d

    # ------------------------------------------------------------------ #
    # matrices
    # ------------------------------------------------------------------ #
    def row(self, y: int, class_indices: Optional[Sequence[int]] = None) -> List[float]:
        """``D_LCA(:, y)`` for the requested columns (default: all classes)."""
        cols = class_indices if class_indices is not None else range(self.num_classes)
        return [self(int(c), int(y)) for c in cols]

    def matrix(
        self,
        class_indices: Optional[Sequence[int]] = None,
    ) -> List[List[float]]:
        """Full ``n x n`` LCA distance matrix ``M[i, k] = D_LCA(i, k)``.

        Row index ``i`` is the *prediction* class and column index ``k`` is the
        ground-truth class, matching ``M[targets]`` indexing in Algorithm 1.
        """
        idx = (
            [int(c) for c in class_indices]
            if class_indices is not None
            else list(range(self.num_classes))
        )
        return [[self(i, k) for k in idx] for i in idx]

    def asymmetric_matrix(self) -> List[List[float]]:
        """Variant of :meth:`matrix` using node-level ``f`` evaluations."""
        return [
            [self.distance_by_node(i, k) for k in range(self.num_classes)]
            for i in range(self.num_classes)
        ]


def pairwise_lca_matrix(
    hierarchy: WordNetHierarchy,
    class_indices: Optional[Sequence[int]] = None,
    mode: str = DEFAULT_DISTANCE_MODE,
) -> List[List[float]]:
    """Build the ``n x n`` pairwise LCA distance matrix for ``hierarchy``."""
    return LcaDistance(hierarchy, mode=mode).matrix(class_indices)


def lca_distance_matrix_from_scores(
    class_scores: Sequence[float],
    lca_class_indices: Dict[Tuple[int, int], int],
    num_classes: Optional[int] = None,
    mode: str = DEFAULT_DISTANCE_MODE,
) -> List[List[float]]:
    """Assemble a distance matrix from pre-computed scores + LCA lookups.

    ``lca_class_indices`` maps (y, y') (unordered) to the class index of the
    lowest common ancestor.
    """
    n = int(num_classes) if num_classes is not None else len(class_scores)
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):  # prediction
        for k in range(n):  # ground truth
            if i == k:
                continue
            lca = lca_class_indices.get((k, i), lca_class_indices.get((i, k), k))
            matrix[i][k] = lca_distance_from_scores(
                i, k, lca, class_scores, mode=mode
            )
    return matrix


def reverse_lca_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """``reverse_LCA_matrix = 1 - LCA_matrix`` (Algorithm 1).

    With a distance matrix whose diagonal is zero, the reversed matrix has a
    diagonal of ones, as required by the addendum's sanity check.
    """
    return [[1.0 - float(v) for v in row] for row in matrix]


def sanity_check_matrix(
    matrix: Sequence[Sequence[float]], atol: float = 1e-5, check_symmetry: bool = True
) -> None:
    """Raise ``AssertionError`` if ``matrix`` is not a valid LCA distance matrix.

    Checks the addendum's requirements: zero diagonal and (optionally) symmetry.
    """
    n = len(matrix)
    for i in range(n):
        assert len(matrix[i]) == n, "LCA matrix must be square."
        assert abs(float(matrix[i][i])) <= atol, (
            f"LCA matrix must have a zero diagonal, got {matrix[i][i]} at ({i}, {i})."
        )
    if check_symmetry:
        for i in range(n):
            for k in range(i + 1, n):
                assert abs(float(matrix[i][k]) - float(matrix[k][i])) <= atol, (
                    "LCA matrix must be symmetric."
                )
    for row in reverse_lca_matrix(matrix):
        pass
