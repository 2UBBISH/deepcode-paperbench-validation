"""Latent class-hierarchy construction via K-means clustering on image features.

Implements the latent taxonomy of the paper:

* §4.3.1 "Inferring Class Taxonomy from a Pretrained Model via K-Means Clustering"
* §E.1   "K-mean Clustering for Latent Class Hierarchy Construction"

Pipeline (paper §E.1):

1. take a pretrained source model ``M`` and in-distribution data ``(X, Y)`` with ``k`` classes;
2. extract in-distribution features ``M(X)`` and average them per class -> ``k``  average class
   features;
3. run a 9-layer hierarchical clustering on the average class features: for ``i = 1..9`` run
   K-means with ``2^i`` centres *independently per level* (``2^9 < 1000`` for ImageNet);
4. for every pair of classes find the cluster level at which both classes live in the same
   cluster; that level's height is the pairwise LCA height (the K-means analogue of the WordNet
   LCA node);
5. "By definition, all classes share a base cluster level of 10" (paper §E.1), i.e. pairs that
   never share a cluster get the base height ``10`` -- which matches
   ``log2(1000) ~= 9.97 ~= 10``, the information content of a class leaf.

Two equivalent views of the resulting hierarchy are exposed:

* a *)similarity* view (:meth:`LatentHierarchy.similarity_matrix`) whose maximum is the base
  level ``10`` on the diagonal and which is the raw matrix that
  :func:`src.hierarchy.lca_matrix.process_lca_matrix` expects with
  ``latent_hierarchy=True`` (the Addendum inverts latent matrices with ``max(M) - M``, which
  turns this similarity into the LCA distance);
* a *)distance* view (:meth:`LatentHierarchy.distance_matrix`,
  ``d(i, j) = base_level - share_level(i, j)``) with a zero diagonal, plus a
  :class:`~src.hierarchy.wordnet.WordNetHierarchy`-compatible tree
  (:meth:`LatentHierarchy.to_wordnet_hierarchy`) so that the standard information-content and
  depth scorers can be reused unchanged.  Both views are monotonically equivalent (deeper common
  cluster -> smaller distance), so downstream correlations are unaffected by the choice.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .wordnet import IMAGENET_NUM_CLASSES, WordNetHierarchy

logger = logging.getLogger(__name__)

try:  # numpy is needed for the clustering itself; keep the import soft for metadata-only use.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a hard requirement for the experiments
    _np = None  # type: ignore

__all__ = [
    "DEFAULT_MAX_LEVEL",
    "DEFAULT_BASE_LEVEL",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_KMEANS_SEED",
    "DEFAULT_N_INIT",
    "LATENT_NODE_PREFIX",
    "CLASS_NODE_PREFIX",
    "ClassFeatureAccumulator",
    "LatentHierarchy",
    "class_mean_features",
    "extract_class_features",
    "level_cluster_count",
    "kmeans_levels",
    "numpy_kmeans",
    "fit_kmeans",
    "build_latent_hierarchy",
    "latent_hierarchy_from_features",
    "latent_hierarchy_from_class_features",
    "latent_hierarchies_from_sources",
    "extract_latent_hierarchy_from_model",
    "latent_lca_matrix",
    "latent_distance_matrix",
    "synthetic_latent_hierarchy",
]


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
DEFAULT_MAX_LEVEL: int = 9
"""Number of K-means levels (``2^i`` centres for ``i = 1..9``, §4.3.1 / §E.1)."""

DEFAULT_BASE_LEVEL: int = 10
"""Base cluster level shared by all classes (§E.1); ``2^9 < 1000 < 2^10``."""

DEFAULT_NUM_CLASSES: int = IMAGENET_NUM_CLASSES
DEFAULT_KMEANS_SEED: int = 0
DEFAULT_N_INIT: int = 10
DEFAULT_FEATURE_NORMALIZE: bool = False

LATENT_NODE_PREFIX: str = "L"
CLASS_NODE_PREFIX: str = "class_"

_EPS = 1e-12


def _require_numpy():
    if _np is None:  # pragma: no cover
        raise ImportError(
            "numpy is required for latent-hierarchy construction (pip install numpy)."
        )
    return _np


def level_cluster_count(level: int, num_classes: int = DEFAULT_NUM_CLASSES) -> int:
    """Number of K-means centres at ``level`` (1-based): ``min(2 ** level, num_classes)``."""
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    return int(max(1, min(2 ** int(level), int(num_classes))))


# --------------------------------------------------------------------------------------
# class feature extraction (step 2 of §E.1)
# --------------------------------------------------------------------------------------
class ClassFeatureAccumulator:
    """Streaming accumulator of per-class *average* features (``kX`` in §E.1).

    Accumulates ``sum``/``count`` per class index so that features can be streamed from a
    dataloader without materialising the whole feature matrix in memory.
    """

    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES, dim: Optional[int] = None):
        np = _require_numpy()
        self.num_classes = int(num_classes)
        self.dim = int(dim) if dim is not None else None
        self.sums: Optional["_np.ndarray"] = None
        self.counts = np.zeros(self.num_classes, dtype=np.float64)
        self._feature_sum_total: Optional["_np.ndarray"] = None

    def _ensure(self, dim: int) -> None:
        np = _require_numpy()
        if self.sums is None:
            self.sums = np.zeros((self.num_classes, int(dim)), dtype=np.float64)
            self._feature_sum_total = np.zeros(int(dim), dtype=np.float64)

    def update(self, features: Any, labels: Any) -> None:
        """Add a batch of ``features`` (N, D) with integer ``labels`` (N,)."""
        np = _require_numpy()
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        labs = np.asarray(labels).reshape(-1).astype(np.int64)
        self._ensure(feats.shape[1])
        valid = (labs >= 0) & (labs < self.num_classes)
        if not np.any(valid):
            return
        feats_v = feats[valid]
        labs_v = labs[valid]
        np.add.at(self.sums, labs_v, feats_v)
        np.add.at(self.counts, labs_v, 1.0)
        self._feature_sum_total += feats_v.sum(axis=0)

    @property
    def total_seen(self) -> float:
        return float(self.counts.sum())

    def result(
        self,
        normalize: bool = DEFAULT_FEATURE_NORMALIZE,
        empty_fill_mean: bool = True,
    ) -> Tuple[Any, Any]:
        """Return ``(class_features, counts)`` with shape ``(num_classes, dim)``."""
        np = _require_numpy()
        if self.sums is None:
            raise ValueError("No features were accumulated.")
        counts = self.counts.copy()
        denom = np.maximum(counts[:, None], 1.0)
        means = self.sums / denom
        empty = counts <= 0
        if np.any(empty) and empty_fill_mean:
            if counts.sum() > 0:
                fill = self.sums[~empty].sum(axis=0) / max(counts[~empty].sum(), 1.0)
            else:  # pragma: no cover - degenerate
                fill = np.zeros_like(means[0])
            means[empty] = fill
            logger.warning(
                "%d class(es) had no samples; filled with the dataset mean feature.",
                int(empty.sum()),
            )
        if normalize:
            means = l2_normalize(means)
        return means, counts


def l2_normalize(features: Any, eps: float = 1e-12) -> Any:
    """Row-wise L2 normalisation (optional preprocessing before K-means)."""
    np = _require_numpy()
    feats = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(feats, axis=-1, keepdims=True)
    return feats / np.maximum(norm, eps)


def class_mean_features(
    features: Any,
    labels: Any,
    num_classes: int = DEFAULT_NUM_CLASSES,
    normalize: bool = DEFAULT_FEATURE_NORMALIZE,
    empty_fill_mean: bool = True,
) -> Tuple[Any, Any]:
    """Average ``features`` per class -> ``k`` average class features (``kX``).

    Returns ``(class_features (k, D), counts (k,))``.  Classes without samples are filled with
    the dataset mean feature (and flagged through ``counts == 0``).
    """
    np = _require_numpy()
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    acc = ClassFeatureAccumulator(num_classes=num_classes, dim=feats.shape[1])
    acc.update(feats, labels)
    return acc.result(normalize=normalize, empty_fill_mean=empty_fill_mean)


def extract_class_features(features: Any, labels: Any, **kwargs) -> Tuple[Any, Any]:
    """Alias of :func:`class_mean_features` (paper notation: ``M(X)`` -> ``kX``)."""
    return class_mean_features(features, labels, **kwargs)


# --------------------------------------------------------------------------------------
# K-means
# --------------------------------------------------------------------------------------
def numpy_kmeans(
    X: Any,
    n_clusters: int,
    seed: int = 0,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Any:
    """Small k-means++ implementation used when scikit-learn is unavailable.

    Returns the integer cluster assignment for each row of ``X``.
    """
    np = _require_numpy()
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    n_clusters = int(max(1, min(n_clusters, n_samples)))
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=np.int64)

    rng = np.random.RandomState(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(max(1, int(n_init))):
        # k-means++ initialisation
        centres = [X[rng.randint(n_samples)]]
        for _c in range(1, n_clusters):
            d2 = np.min(
                np.stack([((X - c) ** 2).sum(axis=1) for c in centres], axis=1), axis=1
            )
            total = d2.sum()
            if total <= 0:
                centres.append(X[rng.randint(n_samples)])
            else:
                probs = d2 / total
                centres.append(X[rng.choice(n_samples, p=probs)])
        centres = np.stack(centres, axis=0)

        labels = np.zeros(n_samples, dtype=np.int64)
        for _it in range(int(max_iter)):
            dists = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(n_clusters):
                members = labels == c
                if np.any(members):
                    centres[c] = X[members].mean(axis=0)
        inertia = float(((X - centres[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels.astype(np.int64)


def fit_kmeans(
    X: Any,
    n_clusters: int,
    seed: int = DEFAULT_KMEANS_SEED,
    n_init: int = DEFAULT_N_INIT,
    init: str = "k-means++",
    backend: str = "auto",
) -> Any:
    """Fit K-means and return integer cluster assignments.

    ``backend`` may be ``"auto"``, ``"sklearn"`` or ``"numpy"``.
    """
    np = _require_numpy()
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    n_clusters = int(max(1, min(n_clusters, n_samples)))
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=np.int64)

    use_sklearn = backend in ("auto", "sklearn")
    if use_sklearn:
        try:
            from sklearn.cluster import KMeans  # type: ignore

            km = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=max(1, int(n_init)),
                random_state=int(seed),
            )
            return np.asarray(km.fit_predict(X), dtype=np.int64)
        except ImportError:
            if backend == "sklearn":
                raise
            logger.info("scikit-learn not available; using the numpy K-means fallback.")
    return numpy_kmeans(X, n_clusters, seed=seed, n_init=min(max(1, int(n_init)), 5))


def kmeans_levels(
    class_features: Any,
    max_level: int = DEFAULT_MAX_LEVEL,
    num_classes: Optional[int] = None,
    seed: int = DEFAULT_KMEANS_SEED,
    n_init: int = DEFAULT_N_INIT,
    init: str = "k-means++",
    normalize: bool = DEFAULT_FEATURE_NORMALIZE,
    backend: str = "auto",
) -> List[Any]:
    """Run the ``max_level`` independent K-means clusterings of §E.1.

    For ``i = 1, 2, ..., max_level`` a K-means with ``2^i`` centres is fitted *independently* on
    the average class features.  Returns a list whose ``i-1``-th entry is the integer cluster
    assignment (length ``k``) of level ``i``.
    """
    np = _require_numpy()
    X = np.asarray(class_features, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if num_classes is None:
        num_classes = X.shape[0]
    if normalize:
        X = l2_normalize(X)

    assignments: List[Any] = []
    for level in range(1, int(max_level) + 1):
        n_clusters = level_cluster_count(level, num_classes=X.shape[0])
        labels = fit_kmeans(
            X,
            n_clusters,
            seed=int(seed) + level,
            n_init=n_init,
            init=init,
            backend=backend,
        )
        assignments.append(labels.astype(np.int64))
        logger.debug("level %d: %d clusters over %d classes", level, n_clusters, X.shape[0])
    return assignments


# --------------------------------------------------------------------------------------
# the latent hierarchy
# --------------------------------------------------------------------------------------
class LatentHierarchy:
    """Class taxonomy induced by hierarchical K-means on average class features (§4.3.1).

    Parameters
    ----------
    assignments:
        List of integer arrays, the ``i-1``-th entry holding the level-``i`` cluster id
        (``2^i`` clusters) for every class index.
    base_level:
        Height shared by every pair of classes (§E.1): ``10``.
    num_classes:
        Number of classes covered by the hierarchy.
    """

    def __init__(
        self,
        assignments: Sequence[Any],
        base_level: int = DEFAULT_BASE_LEVEL,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        source_model: Optional[str] = None,
        seed: Optional[int] = None,
        normalize_features: bool = DEFAULT_FEATURE_NORMALIZE,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        np = _require_numpy()
        if not assignments:
            raise ValueError("A latent hierarchy needs at least one K-means level.")
        self.assignments: List[Any] = [
            np.asarray(a, dtype=np.int64).reshape(-1).copy() for a in assignments
        ]
        sizes = {a.shape[0] for a in self.assignments}
        if len(sizes) != 1:
            raise ValueError(f"All levels must cover the same classes; got sizes {sorted(sizes)}")
        inferred = int(next(iter(sizes)))
        self.num_classes = int(num_classes) if num_classes else inferred
        if self.num_classes != inferred:
            raise ValueError(
                f"num_classes={self.num_classes} does not match assignment length {inferred}."
            )
        self.base_level = int(base_level)
        self.class_names = [str(c) for c in class_names] if class_names is not None else None
        self.source_model = source_model
        self.seed = seed
        self.normalize_features = bool(normalize_features)
        self.metadata: Dict[str, Any] = dict(metadata or {})

        self._share_levels: Optional[Any] = None
        self._tree: Optional[WordNetHierarchy] = None
        self._scorers: Dict[str, Any] = {}

    # ------------------------------------------------------------------ basics
    @property
    def num_levels(self) -> int:
        return len(self.assignments)

    @property
    def max_level(self) -> int:
        return len(self.assignments)

    def cluster_count(self, level: int) -> int:
        """Number of K-means centres at ``level`` (1-based)."""
        return level_cluster_count(level, num_classes=self.num_classes)

    def level_assignments(self, level: int) -> Any:
        """Cluster ids of all classes at ``level`` (1-based)."""
        if not 1 <= int(level) <= self.num_levels:
            raise ValueError(f"level must be in [1, {self.num_levels}], got {level}")
        return self.assignments[int(level) - 1]

    def cluster_of(self, class_index: int, level: int) -> int:
        return int(self.level_assignments(level)[int(class_index)])

    def clusters_at_level(self, level: int) -> Dict[int, List[int]]:
        """``{cluster_id: [class indices]}`` for one level."""
        np = _require_numpy()
        assign = self.level_assignments(level)
        out: Dict[int, List[int]] = {}
        for c in np.unique(assign):
            out[int(c)] = np.where(assign == c)[0].astype(int).tolist()
        return out

    def class_name(self, class_index: int) -> str:
        if self.class_names is not None and 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return str(class_index)

    # ------------------------------------------------------------------ pairwise LCA height
    def share_levels(self) -> Any:
        """``(k, k)`` matrix holding the deepest level at which two classes share a cluster.

        Entry ``0`` means the two classes never share a cluster (their common ancestor is the
        base cluster level).  The diagonal holds :attr:`max_level`.
        """
        if self._share_levels is None:
            np = _require_numpy()
            share = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            for level, assign in enumerate(self.assignments, start=1):
                same = assign[:, None] == assign[None, :]
                share = np.where(same, level, share)  # levels ascend -> keep the deepest match
            self._share_levels = share
        return self._share_levels

    def share_level(self, class_a: int, class_b: int) -> int:
        """Deepest level at which ``class_a`` and ``class_b`` share a cluster (0 if never).

        Levels are scanned from the finest (``2^max_level`` clusters) to the coarsest
        (``2`` clusters); the *first* level at which both classes fall into the same cluster
        (i.e. the most specific shared cluster) is the K-means analogue of the LCA node.
        """
        if class_a == class_b:
            return self.max_level
        return int(self.share_levels()[int(class_a), int(class_b)])

    def lca_height(self, class_a: int, class_b: int) -> int:
        """Pairwise LCA height (§E.1): base level ``10`` if the classes never share a cluster."""
        if class_a == class_b:
            return 0
        shared = int(self.share_levels()[int(class_a), int(class_b)])
        return int(self.base_level) if shared <= 0 else int(self.base_level) - shared

    # ------------------------------------------------------------------ distance / similarity
    def similarity(self, class_a: int, class_b: int) -> float:
        """Similarity view: ``base_level`` for identical classes, else the shared level (0..9)."""
        if class_a == class_b:
            return float(self.base_level)
        return float(self.share_levels()[int(class_a), int(class_b)])

    def distance(self, class_a: int, class_b: int) -> float:
        """Distance view: ``base_level - share_level`` with a zero diagonal."""
        if class_a == class_b:
            return 0.0
        return float(self.lca_height(class_a, class_b))

    def similarity_matrix(self) -> Any:
        """Raw latent LCA matrix in *similarity* form (maximum on the diagonal).

        This is the matrix that :func:`src.hierarchy.lca_matrix.process_lca_matrix` inverts with
        ``latent_hierarchy=True`` (Addendum) to obtain the LCA distance matrix.
        """
        np = _require_numpy()
        share = self.share_levels()
        sim = share.astype(np.float64).copy()
        np.fill_diagonal(sim, float(self.base_level))
        return sim

    def latent_lca_matrix(self) -> Any:
        """Alias of :meth:`similarity_matrix` (raw latent taxonomy matrix)."""
        return self.similarity_matrix()

    def distance_matrix(self) -> Any:
        """``(k, k)`` LCA distance matrix ``d(i, j) = base_level - share_level(i, j)``."""
        np = _require_numpy()
        dist = float(self.base_level) - self.similarity_matrix()
        np.fill_diagonal(dist, 0.0)
        return dist

    def reverse_lca_matrix(self, temperature: float = 1.0) -> Any:
        """``1 - M_LCA`` after the Addendum processing pipeline (taxonomy soft labels)."""
        from .lca_matrix import reverse_lca_matrix as _reverse

        return _reverse(self.processed_matrix(temperature=temperature))

    def processed_matrix(self, temperature: float = 1.0, scale: bool = True) -> Any:
        """Apply ``invert -> ** temperature -> MinMax`` to the raw latent matrix."""
        from .lca_matrix import process_lca_matrix

        matrix = process_lca_matrix(
            self.latent_lca_matrix(),
            temperature=float(temperature),
            latent_hierarchy=True,
            scale=scale,
        )
        return matrix

    # ------------------------------------------------------------------ tree / scorer views
    def parent_map(self) -> Dict[str, Optional[str]]:
        """Build child -> parent links for the cluster tree.

        Level-``i`` clusters (``i > 1``) are attached to the level-``i-1`` cluster containing the
        majority of their member classes; level-1 clusters are roots.  Class nodes
        ``class_<k>`` are attached to their level-``max_level`` cluster, which places every class
        leaf at depth ``base_level`` (``= max_level + 1 = 10``).
        """
        np = _require_numpy()
        parents: Dict[str, Optional[str]] = {}
        for level in range(1, self.num_levels + 1):
            assign = self.level_assignments(level)
            for c in np.unique(assign):
                node = node_id(level, int(c))
                if level == 1:
                    parents[node] = None
                    continue
                members = np.where(assign == c)[0]
                prev = self.level_assignments(level - 1)
                values, counts = np.unique(prev[members], return_counts=True)
                parent_cluster = int(values[int(np.argmax(counts))])
                parents[node] = node_id(level - 1, parent_cluster)
        top = self.level_assignments(self.num_levels)
        for k in range(self.num_classes):
            parents[class_node_id(k)] = node_id(self.num_levels, int(top[k]))
        return parents

    def to_wordnet_hierarchy(self) -> WordNetHierarchy:
        """Adapt the latent taxonomy to the :class:`WordNetHierarchy` interface.

        The resulting tree can be fed to :class:`src.hierarchy.info_content.HierarchyScorer`, so
        the standard ``D_LCA^I`` / ``D_LCA^P`` implementations apply unchanged.  With the default
        ``base_level = 10`` every class leaf sits at depth ``10`` and internal nodes at depth
        ``= level``, so the depth score is ``P(LCA) = level`` and the information content of an
        LCA node is ``log2(cluster size)`` -- i.e. ``D_LCA^I ~= 10 - share_level``.
        """
        if self._tree is None:
            class_to_leaf = {k: class_node_id(k) for k in range(self.num_classes)}
            self._tree = WordNetHierarchy(
                parents=self.parent_map(),
                class_to_leaf=class_to_leaf,
                class_names=self.class_names,
            )
        return self._tree

    hierarchy = to_wordnet_hierarchy  # convenience alias

    def scorer(self, mode: str = "information", num_leaves: Optional[int] = None, base: float = 2.0):
        """Cached :class:`HierarchyScorer` over the latent tree (``mode`` in information/depth)."""
        if mode not in self._scorers:
            from .info_content import HierarchyScorer

            self._scorers[mode] = HierarchyScorer(
                self.to_wordnet_hierarchy(), mode=mode, num_leaves=num_leaves, base=base
            )
        return self._scorers[mode]

    def lca_distance(self, y_prime: int, y: int, mode: str = "information") -> float:
        """``D_LCA(y', y)`` computed with the standard scorers on the latent tree."""
        return float(self.scorer(mode=mode).lca_distance(y_prime, y))

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        np = _require_numpy()
        return {
            "assignments": [a.astype(int).tolist() for a in self.assignments],
            "base_level": self.base_level,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "source_model": self.source_model,
            "seed": self.seed,
            "normalize_features": self.normalize_features,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LatentHierarchy":
        return cls(
            assignments=payload["assignments"],
            base_level=payload.get("base_level", DEFAULT_BASE_LEVEL),
            num_classes=payload.get("num_classes"),
            class_names=payload.get("class_names"),
            source_model=payload.get("source_model"),
            seed=payload.get("seed"),
            normalize_features=payload.get("normalize_features", DEFAULT_FEATURE_NORMALIZE),
            metadata=payload.get("metadata"),
        )

    def save(self, path: str) -> str:
        """Persist the hierarchy as ``<path>.json`` (assignments + metadata)."""
        np = _require_numpy()
        path = _normalise_path(path)
        if not path.endswith(".json"):
            path = path + ".json"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.to_dict()
        payload["assignments"] = [list(map(int, a)) for a in payload["assignments"]]
        with open(path, "w") as fh:
            json.dump(payload, fh)
        logger.info("Saved latent hierarchy to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "LatentHierarchy":
        path = _normalise_path(path)
        if not path.endswith(".json"):
            path = path + ".json"
        with open(path) as fh:
            payload = json.load(fh)
        return cls.from_dict(payload)

    @classmethod
    def from_features(
        cls,
        class_features: Any,
        max_level: int = DEFAULT_MAX_LEVEL,
        base_level: int = DEFAULT_BASE_LEVEL,
        class_names: Optional[Sequence[str]] = None,
        source_model: Optional[str] = None,
        seed: int = DEFAULT_KMEANS_SEED,
        n_init: int = DEFAULT_N_INIT,
        init: str = "k-means++",
        normalize: bool = DEFAULT_FEATURE_NORMALIZE,
        backend: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LatentHierarchy":
        assignments = kmeans_levels(
            class_features,
            max_level=max_level,
            num_classes=_num_rows(class_features),
            seed=seed,
            n_init=n_init,
            init=init,
            normalize=normalize,
            backend=backend,
        )
        return cls(
            assignments=assignments,
            base_level=base_level,
            num_classes=_num_rows(class_features),
            class_names=class_names,
            source_model=source_model,
            seed=seed,
            normalize_features=normalize,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ dunder
    def __len__(self) -> int:
        return self.num_classes

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LatentHierarchy(num_classes={self.num_classes}, num_levels={self.num_levels}, "
            f"base_level={self.base_level}, source_model={self.source_model!r})"
        )


def _num_rows(array: Any) -> int:
    np = _require_numpy()
    arr = np.asarray(array)
    return int(arr.shape[0]) if arr.ndim > 1 else int(arr.shape[0])


def _normalise_path(path: str) -> str:
    return os.path.expanduser(str(path))


def node_id(level: int, cluster: int) -> str:
    """Node id of the ``cluster``-th cluster at ``level`` (1-based level)."""
    return f"{LATENT_NODE_PREFIX}{int(level)}_{int(cluster)}"


def class_node_id(class_index: int) -> str:
    """Leaf node id of a class in the latent tree."""
    return f"{CLASS_NODE_PREFIX}{int(class_index)}"


# --------------------------------------------------------------------------------------
# top-level builders
# --------------------------------------------------------------------------------------
def build_latent_hierarchy(
    class_features: Any,
    max_level: int = DEFAULT_MAX_LEVEL,
    base_level: int = DEFAULT_BASE_LEVEL,
    class_names: Optional[Sequence[str]] = None,
    source_model: Optional[str] = None,
    seed: int = DEFAULT_KMEANS_SEED,
    n_init: int = DEFAULT_N_INIT,
    init: str = "k-means++",
    normalize: bool = DEFAULT_FEATURE_NORMALIZE,
    backend: str = "auto",
    metadata: Optional[Dict[str, Any]] = None,
) -> LatentHierarchy:
    """Build a latent hierarchy from ``k`` average class features (§E.1)."""
    return LatentHierarchy.from_features(
        class_features,
        max_level=max_level,
        base_level=base_level,
        class_names=class_names,
        source_model=source_model,
        seed=seed,
        n_init=n_init,
        init=init,
        normalize=normalize,
        backend=backend,
        metadata=metadata,
    )


def latent_hierarchy_from_class_features(class_features: Any, **kwargs) -> LatentHierarchy:
    """Alias of :func:`build_latent_hierarchy`."""
    return build_latent_hierarchy(class_features, **kwargs)


def latent_hierarchy_from_features(
    features: Any,
    labels: Any,
    num_classes: int = DEFAULT_NUM_CLASSES,
    normalize_features: bool = DEFAULT_FEATURE_NORMALIZE,
    **kwargs,
) -> LatentHierarchy:
    """Convenience wrapper: average ``features`` per class, then cluster (§E.1)."""
    class_features, counts = class_mean_features(
        features, labels, num_classes=num_classes, normalize=False
    )
    kwargs.setdefault("normalize", normalize_features)
    kwargs.setdefault("source_model", kwargs.get("source_model"))
    hierarchy = build_latent_hierarchy(class_features, **kwargs)
    hierarchy.metadata.setdefault("num_samples", float(counts.sum()))
    hierarchy.metadata.setdefault("classes_without_samples", int((counts <= 0).sum()))
    return hierarchy


def latent_hierarchies_from_sources(
    source_class_features: Dict[str, Any],
    class_names: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dict[str, LatentHierarchy]:
    """Build one latent hierarchy per source model (the 75 hierarchies of Table 4)."""
    hierarchies: Dict[str, LatentHierarchy] = {}
    for name, feats in source_class_features.items():
        logger.info("Building latent hierarchy from source model %s", name)
        hierarchies[name] = build_latent_hierarchy(
            feats, class_names=class_names, source_model=name, **kwargs
        )
    return hierarchies


def extract_latent_hierarchy_from_model(
    model: Any,
    loader: Iterable,
    num_classes: int = DEFAULT_NUM_CLASSES,
    class_names: Optional[Sequence[str]] = None,
    source_model: Optional[str] = None,
    device: Optional[Any] = None,
    feature_fn: Optional[Callable[[Any], Any]] = None,
    max_batches: Optional[int] = None,
    normalize_features: bool = DEFAULT_FEATURE_NORMALIZE,
    feature_cache: Optional[Any] = None,
    **kwargs,
) -> LatentHierarchy:
    """Extract ``M(X)`` with ``model`` over ``loader`` and build its latent hierarchy.

    ``feature_fn`` receives the input tensor and must return the penultimate feature batch
    (``M(X)``).  When it is omitted, common zoo conventions are tried: ``model.features(x)``,
    ``model(x)`` returning ``(features, logits)``, and ``model(x)`` returning bare features.
    """
    np = _require_numpy()
    acc = ClassFeatureAccumulator(num_classes=num_classes)
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images, labels = _unpack_batch(batch)
        if feature_cache is not None:
            features = feature_cache[labels]  # pragma: no cover - optional fast path
        else:
            features = _forward_features(model, images, device=device, feature_fn=feature_fn)
        acc.update(features, labels)
    class_features, counts = acc.result(normalize=False)
    hierarchy = build_latent_hierarchy(
        class_features,
        class_names=class_names,
        source_model=source_model,
        normalize=normalize_features,
        **kwargs,
    )
    hierarchy.metadata.setdefault("num_samples", float(counts.sum()))
    return hierarchy


def _unpack_batch(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        labels = batch.get("label", batch.get("labels"))
        return images, labels
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Cannot unpack a batch into (images, labels).")


def _forward_features(
    model: Any,
    images: Any,
    device: Optional[Any] = None,
    feature_fn: Optional[Callable[[Any], Any]] = None,
) -> Any:
    import torch  # local import: torch is only needed for the model-driven path

    if device is not None and hasattr(images, "to"):
        images = images.to(device)
    with torch.no_grad():
        if feature_fn is not None:
            out = feature_fn(images)
        elif hasattr(model, "features") and callable(getattr(model, "features")):
            out = model.features(images)
        else:
            out = model(images)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "detach"):
        out = out.detach().float().cpu().numpy()
    return out


# --------------------------------------------------------------------------------------
# convenience matrix helpers
# --------------------------------------------------------------------------------------
def latent_lca_matrix(hierarchy: LatentHierarchy, processed: bool = False, temperature: float = 1.0):
    """Raw (similarity form) or processed LCA matrix of a latent hierarchy."""
    return hierarchy.processed_matrix(temperature=temperature) if processed else hierarchy.latent_lca_matrix()


def latent_distance_matrix(hierarchy: LatentHierarchy):
    """Distance-form latent LCA matrix (zero diagonal)."""
    return hierarchy.distance_matrix()


def synthetic_latent_hierarchy(
    num_classes: int = 16,
    max_level: int = 3,
    base_level: int = DEFAULT_BASE_LEVEL,
    seed: int = 0,
    class_names: Optional[Sequence[str]] = None,
) -> LatentHierarchy:
    """Deterministic toy hierarchy (offline tests): classes are embedded on a grid."""
    np = _require_numpy()
    rng = np.random.RandomState(seed)
    side = int(math.ceil(math.sqrt(num_classes)))
    coords = np.array(
        [[i % side, i // side] for i in range(num_classes)], dtype=np.float64
    )
    coords += 0.05 * rng.randn(*coords.shape)
    hierarchy = LatentHierarchy.from_features(
        coords,
        max_level=max_level,
        base_level=base_level,
        class_names=class_names,
        source_model="synthetic",
        seed=seed,
    )
    return hierarchy
