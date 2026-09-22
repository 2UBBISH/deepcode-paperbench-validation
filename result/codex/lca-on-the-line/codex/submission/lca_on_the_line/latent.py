"""Latent class hierarchies built with K-means clustering (Section 4.3.1).

Following Appendix E.1:

1. take a pretrained source model ``M`` and the in-distribution images ``X``,
   extract ``M(X)`` and average the features per class -> ``kX`` (k = 1000);
2. run a 9-layer hierarchical clustering on ``kX`` with ``2**i`` centres for
   ``i = 1..9`` (``2**9 < 1000``);
3. the pairwise *LCA height* is the cluster level at which two classes first
   share a cluster; all classes share a base cluster level of 10.

The resulting height matrix is a *similarity* (larger == closer), so -- as the
paper addendum notes -- it is inverted before being turned into an LCA distance
matrix (``process_lca_matrix`` below).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import minmax_scale

DEFAULT_N_LEVELS = 9
BASE_CLUSTER_LEVEL = 10


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def kmeans_cluster_levels(
    class_features: np.ndarray,
    n_levels: int = DEFAULT_N_LEVELS,
    seed: int = 0,
    normalize: bool = True,
) -> List[np.ndarray]:
    """K-means at every hierarchy level; returns the label array per level.

    Level ``i`` (1-indexed) uses ``2**i`` clusters; ``k-means`` is run
    independently on each level exactly as described in Appendix E.1.
    """
    from sklearn.cluster import KMeans

    feats = np.asarray(class_features, dtype=np.float64)
    if normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-12, None)

    labels: List[np.ndarray] = []
    for level in range(1, n_levels + 1):
        k = min(2 ** level, feats.shape[0])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels.append(km.fit_predict(feats))
    return labels


def _share_matrix(labels: np.ndarray) -> np.ndarray:
    """Boolean ``(n, n)`` matrix: do i and j fall in the same cluster?"""
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


def latent_lca_height_matrix(
    class_features: np.ndarray,
    n_levels: int = DEFAULT_N_LEVELS,
    seed: int = 0,
    labels_per_level: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Pairwise LCA *height* (similarity) matrix.

    ``H[i, j]`` is the deepest hierarchy level at which classes ``i`` and ``j``
    share a cluster (0 when they only meet at the root).  The diagonal is
    ``n_levels`` (a class always shares a cluster with itself), which makes the
    inverted matrix have a zero diagonal.
    """
    if labels_per_level is None:
        labels_per_level = kmeans_cluster_levels(
            class_features, n_levels=n_levels, seed=seed
        )
    labels_per_level = list(labels_per_level)
    n = len(labels_per_level[0])
    height = np.zeros((n, n), dtype=np.float64)
    for level, labels in enumerate(labels_per_level, start=1):
        same = _share_matrix(labels)
        # keep the deepest level at which the pair is together
        height[same] = level
    np.fill_diagonal(height, float(n_levels))
    return height


def latent_lca_distance_matrix(
    class_features: np.ndarray,
    n_levels: int = DEFAULT_N_LEVELS,
    seed: int = 0,
) -> np.ndarray:
    """Inverted (distance) version of the latent LCA matrix, diagonal zero."""
    height = latent_lca_height_matrix(
        class_features, n_levels=n_levels, seed=seed
    )
    return height.max() - height


def process_lca_matrix(
    lca_matrix_raw: Optional[np.ndarray],
    tree_prefix: str,
    temperature: float = 1.0,
    per_column: bool = True,
):
    """Normalise a raw LCA matrix, exactly as in the paper addendum.

    WordNet matrices are already distances; latent-hierarchy matrices are
    similarities and get inverted first.  The result is raised to
    ``temperature`` and min-max scaled into ``[0, 1]`` with a zero diagonal.

    ``per_column=True`` reproduces the addendum snippet literally
    (``MinMaxScaler().fit_transform`` scales every column independently); pass
    ``False`` for a single global min-max scaling.
    """
    if lca_matrix_raw is None:
        return None
    raw = np.asarray(lca_matrix_raw, dtype=np.float64)
    if tree_prefix != "WordNet":
        result_matrix = np.max(raw) - raw
    else:
        result_matrix = raw
    result_matrix = result_matrix ** temperature

    if per_column:
        from sklearn.preprocessing import MinMaxScaler

        result_matrix = MinMaxScaler().fit_transform(result_matrix)
    else:
        result_matrix = minmax_scale(result_matrix.ravel()).reshape(raw.shape)
    # guard the sanity checks requested by the addendum
    np.fill_diagonal(result_matrix, 0.0)

    import torch

    return torch.from_numpy(result_matrix)


# --------------------------------------------------------------------------- #
# benchmark helpers
# --------------------------------------------------------------------------- #
def class_features_from_logits_and_features(
    features: np.ndarray,
    targets: Sequence[int],
    n_classes: int = 1000,
    normalize: bool = False,
) -> np.ndarray:
    """Average the per-image features of each class -> ``kX``."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets)
    dim = features.shape[1]
    out = np.zeros((n_classes, dim), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for idx, cls in enumerate(targets):
        out[cls] += features[idx]
        counts[cls] += 1
    counts = np.clip(counts, 1, None)
    out = out / counts[:, None]
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(norms, 1e-12, None)
    return out


def latent_hierarchy_correlation(
    id_predictions: Sequence[np.ndarray],
    ood_predictions: Sequence[np.ndarray],
    targets: Sequence[int],
    matrices: Sequence[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Table 4: correlation (PEA) between ID LCA and OOD Top-1.

    ``id_predictions`` is a list of per-model ImageNet predictions (one array
    per model), ``ood_predictions`` the corresponding OOD predictions,
    ``targets`` the ground-truth labels and ``matrices`` the latent distance
    matrices of the source models.
    """
    from .lca import dataset_lca_from_matrix
    from .metrics import pea

    n_models = len(id_predictions)
    per_matrix: List[np.ndarray] = []
    id_lca_by_matrix: List[List[float]] = []
    ood_top1 = np.array(
        [
            np.mean([np.asarray(p) == np.asarray(t) for p, t in zip(pred, targets)])
            for pred in ood_predictions
        ]
    )
    for matrix in matrices:
        id_lca = np.array(
            [
                dataset_lca_from_matrix(matrix, id_predictions[m], targets)
                for m in range(n_models)
            ]
        )
        id_lca_by_matrix.append(id_lca.tolist())
        per_matrix.append(pea(id_lca, ood_top1))
    per_matrix_arr = np.array(per_matrix)
    return {
        "per_matrix_pea": per_matrix_arr,
        "mean": float(per_matrix_arr.mean()),
        "min": float(per_matrix_arr.min()),
        "max": float(per_matrix_arr.max()),
        "std": float(per_matrix_arr.std()),
        "ood_top1": ood_top1,
    }
