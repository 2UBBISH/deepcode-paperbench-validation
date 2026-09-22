"""LCA / ELCA distances (Section 2 and Appendix D.2/D.3).

Given a ground-truth class ``y``, a prediction ``y'`` and their lowest common
ancestor ``N_LCA(y, y')`` in a taxonomy ``T``, the paper defines::

    D_LCA(y', y) := f(y) - f(N_LCA(y, y'))

with ``f`` either the tree depth ``P`` or the information content ``I``.  The
per-dataset distance averages the severity of the mistakes::

    D_LCA(model, M) := 1/n * sum_i D_LCA(y'_i, y_i)   <=>   y_i != y'_i

All functions here work either directly on a :class:`WordNetHierarchy` or on a
pre-computed ``(n_classes, n_classes)`` distance matrix (which is what the
latent-hierarchy experiments use).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .hierarchy import WordNetHierarchy


def _as_int_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).ravel()


# --------------------------------------------------------------------------- #
# pairwise distances for a single (prediction, ground-truth) pair
# --------------------------------------------------------------------------- #
def _leaf_index(hierarchy: WordNetHierarchy, synset_or_index) -> int:
    if isinstance(synset_or_index, (int, np.integer)):
        return int(synset_or_index)
    return hierarchy.index_of[synset_or_index]


def d_lca_information(hierarchy: WordNetHierarchy, y_pred, y_true) -> float:
    """``D_LCA^I(y', y) = I(y) - I(LCA(y, y'))``."""
    i = _leaf_index(hierarchy, y_pred)
    j = _leaf_index(hierarchy, y_true)
    a, b = hierarchy.class_synsets[i], hierarchy.class_synsets[j]
    anc = hierarchy.lca(a, b)
    return hierarchy.information(a) - hierarchy.information(anc)


def d_lca_path(hierarchy: WordNetHierarchy, y_pred, y_true) -> float:
    """``D_LCA^P(y', y) = (P(y)-P(LCA)) + (P(y')-P(LCA))``."""
    i = _leaf_index(hierarchy, y_pred)
    j = _leaf_index(hierarchy, y_true)
    a, b = hierarchy.class_synsets[i], hierarchy.class_synsets[j]
    anc = hierarchy.lca(a, b)
    return float(
        (hierarchy.depth[a] - hierarchy.depth[anc])
        + (hierarchy.depth[b] - hierarchy.depth[anc])
    )


# --------------------------------------------------------------------------- #
# dataset level distances
# --------------------------------------------------------------------------- #
def _resolve_matrix(
    hierarchy: Optional[WordNetHierarchy],
    matrix: Optional[np.ndarray],
    score: str,
) -> np.ndarray:
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)
    if hierarchy is None:
        raise ValueError("either `hierarchy` or `matrix` must be provided")
    return hierarchy.lca_distance_matrix(score=score)


def dataset_lca(
    hierarchy: Optional[WordNetHierarchy] = None,
    predictions=None,
    targets=None,
    score: str = "information",
    matrix: Optional[np.ndarray] = None,
    return_details: bool = False,
):
    """Average LCA distance over the misclassified samples of a dataset.

    ``predictions`` are class indices into the same 0..999 ImageNet space used
    by the hierarchy (or distance matrix).  Correctly classified samples are
    skipped, exactly as in the paper's equation.
    """
    if predictions is None or targets is None:
        raise ValueError("`predictions` and `targets` are required")
    preds = _as_int_array(predictions)
    tgts = _as_int_array(targets)
    if preds.shape != tgts.shape:
        raise ValueError("predictions and targets must have matching shapes")

    dist = _resolve_matrix(hierarchy, matrix, score)
    wrong = preds != tgts
    if not np.any(wrong):
        value = 0.0
        per_sample = np.zeros_like(preds, dtype=np.float64)
    else:
        per_sample = np.zeros_like(preds, dtype=np.float64)
        per_sample[wrong] = dist[preds[wrong], tgts[wrong]]
        value = float(per_sample[wrong].mean())

    if return_details:
        return value, {
            "per_sample": per_sample,
            "wrong": wrong,
            "top1": float((~wrong).mean()),
        }
    return value


def dataset_lca_from_matrix(
    matrix: np.ndarray, predictions, targets, return_details: bool = False
):
    return dataset_lca(
        matrix=matrix,
        predictions=predictions,
        targets=targets,
        return_details=return_details,
    )


def dataset_elca(
    probabilities: Sequence[Sequence[float]],
    targets,
    hierarchy: Optional[WordNetHierarchy] = None,
    matrix: Optional[np.ndarray] = None,
    score: str = "information",
    chunk_size: int = 4096,
) -> float:
    """Expected Lowest Common Ancestor distance (Appendix D.3).

    ``D_ELCA = 1/(n*K) * sum_i sum_k p_{k,i} * D_LCA(k, y_i)``.

    Note that ``D_ELCA`` is defined in the appendix only (Table 8), i.e. it is
    reported here for completeness.  It is sensitive to the logit temperature
    (the paper explicitly warns not to compare it across modalities).
    """
    tgts = _as_int_array(targets)
    dist = _resolve_matrix(hierarchy, matrix, score)
    n = len(tgts)
    # streamed so that a 50k x 1000 probability matrix never has to be held in
    # memory together with the distance matrix
    total = 0.0
    n_classes = dist.shape[0]
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        probs = np.asarray(probabilities[start:stop], dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError("`probabilities` must be (n_samples, n_classes)")
        if probs.shape[1] != n_classes:
            raise ValueError(
                "distance matrix has %d classes but probabilities have %d"
                % (n_classes, probs.shape[1])
            )
        d_to_truth = dist[:, tgts[start:stop]].T  # (chunk, K)
        total += float((probs * d_to_truth).sum())
    return float(total / (n * n_classes))


def dataset_elca_unscaled(
    probabilities: Sequence[Sequence[float]],
    targets,
    hierarchy: Optional[WordNetHierarchy] = None,
    matrix: Optional[np.ndarray] = None,
    score: str = "information",
    chunk_size: int = 4096,
) -> float:
    """``1/n * sum_i sum_k p_{k,i} D_LCA(k, y_i)`` -- the un-normalised variant.

    Because the appendix equation contains an extra ``1/K`` factor that makes
    the reported magnitudes hard to obtain in practice, we expose this variant
    as well so that users can reproduce either convention.
    """
    tgts = _as_int_array(targets)
    dist = _resolve_matrix(hierarchy, matrix, score)
    n = len(tgts)
    total = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        probs = np.asarray(probabilities[start:stop], dtype=np.float64)
        d_to_truth = dist[:, tgts[start:stop]].T
        total += float((probs * d_to_truth).sum())
    return float(total / n)


def topk_accuracy(logits: np.ndarray, targets, k: int = 1) -> float:
    """Top-k accuracy (``topk_accuracy(logits, targets, 5)`` -> Top-5)."""
    logits = np.asarray(logits, dtype=np.float64)
    tgts = _as_int_array(targets)
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    hits = (topk == tgts[:, None]).any(axis=1)
    return float(hits.mean())


def softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def elca_from_logits(
    logits: np.ndarray,
    targets,
    hierarchy: Optional[WordNetHierarchy] = None,
    matrix: Optional[np.ndarray] = None,
    score: str = "information",
    chunk_size: int = 4096,
) -> float:
    """``D_ELCA`` computed straight from logits, one chunk at a time.

    Avoids materialising the full ``(n_samples, 1000)`` probability matrix.
    """
    logits = np.asarray(logits, dtype=np.float64)
    tgts = _as_int_array(targets)
    dist = _resolve_matrix(hierarchy, matrix, score)
    n = len(tgts)
    n_classes = dist.shape[0]
    total = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        probs = softmax(logits[start:stop], axis=1)
        d_to_truth = dist[:, tgts[start:stop]].T
        total += float((probs * d_to_truth).sum())
    return float(total / (n * n_classes))


def classify_summary(
    logits: np.ndarray,
    targets,
    hierarchy: Optional[WordNetHierarchy] = None,
    matrix: Optional[np.ndarray] = None,
    score: str = "information",
) -> dict:
    """Bundle of the numbers reported for a single model/dataset."""
    logits = np.asarray(logits, dtype=np.float64)
    tgts = _as_int_array(targets)
    preds = logits.argmax(axis=1)
    out = {
        "top1": topk_accuracy(logits, tgts, 1),
        "top5": topk_accuracy(logits, tgts, 5),
    }
    if hierarchy is not None or matrix is not None:
        lca_value, details = dataset_lca(
            hierarchy=hierarchy,
            matrix=matrix,
            predictions=preds,
            targets=tgts,
            score=score,
            return_details=True,
        )
        out["lca"] = lca_value
        out["elca"] = elca_from_logits(
            logits, tgts, hierarchy=hierarchy, matrix=matrix, score=score
        )
        out["n_mistakes"] = int(details["wrong"].sum())
    return out
