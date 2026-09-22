"""Dataset-level LCA and ELCA metrics (paper Sections 2 and D.3).

This module implements the *aggregate* mistake-severity measurements that give
"LCA-on-the-Line" its x-axis:

* :math:`D_{LCA}(\\text{model}, \\mathcal{M})` (plain LCA distance, Section 2)::

      D_LCA(model, M) := (1/n) * sum_i D_LCA(y_hat_i, y_i)   <=> y_i != y_hat_i

  Only *misclassified* samples contribute (the indicator ``y_i != y_hat_i``).

* :math:`D_{ELCA}(\\text{model}, \\mathcal{M})` (expected LCA distance, Section D.3)::

      D_ELCA(model, M) := (1 / nK) * sum_i sum_k p_hat_{k,i} * D_LCA(k, y_i)

  where ``p_hat_{.,i} = softmax(logits_i)`` over all ``K`` classes.

The pairwise term :math:`D_{LCA}(y', y)=I(y)-I(N_{LCA}(y,y'))` (information
content, base 2) is delegated to :mod:`src.hierarchy.info_content` /
:mod:`src.hierarchy.lca`; this module only performs the aggregation over a
dataset of predictions / logits.

Both measures are *distances* (lower is better) and are reported with a
downward arrow in the paper's tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

try:  # numpy is a hard requirement at runtime but keep import defensive
    import numpy as _np
except Exception:  # pragma: no cover - numpy always present in practice
    _np = None  # type: ignore

from ..hierarchy.info_content import HierarchyScorer
from ..hierarchy.lca import DEFAULT_DISTANCE_MODE, LcaDistance, pairwise_lca_matrix
from ..hierarchy.wordnet import IMAGENET_NUM_CLASSES, WordNetHierarchy

logger = logging.getLogger(__name__)

__all__ = [
    "ModelMetrics",
    "LcaMetric",
    "softmax",
    "topk_accuracy",
    "top1_accuracy",
    "top5_accuracy",
    "lca_distance_dataset",
    "elca_distance_dataset",
    "expected_lca_distance_dataset",
    "evaluate_model_outputs",
    "DEFAULT_SOFTMAX_TEMPERATURE",
]

DEFAULT_SOFTMAX_TEMPERATURE: float = 1.0
ArrayLike = Any  # numpy.ndarray / torch.Tensor / nested lists


def _require_numpy():
    if _np is None:  # pragma: no cover
        raise ImportError("numpy is required for dataset-level LCA/ELCA metrics")
    return _np


def _to_numpy(x: ArrayLike):
    """Convert torch tensors / lists to a float64 numpy array (no-op for numpy)."""
    np = _require_numpy()
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Basic primitives: softmax / top-k accuracy
# ---------------------------------------------------------------------------
def softmax(
    logits: ArrayLike,
    temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    axis: int = -1,
) -> "Any":
    """Numerically stable softmax with optional temperature.

    ``softmax(x / T)``; ``T = 1`` reproduces the standard definition used in
    Section D.3 for ELCA.  Note that the paper warns ELCA is sensitive to the
    logit temperature, so it must not be compared across modalities.
    """
    np = _require_numpy()
    x = np.asarray(_to_numpy(logits), dtype=np.float64)
    if temperature is None or temperature <= 0:
        temperature = 1.0
    x = x / float(temperature)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def topk_accuracy(logits: ArrayLike, targets: ArrayLike, k: int = 1) -> float:
    """Top-``k`` accuracy of ``logits`` (n, K) against ``targets`` (n,)."""
    np = _require_numpy()
    logits = np.asarray(_to_numpy(logits), dtype=np.float64)
    targets = np.asarray(_to_numpy(targets), dtype=np.int64).reshape(-1)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same length")
    k = int(max(1, min(k, logits.shape[1])))
    if logits.shape[0] == 0:
        return float("nan")
    topk = np.argsort(-logits, axis=1)[:, :k]
    correct = (topk == targets[:, None]).any(axis=1)
    return float(np.mean(correct))


def top1_accuracy(logits: ArrayLike, targets: ArrayLike) -> float:
    """Top-1 accuracy (fraction of correctly classified samples)."""
    return topk_accuracy(logits, targets, k=1)


def top5_accuracy(logits: ArrayLike, targets: ArrayLike) -> float:
    """Top-5 accuracy."""
    return topk_accuracy(logits, targets, k=5)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    """Dataset-level measurements for one model on one dataset (cf. Table 1/8).

    Attributes
    ----------
    top1, top5:
        Accuracies in ``[0, 1]``.
    lca:
        Mean information-content LCA distance over misclassified samples
        (Section 2).  ``nan`` when the model is perfect (no mistakes).
    lca_all:
        Mean LCA distance including correctly classified samples (0 for those);
        equals ``lca * (1 - top1)``.
    elca:
        Expected LCA distance (Section D.3).
    num_samples, num_classes:
        Dataset size and number of candidate classes ``K``.
    num_misclassified:
        Number of samples that contribute to ``lca``.
    """

    top1: float = float("nan")
    top5: float = float("nan")
    lca: float = float("nan")
    lca_all: float = float("nan")
    elca: float = float("nan")
    num_samples: int = 0
    num_classes: int = 0
    num_misclassified: int = 0
    name: str = ""
    dataset: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name or 'model'}@{self.dataset or 'dataset'}: "
            f"top1={self.top1:.4f} top5={self.top5:.4f} "
            f"lca={self.lca:.4f} elca={self.elca:.4f}"
        )


# ---------------------------------------------------------------------------
# Main metric driver
# ---------------------------------------------------------------------------
class LcaMetric:
    """Compute dataset-level LCA / ELCA for an arbitrary class hierarchy.

    Parameters
    ----------
    hierarchy:
        A :class:`~src.hierarchy.wordnet.WordNetHierarchy` (WordNet or a latent
        hierarchy exported to tree form).  Optional if ``matrix`` is given.
    matrix:
        Optional pre-computed ``K x K`` pairwise distance matrix with
        ``matrix[k, y] = D_LCA(k, y)`` (prediction ``k``, ground-truth ``y``).
        Passing it avoids recomputing LCAs and is the fast path used by the
        evaluation driver (it is also how latent-hierarchy matrices are fed in).
    mode:
        ``"information"`` (default, used for LCA measurements) or ``"depth"``
        (``D_LCA^P``, used for linear-probing experiments).
    num_classes:
        Number of classes ``K``; inferred from ``matrix`` / hierarchy.
    base, num_leaves:
        Passed through to :class:`~src.hierarchy.info_content.HierarchyScorer`
        for the information-content variant (base 2 per Section D.2.1).
    """

    def __init__(
        self,
        hierarchy: Optional[WordNetHierarchy] = None,
        matrix: Optional[Sequence[Sequence[float]]] = None,
        mode: str = DEFAULT_DISTANCE_MODE,
        num_classes: Optional[int] = None,
        base: float = 2.0,
        num_leaves: Optional[int] = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.mode = mode
        self.base = float(base)
        self.num_leaves = num_leaves
        self._matrix: Optional[Any] = None
        self._scorer: Optional[HierarchyScorer] = None
        self._distance_fn: Optional[LcaDistance] = None

        if matrix is not None:
            self._matrix = _require_numpy().asarray(
                _to_numpy(matrix), dtype=np.float64
            )
            if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
                raise ValueError("matrix must be a square K x K distance matrix")
            inferred = int(self._matrix.shape[0])
        else:
            inferred = IMAGENET_NUM_CLASSES if hierarchy is None else hierarchy.num_classes

        self.num_classes = int(num_classes or inferred)
        if self._matrix is None and hierarchy is not None:
            self._distance_fn = LcaDistance(
                hierarchy, mode=mode, num_leaves=num_leaves, base=base
            )
            self._scorer = self._distance_fn.scorer
        if self._matrix is not None and self._matrix.shape[0] != self.num_classes:
            # keep them consistent; the matrix governs
            self.num_classes = int(self._matrix.shape[0])

    # -- pairwise distances -------------------------------------------------
    @property
    def matrix(self) -> "Any":
        """``K x K`` pairwise distance matrix (``M[k, y] = D_LCA(k, y)``), cached."""
        if self._matrix is None:
            if self.hierarchy is None:
                raise ValueError(
                    "either a hierarchy or a pre-computed matrix is required"
                )
            if self._distance_fn is not None:
                self._matrix = _require_numpy().asarray(
                    self._distance_fn.matrix(), dtype=np.float64
                )
            else:
                self._matrix = _require_numpy().asarray(
                    pairwise_lca_matrix(self.hierarchy, mode=self.mode),
                    dtype=np.float64,
                )
        return self._matrix

    def distance(self, y_pred: int, y_true: int) -> float:
        """Single-pair ``D_LCA(y_pred, y_true)``."""
        y_pred = int(y_pred)
        y_true = int(y_true)
        if y_pred == y_true:
            return 0.0
        if self._matrix is None and self._distance_fn is not None:
            return float(self._distance_fn.distance(y_pred, y_true))
        return float(self.matrix[y_pred, y_true])

    # -- aggregation --------------------------------------------------------
    def dataset_lca(
        self,
        predictions: ArrayLike,
        targets: ArrayLike,
        misclassified_only: bool = True,
        normalize_by_n: bool = True,
        return_details: bool = False,
    ) -> Union[float, Tuple[float, "Any"]]:
        """``D_LCA(model, M)`` -- mean LCA distance over a dataset.

        Parameters
        ----------
        predictions:
            Predicted class indices, shape ``(n,)``.
        targets:
            Ground-truth class indices, shape ``(n,)``.
        misclassified_only:
            If ``True`` (paper definition), only samples with
            ``y_i != y_hat_i`` contribute, and the mean is over those samples.
            If ``False``, correctly classified samples contribute 0 and the mean
            is over all ``n`` samples (``lca_all`` in :class:`ModelMetrics`).
        normalize_by_n:
            When ``misclassified_only=False`` this is ignored.  When
            ``misclassified_only=True`` and ``normalize_by_n=False``, the sum is
            divided by ``n`` instead of by the number of mistakes -- which is
            exactly ``lca_all``.
        return_details:
            Also return the per-sample distance array (length ``n``).
        """
        np = _require_numpy()
        preds = np.asarray(_to_numpy(predictions), dtype=np.int64).reshape(-1)
        tgts = np.asarray(_to_numpy(targets), dtype=np.int64).reshape(-1)
        if preds.shape[0] != tgts.shape[0]:
            raise ValueError("predictions and targets must have the same length")
        n = int(preds.shape[0])
        if n == 0:
            val = float("nan")
            return (val, np.zeros((0,), dtype=np.float64)) if return_details else val

        wrong = preds != tgts
        per_sample = np.zeros((n,), dtype=np.float64)
        if wrong.any():
            # vectorized lookup: D_LCA(pred_i, true_i) for every mistake
            per_sample[wrong] = self.matrix[preds[wrong], tgts[wrong]]

        if misclassified_only and wrong.any():
            denom = float(n) if not normalize_by_n else float(wrong.sum())
            value = float(per_sample.sum() / max(denom, 1.0))
        else:
            value = float(per_sample.sum() / float(n))

        if return_details:
            return value, per_sample
        return value

    def dataset_elca(
        self,
        logits: ArrayLike,
        targets: ArrayLike,
        temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
        normalized: bool = True,
        return_details: bool = False,
    ) -> Union[float, Tuple[float, "Any"]]:
        """``D_ELCA(model, M)`` -- expected LCA distance (Section D.3).

        ``D_ELCA = (1 / nK) * sum_i sum_k p_hat_{k,i} * D_LCA(k, y_i)`` where
        ``p_hat = softmax(logits)``.  With ``normalized=False`` the ``1/K``
        factor is dropped, returning ``(1/n) sum_i sum_k p * D``.
        """
        np = _require_numpy()
        probs = softmax(logits, temperature=temperature)
        tgts = np.asarray(_to_numpy(targets), dtype=np.int64).reshape(-1)
        if probs.shape[0] != tgts.shape[0]:
            raise ValueError("logits and targets must have the same length")
        n = int(probs.shape[0])
        if n == 0:
            val = float("nan")
            zeros = np.zeros((0,), dtype=np.float64)
            return (val, zeros) if return_details else val

        # columns of the distance matrix indexed by the ground-truth labels:
        # d_gt[i, k] = D_LCA(k, y_i) = matrix[k, y_i]
        d_gt = self.matrix[:, tgts].T  # (n, K)
        per_sample = np.einsum("nk,nk->n", probs, d_gt)
        total = float(per_sample.sum())
        value = total / (float(n) * float(self.num_classes) if normalized else float(n))
        if return_details:
            return float(value), per_sample
        return float(value)

    # -- convenience --------------------------------------------------------
    def evaluate(
        self,
        logits: ArrayLike,
        targets: ArrayLike,
        name: str = "",
        dataset: str = "",
        temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
        compute_top5: bool = True,
        compute_lca: bool = True,
        compute_elca: bool = True,
    ) -> ModelMetrics:
        """Full measurement block for one model/dataset (Table 1 / Table 8 row)."""
        np = _require_numpy()
        logits = np.asarray(_to_numpy(logits), dtype=np.float64)
        tgts = np.asarray(_to_numpy(targets), dtype=np.int64).reshape(-1)
        if logits.ndim != 2:
            raise ValueError("logits must be a 2-D array of shape (n, K)")

        metrics = ModelMetrics(
            name=name,
            dataset=dataset,
            num_samples=int(logits.shape[0]),
            num_classes=int(logits.shape[1]),
        )
        preds = np.argmax(logits, axis=1)
        metrics.top1 = float(np.mean(preds == tgts)) if tgts.size else float("nan")
        if compute_top5:
            metrics.top5 = topk_accuracy(logits, tgts, k=5)

        if compute_lca:
            wrong = preds != tgts
            metrics.num_misclassified = int(wrong.sum())
            metrics.lca = self.dataset_lca(
                preds, tgts, misclassified_only=True, normalize_by_n=True
            )
            metrics.lca_all = self.dataset_lca(
                preds, tgts, misclassified_only=True, normalize_by_n=False
            )
        if compute_elca:
            metrics.elca = self.dataset_elca(
                logits, tgts, temperature=temperature, normalized=True
            )
        return metrics


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------
def _resolve_metric(
    hierarchy: Optional[WordNetHierarchy],
    lca_matrix: Optional[Sequence[Sequence[float]]],
    mode: str,
    num_classes: Optional[int],
    base: float = 2.0,
    num_leaves: Optional[int] = None,
    metric: Optional[LcaMetric] = None,
) -> LcaMetric:
    if metric is not None:
        return metric
    return LcaMetric(
        hierarchy=hierarchy,
        matrix=lca_matrix,
        mode=mode,
        num_classes=num_classes,
        base=base,
        num_leaves=num_leaves,
    )


def lca_distance_dataset(
    predictions: ArrayLike,
    targets: ArrayLike,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Sequence[Sequence[float]]] = None,
    mode: str = DEFAULT_DISTANCE_MODE,
    num_classes: Optional[int] = None,
    misclassified_only: bool = True,
    normalize_by_n: bool = True,
    return_details: bool = False,
    metric: Optional[LcaMetric] = None,
    base: float = 2.0,
    num_leaves: Optional[int] = None,
) -> Union[float, Tuple[float, "Any"]]:
    """Functional ``D_LCA(model, M)`` (Section 2).

    By default, the mean is taken over misclassified samples only, matching the
    paper's definition ``... <=> y_i != y_hat_i``.
    """
    m = _resolve_metric(hierarchy, lca_matrix, mode, num_classes, base, num_leaves, metric)
    return m.dataset_lca(
        predictions,
        targets,
        misclassified_only=misclassified_only,
        normalize_by_n=normalize_by_n,
        return_details=return_details,
    )


def elca_distance_dataset(
    logits: ArrayLike,
    targets: ArrayLike,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Sequence[Sequence[float]]] = None,
    mode: str = DEFAULT_DISTANCE_MODE,
    num_classes: Optional[int] = None,
    temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    normalized: bool = True,
    return_details: bool = False,
    metric: Optional[LcaMetric] = None,
    base: float = 2.0,
    num_leaves: Optional[int] = None,
) -> Union[float, Tuple[float, "Any"]]:
    """Functional ``D_ELCA(model, M)`` (Section D.3)."""
    m = _resolve_metric(hierarchy, lca_matrix, mode, num_classes, base, num_leaves, metric)
    return m.dataset_elca(
        logits,
        targets,
        temperature=temperature,
        normalized=normalized,
        return_details=return_details,
    )


expected_lca_distance_dataset = elca_distance_dataset  # paper's long-form name


def evaluate_model_outputs(
    logits: ArrayLike,
    targets: ArrayLike,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Sequence[Sequence[float]]] = None,
    mode: str = DEFAULT_DISTANCE_MODE,
    num_classes: Optional[int] = None,
    temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    name: str = "",
    dataset: str = "",
    compute_top5: bool = True,
    compute_lca: bool = True,
    compute_elca: bool = True,
    metric: Optional[LcaMetric] = None,
    base: float = 2.0,
    num_leaves: Optional[int] = None,
) -> ModelMetrics:
    """Compute Top-1/Top-5/LCA/ELCA for a single model on a single dataset.

    This is the routine used by ``src/eval/evaluate_models.py`` to fill one row
    of Table 1 / Table 2 / Table 8.
    """
    m = _resolve_metric(hierarchy, lca_matrix, mode, num_classes, base, num_leaves, metric)
    return m.evaluate(
        logits,
        targets,
        name=name,
        dataset=dataset,
        temperature=temperature,
        compute_top5=compute_top5,
        compute_lca=compute_lca,
        compute_elca=compute_elca,
    )
