"""Metrics for the LBCS reproduction.

This module implements the measurement layer used by every experiment driver:

* the two objectives of Refined Coreset Selection (RCS)
  - ``f1(m) = (1/n) * sum_i l(h(x_i; theta(m)), y_i)``   -- Eq. (1), primary
  - ``f2(m) = ||m||_0``                                   -- Eq. (2), secondary
* post-selection target-model measurement: top-1 test accuracy (%) and the
  achieved coreset size (Section 5.1 / 5.2 / 5.3),
* the "average accuracy brought by per data point within the selected coreset"
  reported in Appendix E.1, i.e. accuracy / coreset size,
* aggregation utilities for the paper's ``mean +- std`` reporting protocol
  (20 repeats in Section 5.1, 10 repeats in Section 5.2 / 5.3 / 6).

All functions are conservative: PyTorch is an optional (soft) dependency so
that mask-only unit validation can run without it.  Nothing in this module
implements a paper formula beyond the equations quoted above.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Labels / constants
# --------------------------------------------------------------------------- #
LBCS_LABEL = "LBCS (ours)"
SIZE_LABEL = "Coreset size (ours)"
ACCURACY_LABEL = "Test accuracy (%)"
PER_POINT_LABEL = "Accuracy per data point (%)"

#: Measurement columns produced by the comparison drivers (Tables 2 / 3).
MEASUREMENT_COLUMNS = (
    ACCURACY_LABEL,
    SIZE_LABEL,
    PER_POINT_LABEL,
    "f1 (m)",
    "f2 (m)",
)

#: Statistical reporting protocol of the paper (number of repeats per section).
PAPER_REPEATS: Dict[str, int] = {
    "section5.1": 20,
    "section5.2": 10,
    "section5.3": 10,
    "section6": 10,
}

ArrayLike = Union[np.ndarray, Sequence[float]]


# --------------------------------------------------------------------------- #
# Mask / size helpers  (f2 and derived quantities)
# --------------------------------------------------------------------------- #
def _as_numpy_mask(mask: Any) -> np.ndarray:
    """Coerce a mask (numpy, torch, list, or grouped) into a 1-D float array."""
    if mask is None:
        raise ValueError("mask must not be None")
    if _TORCH_AVAILABLE and isinstance(mask, torch.Tensor):  # pragma: no cover
        return mask.detach().cpu().numpy().reshape(-1).astype(np.float64)
    arr = np.asarray(mask, dtype=np.float64).reshape(-1)
    return arr


def binarize_mask(mask: Any, threshold: float = 0.0) -> np.ndarray:
    """Project a mask onto ``{0, 1}``.

    Follows the Appendix A rule used at the end of the LexiFlow search:
    values in ``[-1, 0)`` map to ``0`` and values in ``[0, 1]`` map to ``1``
    (i.e. the threshold is inclusive on the upper half).  Masks that are
    already binary pass through unchanged.
    """
    arr = _as_numpy_mask(mask)
    return (arr >= threshold).astype(np.float64)


def coreset_size(mask: Any) -> int:
    """``f2(m) = ||m||_0``: number of examples selected into the coreset (Eq. 2)."""
    arr = _as_numpy_mask(mask)
    return int(np.count_nonzero(arr))


#: Alias used by the drivers; identical to :func:`coreset_size` (Eq. 2).
f2_value = coreset_size


def mask_size(mask: Any) -> int:
    """Alias of :func:`coreset_size`."""
    return coreset_size(mask)


def l0_norm(mask: Any) -> int:
    """Alias of :func:`coreset_size` (L0 norm of the mask)."""
    return coreset_size(mask)


def selected_indices(mask: Any) -> np.ndarray:
    """Indices of the examples inside the coreset (``m_i = 1``)."""
    arr = _as_numpy_mask(mask)
    return np.flatnonzero(arr != 0).astype(np.int64)


def compressed_to_full_mask(indices: Any, n: int, dtype: float = np.float32) -> np.ndarray:
    """Expand an index list into a binary mask ``m in {0,1}^n``."""
    mask = np.zeros(int(n), dtype=dtype)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size:
        mask[idx] = 1
    return mask


def size_reduction(k: int, size: int) -> float:
    """Absolute reduction ``k - ||m||_0`` achieved by optimizing the size."""
    return float(int(k) - int(size))


def relative_size_reduction(k: int, size: int) -> float:
    """Relative reduction ``(k - ||m||_0) / k`` (fraction of the predefined size)."""
    k = int(k)
    if k <= 0:
        return 0.0
    return float(k - int(size)) / float(k)


# --------------------------------------------------------------------------- #
# Accuracy / loss measurement
# --------------------------------------------------------------------------- #
def _unpack_batch(batch: Any) -> Tuple[Any, Any]:
    """Return ``(inputs, targets)`` from a 2- or 3-tuple batch."""
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("batch must contain at least (inputs, targets)")
    raise TypeError("batch must be a tuple/list of (inputs, targets[, index])")


def _forward_logits(model: Any, inputs: Any) -> Any:
    """Robust forward pass returning raw logits."""
    out = model(inputs)
    logits = out
    if isinstance(out, (list, tuple)):
        logits = out[0]
    elif hasattr(out, "logits"):  # HF-style outputs
        logits = out.logits
    elif isinstance(out, dict):
        for key in ("logits", "out", "output"):
            if key in out:
                logits = out[key]
                break
    return logits


def top1_accuracy(
    model: Any,
    loader: Any,
    device: Any = None,
    max_batches: Optional[int] = None,
    return_counts: bool = False,
) -> Union[float, Tuple[float, int, int]]:
    """Top-1 accuracy (%) of ``model`` over ``loader``.

    The targets may live on any device; batches of the form ``(x, y)`` or
    ``(x, y, index)`` are both supported (the latter is what the coreset
    bookkeeping loaders emit).  Accuracy is reported in percent, matching the
    paper's tables.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for accuracy evaluation")
    if model is None or loader is None:
        raise ValueError("model and loader are required")

    was_training = getattr(model, "training", False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= int(max_batches):
                break
            inputs, targets = _unpack_batch(batch)
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            if hasattr(targets, "to"):
                targets = targets.to(device)
            if not _TORCH_AVAILABLE or not isinstance(inputs, torch.Tensor):
                inputs = torch.as_tensor(np.asarray(inputs))  # pragma: no cover
            logits = _forward_logits(model, inputs)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.shape[0])
    if was_training:
        model.train()
    acc = 100.0 * float(correct) / float(total) if total else 0.0
    if return_counts:
        return acc, correct, total
    return acc


#: Aliases used by the experiment drivers.
evaluate_accuracy = top1_accuracy
accuracy = top1_accuracy


def per_sample_cross_entropy(
    model: Any,
    loader: Any,
    device: Any = None,
    criterion: Any = None,
    max_batches: Optional[int] = None,
    return_count: bool = False,
) -> Union[float, Tuple[float, int]]:
    """Mean cross-entropy loss of ``model`` over ``loader``.

    With ``criterion=None`` the standard (mean-reduction) cross-entropy of the
    network output is used, which is the same ``l(.)`` used in Eq. (1).
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for loss evaluation")
    if model is None or loader is None:
        raise ValueError("model and loader are required")
    crit = criterion if criterion is not None else nn.CrossEntropyLoss()

    was_training = getattr(model, "training", False)
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= int(max_batches):
                break
            inputs, targets = _unpack_batch(batch)
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            if hasattr(targets, "to"):
                targets = targets.to(device)
            logits = _forward_logits(model, inputs)
            loss = crit(logits, targets)
            bs = int(targets.shape[0])
            total_loss += float(loss.item()) * bs
            total += bs
    if was_training:
        model.train()
    mean_loss = total_loss / float(total) if total else 0.0
    if return_count:
        return mean_loss, total
    return mean_loss


def f1_value(
    model: Any,
    loader: Any,
    device: Any = None,
    criterion: Any = None,
    max_batches: Optional[int] = None,
) -> float:
    """``f1(m) = (1/n) sum_i l(h(x_i; theta(m)), y_i)`` (Eq. (1)).

    ``model`` is the converged proxy network ``theta(m)`` and ``loader``
    iterates the *full* dataset ``D`` (n examples).
    """
    return float(per_sample_cross_entropy(model, loader, device=device,
                                          criterion=criterion,
                                          max_batches=max_batches))


def objective_vector(f1: float, f2: float) -> np.ndarray:
    """``F(m) = [f1(m), f2(m)]`` with lexicographic priority ``f1 > f2``."""
    return np.asarray([float(f1), float(f2)], dtype=np.float64)


def accuracy_per_datapoint(acc: float, size: int, scale: float = 1.0) -> float:
    """Average accuracy brought by per data point (Appendix E.1).

    ``scale=100`` reports it in percent-of-percent units; the default keeps the
    raw ``accuracy / coreset_size`` ratio used in the paper's figure.
    """
    size = int(size)
    if size <= 0:
        return 0.0
    return float(acc) / float(size) * float(scale)


#: Alias matching the appendix wording.
average_accuracy_per_datapoint = accuracy_per_datapoint


def per_point_accuracy(cells: Dict[Any, Any], method: str) -> Dict[Any, float]:
    """Compute per-data-point accuracy for every aggregated cell."""
    out: Dict[Any, float] = {}
    for key, cell in cells.items():
        acc_map = getattr(cell, "accuracy", None)
        if acc_map is None:
            continue
        if isinstance(acc_map, dict):
            mean_acc = acc_map.get(method)
            if isinstance(mean_acc, tuple):
                mean_acc = mean_acc[0]
            size = None
            sizes = getattr(cell, "coreset_size", None)
            if isinstance(sizes, dict):
                size = sizes.get(LBCS_LABEL, sizes.get(method))
            elif sizes is not None:
                size = sizes
            if mean_acc is not None and size:
                out[key] = accuracy_per_datapoint(float(mean_acc), int(size))
    return out


# --------------------------------------------------------------------------- #
# Statistical aggregation (mean +- std)
# --------------------------------------------------------------------------- #
def mean_std(values: ArrayLike, ddof: int = 1) -> Tuple[float, float]:
    """Return ``(mean, std)`` of ``values``; empty input yields ``(0.0, 0.0)``."""
    arr = np.asarray(list(values) if isinstance(values, (list, tuple)) else values,
                     dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=ddof)) if arr.size > 1 else 0.0
    return mean, std


def format_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    """Render ``"mean +- std"`` exactly as in the paper's tables."""
    fmt = "{:." + str(int(decimals)) + "f}"
    return fmt.format(float(mean)) + " +- " + fmt.format(float(std))


def summarize(values: ArrayLike, ddof: int = 1) -> Dict[str, Any]:
    """Return ``{"mean", "std", "n", "min", "max", "values"}`` for a metric."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mean, std = mean_std(arr, ddof=ddof)
    return {
        "mean": mean,
        "std": std,
        "n": int(arr.size),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "values": arr.tolist(),
    }


def std_error(values: ArrayLike, ddof: int = 1) -> float:
    """Standard error of the mean (useful for qualitative trend checks)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=ddof) / math.sqrt(arr.size))


def confidence_interval(values: ArrayLike, z: float = 1.96,
                        ddof: int = 1) -> Tuple[float, float]:
    """Normal-approximation confidence interval of the mean."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    se = std_error(arr, ddof=ddof)
    return mean - z * se, mean + z * se


def aggregate_records(
    records: Iterable[Dict[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
    ddof: int = 1,
) -> Dict[Tuple[Any, ...], Dict[str, Dict[str, float]]]:
    """Aggregate per-repeat records into ``{group: {metric: {mean, std, n}}}``.

    ``records`` is the list of per-repeat dictionaries written by the drivers
    (each entry holds the grouping fields plus the measured metrics).  Missing
    metrics are skipped rather than raising, so partially failed repeats still
    contribute to the remaining metrics.
    """
    grouped: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    for record in records:
        if record is None:
            continue
        key = tuple(record.get(k) for k in group_keys)
        bucket = grouped.setdefault(key, {vk: [] for vk in value_keys})
        for vk in value_keys:
            value = record.get(vk)
            if value is None:
                continue
            try:
                bucket[vk].append(float(value))
            except (TypeError, ValueError):
                continue

    out: Dict[Tuple[Any, ...], Dict[str, Dict[str, float]]] = {}
    for key, bucket in grouped.items():
        out[key] = {
            vk: summarize(vals, ddof=ddof) for vk, vals in bucket.items() if vals
        }
    return out


def mean_of(records: Iterable[Dict[str, Any]], key: str,
            default: float = 0.0) -> float:
    """Mean of ``records[*][key]`` (ignoring missing entries)."""
    vals = [float(r[key]) for r in records if r is not None and r.get(key) is not None]
    if not vals:
        return default
    return float(np.mean(vals))


def count_failures(records: Iterable[Dict[str, Any]]) -> int:
    """Number of repeats marked as failed (``record["failed"] is True``)."""
    return int(sum(1 for r in records if r is not None and r.get("failed")))


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class Measurement:
    """One measured coreset: accuracy, size, per-point accuracy and ``F(m)``."""

    method: str
    accuracy: Optional[float] = None
    coreset_size: Optional[int] = None
    f1: Optional[float] = None
    f2: Optional[float] = None
    accuracy_per_point: Optional[float] = None
    repeat: int = 0
    dataset: Optional[str] = None
    k: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accuracy_per_point is None and self.accuracy is not None \
                and self.coreset_size:
            self.accuracy_per_point = accuracy_per_datapoint(self.accuracy,
                                                             self.coreset_size)
        if self.f2 is None and self.coreset_size is not None:
            self.f2 = int(self.coreset_size)

    @property
    def F(self) -> Optional[np.ndarray]:
        """``F(m) = [f1, f2]`` when both objectives are known."""
        if self.f1 is None or self.f2 is None:
            return None
        return objective_vector(self.f1, self.f2)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "method": self.method,
            "accuracy": self.accuracy,
            "coreset_size": self.coreset_size,
            "f1": self.f1,
            "f2": self.f2,
            "accuracy_per_point": self.accuracy_per_point,
            "repeat": self.repeat,
        }
        if self.dataset is not None:
            out["dataset"] = self.dataset
        if self.k is not None:
            out["k"] = self.k
        out.update(self.extras)
        return out


@dataclass
class AggregateMeasurement:
    """``mean +- std`` of one cell (dataset, k, condition, ...) for one method."""

    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    size_mean: float = 0.0
    size_std: float = 0.0
    per_point_mean: float = 0.0
    per_point_std: float = 0.0
    f1_mean: float = 0.0
    f1_std: float = 0.0
    f2_mean: float = 0.0
    f2_std: float = 0.0
    repeats: int = 0
    failures: int = 0

    def accuracy_str(self, decimals: int = 1) -> str:
        return format_mean_std(self.accuracy_mean, self.accuracy_std, decimals)

    def size_str(self, decimals: int = 1) -> str:
        return format_mean_std(self.size_mean, self.size_std, decimals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "size_mean": self.size_mean,
            "size_std": self.size_std,
            "per_point_mean": self.per_point_mean,
            "per_point_std": self.per_point_std,
            "f1_mean": self.f1_mean,
            "f1_std": self.f1_std,
            "f2_mean": self.f2_mean,
            "f2_std": self.f2_std,
            "repeats": self.repeats,
            "failures": self.failures,
        }


def aggregate_measurements(
    measurements: Iterable[Measurement],
    accuracy_key: str = "accuracy",
    size_key: str = "coreset_size",
    ddof: int = 1,
) -> AggregateMeasurement:
    """Aggregate repeated :class:`Measurement` objects into mean +- std."""
    accs: List[float] = []
    sizes: List[float] = []
    per_point: List[float] = []
    f1s: List[float] = []
    f2s: List[float] = []
    failures = 0
    for m in measurements:
        if m is None:
            continue
        if getattr(m, "extras", None) and m.extras.get("failed"):
            failures += 1
        acc = getattr(m, accuracy_key)
        if acc is not None:
            accs.append(float(acc))
        sz = getattr(m, size_key)
        if sz is not None:
            sizes.append(float(sz))
        if m.accuracy_per_point is not None:
            per_point.append(float(m.accuracy_per_point))
        if m.f1 is not None:
            f1s.append(float(m.f1))
        if m.f2 is not None:
            f2s.append(float(m.f2))

    acc_mean, acc_std = mean_std(accs, ddof=ddof)
    size_mean, size_std = mean_std(sizes, ddof=ddof)
    pp_mean, pp_std = mean_std(per_point, ddof=ddof)
    f1_mean, f1_std = mean_std(f1s, ddof=ddof)
    f2_mean, f2_std = mean_std(f2s, ddof=ddof)
    return AggregateMeasurement(
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        size_mean=size_mean,
        size_std=size_std,
        per_point_mean=pp_mean,
        per_point_std=pp_std,
        f1_mean=f1_mean,
        f1_std=f1_std,
        f2_mean=f2_mean,
        f2_std=f2_std,
        repeats=len(accs),
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# Qualitative direction checks (paper claims)
# --------------------------------------------------------------------------- #
def decreased(initial: Optional[float], achieved: Optional[float],
              tol: float = 0.0) -> bool:
    """True when ``achieved`` is lower than ``initial`` (Section 5.1 claim)."""
    if initial is None or achieved is None:
        return False
    return float(achieved) < float(initial) - abs(float(tol))


def non_increasing(sequence: Sequence[float], tol: float = 0.0) -> bool:
    """True when ``sequence`` is (approximately) non-increasing."""
    vals = [float(v) for v in sequence]
    for prev, cur in zip(vals, vals[1:]):
        if cur > prev + abs(float(tol)):
            return False
    return True


def non_decreasing(sequence: Sequence[float], tol: float = 0.0) -> bool:
    """True when ``sequence`` is (approximately) non-decreasing."""
    vals = [float(v) for v in sequence]
    for prev, cur in zip(vals, vals[1:]):
        if cur < prev - abs(float(tol)):
            return False
    return True


def best_method(
    accuracies: Dict[str, float],
    higher_is_better: bool = True,
    exclude: Optional[Sequence[str]] = None,
) -> str:
    """Name of the best method in a ``{method: accuracy}`` mapping."""
    exclude = set(exclude or ())
    candidates = {k: v for k, v in accuracies.items()
                  if k not in exclude and v is not None}
    if not candidates:
        return ""
    if higher_is_better:
        return max(candidates.items(), key=lambda kv: float(kv[1]))[0]
    return min(candidates.items(), key=lambda kv: float(kv[1]))[0]


def rank_methods(accuracies: Dict[str, float],
                 higher_is_better: bool = True) -> List[str]:
    """Methods sorted from best to worst accuracy."""
    items = [(k, v) for k, v in accuracies.items() if v is not None]
    items.sort(key=lambda kv: float(kv[1]), reverse=higher_is_better)
    return [k for k, _ in items]


def accuracy_gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """``a - b`` when both are available (used for size-matched comparisons)."""
    if a is None or b is None:
        return None
    return float(a) - float(b)


# --------------------------------------------------------------------------- #
# Table rendering helpers
# --------------------------------------------------------------------------- #
def render_table(headers: Sequence[str],
                 rows: Sequence[Sequence[Any]],
                 separator: str = " | ",
                 ) -> str:
    """Render a plain-text (Markdown-ish) table for logging / artifacts."""
    str_rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    def fmt(row: Sequence[str]) -> str:
        return separator.join(cell.ljust(widths[i]) for i, cell in enumerate(row))
    lines = [fmt([str(h) for h in headers])]
    lines.append("-" * len(lines[0]))
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def records_to_rows(
    records: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
    decimals: int = 1,
) -> Tuple[List[str], List[List[str]]]:
    """Aggregate records and render them as ``(headers, rows)``."""
    agg = aggregate_records(records, group_keys, value_keys)
    headers = list(group_keys)
    for vk in value_keys:
        headers.append(vk)
    rows: List[List[str]] = []
    for key in sorted(agg.keys(), key=lambda t: tuple(str(x) for x in t)):
        row: List[str] = [str(v) for v in key]
        for vk in value_keys:
            stats = agg[key].get(vk)
            if not stats:
                row.append("-")
            else:
                row.append(format_mean_std(stats["mean"], stats["std"], decimals))
        rows.append(row)
    return headers, rows


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
class _TinyModel:
    """Minimal stand-in exposing ``__call__`` for a torch-free smoke test."""

    def __init__(self, logits: np.ndarray, targets: np.ndarray) -> None:
        self.logits = logits
        self.targets = targets
        self.training = True

    def __call__(self, inputs):  # pragma: no cover - not used without torch
        return self.logits

    def eval(self) -> "_TinyModel":
        self.training = False
        return self

    def train(self) -> "_TinyModel":
        self.training = True
        return self


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of mask/size/statistics helpers (no torch required)."""
    report: Dict[str, Any] = {}

    mask = np.array([1, 0, 1, 1, 0], dtype=np.float32)
    report["size"] = coreset_size(mask)
    assert report["size"] == 3, "||m||_0 must equal 3"
    assert mask_size(mask) == 3 and f2_value(mask) == 3
    assert selected_indices(mask).tolist() == [0, 2, 3]

    # Appendix A projection: [-1,0) -> 0, [0,1] -> 1
    relaxed = np.array([-1.0, -1e-12, 0.0, 1e-12, 1.0], dtype=np.float64)
    proj = binarize_mask(relaxed)
    report["projection"] = proj.tolist()
    assert proj.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0], "projection rule mismatch"

    full = compressed_to_full_mask([1, 3], n=5)
    assert coreset_size(full) == 2 and full[1] == 1 and full[3] == 1

    mean, std = mean_std([1.0, 2.0, 3.0])
    report["mean_std"] = (mean, std)
    assert abs(mean - 2.0) < 1e-12 and abs(std - 1.0) < 1e-12
    assert format_mean_std(1.92, 0.33) == "1.92 +- 0.33"
    assert mean_std([]) == (0.0, 0.0)

    stats = summarize([1.0, 2.0, 3.0])
    assert stats["n"] == 3 and abs(stats["min"] - 1.0) < 1e-12
    report["summarize"] = {k: v for k, v in stats.items() if k != "values"}

    report["per_point"] = accuracy_per_datapoint(79.7, 956)
    assert abs(report["per_point"] - 79.7 / 956.0) < 1e-12

    assert abs(relative_size_reduction(1000, 956) - 0.044) < 1e-9
    assert size_reduction(1000, 956) == 44

    agg = aggregate_records(
        [
            {"dataset": "SVHN", "k": 1000, "accuracy": 70.6, "coreset_size": 970},
            {"dataset": "SVHN", "k": 1000, "accuracy": 70.4, "coreset_size": 974},
            {"dataset": "SVHN", "k": 2000, "accuracy": 78.3, "coreset_size": 1902},
        ],
        group_keys=("dataset", "k"),
        value_keys=("accuracy", "coreset_size"),
    )
    report["groups"] = len(agg)
    assert len(agg) == 2
    acc_stats = agg[("SVHN", 1000)]["accuracy"]
    assert abs(acc_stats["mean"] - 70.5) < 1e-9 and acc_stats["n"] == 2

    headers, rows = records_to_rows(
        [{"dataset": "SVHN", "k": 1000, "accuracy": 70.6}],
        group_keys=("dataset", "k"), value_keys=("accuracy",),
    )
    assert headers == ["dataset", "k", "accuracy"] and len(rows) == 1

    assert decreased(3.21, 1.92) and not decreased(1.92, 3.21)
    assert non_increasing([1.92, 2.26, 2.48]) is False
    assert non_decreasing([1.92, 2.26, 2.48]) is True
    assert non_increasing([200.0, 190.7, 185.0]) is True

    assert best_method({"a": 79.7, "b": 80.3}) == "b"
    assert rank_methods({"a": 79.7, "b": 80.3}) == ["b", "a"]
    assert accuracy_gap(80.3, 79.7) == 80.3 - 79.7

    m = Measurement(method="LBCS", accuracy=79.7, coreset_size=956)
    report["measurement"] = m.to_dict()
    assert m.f2 == 956 and abs(m.accuracy_per_point - 79.7 / 956.0) < 1e-12
    assert m.F is not None or m.f1 is None

    aggregate = aggregate_measurements([m, Measurement("LBCS", 79.5, 958)])
    assert aggregate.repeats == 2
    assert abs(aggregate.accuracy_mean - 79.6) < 1e-9
    assert aggregate.accuracy_str() == "79.60 +- 0.14"

    if _TORCH_AVAILABLE:  # pragma: no cover - torch path
        import torch as _torch

        class _Net(_torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = _torch.nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        loader = [(_torch.randn(8, 4), _torch.zeros(8, dtype=_torch.long))]
        acc = top1_accuracy(_Net(), loader)
        assert 0.0 <= acc <= 100.0
        report["torch_accuracy"] = acc
    report["ok"] = True
    if verbose:
        print("utils.metrics self-test:", report)
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest()
