"""Evaluation metrics for the FOA reproduction (Section 4, "Evaluation Metrics").

Paper specification
-------------------
The paper (Section 4, *Evaluation Metrics*) reports two numbers:

1. **Classification Accuracy (%)**, higher-is-better, computed on the OOD test
   samples (ImageNet-C / R / V2 / Sketch).
2. **Expected Calibration Error (ECE) (%)**, lower-is-better, which "measures the
   difference between predicted probabilities and actual outcomes in a
   probabilistic model (Naeini et al., 2015)".

The paper reports ECE per corruption and the average over the 15 corruptions of
ImageNet-C (Tables 16 and 17) as well as a single average in Tables 2 and 3.

Documented ambiguity (plan item (d))
------------------------------------
The paper cites Naeini et al. (2015) without fixing a bin count and without
stating whether the bins are equal-width or equal-mass.  Because the standard
practice (and the ECE of Guo et al. 2017) is equal-width probability bins, we
use **15 equal-width bins** over [0, 1] as the default and record the choice in
the returned metadata.  ``n_bins`` is a parameter everywhere so a sweep can be
performed without touching this file.

ECE definition used here (standard, equal-width):

    ECE = sum_{m=1}^{M} (|B_m| / n) * | acc(B_m) - conf(B_m) |

where ``B_m`` are the samples whose top-1 confidence falls in bin ``m``,
``acc(B_m)`` is the fraction of correct predictions in the bin and
``conf(B_m)`` the mean predicted confidence of the bin.  The last bin is
closed on the right so that a confidence of exactly 1.0 is counted.

Everything in this module is pure NumPy/PyTorch post-processing on already
computed logits -- it never touches the model and never enables gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is an optional convenience for tensor inputs
    import torch
except Exception:  # pragma: no cover - torch is a hard requirement in practice
    torch = None  # type: ignore


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Default number of equal-width probability bins used for the ECE.
DEFAULT_ECE_BINS = 15

#: The 15 standard ImageNet-C corruptions (order used by Tables 16/17).
IMAGENET_C_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

#: The four corruption families used in the Tables 16/17 group headers.
CORRUPTION_GROUPS: Dict[str, List[str]] = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

#: Paper reference numbers (ViT-Base, ImageNet-C severity level 5) used by the
#: scripts for a printed comparison. Kept here so runners do not duplicate them.
PAPER_REFERENCES: Dict[str, Dict[str, float]] = {
    # Table 2 / Table 16 -- full precision, average over 15 corruptions.
    "foa_full_precision": {"accuracy": 66.3, "ece": 3.2},
    "no_adapt_full_precision": {"accuracy": 55.5, "ece": 10.5},
    "lame": {"accuracy": 54.1, "ece": 11.0},
    "t3a": {"accuracy": 56.9, "ece": 26.8},
    "tent": {"accuracy": 59.6, "ece": 18.5},
    "cotta": {"accuracy": 61.7, "ece": 6.5},
    "sar": {"accuracy": 62.7, "ece": 7.0},
    # Table 4 / Table 17 -- quantized, average over 15 corruptions.
    "foa_8bit": {"accuracy": 63.5, "ece": 3.8},
    "foa_6bit": {"accuracy": 55.8, "ece": 5.5},
    # Table 2 -- single corruption (Gaussian noise, severity 5) sanity anchor.
    "no_adapt_gaussian": {"accuracy": 56.8, "ece": 7.5},
    # Table 3 -- ImageNet-R / V2 / Sketch.
    "foa_imagenet_r": {"accuracy": 63.8, "ece": None},
    "foa_imagenet_v2": {"accuracy": 75.4, "ece": None},
    "foa_imagenet_sketch": {"accuracy": 49.9, "ece": None},
}


# --------------------------------------------------------------------------- #
# Small conversion helpers
# --------------------------------------------------------------------------- #

def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor / list / ndarray to a detached NumPy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def _as_float64(x: Any) -> np.ndarray:
    return np.asarray(_to_numpy(x), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #

def top1_predictions(logits: Any) -> np.ndarray:
    """Return the arg-max (top-1) class index for each row of ``logits``.

    Args:
        logits: ``[N, C]`` or ``[C]`` array-like of (unnormalised) scores.

    Returns:
        Integer array of shape ``[N]``.
    """
    arr = _as_float64(logits)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"logits must be 1D or 2D, got shape {arr.shape}")
    return np.argmax(arr, axis=1)


def compute_accuracy(
    logits: Any,
    targets: Any,
    ignore_index: Optional[int] = None,
) -> float:
    """Top-1 classification accuracy **in percent**.

    Matching Table 2/3/16/17 which report accuracy as a percentage.

    Args:
        logits: ``[N, C]`` scores (or ``[N]`` already-computed predictions).
        targets: ``[N]`` integer ground-truth labels.
        ignore_index: optional label value to exclude from the average.

    Returns:
        Accuracy in percent (``0..100``).
    """
    preds = top1_predictions(logits) if np.asarray(_to_numpy(logits)).ndim >= 2 else _to_numpy(logits).astype(np.int64).ravel()
    tgt = _to_numpy(targets).astype(np.int64).ravel()

    if preds.shape[0] != tgt.shape[0]:
        raise ValueError(
            f"prediction/target length mismatch: {preds.shape[0]} vs {tgt.shape[0]}"
        )
    if tgt.size == 0:
        return float("nan")

    correct = preds == tgt
    if ignore_index is not None:
        mask = tgt != ignore_index
        if not np.any(mask):
            return float("nan")
        correct = correct[mask]

    return float(100.0 * np.mean(correct))


def accuracy_top1(logits: Any, targets: Any) -> float:
    """Alias of :func:`compute_accuracy` (fraction of correctly classified)."""
    return compute_accuracy(logits, targets)


def top1_accuracy(logits: Any, targets: Any) -> float:
    """Alias of :func:`compute_accuracy` (percent)."""
    return compute_accuracy(logits, targets)


def accuracy(logits: Any, targets: Any) -> float:
    """Alias of :func:`compute_accuracy` (percent)."""
    return compute_accuracy(logits, targets)


def compute_topk_accuracy(logits: Any, targets: Any, k: int = 5) -> float:
    """Top-k accuracy in percent (not used by the paper, provided for sanity)."""
    arr = _as_float64(logits)
    if arr.ndim == 1:
        arr = arr[None, :]
    tgt = _to_numpy(targets).astype(np.int64).ravel()
    if tgt.size == 0:
        return float("nan")
    k = int(min(max(k, 1), arr.shape[1]))
    topk = np.argpartition(-arr, kth=k - 1, axis=1)[:, :k]
    hit = (topk == tgt[:, None]).any(axis=1)
    return float(100.0 * np.mean(hit))


# --------------------------------------------------------------------------- #
# Expected Calibration Error
# --------------------------------------------------------------------------- #

def softmax(logits: Any, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over the last axis (returns float64)."""
    arr = _as_float64(logits)
    if arr.ndim == 1:
        arr = arr[None, :]
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def confidences_and_correct(
    logits: Any,
    targets: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-sample ``(top1_confidence, correct_flag)``.

    ``top1_confidence`` is the maximum softmax probability; ``correct_flag`` is a
    boolean array.
    """
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    conf = probs[np.arange(preds.shape[0]), preds]
    tgt = _to_numpy(targets).astype(np.int64).ravel()
    if tgt.shape[0] != preds.shape[0]:
        raise ValueError(
            f"prediction/target length mismatch: {preds.shape[0]} vs {tgt.shape[0]}"
        )
    return conf.astype(np.float64), (preds == tgt)


def bin_indices(confidences: Any, n_bins: int = DEFAULT_ECE_BINS) -> np.ndarray:
    """Map confidences in [0, 1] to equal-width bin indices ``0..n_bins-1``.

    The highest bin is closed on the right, so a confidence of exactly 1.0 lands
    in the last bin (mirrors ``np.digitize`` with the boundary shift below).
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    bins = np.floor(conf * n_bins).astype(np.int64)
    # confidence == 1.0 -> index n_bins (out of range); clamp into the last bin.
    np.clip(bins, 0, n_bins - 1, out=bins)
    return bins


def compute_ece(
    logits: Any = None,
    targets: Any = None,
    n_bins: int = DEFAULT_ECE_BINS,
    confidences: Any = None,
    correct: Any = None,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Expected Calibration Error in percent (equal-width bins).

    Can be called either with ``logits``/``targets`` or with pre-computed
    ``confidences``/``correct`` (used by the streaming accumulator so a forward
    pass is only done once).

    Args:
        logits: ``[N, C]`` scores; ignored if ``confidences`` is given.
        targets: ``[N]`` integer labels; ignored if ``correct`` is given.
        n_bins: number of equal-width probability bins (default 15, documented
            ambiguity (d) of the reproduction plan).
        confidences: optional ``[N]`` array of top-1 probabilities.
        correct: optional ``[N]`` boolean array of correctness.
        return_details: if True also return the per-bin decomposition.

    Returns:
        ECE in percent (``0..100``), or ``(ece, details)`` when
        ``return_details`` is True.  ``nan`` for an empty input.
    """
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")

    if confidences is None or correct is None:
        if logits is None or targets is None:
            raise ValueError("provide either (logits, targets) or (confidences, correct)")
        confidences, correct = confidences_and_correct(logits, targets)

    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()

    if conf.shape[0] != corr.shape[0]:
        raise ValueError(
            f"confidence/correct length mismatch: {conf.shape[0]} vs {corr.shape[0]}"
        )
    n = conf.shape[0]
    if n == 0:
        return (float("nan"), {}) if return_details else float("nan")

    idx = bin_indices(conf, n_bins)

    bin_counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    bin_conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    bin_correct_sum = np.bincount(idx, weights=corr, minlength=n_bins)

    non_empty = bin_counts > 0
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf[non_empty] = bin_conf_sum[non_empty] / bin_counts[non_empty]
    bin_acc[non_empty] = bin_correct_sum[non_empty] / bin_counts[non_empty]

    gaps = np.abs(bin_acc - bin_conf)
    weights = bin_counts / float(n)
    ece_fraction = float(np.sum(weights * gaps))
    ece_percent = 100.0 * ece_fraction

    if not return_details:
        return ece_percent

    details: Dict[str, Any] = {
        "n_bins": n_bins,
        "binning": "equal_width",
        "num_samples": int(n),
        "bin_counts": bin_counts.tolist(),
        "bin_accuracy": bin_acc.tolist(),
        "bin_confidence": bin_conf.tolist(),
        "bin_gap": gaps.tolist(),
        "ece_fraction": ece_fraction,
        "ece_percent": ece_percent,
    }
    return ece_percent, details


def expected_calibration_error(
    logits: Any,
    targets: Any,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Alias of :func:`compute_ece` (percent, equal-width bins)."""
    return float(compute_ece(logits=logits, targets=targets, n_bins=n_bins))


def ece(logits: Any, targets: Any, n_bins: int = DEFAULT_ECE_BINS) -> float:
    """Alias of :func:`compute_ece` (percent, equal-width bins)."""
    return float(compute_ece(logits=logits, targets=targets, n_bins=n_bins))


def negative_log_likelihood(logits: Any, targets: Any) -> float:
    """Mean NLL (natural log) -- auxiliary metric, not reported by the paper."""
    probs = softmax(logits)
    tgt = _to_numpy(targets).astype(np.int64).ravel()
    if tgt.size == 0:
        return float("nan")
    eps = 1e-12
    p_true = np.clip(probs[np.arange(tgt.shape[0]), tgt], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


# --------------------------------------------------------------------------- #
# Streaming accumulation (one forward pass, online TTA friendly)
# --------------------------------------------------------------------------- #

@dataclass
class MetricAccumulator:
    """Streaming accuracy + ECE accumulator for online test-time adaptation.

    The FOA loop is single-pass over an ordered test stream, so every batch must
    contribute to the final metric exactly once.  This class stores the
    per-sample confidences/correctness (cheap: one float + one bool per sample)
    and computes the metrics once at the end, which is mathematically identical
    to the batch-wise computation of :func:`compute_ece`.

    Attributes:
        n_bins: number of equal-width ECE bins (default 15).
        num_samples: number of samples seen so far.
        num_correct: number of correctly classified samples so far.
        per_corruption: mapping ``corruption -> {"correct": int, "total": int}``
            used for the per-corruption tables (Tables 16/17).
    """

    n_bins: int = DEFAULT_ECE_BINS
    num_samples: int = 0
    num_correct: int = 0
    _confidences: List[np.ndarray] = field(default_factory=list, repr=False)
    _correct: List[np.ndarray] = field(default_factory=list, repr=False)
    per_batch: List[Dict[str, float]] = field(default_factory=list, repr=False)
    per_corruption: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Any = None, n_bins: Optional[int] = None) -> "MetricAccumulator":
        """Build from a config object/dict using ``eval.ece_bins``."""
        if n_bins is None:
            n_bins = _cfg_get(cfg, "eval", "ece_bins", default=DEFAULT_ECE_BINS)
        return cls(n_bins=int(n_bins))

    # -- accumulation ------------------------------------------------------ #
    def update(
        self,
        logits: Any,
        targets: Any,
        corruption: Optional[str] = None,
    ) -> Dict[str, float]:
        """Accumulate one batch; returns the batch-level ``{accuracy, ece}``."""
        conf, corr = confidences_and_correct(logits, targets)
        self._confidences.append(conf)
        self._correct.append(corr)
        n = int(conf.shape[0])
        n_correct = int(np.sum(corr))
        self.num_samples += n
        self.num_correct += n_correct

        if corruption is not None:
            entry = self.per_corruption.setdefault(
                corruption, {"correct": 0, "total": 0, "confidences": [], "correct_flags": []}
            )
            entry["correct"] += n_correct
            entry["total"] += n
            entry["confidences"].append(conf)
            entry["correct_flags"].append(corr)

        batch_acc = float(100.0 * np.mean(corr)) if n else float("nan")
        batch_ece = float(compute_ece(confidences=conf, correct=corr, n_bins=self.n_bins)) if n else float("nan")
        record = {"accuracy": batch_acc, "ece": batch_ece, "num_samples": n}
        self.per_batch.append(record)
        return record

    def update_predictions(
        self,
        predictions: Any,
        confidences: Any,
        targets: Any,
        corruption: Optional[str] = None,
    ) -> Dict[str, float]:
        """Accumulate from already-computed predictions/confidences (no softmax)."""
        preds = _to_numpy(predictions).astype(np.int64).ravel()
        conf = np.asarray(confidences, dtype=np.float64).ravel()
        tgt = _to_numpy(targets).astype(np.int64).ravel()
        if not (preds.shape[0] == conf.shape[0] == tgt.shape[0]):
            raise ValueError("predictions/confidences/targets must have equal length")
        corr = preds == tgt
        self._confidences.append(conf)
        self._correct.append(corr)
        n = int(conf.shape[0])
        n_correct = int(np.sum(corr))
        self.num_samples += n
        self.num_correct += n_correct
        if corruption is not None:
            entry = self.per_corruption.setdefault(
                corruption, {"correct": 0, "total": 0, "confidences": [], "correct_flags": []}
            )
            entry["correct"] += n_correct
            entry["total"] += n
            entry["confidences"].append(conf)
            entry["correct_flags"].append(corr)
        batch_acc = float(100.0 * np.mean(corr)) if n else float("nan")
        batch_ece = float(compute_ece(confidences=conf, correct=corr, n_bins=self.n_bins)) if n else float("nan")
        record = {"accuracy": batch_acc, "ece": batch_ece, "num_samples": n}
        self.per_batch.append(record)
        return record

    # -- results ----------------------------------------------------------- #
    def compute(self) -> Dict[str, Any]:
        """Return ``{accuracy, ece, num_samples, n_bins}`` over everything seen."""
        if self.num_samples == 0:
            return {
                "accuracy": float("nan"),
                "ece": float("nan"),
                "num_samples": 0,
                "n_bins": self.n_bins,
            }
        conf = np.concatenate(self._confidences)
        corr = np.concatenate(self._correct)
        return {
            "accuracy": float(100.0 * np.mean(corr)),
            "ece": float(compute_ece(confidences=conf, correct=corr, n_bins=self.n_bins)),
            "num_samples": int(self.num_samples),
            "n_bins": self.n_bins,
        }

    def compute_per_corruption(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{corruption: {accuracy, ece, num_samples}}`` (Tables 16/17)."""
        out: Dict[str, Dict[str, Any]] = {}
        for name, entry in self.per_corruption.items():
            total = int(entry.get("total", 0))
            if total == 0:
                out[name] = {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
                continue
            conf = np.concatenate(entry["confidences"]) if entry.get("confidences") else np.zeros(0)
            corr = np.concatenate(entry["correct_flags"]) if entry.get("correct_flags") else np.zeros(0, dtype=bool)
            out[name] = {
                "accuracy": float(100.0 * entry["correct"] / total),
                "ece": float(compute_ece(confidences=conf, correct=corr, n_bins=self.n_bins))
                if conf.size
                else float("nan"),
                "num_samples": total,
            }
        return out

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.num_samples = 0
        self.num_correct = 0
        self._confidences.clear()
        self._correct.clear()
        self.per_batch.clear()
        self.per_corruption.clear()

    # Convenience mirrors so callers can treat this like the metric functions
    def accuracy(self) -> float:
        return float(self.compute()["accuracy"])

    def ece(self) -> float:
        return float(self.compute()["ece"])


# --------------------------------------------------------------------------- #
# Aggregation across corruptions / benchmarks
# --------------------------------------------------------------------------- #

def average_over_corruptions(
    per_corruption: Mapping[str, Mapping[str, float]],
    corruptions: Optional[Sequence[str]] = None,
    metric: str = "accuracy",
) -> float:
    """Average a metric over the 15 ImageNet-C corruptions.

    Missing corruptions are skipped (their ``nan`` values ignored).  If a
    corruption is present but its value is ``nan`` it is skipped as well.
    """
    names: Iterable[str] = corruptions if corruptions is not None else list(per_corruption.keys())
    values = [
        float(per_corruption[name][metric])
        for name in names
        if name in per_corruption
        and per_corruption[name].get(metric) is not None
        and not np.isnan(float(per_corruption[name][metric]))
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize_corruptions(
    per_corruption: Mapping[str, Mapping[str, float]],
    corruptions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the Table 2/16 style summary: mean accuracy and mean ECE.

    Returns a dict with ``accuracy``/``ece`` means plus the per-corruption table
    and the four family sub-averages (noise/blur/weather/digital).
    """
    names = list(corruptions) if corruptions is not None else list(per_corruption.keys())
    summary: Dict[str, Any] = {
        "accuracy": average_over_corruptions(per_corruption, names, "accuracy"),
        "ece": average_over_corruptions(per_corruption, names, "ece"),
        "num_corruptions": len([n for n in names if n in per_corruption]),
        "per_corruption": {n: dict(per_corruption[n]) for n in names if n in per_corruption},
    }
    groups: Dict[str, Dict[str, float]] = {}
    for group, members in CORRUPTION_GROUPS.items():
        present = [m for m in members if m in per_corruption]
        if present:
            groups[group] = {
                "accuracy": average_over_corruptions(per_corruption, present, "accuracy"),
                "ece": average_over_corruptions(per_corruption, present, "ece"),
            }
    summary["groups"] = groups
    return summary


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and (population) standard deviation of a sequence, ignoring NaNs."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


# --------------------------------------------------------------------------- #
# Config helper
# --------------------------------------------------------------------------- #

def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup supporting both ``Config``/dict and attribute objects."""
    if cfg is None:
        return default
    node = cfg
    for key in keys:
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(key, None)
        else:
            node = getattr(node, key, None)
    return default if node is None else node


__all__ = [
    "DEFAULT_ECE_BINS",
    "IMAGENET_C_CORRUPTIONS",
    "CORRUPTION_GROUPS",
    "PAPER_REFERENCES",
    "top1_predictions",
    "compute_accuracy",
    "accuracy_top1",
    "top1_accuracy",
    "accuracy",
    "compute_topk_accuracy",
    "softmax",
    "confidences_and_correct",
    "bin_indices",
    "compute_ece",
    "expected_calibration_error",
    "ece",
    "negative_log_likelihood",
    "MetricAccumulator",
    "average_over_corruptions",
    "summarize_corruptions",
    "mean_std",
]
