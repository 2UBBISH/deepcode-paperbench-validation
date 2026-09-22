"""Evaluation metrics for SMM visual reprogramming experiments.

The paper reports **top-1 test accuracy** (mean % over three seeds, with std),
see Sec. 5 ("Experiments are run with three seeds on a single A100 GPU and the
averaged test accuracy is reported").  The paper also reports "Mean % +- Std %"
in the ablation table (Table 3) and average rows per table (e.g. 52.53 for SMM
with ResNet-18, 72.4 for SMM with ViT-B32).

This module therefore provides:

* top-1 (and top-k) accuracy computed either from ``(predicted, target)`` index
  tensors or directly from model logits;
* a streaming accumulator (:class:`AccuracyMeter`) so test loops can avoid
  holding every prediction in memory;
* helpers to aggregate the per-seed results into ``mean +- std`` exactly the way
  the paper's tables are written;
* reporting helpers that render paper-style tables and compare measured
  averages against the values printed in the paper.

Nothing here is learnable; it is pure bookkeeping and glue code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    # accuracy primitives
    "accuracy",
    "top1_accuracy",
    "topk_accuracy",
    "AccuracyMeter",
    "MetricTracker",
    # per-run bookkeeping
    "RunResult",
    "aggregate_seeds",
    "aggregate_results",
    "mean_over_datasets",
    "format_mean_std",
    "format_results_table",
    # comparison with the paper
    "compare_with_reference",
    "compare_run_to_reference",
    "resolve_reference_table",
    "dump_results_json",
    "REFERENCE_TABLES",
    "TABLE1_RESNET18",
    "TABLE1_RESNET50",
    "TABLE2_VIT_B32",
    "TABLE3_ABLATIONS",
]


# ---------------------------------------------------------------------------
# Reference tables (mean accuracy in %).  Kept locally so this module is usable
# even without ``smm_vr.data.dataset_stats``; ``resolve_reference_table``
# prefers the richer tables from that module when it is importable.
# ---------------------------------------------------------------------------

TABLE1_RESNET18 = {
    "cifar10": 72.8,
    "cifar100": 39.4,
    "svhn": 84.4,
    "gtsrb": 80.4,
    "flowers102": 38.7,
    "dtd": 33.6,
    "ucf101": 28.7,
    "food101": 17.5,
    "sun397": 16.0,
    "eurosat": 92.2,
    "oxfordpets": 74.1,
    "average": 52.53,
}

TABLE1_RESNET50 = {
    "cifar10": 76.1,
    "cifar100": 43.4,
    "svhn": 87.4,
    "gtsrb": 83.9,
    "flowers102": 42.4,
    "dtd": 36.9,
    "ucf101": 31.9,
    "food101": 19.8,
    "sun397": 18.2,
    "eurosat": 94.3,
    "oxfordpets": 32.4,
    "average": 56.35,
}

TABLE2_VIT_B32 = {
    "cifar10": 97.4,
    "cifar100": 82.6,
    "svhn": 89.7,
    "gtsrb": 80.5,
    "flowers102": 79.1,
    "dtd": 45.6,
    "ucf101": 42.6,
    "food101": 64.8,
    "sun397": 36.7,
    "eurosat": 93.5,
    "oxfordpets": 83.8,
    "average": 72.4,
}

TABLE3_ABLATIONS = {
    "ours": 52.53,
    "single_channel_fmask": 49.70,
    "only_delta": 46.85,
    "only_fmask": 42.59,
}

REFERENCE_TABLES: Dict[str, Any] = {
    "table1_resnet18": {"ours": TABLE1_RESNET18},
    "table1_resnet50": {"ours": TABLE1_RESNET50},
    "table2_vit_b32": {"ours": TABLE2_VIT_B32},
    "table3_ablations": TABLE3_ABLATIONS,
}

# Tolerances used by :func:`compare_with_reference` (absolute accuracy % points).
DEFAULT_AVERAGE_TOLERANCE = 2.0
DEFAULT_PER_DATASET_TOLERANCE = 5.0

_TABLE_ALIASES = {
    "table1": "table1_resnet18",
    "table1resnet": "table1_resnet18",
    "resnet18": "table1_resnet18",
    "resnet50": "table1_resnet50",
    "table2": "table2_vit_b32",
    "vit": "table2_vit_b32",
    "vitb32": "table2_vit_b32",
    "table3": "table3_ablations",
    "ablations": "table3_ablations",
}


def _canonical(name: Any) -> str:
    """Normalise a dataset/table name to the keys used in this module."""
    if not isinstance(name, str):
        return str(name)
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_reference_table(table: str) -> Any:
    """Look up a reference table by name (tolerates aliases and spellings).

    Prefers ``smm_vr.data.dataset_stats`` when importable (it holds the full set
    of paper tables), and otherwise falls back to the tables defined here.
    """
    key = _TABLE_ALIASES.get(_canonical(table), _canonical(table))
    try:  # pragma: no cover - optional dependency
        from ..data import dataset_stats as _ds  # type: ignore

        ref = getattr(_ds, "REFERENCE_TABLES", None)
        if isinstance(ref, Mapping):
            for cand in (key, table):
                if cand in ref:
                    return ref[cand]
            norm = {_canonical(k): v for k, v in ref.items()}
            if key in norm:
                return norm[key]
    except Exception:
        pass
    if key not in REFERENCE_TABLES:
        raise KeyError(
            f"Unknown reference table {table!r}. Available: {sorted(REFERENCE_TABLES)}"
        )
    return REFERENCE_TABLES[key]


# ---------------------------------------------------------------------------
# Accuracy primitives
# ---------------------------------------------------------------------------


def _as_long_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """Coerce indices/logits/sequences into a flat ``long`` tensor."""
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.dim() > 1:  # e.g. one-hot labels or logits passed by mistake
            t = t.argmax(dim=-1)
        t = t.reshape(-1).to(torch.long)
    else:
        t = torch.as_tensor(list(x), dtype=torch.long).reshape(-1)
    if device is not None:
        t = t.to(device)
    return t


def accuracy(
    predicted: Any,
    target: Any,
    ignore_index: Optional[int] = None,
) -> float:
    """Top-1 accuracy as a fraction in ``[0, 1]``.

    ``predicted``/``target`` may be tensors or sequences of ints.  If
    ``predicted`` has more than one dimension it is treated as logits and the
    argmax over the last dimension is taken.
    """
    pred = _as_long_tensor(predicted)
    tgt = _as_long_tensor(target, device=pred.device)
    if ignore_index is not None:
        keep = tgt != ignore_index
        pred, tgt = pred[keep], tgt[keep]
    if tgt.numel() == 0:
        return 0.0
    return float((pred == tgt).sum().item()) / float(tgt.numel())


# Paper-facing alias: top-1 accuracy.
top1_accuracy = accuracy


def topk_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    k: int = 5,
    ignore_index: Optional[int] = None,
) -> float:
    """Top-k accuracy as a fraction in ``[0, 1]`` (logits or decoded indices)."""
    tgt = _as_long_tensor(target)
    if isinstance(predicted, torch.Tensor) and predicted.dim() >= 2:
        logits = predicted.detach()
        if logits.dim() > 2:
            logits = logits.reshape(logits.shape[0], -1)
        kk = min(int(k), logits.shape[1])
        top = logits.topk(kk, dim=1).indices
        hit = (top == tgt.unsqueeze(1).to(top.device)).any(dim=1)
    else:  # already-decoded predictions: only top-1 is meaningful
        pred = _as_long_tensor(predicted)
        hit = pred.to(tgt.device) == tgt
    if ignore_index is not None:
        hit = hit[tgt != ignore_index]
    if hit.numel() == 0:
        return 0.0
    return float(hit.sum().item()) / float(hit.numel())


class AccuracyMeter:
    """Streaming top-1 (and top-k) accuracy accumulator.

    Example::

        meter = AccuracyMeter()
        with torch.no_grad():
            for images, targets in loader:
                logits = model(f_in(images))
                meter.update(logits, targets)
        meter.accuracy  # percent, as reported in the paper tables
    """

    def __init__(self, topk: Sequence[int] = (1,), ignore_index: Optional[int] = None):
        self.topk = tuple(int(k) for k in topk)
        self.ignore_index = ignore_index
        self.num_correct: Dict[int, int] = {k: 0 for k in self.topk}
        self.num_samples = 0
        self.num_batches = 0

    def reset(self) -> None:
        self.num_correct = {k: 0 for k in self.topk}
        self.num_samples = 0
        self.num_batches = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.  ``logits`` are pre-softmax scores ``(B, C)``."""
        tgt = _as_long_tensor(target)
        scores = logits.detach()
        if scores.dim() > 2:
            scores = scores.reshape(scores.shape[0], -1)
        if self.ignore_index is not None:
            keep = (tgt != self.ignore_index).to(scores.device)
            scores, tgt = scores[keep], tgt[keep.to(tgt.device)]
        if tgt.numel() == 0:
            return
        tgt = tgt.to(scores.device)
        for k in self.topk:
            kk = min(k, scores.shape[1])
            top = scores.topk(kk, dim=1).indices
            hits = (top == tgt.unsqueeze(1)).any(dim=1)
            self.num_correct[k] += int(hits.sum().item())
        self.num_samples += int(tgt.numel())
        self.num_batches += 1

    # -- properties -------------------------------------------------------
    def get(self, k: int = 1) -> float:
        """Accuracy for a given ``k`` in percent (paper convention)."""
        if self.num_samples == 0:
            return 0.0
        return 100.0 * float(self.num_correct.get(k, 0)) / float(self.num_samples)

    @property
    def accuracy(self) -> float:
        """Top-1 accuracy in percent (paper convention)."""
        return self.get(1)

    # Common aliases used elsewhere in the codebase.
    top1 = accuracy
    acc1 = accuracy

    @property
    def error(self) -> float:
        return 100.0 - self.get(1)

    @property
    def topk_values(self) -> Dict[int, float]:
        return {k: self.get(k) for k in self.topk}

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, Any] = {"num_samples": self.num_samples, "acc@1": self.get(1)}
        for k in self.topk:
            if k != 1:
                out[f"acc@{k}"] = self.get(k)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{self.__class__.__name__}(acc@1={self.get(1):.2f}%, "
            f"n={self.num_samples})"
        )


class MetricTracker:
    """Accumulate arbitrary scalar metrics as running means (epoch logging)."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(value) * int(n)
        self.counts[key] = self.counts.get(key, 0) + int(n)

    def add(self, values: Mapping[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.update(k, v, n=n)

    def mean(self, key: str) -> float:
        if self.counts.get(key, 0) == 0:
            return 0.0
        return self.sums[key] / float(self.counts[key])

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()

    def as_dict(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self.sums}


# ---------------------------------------------------------------------------
# Multi-seed aggregation ("Mean % +- Std %", Table 3 convention)
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single (dataset, backbone, method, seed) run."""

    dataset: str
    backbone: str = "resnet18"
    method: str = "ours"
    seed: int = 0
    accuracy: float = 0.0  # percent
    loss: Optional[float] = None
    mapping: str = "Ilm"
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "method": self.method,
            "seed": int(self.seed),
            "accuracy": float(self.accuracy),
            "mapping": self.mapping,
        }
        if self.loss is not None:
            d["loss"] = float(self.loss)
        d.update(self.extra)
        return d

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.method}/{self.dataset}/{self.backbone}/seed{self.seed}: "
            f"{self.accuracy:.2f}"
        )


def aggregate_seeds(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:
    """Return ``(mean, std)`` of per-seed accuracies (percent).

    Uses the sample standard deviation (``ddof=1``), matching the
    "mean +- std" reporting of Table 3.  NaNs / ``None`` entries are ignored.
    """
    vals: List[float] = []
    for v in values:
        if v is None:
            continue
        fv = float(v)
        if math.isnan(fv):
            continue
        vals.append(fv)
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mean = sum(vals) / n
    if n <= ddof or n <= 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - ddof)
    return mean, math.sqrt(max(var, 0.0))


def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    """Render ``"52.53 +- 0.31"`` (paper Table 3 style)."""
    if decimals <= 0:
        return f"{mean:.0f}"
    return f"{mean:.{decimals}f} +- {std:.2f}"


def aggregate_results(
    results: Iterable[Any],
    *,
    by: Sequence[str] = ("method", "dataset"),
    metric: str = "accuracy",
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """Group per-seed results and compute mean/std per group.

    ``results`` may be :class:`RunResult` objects or plain dicts.  Returns a
    mapping ``group_key -> {"mean", "std", "n", "values"}``.
    """
    groups: Dict[Tuple[Any, ...], List[float]] = {}
    for r in results:
        row = dict(r) if isinstance(r, Mapping) else r.as_dict()
        key = tuple(row.get(k) for k in by)
        groups.setdefault(key, []).append(float(row.get(metric, 0.0)))
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for key, vals in groups.items():
        mean, std = aggregate_seeds(vals)
        out[key] = {"mean": mean, "std": std, "n": len(vals), "values": list(vals)}
    return out


def mean_over_datasets(
    per_dataset: Mapping[str, Any],
    *,
    dataset_order: Optional[Sequence[str]] = None,
) -> float:
    """Average accuracy over datasets, ignoring an ``"average"`` key itself.

    Mirrors the bottom "AVERAGE" row of Tables 1-3 (unweighted mean over
    evaluated target tasks).
    """
    order = list(dataset_order) if dataset_order is not None else list(per_dataset)
    vals: List[float] = []
    for name in order:
        if _canonical(name) == "average" or name not in per_dataset:
            continue
        v = per_dataset[name]
        if isinstance(v, Mapping):
            v = v.get("mean", 0.0)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def format_results_table(
    per_dataset: Mapping[str, Any],
    *,
    dataset_order: Optional[Sequence[str]] = None,
    method: str = "ours",
    decimals: int = 1,
) -> str:
    """Render a small paper-style text table for one method."""
    order = list(dataset_order) if dataset_order is not None else list(per_dataset)
    lines = [f"{'DATASET':<14}{method.upper():>16}"]
    for name in order:
        if name not in per_dataset:
            continue
        v = per_dataset[name]
        if isinstance(v, Mapping):
            txt = f"{float(v.get('mean', 0.0)):.{decimals}f}"
            if v.get("n", 1) > 1:
                txt += f" +- {float(v.get('std', 0.0)):.{decimals}f}"
        else:
            txt = f"{float(v):.{decimals}f}"
        lines.append(f"{name.upper():<14}{txt:>16}")
    avg = mean_over_datasets(per_dataset, dataset_order=order)
    lines.append(f"{'AVERAGE':<14}{avg:>16.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison against the paper
# ---------------------------------------------------------------------------


def compare_with_reference(
    results: Mapping[str, Any],
    *,
    table: str = "table2_vit_b32",
    key: str = "ours",
    average_tolerance: float = DEFAULT_AVERAGE_TOLERANCE,
    per_dataset_tolerance: float = DEFAULT_PER_DATASET_TOLERANCE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare measured accuracies against the paper's reported values.

    ``results`` maps dataset name -> measured accuracy (float) or a mapping with
    a ``"mean"`` key.  Returns a report with per-dataset deltas, the average
    delta and booleans indicating whether the tolerances were met.
    """
    reference_all = resolve_reference_table(table)
    if (
        isinstance(reference_all, Mapping)
        and key in reference_all
        and isinstance(reference_all[key], Mapping)
    ):
        reference: Dict[str, Any] = dict(reference_all[key])
    else:
        reference = dict(reference_all)

    measured: Dict[str, float] = {}
    for name, v in results.items():
        if _canonical(name) == "average":
            continue
        if isinstance(v, Mapping):
            v = v.get("mean", 0.0)
        measured[_canonical(name)] = float(v)

    per_dataset: Dict[str, Dict[str, Any]] = {}
    for ref_name, ref_val in reference.items():
        if _canonical(ref_name) == "average":
            continue
        cn = _canonical(ref_name)
        if cn not in measured:
            continue
        delta = measured[cn] - float(ref_val)
        per_dataset[str(ref_name)] = {
            "measured": measured[cn],
            "reference": float(ref_val),
            "delta": delta,
            "within_tolerance": abs(delta) <= per_dataset_tolerance,
        }

    measured_avg = (
        sum(v["measured"] for v in per_dataset.values()) / len(per_dataset)
        if per_dataset
        else 0.0
    )
    ref_avg = reference.get("average")
    if ref_avg is None and per_dataset:
        ref_avg = sum(v["reference"] for v in per_dataset.values()) / len(per_dataset)
    ref_avg = float(ref_avg) if ref_avg is not None else 0.0
    avg_delta = measured_avg - ref_avg

    report: Dict[str, Any] = {
        "table": table,
        "key": key,
        "num_datasets": len(per_dataset),
        "per_dataset": per_dataset,
        "measured_average": measured_avg,
        "reference_average": ref_avg,
        "average_delta": avg_delta,
        "average_within_tolerance": abs(avg_delta) <= average_tolerance,
    }
    if verbose:  # pragma: no cover - convenience
        print(format_results_table(measured, method=key))
        print(f"reference average = {ref_avg:.2f}, delta = {avg_delta:+.2f}")
    return report


def compare_run_to_reference(
    results: Iterable[Any],
    *,
    table: str = "table2_vit_b32",
    key: str = "ours",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Aggregate per-seed results then compare the averages to a paper table."""
    aggregated = aggregate_results(results)
    per_dataset = {k[-1]: v for k, v in aggregated.items()}
    return compare_with_reference(per_dataset, table=table, key=key, **kwargs)


def dump_results_json(path: str, results: Iterable[Any], **kwargs: Any) -> str:
    """Serialise results (list of :class:`RunResult`/dicts) to JSON."""
    rows = [dict(r) if isinstance(r, Mapping) else r.as_dict() for r in results]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, **kwargs)
    return path
