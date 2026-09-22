#!/usr/bin/env python
"""Figure 3 driver: task performance vs. relative inference efficiency.

Reproduces Figure 3 of *APT: Adaptive Pruning and Tuning Pretrained Language
Models for Efficient Training and Inference* (Section 5.4):

    "Figure 3. Task performance vs. relative inference efficiency on RoBERTa,
     T5, and LLaMA-2 7B models with APT and baselines."

The paper also says (Section 5.4) "We demonstrate the end-task performance of
APT comparing to fine-tuning (FT), LoRA-tuning (LoRA), and pruning baselines in
Table 2 and Table 3 ... when pruning RoBERTa models to 60% sparsity, APT
converges 8.4x faster than the LoRA+Prune baseline", i.e. the figure plots task
performance against relative *inference* efficiency while the pruning sparsity
is swept.

This script is a **driver only**: it never re-implements APT or any baseline.
It sweeps the target sparsity for a given model family, dispatches every
(model, method, task, sparsity, seed) cell to the existing training harnesses
(:mod:`scripts.train_apt`, :mod:`scripts.train_baseline`), measures task
quality together with inference efficiency (latency, throughput, peak memory),
normalises every efficiency number against the fine-tuned (FT) reference
(FT = 100%, following Section 5.3), builds the Figure 3 data series
(performance vs. relative inference efficiency), optionally renders a
matplotlib figure, and persists the sweep as JSON/CSV.

Scope (per the reproduction plan / Addendum): small models only, i.e. RoBERTa
and T5.  LLaMA-2 7B pruning is intentionally excluded, therefore the figure is
produced with the RoBERTa and T5 panels.

Usage examples
--------------
    python scripts/run_figure3_sparsity.py --dry-run
    python scripts/run_figure3_sparsity.py --model roberta \
        --sparsities 0.2 0.4 0.6 --seeds 42
    python scripts/run_figure3_sparsity.py --models roberta t5 --quiet
    python scripts/run_figure3_sparsity.py --print-reference
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apt", "configs"
)
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "default.yaml")

MODEL_DEFAULT = "roberta"
# LLaMA-2 7B is out of the reproduction scope (Addendum): small models only.
DEFAULT_MODELS: Tuple[str, ...] = ("roberta", "t5")
IN_SCOPE_MODELS: Tuple[str, ...] = ("roberta", "bert", "t5")

METHOD_ORDER: Tuple[str, ...] = (
    "ft",
    "lora",
    "lora_prune",
    "prune_distill",
    "lora_prune_distill",
    "apt",
)
BASELINE_METHODS: Tuple[str, ...] = METHOD_ORDER[:-1]

METHOD_ALIASES: Dict[str, str] = {
    "ft": "ft",
    "finetune": "ft",
    "fine_tune": "ft",
    "fine-tuning": "ft",
    "full": "ft",
    "lora": "lora",
    "lora_tuning": "lora",
    "lora-prune": "lora_prune",
    "lora+prune": "lora_prune",
    "mask_tuning": "lora_prune",
    "mask-tuning": "lora_prune",
    "prune": "lora_prune",
    "prune_distill": "prune_distill",
    "prune+distill": "prune_distill",
    "cofi": "prune_distill",
    "cofi_pruning": "prune_distill",
    "lora_prune_distill": "lora_prune_distill",
    "lora+prune+distill": "lora_prune_distill",
    "lora_distill": "lora_prune_distill",
    "lora+distill": "lora_prune_distill",
    "apt": "apt",
}

METHOD_DISPLAY: Dict[str, str] = {
    "ft": "FT",
    "lora": "LoRA",
    "lora_prune": "LoRA+Prune",
    "prune_distill": "Prune+Distill",
    "lora_prune_distill": "LoRA+Prune+Distill",
    "apt": "APT",
}

MODEL_ALIASES: Dict[str, str] = {
    "roberta": "roberta",
    "roberta-base": "roberta",
    "roberta_base": "roberta",
    "bert": "bert",
    "bert-base": "bert",
    "bert_base": "bert",
    "t5": "t5",
    "t5-base": "t5",
    "t5_base": "t5",
    # Excluded from the reproduction scope but recognised so we can warn.
    "llama": "llama",
    "llama2": "llama",
    "llama-2": "llama",
    "llama-2-7b": "llama",
    "llama2-7b": "llama",
    "llama-2-13b": "llama",
}

TASK_ALIASES: Dict[str, str] = {
    "sst-2": "sst2",
    "sst_2": "sst2",
    "sst2": "sst2",
    "mnli": "mnli",
    "mnli-mm": "mnli",
    "qnli": "qnli",
    "qqp": "qqp",
    "mrpc": "mrpc",
    "cola": "cola",
    "rte": "rte",
    "stsb": "stsb",
    "sts-b": "stsb",
    "squad": "squad_v2",
    "squad_v2": "squad_v2",
    "squad2": "squad_v2",
    "cnndm": "cnndm",
    "cnn_dailymail": "cnndm",
    "cnn-dailymail": "cnndm",
}

# Tasks used for the y-axis (task performance) of each model panel.
MODEL_TASKS: Dict[str, Tuple[str, ...]] = {
    "roberta": ("sst2", "mnli"),
    "bert": ("sst2", "mnli"),
    "t5": ("sst2", "mnli"),
}

# Figure 3 sweeps performance against inference efficiency across sparsity.
DEFAULT_SPARSITIES: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)

TTA_FRACTION = 0.97
RELATIVE_SCALE = 100.0
SMALL_MODEL_INF_BATCH = 128

EFFICIENCY_KEYS: Tuple[str, ...] = ("train_time", "train_mem", "inf_time", "inf_mem")

# --------------------------------------------------------------------------- #
# Appendix I (Table 11) absolute FT efficiency -- used to normalise the sweep
# when FT has not been trained locally.  Values are taken verbatim from the
# paper's raw efficiency table (RoBERTa: 127 s, 2696 MB, 220.8 ms, 1157 MB;
# T5: 366 s, 7217 MB, 248.1 ms, 2347 MB).
# --------------------------------------------------------------------------- #

TABLE11_FT_REFERENCE: Dict[str, Dict[str, float]] = {
    "roberta": {
        "tta_seconds": 127.0,
        "train_time_s": 127.0,
        "train_peak_mem_mb": 2696.0,
        "inf_time_ms": 220.8,
        "inf_mem_mb": 1157.0,
    },
    "t5": {
        "tta_seconds": 366.0,
        "train_time_s": 366.0,
        "train_peak_mem_mb": 7217.0,
        "inf_time_ms": 248.1,
        "inf_mem_mb": 2347.0,
    },
}

# --------------------------------------------------------------------------- #
# Anchor points known from Table 2 (60% sparsity) used as sanity-check markers
# on the figure.  Task-accuracy anchors are deliberately left as ``None``
# instead of being fabricated; the inference-efficiency anchors are published.
# --------------------------------------------------------------------------- #

FIGURE3_REFERENCES: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {
    "roberta": {
        "0.6": {
            "ft": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
            "lora": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
            "lora_prune": {"quality": None, "inf_time_pct": 38.0, "inf_mem_pct": 75.1},
            "prune_distill": {"quality": None, "inf_time_pct": 38.6, "inf_mem_pct": 79.2},
            "lora_prune_distill": {"quality": None, "inf_time_pct": 39.4,
                                   "inf_mem_pct": 82.3},
            "apt": {"quality": None, "inf_time_pct": 41.3, "inf_mem_pct": 78.1},
        },
        "0.0": {
            "ft": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
            "lora": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
        },
    },
    "t5": {
        "0.6": {
            "ft": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
            "lora": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
            "lora_prune": {"quality": None, "inf_time_pct": 47.1, "inf_mem_pct": 73.4},
            "apt": {"quality": None, "inf_time_pct": 74.6, "inf_mem_pct": 81.5},
        },
        "0.0": {
            "ft": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
            "lora": {"quality": None, "inf_time_pct": 100.0, "inf_mem_pct": 100.0},
        },
    },
}

FIGURE3_NOTE = (
    "Figure 3 in the paper contains RoBERTa, T5 and LLaMA-2 7B panels. "
    "LLaMA-2 7B pruning is outside the reproduction scope (Addendum), so the "
    "figure is produced for the RoBERTa and T5 panels only."
)

TABLE6_GROUPS: Dict[str, Dict[str, Any]] = {
    "glue-big": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40,
                 "distill_epochs": 20, "max_seq_length": 128},
    "glue-small": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40,
                   "distill_epochs": 20, "max_seq_length": 128},
    "squad": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40,
              "distill_epochs": 20, "max_seq_length": 384, "doc_stride": 128,
              "max_query_length": 64},
    "cnndm": {"learning_rate": 1e-4, "batch_size": 16, "epochs": 16,
              "distill_epochs": 6, "max_seq_length": 512, "max_target_length": 128},
}

GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")
SQUAD_TASKS = ("squad_v2", "squad2", "squad")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum", "samsum")

MODEL_NAMES: Dict[str, str] = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "t5": "t5-base",
    "llama": "meta-llama/Llama-2-7b-hf",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable primitives."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()
                if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    for attr in ("tolist", "item"):
        if hasattr(obj, attr):
            try:
                return _json_safe(getattr(obj, attr)())
            except Exception:  # pragma: no cover - defensive
                pass
    if hasattr(obj, "__dict__"):
        drop = {"trainer", "model", "tokenizer", "optimizer", "scheduler"}
        return {k: _json_safe(v) for k, v in vars(obj).items() if k not in drop}
    return str(obj)


def _to_float(value: Any) -> Optional[float]:
    """Best-effort float conversion; returns ``None`` when not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_model(model: Optional[str]) -> str:
    """Canonicalise a model family name."""
    if not model:
        return MODEL_DEFAULT
    key = str(model).strip().lower().replace("_", "-")
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    key2 = key.replace("-", "")
    if key2 in MODEL_ALIASES:
        return MODEL_ALIASES[key2]
    for alias, canonical in MODEL_ALIASES.items():
        if key.startswith(alias):
            return canonical
    return key2


def normalize_method(method: Optional[str]) -> str:
    """Canonicalise a method key (accepts ``LoRA+Prune`` style aliases)."""
    if not method:
        return "apt"
    key = str(method).strip().lower().replace(" ", "")
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    key2 = key.replace("-", "_").replace("+", "_")
    while "__" in key2:
        key2 = key2.replace("__", "_")
    if key2 in METHOD_ALIASES:
        return METHOD_ALIASES[key2]
    if key2 in METHOD_ORDER:
        return key2
    raise KeyError(f"unknown method: {method!r} (known: {sorted(METHOD_ORDER)})")


def normalize_task(task: Optional[str]) -> str:
    """Canonicalise a task name."""
    if not task:
        return "sst2"
    key = str(task).strip().lower().replace(" ", "")
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    key2 = key.replace("-", "_")
    if key2 in TASK_ALIASES:
        return TASK_ALIASES[key2]
    return key2


def normalize_sparsity(sparsity: Optional[float] = None,
                       density: Optional[float] = None) -> float:
    """Return sparsity in ``[0, 1]``, accepting percent inputs or a density."""
    if density is not None:
        value = 1.0 - float(density)
    elif sparsity is None:
        value = 0.6
    else:
        value = float(sparsity)
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def density_for_sparsity(sparsity: Optional[float]) -> float:
    """Density (= retained fraction) for a sparsity value."""
    return max(0.0, min(1.0, 1.0 - normalize_sparsity(sparsity)))


def sparsity_for_density(density: Optional[float]) -> float:
    """Sparsity for a density (retained fraction)."""
    return normalize_sparsity(density=density)


def density_label(density: Optional[float]) -> str:
    """Human readable density label, e.g. ``"40%"``."""
    value = _to_float(density)
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def sparsity_for_method(method: str, sparsity: float = 0.6) -> float:
    """Dense methods (FT, LoRA) keep the full model."""
    method = normalize_method(method)
    if method in ("ft", "lora"):
        return 0.0
    return normalize_sparsity(sparsity)


def parse_seeds(seeds: Any, default: Sequence[int] = DEFAULT_SEEDS) -> Tuple[int, ...]:
    """Parse a seed specification into a tuple of ints."""
    if seeds is None or seeds == "":
        return tuple(default)
    if isinstance(seeds, bool):
        return tuple(default)
    if isinstance(seeds, (int, float)):
        return (int(seeds),)
    if isinstance(seeds, str):
        parts = [p for p in seeds.replace(",", " ").split() if p]
        return tuple(int(p) for p in parts) if parts else tuple(default)
    try:
        parsed = tuple(int(s) for s in seeds)
    except TypeError:
        return (int(seeds),)
    return parsed or tuple(default)


def parse_sparsities(values: Any, default: Sequence[float] = DEFAULT_SPARSITIES
                     ) -> Tuple[float, ...]:
    """Parse/validate a sparsity sweep specification."""
    if values is None or values == "":
        return tuple(float(v) for v in default)
    if isinstance(values, bool):
        return tuple(float(v) for v in default)
    if isinstance(values, (int, float)):
        return (normalize_sparsity(float(values)),)
    if isinstance(values, str):
        parts = [p for p in values.replace(",", " ").split() if p]
        if not parts:
            return tuple(float(v) for v in default)
        return tuple(normalize_sparsity(float(p)) for p in parts)
    return tuple(normalize_sparsity(float(v)) for v in values)


def split_methods(value: Any, default: Sequence[str] = METHOD_ORDER) -> List[str]:
    """Parse a comma/space separated method list into canonical keys."""
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        items: List[Any] = [p for p in value.replace(",", " ").split() if p]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]
    out: List[str] = []
    for item in items:
        key = normalize_method(item)
        if key not in out:
            out.append(key)
    return out


def split_models(value: Any, default: Sequence[str] = DEFAULT_MODELS) -> List[str]:
    """Parse a comma/space separated model list into canonical names."""
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        items: List[Any] = [p for p in value.replace(",", " ").split() if p]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]
    out: List[str] = []
    for item in items:
        key = normalize_model(item)
        if key not in out:
            out.append(key)
    return out


def default_methods(model: str) -> List[str]:
    """Methods plotted for a model panel (T5 omits the distill-free baselines)."""
    model = normalize_model(model)
    if model == "t5":
        return ["ft", "lora", "lora_prune", "apt"]
    return list(METHOD_ORDER)


def tasks_for_model(model: str) -> List[str]:
    """Task list used to average the y-axis for a model panel."""
    return list(MODEL_TASKS.get(normalize_model(model), ("sst2", "mnli")))


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and (population) std of the finite values of ``values``."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return mean, var ** 0.5


# --------------------------------------------------------------------------- #
# Metric / efficiency bookkeeping
# --------------------------------------------------------------------------- #

def primary_from_metrics(task: str, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract the paper's primary metric for a task from a metric dict."""
    if not metrics:
        return None
    task = normalize_task(task)
    try:  # canonical implementation
        from apt.eval.metrics import primary_metric as _pm  # noqa: PLC0415

        value = _to_float(_pm(task, metrics))
        if value is not None:
            return value
    except Exception:
        pass
    candidates = {
        "sst2": ("accuracy", "acc"),
        "mnli": ("accuracy", "acc"),
        "qnli": ("accuracy", "acc"),
        "qqp": ("accuracy", "f1", "acc"),
        "mrpc": ("accuracy", "f1", "acc"),
        "cola": ("matthews_correlation", "matthews", "accuracy"),
        "rte": ("accuracy", "acc"),
        "stsb": ("spearmanr", "spearman", "pearsonr"),
        "squad_v2": ("f1", "exact", "em"),
        "cnndm": ("rougeL", "rouge_l", "rougeLsum", "rouge1"),
    }.get(task, ("accuracy", "f1", "primary"))
    for key in candidates:
        if key in metrics:
            value = _to_float(metrics[key])
            if value is not None:
                return value
    if "primary" in metrics:
        return _to_float(metrics["primary"])
    for key, value in metrics.items():
        if key in ("loss", "epoch", "step"):
            continue
        value = _to_float(value)
        if value is not None:
            return value
    return None


def primary_from_summary(summary: Optional[Dict[str, Any]], task: str) -> Optional[float]:
    """Extract the primary metric from a training summary dict."""
    if not summary:
        return None
    if summary.get("primary") is not None:
        value = _to_float(summary.get("primary"))
        if value is not None:
            return value
    task = normalize_task(task)
    for key in ("metrics", "per_task", "results"):
        block = summary.get(key)
        if isinstance(block, dict):
            entry = block.get(task)
            if isinstance(entry, dict):
                value = primary_from_metrics(task, entry)
                if value is not None:
                    return value
            value = primary_from_metrics(task, block)
            if value is not None:
                return value
    return primary_from_metrics(task, summary)


def glue_average(values: Any, tasks: Optional[Sequence[str]] = None,
                 default: Optional[float] = None) -> Optional[float]:
    """Mean of the finite primary metrics in ``values`` (dict or sequence)."""
    if isinstance(values, dict):
        if tasks:
            nums = [_to_float(values.get(normalize_task(t))) for t in tasks]
        else:
            nums = [_to_float(v) for v in values.values()]
    else:
        nums = [_to_float(v) for v in (values or [])]
    nums = [n for n in nums if n is not None]
    if not nums:
        return default
    return sum(nums) / len(nums)


def normalise_from_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Pull the absolute efficiency numbers out of a training summary."""
    out: Dict[str, Optional[float]] = {k: None for k in EFFICIENCY_KEYS}
    out.update({"tta_seconds": None, "throughput": None})
    if not summary:
        return out

    blocks: List[Dict[str, Any]] = []
    for key in ("efficiency", "raw"):
        block = summary.get(key)
        if isinstance(block, dict):
            blocks.append(block)
    blocks.append(summary)

    aliases: Dict[str, Tuple[str, ...]] = {
        "train_time": ("train_time_s", "train_time", "training_time", "wall_time_s",
                       "time_s"),
        "train_mem": ("train_peak_mem_mb", "train_mem_mb", "train_memory_mb",
                      "peak_mem_mb"),
        "inf_time": ("inf_time_ms", "inference_time_ms", "latency_ms", "inf_time"),
        "inf_mem": ("inf_mem_mb", "inference_mem_mb", "inf_peak_mem_mb", "inf_memory_mb"),
        "tta_seconds": ("tta_seconds", "tta", "time_to_accuracy"),
        "throughput": ("inf_throughput", "throughput", "samples_per_sec",
                       "samples_per_second"),
    }
    for key, names in aliases.items():
        for block in blocks:
            for name in names:
                if name in block and block[name] is not None:
                    value = _to_float(block[name])
                    if value is not None:
                        out[key] = value
                        break
            if out[key] is not None:
                break
    return out


def compute_tta(history: Optional[Sequence[Any]], reference: Optional[float], *,
                fraction: float = TTA_FRACTION,
                higher_is_better: bool = True) -> Optional[float]:
    """Linearly interpolated time to reach ``fraction`` of ``reference``."""
    if not history or reference is None:
        return None
    target = float(reference) * float(fraction)
    points: List[Tuple[float, float]] = []
    for entry in history:
        elapsed: Optional[float] = None
        metric: Optional[float] = None
        if isinstance(entry, dict):
            metric = _to_float(entry.get("metric", entry.get("primary")))
            elapsed = _to_float(
                entry.get("elapsed", entry.get("time", entry.get("train_time_s")))
            )
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            elapsed, metric = _to_float(entry[0]), _to_float(entry[1])
        else:
            continue
        if metric is None or elapsed is None:
            continue
        points.append((float(elapsed), float(metric)))
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    for index, (elapsed, metric) in enumerate(points):
        reached = metric >= target if higher_is_better else metric <= target
        if reached and index == 0:
            return elapsed
        if not reached:
            continue
        if index == 0:
            return elapsed
        prev_elapsed, prev_metric = points[index - 1]
        span = abs(metric - prev_metric)
        if span < 1e-12:
            return elapsed
        frac = abs(target - prev_metric) / span
        return prev_elapsed + frac * (elapsed - prev_elapsed)
    return None


def relative_efficiency(raw: Dict[str, Optional[float]],
                        reference: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Normalise absolute efficiency numbers to the FT reference (FT = 100%)."""
    out: Dict[str, Optional[float]] = {k: None for k in EFFICIENCY_KEYS}
    out["tta"] = None
    out["throughput"] = None

    for key in ("train_time", "train_mem", "inf_time", "inf_mem"):
        value = raw.get(key)
        ref = reference.get(key)
        if ref is None:
            ref = reference.get(key + "_s", reference.get(key + "_mb"))
        if value is None or ref in (None, 0):
            continue
        out[key] = RELATIVE_SCALE * float(value) / float(ref)

    tta = raw.get("tta_seconds")
    ref_tta = reference.get("tta_seconds")
    if tta is None and raw.get("train_time") not in (None, 0) and ref_tta:
        # Without an evaluation history, total training time is the TTA proxy.
        tta = raw.get("train_time")
    if tta is not None and ref_tta:
        out["tta"] = RELATIVE_SCALE * float(tta) / float(ref_tta)

    thr = raw.get("throughput")
    ref_thr = reference.get("throughput")
    if thr is not None and ref_thr:
        out["throughput"] = RELATIVE_SCALE * float(thr) / float(ref_thr)
    elif out.get("inf_time"):
        # Relative speedup expressed as a percentage of FT (FT = 100%).
        out["throughput"] = RELATIVE_SCALE * RELATIVE_SCALE / float(out["inf_time"])
    return out


def speedup_from_relative(inf_time_pct: Optional[float]) -> Optional[float]:
    """Convert a relative inference time (FT = 100%) into a speedup factor."""
    value = _to_float(inf_time_pct)
    if value in (None, 0):
        return None
    return RELATIVE_SCALE / value


def efficiency_reference(rows: Sequence[Dict[str, Any]], model: str,
                         ft_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the FT efficiency reference used to normalise the sweep."""
    model = normalize_model(model)
    if ft_summary:
        raw = normalise_from_summary(ft_summary)
        if any(raw.get(k) is not None for k in EFFICIENCY_KEYS):
            return {"source": "measured", "model": model, "raw": raw}
    for row in rows:
        if normalize_method(row.get("method", "")) == "ft":
            raw = row.get("raw") or {}
            if any(raw.get(k) is not None for k in EFFICIENCY_KEYS):
                return {"source": "measured", "model": model, "raw": raw}
    table = TABLE11_FT_REFERENCE.get(model)
    if table:
        raw = {
            "train_time": table["train_time_s"],
            "train_mem": table["train_peak_mem_mb"],
            "inf_time": table["inf_time_ms"],
            "inf_mem": table["inf_mem_mb"],
            "tta_seconds": table["tta_seconds"],
            "throughput": None,
        }
        return {"source": "table11", "model": model, "raw": raw}
    return {"source": "unknown", "model": model, "raw": {}}


# --------------------------------------------------------------------------- #
# Config / training dispatch
# --------------------------------------------------------------------------- #

def table6_group_for(model_type: str, task: str) -> str:
    """Map a (model, task) pair onto the Table 6 hyper-parameter column."""
    task = normalize_task(task)
    if task in SQUAD_TASKS or task.startswith("squad"):
        return "squad"
    if task in SEQ2SEQ_TASKS:
        return "cnndm"
    return "glue-big" if task in GLUE_BIG_TASKS else "glue-small"


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML file, returning ``{}`` when unavailable."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml  # noqa: PLC0415

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"could not load config {path}: {exc}")
        return {}


def resolve_config_path(path: Optional[str], model: str, task: str) -> Optional[str]:
    """Resolve an explicit path/filename/alias or (model, task) into a config."""
    if path:
        for candidate in (path, os.path.join(CONFIG_DIR, path),
                          os.path.join(CONFIG_DIR, f"{path}.yaml")):
            if os.path.isfile(candidate):
                return candidate
    model = normalize_model(model)
    task = normalize_task(task)
    candidates = [
        f"{model}_{task}.yaml",
        {"squad_v2": "squad.yaml"}.get(task, ""),
        {"cnndm": "t5_cnndm.yaml"}.get(task, ""),
        f"{model}_sst2.yaml",
        "default.yaml",
    ]
    for name in candidates:
        if not name:
            continue
        candidate = os.path.join(CONFIG_DIR, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_config(model: str = MODEL_DEFAULT, task: str = "sst2",
                 sparsity: float = 0.6, seed: int = 42,
                 overrides: Optional[Dict[str, Any]] = None,
                 method: str = "apt") -> Dict[str, Any]:
    """Assemble the configuration for one Figure 3 cell.

    Precedence: ``default.yaml`` < task config < Table 6 group < method
    defaults < caller overrides.
    """
    model = normalize_model(model)
    task = normalize_task(task)
    method = normalize_method(method)
    sparsity = normalize_sparsity(sparsity)

    group = table6_group_for(model, task)
    config: Dict[str, Any] = dict(load_yaml(DEFAULT_CONFIG_FILE))
    config.update(load_yaml(resolve_config_path(None, model, task)))

    config.update({
        "model_name_or_path": MODEL_NAMES.get(model, model),
        "model_type": model,
        "task": task,
        "table6_group": group,
        "seed": int(seed),
    })
    for key, value in TABLE6_GROUPS.get(group, {}).items():
        config.setdefault(key, value)

    # APT / Algorithm 1 defaults (Section 4, Table 6, Appendix A, Addendum).
    config.setdefault("target_sparsity", sparsity)
    config.setdefault("initial_rank", 8)
    config.setdefault("scaling", 2.0)
    config.setdefault("mask_alpha", 0.01)
    config.setdefault("ema_beta", 0.85)
    config.setdefault("tau", 4)
    config.setdefault("top_fraction", 0.5)
    config.setdefault("tuning_budget_initial", 1.0)
    config.setdefault("tuning_budget_final", 2.0)
    config.setdefault("use_distillation", True)
    config.setdefault("use_kurtosis", True)
    config.setdefault("optimizer", "adamw")
    config.setdefault("weight_decay", 0.01)
    config.setdefault("warmup_ratio", 0.06)
    config.setdefault("max_grad_norm", 1.0)
    config.setdefault("inference_batch_size", SMALL_MODEL_INF_BATCH)
    config.setdefault("sequence_length", 128)

    # Distillation weighting: GLUE 1.0/0.9, SQuAD & CNN/DM 0.1/0.9 (Appendix A).
    config.setdefault("pred_distill_weight",
                      0.1 if group in ("squad", "cnndm") else 1.0)
    config.setdefault("layer_distill_weight", 0.9)

    # Dense rows of the figure.
    if method in ("ft", "lora"):
        config["target_sparsity"] = 0.0
        config["use_distillation"] = False
    elif method == "lora_prune":
        # LoRA+Prune tunes a static LoRA (Mask Tuning), i.e. no rank growth.
        config["tuning_budget_final"] = config.get("tuning_budget_initial", 1.0)
        config["use_distillation"] = False

    config["method"] = method
    config["sparsity"] = sparsity
    config["density"] = density_for_sparsity(sparsity)
    if overrides:
        config.update(overrides)
    return config


def resolve_trainer(method: str) -> Callable[..., Dict[str, Any]]:
    """Return a ``trainer(method, config) -> summary`` callable for a method."""
    method = normalize_method(method)

    def _apt_trainer(method_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from scripts.train_apt import run_training  # noqa: PLC0415

            return run_training(cfg)
        except ImportError:
            pass
        try:
            from apt.training import train_apt  # noqa: PLC0415

            return train_apt(cfg)
        except Exception as exc:
            raise RuntimeError(f"APT trainer unavailable: {exc}") from exc

    def _baseline_trainer(method_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from scripts.train_baseline import run_baseline  # noqa: PLC0415

            return run_baseline(method_key, cfg)
        except ImportError:
            pass
        try:
            from apt.baselines import get_method  # noqa: PLC0415

            trainer_cls = get_method(method_key)
            trainer = trainer_cls(**cfg)
            return trainer.fit()
        except Exception as exc:
            raise RuntimeError(f"baseline trainer unavailable ({method_key}): {exc}") from exc

    return _apt_trainer if method == "apt" else _baseline_trainer


def external_repo_available(name: str = "cofi") -> bool:
    """Best-effort check that an external baseline repo can be imported."""
    if normalize_method(name) == "lora_prune":
        try:
            from apt.baselines import mask_tuning as _mt  # noqa: PLC0415

            return bool(_mt.external_repo_available())
        except Exception:
            return False
    try:
        from apt.baselines import cofi as _cofi  # noqa: PLC0415

        return bool(_cofi.external_repo_available())
    except Exception:
        return False


def proxy_note(method: str) -> str:
    """Document whether a baseline uses the external repo or the in-repo proxy."""
    method = normalize_method(method)
    if method in ("ft", "lora", "apt"):
        return ""
    repo = "retraining-free-pruning" if method == "lora_prune" else "CoFiPruning"
    if external_repo_available(method):
        return f"{METHOD_DISPLAY[method]} uses the external {repo} repository"
    return (f"{METHOD_DISPLAY[method]} uses the in-repo reimplementation because the "
            f"external {repo} repository is unavailable")


def train_one(method: str, model: str, task: str, *, sparsity: float = 0.6,
              seed: int = 42, config_overrides: Optional[Dict[str, Any]] = None,
              trainer: Optional[Callable[..., Dict[str, Any]]] = None,
              verbose: bool = True) -> Dict[str, Any]:
    """Train a single (method, model, task, sparsity, seed) cell."""
    method = normalize_method(method)
    model = normalize_model(model)
    task = normalize_task(task)
    config = build_config(model, task, sparsity, seed, config_overrides, method=method)
    runner = trainer or resolve_trainer(method)
    if verbose:
        print(f"[figure3] {METHOD_DISPLAY.get(method, method):<20} {model:<8} "
              f"{task:<8} sparsity={config.get('sparsity', sparsity):.2f} seed={seed}")
    started = time.time()
    try:
        summary = runner(method, config)
    except TypeError:
        summary = runner(config)  # tolerate narrower trainer signatures
    summary = dict(summary or {})
    summary.setdefault("method", method)
    summary.setdefault("model", model)
    summary.setdefault("task", task)
    summary["sparsity"] = config.get("sparsity", sparsity)
    summary["seed"] = seed
    summary["wall_time_s"] = time.time() - started
    return summary


def collect_metrics(method: str, model: str, tasks: Optional[Sequence[str]] = None, *,
                    sparsity: float = 0.6, seeds: Sequence[int] = DEFAULT_SEEDS,
                    trainer: Optional[Callable[..., Dict[str, Any]]] = None,
                    verbose: bool = True,
                    config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train/aggregate one (method, sparsity) point across tasks and seeds."""
    method = normalize_method(method)
    model = normalize_model(model)
    tasks = [normalize_task(t) for t in (tasks or tasks_for_model(model))]
    sparsity = normalize_sparsity(sparsity)

    per_task: Dict[str, List[Optional[float]]] = {t: [] for t in tasks}
    primary_per_seed: List[Optional[float]] = []
    raw_records: List[Dict[str, Optional[float]]] = []
    histories: List[List[Any]] = []
    failures: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_values: List[Optional[float]] = []
        for task in tasks:
            try:
                summary = train_one(method, model, task, sparsity=sparsity, seed=int(seed),
                                    config_overrides=config_overrides, trainer=trainer,
                                    verbose=verbose)
            except Exception as exc:  # keep sweeping
                warnings.warn(f"[figure3] {method}/{model}/{task} seed={seed} failed: {exc}")
                failures.append({"method": method, "model": model, "task": task,
                                 "sparsity": sparsity, "seed": int(seed), "error": str(exc)})
                per_task[task].append(None)
                continue
            value = primary_from_summary(summary, task)
            per_task[task].append(value)
            seed_values.append(value)
            raw_records.append(normalise_from_summary(summary))
            if isinstance(summary.get("history"), list):
                histories.append(summary["history"])
            summaries.append(_json_safe(summary))

        seed_mean = mean_std(seed_values)[0] if seed_values else None
        primary_per_seed.append(seed_mean)

    primary_mean, primary_std = mean_std(primary_per_seed)
    per_task_summary: Dict[str, Dict[str, Optional[float]]] = {}
    task_means: List[Optional[float]] = []
    for task, values in per_task.items():
        tmean, tstd = mean_std(values)
        per_task_summary[task] = {"mean": tmean, "std": tstd, "values": values}
        task_means.append(tmean)
    aggregate, aggregate_std = mean_std(task_means)

    raw_agg: Dict[str, Optional[float]] = {}
    for key in EFFICIENCY_KEYS + ("tta_seconds", "throughput"):
        raw_agg[key] = mean_std([record.get(key) for record in raw_records])[0]

    return {
        "method": method,
        "display_name": METHOD_DISPLAY.get(method, method),
        "model": model,
        "sparsity": sparsity,
        "density": density_for_sparsity(sparsity),
        "primary": primary_mean,
        "primary_std": primary_std,
        "primary_seed_values": primary_per_seed,
        "accuracy": aggregate,
        "accuracy_std": aggregate_std,
        "per_task": per_task_summary,
        "task_means": task_means,
        "raw": raw_agg,
        "history": histories[0] if histories else None,
        "num_runs": len(raw_records),
        "summaries": summaries,
        "failures": failures,
    }


def build_rows(model: str = MODEL_DEFAULT, *, methods: Optional[Sequence[str]] = None,
               sparsities: Optional[Sequence[float]] = None,
               tasks: Optional[Sequence[str]] = None,
               seeds: Sequence[int] = DEFAULT_SEEDS,
               trainer: Optional[Callable[..., Dict[str, Any]]] = None,
               verbose: bool = True,
               config_overrides: Optional[Dict[str, Any]] = None
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the whole sparsity sweep for one model panel."""
    model = normalize_model(model)
    methods = [normalize_method(m) for m in (methods or default_methods(model))]
    sparsity_list = [normalize_sparsity(s) for s in (sparsities or DEFAULT_SPARSITIES)]
    tasks = [normalize_task(t) for t in (tasks or tasks_for_model(model))]

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for method in methods:
        # Dense methods (FT/LoRA) are a single point at sparsity 0 on the figure.
        sweep: Sequence[float] = [0.0] if method in ("ft", "lora") else sparsity_list
        for sparsity in sweep:
            row = collect_metrics(method, model, tasks, sparsity=float(sparsity),
                                  seeds=seeds, trainer=trainer, verbose=verbose,
                                  config_overrides=config_overrides)
            failures.extend(row.pop("failures", []) or [])
            rows.append(row)
    return rows, failures


# --------------------------------------------------------------------------- #
# Figure 3 series construction
# --------------------------------------------------------------------------- #

def figure3_series(rows: Sequence[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Build the Figure 3 data series (performance vs. relative inference eff.)."""
    model = normalize_model(model)
    reference = efficiency_reference(rows, model)
    ref_raw = reference.get("raw") or {}

    series: Dict[str, Any] = {
        "model": model,
        "x_axis": "relative inference efficiency (FT = 100%, lower = more efficient)",
        "x_metric": "inf_time",
        "y_axis": "task performance",
        "reference": reference,
        "points": [],
        "series": {},
        "note": "" if model in IN_SCOPE_MODELS else FIGURE3_NOTE,
    }

    for row in rows:
        rel = relative_efficiency(row.get("raw") or {}, ref_raw)
        method = normalize_method(row.get("method", ""))
        point = {
            "method": method,
            "display_name": row.get("display_name") or METHOD_DISPLAY.get(method, method),
            "sparsity": _to_float(row.get("sparsity")),
            "density": _to_float(row.get("density")),
            "density_label": density_label(row.get("density")),
            "accuracy": _to_float(row.get("accuracy")),
            "accuracy_std": _to_float(row.get("accuracy_std")),
            "primary": _to_float(row.get("primary")),
            "primary_std": _to_float(row.get("primary_std")),
            "per_task": {k: (v or {}).get("mean")
                         for k, v in (row.get("per_task") or {}).items()},
            "raw": {k: _to_float((row.get("raw") or {}).get(k))
                    for k in EFFICIENCY_KEYS + ("tta_seconds", "throughput")},
            "relative": {k: _to_float(rel.get(k))
                         for k in ("train_time", "train_mem", "inf_time", "inf_mem",
                                   "tta", "throughput")},
        }
        point["speedup"] = speedup_from_relative(point["relative"]["inf_time"])
        point["inf_mem_pct"] = point["relative"]["inf_mem"]
        series["points"].append(point)
        series["series"].setdefault(method, []).append({
            "sparsity": point["sparsity"],
            "density": point["density"],
            "accuracy": point["accuracy"],
            "accuracy_std": point["accuracy_std"],
            "x_inf_time_pct": point["relative"]["inf_time"],
            "speedup": point["speedup"],
            "inf_mem_pct": point["inf_mem_pct"],
        })

    for values in series["series"].values():
        values.sort(key=lambda p: (p["x_inf_time_pct"]
                                   if p["x_inf_time_pct"] is not None else 1e9,
                                   p["sparsity"] if p["sparsity"] is not None else 0.0))
    return series


def reference_points(model: str) -> List[Dict[str, Any]]:
    """Published anchor points (from Table 2) used as figure markers."""
    model = normalize_model(model)
    out: List[Dict[str, Any]] = []
    for sparsity_key, per_method in (FIGURE3_REFERENCES.get(model) or {}).items():
        for method, values in per_method.items():
            out.append({
                "model": model,
                "method": method,
                "display_name": METHOD_DISPLAY.get(method, method),
                "sparsity": float(sparsity_key),
                "speedup": speedup_from_relative(values.get("inf_time_pct")),
                "inf_time_pct": values.get("inf_time_pct"),
                "inf_mem_pct": values.get("inf_mem_pct"),
                "quality": values.get("quality"),
            })
    return out


def print_reference_points(model: str) -> None:
    """Print the published anchor points without training."""
    print(f"\nReference anchors (Figure 3, {normalize_model(model)}) -- Table 2:")
    print(f"{'Method':<20}{'Sparsity':>9}{'Inf Time %':>12}{'Speedup':>9}{'Inf Mem %':>10}")
    print("-" * 60)
    for point in reference_points(model):
        speedup = point["speedup"]
        inf_time = point["inf_time_pct"]
        inf_mem = point["inf_mem_pct"]
        print(f"{point['display_name']:<20}{point['sparsity'] * 100:>8.0f}%"
              f"{(inf_time if inf_time is not None else float('nan')):>12.1f}"
              f"{(speedup if speedup is not None else float('nan')):>9.2f}"
              f"{(inf_mem if inf_mem is not None else float('nan')):>10.1f}")
    print(f"\nNote: {FIGURE3_NOTE}\n")


def compare_to_reference(series: Dict[str, Any], *, tolerance: float = 1.0
                         ) -> List[Dict[str, Any]]:
    """Diff measured sweep points against published anchor points."""
    model = normalize_model(series.get("model", MODEL_DEFAULT))
    refs = {(p["method"], round(p["sparsity"], 3)): p for p in reference_points(model)}
    diffs: List[Dict[str, Any]] = []
    for point in series.get("points", []):
        key = (point["method"], round(point.get("sparsity") or 0.0, 3))
        ref = refs.get(key)
        if not ref:
            continue
        measured = point.get("relative", {}).get("inf_time")
        published = ref.get("inf_time_pct")
        delta = None
        if measured is not None and published is not None:
            delta = measured - published
        diffs.append({
            "method": point["method"],
            "display_name": point.get("display_name"),
            "sparsity": point.get("sparsity"),
            "measured_inf_time_pct": measured,
            "reference_inf_time_pct": published,
            "delta": delta,
            "within_tolerance": None if delta is None else abs(delta) <= tolerance,
            "measured_accuracy": point.get("accuracy"),
            "reference_quality": ref.get("quality"),
        })
    return diffs


# --------------------------------------------------------------------------- #
# Rendering / persistence
# --------------------------------------------------------------------------- #

def format_figure3_table(series: Dict[str, Any], *, digits: int = 1) -> str:
    """Render the sweep as a markdown-style table (the paper's figure data)."""
    model = series.get("model", MODEL_DEFAULT)
    header = (f"{'Method':<20}{'Sparsity':>9}{'Density':>9}{'Perf.':>9}"
              f"{'InfTime%':>10}{'Speedup':>9}{'InfMem%':>9}")
    lines = [f"## Figure 3 data -- {model}", "", header, "-" * 75]

    def cell(value: Any, fmt: str) -> str:
        return "-" if value is None else format(value, fmt)

    for point in series.get("points", []):
        sparsity = point.get("sparsity")
        sparsity_text = "-" if sparsity is None else f"{sparsity * 100:.0f}%"
        lines.append(
            f"{str(point.get('display_name') or ''):<20}"
            f"{sparsity_text:>9}"
            f"{str(point.get('density_label') or '-'):>9}"
            f"{cell(point.get('accuracy'), f'.{digits}f'):>9}"
            f"{cell(point.get('relative', {}).get('inf_time'), f'.{digits}f'):>10}"
            f"{cell(point.get('speedup'), '.2f'):>9}"
            f"{cell(point.get('inf_mem_pct'), f'.{digits}f'):>9}"
        )
    if series.get("note"):
        lines += ["", f"Note: {series['note']}"]
    return "\n".join(lines)


def plot_figure3(series: Dict[str, Any], path: Optional[str] = None) -> Optional[str]:
    """Render the Figure 3 scatter/line plot; returns the saved path (or ``None``)."""
    if not path:
        return None
    try:  # matplotlib is optional
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.warn(f"matplotlib unavailable, skipping plot ({exc})")
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for method, points in (series.get("series") or {}).items():
        xs = [p["x_inf_time_pct"] for p in points if p["x_inf_time_pct"] is not None]
        ys = [p["accuracy"] for p in points if p["x_inf_time_pct"] is not None]
        if not xs or not ys:
            continue
        ax.plot(xs, ys, marker="o", label=METHOD_DISPLAY.get(method, method))
    for ref in reference_points(series.get("model", MODEL_DEFAULT)):
        if ref.get("quality") is None or ref.get("inf_time_pct") is None:
            continue
        ax.scatter([ref["inf_time_pct"]], [ref["quality"]], marker="x",
                   label=f"{ref['display_name']} (paper)")
    ax.set_xlabel(series.get("x_axis", "relative inference efficiency"))
    ax.set_ylabel(series.get("y_axis", "task performance"))
    ax.set_title(f"Figure 3 -- {series.get('model', '')}")
    ax.invert_xaxis()  # lower relative inference time = more efficient
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_csv(series: Dict[str, Any], path: str) -> str:
    """Persist the sweep as CSV (one row per point)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = ["model", "method", "display_name", "sparsity", "density", "accuracy",
              "accuracy_std", "inf_time_pct", "speedup", "inf_mem_pct",
              "train_time_pct", "train_mem_pct"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in series.get("points", []):
            rel = point.get("relative", {}) or {}
            writer.writerow({
                "model": series.get("model", ""),
                "method": point.get("method", ""),
                "display_name": point.get("display_name", ""),
                "sparsity": point.get("sparsity"),
                "density": point.get("density"),
                "accuracy": point.get("accuracy"),
                "accuracy_std": point.get("accuracy_std"),
                "inf_time_pct": rel.get("inf_time"),
                "speedup": point.get("speedup"),
                "inf_mem_pct": point.get("inf_mem_pct"),
                "train_time_pct": rel.get("train_time"),
                "train_mem_pct": rel.get("train_mem"),
            })
    return path


def save_results(payload: Dict[str, Any], path: str) -> str:
    """Persist the payload as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)
    return path


def default_output_path(model: str, name: str = "figure3") -> str:
    """Default JSON output location for a model panel."""
    return os.path.join("outputs", "figure3", f"{name}_{normalize_model(model)}.json")


def summary_rows(series: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the series into report rows (one per (method, sparsity))."""
    rows: List[Dict[str, Any]] = []
    for method, points in (series.get("series") or {}).items():
        for point in points:
            rows.append({
                "model": series.get("model"),
                "method": method,
                "display_name": METHOD_DISPLAY.get(method, method),
                "sparsity": point.get("sparsity"),
                "density": point.get("density"),
                "accuracy": point.get("accuracy"),
                "accuracy_std": point.get("accuracy_std"),
                "inf_time_pct": point.get("x_inf_time_pct"),
                "speedup": point.get("speedup"),
                "inf_mem_pct": point.get("inf_mem_pct"),
            })
    return rows


# --------------------------------------------------------------------------- #
# Top level driver
# --------------------------------------------------------------------------- #

def run_figure3(model: str = MODEL_DEFAULT, *, models: Optional[Sequence[str]] = None,
                methods: Optional[Sequence[str]] = None,
                sparsities: Optional[Sequence[float]] = None,
                tasks: Optional[Sequence[str]] = None,
                seeds: Sequence[int] = DEFAULT_SEEDS,
                config_overrides: Optional[Dict[str, Any]] = None,
                trainer: Optional[Callable[..., Dict[str, Any]]] = None,
                output_path: Optional[str] = None,
                plot_path: Optional[str] = None,
                dry_run: bool = False,
                verbose: bool = True) -> Dict[str, Any]:
    """Sweep sparsity and build the Figure 3 payload for one or more models."""
    model_list = split_models(models) if models else [normalize_model(model)]
    for name in model_list:
        if name not in IN_SCOPE_MODELS:
            raise ValueError(
                f"model {name!r} is outside the reproduction scope: {FIGURE3_NOTE}"
            )
    sparsity_list = parse_sparsities(sparsities)
    seed_list = list(seeds)

    payload: Dict[str, Any] = {
        "figure": "figure3",
        "caption": ("Task performance vs. relative inference efficiency on RoBERTa "
                    "and T5 models with APT and baselines"),
        "sparsities": list(sparsity_list),
        "seeds": seed_list,
        "models": {},
        "rows": {},
        "reference": FIGURE3_REFERENCES,
        "note": FIGURE3_NOTE,
        "failures": [],
        "plot_path": None,
        "output_path": None,
    }

    for name in model_list:
        if dry_run:
            payload["models"][name] = {
                "model": name,
                "points": [],
                "series": {},
                "reference_points": reference_points(name),
                "comparison": [],
                "note": "dry-run: no training performed",
            }
            payload["rows"][name] = []
            continue

        method_list = [normalize_method(m) for m in (methods or default_methods(name))]
        rows, failures = build_rows(name, methods=method_list, sparsities=sparsity_list,
                                    tasks=tasks or tasks_for_model(name), seeds=seed_list,
                                    trainer=trainer, verbose=verbose,
                                    config_overrides=config_overrides)
        payload["failures"].extend(failures)
        series = figure3_series(rows, name)
        series["reference_points"] = reference_points(name)
        series["comparison"] = compare_to_reference(series)
        series["proxies"] = [proxy_note(m) for m in method_list if proxy_note(m)]
        if plot_path:
            per_model_plot = plot_path if len(model_list) == 1 else plot_path.replace(
                ".png", f"_{name}.png"
            )
            series["plot_path"] = plot_figure3(series, per_model_plot)
            payload["plot_path"] = series["plot_path"] or payload["plot_path"]
        payload["models"][name] = series
        payload["rows"][name] = summary_rows(series)

    if output_path is None:
        output_path = default_output_path(model_list[0])
    payload["output_path"] = save_results(payload, output_path)
    payload["csv_path"] = save_csv(payload["models"][model_list[0]],
                                   output_path.replace(".json", ".csv")) \
        if payload["models"].get(model_list[0], {}).get("points") else None

    if verbose:
        for name in model_list:
            series = payload["models"][name]
            if series.get("points"):
                print("\n" + format_figure3_table(series))
                for diff in series.get("comparison", []) or []:
                    if diff.get("delta") is None:
                        continue
                    flag = "OK  " if diff.get("within_tolerance") else "DIFF"
                    print(f"  [{flag}] {diff['display_name']} @ "
                          f"{(diff['sparsity'] or 0) * 100:.0f}%: measured "
                          f"{diff['measured_inf_time_pct']:.1f}% vs paper "
                          f"{diff['reference_inf_time_pct']:.1f}%")
        print(f"[figure3] results -> {payload['output_path']}")
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=("Reproduce Figure 3 of APT: task performance vs. relative "
                     "inference efficiency across sparsity (RoBERTa, T5)."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=MODEL_DEFAULT,
                        help="model family (roberta|t5); LLaMA is out of scope")
    parser.add_argument("--models", nargs="*", default=None,
                        help="multiple model panels, e.g. --models roberta t5")
    parser.add_argument("--methods", nargs="*", default=None,
                        help=f"methods to sweep (default: {list(METHOD_ORDER)})")
    parser.add_argument("--sparsities", nargs="*", type=float, default=None,
                        help=f"sparsity sweep (default: {list(DEFAULT_SPARSITIES)})")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="tasks averaged for the y-axis")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help=f"random seeds (default: {list(DEFAULT_SEEDS)})")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--plot", default=None, help="write a PNG figure to this path")
    parser.add_argument("--config", default=None, help="extra YAML config to merge")
    parser.add_argument("--dry-run", action="store_true",
                        help="print reference anchors without training")
    parser.add_argument("--print-reference", action="store_true",
                        help="print published anchor points and exit")
    parser.add_argument("--self-test", action="store_true",
                        help="run dependency-light checks")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect CLI-provided config overrides."""
    overrides: Dict[str, Any] = {}
    explicit = getattr(args, "config", None)
    if explicit:
        overrides.update(load_yaml(explicit))
    return overrides


def main(config: Optional[Dict[str, Any]] = None, argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``main.py figure3`` and direct CLI usage."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        return 0 if _self_test() else 1

    config = dict(config or {})
    model = normalize_model(config.get("model", args.model))
    if args.print_reference:
        for name in (split_models(args.models) if args.models else [model]):
            print_reference_points(name)
        return 0

    overrides = dict(cli_overrides(args))
    explicit = dict(config)
    explicit.pop("model", None)
    overrides.update(explicit)

    try:
        payload = run_figure3(
            model,
            models=split_models(args.models) if args.models else None,
            methods=split_methods(args.methods) if args.methods else None,
            sparsities=parse_sparsities(args.sparsities) if args.sparsities else None,
            tasks=[normalize_task(t) for t in args.tasks] if args.tasks else None,
            seeds=parse_seeds(args.seeds),
            config_overrides=overrides or None,
            output_path=args.output,
            plot_path=args.plot,
            dry_run=bool(args.dry_run),
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"[figure3] error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for name, series in (payload.get("models") or {}).items():
            print(f"\n[dry-run] Figure 3 reference anchors for {name}:")
            for point in series.get("reference_points", []):
                speedup = point.get("speedup")
                speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
                print(f"  {point['display_name']:<20} sparsity={point['sparsity'] * 100:>3.0f}%  "
                      f"inf_time={point['inf_time_pct']}%  speedup={speedup_text}")
        print(f"\n[dry-run] note: {FIGURE3_NOTE}")
    return 0


# --------------------------------------------------------------------------- #
# Self test (dependency-light)
# --------------------------------------------------------------------------- #

def _self_test() -> bool:
    """Validate alias handling, series construction and normalisation math."""
    ok = True

    def check(condition: bool, message: str) -> None:
        nonlocal ok
        if not condition:
            ok = False
            print(f"  FAIL: {message}")

    # Aliases ------------------------------------------------------------- #
    check(normalize_method("LoRA+Prune") == "lora_prune", "method alias LoRA+Prune")
    check(normalize_method("Prune+Distill") == "prune_distill", "method alias Prune+Distill")
    check(normalize_method("LoRA+Distill") == "lora_prune_distill", "alias LoRA+Distill")
    check(normalize_method("apt") == "apt", "method alias apt")
    check(normalize_task("SST-2") == "sst2", "task alias SST-2")
    check(normalize_task("cnn_dailymail") == "cnndm", "task alias cnn_dailymail")
    check(normalize_model("roberta-base") == "roberta", "model alias roberta-base")
    check(normalize_model("t5-base") == "t5", "model alias t5-base")
    check(normalize_model("llama-2-7b") == "llama", "model alias llama-2-7b")

    # Sparsity / density -------------------------------------------------- #
    check(normalize_sparsity(60) == 0.6, "sparsity percent handling")
    check(normalize_sparsity(density=0.4) == 0.6, "density conversion")
    check(abs(density_for_sparsity(0.6) - 0.4) < 1e-9, "density for sparsity")
    check(sparsity_for_method("ft", 0.6) == 0.0, "FT is dense")
    check(sparsity_for_method("lora", 0.6) == 0.0, "LoRA is dense")
    check(sparsity_for_method("apt", 0.6) == 0.6, "APT uses target sparsity")
    check(density_label(0.4) == "40%", "density label")

    # Table 6 wiring ------------------------------------------------------ #
    check(table6_group_for("t5", "cnndm") == "cnndm", "Table 6 CNN/DM group")
    check(table6_group_for("roberta", "mnli") == "glue-big", "Table 6 GLUE-big group")
    check(table6_group_for("bert", "cola") == "glue-small", "Table 6 GLUE-small group")
    check(table6_group_for("roberta", "squad_v2") == "squad", "Table 6 SQuAD group")

    # Config -------------------------------------------------------------- #
    cfg = build_config("roberta", "sst2", 0.6, 42)
    check(abs(cfg["target_sparsity"] - 0.6) < 1e-9, "config target sparsity")
    check(cfg["initial_rank"] == 8, "config initial rank")
    check(abs(cfg["scaling"] - 2.0) < 1e-12, "config scaling")
    check(abs(cfg["mask_alpha"] - 0.01) < 1e-12, "config mask alpha")
    check(abs(cfg["ema_beta"] - 0.85) < 1e-12, "config EMA beta")
    check(cfg["tau"] == 4, "config tau")
    check(abs(cfg["top_fraction"] - 0.5) < 1e-12, "config top fraction")
    check(abs(cfg["pred_distill_weight"] - 1.0) < 1e-12, "GLUE pred distill weight")
    check(abs(cfg["layer_distill_weight"] - 0.9) < 1e-12, "GLUE layer distill weight")
    cnndm_cfg = build_config("t5", "cnndm", 0.6, 42)
    check(abs(cnndm_cfg["pred_distill_weight"] - 0.1) < 1e-12, "CNN/DM pred weight")
    check(abs(cnndm_cfg["layer_distill_weight"] - 0.9) < 1e-12, "CNN/DM layer weight")
    ft_cfg = build_config("roberta", "sst2", 0.6, 42, method="ft")
    check(ft_cfg["target_sparsity"] == 0.0, "FT config is dense")
    check(ft_cfg["use_distillation"] is False, "FT config disables distillation")
    lora_prune_cfg = build_config("roberta", "mnli", 0.6, 42, method="lora_prune")
    check(lora_prune_cfg["use_distillation"] is False, "LoRA+Prune config has no distillation")

    # Efficiency normalisation from the Table 11 absolute numbers:
    # RoBERTa APT = 752s / 127s = 592.1% train time, 41.3% inference time.
    ref_raw = {
        "train_time": TABLE11_FT_REFERENCE["roberta"]["train_time_s"],
        "train_mem": TABLE11_FT_REFERENCE["roberta"]["train_peak_mem_mb"],
        "inf_time": TABLE11_FT_REFERENCE["roberta"]["inf_time_ms"],
        "inf_mem": TABLE11_FT_REFERENCE["roberta"]["inf_mem_mb"],
        "tta_seconds": TABLE11_FT_REFERENCE["roberta"]["tta_seconds"],
        "throughput": None,
    }
    rel = relative_efficiency(
        {"train_time": 752.0, "train_mem": 1890.0, "inf_time": 91.3,
         "inf_mem": 904.0, "tta_seconds": None, "throughput": None},
        ref_raw,
    )
    check(abs(rel["train_time"] - 592.1) < 0.05, f"APT train time ({rel['train_time']:.1f})")
    check(abs(rel["train_mem"] - 70.1) < 0.05, f"APT train mem ({rel['train_mem']:.1f})")
    check(abs(rel["inf_time"] - 41.3) < 0.05, f"APT inf time ({rel['inf_time']:.1f})")
    check(abs(rel["inf_mem"] - 78.1) < 0.05, f"APT inf mem ({rel['inf_mem']:.1f})")
    # APT TTA proxy falls back to total training time: 752/127 = 592.1%.
    check(abs(rel["tta"] - 592.1) < 0.05, f"APT TTA ({rel['tta']:.1f})")
    check(abs(rel["throughput"] - 242.1) < 0.2, f"APT speedup ({rel['throughput']:.1f})")

    # T5: FT 366s / 7217 MB / 248.1 ms / 2347 MB; APT 1774s / 5332 MB / 185 ms.
    t5_rel = relative_efficiency(
        {"train_time": 1774.0, "train_mem": 5332.0, "inf_time": 185.0,
         "inf_mem": 1913.0, "tta_seconds": None, "throughput": None},
        {"train_time": 366.0, "train_mem": 7217.0, "inf_time": 248.1,
         "inf_mem": 2347.0, "tta_seconds": 366.0, "throughput": None},
    )
    check(abs(t5_rel["train_time"] - 484.7) < 0.05, f"T5 APT train time ({t5_rel['train_time']:.1f})")
    check(abs(t5_rel["inf_time"] - 74.6) < 0.05, f"T5 APT inf time ({t5_rel['inf_time']:.1f})")
    check(abs(t5_rel["inf_mem"] - 81.5) < 0.05, f"T5 APT inf mem ({t5_rel['inf_mem']:.1f})")

    speedup = speedup_from_relative(100.0 / 2.42)
    check(speedup is not None and 2.3 < speedup < 2.5, "speedup conversion")
    check(speedup_from_relative(None) is None, "speedup of None")

    # TTA interpolation: reaches 97% of the reference between the two evals.
    tta = compute_tta([{"elapsed": 340.0, "metric": 90.0},
                       {"elapsed": 780.0, "metric": 95.1}], reference=95.1, fraction=0.97)
    check(tta is not None and 340.0 < tta < 780.0,
          f"TTA interpolation within bracket (got {tta})")
    check(compute_tta([], 95.1) is None, "TTA empty history")
    check(abs(compute_tta([{"elapsed": 127.0, "metric": 95.1}], 95.1) - 127.0) < 1e-9,
          "TTA immediate hit")

    # Parsing / defaults -------------------------------------------------- #
    check(parse_sparsities([0.2, 0.6]) == (0.2, 0.6), "sparsity parsing")
    check(parse_sparsities("0.2,0.6") == (0.2, 0.6), "sparsity string parsing")
    check(parse_sparsities(None) == DEFAULT_SPARSITIES, "sparsity defaults")
    check(parse_seeds([42, 43]) == (42, 43), "seed parsing")
    check(parse_seeds(None) == DEFAULT_SEEDS, "seed defaults")
    check(parse_seeds(42) == (42,), "single seed")
    check(split_methods(None) == list(METHOD_ORDER), "method defaults")
    check(split_models("roberta,t5") == ["roberta", "t5"], "model list parsing")
    check(default_methods("t5") == ["ft", "lora", "lora_prune", "apt"], "T5 method set")
    check("lora_prune_distill" in default_methods("roberta"), "RoBERTa method set")
    check(tasks_for_model("roberta") == ["sst2", "mnli"], "task set")

    # Aggregation --------------------------------------------------------- #
    check(abs(glue_average({"sst2": 94.5, "mnli": 86.4}) - 90.45) < 1e-9, "GLUE average")
    check(glue_average([None, None]) is None, "GLUE average of Nones")
    mean, std = mean_std([1.0, 3.0])
    check(abs(mean - 2.0) < 1e-12 and abs(std - 1.0) < 1e-12, "mean/std")
    check(mean_std([None]) == (None, None), "mean/std of None")

    # Series construction with synthetic rows (no training needed) -------- #
    synthetic_rows = [
        {"method": "ft", "display_name": "FT", "sparsity": 0.0, "density": 1.0,
         "accuracy": 91.2, "accuracy_std": 0.0, "primary": 91.2, "primary_std": 0.0,
         "per_task": {"sst2": {"mean": 94.8}, "mnli": {"mean": 87.6}},
         "raw": {"train_time": 127.0, "train_mem": 2696.0, "inf_time": 220.8,
                 "inf_mem": 1157.0, "tta_seconds": 127.0, "throughput": None}},
        {"method": "apt", "display_name": "APT", "sparsity": 0.6, "density": 0.4,
         "accuracy": 90.45, "accuracy_std": 0.1, "primary": 90.45, "primary_std": 0.1,
         "per_task": {"sst2": {"mean": 94.5}, "mnli": {"mean": 86.4}},
         "raw": {"train_time": 752.0, "train_mem": 1890.0, "inf_time": 91.3,
                 "inf_mem": 904.0, "tta_seconds": None, "throughput": None}},
    ]
    series = figure3_series(synthetic_rows, "roberta")
    check(len(series["points"]) == 2, "series point count")
    check(series["reference"]["source"] == "table11", "series uses Table 11 reference")
    apt_point = [p for p in series["points"] if p["method"] == "apt"][0]
    check(abs(apt_point["relative"]["inf_time"] - 41.3) < 0.05, "series APT inf time")
    check(abs(apt_point["relative"]["inf_mem"] - 78.1) < 0.05, "series APT inf mem")
    check(apt_point["speedup"] is not None and apt_point["speedup"] > 2.3, "series speedup")
    check("apt" in series["series"], "series grouping")
    check(series["note"] == "", "in-scope series note empty")

    table = format_figure3_table(series)
    check("Figure 3 data" in table and "APT" in table, "figure table rendering")
    check("40%" in table, "figure table density label")

    refs = reference_points("roberta")
    check(len(refs) > 0, "reference points")
    check(any(p["method"] == "apt" and abs((p["sparsity"] or 0) - 0.6) < 1e-9 for p in refs),
          "APT anchor at 60% sparsity")

    diffs = compare_to_reference(series, tolerance=1.0)
    check(isinstance(diffs, list) and len(diffs) == 2, "comparison diff count")
    check(all(d["delta"] is not None for d in diffs), "comparison deltas")

    rows = summary_rows(series)
    check(len(rows) == 2, "summary rows")

    # Summary parsing ----------------------------------------------------- #
    parsed = normalise_from_summary({
        "efficiency": {"train_time_s": 752.0, "train_peak_mem_mb": 1890.0,
                       "inf_time_ms": 91.3, "inf_mem_mb": 904.0}})
    check(abs(parsed["train_time"] - 752.0) < 1e-9, "summary train time alias")
    check(abs(parsed["inf_mem"] - 904.0) < 1e-9, "summary inf mem alias")
    check(primary_from_summary({"metrics": {"sst2": {"accuracy": 94.5}}}, "sst2") == 94.5,
          "primary from summary metrics")
    check(primary_from_summary({"accuracy": 94.5}, "sst2") == 94.5, "primary fallback")

    # Out-of-scope guard -------------------------------------------------- #
    try:
        run_figure3("roberta", models=["llama"], dry_run=True, verbose=False)
        check(False, "LLaMA should be rejected")
    except ValueError:
        check(True, "LLaMA scope guard")

    # Dry-run payload ----------------------------------------------------- #
    payload = run_figure3("t5", sparsities=[0.6], dry_run=True, verbose=False,
                          output_path=os.path.join("outputs", "figure3", "_selftest.json"))
    check("t5" in payload["models"], "dry-run payload")
    check(bool(payload["models"]["t5"]["reference_points"]), "dry-run reference anchors")
    check(payload["note"] == FIGURE3_NOTE, "dry-run note")

    # JSON safety --------------------------------------------------------- #
    check(json.dumps(_json_safe(payload)) is not None, "payload JSON-serialisable")

    print("self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(main())
