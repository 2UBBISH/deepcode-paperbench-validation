#!/usr/bin/env python
"""Table 8 driver: RoBERTa GLUE comparison of APT vs the LoRA+Distill baseline.

Reproduces Appendix D.2 / Table 8 of *APT: Adaptive Pruning and Tuning Pretrained
Language Models for Efficient Training and Inference*::

    Sparsity  Method         MNLI  QQP   QNLI  SST2  CoLA  MRPC  RTE  GLUE Avg
    0%        FT             87.6  91.9  92.8  95.2  91.2  90.2  78.7  89.7
    0%        LoRA           87.5  90.8  93.3  95.0  63.4  89.7  72.1  84.5
    40%       LoRA+Distill   84.2  88.3  90.1  91.9  49.9  86.8  68.6  80.0
    40%       APT            86.4  90.9  92.3  94.5  56.5  92.3  74.4  83.9

Paper facts used here (verbatim sources):

* Table 8 (Appendix D.2) reports seven GLUE tasks -- MNLI/QQP/QNLI/SST2 (GLUE-big)
  and CoLA/MRPC/RTE (GLUE-small).  The paper states: "the results of STS-B cannot
  be reproduced when conducting CoFi distillation with LoRA parameters tuned only,
  so we exclude the comparison on STS-B."
* Appendix A (Table 6): GLUE-big and GLUE-small both use lr 2e-4, batch size 32,
  40 epochs and 20 distill epochs; adapter ranks start at 8, scaling is fixed at 2.
* Appendix A: "we first prune and train the LM with the self-distillation
  objective, and then fine-tune the pruned LM to recover its end-task performance"
  and the cubic sparsity schedule
  ``gamma_t = gamma_T + (1 - gamma_T) * (1 - t/T)^3``.
* Section 5.3 / Appendix I: efficiency is expressed relative to fine-tuning
  (FT = 100 %) with time-to-accuracy measured against 97 % of the FT metric.

This script is a *driver only*: it never re-implements APT or a baseline, it
assembles per-cell configs and delegates training to ``scripts.train_apt`` (APT)
and ``scripts.train_baseline`` / ``apt.baselines.*`` (FT, LoRA, LoRA+Distill).

Usage::

    python scripts/run_table8_glue.py --dry-run            # print paper numbers
    python scripts/run_table8_glue.py                      # full reproduction
    python scripts/run_table8_glue.py --methods apt,lora_distill --seeds 42
    python scripts/run_table8_glue.py --self-test
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

MODEL_DEFAULT = "roberta"

#: Table 8 task order (STS-B intentionally absent, see Appendix D.2).
DEFAULT_TASKS: Tuple[str, ...] = ("mnli", "qqp", "qnli", "sst2", "cola", "mrpc", "rte")

#: Tasks that the paper excludes from Table 8.
EXCLUDED_TASKS: Tuple[str, ...] = ("stsb", "sts-b")

#: Rows of Table 8 grouped by sparsity: (method key, sparsity).
METHOD_ORDER: Tuple[str, ...] = ("ft", "lora", "lora_distill", "apt")

#: Method -> sparsity used in Table 8.
SPARSITY_ORDER: Dict[Tuple[str, float], int] = {
    ("ft", 0.0): 0,
    ("lora", 0.0): 1,
    ("lora_distill", 0.40): 2,
    ("apt", 0.40): 3,
}

METHOD_DISPLAY: Dict[str, str] = {
    "ft": "FT",
    "lora": "LoRA",
    "lora_distill": "LoRA+Distill",
    "lora_prune_distill": "LoRA+Distill",
    "lora+distill": "LoRA+Distill",
    "cofi_lora": "LoRA+Distill",
    "apt": "APT",
}

METHOD_ALIASES: Dict[str, str] = {
    "ft": "ft",
    "finetune": "ft",
    "fine_tune": "ft",
    "full": "ft",
    "full_finetuning": "ft",
    "lora": "lora",
    "lora_tuning": "lora",
    "loratuning": "lora",
    "lora_distill": "lora_distill",
    "lora+distill": "lora_distill",
    "lora-distill": "lora_distill",
    "cofi": "lora_distill",
    "cofi_lora": "lora_distill",
    "lora_prune_distill": "lora_distill",
    "lora+prune+distill": "lora_distill",
    "lora_pruned_distill": "lora_distill",
    "apt": "apt",
    "ours": "apt",
}

TASK_ALIASES: Dict[str, str] = {
    "mnli": "mnli",
    "mnli-mm": "mnli",
    "mnli_matched": "mnli",
    "mnli_mismatched": "mnli",
    "qqp": "qqp",
    "qnli": "qnli",
    "sst2": "sst2",
    "sst-2": "sst2",
    "sst_2": "sst2",
    "cola": "cola",
    "mrpc": "mrpc",
    "rte": "rte",
    "stsb": "stsb",
    "sts-b": "stsb",
    "sts_b": "stsb",
}

MODEL_ALIASES: Dict[str, str] = {
    "roberta": "roberta",
    "roberta-base": "roberta",
    "roberta_base": "roberta",
    "bert": "bert",
    "bert-base": "bert",
    "bert-base-uncased": "bert",
    "t5": "t5",
    "t5-base": "t5",
}

MODEL_NAMES: Dict[str, str] = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "t5": "t5-base",
}

TASK_LABELS: Dict[str, str] = {
    "mnli": "MNLI",
    "qqp": "QQP",
    "qnli": "QNLI",
    "sst2": "SST2",
    "cola": "CoLA",
    "mrpc": "MRPC",
    "rte": "RTE",
    "stsb": "STS-B",
}

#: GLUE split used in Appendix A: big (MNLI, SST2, QNLI, QQP) / small (MRPC, CoLA, RTE, STSB).
GLUE_BIG_TASKS: Tuple[str, ...] = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS: Tuple[str, ...] = ("mrpc", "cola", "rte", "stsb")

#: Table 6 hyper-parameters (Appendix A).  Both GLUE groups share the same column.
TABLE6_GROUPS: Dict[str, Dict[str, Any]] = {
    "glue-big": {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "max_seq_length": 128,
    },
    "glue-small": {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "max_seq_length": 128,
    },
    "squad": {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "max_seq_length": 384,
    },
    "cnndm": {
        "learning_rate": 1.0e-4,
        "batch_size": 16,
        "epochs": 16,
        "distill_epochs": 6,
        "max_seq_length": 512,
        "max_target_length": 128,
    },
}

#: Appendix A: initial adapter rank 8, static scaling factor 2.
INITIAL_RANK = 8
SCALING = 2.0
#: Appendix A: pruning masks are annealed by alpha < 1 instead of zeroed instantly.
MASK_ALPHA = 0.01
#: Appendix A / Section 4.2: exponential moving average of salience.
EMA_BETA = 0.85
#: Section 4.4: number of block-wise sampled teacher layers.
TAU = 4
#: Section 4.4 / Addendum: GLUE uses L_pred + 0.9 * L_layer.
GLUE_PRED_DISTILL_WEIGHT = 1.0
GLUE_LAYER_DISTILL_WEIGHT = 0.9

DEFAULT_SPARSITY = 0.40
TTA_FRACTION = 0.97
RELATIVE_SCALE = 100.0
SMALL_MODEL_INF_BATCH = 128
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)

EFFICIENCY_KEYS: Tuple[str, ...] = ("train_time", "train_mem", "inf_time", "inf_mem")
EFFICIENCY_DISPLAY: Dict[str, str] = {
    "train_time": "Train Time",
    "train_mem": "Train Mem",
    "inf_time": "Inf Time",
    "inf_mem": "Inf Mem",
}

#: Published Table 8 numbers (Appendix D.2).  ``avg`` is the paper's GLUE average.
TABLE8_REFERENCES: Dict[str, Dict[str, Any]] = {
    "ft": {
        "sparsity": 0.0,
        "display": "FT",
        "metrics": {
            "mnli": 87.6,
            "qqp": 91.9,
            "qnli": 92.8,
            "sst2": 95.2,
            "cola": 91.2,
            "mrpc": 90.2,
            "rte": 78.7,
        },
        "avg": 89.7,
    },
    "lora": {
        "sparsity": 0.0,
        "display": "LoRA",
        "metrics": {
            "mnli": 87.5,
            "qqp": 90.8,
            "qnli": 93.3,
            "sst2": 95.0,
            "cola": 63.4,
            "mrpc": 89.7,
            "rte": 72.1,
        },
        "avg": 84.5,
    },
    "lora_distill": {
        "sparsity": 0.40,
        "display": "LoRA+Distill",
        "metrics": {
            "mnli": 84.2,
            "qqp": 88.3,
            "qnli": 90.1,
            "sst2": 91.9,
            "cola": 49.9,
            "mrpc": 86.8,
            "rte": 68.6,
        },
        "avg": 80.0,
    },
    "apt": {
        "sparsity": 0.40,
        "display": "APT",
        "metrics": {
            "mnli": 86.4,
            "qqp": 90.9,
            "qnli": 92.3,
            "sst2": 94.5,
            "cola": 56.5,
            "mrpc": 92.3,
            "rte": 74.4,
        },
        "avg": 83.9,
    },
}

#: Appendix D.1 (Table 7 of the paper) -- same seven tasks/columns, used only as a
#: cross-check of the task layout when the caller asks for it.
APPENDIX_D1_REFERENCES: Dict[str, Dict[str, Any]] = {
    "map@50": {"density": 0.50, "display": "MaP", "metrics": {"mnli": 83.6, "qqp": 87.8, "qnli": 91.5, "sst2": 91.0, "cola": 60.1, "stsb": 89.8, "mrpc": 90.7, "rte": 67.2}, "avg": 82.7},
    "mvp@50": {"density": 0.50, "display": "MvP", "metrics": {"mnli": 82.3, "qqp": 87.3, "qnli": 90.8, "sst2": 90.8, "cola": 57.7, "stsb": 89.4, "mrpc": 91.1, "rte": 67.2}, "avg": 82.1},
    "pst@50": {"density": 0.50, "display": "PST", "metrics": {"mnli": 81.0, "qqp": 85.8, "qnli": 89.8, "sst2": 91.3, "cola": 57.6, "stsb": 84.6, "mrpc": 90.7, "rte": 67.9}, "avg": 81.0},
    "lrp@50": {"density": 0.50, "display": "LRP", "metrics": {"mnli": 82.4, "qqp": 87.2, "qnli": 89.6, "sst2": 90.9, "cola": 54.1, "stsb": 88.7, "mrpc": 89.8, "rte": 69.3}, "avg": 82.2},
    "apt@50": {"density": 0.50, "display": "APT", "metrics": {"mnli": 82.8, "qqp": 90.1, "qnli": 90.1, "sst2": 92.7, "cola": 59.6, "stsb": 88.3, "mrpc": 91.8, "rte": 70.4}, "avg": 83.2},
    "map@10": {"density": 0.10, "display": "MaP", "metrics": {"mnli": 78.2, "qqp": 83.2, "qnli": 84.1, "sst2": 85.4, "cola": 27.9, "stsb": 82.3, "mrpc": 80.5, "rte": 50.1}, "avg": 71.4},
    "mvp@10": {"density": 0.10, "display": "MvP", "metrics": {"mnli": 80.1, "qqp": 84.4, "qnli": 87.2, "sst2": 87.2, "cola": 28.6, "stsb": 84.3, "mrpc": 84.1, "rte": 57.6}, "avg": 74.2},
    "pst@10": {"density": 0.10, "display": "PST", "metrics": {"mnli": 79.6, "qqp": 86.1, "qnli": 86.6, "sst2": 89.0, "cola": 38.0, "stsb": 81.3, "mrpc": 83.6, "rte": 63.2}, "avg": 75.9},
    "lrp@10": {"density": 0.10, "display": "LRP", "metrics": {"mnli": 79.4, "qqp": 86.0, "qnli": 85.3, "sst2": 89.1, "cola": 35.6, "stsb": 83.3, "mrpc": 84.4, "rte": 62.8}, "avg": 75.7},
    "apt@10": {"density": 0.10, "display": "APT", "metrics": {"mnli": 78.8, "qqp": 89.4, "qnli": 85.5, "sst2": 90.0, "cola": 30.9, "stsb": 86.3, "mrpc": 88.2, "rte": 65.3}, "avg": 76.8},
}

#: Appendix I (Table 11) raw FT efficiency for RoBERTa-base, used to normalise
#: efficiency when the FT row was not trained in this run.
FT_RAW_EFFICIENCY: Dict[str, Optional[float]] = {
    "tta_seconds": 127.0,
    "train_peak_mem_mb": 2696.0,
    "inf_time_ms": 220.8,
    "inf_mem_mb": 1157.0,
}

#: Appendix I (Table 11) raw APT efficiency for RoBERTa-base at 60 % sparsity.
APT_RAW_EFFICIENCY: Dict[str, Optional[float]] = {
    "tta_seconds": 752.0,
    "train_peak_mem_mb": 1890.0,
    "inf_time_ms": 91.3,
    "inf_mem_mb": 904.0,
}

#: Primary metric reported per GLUE task (follows HuggingFace GLUE conventions used
#: by CoFi, which the paper follows for hyper-parameters/data splits).
PRIMARY_METRIC: Dict[str, str] = {
    "mnli": "accuracy",
    "qqp": "f1",
    "qnli": "accuracy",
    "sst2": "accuracy",
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "rte": "accuracy",
    "stsb": "spearmanr",
}

#: External repositories the LoRA+Distill baseline is defined by (paper Section 5.2).
EXTERNAL_REPOS: Dict[str, Dict[str, str]] = {
    "cofi": {
        "url": "https://github.com/princeton-nlp/CoFiPruning",
        "env": "COFI_DIR",
    },
}

_RESULT_KEYS = ("trainer", "model", "tokenizer", "config")


# --------------------------------------------------------------------------------------
# Small normalisation helpers
# --------------------------------------------------------------------------------------


def normalize_task(task: Optional[str]) -> str:
    """Canonicalise a GLUE task name (``SST-2`` -> ``sst2``)."""
    if task is None:
        return "sst2"
    key = str(task).strip().lower().replace(" ", "")
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    key2 = key.replace("_", "-")
    if key2 in TASK_ALIASES:
        return TASK_ALIASES[key2]
    key3 = key.replace("-", "")
    if key3 in TASK_ALIASES:
        return TASK_ALIASES[key3]
    return key


def normalize_method(method: Optional[str]) -> str:
    """Canonicalise a method name (``LoRA+Distill`` -> ``lora_distill``)."""
    if method is None:
        return "apt"
    key = str(method).strip().lower()
    for candidate in (key, key.replace(" ", ""), key.replace(" ", "_"), key.replace("-", "_")):
        if candidate in METHOD_ALIASES:
            return METHOD_ALIASES[candidate]
    key2 = key.replace("+", "_").replace("-", "_")
    if key2 in METHOD_ALIASES:
        return METHOD_ALIASES[key2]
    raise KeyError(
        "unknown method {!r} (known: {})".format(method, ", ".join(sorted(set(METHOD_ALIASES.values()))))
    )


def normalize_model(model: Optional[str]) -> str:
    """Canonicalise a model family name."""
    if model is None:
        return MODEL_DEFAULT
    key = str(model).strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    for alias, family in MODEL_ALIASES.items():
        if key.startswith(alias):
            return family
    return key


def display_name(method: str) -> str:
    """Paper table row label for a method key."""
    try:
        key = normalize_method(method)
    except KeyError:
        return str(method)
    return METHOD_DISPLAY.get(key, key)


def normalize_sparsity(sparsity: Optional[float] = None, density: Optional[float] = None) -> float:
    """Return sparsity, converting density (``1 - sparsity``) when needed."""
    if sparsity is not None:
        value = float(sparsity)
    elif density is not None:
        value = 1.0 - float(density)
    else:
        value = DEFAULT_SPARSITY
    value = min(max(value, 0.0), 0.99)
    # accept percentages such as 40 meaning 40 %
    if value > 1.0:
        value = value / 100.0
    return round(value, 4)


def sparsity_for_density(density: Optional[float]) -> float:
    """``sparsity = 1 - density`` (Section 4.2 definition of gamma_T)."""
    return normalize_sparsity(density=density)


def density_for_sparsity(sparsity: Optional[float]) -> float:
    """``density = 1 - sparsity``."""
    return round(1.0 - normalize_sparsity(sparsity=sparsity), 4)


def density_label(density: Optional[float]) -> str:
    """Format a density as ``50%`` / ``10%``."""
    if density is None:
        return "n/a"
    value = float(density)
    if value > 1.0:
        value = value / 100.0
    return "{:g}%".format(round(value * 100.0, 4))


def sparsity_for_method(method: str, sparsity: float = DEFAULT_SPARSITY) -> float:
    """Table 8 sparsity for a method: FT/LoRA are dense, APT/LoRA+Distill pruned."""
    key = normalize_method(method)
    if key in ("ft", "lora"):
        return 0.0
    return normalize_sparsity(sparsity=sparsity)


def glue_big_or_small(task: str) -> str:
    """Table 6 column for a GLUE task."""
    task = normalize_task(task)
    return "glue-big" if task in GLUE_BIG_TASKS else "glue-small"


def parse_seeds(seeds: Any, default: Sequence[int] = DEFAULT_SEEDS) -> Tuple[int, ...]:
    """Parse ``"42,43"`` / ``[42, 43]`` / ``42`` into a tuple of ints."""
    if seeds is None:
        return tuple(default)
    if isinstance(seeds, (int, float)):
        return (int(seeds),)
    if isinstance(seeds, str):
        parts = [p for p in seeds.replace(";", ",").split(",") if p.strip()]
        if not parts:
            return tuple(default)
        return tuple(int(float(p)) for p in parts)
    if isinstance(seeds, Iterable):
        return tuple(int(float(s)) for s in seeds)
    return tuple(default)


def split_methods(value: Any, default: Sequence[str] = METHOD_ORDER) -> List[str]:
    """Parse a method list from CLI/config input."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [v for v in value.replace(";", ",").split(",") if v.strip()]
    else:
        items = list(value)
    out: List[str] = []
    for item in items:
        key = normalize_method(item)
        if key not in out:
            out.append(key)
    return out


def split_tasks(value: Any, default: Sequence[str] = DEFAULT_TASKS) -> List[str]:
    """Parse a task list from CLI/config input."""
    if value is None:
        return [normalize_task(t) for t in default]
    if isinstance(value, str):
        items = [v for v in value.replace(";", ",").split(",") if v.strip()]
    else:
        items = list(value)
    out: List[str] = []
    for item in items:
        key = normalize_task(item)
        if key in EXCLUDED_TASKS:
            warnings.warn(
                "task {!r} is excluded from Table 8 (Appendix D.2: STS-B cannot be "
                "reproduced with CoFi distillation when only LoRA parameters are "
                "tuned)".format(item)
            )
        if key not in out:
            out.append(key)
    return out


def default_tasks() -> List[str]:
    """The seven Table 8 tasks in paper order."""
    return [normalize_task(t) for t in DEFAULT_TASKS]


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and (population) standard deviation ignoring ``None`` values."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return mean, var ** 0.5


def glue_average(values: Any, tasks: Optional[Sequence[str]] = None, default: Optional[float] = None) -> Optional[float]:
    """Mean primary metric over GLUE tasks.

    ``values`` may be a ``{task: score}`` mapping or a sequence of scores.  STS-B is
    dropped when present (Table 8 excludes it).
    """
    if values is None:
        return default
    if isinstance(values, dict):
        keys = [t for t in (tasks or values.keys()) if t in values]
        keys = [normalize_task(k) for k in keys if normalize_task(k) not in EXCLUDED_TASKS]
        nums = [values[t] for t in keys if values.get(t) is not None]
    else:
        nums = [v for v in values if v is not None]
    nums = [float(v) for v in nums]
    if not nums:
        return default
    return sum(nums) / len(nums)


# --------------------------------------------------------------------------------------
# Metrics / efficiency helpers
# --------------------------------------------------------------------------------------


def primary_from_metrics(task: str, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract the paper's primary metric for a task."""
    if not metrics:
        return None
    task = normalize_task(task)
    try:  # preferred: shared metric dispatch
        from apt.eval.metrics import primary_metric as _primary_metric  # type: ignore

        value = _primary_metric(task, metrics)
        if value is not None:
            return float(value)
    except Exception:
        pass
    key = PRIMARY_METRIC.get(task, "accuracy")
    for candidate in (key, "accuracy", "f1", "matthews_correlation", "spearmanr", "primary"):
        if candidate in metrics and metrics[candidate] is not None:
            return float(metrics[candidate])
    return None


def primary_from_summary(summary: Optional[Dict[str, Any]], task: str) -> Optional[float]:
    """Pull the task metric out of a training summary produced by a trainer."""
    if not summary:
        return None
    if summary.get("primary") is not None and "metrics" not in summary:
        return float(summary["primary"])
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        if "primary" in metrics and metrics["primary"] is not None:
            return float(metrics["primary"])
        value = primary_from_metrics(task, metrics)
        if value is not None:
            return value
    for key in ("primary", "metric", "accuracy", "f1"):
        if summary.get(key) is not None:
            return float(summary[key])
    return None


def normalise_from_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract the four absolute efficiency numbers from a training summary."""
    out: Dict[str, Optional[float]] = {
        "tta_seconds": None,
        "train_time_s": None,
        "train_peak_mem_mb": None,
        "inf_time_ms": None,
        "inf_mem_mb": None,
        "inf_throughput": None,
    }
    if not summary:
        return out
    raw = summary.get("efficiency") if isinstance(summary.get("efficiency"), dict) else summary
    for key in list(out.keys()):
        value = raw.get(key)
        if value is None and summary is not raw:
            value = summary.get(key)
        if value is not None:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = None
    if out["tta_seconds"] is None:
        out["tta_seconds"] = out["train_time_s"]
    return out


def compute_tta(
    history: Any,
    reference: Optional[float],
    *,
    fraction: float = TTA_FRACTION,
    higher_is_better: bool = True,
) -> Optional[float]:
    """Time to reach ``fraction * reference`` accuracy, linearly interpolated.

    ``history`` may be a list of ``(elapsed_seconds, metric)`` pairs or a list of
    dicts with ``time``/``elapsed``/``seconds`` and ``metric``/``primary`` keys.
    """
    if history is None or reference is None:
        return None
    points: List[Tuple[float, float]] = []
    for item in history:
        t = m = None
        if isinstance(item, dict):
            for key in ("time", "elapsed", "seconds", "elapsed_seconds", "train_time_s"):
                if item.get(key) is not None:
                    t = float(item[key])
                    break
            for key in ("metric", "primary", "value", "accuracy"):
                if item.get(key) is not None:
                    m = float(item[key])
                    break
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            t, m = float(item[0]), float(item[1])
        if t is not None and m is not None:
            points.append((t, m))
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    target = float(reference) * float(fraction)
    reached = None
    if higher_is_better:
        if points[0][1] >= target:
            reached = points[0][0]
        else:
            for (t0, m0), (t1, m1) in zip(points, points[1:]):
                if m1 >= target:
                    if m1 == m0:
                        reached = t1
                    else:
                        reached = t0 + (t1 - t0) * (target - m0) / (m1 - m0)
                    break
    else:
        if points[0][1] <= target:
            reached = points[0][0]
        else:
            for (t0, m0), (t1, m1) in zip(points, points[1:]):
                if m1 <= target:
                    if m1 == m0:
                        reached = t1
                    else:
                        reached = t0 + (t1 - t0) * (m0 - target) / (m0 - m1)
                    break
    if reached is None:
        return None
    return max(float(reached), 0.0)


def relative_efficiency(
    raw: Optional[Dict[str, Any]],
    reference: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Express absolute efficiency numbers as percentages of the FT reference.

    FT itself is 100 % on every axis; lower is better for all four metrics.
    """
    out: Dict[str, Optional[float]] = {key: None for key in EFFICIENCY_KEYS}
    if not raw:
        return out
    if not reference:
        reference = FT_RAW_EFFICIENCY
    pairs = (
        ("train_time", raw.get("tta_seconds") if raw.get("tta_seconds") is not None else raw.get("train_time_s"), reference.get("tta_seconds")),
        ("train_mem", raw.get("train_peak_mem_mb"), reference.get("train_peak_mem_mb")),
        ("inf_time", raw.get("inf_time_ms"), reference.get("inf_time_ms")),
        ("inf_mem", raw.get("inf_mem_mb"), reference.get("inf_mem_mb")),
    )
    for key, value, base in pairs:
        try:
            if value is None or base in (None, 0):
                out[key] = None
            else:
                out[key] = RELATIVE_SCALE * float(value) / float(base)
        except (TypeError, ValueError):
            out[key] = None
    return out


# --------------------------------------------------------------------------------------
# Config assembly
# --------------------------------------------------------------------------------------


def _config_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, "apt", "configs"))


def _candidate_config_files(model: str, task: str) -> List[str]:
    model = normalize_model(model)
    task = normalize_task(task)
    candidates = ["{}_{}.yaml".format(model, task)]
    if task in ("squad", "squad_v2"):
        candidates.append("squad.yaml")
    if task in ("cnndm", "cnn_dailymail"):
        candidates.append("t5_cnndm.yaml" if model == "t5" else "cnndm.yaml")
    candidates.append("default.yaml")
    return candidates


def _default_config_path(model: str = MODEL_DEFAULT, task: str = "sst2") -> Optional[str]:
    directory = _config_dir()
    for name in _candidate_config_files(model, task):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML file, returning ``{}`` when unavailable."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_base_config(model: str, task: str) -> Dict[str, Any]:
    """Base config: task YAML (falling back to ``default.yaml``), plus Table 6."""
    config = load_yaml(_default_config_path(model, task))
    if not config:
        config = load_yaml(os.path.join(_config_dir(), "default.yaml"))
    return config


def build_config(
    model: str = MODEL_DEFAULT,
    task: str = "sst2",
    sparsity: float = DEFAULT_SPARSITY,
    seed: int = 42,
    overrides: Optional[Dict[str, Any]] = None,
    method: str = "apt",
) -> Dict[str, Any]:
    """Assemble the config for one Table 8 cell.

    Precedence: ``default.yaml`` < task config < Table 6 group < APT defaults <
    caller overrides.  Table 6 values are only set when the key is absent, so an
    explicit user value always wins.
    """
    model = normalize_model(model)
    task = normalize_task(task)
    method = normalize_method(method)
    sparsity = normalize_sparsity(sparsity=sparsity)

    config: Dict[str, Any] = {}
    config.update(_load_base_config(model, task))

    config.setdefault("model_type", model)
    config.setdefault("model_name_or_path", MODEL_NAMES.get(model, model))
    config["task"] = task
    group = glue_big_or_small(task)
    config["table6_group"] = group
    config.setdefault("method", method)

    for key, value in TABLE6_GROUPS.get(group, {}).items():
        config.setdefault(key, value)

    # APT / Algorithm 1 defaults (Appendix A + Section 4.1-4.4).
    apt_defaults: Dict[str, Any] = {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "target_sparsity": sparsity,
        "initial_rank": INITIAL_RANK,
        "scaling": SCALING,
        "mask_alpha": MASK_ALPHA,
        "ema_beta": EMA_BETA,
        "tau": TAU,
        "pred_distill_weight": GLUE_PRED_DISTILL_WEIGHT,
        "layer_distill_weight": GLUE_LAYER_DISTILL_WEIGHT,
        "use_distillation": True,
        "use_kurtosis": True,
        "top_fraction": 0.5,
        "tuning_budget_initial": 1.0,
        "tuning_budget_final": 2.0,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "lr_kind": "linear",
        "max_grad_norm": 1.0,
        "max_seq_length": 128,
        "dynamic_padding": False,
        "inference_batch_size": SMALL_MODEL_INF_BATCH,
        "sequence_length": 128,
        "num_workers": 0,
    }
    for key, value in apt_defaults.items():
        config.setdefault(key, value)

    # Always force the cell-specific values.
    config["target_sparsity"] = sparsity
    config["seed"] = int(seed)

    if method in ("ft", "lora"):
        # Dense baselines: no pruning, no distillation (Table 8 rows at 0 %).
        config["target_sparsity"] = 0.0
        config["use_distillation"] = False

    if overrides:
        config.update(overrides)
        # re-force cell identity after user overrides of unrelated keys
        config.setdefault("target_sparsity", sparsity)

    return config


# --------------------------------------------------------------------------------------
# Trainer dispatch
# --------------------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _make_config_obj(cls: Any, data: Dict[str, Any]) -> Any:
    """Instantiate a baseline config dataclass from a dict when possible."""
    if cls is None:
        return data
    try:
        factory = getattr(cls, "from_dict", None)
        if callable(factory):
            try:
                return factory(data)
            except TypeError:
                return factory(**data)
    except Exception:
        pass
    try:
        return cls(**data)
    except Exception:
        return data


def _invoke(trainer: Callable[..., Any], method: str, config: Dict[str, Any]) -> Any:
    """Call a trainer with either ``(method, config)`` or ``(config)``."""
    try:
        signature = inspect.signature(trainer)
        required = 0
        for param in signature.parameters.values():
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and param.default is inspect._empty:
                required += 1
        if required >= 2:
            return trainer(method, config)
        if required == 1:
            return trainer(config)
        return trainer()
    except (TypeError, ValueError):
        return trainer(method, config)


def _call_baseline(fn: Callable[..., Any], config_obj: Any) -> Any:
    """Call a ``train_*`` baseline entry point with flexible keyword shapes."""
    for kwargs in ({"config": config_obj}, {"cfg": config_obj}, {}):
        try:
            if kwargs:
                return fn(**kwargs)
            return fn(config_obj)
        except TypeError as exc:
            last = exc
            continue
    raise last  # pragma: no cover


def _train_apt(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """APT row: prune + self-distill, then recover (Algorithm 1)."""
    try:
        from scripts.train_apt import run_training

        return run_training(dict(config))
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn("scripts.train_apt.run_training unavailable ({}); using apt.training".format(exc))
    from apt.training import train_apt

    return train_apt(config=dict(config))


def _train_ft(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """FT row: full-parameter fine-tuning (upper bound / normalisation reference)."""
    from apt.baselines.ft import FTConfig, train_ft

    cfg = _make_config_obj(FTConfig, dict(config))
    return _call_baseline(train_ft, cfg)


def _train_lora(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """LoRA row: rank-8 LoRA on q/v (matching APT's adapter capacity)."""
    from apt.baselines.lora import LoRAConfig, train_lora

    cfg = _make_config_obj(LoRAConfig, dict(config))
    return _call_baseline(train_lora, cfg)


def _train_lora_distill(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """LoRA+Distill row: CoFi-style pruning/distillation with LoRA-only tuning.

    Appendix D.2 describes the baseline as "conducting CoFi distillation with LoRA
    parameters tuned only"; the plan maps it onto
    ``apt.baselines.lora_prune_distill`` (CoFi with the tunable set restricted to
    LoRA + L0 gates), which wraps the external CoFiPruning repo when present.
    """
    payload = dict(config)
    payload.setdefault("method", "lora_distill")
    payload["tuning_only_lora"] = True
    payload.setdefault("lora_only", True)
    try:
        from apt.baselines.lora_prune_distill import (  # type: ignore
            LoRAPruneDistillConfig,
            train_lora_prune_distill,
        )

        cfg = _make_config_obj(LoRAPruneDistillConfig, payload)
        return _call_baseline(train_lora_prune_distill, cfg)
    except Exception as exc:
        warnings.warn("lora_prune_distill unavailable ({}); falling back to cofi".format(exc))
    from apt.baselines.cofi import CoFiConfig, train_cofi

    cfg = _make_config_obj(CoFiConfig, payload)
    return _call_baseline(train_cofi, cfg)


def resolve_trainer(method: str) -> Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]]:
    """Return the trainer callable for a Table 8 method (``(method, config)``)."""
    key = normalize_method(method)
    mapping = {
        "apt": _train_apt,
        "ft": _train_ft,
        "lora": _train_lora,
        "lora_distill": _train_lora_distill,
    }
    if key not in mapping:
        return None
    if key in ("ft", "lora", "lora_distill") and not _torch_available():
        # Import errors surface when the trainer is actually invoked.
        return mapping[key]
    return mapping[key]


def external_repo_available(name: str = "cofi") -> bool:
    """Whether the external baseline repository backing LoRA+Distill was found."""
    try:
        if name == "cofi":
            from apt.baselines.cofi import external_repo_available as _available

            return bool(_available())
    except Exception:
        return False
    return False


def proxy_note(method: str) -> str:
    """Human readable note about in-repo vs external implementations."""
    key = normalize_method(method)
    if key == "lora_distill" and not external_repo_available("cofi"):
        return (
            "in-repo CoFi-style LoRA+Distill proxy (external CoFiPruning checkout not "
            "found; set COFI_DIR or clone {})".format(EXTERNAL_REPOS["cofi"]["url"])
        )
    return ""


# --------------------------------------------------------------------------------------
# Training / aggregation
# --------------------------------------------------------------------------------------


def train_one(
    method: str,
    model: str,
    task: str,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seed: int = 42,
    config_overrides: Optional[Dict[str, Any]] = None,
    trainer: Optional[Callable[..., Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a single ``(method, model, task, sparsity, seed)`` cell.

    Returns a JSON-safe summary including the primary metric and efficiency numbers.
    """
    method = normalize_method(method)
    task = normalize_task(task)
    cell_sparsity = sparsity_for_method(method, sparsity)
    config = build_config(model, task, cell_sparsity, seed, config_overrides, method=method)

    trainer_fn = trainer or resolve_trainer(method)
    started = time.time()
    if trainer_fn is None:
        raise RuntimeError("no trainer available for method {!r}".format(method))
    summary = _invoke(trainer_fn, method, config)
    elapsed = time.time() - started

    summary = summary if isinstance(summary, dict) else {}
    primary = primary_from_summary(summary, task)
    efficiency = normalise_from_summary(summary)
    if efficiency.get("train_time_s") is None:
        efficiency["train_time_s"] = elapsed
    if efficiency.get("tta_seconds") is None:
        tta = compute_tta(
            summary.get("history"),
            summary.get("reference_metric") or summary.get("reference"),
        )
        if tta is not None:
            efficiency["tta_seconds"] = tta

    out = {
        "method": method,
        "method_display": display_name(method),
        "model": normalize_model(model),
        "task": task,
        "sparsity": cell_sparsity,
        "density": density_for_sparsity(cell_sparsity),
        "seed": int(seed),
        "primary": primary,
        "metrics": summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {},
        "efficiency": efficiency,
        "num_parameters": summary.get("num_parameters"),
        "num_tuning_parameters": summary.get("num_tuning_parameters"),
        "elapsed_seconds": elapsed,
        "note": proxy_note(method),
    }
    if verbose:
        value = "{:.2f}".format(primary) if primary is not None else "n/a"
        print(
            "[table8] {} | {} | {} | seed {} | {} | {:.1f}s".format(
                out["method_display"], out["model"], task, seed, value, elapsed
            )
        )
    return out


def collect_metrics(
    method: str,
    model: str,
    tasks: Sequence[str],
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    trainer: Optional[Callable[..., Any]] = None,
    verbose: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one method across all tasks/seeds and aggregate the results."""
    method = normalize_method(method)
    cell_sparsity = sparsity_for_method(method, sparsity)
    seeds = parse_seeds(seeds)

    per_task: Dict[str, Optional[float]] = {}
    per_task_std: Dict[str, Optional[float]] = {}
    per_task_metrics: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    raw_eff: Dict[str, List[Optional[float]]] = {key: [] for key in ("tta_seconds", "train_peak_mem_mb", "inf_time_ms", "inf_mem_mb", "train_time_s")}
    histories: Dict[str, Any] = {}
    notes: List[str] = []
    first_metrics: Dict[str, Any] = {}

    for task in tasks:
        task = normalize_task(task)
        values: List[Optional[float]] = []
        for seed in seeds:
            try:
                result = train_one(
                    method,
                    model,
                    task,
                    sparsity=cell_sparsity,
                    seed=seed,
                    config_overrides=config_overrides,
                    trainer=trainer,
                    verbose=verbose,
                )
            except Exception as exc:  # keep going; record the failure
                message = "{} | {} | {} | seed {}: {}".format(method, model, task, seed, exc)
                warnings.warn(message)
                failures.append({"method": method, "task": task, "seed": int(seed), "error": str(exc)})
                values.append(None)
                continue
            values.append(result.get("primary"))
            if result.get("metrics") and task not in first_metrics:
                first_metrics[task] = result["metrics"]
            eff = result.get("efficiency") or {}
            for key in raw_eff:
                raw_eff[key].append(eff.get(key))
            if result.get("note") and result["note"] not in notes:
                notes.append(result["note"])

        average, std = mean_std(values)
        per_task[task] = average
        per_task_std[task] = std
        per_task_metrics[task] = {"mean": average, "std": std, "n": len([v for v in values if v is not None])}

    raw: Dict[str, Optional[float]] = {}
    for key, values in raw_eff.items():
        average, _ = mean_std(values)
        raw[key] = average

    return {
        "method": method,
        "method_display": display_name(method),
        "model": normalize_model(model),
        "sparsity": cell_sparsity,
        "density": density_for_sparsity(cell_sparsity),
        "tasks": list(tasks),
        "per_task": per_task,
        "per_task_std": per_task_std,
        "per_task_metrics": per_task_metrics,
        "glue_avg": glue_average(per_task, tasks),
        "glue_avg_std": (mean_std([per_task_std.get(t) for t in tasks if per_task_std.get(t) is not None])[1]),
        "raw": raw,
        "raw_metrics": first_metrics,
        "histories": histories,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "failures": failures,
        "notes": notes,
    }


def build_rows(
    model: str = MODEL_DEFAULT,
    *,
    methods: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    trainer: Optional[Callable[..., Any]] = None,
    verbose: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Produce one Table 8 row per ``(method, sparsity)`` cell."""
    methods = split_methods(methods)
    tasks = split_tasks(tasks)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    dense_cache: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        cell_sparsity = sparsity_for_method(method, sparsity)
        if method == "ft" and "ft" in dense_cache:
            result = dense_cache["ft"]
        else:
            result = collect_metrics(
                method,
                model,
                tasks,
                sparsity=cell_sparsity,
                seeds=seeds,
                trainer=trainer,
                verbose=verbose,
                config_overrides=config_overrides,
            )
            if method == "ft":
                dense_cache["ft"] = result
        failures.extend(result.get("failures") or [])
        row = dict(result)
        row["method"] = method
        row["method_display"] = display_name(method)
        row["sparsity"] = cell_sparsity
        row["density"] = density_for_sparsity(cell_sparsity)
        rows.append(row)

    rows = normalise_rows(rows)
    return rows, failures


def reference_row(method: str, sparsity: Optional[float] = None) -> Dict[str, Any]:
    """Published Table 8 row for a method (sparsity selects the row)."""
    key = normalize_method(method)
    entry = TABLE8_REFERENCES.get(key)
    if entry is None:
        return {}
    if sparsity is not None and abs(normalize_sparsity(sparsity=sparsity) - float(entry.get("sparsity", 0.0))) > 1e-6:
        warnings.warn(
            "Table 8 has no published row for {} at sparsity {} "
            "(published sparsity is {})".format(key, sparsity, entry.get("sparsity"))
        )
    return json.loads(json.dumps(entry))


def absolute_row(method: str) -> Dict[str, Any]:
    """Appendix I raw efficiency numbers for RoBERTa-base FT / APT."""
    key = normalize_method(method)
    if key == "ft":
        return dict(FT_RAW_EFFICIENCY)
    if key == "apt":
        return dict(APT_RAW_EFFICIENCY)
    return {}


def efficiency_reference(rows: Sequence[Dict[str, Any]], model: str = MODEL_DEFAULT) -> Dict[str, Any]:
    """Reference efficiency used to normalise Table 8 efficiency columns.

    Prefers the measured FT row; falls back to Appendix I (Table 11) FT numbers.
    """
    for row in rows:
        if normalize_method(row.get("method", "")) == "ft":
            raw = row.get("raw") or {}
            if raw.get("tta_seconds") or raw.get("train_peak_mem_mb"):
                return {
                    "tta_seconds": raw.get("tta_seconds"),
                    "train_peak_mem_mb": raw.get("train_peak_mem_mb"),
                    "inf_time_ms": raw.get("inf_time_ms"),
                    "inf_mem_mb": raw.get("inf_mem_mb"),
                }
            break
    if normalize_model(model) == "roberta":
        return dict(FT_RAW_EFFICIENCY)
    return {}


def normalise_rows(rows: Sequence[Dict[str, Any]], reference: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Attach FT-relative efficiency columns (FT = 100 %)."""
    rows = [dict(row) for row in rows]
    base = reference or efficiency_reference(rows)
    for row in rows:
        method = normalize_method(row.get("method", ""))
        if method == "ft":
            row["relative"] = {key: RELATIVE_SCALE for key in EFFICIENCY_KEYS}
        else:
            row["relative"] = relative_efficiency(row.get("raw"), base)
    return rows


def compare_to_reference(
    rows: Sequence[Dict[str, Any]],
    *,
    tolerance: float = 1.0,
    tasks: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Diff measured Table 8 rows against the published numbers."""
    task_list = split_tasks(tasks) if tasks else default_tasks()
    comparisons: List[Dict[str, Any]] = []
    for row in rows:
        method = normalize_method(row.get("method", ""))
        ref = reference_row(method, row.get("sparsity"))
        if not ref:
            continue
        ref_metrics = ref.get("metrics", {})
        per_task: Dict[str, Dict[str, Any]] = {}
        for task in task_list:
            measured = (row.get("per_task") or {}).get(task)
            published = ref_metrics.get(task)
            delta = None
            within = None
            if measured is not None and published is not None:
                delta = float(measured) - float(published)
                within = abs(delta) <= float(tolerance)
            per_task[task] = {"measured": measured, "published": published, "delta": delta, "within": within}
        measured_avg = row.get("glue_avg")
        published_avg = ref.get("avg")
        avg_delta = None
        avg_within = None
        if measured_avg is not None and published_avg is not None:
            avg_delta = float(measured_avg) - float(published_avg)
            avg_within = abs(avg_delta) <= float(tolerance)
        comparisons.append(
            {
                "method": method,
                "method_display": display_name(method),
                "sparsity": row.get("sparsity"),
                "tasks": per_task,
                "glue_avg": {
                    "measured": measured_avg,
                    "published": published_avg,
                    "delta": avg_delta,
                    "within": avg_within,
                },
            }
        )
    return comparisons


# --------------------------------------------------------------------------------------
# Rendering / persistence
# --------------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return "{:.{d}f}".format(float(value), d=digits)
    except (TypeError, ValueError):
        return str(value)


def _fmt_relative(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return "{:.{d}f}%".format(float(value), d=digits)
    except (TypeError, ValueError):
        return str(value)


def format_table8(
    rows: Sequence[Dict[str, Any]],
    model: str = MODEL_DEFAULT,
    *,
    tasks: Optional[Sequence[str]] = None,
    digits: int = 1,
    show_efficiency: bool = False,
) -> str:
    """Render a markdown-style Table 8 block."""
    task_list = split_tasks(tasks) if tasks else default_tasks()
    model = normalize_model(model)
    header = ["Sparsity", "Method"] + [TASK_LABELS.get(t, t) for t in task_list] + ["GLUE Avg."]
    if show_efficiency:
        header += [EFFICIENCY_DISPLAY[key] for key in EFFICIENCY_KEYS]
    lines = ["# Table 8: {} GLUE pruning, APT vs LoRA+Distill (Appendix D.2)".format(model)]
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    children: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        children.setdefault(float(row.get("sparsity", 0.0)), []).append(row)

    for sparsity in sorted(children.keys()):
        block = children[sparsity]
        block.sort(key=lambda r: SPARSITY_ORDER.get((normalize_method(r.get("method", "")), sparsity), 99))
        for index, row in enumerate(block):
            label = "{:g}%".format(round(sparsity * 100.0, 4)) if index == 0 else ""
            cells = [label, str(row.get("method_display") or display_name(row.get("method", "")))]
            for task in task_list:
                cells.append(_fmt((row.get("per_task") or {}).get(task), digits))
            cells.append(_fmt(row.get("glue_avg"), digits))
            if show_efficiency:
                relative = row.get("relative") or {}
                cells += [_fmt_relative(relative.get(key), digits) for key in EFFICIENCY_KEYS]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_reference_table(tasks: Optional[Sequence[str]] = None, digits: int = 1) -> None:
    """Print the published Table 8 without training anything."""
    task_list = split_tasks(tasks) if tasks else default_tasks()
    header = ["Sparsity", "Method"] + [TASK_LABELS.get(t, t) for t in task_list] + ["GLUE Avg."]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for sparsity in sorted({float(v.get("sparsity", 0.0)) for v in TABLE8_REFERENCES.values()}):
        block = [(k, v) for k, v in TABLE8_REFERENCES.items() if float(v.get("sparsity", 0.0)) == sparsity]
        block.sort(key=lambda kv: SPARSITY_ORDER.get((normalize_method(kv[0]), sparsity), 99))
        for index, (key, entry) in enumerate(block):
            label = "{:g}%".format(round(sparsity * 100.0, 4)) if index == 0 else ""
            cells = [label, entry.get("display", display_name(key))]
            cells += [_fmt(entry.get("metrics", {}).get(t), digits) for t in task_list]
            cells.append(_fmt(entry.get("avg"), digits))
            print("| " + " | ".join(cells) + " |")


def print_appendix_d1_table(digits: int = 1) -> None:
    """Print the Appendix D.1 (paper Table 7) per-task reference numbers."""
    tasks = ("mnli", "qqp", "qnli", "sst2", "cola", "stsb", "mrpc", "rte")
    header = ["Density", "Method"] + [TASK_LABELS.get(t, t) for t in tasks] + ["GLUE Avg."]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for density in (0.50, 0.10):
        block = [(k, v) for k, v in APPENDIX_D1_REFERENCES.items() if float(v.get("density", 0.0)) == density]
        for index, (key, entry) in enumerate(block):
            label = density_label(density) if index == 0 else ""
            cells = [label, entry.get("display", key)]
            cells += [_fmt(entry.get("metrics", {}).get(t), digits) for t in tasks]
            cells.append(_fmt(entry.get("avg"), digits))
            print("| " + " | ".join(cells) + " |")


def _json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k not in _RESULT_KEYS}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def save_results(payload: Dict[str, Any], path: str) -> str:
    """Persist a Table 8 payload as JSON."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    safe = _json_safe(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(safe, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def default_output_path(model: str = MODEL_DEFAULT) -> str:
    """Default JSON destination for a model family."""
    return os.path.join("outputs", "table8", "table8_{}.json".format(normalize_model(model)))


def load_reference_json(path: Optional[str]) -> None:
    """Merge official numbers from a JSON file into ``TABLE8_REFERENCES``."""
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return
    for key, entry in data.items():
        try:
            method = normalize_method(key)
        except KeyError:
            continue
        if not isinstance(entry, dict):
            continue
        target = TABLE8_REFERENCES.setdefault(method, {"sparsity": 0.0, "display": display_name(method), "metrics": {}})
        for field, value in entry.items():
            if field == "metrics" and isinstance(value, dict):
                target.setdefault("metrics", {}).update({normalize_task(k): v for k, v in value.items()})
            else:
                target[field] = value


# --------------------------------------------------------------------------------------
# Top level driver
# --------------------------------------------------------------------------------------


def run_table8(
    model: str = MODEL_DEFAULT,
    *,
    methods: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config_overrides: Optional[Dict[str, Any]] = None,
    trainer: Optional[Callable[..., Any]] = None,
    reference: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    show_efficiency: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Reproduce Table 8: train every row, normalise, diff, save.

    Returns a payload with ``rows``, ``reference``, ``comparison``, ``table`` and
    ``failures``.
    """
    model = normalize_model(model)
    methods = split_methods(methods)
    tasks = split_tasks(tasks)
    sparsity = normalize_sparsity(sparsity=sparsity)
    seeds = parse_seeds(seeds)

    if tasks == []:
        tasks = default_tasks()

    if dry_run:
        rows: List[Dict[str, Any]] = []
        for method in methods:
            cell_sparsity = sparsity_for_method(method, sparsity)
            published = reference_row(method, cell_sparsity)
            if not published:
                continue
            rows.append(
                {
                    "method": method,
                    "method_display": display_name(method),
                    "model": model,
                    "sparsity": cell_sparsity,
                    "density": density_for_sparsity(cell_sparsity),
                    "per_task": dict(published.get("metrics", {})),
                    "per_task_std": {t: 0.0 for t in published.get("metrics", {})},
                    "glue_avg": published.get("avg"),
                    "raw": absolute_row(method),
                    "relative": {key: RELATIVE_SCALE for key in EFFICIENCY_KEYS} if method == "ft" else None,
                    "notes": [],
                    "failures": [],
                    "dry_run": True,
                }
            )
        rows = normalise_rows(rows)
        payload = {
            "model": model,
            "sparsity": sparsity,
            "tasks": tasks,
            "methods": methods,
            "seeds": list(seeds),
            "rows": rows,
            "reference": TABLE8_REFERENCES,
            "comparison": compare_to_reference(rows, tasks=tasks),
            "failures": [],
            "table": format_table8(rows, model, tasks=tasks, digits=1, show_efficiency=show_efficiency),
            "dry_run": True,
            "generated_at": time.time(),
        }
        if output_path:
            payload["result_path"] = save_results(payload, output_path)
        return payload

    rows, failures = build_rows(
        model,
        methods=methods,
        tasks=tasks,
        sparsity=sparsity,
        seeds=seeds,
        trainer=trainer,
        verbose=verbose,
        config_overrides=config_overrides,
    )
    if reference:
        rows = normalise_rows(rows, reference=reference)

    payload = {
        "model": model,
        "sparsity": sparsity,
        "tasks": tasks,
        "methods": methods,
        "seeds": list(seeds),
        "rows": rows,
        "reference": TABLE8_REFERENCES,
        "comparison": compare_to_reference(rows, tasks=tasks),
        "failures": failures,
        "table": format_table8(rows, model, tasks=tasks, digits=1, show_efficiency=show_efficiency),
        "dry_run": False,
        "generated_at": time.time(),
    }

    output_path = output_path or default_output_path(model)
    try:
        payload["result_path"] = save_results(payload, output_path)
    except Exception as exc:  # pragma: no cover - disk issues shouldn't kill a run
        warnings.warn("could not save Table 8 results to {}: {}".format(output_path, exc))

    if verbose:
        print("")
        print(payload["table"])
        notes = sorted({n for row in rows for n in (row.get("notes") or [])})
        for note in notes:
            print("[table8] note: {}".format(note))
        if failures:
            print("[table8] {} cell(s) failed; see the 'failures' key of the payload.".format(len(failures)))
        if payload.get("result_path"):
            print("[table8] results written to {}".format(payload["result_path"]))
    return payload


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_table8_glue",
        description="Reproduce Table 8 (Appendix D.2): RoBERTa GLUE, APT vs LoRA+Distill.",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT, help="model family (default: roberta)")
    parser.add_argument("--methods", default=",".join(METHOD_ORDER), help="comma separated methods")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma separated GLUE tasks")
    parser.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY, help="target sparsity for pruned rows (default 0.40)")
    parser.add_argument("--density", type=float, default=None, help="target density (overrides --sparsity)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), help="comma separated seeds")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--reference-json", default=None, help="JSON file with published numbers to merge")
    parser.add_argument("--config", default=None, help="YAML config path supplying overrides")
    parser.add_argument("--dry-run", action="store_true", help="print the published Table 8 without training")
    parser.add_argument("--no-efficiency", action="store_true", help="omit efficiency columns from the table")
    parser.add_argument("--appendix-d1", action="store_true", help="also print Appendix D.1 reference numbers")
    parser.add_argument("--self-test", action="store_true", help="run dependency-light validation and exit")
    parser.add_argument("--quiet", action="store_true", help="suppress per-cell progress output")
    parser.add_argument("extra", nargs="*", help="extra config overrides as key=value")
    return parser


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect config overrides from parsed CLI args."""
    overrides: Dict[str, Any] = {}
    raw = getattr(args, "config", None)
    if raw:
        overrides.update(load_yaml(raw))
    for token in getattr(args, "extra", None) or []:
        if "=" in token:
            key, value = token.split("=", 1)
            overrides[key.strip()] = _coerce_scalar(value.strip())
    method = getattr(args, "model", None)
    if method:
        overrides.setdefault("model_type", normalize_model(method))
        overrides.setdefault("model_name_or_path", MODEL_NAMES.get(normalize_model(method), method))
    return overrides


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def main(config: Optional[Dict[str, Any]] = None, argv: Optional[List[str]] = None) -> int:
    """Programmatic / CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "self_test", False):
        return 0 if _self_test() else 1

    if getattr(args, "appendix_d1", False):
        print("Appendix D.1 reference numbers (paper Table 7):")
        print_appendix_d1_table()

    load_reference_json(getattr(args, "reference_json", None))

    overrides = cli_overrides(args)
    if config:
        overrides.update(config)

    methods = split_methods(args.methods)
    tasks = split_tasks(args.tasks)
    sparsity = normalize_sparsity(sparsity=args.sparsity, density=args.density)
    seeds = parse_seeds(args.seeds)

    payload = run_table8(
        args.model,
        methods=methods,
        tasks=tasks,
        sparsity=sparsity,
        seeds=seeds,
        config_overrides=overrides or None,
        output_path=args.output,
        dry_run=bool(args.dry_run),
        show_efficiency=not bool(args.no_efficiency),
        verbose=not bool(args.quiet),
    )

    if args.dry_run and not args.quiet:
        print("")
        print("Published Table 8:")
        print_reference_table(tasks=tasks)

    stats = _summarise(payload)
    if not args.quiet:
        print("")
        print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


def _summarise(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, JSON-safe summary printed at the end of a run."""
    averages = {}
    for row in payload.get("rows", []):
        key = "{}|{}".format(row.get("method"), round(float(row.get("sparsity", 0.0)), 4))
        averages[key] = row.get("glue_avg")
    return {
        "model": payload.get("model"),
        "tasks": payload.get("tasks"),
        "methods": payload.get("methods"),
        "seeds": payload.get("seeds"),
        "glue_avg": averages,
        "num_failures": len(payload.get("failures") or []),
        "result_path": payload.get("result_path"),
        "dry_run": payload.get("dry_run", False),
    }


# --------------------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------------------


def _fake_trainer(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic stand-in trainer used by the self test (no torch needed)."""
    task = normalize_task(config.get("task", "sst2"))
    key = normalize_method(method)
    entry = TABLE8_REFERENCES.get(key, {})
    reference = entry.get("metrics", {}).get(task)
    if reference is None:
        reference = 80.0
    if data_dir := config.get("__noise__"):
        reference = reference + float(data_dir)
    return {
        "metrics": {"accuracy": reference},
        "primary": reference,
        "train_time_s": 400.0,
        "train_peak_mem_mb": 1800.0,
        "inf_time_ms": 100.0,
        "inf_mem_mb": 900.0,
        "history": [(0.0, reference * 0.5), (400.0, reference)],
        "reference_metric": reference,
    }


def _self_test() -> bool:  # noqa: C901 - explicit assertions are the point
    import io
    import contextlib

    failures: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # --- task / method normalisation ---------------------------------------------
    check(normalize_task("SST-2") == "sst2", "SST-2 should normalise to sst2")
    check(normalize_task("mnli-mm") == "mnli", "mnli-mm should normalise to mnli")
    check(normalize_task("STS-B") == "stsb", "STS-B should normalise to stsb")
    check(normalize_method("LoRA+Distill") == "lora_distill", "LoRA+Distill alias")
    check(normalize_method("LoRA+Prune+Distill") == "lora_distill", "LoRA+Prune+Distill alias")
    check(normalize_method("APT") == "apt", "APT alias")
    check(display_name("lora_distill") == "LoRA+Distill", "display name for lora_distill")
    check(normalize_model("roberta-base") == "roberta", "roberta alias")
    check(glue_big_or_small("qqp") == "glue-big", "QQP is GLUE-big")
    check(glue_big_or_small("rte") == "glue-small", "RTE is GLUE-small")

    # --- sparsity / density ------------------------------------------------------
    check(sparsity_for_method("ft", 0.40) == 0.0, "FT is dense in Table 8")
    check(sparsity_for_method("lora", 0.40) == 0.0, "LoRA is dense in Table 8")
    check(abs(sparsity_for_method("apt", 0.40) - 0.40) < 1e-9, "APT uses 40% sparsity")
    check(abs(sparsity_for_method("lora_distill", 0.40) - 0.40) < 1e-9, "LoRA+Distill uses 40% sparsity")
    check(abs(sparsity_for_density(0.50) - 0.50) < 1e-9, "density 50% -> sparsity 0.50")
    check(density_label(0.10) == "10%", "density label")

    # --- published averages reproduce from the per-task rows ----------------------
    for key, entry in TABLE8_REFERENCES.items():
        computed = glue_average(entry["metrics"], TABLE8_REFERENCES[key]["metrics"].keys())
        check(
            computed is not None and abs(computed - float(entry["avg"])) < 0.06,
            "Table 8 average mismatch for {}: computed {} vs published {}".format(key, computed, entry["avg"]),
        )
    check(abs(glue_average(TABLE8_REFERENCES["apt"]["metrics"]) - 83.9) < 0.06, "APT GLUE avg 83.9")
    check(abs(glue_average(TABLE8_REFERENCES["lora_distill"]["metrics"]) - 80.0) < 0.06, "LoRA+Distill GLUE avg 80.0")
    check(abs(glue_average(TABLE8_REFERENCES["ft"]["metrics"]) - 89.7) < 0.06, "FT GLUE avg 89.7")
    check(glue_average({"mnli": 80.0, "stsb": 90.0}) == 80.0, "STS-B must be excluded from the average")

    # --- efficiency normalisation matches Appendix I ------------------------------
    relative = relative_efficiency(APT_RAW_EFFICIENCY, FT_RAW_EFFICIENCY)
    check(abs(relative["train_time"] - 592.1) < 0.2, "APT train time should be 592.1% (752/127)")
    check(abs(relative["train_mem"] - 70.1) < 0.2, "APT train mem should be 70.1%")
    check(abs(relative["inf_time"] - 41.3) < 0.2, "APT inf time should be 41.3%")
    check(abs(relative["inf_mem"] - 78.1) < 0.2, "APT inf mem should be 78.1%")

    # --- TTA interpolation --------------------------------------------------------
    history = [(0.0, 50.0), (100.0, 90.0)]
    tta = compute_tta(history, 95.0, fraction=1.0)
    check(tta is not None and abs(tta - 88.888) < 0.05, "TTA interpolation: {}".format(tta))
    check(compute_tta([], 95.0) is None, "empty history -> None")

    # --- Table 6 wiring -----------------------------------------------------------
    cfg = build_config("roberta", "mnli", 0.40, seed=42)
    check(cfg["table6_group"] == "glue-big", "MNLI uses the GLUE-big column")
    check(cfg["learning_rate"] == 2.0e-4, "GLUE learning rate is 2e-4")
    check(cfg["batch_size"] == 32, "GLUE batch size is 32")
    check(cfg["epochs"] == 40, "GLUE epochs is 40")
    check(cfg["distill_epochs"] == 20, "GLUE distill epochs is 20")
    check(cfg["initial_rank"] == 8, "initial adapter rank is 8")
    check(cfg["scaling"] == 2.0, "scaling factor is 2")
    check(abs(cfg["mask_alpha"] - 0.01) < 1e-12, "mask decay alpha is 0.01")
    check(abs(cfg["ema_beta"] - 0.85) < 1e-12, "EMA beta is 0.85")
    check(cfg["tau"] == 4, "tau is 4")
    check(abs(cfg["pred_distill_weight"] - 1.0) < 1e-12, "GLUE pred distill weight is 1.0")
    check(abs(cfg["layer_distill_weight"] - 0.9) < 1e-12, "GLUE layer distill weight is 0.9")
    check(cfg["target_sparsity"] == 0.40, "cell sparsity respected")
    check(cfg["seed"] == 42, "seed respected")
    dense = build_config("roberta", "cola", 0.40, seed=7, method="lora")
    check(dense["target_sparsity"] == 0.0, "dense baseline gets 0 sparsity")
    check(dense["use_distillation"] is False, "dense baseline disables distillation")
    check(dense["table6_group"] == "glue-small", "CoLA uses the GLUE-small column")

    # --- row aggregation + rendering with a fake trainer --------------------------
    rows, failures = build_rows(
        "roberta",
        methods=("ft", "lora_distill", "apt"),
        tasks=("sst2", "mnli"),
        sparsity=0.40,
        seeds=(42,),
        trainer=_fake_trainer,
        verbose=False,
    )
    check(not failures, "fake trainer should not fail: {}".format(failures))
    check(len(rows) == 3, "expected three rows, got {}".format(len(rows)))
    by_method = {row["method"]: row for row in rows}
    check(abs(by_method["apt"]["glue_avg"] - 90.45) < 0.1, "APT fake avg should average sst2/mnli")
    check(by_method["apt"]["sparsity"] == 0.40, "APT row sparsity")
    check(by_method["ft"]["sparsity"] == 0.0, "FT row sparsity")
    check(by_method["apt"]["relative"]["train_time"] is not None, "relative efficiency computed")

    table = format_table8(rows, "roberta", tasks=("sst2", "mnli"))
    check("APT" in table and "LoRA+Distill" in table, "table renders method rows")
    check("40%" in table, "table renders the sparsity block label")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_reference_table()
    check("83.9" in buffer.getvalue(), "reference table prints APT GLUE avg")

    # --- dry run end to end -------------------------------------------------------
    payload = run_table8("roberta", seeds=(42,), dry_run=True, verbose=False)
    check(payload["dry_run"] is True, "dry-run flag")
    check(len(payload["rows"]) == 4, "dry run produces the four published rows")
    check("83.9" in payload["table"], "dry-run table contains the APT average")
    for comparison in payload["comparison"]:
        check(bool(comparison["glue_avg"]["within"]), "published numbers must match themselves")

    if failures:
        print("SELF TEST FAILURES:")
        for item in failures:
            print(" - {}".format(item))
        return False
    print("run_table8_glue self-test passed ({} checks)".format(0))
    return True


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
