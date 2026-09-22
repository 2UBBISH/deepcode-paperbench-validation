#!/usr/bin/env python
"""Reproduce **Table 2** of *APT: Adaptive Pruning and Tuning Pretrained Language
Models for Efficient Training and Inference* (Sec. 5.4, Table 2).

Table 2 reports end-task performance **and** relative training / inference
efficiency of APT and its baselines when pruning RoBERTa-base and T5-base to
``60 %`` sparsity:

================  =======  ======  ========  ==========  =========  =========  =========
Model / Method    MNLI     SST2    CNN/DM    Train Time  Train Mem  Inf Time   Inf Mem
================  =======  ======  ========  ==========  =========  =========  =========
*RoBERTa-base*    87.6     94.8    82.9      100.0 %     100.0 %    100.0 %    100.0 %
FT
LoRA              87.5     95.1    83.0      2137.0 %    60.5 %     100.0 %    100.0 %
LoRA+Prune        84.0     93.0    79.2      5128.3 %    60.5 %     38.0 %     75.1 %
Prune+Distill     87.3     94.5    --        1495.3 %    168.5 %    38.6 %     79.2 %
LoRA+Prune+Dist.  84.2     91.9    --        6534.6 %    141.4 %    39.4 %     82.3 %
APT               86.4     94.5    81.8      592.1 %     70.1 %     41.3 %     78.1 %
*T5-base*         ...      ...     42.1/20.3/39.4 (FT ROUGE-1/2/L)
================  =======  ======  ========  ==========  =========  =========  =========

The script is a thin *driver*: it never re-implements a method.  For every
``(model, method)`` pair it asks the existing harness

* :mod:`scripts.train_baseline` -- FT / LoRA / LoRA+Prune / Prune+Distill /
  LoRA+Prune+Distill, and
* :mod:`scripts.train_apt` -- APT,

to *train* the model, then normalises the measured wall-clock time and peak
memory by the fully fine-tuned (FT) reference, exactly as described in Sec. 5.3
("time-to-accuracy to 97 % of the fully fine-tuned baseline", peak memory via
``torch.cuda.max_memory_allocated()``) and Appendix I (absolute numbers).
Finally it prints a paper-style markdown table and diffs the measured values
against the numbers published in Table 2 / Table 11.

Usage
-----
::

    python scripts/run_table2.py                       # self test + plan
    python scripts/run_table2.py --self-test
    python scripts/run_table2.py --model roberta --methods ft,lora,apt
    python scripts/run_table2.py --model t5 --methods ft,lora,lora_prune_distill
    python scripts/run_table2.py --dry-run             # only print the plan
    python scripts/run_table2.py --seeds 42,43,44      # mean/std over seeds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:  # allow `python scripts/run_table2.py`
    sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------- #
# constants -- Table 2 / Table 11 of the paper
# --------------------------------------------------------------------------- #
TABLE2_REFERENCES: Dict[str, Dict[str, Dict[str, float]]] = {
    "roberta": {
        "ft": {
            "mnli": 87.6, "sst2": 94.8, "squad_v2": 82.9,
            "train_time": 100.0, "train_mem": 100.0, "inf_time": 100.0, "inf_mem": 100.0,
        },
        "lora": {
            "mnli": 87.5, "sst2": 95.1, "squad_v2": 83.0,
            "train_time": 2137.0, "train_mem": 60.5, "inf_time": 100.0, "inf_mem": 100.0,
        },
        "mask_tuning": {
            "mnli": 84.0, "sst2": 93.0, "squad_v2": 79.2,
            "train_time": 5128.3, "train_mem": 60.5, "inf_time": 38.0, "inf_mem": 75.1,
        },
        "cofi": {
            "mnli": 87.3, "sst2": 94.5, "squad_v2": None,
            "train_time": 1495.3, "train_mem": 168.5, "inf_time": 38.6, "inf_mem": 79.2,
        },
        "lora_prune_distill": {
            "mnli": 84.2, "sst2": 91.9, "squad_v2": None,
            "train_time": 6534.6, "train_mem": 141.4, "inf_time": 39.4, "inf_mem": 82.3,
        },
        "apt": {
            "mnli": 86.4, "sst2": 94.5, "squad_v2": 81.8,
            "train_time": 592.1, "train_mem": 70.1, "inf_time": 41.3, "inf_mem": 78.1,
        },
    },
    "t5": {
        # CNN/DM is reported as ROUGE-1/2/L (summarisation); MNLI/SST2 are
        # classification accuracies (T5 runs those tasks as text-to-text).
        "ft": {
            "mnli": 87.1, "sst2": 95.2, "cnndm": 42.1, "cnndm_2": 20.3, "cnndm_l": 39.4,
            "train_time": 100.0, "train_mem": 100.0, "inf_time": 100.0, "inf_mem": 100.0,
        },
        "lora": {
            "mnli": 87.0, "sst2": 95.0, "cnndm": 38.7, "cnndm_2": 17.2, "cnndm_l": 36.0,
            "train_time": 255.5, "train_mem": 62.0, "inf_time": 100.0, "inf_mem": 100.0,
        },
        "mask_tuning": {
            "mnli": 80.9, "sst2": 92.3, "cnndm": 36.7, "cnndm_2": 15.7, "cnndm_l": 33.9,
            "train_time": 4523.5, "train_mem": 62.0, "inf_time": 47.1, "inf_mem": 73.4,
        },
        "apt": {
            "mnli": 87.0, "sst2": 95.0, "cnndm": 38.6, "cnndm_2": 17.0, "cnndm_l": 35.8,
            "train_time": 484.7, "train_mem": 73.9, "inf_time": 74.6, "inf_mem": 81.5,
        },
    },
}

# Appendix I -- absolute efficiency of the FT and APT runs used to derive the
# relative values above (seconds / MB for training, ms / MB for inference).
TABLE11_REFERENCES: Dict[str, Dict[str, Dict[str, float]]] = {
    "roberta": {
        "ft": {"tta_seconds": 127.0, "train_mem_mb": 2696.0, "inf_time_ms": 220.8, "inf_mem_mb": 1157.0},
        "apt": {"tta_seconds": 752.0, "train_mem_mb": 1890.0, "inf_time_ms": 91.3, "inf_mem_mb": 904.0},
    },
    "t5": {
        "ft": {"tta_seconds": 366.0, "train_mem_mb": 7217.0, "inf_time_ms": 248.1, "inf_mem_mb": 2347.0},
        "apt": {"tta_seconds": 1774.0, "train_mem_mb": 5332.0, "inf_time_ms": 185.0, "inf_mem_mb": 1913.0},
    },
}

METHOD_ORDER: Tuple[str, ...] = (
    "ft",
    "lora",
    "mask_tuning",
    "cofi",
    "lora_prune_distill",
    "apt",
)

METHOD_DISPLAY: Dict[str, str] = {
    "ft": "FT",
    "lora": "LoRA",
    "mask_tuning": "LoRA+Prune",
    "cofi": "Prune+Distill",
    "lora_prune_distill": "LoRA+Prune+Distill",
    "apt": "APT",
}

METHOD_ALIASES: Dict[str, str] = {
    "ft": "ft", "finetune": "ft", "fine-tuning": "ft", "full": "ft", "fullft": "ft",
    "lora": "lora",
    "lora+prune": "mask_tuning", "mask_tuning": "mask_tuning", "mask-tuning": "mask_tuning",
    "mask": "mask_tuning", "retraining_free": "mask_tuning", "retraining-free": "mask_tuning",
    "prune+distill": "cofi", "cofi": "cofi", "cofipruning": "cofi", "cofi_pruning": "cofi",
    "lora+prune+distill": "lora_prune_distill", "lora_prune_distill": "lora_prune_distill",
    "lora+prune+distil": "lora_prune_distill", "lpd": "lora_prune_distill",
    "apt": "apt",
}

# --------------------------------------------------------------------------- #
# Table 2 protocol constants (Sec. 5.3, Sec. 5.4, Addendum)
# --------------------------------------------------------------------------- #
DEFAULT_SPARSITY = 0.60
TTA_FRACTION = 0.97
TTA_SCALE = 100.0          # relative values are reported in %
SMALL_MODEL_INF_BATCH = 128
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)   # paper: mean/std over 3 seeds
MODEL_TASKS: Dict[str, Tuple[str, ...]] = {
    "roberta": ("mnli", "sst2", "squad_v2"),
    "t5": ("mnli", "sst2", "cnndm"),
}
EFFICIENCY_KEYS: Tuple[str, ...] = ("train_time", "train_mem", "inf_time", "inf_mem")
EFFICIENCY_DISPLAY: Dict[str, str] = {
    "train_time": "Train Time",
    "train_mem": "Train Mem",
    "inf_time": "Inf Time",
    "inf_mem": "Inf Mem",
}
MODEL_ALIASES: Dict[str, str] = {
    "roberta": "roberta", "roberta-base": "roberta", "roberta_base": "roberta",
    "t5": "t5", "t5-base": "t5", "t5_base": "t5",
}
TASK_ALIASES: Dict[str, str] = {
    "sst-2": "sst2", "sst_2": "sst2", "sst2": "sst2",
    "mnli-mm": "mnli", "mnli_matched": "mnli", "mnli": "mnli",
    "squad": "squad_v2", "squad2": "squad_v2", "squad-v2": "squad_v2", "squad_v2": "squad_v2",
    "cnn_dailymail": "cnndm", "cnn-dailymail": "cnndm", "cnndm": "cnndm",
}


# --------------------------------------------------------------------------- #
# small generic helpers
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return var ** 0.5


def _fmt(value: Optional[float], digits: int = 1, dash: str = "--") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_model(model: Optional[str]) -> str:
    """Map model aliases to ``"roberta"`` / ``"t5"``."""
    key = (model or "roberta").strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    raise KeyError(f"unknown model {model!r}; expected one of {sorted(set(MODEL_ALIASES))}")


def normalize_method(method: str) -> str:
    """Map method names from the paper ("LoRA+Prune") to registry keys."""
    key = (method or "").strip().lower().replace(" ", "")
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    if key in METHOD_ORDER:
        return key
    raise KeyError(f"unknown method {method!r}; expected one of {list(METHOD_ORDER)}")


def normalize_task(task: str) -> str:
    key = (task or "").strip().lower()
    return TASK_ALIASES.get(key, key)


def split_methods(text: Optional[str], model: str = "roberta") -> List[str]:
    """Parse a ``--methods`` string, expanding ``all`` per-model."""
    if not text or not str(text).strip():
        return list(default_methods(model))
    out: List[str] = []
    for chunk in str(text).replace(";", ",").split(","):
        name = chunk.strip()
        if not name:
            continue
        low = name.lower()
        if low in ("all", "table2"):
            for m in default_methods(model):
                if m not in out:
                    out.append(m)
            continue
        key = normalize_method(name)
        if key not in out:
            out.append(key)
    return out


def default_methods(model: str) -> Tuple[str, ...]:
    """Methods reported for a model in Table 2 (order follows the paper)."""
    if normalize_model(model) == "t5":
        return ("ft", "lora", "mask_tuning", "apt")
    return ("ft", "lora", "mask_tuning", "cofi", "lora_prune_distill", "apt")


def tasks_for_model(model: str, tasks: Optional[Sequence[str]] = None) -> List[str]:
    if tasks:
        return [normalize_task(t) for t in tasks]
    return list(MODEL_TASKS[normalize_model(model)])


# --------------------------------------------------------------------------- #
# efficiency normalisation (Sec. 5.3 / Appendix I)
# --------------------------------------------------------------------------- #
def relative_efficiency(
    raw: Dict[str, Optional[float]],
    reference: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """Normalise absolute efficiency numbers by the FT reference (FT == 100 %).

    ``raw`` / ``reference`` accept the aliases ``tta_seconds``/``train_time_s``/
    ``train_time``, ``train_peak_mem_mb``/``train_mem_mb``/``train_mem``,
    ``inf_time_ms``/``inf_time`` and ``inf_mem_mb``/``inf_mem``.
    ``torch.cuda.max_memory_allocated()`` supplies the peak memory values.
    """
    def pick(d: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for k in keys:
            if k in d and d[k] is not None:
                return _to_float(d[k])
        return None

    time_keys = ("tta_seconds", "train_time_s", "train_time")
    mem_keys = ("train_peak_mem_mb", "train_mem_mb", "peak_train_mem_mb", "train_mem")
    itime_keys = ("inf_time_ms", "inference_time_ms", "inf_time")
    imem_keys = ("inf_mem_mb", "inference_mem_mb", "inf_mem")

    out: Dict[str, Optional[float]] = {}
    for name, keys in (
        ("train_time", time_keys),
        ("train_mem", mem_keys),
        ("inf_time", itime_keys),
        ("inf_mem", imem_keys),
    ):
        value = pick(raw, keys)
        ref = pick(reference, keys)
        if value is None or ref in (None, 0.0):
            out[name] = None
        else:
            out[name] = value / ref * TTA_SCALE
    return out


def normalise_from_summary(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract the four absolute efficiency numbers from a training summary."""
    return {
        "tta_seconds": _to_float(summary.get("tta_seconds")),
        "train_peak_mem_mb": _to_float(summary.get("train_peak_mem_mb")),
        "inf_time_ms": _to_float(summary.get("inf_time_ms")),
        "inf_mem_mb": _to_float(summary.get("inf_mem_mb")),
    }


def compute_tta(
    history: Sequence[Dict[str, Any]],
    reference: float,
    *,
    fraction: float = TTA_FRACTION,
    higher_is_better: bool = True,
) -> Optional[float]:
    """Time-to-accuracy: linearly interpolated wall-clock seconds to reach
    ``fraction`` of the FT reference metric (Sec. 5.3)."""
    if not history or reference is None:
        return None
    target = float(reference) * fraction
    if target <= 0:
        return None
    reached = (lambda v: v >= target) if higher_is_better else (lambda v: v <= target)
    prev_t: Optional[float] = None
    prev_v: Optional[float] = None
    for rec in history:
        t = _to_float(rec.get("elapsed_seconds", rec.get("time")))
        v = _to_float(rec.get("metric", rec.get("value", rec.get("primary"))))
        if t is None or v is None:
            continue
        if reached(v):
            if prev_t is None or v == prev_v:
                return t
            span_v = (v - prev_v) if higher_is_better else (prev_v - v)
            if span_v <= 0:
                return t
            frac = (target - prev_v) / (v - prev_v)
            frac = max(0.0, min(1.0, frac))
            return prev_t + frac * (t - prev_t)
        prev_t, prev_v = t, v
    return None


# --------------------------------------------------------------------------- #
# harness adapters (lazy: heavy deps loaded only when actually training)
# --------------------------------------------------------------------------- #
def _load_module(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"could not import {name}: {exc}")
        return None


def build_config(model: str, task: str, sparsity: float, seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the config dict for one (model, task) cell of Table 2."""
    config: Dict[str, Any] = {
        "model_type": model,
        "model_name_or_path": "roberta-base" if model == "roberta" else "t5-base",
        "task": task,
        "target_sparsity": float(sparsity),
        "seed": int(seed),
        "inference_batch_size": SMALL_MODEL_INF_BATCH,
    }
    if model == "roberta":
        config.update({
            "learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20,
            "max_seq_length": 384 if task == "squad_v2" else 128,
            "doc_stride": 128, "max_query_length": 64,
            "table6_group": "squad" if task == "squad_v2" else "glue-big",
        })
        config["pred_distill_weight"] = 0.1 if task == "squad_v2" else 1.0
        config["layer_distill_weight"] = 0.9
    else:
        config.update({
            "learning_rate": 1e-4 if task == "cnndm" else 2e-4,
            "batch_size": 16 if task == "cnndm" else 32,
            "epochs": 16 if task == "cnndm" else 40,
            "distill_epochs": 6 if task == "cnndm" else 20,
            "max_seq_length": 512 if task == "cnndm" else 128,
            "max_target_length": 128,
            "table6_group": "cnndm" if task == "cnndm" else "glue-big",
        })
        config["pred_distill_weight"] = 0.1
        config["layer_distill_weight"] = 0.9
    if overrides:
        config.update(overrides)
    return config


def resolve_trainer(method: str):
    """Return a callable ``trainer(config, **kwargs) -> Dict[str, Any]``.

    ``apt`` is dispatched to :mod:`scripts.train_apt`, everything else to
    :mod:`scripts.train_baseline` (in-repo FT/LoRA and the wrapped Mask Tuning /
    CoFi baselines).
    """
    if method == "apt":
        mod = _load_module("scripts.train_apt")
        if mod is None:
            mod = _load_module("apt.training")
        if mod is None:
            return None

        def _run(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            if hasattr(mod, "run_training"):
                return mod.run_training(config, **kwargs)
            if hasattr(mod, "train_apt"):
                return mod.train_apt(config, **kwargs)
            raise AttributeError("no train entry point found in scripts.train_apt")

        return _run

    mod = _load_module("scripts.train_baseline")
    if mod is None:
        return None

    def _run_baseline(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if hasattr(mod, "run_baseline"):
            return mod.run_baseline(method, config, **kwargs)
        if hasattr(mod, "run_methods"):
            results = mod.run_methods([method], config, **kwargs)
            return results.get(method, {})
        raise AttributeError("no train entry point found in scripts.train_baseline")

    return _run_baseline


def train_one(
    method: str,
    model: str,
    task: str,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seed: int = 42,
    config_overrides: Optional[Dict[str, Any]] = None,
    trainer: Any = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a single (method, model, task) cell; returns a raw summary dict."""
    config = build_config(model, task, sparsity, seed, config_overrides)
    run = trainer or resolve_trainer(method)
    if run is None:
        return {"method": method, "model": model, "task": task, "error": "harness unavailable"}
    if verbose:
        print(f"[table2] training {METHOD_DISPLAY.get(method, method)} "
              f"on {model}/{task} (sparsity={sparsity:.0%}, seed={seed})")
    started = time.time()
    summary = run(config)
    if not isinstance(summary, dict):
        summary = {"metrics": summary}
    summary.setdefault("method", method)
    summary.setdefault("model", model)
    summary.setdefault("task", task)
    summary.setdefault("wall_clock_s", time.time() - started)
    return summary


def collect_metrics(
    method: str,
    model: str,
    tasks: Sequence[str],
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = (42,),
    trainer: Any = None,
    verbose: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one method over all tasks (and optionally seeds); aggregate mean/std."""
    per_task: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tasks}
    for task in tasks:
        for seed in seeds:
            summary = train_one(
                method, model, task,
                sparsity=sparsity, seed=seed, trainer=trainer,
                verbose=verbose, config_overrides=config_overrides,
            )
            per_task[task].append(summary)

    row: Dict[str, Any] = {
        "method": method,
        "display": METHOD_DISPLAY.get(method, method),
        "model": model,
        "sparsity": float(sparsity),
        "seeds": list(seeds),
        "quality": {},
        "quality_std": {},
        "efficiency_raw": {},
        "efficiency": {},
    }
    for task, summaries in per_task.items():
        primaries: List[float] = []
        for s in summaries:
            value = primary_from_summary(task, s)
            if value is None:
                value = _to_float(s.get("primary"))
            if value is not None:
                primaries.append(float(value))
        row["quality"][task] = _mean(primaries)
        row["quality_std"][task] = _std(primaries) if len(primaries) > 1 else None

        for key in ("tta_seconds", "train_peak_mem_mb", "train_time_s",
                    "inf_time_ms", "inf_mem_mb", "inf_throughput"):
            values = [_to_float(s.get(key)) for s in summaries]
            values = [v for v in values if v is not None]
            if values:
                row["efficiency_raw"][key] = _mean(values)
    return row


def primary_from_summary(task: str, summary: Dict[str, Any]) -> Optional[float]:
    """Pull the paper's primary metric out of a training summary dict."""
    task = normalize_task(task)
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    if isinstance(metrics, dict):
        if task == "cnndm":
            for key in ("rouge1", "rouge-1", "rougeL", "rouge_l"):
                if key in metrics:
                    return _to_float(metrics[key])
        mod = _load_module("apt.eval.metrics")
        if mod is not None and hasattr(mod, "primary_metric"):
            try:
                value = mod.primary_metric(task, metrics)
                if value is not None:
                    return float(value)
            except Exception:
                pass
        for key in (task, "accuracy", "acc", "f1", "matthews_correlation", "spearmanr", "primary"):
            if key in metrics:
                return _to_float(metrics[key])
    return _to_float(summary.get("primary")) if isinstance(summary, dict) else None


# --------------------------------------------------------------------------- #
# reference / fallback rows
# --------------------------------------------------------------------------- #
def reference_row(model: str, method: str) -> Dict[str, Any]:
    """Paper values for one ``(model, method)`` cell of Table 2."""
    model = normalize_model(model)
    method = normalize_method(method)
    refs = TABLE2_REFERENCES.get(model, {}).get(method, {})
    return {
        "method": method,
        "display": METHOD_DISPLAY.get(method, method),
        "model": model,
        "quality": {t: refs.get(t) for t in MODEL_TASKS[model]},
        "efficiency": {k: refs.get(k) for k in EFFICIENCY_KEYS},
        "source": "reference",
    }


def absolute_row(model: str, method: str) -> Dict[str, Any]:
    """Appendix I absolute efficiency numbers for FT / APT."""
    model = normalize_model(model)
    method = normalize_method(method)
    raw = TABLE11_REFERENCES.get(model, {}).get(method, {})
    return {"method": method, "model": model, "efficiency_raw": dict(raw), "source": "table11"}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _quality_string(row: Dict[str, Any], tasks: Sequence[str]) -> List[str]:
    cells: List[str] = []
    for task in tasks:
        value = row.get("quality", {}).get(task)
        std = row.get("quality_std", {}).get(task)
        if task == "cnndm":
            # ROUGE-1/2/L, e.g. "38.6/17.0/35.8"
            triple = [value]
            ref = row.get("quality", {}).get("cnndm_tuple")
            if isinstance(ref, (list, tuple)) and len(ref) == 3:
                triple = list(ref)
            cells.append("/".join(_fmt(_to_float(v)) for v in triple))
        elif value is None:
            cells.append("--")
        elif std:
            cells.append(f"{_fmt(value)}±{_fmt(std)}")
        else:
            cells.append(_fmt(value))
    return cells


def format_table2(rows: Sequence[Dict[str, Any]], model: str, *, digits: int = 1) -> str:
    """Render the Table 2 markdown block/latex-like fixed-width table."""
    model = normalize_model(model)
    tasks = MODEL_TASKS[model]
    eff = row_efficiency(rows)
    header = f"{'Method':<20}" + "".join(f"{t.upper():>14}" for t in tasks)
    header += "".join(f"{EFFICIENCY_DISPLAY[k]:>14}" for k in EFFICIENCY_KEYS)
    lines = [f"Model: {model}", header, "-" * len(header)]
    for row in rows:
        cells = _quality_string(row, tasks)
        eff_cells = [
            ("--" if eff.get(row["method"], {}).get(k) is None
             else f"{_fmt(eff[row['method']][k], digits)} %")
            for k in EFFICIENCY_KEYS
        ]
        lines.append(
            f"{row.get('display', row['method']):<20}"
            + "".join(f"{c:>14}" for c in cells)
            + "".join(f"{c:>14}" for c in eff_cells)
        )
    lines.append("-" * len(header))
    lines.append("All efficiency numbers are relative to FT (FT = 100 %); "
                 "lower is better, ROUGE shown as R-1/R-2/R-L.")
    return "\n".join(lines)


def row_efficiency(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Relative efficiency for every row, normalising by the FT row of the batch."""
    ft_raw: Dict[str, Optional[float]] = {}
    for row in rows:
        if row.get("method") == "ft" and row.get("efficiency_raw"):
            ft_raw = row["efficiency_raw"]
            break
    if not ft_raw:  # no measured FT row -> use Appendix I absolute numbers
        for row in rows:
            if row.get("method") == "ft":
                model = row.get("model", "roberta")
                ft_raw = dict(TABLE11_REFERENCES.get(model, {}).get("ft", {}))
                break

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        method = row["method"]
        measured = row.get("efficiency_raw") or {}
        if measured:
            out[method] = relative_efficiency(measured, ft_raw)
            continue
        ref = row.get("efficiency")
        if isinstance(ref, dict) and any(v is not None for v in ref.values()):
            out[method] = {k: ref.get(k) for k in EFFICIENCY_KEYS}
        else:
            out[method] = {k: None for k in EFFICIENCY_KEYS}
    return out


def compare_to_reference(rows: Sequence[Dict[str, Any]], model: str, *, tolerance: float = 1.0) -> List[Dict[str, Any]]:
    """Diff measured rows against the published Table 2 values."""
    model = normalize_model(model)
    eff = row_efficiency(rows)
    diffs: List[Dict[str, Any]] = []
    for row in rows:
        method = row["method"]
        refs = TABLE2_REFERENCES.get(model, {}).get(method)
        if not refs:
            continue
        entry: Dict[str, Any] = {"method": method, "display": row.get("display", method), "diff": {}}
        for task in MODEL_TASKS[model]:
            measured = row.get("quality", {}).get(task)
            expected = refs.get(task)
            if measured is None or expected is None:
                continue
            entry["diff"][task] = {
                "measured": measured,
                "expected": float(expected),
                "delta": measured - float(expected),
                "ok": abs(measured - float(expected)) <= tolerance,
            }
        for key in EFFICIENCY_KEYS:
            measured = eff.get(method, {}).get(key)
            expected = refs.get(key)
            if measured is None or expected is None:
                continue
            entry["diff"][key] = {
                "measured": measured,
                "expected": float(expected),
                "delta": measured - float(expected),
                "ok": abs(measured - float(expected)) <= max(tolerance, 2.0 * tolerance),
            }
        diffs.append(entry)
    return diffs


def print_reference_table(model: str, methods: Optional[Sequence[str]] = None) -> None:
    """Print the published Table 2 values (no training)."""
    model = normalize_model(model)
    methods = list(methods or default_methods(model))
    rows = [reference_row(model, m) for m in methods]
    print(format_table2(rows, model))


def save_results(payload: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return path


def default_output_path(model: str) -> str:
    return os.path.join("outputs", "table2", f"table2_{normalize_model(model)}.json")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_table2(
    model: str = "roberta",
    *,
    methods: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = (42,),
    config_overrides: Optional[Dict[str, Any]] = None,
    trainer: Any = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train every requested Table 2 cell and return the aggregated payload."""
    model = normalize_model(model)
    methods = [normalize_method(m) for m in (methods or default_methods(model))]
    tasks = tasks_for_model(model, tasks)

    if dry_run:
        return {
            "model": model,
            "sparsity": float(sparsity),
            "seeds": list(seeds),
            "tasks": tasks,
            "methods": methods,
            "dry_run": True,
            "reference": {m: reference_row(model, m) for m in methods},
        }

    rows: List[Dict[str, Any]] = []
    for method in methods:
        row = collect_metrics(
            method, model, tasks,
            sparsity=sparsity, seeds=seeds,
            trainer=trainer, verbose=verbose,
            config_overrides=config_overrides,
        )
        rows.append(row)

    payload: Dict[str, Any] = {
        "model": model,
        "sparsity": float(sparsity),
        "seeds": list(seeds),
        "tasks": tasks,
        "methods": methods,
        "rows": rows,
        "reference": {m: reference_row(model, m) for m in methods},
        "comparison": compare_to_reference(rows, model),
    }
    payload["table"] = format_table2(rows, model)
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_table2.py",
        description="Reproduce Table 2 of the APT paper (RoBERTa/T5, 60% sparsity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="roberta",
                        help="roberta / t5 (aliases accepted)")
    parser.add_argument("--methods", default=None,
                        help="comma separated methods; 'all' expands to the Table 2 rows")
    parser.add_argument("--tasks", default=None, help="comma separated task subset")
    parser.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY,
                        help="target sparsity (paper: 0.60 for RoBERTa/T5)")
    parser.add_argument("--seeds", default="42",
                        help="comma separated seeds, e.g. 42,43,44 (paper: 3 seeds, mean/std)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run plan and the reference table only")
    parser.add_argument("--self-test", action="store_true", help="run internal sanity checks")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_seeds(text: Optional[str]) -> List[int]:
    if not text:
        return [42]
    out: List[int] = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out or [42]


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Extra config values merged into every cell's config."""
    overrides: Dict[str, Any] = {}
    if getattr(args, "sparsity", None) is not None:
        overrides["target_sparsity"] = float(args.sparsity)
    return overrides


def main(config: Optional[Dict[str, Any]] = None, argv: Optional[List[str]] = None) -> int:
    """Entry point shared by :mod:`main.py` and the command line."""
    config = dict(config or {})
    known = {a.dest for a in build_parser()._actions}
    merged: Dict[str, Any] = {}
    for key, value in config.items():
        merged[key.replace("-", "_")] = value
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args([f"--{k.replace('_', '-')}={v}" if isinstance(v, bool) and v else
                                      (f"--{k.replace('_', '-')}" if isinstance(v, bool) else
                                       f"--{k.replace('_', '-')}={v}")
                                      for k, v in merged.items() if k in known] + cli_argv)

    if args.self_test:
        return 0 if _self_test() else 1

    model = normalize_model(args.model)
    methods = split_methods(args.methods, model)
    tasks = tasks_for_model(model, args.tasks.split(",") if args.tasks else None)
    seeds = parse_seeds(args.seeds)

    print(f"== APT Table 2 reproduction: model={model}, sparsity={args.sparsity:.0%}, "
          f"methods={methods}, tasks={tasks}, seeds={seeds}")
    print()
    print("Published Table 2 values (reference):")
    print_reference_table(model, methods)
    print()

    payload = run_table2(
        model,
        methods=methods,
        tasks=tasks,
        sparsity=args.sparsity,
        seeds=seeds,
        config_overrides=cli_overrides(args),
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if args.dry_run:
        print("Dry run complete -- no training performed.")
        return 0

    print()
    print("Measured results:")
    print(payload["table"])
    print()
    print("Comparison against Table 2 (deltas):")
    for entry in payload.get("comparison", []):
        parts = [
            f"{name}={v['measured']:.1f} (ref {v['expected']:.1f}, d={v['delta']:+.1f})"
            for name, v in entry["diff"].items()
        ]
        status = "OK " if all(v["ok"] for v in entry["diff"].values()) else "DIFF"
        print(f"  [{status}] {entry['display']:<20} " + ", ".join(parts))

    path = args.output or default_output_path(model)
    save_results(_json_safe(payload), path)
    print(f"\nResults written to {path}")
    return 0


def _json_safe(obj: Any) -> Any:
    """Drop non-serialisable training objects from a result payload."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()
                if k not in ("trainer", "model", "tokenizer")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def _self_test() -> bool:
    ok = True

    def check(name: str, condition: bool) -> None:
        nonlocal ok
        print(f"  [{'ok' if condition else 'FAIL'}] {name}")
        ok = ok and bool(condition)

    print("run_table2 self test")
    check("model aliasing", normalize_model("roberta-base") == "roberta" and normalize_model("t5") == "t5")
    check("method aliasing",
          normalize_method("LoRA+Prune") == "mask_tuning"
          and normalize_method("Prune+Distill") == "cofi"
          and normalize_method("LoRA+Prune+Distill") == "lora_prune_distill")

    roberta_defaults = default_methods("roberta")
    check("roberta method list", "cofi" in roberta_defaults and "lora_prune_distill" in roberta_defaults)
    check("t5 method list", default_methods("t5") == ("ft", "lora", "mask_tuning", "apt"))

    # Appendix I -> Table 2 normalisation: 752 s / 127 s == 592.1 %
    rel = relative_efficiency(
        {"tta_seconds": 752.0, "train_peak_mem_mb": 1890.0, "inf_time_ms": 91.3, "inf_mem_mb": 904.0},
        {"tta_seconds": 127.0, "train_peak_mem_mb": 2696.0, "inf_time_ms": 220.8, "inf_mem_mb": 1157.0},
    )
    check("roberta APT train_time ~592.1%", abs(rel["train_time"] - 592.1) < 0.2)
    check("roberta APT train_mem ~70.1%", abs(rel["train_mem"] - 70.1) < 0.2)
    check("roberta APT inf_time ~41.3%", abs(rel["inf_time"] - 41.3) < 0.2)
    check("roberta APT inf_mem ~78.1%", abs(rel["inf_mem"] - 78.1) < 0.2)

    t5_rel = relative_efficiency(
        {"tta_seconds": 1774.0, "train_peak_mem_mb": 5332.0, "inf_time_ms": 185.0, "inf_mem_mb": 1913.0},
        {"tta_seconds": 366.0, "train_peak_mem_mb": 7217.0, "inf_time_ms": 248.1, "inf_mem_mb": 2347.0},
    )
    check("t5 APT train_time ~484.7%", abs(t5_rel["train_time"] - 484.7) < 0.2)
    check("t5 APT train_mem ~73.9%", abs(t5_rel["train_mem"] - 73.9) < 0.2)
    check("t5 APT inf_time ~74.6%", abs(t5_rel["inf_time"] - 74.6) < 0.2)
    check("t5 APT inf_mem ~81.5%", abs(t5_rel["inf_mem"] - 81.5) < 0.2)

    # TTA interpolation: reach 97% of 95.0 at some point between the two evals
    history = [
        {"elapsed_seconds": 100.0, "metric": 80.0},
        {"elapsed_seconds": 200.0, "metric": 95.0},
    ]
    tta = compute_tta(history, 95.0)
    check("tta interpolation", tta is not None and 100.0 < tta <= 200.0)
    check("tta unreachable", compute_tta(history, 200.0) is None)

    # sparsity default and task lists follow the paper
    check("default sparsity 60%", abs(DEFAULT_SPARSITY - 0.60) < 1e-9)
    check("roberta tasks", tasks_for_model("roberta") == ["mnli", "sst2", "squad_v2"])
    check("t5 tasks", tasks_for_model("t5") == ["mnli", "sst2", "cnndm"])
    check("task aliasing", normalize_task("SST-2") == "sst2" and normalize_task("cnn_dailymail") == "cnndm")

    # reference tables carry the published numbers
    apt_roberta = reference_row("roberta", "apt")
    check("reference apt roberta mnli 86.4", abs(apt_roberta["quality"]["mnli"] - 86.4) < 1e-9)
    check("reference apt roberta inf_mem 78.1", abs(apt_roberta["efficiency"]["inf_mem"] - 78.1) < 1e-9)
    row = reference_row("t5", "mask_tuning")
    check("reference t5 LoRA+Prune 4523.5%", abs(row["efficiency"]["train_time"] - 4523.5) < 1e-9)

    # rendering does not explode and mentions every method
    text = format_table2([reference_row("roberta", m) for m in default_methods("roberta")], "roberta")
    check("render contains APT", "APT" in text)
    check("render contains LoRA+Prune+Distill", "LoRA+Prune+Distill" in text)

    # dry run never touches torch
    payload = run_table2("roberta", methods=["apt"], dry_run=True)
    check("dry run payload", payload["dry_run"] and payload["methods"] == ["apt"])

    print("self test", "passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if _self_test() else 1)
    sys.exit(main())
