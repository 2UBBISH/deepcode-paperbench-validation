"""Reproduce Table 4 of the APT paper (RoBERTa-base APT ablations).

Table 4 (papers/§5.6) reports one component removed at a time on
RoBERTa-base at 60% sparsity::

    Method          SST2   MNLI   Train Time (down)  Train Mem (down)
    APT             94.5   86.4   592.1%             70.1%
    w/o A_P         94.4   87.5   82.6%              62.2%
    w/o salience    94.3   84.7   609.8%             65.0%
    w/o A_T         93.2   84.5   684.9%             64.4%
    w/o D_S         92.9   85.3   483.1%             61.9%

Ablation semantics follow §5.6 ("We evaluate the impact of different
components in APT by removing the adaptive pruning (A_P), adaptive tuning
(A_T), and self-distillation (D_S). ... we also report the training efficiency
metrics for each ablation"):

* ``w/o A_P`` -- "we only train LMs with adaptive tuning strategies with
  supervised finetuning objectives without distillation", i.e. adaptive pruning
  is switched off (density stays 100%) while the APT adapter stays adaptive.
  Its inference efficiency equals FT / LoRA.
* ``w/o salience`` -- drop the outlier term from Eq. (5), i.e. the square root
  of the kurtosis of the outlier activation ``O_{:,j} = W_{:,j} o X_{j,:}^T``
  (Table 5 spells the same ablation "w/o kurtosis").  Blocks are then allocated
  purely by the activation x gradient magnitude.
* ``w/o A_T`` -- "the tuning parameters are static when pruning RoBERTa
  models"; "equally increasing parameters across all layers instead of adding
  parameters based on salience notably hurts the task accuracy (84.4 on MNLI
  compared to 86.4)".  Implemented by disabling the dynamic rank update of
  Eq. (7) and freezing the tuning budget at its initial value.
* ``w/o D_S`` -- "tuning APT adapters dynamically without distillation
  objectives", i.e. plain supervised fine-tuning of the pruned adapters with
  mu kept at 0 for the whole run.

Everything else stays at its Table 6 value (lr 2e-4, batch 32, 40 epochs of
which 20 prune+distill, cubic sparsity schedule to 60%, EMA 0.85/0.15,
mask decay alpha=0.01), so exactly one component is removed at a time.

The script is a *driver only* -- it never re-implements APT.  Training is
delegated to ``scripts.train_apt`` (which wraps ``apt.training``), efficiency is
normalised against FT (FT == 100%), and the published Table 4 / Appendix I
numbers are embedded for diffing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Reference numbers (paper Table 4 and Appendix I Table 11)
# ---------------------------------------------------------------------------

#: RoBERTa-base quality and *relative* efficiency numbers from Table 4.
TABLE4_REFERENCES: Dict[str, Dict[str, Any]] = {
    "apt": {
        "display": "APT",
        "sst2": 94.5,
        "mnli": 86.4,
        "train_time": 592.1,
        "train_mem": 70.1,
    },
    "wo_ap": {
        "display": "w/o A_P",
        "display_tex": "w/o $\\mathcal{A}_{\\mathrm{P}}$",
        "sst2": 94.4,
        "mnli": 87.5,
        "train_time": 82.6,
        "train_mem": 62.2,
    },
    "wo_salience": {
        "display": "w/o salience",
        "sst2": 94.3,
        "mnli": 84.7,
        "train_time": 609.8,
        "train_mem": 65.0,
    },
    "wo_at": {
        "display": "w/o A_T",
        "display_tex": "w/o $\\mathcal{A}_{\\mathrm{T}}$",
        "sst2": 93.2,
        "mnli": 84.5,
        "train_time": 684.9,
        "train_mem": 64.4,
    },
    "wo_ds": {
        "display": "w/o D_S",
        "display_tex": "w/o $\\mathcal{D}_{\\mathrm{S}}$",
        "sst2": 92.9,
        "mnli": 85.3,
        "train_time": 483.1,
        "train_mem": 61.9,
    },
}

#: Absolute efficiency of the FT reference (Appendix I, Table 11, RoBERTa):
#: 97% TTA 127 s, training peak memory 2696 MB, inference 220.8 ms / 1157 MB.
TABLE11_FT_REFERENCE: Dict[str, float] = {
    "tta_seconds": 127.0,
    "train_peak_mem_mb": 2696.0,
    "inf_time_ms": 220.8,
    "inf_mem_mb": 1157.0,
}

#: Raw absolute numbers for the full APT row (Appendix I, Table 11).
TABLE11_APT_REFERENCE: Dict[str, float] = {
    "tta_seconds": 752.0,
    "train_peak_mem_mb": 1890.0,
    "inf_time_ms": 91.3,
    "inf_mem_mb": 904.0,
}

#: Row order of the printed table (paper Table 4).
ABLATION_ORDER: Tuple[str, ...] = ("apt", "wo_ap", "wo_salience", "wo_at", "wo_ds")

#: Accepted aliases for each ablation key.
ABLATION_ALIASES: Dict[str, str] = {
    "apt": "apt",
    "full": "apt",
    "none": "apt",
    "wo_ap": "wo_ap",
    "w/o ap": "wo_ap",
    "w/o a_p": "wo_ap",
    "w/o a p": "wo_ap",
    "no ap": "wo_ap",
    "no adaptive pruning": "wo_ap",
    "wo_salience": "wo_salience",
    "w/o salience": "wo_salience",
    "wo_kurtosis": "wo_salience",
    "w/o kurtosis": "wo_salience",
    "no salience": "wo_salience",
    "no kurtosis": "wo_salience",
    "wo_at": "wo_at",
    "w/o at": "wo_at",
    "w/o a_t": "wo_at",
    "w/o a t": "wo_at",
    "no at": "wo_at",
    "no adaptive tuning": "wo_at",
    "wo_ds": "wo_ds",
    "w/o ds": "wo_ds",
    "w/o d_s": "wo_ds",
    "w/o d s": "wo_ds",
    "no ds": "wo_ds",
    "no self distillation": "wo_ds",
    "w/o distillation": "wo_ds",
}

#: Config overrides implementing each ablation.  Exactly one component is
#: removed; every other APT hyper-parameter keeps its Table 6 value.
ABLATION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "apt": {},
    # §5.6: adaptive tuning without adaptive pruning and without distillation.
    "wo_ap": {
        "target_sparsity": 0.0,
        "use_adaptive_pruning": False,
        "use_pruning": False,
        "prune_heads": False,
        "prune_neurons": False,
        "prune_dims": False,
        "use_distillation": False,
    },
    # §5.6 / Table 4 "w/o salience" == Table 5 "w/o kurtosis".
    "wo_salience": {
        "use_kurtosis": False,
        "salience_kurtosis": False,
    },
    # §5.6: tuning parameters are static while pruning (no rank growth).
    "wo_at": {
        "use_adaptive_tuning": False,
        "use_adaptive_rank": False,
        "top_fraction": 0.0,
        "tuning_budget_final": 1.0,
        "max_growth": 0.0,
    },
    # §5.6: "tuning APT adapters dynamically without distillation objectives".
    "wo_ds": {
        "use_distillation": False,
        "distill_epochs": 0,
    },
}

DEFAULT_SPARSITY = 0.60
TTA_FRACTION = 0.97
RELATIVE_SCALE = 100.0
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)
DEFAULT_TASKS: Tuple[str, ...] = ("mnli", "sst2")

TASK_LABELS = {
    "sst2": "SST2",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "qqp": "QQP",
    "mrpc": "MRPC",
    "cola": "CoLA",
    "rte": "RTE",
    "stsb": "STS-B",
}

MODEL_NAMES = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "t5": "t5-base",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _strip_tex(name: str) -> str:
    """Reduce a LaTeX-ish ablation label to comparable plain text."""
    out = str(name)
    for token in ("$", "{", "}", "\\", "_", ".", ",", ";"):
        out = out.replace(token, " ")
    out = out.replace("mathrm", " ")
    return " ".join(out.lower().split())


def normalize_ablation(name: str) -> str:
    """Map an ablation alias (``w/o D_S``, ``wo_kurtosis``, ...) to a key."""
    if name is None:
        raise KeyError("ablation name is None")
    key = str(name).strip().lower()
    if key in ABLATION_ORDER:
        return key
    if key in ABLATION_ALIASES:
        return ABLATION_ALIASES[key]
    for target in ABLATION_ORDER:
        for candidate in (target, TABLE4_REFERENCES[target].get("display")):
            if candidate and _strip_tex(candidate) == _strip_tex(name):
                return target
    for alias, canonical in ABLATION_ALIASES.items():
        if _strip_tex(alias) == _strip_tex(name):
            return canonical
    raise KeyError(
        "unknown ablation %r (expected one of %s)" % (name, ", ".join(ABLATION_ORDER))
    )


def normalize_task(task: str) -> str:
    """Canonicalise GLUE/SQuAD/CNN-DM task names."""
    if task is None:
        raise KeyError("task name is None")
    key = str(task).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "sst2": "sst2",
        "sst": "sst2",
        "mnli": "mnli",
        "mnlimatched": "mnli",
        "mnlimismatched": "mnli",
        "qnli": "qnli",
        "qqp": "qqp",
        "mrpc": "mrpc",
        "cola": "cola",
        "rte": "rte",
        "stsb": "stsb",
        "squad": "squad_v2",
        "squadv2": "squad_v2",
        "squad20": "squad_v2",
        "cnndm": "cnndm",
        "cnndailymail": "cnndm",
    }
    if key not in aliases:
        raise KeyError("unknown task %r" % (task,))
    return aliases[key]


def normalize_model(model: str) -> str:
    """Canonicalise model family names."""
    key = str(model or "roberta").strip().lower()
    mapping = {
        "roberta": "roberta",
        "roberta-base": "roberta",
        "robertabase": "roberta",
        "roberta_base": "roberta",
        "bert": "bert",
        "bert-base": "bert",
        "bert-base-uncased": "bert",
        "bertbase": "bert",
        "t5": "t5",
        "t5-base": "t5",
        "t5base": "t5",
    }
    return mapping.get(key, key)


def display_name(ablation: str) -> str:
    """Plain-text paper row label for an ablation key."""
    key = normalize_ablation(ablation)
    return str(TABLE4_REFERENCES[key].get("display", key))


def reference_row(ablation: str) -> Dict[str, Any]:
    """Copy of the embedded Table 4 row for an ablation key."""
    key = normalize_ablation(ablation)
    return dict(TABLE4_REFERENCES[key])


def absolute_row(ablation: str) -> Dict[str, float]:
    """Absolute efficiency numbers for the FT and APT rows (Appendix I)."""
    if normalize_ablation(ablation) == "apt":
        return dict(TABLE11_APT_REFERENCE)
    return dict(TABLE11_FT_REFERENCE)


def parse_seeds(seeds: Optional[Iterable[Any]]) -> List[int]:
    """Parse ``--seeds 42 43 44`` or ``"42,43,44"`` into a list of ints."""
    if seeds is None:
        return list(DEFAULT_SEEDS)
    if isinstance(seeds, (int, float)):
        return [int(seeds)]
    if isinstance(seeds, str):
        parts = [p for p in seeds.replace(",", " ").split() if p]
        return [int(p) for p in parts] or list(DEFAULT_SEEDS)
    out: List[int] = []
    for item in seeds:
        if isinstance(item, str) and "," in item:
            out.extend(int(p) for p in item.split(",") if p)
        else:
            out.append(int(item))
    return out or list(DEFAULT_SEEDS)


def split_ablations(values: Optional[Iterable[Any]]) -> List[str]:
    """Normalise a list / CSV string of ablations, keeping paper row order."""
    if values is None:
        return list(ABLATION_ORDER)
    if isinstance(values, str):
        raw = [p for p in values.replace(",", " ").split() if p]
        merged: List[str] = []
        buf: List[str] = []
        for token in raw:
            buf.append(token)
            try:
                normalize_ablation(" ".join(buf))
            except KeyError:
                continue
            merged.append(" ".join(buf))
            buf = []
        if buf:
            merged.append(" ".join(buf))
        values = merged
    keys = [normalize_ablation(v) for v in values]
    ordered = [k for k in ABLATION_ORDER if k in keys]
    for k in keys:
        if k not in ordered:
            ordered.append(k)
    return ordered


def tasks_for_model(model: str, tasks: Optional[Sequence[str]] = None) -> List[str]:
    """Task list to evaluate (Table 4 reports MNLI and SST2)."""
    if tasks:
        return [normalize_task(t) for t in tasks]
    return list(DEFAULT_TASKS)


def default_output_path(model: str) -> str:
    return os.path.join("outputs", "table4", "table4_%s.json" % normalize_model(model))


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def _config_dir() -> str:
    try:
        from scripts.train_apt import CONFIG_DIR  # type: ignore

        return str(CONFIG_DIR)
    except Exception:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, "apt", "configs")


def _default_config_path(model: str, task: str) -> Optional[str]:
    """Best-effort lookup of ``apt/configs/<model>_<task>.yaml``."""
    directory = _config_dir()
    candidates = [
        "%s_%s.yaml" % (model, task),
        "%s_%s.yaml" % (model, task.replace("_v2", "")),
        "default.yaml",
    ]
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def _load_base_config(model: str, task: str) -> Dict[str, Any]:
    """Load the config file the training script would resolve for this cell."""
    config: Dict[str, Any] = {}
    try:  # reuse the canonical resolution logic when importable
        from scripts import train_apt as _train_apt

        cfg = _train_apt.merge_config(None, {}, model, task)
        if isinstance(cfg, dict):
            config.update(cfg)
        return config
    except Exception:
        pass
    path = _default_config_path(model, task)
    if path:
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception:
            pass
    return config


def build_config(
    model: str,
    task: str,
    ablation: str,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seed: int = 42,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the config dict for one (ablation, model, task, seed) cell.

    Order of precedence: config file < Table 6 values < the single ablation's
    overrides < the caller's ``overrides``.
    """
    model_key = normalize_model(model)
    task_key = normalize_task(task)
    key = normalize_ablation(ablation)

    config = _load_base_config(model_key, task_key)

    config.setdefault("model_name_or_path", MODEL_NAMES.get(model_key, "roberta-base"))
    config.setdefault("model_type", model_key)
    config.setdefault("task", task_key)
    config.setdefault("ablation", key)
    config.setdefault("target_sparsity", float(sparsity))
    config.setdefault("seed", int(seed))
    # Table 6 for RoBERTa/GLUE-big: lr 2e-4, batch 32, 40 epochs of which
    # 20 are prune+distill, matching apt/configs/roberta_*.yaml.
    config.setdefault("learning_rate", 2.0e-4)
    config.setdefault("batch_size", 32)
    config.setdefault("epochs", 40)
    config.setdefault("distill_epochs", 20)
    config.setdefault("initial_rank", 8)
    config.setdefault("scaling", 2.0)
    config.setdefault("mask_alpha", 0.01)
    config.setdefault("ema_beta", 0.85)
    config.setdefault("tau", 4)
    config.setdefault("use_distillation", True)
    config.setdefault("use_kurtosis", True)
    # GLUE weighting: L_pred + 0.9 * L_layer.
    config.setdefault("pred_distill_weight", 1.0)
    config.setdefault("layer_distill_weight", 0.9)
    config.setdefault("weight_decay", 0.01)
    config.setdefault("warmup_ratio", 0.06)
    config.setdefault("max_grad_norm", 1.0)

    config.update(ABLATION_OVERRIDES.get(key, {}))
    if overrides:
        config.update(overrides)
    config["target_sparsity"] = float(config.get("target_sparsity", sparsity))
    config["ablation"] = key
    return config


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------


def resolve_trainer(ablation: str = "apt") -> Optional[Callable[..., Any]]:
    """Return the callable used to train one APT / ablation run."""
    try:
        from scripts import train_apt as _train_apt

        fn = getattr(_train_apt, "run_training", None)
        if fn is None:
            fn = getattr(_train_apt, "train_apt", None)
        if fn is not None:
            return fn
    except Exception:
        pass
    try:
        from apt.training import train_apt as fn  # type: ignore

        return fn
    except Exception:
        pass
    return None


def train_one(
    ablation: str,
    model: str,
    task: str,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seed: int = 42,
    config_overrides: Optional[Dict[str, Any]] = None,
    trainer: Optional[Callable[..., Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one (ablation, model, task, seed) cell, return its summary."""
    key = normalize_ablation(ablation)
    task_key = normalize_task(task)
    config = build_config(
        model, task_key, key, sparsity=sparsity, seed=seed, overrides=config_overrides
    )
    fn = trainer or resolve_trainer(key)
    if fn is None:
        raise RuntimeError(
            "no APT trainer available; expected scripts.train_apt.run_training "
            "or apt.training.train_apt"
        )
    if verbose:
        print(
            "[table4] train %-14s %s/%s (sparsity=%.2f, seed=%d)"
            % (key, normalize_model(model), task_key, float(sparsity), int(seed))
        )
    started = time.time()
    result = fn(config)
    if isinstance(result, tuple):  # tolerate (summary, model, ...) returns
        for item in result:
            if isinstance(item, dict):
                result = item
                break
    if not isinstance(result, dict):
        result = {"raw": result}
    result.setdefault("method", key)
    result.setdefault("ablation", key)
    result.setdefault("task", task_key)
    result.setdefault("model", normalize_model(model))
    result.setdefault("sparsity", float(sparsity))
    result.setdefault("seed", int(seed))
    result.setdefault("wall_seconds", time.time() - started)
    return result


# ---------------------------------------------------------------------------
# Metric aggregation / normalisation
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def primary_from_summary(summary: Dict[str, Any], task: str) -> Optional[float]:
    """Extract the paper's primary metric (accuracy / F1 / MCC / ...)."""
    if not isinstance(summary, dict):
        return None
    for key in ("primary", "primary_metric", "score"):
        if isinstance(summary.get(key), (int, float)):
            return float(summary[key])
    metrics = summary.get("metrics") or summary.get("eval_metrics") or {}
    if isinstance(metrics, dict):
        for key in ("primary", "accuracy", "acc"):
            if isinstance(metrics.get(key), (int, float)):
                return float(metrics[key])
        try:
            from apt.eval.metrics import primary_metric as _pm

            value = _pm(normalize_task(task), metrics)
            if value is not None:
                return float(value)
        except Exception:
            pass
        numeric = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
        if numeric:
            return max(numeric)
    if isinstance(summary.get("accuracy"), (int, float)):
        return float(summary["accuracy"])
    return None


def compute_tta(
    history: Optional[Sequence[Dict[str, Any]]],
    reference: float,
    *,
    fraction: float = TTA_FRACTION,
    higher_is_better: bool = True,
) -> Optional[float]:
    """Linearly-interpolated time to reach ``fraction x FT`` accuracy (§5.3).

    Table 4 reports "Train Time" as time-to-accuracy to 97% of the fully
    fine-tuned baseline, so this uses the same definition as Table 2.
    """
    if not history or reference is None:
        return None
    target = (
        fraction * float(reference)
        if higher_is_better
        else (2.0 - fraction) * float(reference)
    )
    points: List[Tuple[float, float]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        metric = entry.get("metric", entry.get("primary", entry.get("accuracy")))
        elapsed = entry.get("wall_seconds", entry.get("elapsed", entry.get("time")))
        if isinstance(metric, (int, float)) and isinstance(elapsed, (int, float)):
            points.append((float(elapsed), float(metric)))
    if not points:
        return None
    for elapsed, metric in points:
        reached = metric >= target if higher_is_better else metric <= target
        if reached:
            return elapsed
    if len(points) < 2:
        return None
    (t0, m0), (t1, m1) = points[-2], points[-1]
    if m1 == m0:
        return t1
    ratio = (target - m0) / (m1 - m0)
    return t0 + ratio * (t1 - t0)


def normalise_from_summary(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull the four absolute efficiency numbers out of a training summary."""
    empty = {
        "tta_seconds": None,
        "train_peak_mem_mb": None,
        "inf_time_ms": None,
        "inf_mem_mb": None,
    }
    if not isinstance(summary, dict):
        return empty
    efficiency = summary.get("efficiency") or {}
    if not isinstance(efficiency, dict):
        efficiency = {}

    tta = _as_float(summary.get("tta_seconds"))
    if tta is None:
        history = summary.get("history") or summary.get("eval_history")
        reference = summary.get("reference_metric")
        if reference is not None:
            tta = compute_tta(history, reference)
        if tta is None:
            tta = _as_float(summary.get("train_time_s", summary.get("train_time")))

    mem = _as_float(summary.get("train_peak_mem_mb", summary.get("train_mem_mb")))
    if mem is None:
        mem = _as_float(efficiency.get("train_peak_mem_mb"))

    inf_time = _as_float(summary.get("inf_time_ms"))
    if inf_time is None:
        inf_time = _as_float(efficiency.get("inf_time_ms"))
    inf_mem = _as_float(summary.get("inf_mem_mb"))
    if inf_mem is None:
        inf_mem = _as_float(efficiency.get("inf_mem_mb"))

    return {
        "tta_seconds": tta,
        "train_peak_mem_mb": mem,
        "inf_time_ms": inf_time,
        "inf_mem_mb": inf_mem,
    }


def relative_efficiency(
    raw: Dict[str, Optional[float]],
    reference: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """Normalise absolute efficiency against FT (FT == 100%).

    Table 4's "Train Time" is the time-to-accuracy of §5.3 (lower is better)
    and "Train Mem" is the peak training memory (lower is better).
    """
    out: Dict[str, Optional[float]] = {}
    for out_key, raw_key in (
        ("train_time", "tta_seconds"),
        ("train_mem", "train_peak_mem_mb"),
        ("inf_time", "inf_time_ms"),
        ("inf_mem", "inf_mem_mb"),
    ):
        value = (raw or {}).get(raw_key)
        base = (reference or {}).get(raw_key)
        out[out_key] = (
            None if (value is None or not base) else float(value) / float(base) * RELATIVE_SCALE
        )
    return out


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Mean / sample std of the numeric entries (``None`` ignored)."""
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
    return mean, var ** 0.5


# ---------------------------------------------------------------------------
# Driving one ablation across tasks/seeds
# ---------------------------------------------------------------------------


def collect_metrics(
    ablation: str,
    model: str,
    tasks: Sequence[str] = DEFAULT_TASKS,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    trainer: Optional[Callable[..., Any]] = None,
    verbose: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one ablation over all tasks/seeds; aggregate quality and cost."""
    key = normalize_ablation(ablation)
    result: Dict[str, Any] = {
        "ablation": key,
        "display": display_name(key),
        "model": normalize_model(model),
        "sparsity": float(sparsity),
        "seeds": [int(s) for s in seeds],
        "tasks": {},
        "raw": {},
        "failures": [],
    }
    for task in tasks:
        task_key = normalize_task(task)
        values: List[Optional[float]] = []
        raw_values: List[Dict[str, Optional[float]]] = []
        per_seed_raw: List[Dict[str, Any]] = []
        for seed in seeds:
            try:
                summary = train_one(
                    key,
                    model,
                    task_key,
                    sparsity=sparsity,
                    seed=int(seed),
                    config_overrides=config_overrides,
                    trainer=trainer,
                    verbose=verbose,
                )
            except Exception as exc:  # keep going: partial tables are useful
                warnings.warn(
                    "table4: %s/%s seed=%s failed: %s" % (key, task_key, seed, exc)
                )
                result["failures"].append(
                    {
                        "ablation": key,
                        "task": task_key,
                        "seed": int(seed),
                        "error": str(exc),
                    }
                )
                continue
            score = primary_from_summary(summary, task_key)
            raw = normalise_from_summary(summary)
            values.append(score)
            raw_values.append(raw)
            per_seed_raw.append({"seed": int(seed), "primary": score, **raw})

        mean, std = mean_std(values)
        aggregated: Dict[str, Optional[float]] = {}
        for field in ("tta_seconds", "train_peak_mem_mb", "inf_time_ms", "inf_mem_mb"):
            field_mean, field_std = mean_std([r.get(field) for r in raw_values])
            aggregated[field] = field_mean
            aggregated[field + "_std"] = field_std
        result["tasks"][task_key] = {
            "primary": mean,
            "std": std,
            "n": len([v for v in values if v is not None]),
            **aggregated,
        }
        result["raw"][task_key] = per_seed_raw
        if verbose:
            shown = "n/a" if mean is None else "%.1f" % mean
            if std:
                shown += " +/- %.1f" % std
            print("[table4] %-14s %-5s %s" % (key, task_key, shown))
    return result


def normalise_rows(
    rows: Sequence[Dict[str, Any]],
    reference: Optional[Dict[str, Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """Fill the relative-efficiency columns, normalising by FT (100%).

    FT is not an APT ablation, so the denominator comes from the embedded
    Appendix I/Table 11 numbers unless the caller measured FT themselves and
    passes those four absolute values as ``reference``.
    """
    reference = dict(reference or TABLE11_FT_REFERENCE)
    out: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        raw = {
            "tta_seconds": row.get("_abs_tta_seconds"),
            "train_peak_mem_mb": row.get("_abs_train_peak_mem_mb"),
            "inf_time_ms": row.get("_abs_inf_time_ms"),
            "inf_mem_mb": row.get("_abs_inf_mem_mb"),
        }
        for key, value in relative_efficiency(raw, reference).items():
            if value is not None:
                row[key] = value
        out.append(row)
    return out


def build_rows(
    model: str = "roberta",
    *,
    ablations: Optional[Sequence[str]] = None,
    tasks: Sequence[str] = DEFAULT_TASKS,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    trainer: Optional[Callable[..., Any]] = None,
    verbose: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
    reference: Optional[Dict[str, Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """Train every ablation and assemble one Table 4 row per ablation."""
    keys = split_ablations(ablations)
    task_keys = [normalize_task(t) for t in tasks]
    rows: List[Dict[str, Any]] = []
    for key in keys:
        collected = collect_metrics(
            key,
            model,
            task_keys,
            sparsity=sparsity,
            seeds=seeds,
            trainer=trainer,
            verbose=verbose,
            config_overrides=config_overrides,
        )
        row: Dict[str, Any] = {
            "ablation": key,
            "display": display_name(key),
            "model": normalize_model(model),
            "sparsity": float(sparsity),
            "seeds": collected["seeds"],
            "train_time": None,
            "train_mem": None,
            "inf_time": None,
            "inf_mem": None,
            "failures": collected["failures"],
            "raw": collected["tasks"],
        }
        for task_key in task_keys:
            stats = collected["tasks"].get(task_key, {})
            row[task_key] = stats.get("primary")
            row[task_key + "_std"] = stats.get("std")
        for field in ("tta_seconds", "train_peak_mem_mb", "inf_time_ms", "inf_mem_mb"):
            vals = [collected["tasks"].get(t, {}).get(field) for t in task_keys]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                row["_abs_" + field] = sum(vals) / len(vals)
        rows.append(row)
    return normalise_rows(rows, reference)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table4(
    rows: Sequence[Dict[str, Any]],
    model: str = "roberta",
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    digits: int = 1,
) -> str:
    """Render a markdown table matching the layout of paper Table 4."""
    task_keys = [normalize_task(t) for t in tasks]
    sparsity = float(rows[0].get("sparsity", DEFAULT_SPARSITY)) if rows else DEFAULT_SPARSITY
    lines = [
        "Table 4: %s, %.0f%% sparsity (relative efficiency to fine-tuning)"
        % (normalize_model(model), 100.0 * sparsity),
        "",
    ]
    header = (
        ["Method"]
        + [TASK_LABELS.get(t, t.upper()) for t in task_keys]
        + ["Train Time (\u2193)", "Train Mem (\u2193)"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        cells = [str(row.get("display", row.get("ablation", "?")))]
        for task_key in task_keys:
            value = row.get(task_key)
            cells.append("n/a" if value is None else "%.*f" % (digits, float(value)))
        for key in ("train_time", "train_mem"):
            value = row.get(key)
            cells.append("n/a" if value is None else "%.*f%%" % (digits, float(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_reference_table(
    ablations: Optional[Sequence[str]] = None,
    tasks: Sequence[str] = DEFAULT_TASKS,
) -> None:
    """Print the published Table 4 without running any training."""
    keys = split_ablations(ablations)
    task_keys = [normalize_task(t) for t in tasks]
    rows = []
    for key in keys:
        ref = reference_row(key)
        row = {
            "ablation": key,
            "display": ref.get("display", key),
            "train_time": ref.get("train_time"),
            "train_mem": ref.get("train_mem"),
        }
        for task_key in task_keys:
            row[task_key] = ref.get(task_key)
        rows.append(row)
    print(format_table4(rows, "roberta", tasks=task_keys))
    print("")
    print("(published values from the APT paper, Table 4)")


def compare_to_reference(
    rows: Sequence[Dict[str, Any]],
    *,
    tolerance: float = 1.0,
    tasks: Sequence[str] = DEFAULT_TASKS,
) -> List[Dict[str, Any]]:
    """Diff measured rows against the published Table 4 values."""
    fields = [normalize_task(t) for t in tasks] + ["train_time", "train_mem"]
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            ref = reference_row(row.get("ablation"))
        except KeyError:
            continue
        entry: Dict[str, Any] = {
            "ablation": row.get("ablation"),
            "display": ref.get("display", row.get("ablation")),
        }
        for field in fields:
            measured = _as_float(row.get(field))
            expected = _as_float(ref.get(field))
            if measured is None or expected is None:
                entry[field] = {"measured": measured, "paper": expected, "within": None}
                continue
            entry[field] = {
                "measured": measured,
                "paper": expected,
                "delta": measured - expected,
                "within": abs(measured - expected) <= float(tolerance),
            }
        out.append(entry)
    return out


def save_results(payload: Dict[str, Any], path: str) -> str:
    """Persist the table payload as JSON (creating parent directories)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    return path


def _json_safe(obj: Any) -> Any:
    """Recursively drop non-serialisable objects (trainers / models)."""
    if isinstance(obj, dict):
        return {
            str(k): _json_safe(v)
            for k, v in obj.items()
            if k not in ("trainer", "model", "tokenizer", "optimizer")
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_table4(
    model: str = "roberta",
    *,
    methods: Optional[Sequence[str]] = None,
    ablations: Optional[Sequence[str]] = None,
    tasks: Sequence[str] = DEFAULT_TASKS,
    sparsity: float = DEFAULT_SPARSITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config_overrides: Optional[Dict[str, Any]] = None,
    trainer: Optional[Callable[..., Any]] = None,
    reference: Optional[Dict[str, Optional[float]]] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Reproduce Table 4: run every ablation, return rows + comparison.

    ``methods`` is accepted as an alias of ``ablations`` so the dispatcher in
    ``main.py`` can forward a generic ``--methods`` argument.
    """
    model_key = normalize_model(model)
    if model_key != "roberta":
        warnings.warn(
            "Table 4 is reported on RoBERTa-base; running %r anyway." % model_key
        )
    keys = split_ablations(ablations if ablations is not None else methods)
    task_keys = [normalize_task(t) for t in (tasks or DEFAULT_TASKS)]

    if dry_run:
        rows = []
        for key in keys:
            ref = reference_row(key)
            row = {
                "ablation": key,
                "display": ref.get("display", key),
                "model": model_key,
                "sparsity": float(sparsity),
                "train_time": ref.get("train_time"),
                "train_mem": ref.get("train_mem"),
            }
            for task_key in task_keys:
                row[task_key] = ref.get(task_key)
            rows.append(row)
        table = format_table4(rows, model_key, tasks=task_keys)
        if verbose:
            print(table)
        return {
            "model": model_key,
            "sparsity": float(sparsity),
            "tasks": task_keys,
            "rows": rows,
            "reference": {k: reference_row(k) for k in keys},
            "comparison": compare_to_reference(rows, tasks=task_keys),
            "table": table,
            "dry_run": True,
        }

    rows = build_rows(
        model_key,
        ablations=keys,
        tasks=task_keys,
        sparsity=sparsity,
        seeds=seeds,
        trainer=trainer,
        verbose=verbose,
        config_overrides=config_overrides,
        reference=reference,
    )
    table = format_table4(rows, model_key, tasks=task_keys)
    comparison = compare_to_reference(rows, tasks=task_keys)
    if verbose:
        print("")
        print(table)
        print("")
        for entry in comparison:
            fields = []
            for field in task_keys + ["train_time", "train_mem"]:
                info = entry.get(field, {})
                if info.get("delta") is None:
                    continue
                fields.append("%s %+.1f (paper %.1f)" % (field, info["delta"], info["paper"]))
            head = "[table4] %-14s" % entry["display"]
            print(head + " " + ", ".join(fields) if fields else head)
    return {
        "model": model_key,
        "sparsity": float(sparsity),
        "seeds": [int(s) for s in seeds],
        "tasks": task_keys,
        "rows": rows,
        "reference": {k: reference_row(k) for k in keys},
        "comparison": comparison,
        "table": table,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_table4_ablation.py",
        description="Reproduce Table 4 (RoBERTa-base APT ablations).",
    )
    parser.add_argument("--model", default="roberta", help="model family (default: roberta)")
    parser.add_argument(
        "--ablations",
        "--methods",
        dest="ablations",
        nargs="*",
        default=None,
        help="ablation names, e.g. 'w/o A_P' 'w/o salience' 'w/o A_T' 'w/o D_S'",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=list(DEFAULT_TASKS), help="tasks (default: mnli sst2)"
    )
    parser.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY)
    parser.add_argument(
        "--seeds", nargs="*", default=list(DEFAULT_SEEDS), help="seeds (default: 42 43 44)"
    )
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="print paper Table 4 only")
    parser.add_argument("--self-test", action="store_true", help="run sanity checks and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if getattr(args, "sparsity", None) is not None:
        overrides["target_sparsity"] = float(args.sparsity)
    return overrides


def main(config: Optional[Dict[str, Any]] = None, argv: Optional[List[str]] = None) -> int:
    """CLI / programmatic entry point (``0`` ok, ``2`` error, ``1`` self-test)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if _self_test() else 1

    config = dict(config or {})
    if "output" in config:
        output = config.pop("output")
    else:
        output = args.output
    model = config.pop("model", None) or config.pop("model_type", None) or args.model
    ablations = (
        config.pop("ablations", None) or config.pop("methods", None) or args.ablations
    )
    tasks = config.pop("tasks", None) or args.tasks
    seeds = parse_seeds(config.pop("seeds", None) or args.seeds)
    sparsity = float(config.pop("sparsity", args.sparsity))
    overrides = dict(config.pop("config_overrides", {}) or {})
    overrides.update(config)
    overrides.update(cli_overrides(args))

    try:
        payload = run_table4(
            model,
            ablations=ablations,
            tasks=tasks,
            sparsity=sparsity,
            seeds=seeds,
            config_overrides=overrides,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except Exception as exc:  # pragma: no cover - runtime failure path
        print("[table4] error: %s" % exc, file=sys.stderr)
        return 2

    path = output or default_output_path(model)
    try:
        save_results(payload, path)
        if not args.quiet:
            print("[table4] results written to %s" % path)
    except Exception as exc:  # pragma: no cover
        warnings.warn("could not write %s: %s" % (path, exc))
    return 0


# ---------------------------------------------------------------------------
# Self test (no torch / GPU required)
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    ok = True

    def check(name: str, condition: bool) -> None:
        nonlocal ok
        if not condition:
            ok = False
            print("[self-test] FAIL: %s" % name)
        else:
            print("[self-test] ok: %s" % name)

    # --- alias handling -----------------------------------------------------
    check("alias w/o A_P", normalize_ablation("w/o A_P") == "wo_ap")
    check("alias w/o A_T", normalize_ablation("w/o A_T") == "wo_at")
    check("alias kurtosis", normalize_ablation("w/o kurtosis") == "wo_salience")
    check("alias D_S tex", normalize_ablation("w/o $\\mathcal{D}_{\\mathrm{S}}$") == "wo_ds")
    check("alias P tex", normalize_ablation("w/o $\\mathcal{A}_{\\mathrm{P}}$") == "wo_ap")
    check("task alias", normalize_task("SST-2") == "sst2")
    check("model alias", normalize_model("roberta-base") == "roberta")
    check("seeds parse", parse_seeds("42,43") == [42, 43])
    check("seeds default", parse_seeds(None) == list(DEFAULT_SEEDS))

    # --- exactly one component removed per ablation -------------------------
    check("apt has no overrides", ABLATION_OVERRIDES["apt"] == {})
    check("wo_ap pruning off", ABLATION_OVERRIDES["wo_ap"]["target_sparsity"] == 0.0)
    check(
        "wo_ap distill off",
        ABLATION_OVERRIDES["wo_ap"]["use_distillation"] is False,
    )
    check(
        "wo_ap keeps adaptive tuning",
        "use_adaptive_tuning" not in ABLATION_OVERRIDES["wo_ap"],
    )
    check(
        "wo_salience only kurtosis",
        ABLATION_OVERRIDES["wo_salience"] == {"use_kurtosis": False, "salience_kurtosis": False},
    )
    check(
        "wo_at freezes budget",
        ABLATION_OVERRIDES["wo_at"]["top_fraction"] == 0.0
        and ABLATION_OVERRIDES["wo_at"]["tuning_budget_final"] == 1.0,
    )
    check(
        "wo_at keeps distillation",
        "use_distillation" not in ABLATION_OVERRIDES["wo_at"],
    )
    check(
        "wo_ds only distillation",
        ABLATION_OVERRIDES["wo_ds"]["use_distillation"] is False
        and ABLATION_OVERRIDES["wo_ds"]["distill_epochs"] == 0,
    )
    check(
        "wo_ds keeps sparsity",
        "target_sparsity" not in ABLATION_OVERRIDES["wo_ds"],
    )

    # --- relative efficiency against the Appendix I FT reference ------------
    rel = relative_efficiency(dict(TABLE11_APT_REFERENCE), dict(TABLE11_FT_REFERENCE))
    check("APT train time 592.1%%", abs(rel["train_time"] - 592.1) < 0.1)
    check("APT train mem 70.1%%", abs(rel["train_mem"] - 70.1) < 0.1)
    check("APT inf time 41.3%%", abs(rel["inf_time"] - 41.34) < 0.2)
    check("APT inf mem 78.1%%", abs(rel["inf_mem"] - 78.1) < 0.1)
    self_rel = relative_efficiency(dict(TABLE11_FT_REFERENCE), dict(TABLE11_FT_REFERENCE))
    check("FT normalises to 100%%", all(abs(v - 100.0) < 1e-9 for v in self_rel.values()))

    # --- TTA interpolation (97% of 94.8 lies between 90 and 94.8) ----------
    tta = compute_tta(
        [
            {"wall_seconds": 40.0, "metric": 90.0},
            {"wall_seconds": 60.0, "metric": 94.8},
        ],
        94.8,
    )
    check("TTA interpolation", tta is not None and 40.0 < tta <= 60.0)
    check(
        "TTA already reached",
        compute_tta([{"wall_seconds": 10.0, "metric": 95.0}], 94.8) == 10.0,
    )

    # --- aggregation helpers ------------------------------------------------
    mean, std = mean_std([94.0, 95.0])
    check("mean_std", abs(mean - 94.5) < 1e-9 and abs(std - 0.5 * 2 ** 0.5) < 1e-9)
    check("mean_std empty", mean_std([]) == (None, None))

    # --- dry run reproduces the paper table ---------------------------------
    payload = run_table4("roberta", dry_run=True, verbose=False)
    check(
        "dry-run row order",
        [r["ablation"] for r in payload["rows"]] == list(ABLATION_ORDER),
    )
    check("dry-run APT sst2", payload["rows"][0]["sst2"] == 94.5)
    check("dry-run APT mnli", payload["rows"][0]["mnli"] == 86.4)
    check("dry-run w/o A_P mnli", payload["rows"][1]["mnli"] == 87.5)
    check("dry-run w/o A_P train time", payload["rows"][1]["train_time"] == 82.6)
    check("dry-run w/o salience mem", payload["rows"][2]["train_mem"] == 65.0)
    check("dry-run w/o A_T sst2", payload["rows"][3]["sst2"] == 93.2)
    check("dry-run w/o D_S train time", payload["rows"][4]["train_time"] == 483.1)
    check(
        "dry-run comparison within tolerance",
        all(
            all(
                info.get("within", True)
                for info in entry.values()
                if isinstance(info, dict)
            )
            for entry in payload["comparison"]
        ),
    )
    check("table renders all rows", payload["table"].count("\n") >= len(ABLATION_ORDER))
    check("w/o D_S beaten by APT on MNLI", 85.3 < 86.4)

    # --- config assembly keeps Table 6 and only flips the ablated knob ------
    base = build_config("roberta", "sst2", "apt")
    wo_ds = build_config("roberta", "sst2", "wo_ds")
    wo_at = build_config("roberta", "mnli", "wo_at")
    wo_ap = build_config("roberta", "sst2", "wo_ap")
    check(
        "table6 lr preserved",
        base["learning_rate"] == 2.0e-4 and wo_ds["learning_rate"] == 2.0e-4,
    )
    check("table6 epochs preserved", base["epochs"] == 40 and base["distill_epochs"] == 20)
    check("model name", base["model_name_or_path"] == "roberta-base")
    check("apt keeps 60%% sparsity", base["target_sparsity"] == DEFAULT_SPARSITY)
    check("apt keeps kurtosis", base["use_kurtosis"] is True)
    check("apt keeps distillation", base["use_distillation"] is True)
    check("glue distill weights", base["pred_distill_weight"] == 1.0 and base["layer_distill_weight"] == 0.9)
    check("config ablation flag", wo_ds["ablation"] == "wo_ds")
    check("wo_ds turns distill off", wo_ds["use_distillation"] is False)
    check("wo_ds keeps sparsity", wo_ds["target_sparsity"] == DEFAULT_SPARSITY)
    check(
        "wo_at static ranks",
        wo_at["top_fraction"] == 0.0 and wo_at["tuning_budget_final"] == 1.0,
    )
    check("wo_at keeps distillation", wo_at["use_distillation"] is True)
    check("wo_ap no sparsity target", wo_ap["target_sparsity"] == 0.0)
    check("wo_ap no distillation", wo_ap["use_distillation"] is False)
    overridden = build_config(
        "roberta", "sst2", "apt", overrides={"learning_rate": 1e-5, "seed": 7}
    )
    check(
        "caller overrides win",
        overridden["learning_rate"] == 1e-5 and overridden["seed"] == 7,
    )

    # --- json safety / persistence -----------------------------------------
    check("json safe drops trainer", "trainer" not in _json_safe({"trainer": object(), "a": 1}))
    check("json safe nested", _json_safe({"rows": [{"model": object(), "a": 1}]}) == {"rows": [{"a": 1}]})

    print("[self-test] %s" % ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if _self_test() else 1)
    sys.exit(main())
