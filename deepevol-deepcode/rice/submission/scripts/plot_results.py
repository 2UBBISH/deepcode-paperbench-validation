#!/usr/bin/env python
"""Plot / tabulate RICE reproduction results.

This script is intentionally *result-driven*: it never runs RL itself.  It scans
the JSON + TXT artifacts produced by the experiment drivers

    experiments/exp1_fidelity_efficiency.py   -> fidelity (Table 4 / Exp I)
    experiments/exp2_refine_effectiveness.py  -> refining (Table 1 / Exp II, Fig. 2)
    experiments/exp3_explanation_quality.py   -> explanation quality (Exp III / Table 6)
    experiments/exp4_sac_agent.py             -> SAC agent (Exp IV / Fig. 3)
    experiments/exp5_hyperparams.py           -> sensitivity (Exp V / Fig. 7-9)
    scripts/run_ablation.py                   -> combined ablation summaries

and renders the paper's tables (Table 1, 4, 5, 6) and figures (Figure 2, 3, 7, 8, 9)
as CSV / Markdown / (optionally) Matplotlib images under ``results/plots``.

Usage
-----
    python scripts/plot_results.py --results-dir results --out-dir results/plots
    python scripts/plot_results.py --experiment exp2 --format md csv
    python scripts/plot_results.py --no-plots            # tables only

The module keeps a defensive import policy: Matplotlib is optional and the
script degrades to writing CSV/Markdown/text tables when it is missing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - project utility layer
    from rice.utils.io import ensure_dir, get_config, load_json, save_json
except Exception:  # pragma: no cover

    def ensure_dir(path: str) -> str:  # type: ignore[misc]
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def load_json(path: str, default: Any = None) -> Any:  # type: ignore[misc]
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore[misc]
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, default=str)
        return path

    def get_config(name: str = "default", config_dir: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[misc]
        root = config_dir or os.path.join(os.getcwd(), "configs")
        path = name if os.path.isabs(name) else os.path.join(root, name)
        if not path.endswith((".yaml", ".yml")):
            path = path + ".yaml"
        merged: Dict[str, Any] = {}
        default_path = os.path.join(root, "default.yaml")
        try:
            import yaml  # type: ignore

            if os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as fh:
                    merged = yaml.safe_load(fh) or {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    override = yaml.safe_load(fh) or {}
                merged = _deep_merge(merged, override)
        except Exception:
            return merged
        return merged


try:  # pragma: no cover - project logging layer
    from rice.utils.logging import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = 20):  # type: ignore[misc]
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
            logger.addHandler(handler)
            logger.setLevel(level)
        if out_dir:
            try:
                ensure_dir(out_dir)
                fh = logging.FileHandler(os.path.join(out_dir, "plot_log.txt"))
                fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
                logger.addHandler(fh)
            except Exception:
                pass
        return logger


try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

_MATPLOTLIB_AVAILABLE = False
try:  # pragma: no cover - optional
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    _MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = os.path.join("results")
DEFAULT_OUT_DIR = os.path.join("results", "plots")

EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4", "exp5", "ablation")

#: Method display order used across the paper's tables.
METHOD_ORDER: Tuple[str, ...] = (
    "no_refine",
    "ours",
    "statemask_r",
    "ppo_finetune",
    "jsrl",
    "sac_finetune",
    "sil",
    "gail",
    "statemask",
    "random",
    "integrated_gradients",
    "airs",
)

METHOD_LABELS: Dict[str, str] = {
    "no_refine": "No Refine",
    "ours": "Ours",
    "statemask_r": "StateMask-R",
    "ppo_finetune": "PPO fine-tuning",
    "jsrl": "JSRL",
    "sac_finetune": "SAC fine-tuning",
    "sil": "SIL",
    "gail": "GAIL",
    "statemask": "StateMask",
    "random": "Random",
    "integrated_gradients": "Integrated Gradients",
    "airs": "AIRS",
}

ENV_LABELS: Dict[str, str] = {
    "hopper": "Hopper",
    "walker2d": "Walker2d",
    "reacher": "Reacher",
    "halfcheetah": "HalfCheetah",
    "sparse_hopper": "Sparse Hopper",
    "sparse_halfcheetah": "Sparse HalfCheetah",
    "selfish_mining": "Selfish Mining",
    "cage2": "CAGE-2",
    "autodriving": "Auto Driving",
    "malware_mutation": "Malware Mutation",
}

DEFAULT_ENVS: Tuple[str, ...] = (
    "hopper",
    "walker2d",
    "reacher",
    "halfcheetah",
    "selfish_mining",
    "cage2",
    "autodriving",
)

MUJOCO_ENVS: Tuple[str, ...] = ("hopper", "walker2d", "reacher", "halfcheetah")

#: Paper reference values (Table 1 left block: refining methods, explanation = Ours).
REFERENCE_TABLE1: Dict[str, Dict[str, Optional[float]]] = {
    "hopper": {
        "no_refine": 3559.44,
        "ours": 3663.91,
        "statemask_r": 3662.16,
        "ppo_finetune": 3600.62,
        "jsrl": 3625.19,
    },
    "walker2d": {
        "no_refine": 3339.68,
        "ours": 3423.28,
        "statemask_r": 3404.20,
        "ppo_finetune": 3365.74,
        "jsrl": 3390.82,
    },
    "reacher": {
        "no_refine": -4.19,
        "ours": -3.90,
        "statemask_r": -3.97,
        "ppo_finetune": -4.11,
        "jsrl": -4.00,
    },
    "halfcheetah": {
        "no_refine": 4540.50,
        "ours": 4663.75,
        "statemask_r": 4643.63,
        "ppo_finetune": 4560.94,
        "jsrl": 4600.06,
    },
    "selfish_mining": {
        "no_refine": None,
        "ours": None,
        "statemask_r": None,
        "ppo_finetune": None,
        "jsrl": None,
    },
    "cage2": {"no_refine": -23.64, "ours": -20.02, "statemask_r": -22.15, "ppo_finetune": -23.10, "jsrl": -22.54},
    "autodriving": {"no_refine": 10.30, "ours": 17.03, "statemask_r": 15.08, "ppo_finetune": 10.98, "jsrl": 12.61},
}

#: Table 1 right block: explanation methods with refining fixed to Ours.
REFERENCE_TABLE1_EXPLANATIONS: Dict[str, Dict[str, Optional[float]]] = {
    "hopper": {"random": 3575.13, "statemask": 3662.16, "ours": 3663.91},
    "walker2d": {"random": 3350.44, "statemask": 3420.13, "ours": 3423.28},
    "reacher": {"random": -4.13, "statemask": -3.91, "ours": -3.90},
    "halfcheetah": {"random": 4566.32, "statemask": 4655.61, "ours": 4663.75},
    "cage2": {"random": -22.90, "statemask": -20.19, "ours": -20.02},
    "autodriving": {"random": 12.44, "statemask": 16.62, "ours": 17.03},
}

#: Table 6 (Appendix C.3): other explanation methods, refining = Ours (MuJoCo only).
REFERENCE_TABLE6: Dict[str, Dict[str, Optional[float]]] = {
    "hopper": {"random": 3575.13, "integrated_gradients": 3592.71, "airs": 3611.48, "statemask": 3662.16, "ours": 3663.91},
    "walker2d": {"random": 3350.44, "integrated_gradients": 3368.05, "airs": 3381.90, "statemask": 3420.13, "ours": 3423.28},
    "reacher": {"random": -4.13, "integrated_gradients": -4.06, "airs": -4.02, "statemask": -3.91, "ours": -3.90},
    "halfcheetah": {
        "random": 4566.32,
        "integrated_gradients": 4588.47,
        "airs": 4610.92,
        "statemask": 4655.61,
        "ours": 4663.75,
    },
}

#: Table 4: mask-network training wall-clock (seconds) for a fixed sample budget.
REFERENCE_TABLE4: Dict[str, Dict[str, Optional[float]]] = {
    "hopper": {"ours": 12426.0, "statemask": 15393.0},
    "halfcheetah": {"ours": 1317.0, "statemask": 1579.0},
    "cage2": {"ours": 65400.0, "statemask": 79382.0},
}
PAPER_TIME_REDUCTION = 0.168  # ~16.8%

#: Table-4 fixed mask sample budgets (Table 3 / 4 of the reproduction plan).
MASK_SAMPLES: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
}

#: Table 3 hyperparameters per application.
TABLE3: Dict[str, Dict[str, float]] = {
    "hopper": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "walker2d": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "reacher": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "halfcheetah": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "selfish_mining": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "cage2": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "autodriving": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
}

K_VALUES: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
P_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDA_VALUES: Tuple[float, ...] = (0.0, 0.1, 0.01, 0.001)
ALPHA_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)

NEGATIVE_REWARD_ENVS: Tuple[str, ...] = ("reacher", "cage2")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def env_label(env_id: str) -> str:
    key = str(env_id).strip().lower().replace("-", "_")
    return ENV_LABELS.get(key, str(env_id))


def method_label(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    return METHOD_LABELS.get(key, str(method))


def is_negative_env(env_id: str) -> bool:
    key = str(env_id).strip().lower()
    return any(neg in key for neg in NEGATIVE_REWARD_ENVS)


def deep_get(obj: Any, *keys: Any, default: Any = None) -> Any:
    """Nested lookup supporting dicts and sequences (int indices)."""
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
                continue
            if isinstance(key, str):
                lower = {str(k).lower(): v for k, v in cur.items()}
                if key.lower() in lower:
                    cur = lower[key.lower()]
                    continue
            return default
        if isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(key)]
                continue
            except Exception:
                return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        return default
    return cur


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_format(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:.{decimals}f}"


def format_mean_std(mean: Any, std: Any, decimals: int = 2) -> str:
    m = _to_float(mean)
    s = _to_float(std)
    if m is None:
        return "n/a"
    if s is None:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} +- {s:.{decimals}f}"


def series_mean_std(values: Sequence[Any], decimals: int = 2) -> Tuple[Optional[float], Optional[float]]:
    nums = [_to_float(v) for v in (values or [])]
    nums = [n for n in nums if n is not None]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) == 1:
        return mean, None
    var = sum((n - mean) ** 2 for n in nums) / (len(nums) - 1)
    return mean, math.sqrt(var)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:
        return float(obj)
    except Exception:
        return str(obj)


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------
@dataclass
class ResultTable:
    """A renderable table: header row + data rows (all strings)."""

    name: str
    header: List[str]
    rows: List[List[str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add(self, *cells: Any) -> None:
        self.rows.append(["" if c is None else str(c) for c in cells])

    def to_csv(self, path: str) -> str:
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.header)
            writer.writerows(self.rows)
        return path

    def to_markdown(self) -> str:
        lines = [f"### {self.name}", ""]
        lines.append("| " + " | ".join(self.header) + " |")
        lines.append("| " + " | ".join(["---"] * len(self.header)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"> {note}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "header": list(self.header),
            "rows": [list(r) for r in self.rows],
            "notes": list(self.notes),
        }


@dataclass
class PlotSeries:
    """A labelled series for a figure (x values may be numeric or categorical)."""

    label: str
    x: List[Any] = field(default_factory=list)
    y: List[Optional[float]] = field(default_factory=list)
    yerr: List[Optional[float]] = field(default_factory=list)


@dataclass
class Figure:
    name: str
    title: str
    xlabel: str
    ylabel: str
    series: List[PlotSeries] = field(default_factory=list)
    xtick_labels: Optional[List[str]] = None


# --------------------------------------------------------------------------------------
# Result loading
# --------------------------------------------------------------------------------------
class ResultStore:
    """Discovers and lazily loads JSON artifacts produced by experiment drivers."""

    def __init__(self, results_dir: str = DEFAULT_RESULTS_DIR, logger: Any = None) -> None:
        self.results_dir = results_dir
        self.logger = logger
        self._cache: Dict[str, Any] = {}
        self._files: Dict[str, List[str]] = {}

    # -- discovery -------------------------------------------------------------------
    def dir_for(self, experiment: str) -> str:
        exp = str(experiment).strip().lower()
        if os.path.isabs(exp) and os.path.isdir(exp):
            return exp
        candidate = os.path.join(self.results_dir, exp)
        return candidate if os.path.isdir(candidate) else candidate

    def find(self, experiment: str, pattern: str = "*.json") -> List[str]:
        exp = str(experiment).strip().lower()
        if exp in self._files:
            return self._files[exp]
        folder = self.dir_for(exp)
        found: List[str] = []
        if os.path.isdir(folder):
            found = sorted(glob.glob(os.path.join(folder, pattern)))
        if not found and os.path.isdir(self.results_dir):
            # Fall back to a flat naming scheme: results/exp1_hopper.json
            found = sorted(glob.glob(os.path.join(self.results_dir, f"{exp}*{pattern.lstrip('*')}")))
        self._files[exp] = found
        return found

    def load(self, path: str) -> Optional[Dict[str, Any]]:
        if path in self._cache:
            return self._cache[path]
        try:
            data = load_json(path)
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - corrupt artifact
            if self.logger:
                self.logger.warning("Could not read %s (%s)", path, exc)
            return None
        self._cache[path] = data
        return data

    def load_all(self, experiment: str) -> Dict[str, Dict[str, Any]]:
        """Return ``{filename_stem: payload}`` for every artifact of an experiment."""
        out: Dict[str, Dict[str, Any]] = {}
        for path in self.find(experiment):
            data = self.load(path)
            if data is None:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            out[stem] = data
        if not out and self.logger:
            self.logger.info("No %s artifacts found under %s", experiment, self.dir_for(experiment))
        return out

    def available(self) -> Dict[str, List[str]]:
        return {exp: [os.path.basename(p) for p in self.find(exp)] for exp in EXPERIMENTS}


# --------------------------------------------------------------------------------------
# Extraction helpers (payloads differ slightly between drivers -> be tolerant)
# --------------------------------------------------------------------------------------
def extract_env(payload: Dict[str, Any], fallback: Optional[str] = None) -> str:
    env = deep_get(payload, "env_id")
    if env:
        return str(env)
    env = deep_get(payload, "env", "id")
    if env:
        return str(env)
    return fallback or "unknown"


def extract_reward(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort (mean, std) reward from a result payload."""
    for key in ("final_reward", "mean_reward", "reward", "eval_reward", "mean"):
        value = deep_get(payload, key)
        if value is not None and not isinstance(value, (dict, list)):
            std = deep_get(payload, "std") or deep_get(payload, "std_reward") or deep_get(payload, f"{key}_std")
            return _to_float(value), _to_float(std)
    # nested eval block
    mean = deep_get(payload, "eval", "mean_reward")
    std = deep_get(payload, "eval", "std_reward")
    if mean is not None:
        return _to_float(mean), _to_float(std)
    return None, None


def extract_method(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("method", "explanation", "baseline"):
        value = deep_get(payload, key)
        if isinstance(value, str) and value:
            return value.strip().lower()
    return None


def extract_fidelity(payload: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """Return ``{scoring/method: {K: (mean, std)}}`` from an Exp-I payload."""
    out: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

    def _absorb(method: str, block: Any) -> None:
        if not isinstance(block, dict):
            return
        per_k: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for k, value in block.items():
            if isinstance(value, dict):
                mean = _to_float(deep_get(value, "mean"))
                std = _to_float(deep_get(value, "std"))
            else:
                mean, std = _to_float(value), None
            if mean is not None:
                per_k[str(k)] = (mean, std)
        if per_k:
            out.setdefault(method.strip().lower(), {}).update(per_k)

    # shape A: {"fidelity": {"ours": {"0.1": {...}}}}
    fid = deep_get(payload, "fidelity")
    if isinstance(fid, dict):
        if any(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values()) for v in fid.values()):
            for method, block in fid.items():
                _absorb(str(method), block)
        else:
            _absorb(str(payload.get("method", payload.get("scoring", "ours"))), fid)

    # shape B: {"results": {"ours": {"K": ...}}}
    results = deep_get(payload, "results")
    if isinstance(results, dict) and not out:
        for method, block in results.items():
            _absorb(str(method), block)

    # shape C: {"methods": [{"name": "ours", "fidelity": {...}}]}
    methods = deep_get(payload, "methods")
    if isinstance(methods, list) and not out:
        for entry in methods:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("method") or "ours"
                _absorb(str(name), entry.get("fidelity") or entry)

    # shape D: single method payload {"method": "ours", "fidelity_by_K": {...}}
    if not out:
        block = deep_get(payload, "fidelity_by_K")
        if isinstance(block, dict):
            _absorb(str(payload.get("method", payload.get("scoring", "ours"))), block)
    return out


def extract_refine_results(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return ``{method: {"mean":..., "std":..., "env_id":...}}`` from Exp-II payloads."""
    out: Dict[str, Dict[str, Any]] = {}
    methods = deep_get(payload, "methods")
    if isinstance(methods, dict):
        for method, info in methods.items():
            if not isinstance(info, dict):
                continue
            mean = _to_float(deep_get(info, "mean", default=deep_get(info, "final_reward")))
            std = _to_float(deep_get(info, "std", default=deep_get(info, "std_reward")))
            if mean is None:
                mean, std = extract_reward(info)
            if mean is not None or info:
                out[str(method).strip().lower()] = {
                    "mean": mean,
                    "std": std,
                    "env_id": str(deep_get(info, "env_id", default=extract_env(payload))),
                    "raw": info,
                }
    elif isinstance(methods, list):
        for entry in methods:
            if not isinstance(entry, dict):
                continue
            method = extract_method(entry)
            if not method:
                continue
            mean, std = extract_reward(entry)
            out[method] = {"mean": mean, "std": std, "env_id": extract_env(entry, extract_env(payload)), "raw": entry}
    # no-refine reference
    no_refine = deep_get(payload, "no_refine")
    if isinstance(no_refine, dict):
        mean = _to_float(deep_get(no_refine, "mean", default=deep_get(no_refine, "final_reward")))
        std = _to_float(deep_get(no_refine, "std"))
        if mean is not None:
            out.setdefault("no_refine", {"mean": mean, "std": std, "env_id": extract_env(payload), "raw": no_refine})
    elif no_refine is not None and "no_refine" not in out:
        out["no_refine"] = {"mean": _to_float(no_refine), "std": None, "env_id": extract_env(payload), "raw": {}}
    return out


def extract_efficiency(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{ours: seconds, statemask: seconds, reduction: frac, samples: int}``."""
    out: Dict[str, Any] = {}
    eff = deep_get(payload, "efficiency")
    blocks = [eff, deep_get(payload, "training_time"), deep_get(payload, "time")]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for method in ("ours", "statemask"):
            value = block.get(method)
            if isinstance(value, dict):
                value = deep_get(value, "train_time", default=deep_get(value, "time", default=deep_get(value, "seconds")))
            seconds = _to_float(value)
            if seconds is not None and method not in out:
                out[method] = seconds
        reduction = _to_float(block.get("reduction", block.get("time_reduction")))
        if reduction is not None:
            out.setdefault("reduction", reduction)
    for method in ("ours", "statemask"):
        if method in out:
            continue
        seconds = _to_float(deep_get(payload, "artifacts", method, "train_time"))
        if seconds is not None:
            out[method] = seconds
    if "samples" not in out:
        samples = deep_get(payload, "mask_timesteps", default=deep_get(payload, "samples"))
        if samples is not None:
            out["samples"] = samples
    if "ours" in out and "statemask" in out and "reduction" not in out and out["statemask"]:
        out["reduction"] = (out["statemask"] - out["ours"]) / out["statemask"]
    return out


def extract_sweep_points(payload: Dict[str, Any], key_candidates: Sequence[str]) -> List[Dict[str, Any]]:
    """Extract ``[{param, value, mean, std}]`` from an Exp-V sweep payload."""
    points: List[Dict[str, Any]] = []
    for key in key_candidates:
        block = deep_get(payload, key)
        if isinstance(block, dict):
            # {value: {...}} or {"points": [...]}
            inner = block.get("points") if isinstance(block.get("points"), list) else None
            if inner is None:
                inner = [
                    {"value": k, **(v if isinstance(v, dict) else {"mean": v})}
                    for k, v in block.items()
                    if _to_float(k) is not None
                ]
            for entry in inner:
                if not isinstance(entry, dict):
                    continue
                value = _to_float(entry.get("value", entry.get("param_value", entry.get("alpha"))))
                mean, std = extract_reward(entry)
                if value is not None and mean is not None:
                    points.append({"param": key, "value": value, "mean": mean, "std": std, "raw": entry})
        elif isinstance(block, list):
            for entry in block:
                if not isinstance(entry, dict):
                    continue
                value = _to_float(entry.get("value", entry.get("param_value", entry.get("alpha"))))
                mean, std = extract_reward(entry)
                if value is not None and mean is not None:
                    points.append({"param": key, "value": value, "mean": mean, "std": std, "raw": entry})
    return points


# --------------------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------------------
def build_table1(store: ResultStore, include_reference: bool = True) -> ResultTable:
    """Table 1 left block: refining method comparison (explanation = Ours)."""
    table = ResultTable(
        name="Table 1 - Refining effectiveness (explanation = Ours)",
        header=["Environment", "No Refine", "Ours", "StateMask-R", "PPO fine-tuning", "JSRL"],
    )
    methods = ("no_refine", "ours", "statemask_r", "ppo_finetune", "jsrl")
    observed: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

    for stem, payload in store.load_all("exp2").items():
        env = extract_env(payload, fallback=stem)
        if env == "unknown":
            continue
        blocks = {**extract_refine_results(payload)}
        # Some drivers nest per-method payloads under "results"
        nested = deep_get(payload, "results")
        if isinstance(nested, dict):
            blocks.update(extract_refine_results(nested))
        bucket = observed.setdefault(str(env).lower(), {})
        for method, info in blocks.items():
            bucket.setdefault(method, (info.get("mean"), info.get("std")))

    envs = sorted(set(list(observed.keys()) + (list(REFERENCE_TABLE1.keys()) if include_reference else [])))
    for env in envs:
        row_observed = observed.get(env, {})
        row_ref = REFERENCE_TABLE1.get(env, {})
        cells: List[str] = [env_label(env)]
        for method in methods:
            if method in row_observed and row_observed[method][0] is not None:
                cells.append(format_mean_std(*row_observed[method]))
            elif include_reference:
                cells.append(safe_format(row_ref.get(method)))
            else:
                cells.append("n/a")
        table.add(*cells)
    if include_reference and not observed:
        table.notes.append("No exp2 artifacts found - table shows paper reference values (Table 1 left block).")
    elif include_reference:
        table.notes.append("Cells use observed results where available, otherwise paper reference values.")
    return table


def build_table1_explanations(store: ResultStore, include_reference: bool = True) -> ResultTable:
    """Table 1 right block: explanation comparison (refining = Ours)."""
    table = ResultTable(
        name="Table 1 - Explanation quality (refining = Ours)",
        header=["Environment", "Random", "StateMask", "Ours"],
    )
    methods = ("random", "statemask", "ours")
    observed: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

    for stem, payload in store.load_all("exp3").items():
        env = extract_env(payload, fallback=stem)
        if env == "unknown":
            continue
        blocks = extract_refine_results(payload)
        nested = deep_get(payload, "results")
        if isinstance(nested, dict):
            blocks.update(extract_refine_results(nested))
        bucket = observed.setdefault(str(env).lower(), {})
        for method, info in blocks.items():
            bucket.setdefault(method, (info.get("mean"), info.get("std")))

    envs = sorted(set(list(observed.keys()) + (list(REFERENCE_TABLE1_EXPLANATIONS.keys()) if include_reference else [])))
    for env in envs:
        row_observed = observed.get(env, {})
        row_ref = REFERENCE_TABLE1_EXPLANATIONS.get(env, {})
        cells: List[str] = [env_label(env)]
        for method in methods:
            if method in row_observed and row_observed[method][0] is not None:
                cells.append(format_mean_std(*row_observed[method]))
            elif include_reference:
                cells.append(safe_format(row_ref.get(method)))
            else:
                cells.append("n/a")
        table.add(*cells)
    if include_reference:
        table.notes.append("Ours and StateMask are comparable; both should beat Random (strict Ours > StateMask is NOT required).")
    return table


def build_table4(store: ResultStore, include_reference: bool = True) -> ResultTable:
    """Table 4: fidelity across K and mask-training wall-clock efficiency."""
    table = ResultTable(
        name="Table 4 - Fidelity & mask-net training efficiency",
        header=["Environment", "Method"] + [f"K={int(k * 100)}%" for k in K_VALUES] + ["Train time (s)", "vs StateMask"],
    )
    observed: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]] = {}
    efficiency: Dict[str, Dict[str, Any]] = {}

    for stem, payload in store.load_all("exp1").items():
        env = extract_env(payload, fallback=stem.split("_")[0])
        if env == "unknown":
            continue
        fid = extract_fidelity(payload)
        if fid:
            observed.setdefault(str(env).lower(), {}).update(fid)
        eff = extract_efficiency(payload)
        if eff:
            efficiency.setdefault(str(env).lower(), {}).update(eff)

    envs = sorted(set(list(observed.keys()) + list(efficiency.keys()) + (list(REFERENCE_TABLE4.keys()) if include_reference else [])))
    for env in envs:
        methods = list(observed.get(env, {}).keys()) or ["ours", "statemask"]
        methods = [m for m in ("ours", "statemask", "random") if m in methods] or methods
        for method in methods:
            cells: List[str] = [env_label(env), method_label(method)]
            per_k = observed.get(env, {}).get(method, {})
            for k in K_VALUES:
                key = _match_k(per_k, k)
                if key is not None:
                    cells.append(format_mean_std(*per_k[key], decimals=3))
                else:
                    cells.append("n/a")
            ours = _to_float(efficiency.get(env, {}).get("ours"))
            statemask = _to_float(efficiency.get(env, {}).get("statemask"))
            if method == "ours" and ours is not None:
                time_str = safe_format(ours, 0)
                if statemask:
                    red = (statemask - ours) / statemask
                    time_str += f" ({100 * red:.1f}% faster)"
                cells.append(time_str)
                cells.append(f"{100 * (statemask - ours) / statemask:.1f}%" if statemask else "n/a")
            elif method == "statemask" and statemask is not None:
                cells.append(safe_format(statemask, 0))
                cells.append("baseline")
            elif include_reference and method in REFERENCE_TABLE4.get(env, {}):
                ref_time = _to_float(REFERENCE_TABLE4[env][method])
                cells.append(safe_format(ref_time, 0))
                if method == "ours" and REFERENCE_TABLE4[env].get("statemask"):
                    base = REFERENCE_TABLE4[env]["statemask"]
                    cells.append(f"{100 * (base - ref_time) / base:.1f}%")
                elif method == "statemask":
                    cells.append("baseline")
                else:
                    cells.append("n/a")
            else:
                cells.append("n/a")
                cells.append("n/a")
            table.add(*cells)
    if include_reference:
        table.notes.append(f"Paper reports ~{100 * PAPER_TIME_REDUCTION:.1f}% lower mask-training time than StateMask for a fixed budget.")
        table.notes.append("Fidelity = log(d/d_max) - log(l/L); higher is better.")
    return table


def build_table5(store: ResultStore) -> ResultTable:
    """Table 5: RICE vs SIL on the four MuJoCo applications."""
    table = ResultTable(
        name="Table 5 - RICE vs SIL (MuJoCo)",
        header=["Environment", "Ours", "SIL"],
    )
    observed: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    for stem, payload in store.load_all("ablation").items():
        env = extract_env(payload, fallback=stem)
        blocks = extract_refine_results(payload)
        if not blocks:
            continue
        bucket = observed.setdefault(str(env).lower(), {})
        for method, info in blocks.items():
            bucket.setdefault(method, (info.get("mean"), info.get("std")))
    for stem, payload in store.load_all("exp2").items():
        env = extract_env(payload, fallback=stem)
        blocks = extract_refine_results(payload)
        if "sil" in blocks:
            observed.setdefault(str(env).lower(), {})["sil"] = (blocks["sil"]["mean"], blocks["sil"]["std"])
    for env in MUJOCO_ENVS:
        row = observed.get(env, {})
        table.add(
            env_label(env),
            format_mean_std(*row["ours"]) if "ours" in row and row["ours"][0] is not None else "n/a",
            format_mean_std(*row["sil"]) if "sil" in row and row["sil"][0] is not None else "n/a",
        )
    if not observed:
        table.notes.append("No SIL artifacts found (run scripts/run_ablation.py --ablation sil).")
    return table


def build_table6(store: ResultStore, include_reference: bool = True) -> ResultTable:
    """Table 6: other explanation methods (Integrated Gradients, AIRS) vs Random/Ours."""
    table = ResultTable(
        name="Table 6 - Other explanation methods (refining = Ours)",
        header=["Environment", "Random", "Integrated Gradients", "AIRS", "StateMask", "Ours"],
    )
    methods = ("random", "integrated_gradients", "airs", "statemask", "ours")
    observed: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    for stem, payload in store.load_all("exp3").items():
        env = extract_env(payload, fallback=stem)
        blocks = extract_refine_results(payload)
        nested = deep_get(payload, "results")
        if isinstance(nested, dict):
            blocks.update(extract_refine_results(nested))
        bucket = observed.setdefault(str(env).lower(), {})
        for method, info in blocks.items():
            bucket.setdefault(method, (info.get("mean"), info.get("std")))
    for env in MUJOCO_ENVS:
        row_observed = observed.get(env, {})
        row_ref = REFERENCE_TABLE6.get(env, {})
        cells: List[str] = [env_label(env)]
        for method in methods:
            if method in row_observed and row_observed[method][0] is not None:
                cells.append(format_mean_std(*row_observed[method]))
            elif include_reference:
                cells.append(safe_format(row_ref.get(method)))
            else:
                cells.append("n/a")
        table.add(*cells)
    return table


def build_hyperparam_table(store: ResultStore, sweep: str) -> ResultTable:
    """Exp-V sweep table (p / lambda / alpha)."""
    if sweep == "p":
        values, header_name, ref_name = P_VALUES, "p", "Mixed init probability p"
    elif sweep == "lambda":
        values, header_name, ref_name = LAMBDA_VALUES, "lambda", "RND coefficient lambda"
    else:
        values, header_name, ref_name = ALPHA_VALUES, "alpha", "Blinding bonus alpha"
    table = ResultTable(
        name=f"Experiment V - {ref_name} sensitivity",
        header=["Environment"] + [f"{header_name}={v}" for v in values],
    )
    for stem, payload in store.load_all("exp5").items():
        env = extract_env(payload, fallback=stem)
        points = extract_sweep_points(payload, (sweep, f"{sweep}_sweep", "sweeps"))
        if not points and isinstance(payload, dict):
            points = extract_sweep_points(payload, tuple(payload.keys()))
        points = [p for p in points if p.get("param") in (sweep, f"{sweep}_sweep")]
        if not points:
            continue
        by_value = {round(p["value"], 6): p for p in points}
        cells: List[str] = [env_label(env)]
        for value in values:
            point = by_value.get(round(value, 6))
            cells.append(safe_format(point["mean"]) if point else "n/a")
        table.add(*cells)
    if not store.find("exp5"):
        table.notes.append("No exp5 artifacts found (run experiments/exp5_hyperparams.py).")
    if sweep == "p":
        table.notes.append("Expected: p=0 and p=1 are worse than the mixture; p=0.25/0.5 are best.")
    elif sweep == "lambda":
        table.notes.append("Expected: lambda>0 improves over lambda=0; performance largely insensitive to lambda within a range.")
    else:
        table.notes.append("Expected: fidelity is insensitive to alpha (Fig. 9).")
    return table


def _match_k(per_k: Dict[str, Any], k: float) -> Optional[str]:
    """Find the key in ``per_k`` matching the K fraction/percentage ``k``."""
    targets = {round(k, 6), round(k * 100, 6)}
    for key in per_k.keys():
        num = _to_float(key)
        if num is None:
            continue
        for target in targets:
            if abs(num - target) < 1e-6:
                return key
    # string prefix match, e.g. "K=0.1" or "k10"
    for key in per_k.keys():
        text = str(key).lower()
        if f"{k:g}" in text or f"{k * 100:g}" in text:
            return key
    return None


# --------------------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------------------
def build_figure_fidelity(store: ResultStore) -> Optional[Figure]:
    """Figure 1-style fidelity vs K (Exp I)."""
    series: List[PlotSeries] = []
    for stem, payload in sorted(store.load_all("exp1").items()):
        env = extract_env(payload, fallback=stem.split("_")[0])
        fid = extract_fidelity(payload)
        for method, per_k in sorted(fid.items()):
            xs, ys, es = [], [], []
            for k in K_VALUES:
                key = _match_k(per_k, k)
                if key is None:
                    xs.append(k)
                    ys.append(None)
                    es.append(None)
                    continue
                xs.append(k)
                ys.append(per_k[key][0])
                es.append(per_k[key][1])
            series.append(PlotSeries(label=f"{env_label(env)} / {method_label(method)}", x=xs, y=ys, yerr=es))
    if not series:
        return None
    return Figure(
        name="fidelity_vs_K",
        title="Fidelity vs window fraction K",
        xlabel="K (fraction of episode length)",
        ylabel="Fidelity",
        series=series,
        xtick_labels=[f"{int(k * 100)}%" for k in K_VALUES],
    )


def build_figure_refining(store: ResultStore) -> Optional[Figure]:
    """Figure 2-style refining effectiveness (incl. sparse environments)."""
    return _figure_from_refine(store, experiments=("exp2",), name="refining_effectiveness",
                               title="Refining effectiveness (explanation = Ours)")


def build_figure_explanations(store: ResultStore) -> Optional[Figure]:
    """Exp III: explanation quality comparison."""
    return _figure_from_refine(store, experiments=("exp3",), name="explanation_quality",
                               title="Explanation quality (refining = Ours)")


def _figure_from_refine(store: ResultStore, experiments: Sequence[str], name: str, title: str) -> Optional[Figure]:
    env_order: List[str] = []
    data: Dict[str, Dict[str, Optional[float]]] = {}
    for experiment in experiments:
        for stem, payload in sorted(store.load_all(experiment).items()):
            env = extract_env(payload, fallback=stem)
            if env == "unknown":
                continue
            blocks = extract_refine_results(payload)
            if not blocks:
                continue
            bucket = data.setdefault(env, {})
            for method, info in blocks.items():
                if info.get("mean") is not None:
                    bucket.setdefault(method, info["mean"])
            if env not in env_order:
                env_order.append(env)
    if not data:
        return None
    methods = [m for m in METHOD_ORDER if any(m in bucket for bucket in data.values())]
    series = [
        PlotSeries(
            label=method_label(method),
            x=[env_label(env) for env in env_order],
            y=[data.get(env, {}).get(method) for env in env_order],
        )
        for method in methods
    ]
    return Figure(name=name, title=title, xlabel="Environment", ylabel="Episode return", series=series,
                  xtick_labels=[env_label(env) for env in env_order])


def build_figure_sac(store: ResultStore) -> Optional[Figure]:
    """Figure 3-style SAC-agent refining comparison (Exp IV)."""
    fig = _figure_from_refine(store, experiments=("exp4",), name="sac_agent", title="Refining a pre-trained SAC agent")
    if fig is not None:
        return fig
    # Fall back to SAC learning curves if recorded.
    series: List[PlotSeries] = []
    for stem, payload in sorted(store.load_all("exp4").items()):
        history = deep_get(payload, "sac_pretrain", "eval_history", default=deep_get(payload, "eval_history"))
        if isinstance(history, list) and history:
            xs = list(range(len(history)))
            ys = [_to_float(h) for h in history]
            series.append(PlotSeries(label=f"{env_label(extract_env(payload, stem))} / SAC pre-train", x=xs, y=ys))
    if not series:
        return None
    return Figure(name="sac_pretrain", title="SAC pre-training curve", xlabel="evaluation", ylabel="Episode return",
                  series=series)


def build_figure_sweep(store: ResultStore, sweep: str) -> Optional[Figure]:
    """Figures 7-9: hyperparameter sensitivity (Exp V)."""
    if sweep == "p":
        x_values, xlabel, title = P_VALUES, "mixed init probability p", "Sensitivity to p (Fig. 8)"
    elif sweep == "lambda":
        x_values, xlabel, title = LAMBDA_VALUES, "RND coefficient lambda", "Sensitivity to lambda (Fig. 7)"
    else:
        x_values, xlabel, title = ALPHA_VALUES, "blinding bonus alpha", "Sensitivity to alpha (Fig. 9)"
    series: List[PlotSeries] = []
    for stem, payload in sorted(store.load_all("exp5").items()):
        env = extract_env(payload, fallback=stem.split("_")[0])
        points = extract_sweep_points(payload, (sweep, f"{sweep}_sweep", "sweeps"))
        points = [p for p in points if p.get("param") in (sweep, f"{sweep}_sweep")]
        if not points:
            continue
        by_value = {round(p["value"], 6): p for p in points}
        ys = [by_value[round(v, 6)]["mean"] if round(v, 6) in by_value else None for v in x_values]
        es = [by_value[round(v, 6)]["std"] if round(v, 6) in by_value else None for v in x_values]
        series.append(PlotSeries(label=env_label(env), x=list(x_values), y=ys, yerr=es))
    if not series:
        return None
    return Figure(name=f"sweep_{sweep}", title=title, xlabel=xlabel, ylabel="Episode return", series=series,
                  xtick_labels=[f"{v:g}" for v in x_values])


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def render_figure(figure: Figure, out_dir: str, fmt: str = "pdf", dpi: int = 150,
                  figsize: Tuple[float, float] = (7.0, 4.0), logger: Any = None) -> List[str]:
    """Render a :class:`Figure` with Matplotlib, returning written paths."""
    written: List[str] = []
    ensure_dir(out_dir)
    if not _MATPLOTLIB_AVAILABLE:
        if logger:
            logger.info("Matplotlib unavailable - skipping figure '%s'", figure.name)
        return written
    try:
        fig, ax = plt.subplots(figsize=figsize)
        categorical = not all(isinstance(x, (int, float)) for x in figure.series[0].x) if figure.series else False
        n_series = max(1, len(figure.series))
        width = 0.8 / n_series
        if categorical:
            positions = list(range(len(figure.series[0].x)))
            for idx, series in enumerate(figure.series):
                offsets = [p - 0.4 + width * (idx + 0.5) for p in positions]
                ax.bar(offsets, [y if y is not None else 0 for y in series.y], width=width, label=series.label)
            ax.set_xticks(positions)
            ax.set_xticklabels(figure.xtick_labels or [str(x) for x in figure.series[0].x], rotation=15, ha="right")
        else:
            for series in figure.series:
                yerr = series.yerr if any(v is not None for v in series.yerr) else None
                ax.errorbar(series.x, [y if y is not None else math.nan for y in series.y], yerr=yerr,
                            marker="o", capsize=3, label=series.label)
            if figure.xtick_labels:
                ax.set_xticks(list(figure.series[0].x))
                ax.set_xticklabels(figure.xtick_labels)
        ax.set_title(figure.title)
        ax.set_xlabel(figure.xlabel)
        ax.set_ylabel(figure.ylabel)
        ax.grid(alpha=0.3)
        if len(figure.series) > 1:
            ax.legend(fontsize=8)
        fig.tight_layout()
        for extension in ([fmt] if isinstance(fmt, str) else list(fmt)):
            path = os.path.join(out_dir, f"{figure.name}.{extension}")
            fig.savefig(path, dpi=dpi)
            written.append(path)
        if plt is not None:
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - rendering robustness
        if logger:
            logger.warning("Failed to render figure '%s': %s", figure.name, exc)
    return written


def write_tables(tables: Sequence[ResultTable], out_dir: str, formats: Sequence[str] = ("csv", "md")) -> Dict[str, List[str]]:
    written: Dict[str, List[str]] = {}
    ensure_dir(out_dir)
    for table in tables:
        paths: List[str] = []
        slug = _slug(table.name)
        if "csv" in formats:
            paths.append(table.to_csv(os.path.join(out_dir, f"{slug}.csv")))
        if "md" in formats:
            path = os.path.join(out_dir, f"{slug}.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(table.to_markdown())
            paths.append(path)
        written[table.name] = paths
    combined = os.path.join(out_dir, "all_tables.md")
    with open(combined, "w", encoding="utf-8") as fh:
        fh.write("# RICE reproduction - result tables\n\n")
        fh.write(f"Generated from `{os.path.abspath(out_dir)}`\n\n")
        for table in tables:
            fh.write(table.to_markdown())
            fh.write("\n")
    written["combined"] = [combined]
    return written


def _slug(text: str) -> str:
    out = []
    for ch in str(text).lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:80] or "table"


# --------------------------------------------------------------------------------------
# Reporting / trend checks
# --------------------------------------------------------------------------------------
def trend_report(store: ResultStore) -> Dict[str, Any]:
    """Qualitative checks against the paper's in-scope takeaways."""
    checks: Dict[str, Any] = {}

    # Exp II / Table 1: Ours >= every refining baseline.
    for stem, payload in store.load_all("exp2").items():
        env = extract_env(payload, fallback=stem)
        blocks = extract_refine_results(payload)
        ours = (blocks.get("ours") or {}).get("mean")
        if ours is None:
            continue
        env_checks: Dict[str, Any] = {"ours": ours}
        for method in ("ppo_finetune", "statemask_r", "jsrl"):
            other = (blocks.get(method) or {}).get("mean")
            if other is not None:
                env_checks[f"ours_ge_{method}"] = bool(ours >= other - 1e-6)
        no_refine = (blocks.get("no_refine") or {}).get("mean")
        if no_refine is not None:
            env_checks["ours_gt_no_refine"] = bool(ours > no_refine)
            env_checks["beats_no_refine"] = bool(ours > no_refine)
        checks[env] = env_checks

    # Exp III: Ours/StateMask >= Random, Ours comparable to StateMask.
    for stem, payload in store.load_all("exp3").items():
        env = extract_env(payload, fallback=stem)
        blocks = extract_refine_results(payload)
        ours = (blocks.get("ours") or {}).get("mean")
        statemask = (blocks.get("statemask") or {}).get("mean")
        random_mean = (blocks.get("random") or {}).get("mean")
        if ours is None:
            continue
        entry = checks.setdefault(env, {"ours": ours})
        if random_mean is not None:
            entry["ours_ge_random"] = bool(ours >= random_mean - 1e-6)
            if statemask is not None:
                entry["statemask_ge_random"] = bool(statemask >= random_mean - 1e-6)
        if statemask is not None:
            entry["ours_comparable_to_statemask"] = bool(abs(ours - statemask) <= 0.05 * max(1.0, abs(statemask)))

    # Exp I efficiency: our mask training must be faster than StateMask.
    for stem, payload in store.load_all("exp1").items():
        env = extract_env(payload, fallback=stem)
        eff = extract_efficiency(payload)
        ours, sm = _to_float(eff.get("ours")), _to_float(eff.get("statemask"))
        if ours is not None and sm is not None:
            entry = checks.setdefault(env, {})
            entry["mask_time_faster_than_statemask"] = bool(ours < sm)
            entry["mask_time_reduction"] = (sm - ours) / sm

    # Exp V sweeps.
    for sweep, values in (("p", P_VALUES), ("lambda", LAMBDA_VALUES)):
        for stem, payload in store.load_all("exp5").items():
            env = extract_env(payload, fallback=stem)
            points = extract_sweep_points(payload, (sweep, f"{sweep}_sweep"))
            points = [p for p in points if p.get("param") in (sweep, f"{sweep}_sweep")]
            if not points:
                continue
            by_value = {round(p["value"], 6): p["mean"] for p in points}
            entry = checks.setdefault(env, {})
            if sweep == "p":
                interior = [by_value[round(v, 6)] for v in (0.25, 0.5) if round(v, 6) in by_value]
                extremes = [by_value[round(v, 6)] for v in (0.0, 1.0) if round(v, 6) in by_value]
                if interior and extremes:
                    entry["mixed_beats_p0_p1"] = bool(max(interior) >= max(extremes) - 1e-6)
            else:
                nonzero = [v for k, v in by_value.items() if k > 0]
                zero = by_value.get(0.0)
                if nonzero and zero is not None:
                    entry["exploration_helps"] = bool(max(nonzero) >= zero - 1e-6)

    return checks


# --------------------------------------------------------------------------------------
# Top-level API
# --------------------------------------------------------------------------------------
def plot_results(results_dir: str = DEFAULT_RESULTS_DIR, out_dir: str = DEFAULT_OUT_DIR,
                 experiments: Sequence[str] = EXPERIMENTS, formats: Sequence[str] = ("csv", "md"),
                 make_plots: bool = True, include_reference: bool = True, dpi: int = 150,
                 figsize: Tuple[float, float] = (7.0, 4.0), logger: Any = None) -> Dict[str, Any]:
    """Load all artifacts, build tables/figures, write them to ``out_dir``."""
    logger = logger or get_logger("rice.plot", out_dir=out_dir)
    ensure_dir(out_dir)
    store = ResultStore(results_dir=results_dir, logger=logger)
    selected = [e.strip().lower() for e in experiments]

    tables: List[ResultTable] = []
    if "exp2" in selected:
        tables.append(build_table1(store, include_reference=include_reference))
    if "exp3" in selected:
        tables.append(build_table1_explanations(store, include_reference=include_reference))
    if "exp1" in selected:
        tables.append(build_table4(store, include_reference=include_reference))
    if "ablation" in selected or "exp2" in selected:
        tables.append(build_table5(store))
    if "exp3" in selected:
        tables.append(build_table6(store, include_reference=include_reference))
    if "exp5" in selected:
        for sweep in ("p", "lambda", "alpha"):
            tables.append(build_hyperparam_table(store, sweep))

    written = write_tables(tables, out_dir, formats=formats)

    figures: List[Figure] = []
    if "exp1" in selected:
        for fig in (build_figure_fidelity(store),):
            if fig is not None:
                figures.append(fig)
    if "exp2" in selected:
        for fig in (build_figure_refining(store),):
            if fig is not None:
                figures.append(fig)
    if "exp3" in selected:
        for fig in (build_figure_explanations(store),):
            if fig is not None:
                figures.append(fig)
    if "exp4" in selected:
        for fig in (build_figure_sac(store),):
            if fig is not None:
                figures.append(fig)
    if "exp5" in selected:
        for sweep in ("p", "lambda", "alpha"):
            fig = build_figure_sweep(store, sweep)
            if fig is not None:
                figures.append(fig)

    figure_paths: List[str] = []
    if make_plots and figures:
        figure_paths = [p for fig in figures for p in render_figure(fig, out_dir, fmt="pdf", dpi=dpi, figsize=figsize, logger=logger)]

    trends = trend_report(store)
    report = {
        "results_dir": os.path.abspath(results_dir),
        "out_dir": os.path.abspath(out_dir),
        "artifacts_found": store.available(),
        "tables": {table.name: table.to_dict() for table in tables},
        "table_files": written,
        "figure_files": figure_paths,
        "trends": trends,
    }
    save_json(report, os.path.join(out_dir, "plot_report.json"))

    text = format_report(report)
    with open(os.path.join(out_dir, "plot_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Wrote %d tables and %d figures to %s", len(tables), len(figure_paths), out_dir)
    return report


def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("RICE - reproduction result summary")
    lines.append("=" * 88)
    lines.append(f"results dir : {report.get('results_dir')}")
    lines.append(f"output dir  : {report.get('out_dir')}")
    found = report.get("artifacts_found", {})
    lines.append("artifacts   : " + ", ".join(f"{k}={len(v)}" for k, v in found.items()) or "artifacts   : none")
    lines.append("")

    for name, table in report.get("tables", {}).items():
        header = table.get("header", [])
        rows = table.get("rows", [])
        lines.append("-" * 88)
        lines.append(name)
        lines.append("-" * 88)
        col_widths = [max([len(str(header[i]))] + [len(str(r[i])) for r in rows]) for i in range(len(header))] if header else []
        if header:
            lines.append("  " + "  ".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header))))
        for row in rows:
            lines.append("  " + "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))
        for note in table.get("notes", []):
            lines.append(f"  note: {note}")
        lines.append("")

    trends = report.get("trends", {})
    if trends:
        lines.append("-" * 88)
        lines.append("Qualitative trend checks")
        lines.append("-" * 88)
        for env, entry in sorted(trends.items()):
            flags = {k: v for k, v in entry.items() if isinstance(v, bool)}
            lines.append(f"  {env_label(env)}: " + ", ".join(f"{k}={'OK' if v else 'NO'}" for k, v in flags.items()))
        lines.append("")

    if report.get("figure_files"):
        lines.append("Figures: " + ", ".join(os.path.basename(p) for p in report["figure_files"]))
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot / tabulate RICE reproduction results.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory containing exp*/ result JSONs.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to write tables/figures into.")
    parser.add_argument("--experiment", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS),
                        help="Which experiment artifacts to consume.")
    parser.add_argument("--config", default=None, help="Optional config name (e.g. 'hopper') for plot settings.")
    parser.add_argument("--format", nargs="*", default=["csv", "md"], choices=["csv", "md"],
                        help="Table output formats.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure rendering (tables only).")
    parser.add_argument("--no-reference", action="store_true", help="Do not fall back to paper reference values.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            cfg = get_config(args.config)
        except Exception as exc:  # pragma: no cover
            print(f"[warning] could not load config '{args.config}': {exc}")
    plot_cfg = (cfg or {}).get("plot", {}) if isinstance(cfg, dict) else {}

    logger = get_logger("rice.plot", out_dir=args.out_dir, level=10 if args.verbose else 20)
    report = plot_results(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        experiments=args.experiment,
        formats=tuple(args.format),
        make_plots=not args.no_plots,
        include_reference=not args.no_reference,
        dpi=int(plot_cfg.get("dpi", args.dpi) or args.dpi),
        figsize=tuple(plot_cfg.get("figsize", (7.0, 4.0))) if isinstance(plot_cfg.get("figsize"), (list, tuple)) else (7.0, 4.0),
        logger=logger,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
