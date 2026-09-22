#!/usr/bin/env python
"""FOA hyper-parameter sensitivity sweeps (Figure 2, Tables 13-15).

This script reproduces the parameter-sensitivity study of the FOA paper:

    * Figure 2(a)  - CMA-ES population size ``K`` (Table 13)
    * Figure 2(b)  - number of learnable prompts ``N_p``
    * Figure 2(c)  - number of source images ``Q`` used for the ID statistics
    * lambda       - fitness trade-off between entropy and discrepancy (Table 15)
    * shifting EMA - activation-shifting EMA (alpha), varying batch size (Table 14)
    * discrepancy EMA - EMA factor beta on the test-batch statistics

Every sweep simply re-runs Algorithm 1 (``src.method.foa``) with one
hyper-parameter overridden, keeping the frozen, gradient-free protocol of the
paper unchanged: no backward pass is ever performed anywhere in this file.

The sweep definitions (values, per-value overrides, paper reference numbers)
live in ``configs/sensitivity.yaml``.  This module only interprets them.

Usage
-----
    python scripts/run_sensitivity.py --config configs/sensitivity.yaml
    python scripts/run_sensitivity.py --groups lambda num_prompts
    python scripts/run_sensitivity.py --groups population_size --corruptions gaussian_noise
    python scripts/run_sensitivity.py --groups source_samples --limit-batches 4
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# import shims -- allow both "python scripts/run_sensitivity.py" (script mode)
# and "python -m scripts.run_sensitivity" (package mode).
# ----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # package style
    from src.utils.config import Config, config_to_dict, load_config, save_config  # noqa: F401
    from src.method.foa import build_foa, unpack_batch  # noqa: F401
    from src.method.source_stats import (  # noqa: F401
        DEFAULT_NUM_SOURCE_SAMPLES,
        compute_source_statistics,
        load_source_stats,
        save_source_stats,
    )
    from src.models.vit_loader import build_vit  # noqa: F401
except ImportError:  # pragma: no cover - script style fallback
    from utils.config import Config, config_to_dict, load_config, save_config  # type: ignore  # noqa: F401
    from method.foa import build_foa, unpack_batch  # type: ignore  # noqa: F401
    from method.source_stats import (  # type: ignore  # noqa: F401
        DEFAULT_NUM_SOURCE_SAMPLES,
        compute_source_statistics,
        load_source_stats,
        save_source_stats,
    )
    from models.vit_loader import build_vit  # type: ignore  # noqa: F401

# ----------------------------------------------------------------------------
# optional collaborators (kept defensive so the sweep driver always imports)
# ----------------------------------------------------------------------------
try:
    from run_foa import (  # type: ignore
        IMAGENET_C_CORRUPTIONS,
        ResultAccumulator,
        build_test_loader,
        run_over_corruptions,
    )
except ImportError:  # pragma: no cover
    try:
        from scripts.run_foa import (  # type: ignore
            IMAGENET_C_CORRUPTIONS,
            ResultAccumulator,
            build_test_loader,
            run_over_corruptions,
        )
    except ImportError:
        IMAGENET_C_CORRUPTIONS = [
            "gaussian_noise", "shot_noise", "impulse_noise",
            "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
            "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression",
        ]
        ResultAccumulator = None  # type: ignore
        build_test_loader = None  # type: ignore
        run_over_corruptions = None  # type: ignore

try:
    from src.eval.metrics import DEFAULT_ECE_BINS  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_ECE_BINS = 15


# ============================================================================
# sweep registry / paper reference tables
# ============================================================================

SWEEP_GROUPS: Tuple[str, ...] = (
    "population_size",
    "num_prompts",
    "source_samples",
    "lambda",
    "shifting_ema",
    "discrepancy_ema",
)

#: alias -> canonical sweep name (accepts CLI-friendly spellings)
GROUP_ALIASES: Dict[str, str] = {
    "k": "population_size",
    "pop_size": "population_size",
    "population": "population_size",
    "k_sensitivity": "population_size",
    "np": "num_prompts",
    "num_prompt": "num_prompts",
    "prompts": "num_prompts",
    "q": "source_samples",
    "source_samples": "source_samples",
    "source_stats": "source_samples",
    "lambda": "lambda",
    "lambda_value": "lambda",
    "shifting": "shifting_ema",
    "shifting_alpha": "shifting_ema",
    "alpha": "shifting_ema",
    "beta": "discrepancy_ema",
    "discrepancy": "discrepancy_ema",
}

#: default paper reference values (only used when the YAML omits them)
FIGURE2A_REFERENCE: Dict[Any, float] = {
    2: 57.6, 6: 61.3, 10: 63.6, 15: 65.3, 20: 65.6,
    28: 66.3, 40: 66.5, 60: 66.4, 100: 66.2,
}
FIGURE2B_REFERENCE: Dict[Any, float] = {
    1: 65.6, 3: 66.3, 5: 66.2, 8: 66.1, 10: 66.0,
}
FIGURE2C_REFERENCE: Dict[Any, float] = {
    16: 64.4, 32: 66.3, 64: 66.4, 100: 66.4, 200: 66.4,
    400: 66.3, 800: 66.3, 1600: 66.3,
}
LAMBDA_REFERENCE: Dict[Any, float] = {
    0.0: 44.9, 0.1: 60.8, 0.3: 65.9, 0.4: 66.3, 0.5: 66.2,
    0.7: 64.1, 0.1 * 10: 60.8, 1.0: 62.0, 2.0: 58.1, 5.0: 53.7,
}
TABLE13_REFERENCE: Dict[Any, float] = dict(FIGURE2A_REFERENCE)
TABLE14_REFERENCE: Dict[Any, Dict[str, float]] = {
    # batch size -> {"ema": acc, "no_ema": acc}
    1: {"ema": 41.2, "no_ema": 0.1},
    2: {"ema": 51.4, "no_ema": 15.6},
    4: {"ema": 58.3, "no_ema": 38.1},
    8: {"ema": 61.9, "no_ema": 50.2},
    16: {"ema": 63.8, "no_ema": 57.4},
    32: {"ema": 65.4, "no_ema": 62.1},
    64: {"ema": 66.3, "no_ema": 65.0},
}
TABLE15_REFERENCE: Dict[str, Dict[Any, float]] = {
    "imagenet-c": {0.1: 60.8, 0.3: 65.9, 0.4: 66.3, 0.5: 66.2, 1.0: 62.0},
    "imagenet-r": {0.1: 47.8, 0.2: 63.8, 0.3: 63.2, 0.5: 62.1},
}


# ============================================================================
# config helpers
# ============================================================================

def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup working for dict-like and attribute-like configs."""
    if cfg is None:
        return default
    cur = cfg
    for key in keys:
        if isinstance(key, str) and "." in key:
            return _cfg_get(cur, *key.split("."), default=default)
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur


def _cfg_has(cfg: Any, dotted: str) -> bool:
    sentinel = object()
    return _cfg_get(cfg, dotted, default=sentinel) is not sentinel


def _cfg_set(cfg: Any, dotted: str, value: Any) -> None:
    """Set a nested (dotted) config key in place."""
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if nxt is None:
            nxt = {}
            if isinstance(cur, dict):
                cur[part] = nxt
            else:
                setattr(cur, part, nxt)
        cur = nxt
    if isinstance(cur, dict):
        cur[parts[-1]] = value
    else:
        setattr(cur, parts[-1], value)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_device(cfg: Any = None, device: Optional[str] = None):
    import torch

    if device:
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)
    requested = _cfg_get(cfg, "model.device", default="cuda")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_corruptions(cfg: Any, corruptions: Optional[Sequence[str]] = None) -> List[str]:
    if corruptions:
        return [str(c) for c in corruptions]
    listed = _cfg_get(cfg, "data.corruptions", default=None)
    if listed:
        return [str(c) for c in listed]
    single = _cfg_get(cfg, "data.corruption", default=None)
    if single:
        return [str(single)]
    return list(IMAGENET_C_CORRUPTIONS)


def _seed_everything(seed: int = 0) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass
    return seed


def _population_size_rule(dim: int, rule: str = "ceil") -> int:
    """Hansen's default CMA population: K = 4 + 3 * ln(dim)."""
    raw = 4.0 + 3.0 * math.log(max(int(dim), 2))
    rule = (rule or "ceil").lower()
    if rule == "floor":
        return max(2, int(math.floor(raw)))
    if rule == "round":
        return max(2, int(round(raw)))
    if rule == "exact":
        return raw  # type: ignore[return-value]
    return max(2, int(math.ceil(raw)))


# ============================================================================
# single evaluation
# ============================================================================

class _Accumulator:
    """Minimal streaming accuracy/ECE accumulator used when metrics are absent."""

    def __init__(self, n_bins: int = DEFAULT_ECE_BINS):
        self.n_bins = int(n_bins)
        self.reset()

    def reset(self) -> None:
        self._conf: List[np.ndarray] = []
        self._correct: List[np.ndarray] = []

    def update(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets).reshape(-1)
        if logits.ndim != 2:
            return 0.0, 0.0
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == targets)
        self._conf.append(conf)
        self._correct.append(correct)
        return 100.0 * float(correct.mean()), self._ece(conf, correct)

    def _ece(self, conf: np.ndarray, correct: np.ndarray) -> float:
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, self.n_bins - 1)
        total = len(conf)
        ece = 0.0
        for b in range(self.n_bins):
            mask = idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            ece += (n / total) * abs(acc - avg_conf)
        return 100.0 * ece

    def compute(self) -> Dict[str, float]:
        if not self._conf:
            return {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
        conf = np.concatenate(self._conf)
        correct = np.concatenate(self._correct)
        return {
            "accuracy": 100.0 * float(correct.mean()),
            "ece": self._ece(conf, correct),
            "num_samples": int(conf.size),
        }


def _make_accumulator(n_bins: int = DEFAULT_ECE_BINS, class_subset: Optional[Sequence[int]] = None):
    """Prefer ``src.eval.metrics.MetricAccumulator`` when importable."""
    if ResultAccumulator is not None:
        try:
            return ResultAccumulator(ece_bins=n_bins)
        except Exception:
            pass
    try:
        from src.eval.metrics import MetricAccumulator  # type: ignore

        try:
            return MetricAccumulator(n_bins=n_bins)
        except TypeError:
            return MetricAccumulator()
    except Exception:
        return _Accumulator(n_bins=n_bins)


def _run_one_setting(
    cfg: Any,
    *,
    corruptions: Sequence[str],
    device,
    limit_batches: Optional[int] = None,
    model=None,
    source_stats_path: Optional[str] = None,
    class_subset: Optional[Sequence[int]] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Algorithm 1 over ``corruptions`` with the *current* config state."""
    import torch

    if seed is not None:
        _seed_everything(seed)

    # ---- source statistics -------------------------------------------------
    stats = None
    if source_stats_path:
        try:
            stats = load_source_stats(source_stats_path, device="cpu")
        except Exception as exc:  # pragma: no cover
            print(f"[sensitivity] WARNING: could not load source stats '{source_stats_path}': {exc}")
            stats = None
    if stats is None:
        default_path = _cfg_get(cfg, "source_stats.path", default=None)
        if default_path:
            try:
                stats = load_source_stats(default_path, device="cpu")
            except Exception as exc:
                print(f"[sensitivity] WARNING: could not load source stats '{default_path}': {exc}")

    ece_bins = int(_cfg_get(cfg, "eval.ece_bins", default=DEFAULT_ECE_BINS) or DEFAULT_ECE_BINS)

    per_corruption: Dict[str, Any] = {}
    accs: List[float] = []
    eces: List[float] = []
    total_wall = 0.0
    t_start = time.time()

    for corruption in corruptions:
        if seed is not None:
            _seed_everything(seed)
        loader = None
        if build_test_loader is not None:
            try:
                loader = build_test_loader(cfg, corruption=corruption, limit_batches=limit_batches)
            except TypeError:
                try:
                    loader = build_test_loader(cfg, corruption=corruption)
                except Exception as exc:  # pragma: no cover
                    print(f"[sensitivity] WARNING: loader unavailable for {corruption}: {exc}")
                    loader = None
            except Exception as exc:  # pragma: no cover
                print(f"[sensitivity] WARNING: loader unavailable for {corruption}: {exc}")
                loader = None
        if loader is None:
            per_corruption[corruption] = {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
            continue

        # fresh prompt / CMA / shifting state per corruption (Algorithm 1 is online)
        runner = build_foa(cfg=cfg, source_stats=stats, model=model, device=device)
        if class_subset is not None and hasattr(runner, "class_subset"):
            runner.class_subset = list(class_subset)

        acc = _make_accumulator(ece_bins, class_subset)
        n_samples = 0
        n_batches = 0
        import torch as _torch

        for batch in loader:
            images, targets = unpack_batch(batch)
            if images is None:
                continue
            with _torch.no_grad():
                out = runner.adapt_batch(images, targets if targets is not None else None)
            logits = out.get("logits") if isinstance(out, dict) else out
            if logits is None:
                continue
            logits_np = logits.detach().float().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
            n_samples += int(logits_np.shape[0])
            n_batches += 1
            if targets is not None:
                tgt_np = targets.detach().cpu().numpy() if hasattr(targets, "detach") else np.asarray(targets)
                acc.update(logits_np, tgt_np)
            if limit_batches is not None and n_batches >= int(limit_batches):
                break

        summary = acc.compute()
        summary["num_batches"] = n_batches
        per_corruption[corruption] = summary
        if not math.isnan(summary.get("accuracy", float("nan"))):
            accs.append(float(summary["accuracy"]))
            eces.append(float(summary.get("ece", float("nan"))))
        if verbose:
            print(
                f"    [{corruption:<18}] acc={summary.get('accuracy', float('nan')):6.2f} "
                f"ece={summary.get('ece', float('nan')):6.2f} n={summary.get('num_samples', 0)}"
            )
        del runner

    total_wall = time.time() - t_start
    return {
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "ece": float(np.nanmean(eces)) if eces else float("nan"),
        "num_corruptions": len(corruptions),
        "num_samples": int(sum(v.get("num_samples", 0) for v in per_corruption.values())),
        "wall_clock_s": total_wall,
        "per_corruption": per_corruption,
    }


# ============================================================================
# source statistics sweep helper
# ============================================================================

def _ensure_source_stats(
    cfg: Any,
    num_samples: int,
    device,
    model=None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Return a path to a source-statistics checkpoint for ``num_samples`` (Q).

    Q = 32 (the paper default) is loaded from ``source_stats.path`` when it
    exists; other values are computed on the fly with the frozen backbone and
    cached under ``<checkpoint dir>/source_stats_vit_base_q{Q}.pt``.
    """
    import torch  # noqa: F401

    base_path = _cfg_get(cfg, "source_stats.path", default=os.path.join("checkpoints", "source_stats_vit_base.pt"))
    if int(num_samples) == int(DEFAULT_NUM_SOURCE_SAMPLES) and os.path.isfile(str(base_path)):
        return str(base_path)

    directory = cache_dir or os.path.dirname(str(base_path)) or "checkpoints"
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, f"source_stats_vit_base_q{int(num_samples)}.pt")
    if os.path.isfile(target):
        return target

    if verbose:
        print(f"[sensitivity] computing source statistics for Q={num_samples} -> {target}")

    seed = int(_cfg_get(cfg, "source_stats.seed", default=_cfg_get(cfg, "seed", default=0)) or 0)
    _seed_everything(seed)

    stream = None
    try:
        from src.data.datasets import build_source_stream  # type: ignore

        stream = build_source_stream(
            cfg=cfg,
            num_samples=int(num_samples),
            seed=seed,
            device=device,
        )
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f"[sensitivity] WARNING: source stream unavailable ({exc}); reusing default Q path")
        return str(base_path)

    if model is None:
        model = build_vit(
            model_name=_cfg_get(cfg, "model.name", default="vit_base_patch16_224"),
            checkpoint=_cfg_get(cfg, "model.checkpoint", default=None),
            pretrained=bool(_cfg_get(cfg, "model.pretrained", default=True)),
            num_classes=int(_cfg_get(cfg, "model.num_classes", default=1000)),
            device=str(device),
        )
    stats = compute_source_statistics(
        model,
        stream,
        num_samples=int(num_samples),
        device=str(device),
        progress=verbose,
    )
    save_source_stats(stats, target)
    return target


# ============================================================================
# generic sweep driver
# ============================================================================

def _sweep_values(spec: Dict[str, Any]) -> List[Any]:
    for key in ("values", "batch_sizes", "batch_size", "population_sizes", "num_prompts", "quantiles"):
        if key in spec and spec[key] is not None:
            return _as_list(spec[key])
    return []


def _sweep_overrides(spec: Dict[str, Any], value: Any) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    base = spec.get("overrides") or {}
    if isinstance(base, dict):
        overrides.update(base)
    per_value = spec.get("per_value_overrides") or spec.get("value_overrides") or {}
    if isinstance(per_value, dict):
        overrides.update(per_value.get(str(value), per_value.get(value, {})) or {})
    return overrides


def run_sweep(
    cfg: Any,
    name: str,
    spec: Dict[str, Any],
    *,
    corruptions: Optional[Sequence[str]] = None,
    device=None,
    limit_batches: Optional[int] = None,
    model=None,
    source_stats_path: Optional[str] = None,
    verbose: bool = True,
    plot: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one named sweep and return ``{param, values: {...}, reference: {...}}``."""
    name = GROUP_ALIASES.get(name, name)
    values = _sweep_values(spec)
    param = spec.get("param") or {
        "population_size": "cma.population_size",
        "num_prompts": "prompt.num_prompts",
        "source_samples": "source_stats.num_samples",
        "lambda": "fitness.lambda_base",
        "shifting_ema": "shifting.alpha",
        "discrepancy_ema": "fitness.beta",
    }.get(name)

    corruptions = list(corruptions) if corruptions else _resolve_corruptions(cfg)
    seed = int(_cfg_get(cfg, "runner.seed", default=_cfg_get(cfg, "seed", default=0)) or 0)

    if verbose:
        print(f"\n=== sweep '{name}' (param={param}) over {corruptions if len(corruptions) <= 3 else f'{len(corruptions)} corruptions'} ===")

    results: Dict[str, Any] = {}
    reference: Dict[str, Any] = spec.get("reference") or {}

    # -- Table 14 style sweep: one entry per batch size, EMA on/off ----------
    if name == "shifting_ema":
        modes = spec.get("modes") or {"ema": {"shifting.use_ema": True}, "no_ema": {"shifting.use_ema": False}}
        for bs in values:
            bs = int(bs)
            per_mode: Dict[str, Any] = {}
            for mode_name, mode_overrides in modes.items():
                sub_cfg = copy.deepcopy(cfg)
                _cfg_set(sub_cfg, "data.batch_size", bs)
                # isolate the shifting contribution (Table 14 protocol)
                _cfg_set(sub_cfg, "fitness.use_entropy", False)
                _cfg_set(sub_cfg, "fitness.use_discrepancy", False)
                for k, v in (mode_overrides or {}).items():
                    _cfg_set(sub_cfg, k, v)
                for k, v in _sweep_overrides(spec, bs).items():
                    _cfg_set(sub_cfg, k, v)
                if verbose:
                    print(f"  BS={bs} mode={mode_name}")
                out = _run_one_setting(
                    sub_cfg,
                    corruptions=corruptions,
                    device=device,
                    limit_batches=limit_batches,
                    model=model,
                    source_stats_path=source_stats_path,
                    verbose=verbose,
                    seed=seed,
                )
                out["batch_size"] = bs
                out["mode"] = mode_name
                per_mode[mode_name] = out
            entry: Dict[str, Any] = {"batch_size": bs, "modes": per_mode}
            entry["accuracy"] = per_mode.get("ema", {}).get("accuracy")
            entry["ece"] = per_mode.get("ema", {}).get("ece")
            if str(bs) in reference:
                entry["reference"] = reference[str(bs)]
            elif bs in reference:  # type: ignore[operator]
                entry["reference"] = reference[bs]  # type: ignore[index]
            elif bs in TABLE14_REFERENCE:
                entry["reference"] = TABLE14_REFERENCE[bs]
            results[str(bs)] = entry
        return {"sweep": name, "param": param, "corruptions": corruptions, "values": results,
                "reference": reference or TABLE14_REFERENCE}

    # -- Table 15 style sweep: beta over several datasets --------------------
    if name == "discrepancy_ema":
        datasets = _as_list(spec.get("datasets") or [spec.get("dataset") or _cfg_get(cfg, "data.dataset", default="imagenet-c")])
        for ds in datasets:
            ds_name = str(ds)
            per_ds: Dict[str, Any] = {}
            ds_overrides = {}
            if isinstance(spec.get("dataset_overrides"), dict):
                ds_overrides = spec["dataset_overrides"].get(ds_name, {}) or {}
            for beta in values:
                sub_cfg = copy.deepcopy(cfg)
                _cfg_set(sub_cfg, "data.dataset", ds_name)
                if param:
                    _cfg_set(sub_cfg, str(param), beta)
                for k, v in ds_overrides.items():
                    _cfg_set(sub_cfg, k, v)
                for k, v in _sweep_overrides(spec, beta).items():
                    _cfg_set(sub_cfg, k, v)
                if verbose:
                    print(f"  dataset={ds_name} beta={beta}")
                out = _run_one_setting(
                    sub_cfg,
                    corruptions=_resolve_corruptions(sub_cfg),
                    device=device,
                    limit_batches=limit_batches,
                    model=model,
                    source_stats_path=source_stats_path,
                    verbose=verbose,
                    seed=seed,
                )
                out["beta"] = beta
                out["dataset"] = ds_name
                per_ds[str(beta)] = out
            ref = reference.get(ds_name) if isinstance(reference, dict) else None
            if not ref and ds_name in TABLE15_REFERENCE:
                ref = TABLE15_REFERENCE[ds_name]
            results[ds_name] = {"values": per_ds, "reference": ref or {}}
        return {"sweep": name, "param": param, "corruptions": corruptions, "values": results,
                "reference": reference or TABLE15_REFERENCE}

    # -- generic scalar sweep ------------------------------------------------
    pop_rule = spec.get("population_size_rule")
    for value in values:
        sub_cfg = copy.deepcopy(cfg)
        if param:
            _cfg_set(sub_cfg, str(param), value)
        for k, v in _sweep_overrides(spec, value).items():
            _cfg_set(sub_cfg, k, v)

        stats_path = source_stats_path
        if name == "source_samples":
            num_q = int(value)
            if stats_path is None:
                try:
                    stats_path = _ensure_source_stats(sub_cfg, num_q, device, model=model, verbose=verbose)
                except Exception as exc:  # pragma: no cover
                    print(f"[sensitivity] WARNING: Q={num_q} stats failed ({exc}); using default path")
                    stats_path = _cfg_get(sub_cfg, "source_stats.path", default=None)

        if name == "num_prompts":
            # K must follow Hansen's rule for the *new* prompt dimension.
            embed_dim = int(_cfg_get(sub_cfg, "model.embed_dim", default=0) or 0)
            if not embed_dim:
                embed_dim = 768
            dim = embed_dim * max(int(value), 1)
            rule = str(pop_rule or "ceil")
            k_value = _population_size_rule(dim, rule)
            if isinstance(k_value, float) and rule != "exact":
                k_value = int(k_value)
            if rule == "exact":
                k_value = int(round(float(k_value)))
            if verbose:
                print(f"  N_p={value} -> prompt dim {dim}, K={k_value}")
            _cfg_set(sub_cfg, "cma.population_size", int(k_value))

        if verbose:
            print(f"  {param}={value}")
        out = _run_one_setting(
            sub_cfg,
            corruptions=corruptions,
            device=device,
            limit_batches=limit_batches,
            model=model,
            source_stats_path=stats_path,
            verbose=verbose,
            seed=seed,
        )
        out[str(param)] = value
        ref = None
        if isinstance(reference, dict):
            ref = reference.get(str(value), reference.get(value))
        if ref is None:
            if name == "population_size":
                ref = TABLE13_REFERENCE.get(int(value)) if _is_number(value) else None
            elif name == "num_prompts":
                ref = FIGURE2B_REFERENCE.get(int(value)) if _is_number(value) else None
            elif name == "source_samples":
                ref = FIGURE2C_REFERENCE.get(int(value)) if _is_number(value) else None
            elif name == "lambda":
                ref = LAMBDA_REFERENCE.get(_num(value))
        if ref is not None:
            out["reference_accuracy"] = ref
        results[str(value)] = out

    return {
        "sweep": name,
        "param": param,
        "corruptions": corruptions,
        "values": results,
        "reference": reference,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: Any) -> Any:
    try:
        return float(value)
    except Exception:
        return value


def run_sensitivity(
    cfg: Any,
    groups: Optional[Sequence[str]] = None,
    *,
    corruptions: Optional[Sequence[str]] = None,
    device=None,
    limit_batches: Optional[int] = None,
    model=None,
    source_stats_path: Optional[str] = None,
    verbose: bool = True,
    plot: bool = False,
) -> Dict[str, Any]:
    """Run every requested sweep group and return the aggregated results."""
    if groups is None:
        groups = _cfg_get(cfg, "groups", default=None) or list(SWEEP_GROUPS)
    groups = [GROUP_ALIASES.get(str(g), str(g)) for g in groups]

    specs = _cfg_get(cfg, "sensitivity", default={}) or {}
    if hasattr(specs, "to_dict"):
        specs = specs.to_dict()
    if not isinstance(specs, dict):
        specs = {k: getattr(specs, k) for k in dir(specs) if not k.startswith("_")}

    runner_cfg = _cfg_get(cfg, "runner", default={}) or {}
    corruptions = list(corruptions) if corruptions else None
    if corruptions is None:
        cfg_corr = _cfg_get(runner_cfg, "corruption", default=None)
        if cfg_corr:
            corruptions = [str(cfg_corr)]
        else:
            subgroups = _cfg_get(runner_cfg, "corruptions", default=None)
            corruptions = [str(c) for c in subgroups] if subgroups else None
    severity = _cfg_get(runner_cfg, "severity", default=None)
    batch_size = _cfg_get(runner_cfg, "batch_size", default=None)
    if batch_size:
        _cfg_set(cfg, "data.batch_size", int(batch_size))
    if severity is not None:
        _cfg_set(cfg, "data.severity", int(severity))
    if limit_batches is None:
        limit_batches = _cfg_get(runner_cfg, "limit_batches", default=None)
    if limit_batches is not None:
        try:
            limit_batches = int(limit_batches)
        except Exception:
            limit_batches = None

    all_results: Dict[str, Any] = {}
    for group in groups:
        spec = specs.get(group) if isinstance(specs, dict) else None
        if not spec:
            print(f"[sensitivity] WARNING: no specification for sweep '{group}'; skipping")
            continue
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        try:
            all_results[group] = run_sweep(
                cfg,
                group,
                dict(spec),
                corruptions=corruptions,
                device=device,
                limit_batches=limit_batches,
                model=model,
                source_stats_path=source_stats_path,
                verbose=verbose,
                plot=plot,
            )
        except Exception as exc:  # pragma: no cover
            import traceback

            print(f"[sensitivity] ERROR in sweep '{group}': {exc}")
            traceback.print_exc()
            all_results[group] = {"sweep": group, "error": str(exc), "values": {}}
        _maybe_plot(group, all_results.get(group), cfg)

    return {
        "experiment": _cfg_get(cfg, "experiment.name", default="foa_sensitivity"),
        "groups": groups,
        "results": all_results,
    }


def _maybe_plot(group: str, sweep: Dict[str, Any], cfg: Any) -> None:
    """Best-effort Figure 2 plotting; never fatal."""
    if not _cfg_get(cfg, "runner.plot", default=False) or not sweep or "values" not in sweep:
        return
    try:  # pragma: no cover - plotting is optional
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = _cfg_get(cfg, "output_dir", default="./outputs/sensitivity")
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        xs, ys = [], []
        for key, entry in sweep["values"].items():
            if isinstance(entry, dict) and "accuracy" in entry:
                try:
                    xs.append(float(key))
                    ys.append(float(entry["accuracy"]))
                except Exception:
                    continue
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o")
            ax.set_xlabel(str(sweep.get("param", group)))
            ax.set_ylabel("Accuracy (%)")
            fig.tight_layout()
            path = os.path.join(out_dir, f"sensitivity_{group}.png")
            fig.savefig(path, dpi=150)
            print(f"[sensitivity] wrote figure {path}")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[sensitivity] plotting skipped ({exc})")


# ============================================================================
# reporting
# ============================================================================

def print_sensitivity_summary(results: Dict[str, Any]) -> None:
    if not results:
        return
    print("\n" + "=" * 78)
    print("FOA hyper-parameter sensitivity (paper Figure 2 / Tables 13-15)")
    print("=" * 78)
    for group, sweep in (results.get("results") or {}).items():
        if not isinstance(sweep, dict):
            continue
        if "error" in sweep:
            print(f"\n[{group}] ERROR: {sweep['error']}")
            continue
        param = sweep.get("param")
        print(f"\n[{group}]  param={param}")
        header = f"  {'value':>10} {'accuracy':>10} {'ece':>8} {'paper':>8} {'num':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        values = sweep.get("values") or {}
        if group in ("discrepancy_ema",):
            for ds, block in values.items():
                block_values = (block or {}).get("values", {})
                for key, entry in block_values.items():
                    acc = entry.get("accuracy")
                    ece = entry.get("ece")
                    ref = (block.get("reference") or {}).get(str(key))
                    n = entry.get("num_samples")
                    print(_row(key, acc, ece, ref, n))
        elif group == "shifting_ema":
            for key, entry in values.items():
                for mode, sub in (entry.get("modes") or {}).items():
                    ref = None
                    ref_block = entry.get("reference") or TABLE14_REFERENCE.get(int(key), {})
                    if isinstance(ref_block, dict):
                        ref = ref_block.get("ema" if mode == "ema" else "no_ema")
                    print(_row(f"{key}/{mode}", sub.get("accuracy"), sub.get("ece"), ref, sub.get("num_samples")))
        else:
            for key, entry in values.items():
                if not isinstance(entry, dict):
                    continue
                ref = entry.get("reference_accuracy")
                print(_row(key, entry.get("accuracy"), entry.get("ece"), ref, entry.get("num_samples")))
    print("=" * 78)


def _row(key: Any, acc: Any, ece: Any, ref: Any, n: Any) -> str:
    def fmt(v: Any, width: int = 10) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return f"{'n/a':>{width}}"
        try:
            if width == 8:
                return f"{float(v):>{width}.4g}"
            return f"{float(v):>{width}.2f}"
        except Exception:
            return f"{str(v):>{width}}"

    return f"  {str(key):>10} {fmt(acc)} {fmt(ece, 8)} {fmt(ref, 8)} {fmt(n, 8)}"


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOA sensitivity sweeps (Figure 2, Tables 13-15).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=os.path.join("configs", "sensitivity.yaml"),
                        help="sensitivity YAML config")
    parser.add_argument("--extra-config", default=None,
                        help="optional second YAML merged on top of --config")
    parser.add_argument("--groups", nargs="+", default=None,
                        help=f"subsets of {list(SWEEP_GROUPS)} (default: config 'groups')")
    parser.add_argument("--corruptions", nargs="+", default=None,
                        help="corruptions to evaluate (default: config data.corruptions)")
    parser.add_argument("--dataset", default=None, help="override data.dataset")
    parser.add_argument("--severity", type=int, default=None, help="override data.severity")
    parser.add_argument("--batch-size", type=int, default=None, help="override data.batch_size")
    parser.add_argument("--limit-batches", type=int, default=None, help="cap batches per corruption")
    parser.add_argument("--device", default=None, help="torch device (default: config model.device)")
    parser.add_argument("--seed", type=int, default=None, help="global seed")
    parser.add_argument("--checkpoint", default=None, help="ViT checkpoint path/URL override")
    parser.add_argument("--source-stats", default=None, help="source statistics checkpoint")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--plot", action="store_true", help="write Figure-2 style plots")
    parser.add_argument("--quiet", action="store_true", help="reduce logging")
    return parser.parse_args(argv)


def _apply_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    if args.dataset:
        _cfg_set(cfg, "data.dataset", args.dataset)
    if args.severity is not None:
        _cfg_set(cfg, "data.severity", int(args.severity))
    if args.batch_size is not None:
        _cfg_set(cfg, "data.batch_size", int(args.batch_size))
    if args.checkpoint:
        _cfg_set(cfg, "model.checkpoint", args.checkpoint)
    if args.source_stats:
        _cfg_set(cfg, "source_stats.path", args.source_stats)
    if args.seed is not None:
        _cfg_set(cfg, "seed", int(args.seed))
    if args.plot:
        _cfg_set(cfg, "runner.plot", True)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    paths = [args.config] if args.config else []
    if args.extra_config:
        paths.append(args.extra_config)
    cfg: Any
    try:
        cfg = load_config(*paths) if paths else {}
    except Exception as exc:
        print(f"[sensitivity] WARNING: could not load config ({exc}); continuing with CLI overrides only")
        cfg = {}
    _apply_overrides(cfg, args)

    seed = int(_cfg_get(cfg, "seed", default=0) or 0)
    _seed_everything(seed)
    device = _resolve_device(cfg, args.device)
    if not args.quiet:
        print(f"[sensitivity] config={paths} device={device} seed={seed}")

    groups = args.groups or _cfg_get(cfg, "groups", default=None) or list(SWEEP_GROUPS)
    model = None  # built lazily inside build_foa / _ensure_source_stats

    results = run_sensitivity(
        cfg,
        groups,
        corruptions=args.corruptions,
        device=device,
        limit_batches=args.limit_batches,
        model=model,
        source_stats_path=args.source_stats,
        verbose=not args.quiet,
        plot=bool(args.plot),
    )

    print_sensitivity_summary(results)

    output_dir = _cfg_get(cfg, "runner.output_json", default=None) or _cfg_get(cfg, "output_dir", default="./outputs/sensitivity")
    out_path = args.output
    if out_path is None:
        if str(output_dir).endswith(".json"):
            out_path = str(output_dir)
        else:
            name = _cfg_get(cfg, "experiment.name", default="foa_sensitivity")
            out_path = os.path.join(str(output_dir), f"{name}_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    payload = {"config": config_to_dict(cfg) if callable(globals().get("config_to_dict")) else cfg,
               "args": vars(args), "results": results}
    try:
        with open(out_path, "w") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        print(f"[sensitivity] wrote {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[sensitivity] WARNING: could not write {out_path}: {exc}")
    return 0


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "to_dict"):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
