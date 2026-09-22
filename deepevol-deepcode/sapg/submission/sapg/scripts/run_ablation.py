"""Figure 6 ablation / sensitivity study driver for SAPG (paper Section 6.3).

This script reproduces the ablation experiments of the SAPG paper:

* leader-follower aggregation (default, Algorithm 1) vs. the *symmetric*
  aggregation ablation (Section 4.2) vs. *no off-policy* movement of data,
* the *high off-policy ratio* variant (no sub-sampling of the follower union,
  Section 4.3), and
* the per-task entropy-coefficient sweep ``sigma in {0, 0.003, 0.005}``
  (Equation 10, Section 4.5),

for the five IsaacGym task groups of Appendix A:
``regrasping``, ``throw``, ``reorientation`` (Allegro-Kuka, Table 2),
``shadow_hand`` (Table 3) and ``allegro_hand`` (Table 4).

Each (variant, task, seed) cell is trained with the same algorithm
configuration as the main run and the final metric is aggregated over 5 seeds
with the paper's shaded-band formula (Section 5.2)::

    band(t) = (2 / sqrt(n)) * sum_i (mean(t) - y_i(t))^2

The script writes ``ablation_summary.json``, ``ablation_curves.json`` and (when
matplotlib is available) ``fig6_ablation.{pdf,png}`` +
``fig6b_entropy_sweep.{pdf,png}`` into the output directory, and prints a
compact table together with a check of the qualitative findings reported in
Section 6.3.

The module is importable without ``torch``/IsaacGym/``matplotlib``: every heavy
import is performed lazily inside the functions that need it, and ``--synthetic``
runs the complete reporting/analysis path on synthetic histories so the driver
can be smoke-tested on a CPU-only machine.
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

# --------------------------------------------------------------------------- #
# repo import path (so ``python scripts/run_ablation.py`` works directly)
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)          # .../sapg
_REPO_ROOT = os.path.dirname(_PKG_ROOT)     # .../  (contains sapg/ package)
for _p in (_REPO_ROOT, _PKG_ROOT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
TASKS: Tuple[str, ...] = (
    "regrasping",
    "throw",
    "reorientation",
    "shadow_hand",
    "allegro_hand",
)

#: default ablation variants reproduced from Figure 6
DEFAULT_VARIANTS: Tuple[str, ...] = (
    "sapg",
    "entropy",
    "high_off_policy_ratio",
    "symmetric",
    "no_off_policy",
)

#: all variants that can be requested, including the PPO reference row
ALL_VARIANTS: Tuple[str, ...] = DEFAULT_VARIANTS + ("ppo",)

#: entropy coefficients swept in Section 6.3 / Section 6.2
DEFAULT_ENTROPY_GRID: Tuple[float, ...] = (0.0, 0.003, 0.005)

#: default number of seeds (Section 5.2 uses 5 seeds for every experiment)
DEFAULT_NUM_SEEDS: int = 5

#: default training budget in collected transitions (Section 5.2)
DEFAULT_MAX_SAMPLES: float = 2.0e10

#: tasks for which the high off-policy ratio variant is expected to hurt
HARSH_HIGH_OFF_POLICY_TASKS: Tuple[str, ...] = ("shadow_hand", "allegro_hand")

#: metric keys searched for in the training history, in priority order
METRIC_KEYS: Tuple[str, ...] = (
    "episode_return",
    "episode_reward",
    "return",
    "reward",
    "success_rate",
    "successes",
)

#: paper's per-task best entropy coefficient (Table 1 / Section 6.2)
BEST_ENTROPY_BY_TASK: Dict[str, float] = {
    "regrasping": 0.0,
    "throw": 0.0,
    "reorientation": 0.005,
    "shadow_hand": 0.0,
    "allegro_hand": 0.0,
}

#: expected final performance levels (Section 5.2 / Table 1, sigma = 0 unless
#: noted) — used only for the synthetic smoke-test mode and as sanity anchors.
PAPER_FINAL_MEANS: Dict[str, float] = {
    "regrasping": 35.7,
    "throw": 23.7,
    "reorientation": 33.2,
    "shadow_hand": 1.17e4,
    "allegro_hand": 1.23e4,
}

#: qualitative multiplications used by the synthetic mode so that the
#: expectation checks can be exercised end-to-end without a simulator
_SYNTHETIC_FACTORS: Dict[str, float] = {
    "sapg": 1.00,
    "entropy": 0.99,
    "high_off_policy_ratio": 0.90,
    "symmetric": 0.55,
    "no_off_policy": 0.75,
    "ppo": 0.35,
}

#: synthetic factors for the entropy sweep, per task and coefficient
_SYNTHETIC_ENTROPY_FACTORS: Dict[str, Dict[float, float]] = {
    "regrasping": {0.0: 1.00, 0.003: 0.97, 0.005: 0.92},
    "throw": {0.0: 1.00, 0.003: 0.96, 0.005: 0.90},
    "reorientation": {0.0: 1.00, 0.003: 1.09, 0.005: 1.165},
    "shadow_hand": {0.0: 1.00, 0.003: 0.96, 0.005: 0.88},
    "allegro_hand": {0.0: 1.00, 0.003: 0.94, 0.005: 0.83},
}

#: fallback copy of ``configs/ablations.yaml`` (used when PyYAML or the file is
#: unavailable) — mirrors the shipped YAML definition.
DEFAULT_ABLATION_CONFIG: Dict[str, Any] = {
    "base": {
        "num_envs": 24576,
        "num_policies": 6,
        "leader_index": 1,
        "aggregation": "leader_follower",
        "off_policy_weight": 1.0,
        "subsample_off_policy": True,
        "entropy_coefficient": 0.0,
        "critic_coefficient": 4.0,
        "num_seeds": DEFAULT_NUM_SEEDS,
        "max_samples": DEFAULT_MAX_SAMPLES,
    },
    "ablations": {
        "sapg": {
            "aggregation": "leader_follower",
            "subsample_off_policy": True,
            "off_policy_weight": 1.0,
            "entropy_coefficient": 0.0,
            "description": "default leader-follower SAPG (Algorithm 1)",
        },
        "entropy": {
            "aggregation": "leader_follower",
            "subsample_off_policy": True,
            "off_policy_weight": 1.0,
            "entropy_coefficient": [0.0, 0.003, 0.005],
            "entropy_applies_to": "followers",
            "per_block_sigma": True,
            "description": "sigma * (j-1) * H(pi_j) entropy bonus for followers",
        },
        "high_off_policy_ratio": {
            "aggregation": "leader_follower",
            "subsample_off_policy": False,
            "off_policy_weight": 1.0,
            "entropy_coefficient": 0.0,
            "description": "leader consumes the full follower union (no subsampling)",
        },
        "symmetric": {
            "aggregation": "symmetric",
            "subsample_off_policy": True,
            "off_policy_weight": 1.0,
            "entropy_coefficient": 0.0,
            "symmetric_pairs": "all",
            "description": "every policy aggregates from all others (Section 4.2)",
        },
        "no_off_policy": {
            "aggregation": "none",
            "subsample_off_policy": True,
            "off_policy_weight": 0.0,
            "entropy_coefficient": 0.0,
            "description": "independent block policies, no data fusion",
        },
        "ppo": {
            "method": "ppo",
            "aggregation": "none",
            "num_policies": 1,
            "phi_dim": 0,
            "off_policy_weight": 0.0,
            "subsample_off_policy": True,
            "entropy_coefficient": 0.0,
            "description": "vanilla PPO reference (Section 5.2)",
        },
    },
    "lambda_sweep": {"values": [0.0, 0.5, 1.0, 2.0], "field": "off_policy_weight"},
    "num_policies_sweep": {"values": [1, 2, 3, 6, 12], "field": "num_policies"},
    "best_entropy_coefficient": dict(BEST_ENTROPY_BY_TASK),
}


# --------------------------------------------------------------------------- #
# small generic helpers
# --------------------------------------------------------------------------- #
def parse_float_list(spec: Optional[str]) -> Optional[List[float]]:
    """Parse ``"0,0.003,0.005"`` into ``[0.0, 0.003, 0.005]``."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return [float(v) for v in spec]
    text = str(spec).strip()
    if not text:
        return None
    out: List[float] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    return out


def parse_str_list(spec: Optional[str], default: Sequence[str]) -> List[str]:
    """Parse a comma separated name list, falling back to ``default``."""
    if spec is None:
        return list(default)
    if isinstance(spec, (list, tuple)):
        return [str(v).strip() for v in spec]
    text = str(spec).strip()
    if not text:
        return list(default)
    return [c.strip() for c in text.replace(";", ",").split(",") if c.strip()]


def _as_float(value: Any, default: float = float("nan")) -> float:
    """Best-effort conversion of scalars / tensors / 1-element containers."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    for attr in ("item", "detach"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return float(fn().item()) if attr == "detach" else float(fn())
            except Exception:  # pragma: no cover - defensive
                pass
    try:
        seq = list(value)
    except TypeError:
        return default
    if not seq:
        return default
    return _as_float(seq[-1], default)


def _metric_of(history_entry: Any) -> Optional[float]:
    """Extract the paper metric from one logged history row."""
    if history_entry is None:
        return None
    for key in METRIC_KEYS:
        if isinstance(history_entry, dict) and key in history_entry:
            value = _as_float(history_entry.get(key))
            if value == value:  # not NaN
                return value
        elif hasattr(history_entry, key):
            value = _as_float(getattr(history_entry, key))
            if value == value:
                return value
    return None


def _samples_of(history_entry: Any) -> Optional[float]:
    if isinstance(history_entry, dict):
        for key in ("samples", "total_samples", "num_samples", "env_steps"):
            if key in history_entry:
                return _as_float(history_entry.get(key))
    for key in ("samples", "total_samples", "num_samples"):
        if hasattr(history_entry, key):
            return _as_float(getattr(history_entry, key))
    return None


def _curve(history: Sequence[Any], metric: Optional[str] = None) -> Tuple[List[float], List[float]]:
    """Return ``(samples, values)`` extracted from a training history."""
    samples: List[float] = []
    values: List[float] = []
    for idx, row in enumerate(history or []):
        value = None
        if metric is not None and isinstance(row, dict) and metric in row:
            value = _as_float(row.get(metric))
        if value is None or value != value:
            value = _metric_of(row)
        if value is None or value != value:
            continue
        sample = _samples_of(row)
        if sample is None or sample != sample:
            sample = float(idx + 1)
        samples.append(float(sample))
        values.append(float(value))
    return samples, values


def _interp(xs: Sequence[float], ys: Sequence[float], grid: Sequence[float]) -> List[float]:
    """Linear interpolation of ``ys(xs)`` onto ``grid`` (clamped at the edges)."""
    if not xs or not ys:
        return [float("nan")] * len(grid)
    if len(xs) == 1:
        return [float(ys[0])] * len(grid)
    out: List[float] = []
    j = 0
    n = len(xs)
    for g in grid:
        while j < n - 2 and xs[j + 1] < g:
            j += 1
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if x1 <= x0:
            out.append(float(y1))
            continue
        t = (g - x0) / (x1 - x0)
        t = min(max(t, 0.0), 1.0)
        out.append(float(y0 + t * (y1 - y0)))
    return out


def paper_standard_error(curves: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Paper shaded band: ``(2/sqrt(n)) * sum_i (mean - y_i)^2`` (Section 5.2)."""
    curves = [list(c) for c in curves if c]
    if not curves:
        return [], []
    n = min(len(c) for c in curves)
    curves = [c[:n] for c in curves]
    count = len(curves)
    mean = [sum(c[t] for c in curves) / count for t in range(n)]
    band: List[float] = []
    for t in range(n):
        acc = sum((mean[t] - c[t]) ** 2 for c in curves)
        band.append((2.0 / math.sqrt(max(count, 1))) * acc)
    return mean, band


def aggregate_seed_curves(
    histories: Sequence[Sequence[Any]],
    metric: Optional[str] = None,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate per-seed histories into ``samples/mean/band/seed_curves``."""
    extracted = [_curve(h, metric) for h in histories]
    extracted = [(s, v) for s, v in extracted if v]
    if not extracted:
        return {"samples": [], "mean": [], "band": [], "seed_curves": [], "final": float("nan")}

    if num_bins is None:
        num_bins = min(len(v) for _, v in extracted)
    num_bins = max(int(num_bins), 1)

    max_sample = max(s[-1] for s, _ in extracted)
    grid = [max_sample * (i + 1) / num_bins for i in range(num_bins)]
    resampled = [_interp(s, v, grid) for s, v in extracted]
    mean, band = paper_standard_error(resampled)
    finals = [v[-1] for _, v in extracted]
    return {
        "samples": grid,
        "mean": mean,
        "band": band,
        "seed_curves": resampled,
        "seed_finals": finals,
        "final": mean[-1] if mean else float("nan"),
        "final_std": (sum((f - sum(finals) / len(finals)) ** 2 for f in finals) / len(finals)) ** 0.5
        if finals
        else float("nan"),
    }


# --------------------------------------------------------------------------- #
# configuration plumbing
# --------------------------------------------------------------------------- #
def _candidate_ablation_paths(path: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            os.path.join(_PKG_ROOT, "configs", "ablations.yaml"),
            os.path.join(os.getcwd(), "configs", "ablations.yaml"),
            os.path.join(os.getcwd(), "sapg", "configs", "ablations.yaml"),
        ]
    )
    return candidates


def load_ablation_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``configs/ablations.yaml`` (falling back to the built-in copy)."""
    for candidate in _candidate_ablation_paths(path):
        if candidate and os.path.isfile(candidate):
            try:
                import yaml  # local import: optional dependency
            except Exception:
                break
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            except Exception:
                continue
            merged = copy.deepcopy(DEFAULT_ABLATION_CONFIG)
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
            merged.setdefault("ablations", {})
            for name, patch in DEFAULT_ABLATION_CONFIG["ablations"].items():
                merged["ablations"].setdefault(name, dict(patch))
            return merged
    return copy.deepcopy(DEFAULT_ABLATION_CONFIG)


def _config_fields() -> Optional[set]:
    """Field names of :class:`sapg.utils.config.SAPGConfig` (if importable)."""
    try:
        from dataclasses import fields  # noqa: WPS433

        from sapg.utils.config import SAPGConfig  # noqa: WPS433

        return {f.name for f in fields(SAPGConfig)}
    except Exception:
        return None


def _replace_config(config: Any, patch: Dict[str, Any]) -> Any:
    """Return a copy of ``config`` with ``patch`` applied (dataclass aware)."""
    patch = dict(patch or {})
    if config is None:
        return patch
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            merged = to_dict()
            merged.update(patch)
            from sapg.utils.config import SAPGConfig  # noqa: WPS433

            return SAPGConfig.from_dict(merged)
        except Exception:
            pass
    try:
        from dataclasses import replace as _replace  # noqa: WPS433

        return _replace(config, **patch)
    except Exception:
        out = copy.deepcopy(config)
        for key, value in patch.items():
            try:
                setattr(out, key, value)
            except Exception:  # pragma: no cover - defensive
                pass
        return out


def build_variant_config(
    config: Any,
    variant: str,
    ablation: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    entropy_coefficient: Optional[float] = None,
) -> Any:
    """Apply the ablation patch for ``variant`` on top of a task config."""
    variant = str(variant).strip().lower()
    base = dict(ablation.get("base", {}) or {})
    patches = dict(ablation.get("ablations", {}) or {})

    if variant not in patches:
        if variant == "ppo":
            patch = dict(DEFAULT_ABLATION_CONFIG["ablations"]["ppo"])
        else:
            raise ValueError(
                "unknown ablation variant {!r}; available: {}".format(
                    variant, ", ".join(sorted(patches) + ["ppo"])
                )
            )
    else:
        patch = dict(patches[variant])

    combined = dict(base)
    combined.update(patch)

    # never leak reporting-only keys into the dataclass
    combined.pop("description", None)
    combined.pop("entropy_applies_to", None)
    combined.pop("symmetric_pairs", None)
    combined.pop("field", None)

    if variant == "ppo":
        combined["method"] = "ppo"
    if entropy_coefficient is not None:
        combined["entropy_coefficient"] = float(entropy_coefficient)
        combined["per_block_sigma"] = bool(combined.get("per_block_sigma", True))
    if overrides:
        combined.update(overrides)

    fields = _config_fields()
    if fields is not None:
        combined = {k: v for k, v in combined.items() if k in fields}
    return _replace_config(config, combined)


def resolve_task_config(
    task: str,
    base_config: Any = None,
    ablation: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Any:
    """Build a fresh :class:`SAPGConfig` for ``task`` (task group aware)."""
    if base_config is not None:
        return _replace_config(base_config, overrides)
    from sapg.utils.config import build_config  # noqa: WPS433

    return build_config(task, **overrides)


def load_base_config(config_path: Optional[str], task: str, ablation: Dict[str, Any]) -> Any:
    """Load the task YAML (if provided) or build the config from the registry."""
    from sapg.utils.config import SAPGConfig, build_config  # noqa: WPS433

    config = None
    if config_path and os.path.isfile(config_path):
        config = SAPGConfig.from_yaml(config_path)
        try:
            if str(getattr(config, "task", task)).lower() != str(task).lower():
                config = build_config(task, **config.to_dict())
        except Exception:
            pass
    if config is None:
        config = build_config(task)

    base = dict(ablation.get("base", {}) or {})
    base.pop("description", None)
    fields = _config_fields()
    if fields is not None:
        base = {k: v for k, v in base.items() if k in fields}
    if base:
        config = _replace_config(config, base)
    return config


# --------------------------------------------------------------------------- #
# environment / policy / logger factories (all optional-dependency aware)
# --------------------------------------------------------------------------- #
def make_env_for_config(config: Any, task: Optional[str] = None, **overrides: Any) -> Any:
    """Instantiate the vectorised environment for ``task``."""
    from sapg.envs import make_env  # noqa: WPS433

    task = task or getattr(config, "task", "regrasping")
    num_envs = getattr(config, "num_envs", None)
    try:
        return make_env(task, config=config, num_envs=num_envs, **overrides)
    except TypeError:
        return make_env(task, config=config, **overrides)


def make_policy_for_config(config: Any, num_policies: Optional[int] = None) -> Any:
    """Build the φ-conditioned shared-backbone actor/critic policy."""
    from sapg.models.actor import ActorCritic  # noqa: WPS433

    num_policies = int(num_policies if num_policies is not None else getattr(config, "num_policies", 1) or 1)
    kwargs = {
        "obs_dim": getattr(config, "obs_dim", None),
        "action_dim": getattr(config, "action_dim", None),
        "phi_dim": int(getattr(config, "phi_dim", 0) or 0),
        "num_policies": num_policies,
        "config": config,
    }
    if getattr(config, "random_phi", False):
        kwargs["learnable_phi"] = False
    try:
        return ActorCritic(**kwargs)
    except TypeError:
        kwargs.pop("config", None)
        return ActorCritic(**kwargs)


def make_logger_for_config(config: Any, verbose: bool = False) -> Any:
    """Create the experiment logger (console + optional TensorBoard)."""
    try:
        from sapg.utils.logging import make_logger  # noqa: WPS433

        try:
            return make_logger(config, verbose=verbose)
        except TypeError:
            return make_logger(config=config, verbose=verbose)
    except Exception:
        return None


def set_seed(seed: int, env: Any = None) -> int:
    """Seed python / numpy / torch (best effort)."""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np  # noqa: WPS433

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        try:
            seed_fn(seed)
        except Exception:
            pass
    return seed


# --------------------------------------------------------------------------- #
# training one (variant, task, seed) cell
# --------------------------------------------------------------------------- #
def build_trainer_config(config: Any, variant: str, seed: int, log_dir: Optional[str] = None) -> Any:
    patch: Dict[str, Any] = {"seed": int(seed)}
    if log_dir:
        patch["log_dir"] = os.path.join(log_dir, variant, str(seed))
    return _replace_config(config, patch)


def train_cell(
    variant: str,
    task: str,
    seed: int,
    config: Any,
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    log_dir: Optional[str] = None,
    env: Any = None,
    policy: Any = None,
) -> Dict[str, Any]:
    """Train a single (variant, task, seed) cell and return its result dict."""
    cfg = build_trainer_config(config, variant, seed, log_dir)
    set_seed(seed, env)

    env = env if env is not None else make_env_for_config(cfg, task)
    policy = policy if policy is not None else make_policy_for_config(cfg)
    logger = make_logger_for_config(cfg, verbose=verbose)

    method = str(getattr(cfg, "method", "sapg") or "sapg").lower()
    started = time.time()
    trainer: Any = None
    history: List[Dict[str, float]] = []

    if method in ("ppo", "vanilla_ppo", "ppo_baseline"):
        from sapg.baselines.ppo_baseline import train_ppo_baseline  # noqa: WPS433

        trainer, history = train_ppo_baseline(
            cfg,
            env=env,
            policy=policy,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            logger=logger,
        )
    else:
        from sapg.algorithms.sapg import train_sapg  # noqa: WPS433

        trainer, history = train_sapg(
            cfg,
            env=env,
            policy=policy,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            logger=logger,
        )

    return {
        "variant": variant,
        "task": task,
        "seed": int(seed),
        "method": method,
        "history": list(history or []),
        "trainer": trainer,
        "config": cfg,
        "wall_time": time.time() - started,
    }


def run_variant_seeds(
    variant: str,
    task: str,
    config: Any,
    seeds: Sequence[int],
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    log_dir: Optional[str] = None,
    trainer_factory: Optional[Any] = None,
    metric: Optional[str] = None,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``variant`` on ``task`` across ``seeds`` and aggregate the curves."""
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        if trainer_factory is not None:
            results.append(trainer_factory(variant=variant, task=task, seed=seed, config=config))
        else:
            results.append(
                train_cell(
                    variant,
                    task,
                    seed,
                    config,
                    num_iterations=num_iterations,
                    max_samples=max_samples,
                    verbose=verbose,
                    log_dir=log_dir,
                )
            )
    histories = [r.get("history", []) for r in results]
    agg = aggregate_seed_curves(histories, metric=metric, num_bins=num_bins)
    agg.update(
        {
            "variant": variant,
            "task": task,
            "seeds": [int(s) for s in seeds],
            "method": results[0].get("method") if results else None,
            "results": results,
        }
    )
    return agg


# --------------------------------------------------------------------------- #
# synthetic mode (no simulator / no training)
# --------------------------------------------------------------------------- #
def synthetic_history(
    variant: str,
    task: str,
    seed: int,
    num_points: int = 20,
    max_samples: float = 1.0e9,
    entropy_coefficient: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Generate a deterministic, plausible history for reporting smoke-tests."""
    rng = random.Random(hash((variant, task, seed)) & 0xFFFFFFFF)
    base = PAPER_FINAL_MEANS.get(task, 1.0)
    if variant == "entropy":
        factors = _SYNTHETIC_ENTROPY_FACTORS.get(task, {0.0: 1.0})
        sigma = 0.0 if entropy_coefficient is None else float(entropy_coefficient)
        nearest = min(factors, key=lambda k: abs(k - sigma))
        factor = factors[nearest]
    else:
        factor = _SYNTHETIC_FACTORS.get(variant, 1.0)
    asymptote = base * factor * (1.0 + 0.03 * (seed - 2) / 2.0)
    history: List[Dict[str, float]] = []
    for i in range(num_points):
        progress = (i + 1) / num_points
        value = asymptote * (1.0 - math.exp(-3.0 * progress)) * (1.0 + rng.uniform(-0.02, 0.02))
        history.append(
            {
                "samples": max_samples * progress,
                "episode_return": value,
                "episode_reward": value,
                "iteration": i + 1,
            }
        )
    return history


def synthetic_variant_results(
    variant: str,
    task: str,
    seeds: Sequence[int],
    metric: Optional[str] = None,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    histories = [synthetic_history(variant, task, s) for s in seeds]
    agg = aggregate_seed_curves(histories, metric=metric, num_bins=num_bins)
    agg.update({"variant": variant, "task": task, "seeds": [int(s) for s in seeds], "results": []})
    return agg


# --------------------------------------------------------------------------- #
# the ablation study itself
# --------------------------------------------------------------------------- #
def run_ablation(
    tasks: Sequence[str] = TASKS,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    seeds: Sequence[int] = tuple(range(DEFAULT_NUM_SEEDS)),
    ablation: Optional[Dict[str, Any]] = None,
    base_config: Any = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    log_dir: Optional[str] = None,
    metric: Optional[str] = None,
    num_bins: Optional[int] = None,
    synthetic: bool = False,
    trainer_factory: Optional[Any] = None,
    entropy_grid: Optional[Sequence[float]] = None,
    run_entropy_sweep: bool = True,
) -> Dict[str, Any]:
    """Run the Figure-6 ablations and (optionally) the entropy sweep."""
    ablation = ablation or load_ablation_config()

    summary: Dict[str, Any] = {
        "tasks": list(tasks),
        "variants": list(variants),
        "seeds": [int(s) for s in seeds],
        "metric": metric or METRIC_KEYS[0],
        "synthetic": bool(synthetic),
        "variants_summary": {},
        "entropy_sweep": {},
    }

    for task in tasks:
        task_config = resolve_task_config(task, base_config=base_config, ablation=ablation)
        summary["variants_summary"][task] = {}
        for variant in variants:
            variant_config = build_variant_config(task_config, variant, ablation)
            if synthetic:
                agg = synthetic_variant_results(variant, task, seeds, metric=metric, num_bins=num_bins)
            else:
                agg = run_variant_seeds(
                    variant,
                    task,
                    variant_config,
                    seeds,
                    num_iterations=num_iterations,
                    max_samples=max_samples,
                    verbose=verbose,
                    log_dir=log_dir,
                    trainer_factory=trainer_factory,
                    metric=metric,
                    num_bins=num_bins,
                )
            summary["variants_summary"][task][variant] = {
                "final": agg.get("final"),
                "final_std": agg.get("final_std"),
                "mean": agg.get("mean"),
                "band": agg.get("band"),
                "samples": agg.get("samples"),
                "seed_finals": agg.get("seed_finals"),
                "seeds": agg.get("seeds"),
            }
            if verbose or synthetic:
                print(
                    "[ablation] task={:<13} variant={:<22} final={:>12.4f}".format(
                        task, variant, _as_float(agg.get("final"))
                    )
                )

    if run_entropy_sweep:
        grid = list(entropy_grid) if entropy_grid else list(DEFAULT_ENTROPY_GRID)
        for task in tasks:
            task_config = resolve_task_config(task, base_config=base_config, ablation=ablation)
            summary["entropy_sweep"][task] = {}
            for sigma in grid:
                name = "entropy_sigma_{}".format(sigma)
                variant_config = build_variant_config(
                    task_config, "entropy", ablation, entropy_coefficient=sigma
                )
                if synthetic:
                    histories = [
                        synthetic_history("entropy", task, s, entropy_coefficient=sigma) for s in seeds
                    ]
                    agg = aggregate_seed_curves(histories, metric=metric, num_bins=num_bins)
                    agg.update({"variant": name, "task": task, "seeds": [int(s) for s in seeds]})
                else:
                    agg = run_variant_seeds(
                        name,
                        task,
                        variant_config,
                        seeds,
                        num_iterations=num_iterations,
                        max_samples=max_samples,
                        verbose=verbose,
                        log_dir=log_dir,
                        trainer_factory=trainer_factory,
                        metric=metric,
                        num_bins=num_bins,
                    )
                summary["entropy_sweep"][task][float(sigma)] = {
                    "final": agg.get("final"),
                    "final_std": agg.get("final_std"),
                    "mean": agg.get("mean"),
                    "band": agg.get("band"),
                    "samples": agg.get("samples"),
                }
            finals = {s: _as_float(v.get("final")) for s, v in summary["entropy_sweep"][task].items()}
            best_sigma = max(finals.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -float("inf")))[0]
            summary["entropy_sweep"][task]["best_sigma"] = best_sigma
            baseline = finals.get(0.0, float("nan"))
            best = finals.get(best_sigma, float("nan"))
            summary["entropy_sweep"][task]["best_improvement_pct"] = (
                100.0 * (best - baseline) / abs(baseline) if baseline and baseline == baseline and baseline != 0 else 0.0
            )

    summary["expected_findings"] = evaluate_expectations(summary, ablation, tasks)
    return summary


def evaluate_expectations(
    summary: Dict[str, Any],
    ablation: Optional[Dict[str, Any]] = None,
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Check the qualitative Figure-6 findings reported in Section 6.3."""
    ablation = ablation or {}
    tasks = list(tasks or summary.get("tasks", TASKS))
    variants_summary = summary.get("variants_summary", {}) or {}

    def final(task: str, variant: str) -> float:
        entry = (variants_summary.get(task) or {}).get(variant) or {}
        return _as_float(entry.get("final"))

    checks: Dict[str, Any] = {"symmetric_worse": {}, "no_off_policy_worse": {}, "high_off_ratio": {}}

    for task in tasks:
        ref = final(task, "sapg")
        sym = final(task, "symmetric")
        noff = final(task, "no_off_policy")
        high = final(task, "high_off_policy_ratio")
        if ref == ref:
            if sym == sym:
                checks["symmetric_worse"][task] = bool(sym < ref)
            if noff == noff:
                checks["no_off_policy_worse"][task] = bool(noff < ref)
            if high == high and task in HARSH_HIGH_OFF_POLICY_TASKS:
                checks["high_off_ratio"][task] = bool(high < ref)

    checks["symmetric_worse_everywhere"] = bool(checks["symmetric_worse"]) and all(
        checks["symmetric_worse"].values()
    )
    checks["no_off_policy_worse_everywhere"] = bool(checks["no_off_policy_worse"]) and all(
        checks["no_off_policy_worse"].values()
    )
    checks["high_off_ratio_worse_on_hand_tasks"] = bool(checks["high_off_ratio"]) and all(
        checks["high_off_ratio"].values()
    )

    # entropy coefficient ranking versus the per-task best from the paper
    expected = dict(ablation.get("best_entropy_coefficient") or BEST_ENTROPY_BY_TASK)
    entropy_match: Dict[str, Any] = {}
    for task, entry in (summary.get("entropy_sweep") or {}).items():
        if not isinstance(entry, dict) or "best_sigma" not in entry:
            continue
        best = float(entry["best_sigma"])
        want = float(expected.get(task, BEST_ENTROPY_BY_TASK.get(task, 0.0)))
        entropy_match[task] = {
            "best_sigma": best,
            "expected": want,
            "match": bool(abs(best - want) < 1e-9),
            "improvement_pct": entry.get("best_improvement_pct"),
        }
    checks["entropy_best_per_task"] = entropy_match
    checks["entropy_ranking_matches_paper"] = bool(entropy_match) and all(
        v["match"] for v in entropy_match.values()
    )

    checks["all_checks_pass"] = bool(
        checks["symmetric_worse_everywhere"]
        and checks["no_off_policy_worse_everywhere"]
        and checks.get("high_off_ratio_worse_on_hand_tasks", True)
    )
    return checks


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_summary(summary: Dict[str, Any]) -> None:
    """Print the ablation table, the entropy sweep and the expectation checks."""
    tasks = list(summary.get("tasks", []))
    variants = list(summary.get("variants", []))
    variants_summary = summary.get("variants_summary", {})

    header = ["task"] + [v[:18] for v in variants]
    rows: List[List[str]] = []
    for task in tasks:
        row = [task]
        for variant in variants:
            entry = (variants_summary.get(task) or {}).get(variant) or {}
            value = _as_float(entry.get("final"))
            row.append("{:.4g}".format(value) if value == value else "n/a")
        rows.append(row)

    try:
        from sapg.utils.logging import format_table  # noqa: WPS433

        print(format_table([header] + rows))
    except Exception:
        widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
        for r in [header] + rows:
            print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))

    sweep = summary.get("entropy_sweep") or {}
    if sweep:
        print("\nEntropy coefficient sweep (sigma -> final metric):")
        sigmas = sorted({s for t in sweep for s in sweep[t] if isinstance(s, float)})
        head = ["task"] + ["sigma={}".format(s) for s in sigmas] + ["best", "improv.%"]
        rows2: List[List[str]] = []
        for task, entry in sweep.items():
            if not isinstance(entry, dict):
                continue
            row = [task]
            for s in sigmas:
                value = _as_float((entry.get(s) or {}).get("final"))
                row.append("{:.4g}".format(value) if value == value else "n/a")
            row.append(str(entry.get("best_sigma", "-")))
            row.append("{:.1f}".format(_as_float(entry.get("best_improvement_pct"), 0.0)))
            rows2.append(row)
        try:
            from sapg.utils.logging import format_table  # noqa: WPS433

            print(format_table([head] + rows2))
        except Exception:
            for r in [head] + rows2:
                print("  ".join(str(c) for c in r))

    checks = summary.get("expected_findings") or {}
    if checks:
        print("\nQualitative findings (Section 6.3):")
        print("  symmetric worse everywhere        : {}".format(checks.get("symmetric_worse_everywhere")))
        print("  no off-policy worse everywhere    : {}".format(checks.get("no_off_policy_worse_everywhere")))
        print(
            "  high off-policy ratio worse (hand): {}".format(
                checks.get("high_off_ratio_worse_on_hand_tasks")
            )
        )
        print("  entropy ranking matches paper     : {}".format(checks.get("entropy_ranking_matches_paper")))
        for task, entry in (checks.get("entropy_best_per_task") or {}).items():
            print(
                "    - {:<13} best sigma={} (paper {}) improv={:+.1f}%".format(
                    task,
                    entry.get("best_sigma"),
                    entry.get("expected"),
                    _as_float(entry.get("improvement_pct"), 0.0),
                )
            )


def _matplotlib():
    try:
        import matplotlib  # noqa: WPS433

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: WPS433

        return plt
    except Exception:
        return None


def plot_ablation_bars(summary: Dict[str, Any], path: Optional[str] = None, show: bool = False) -> Any:
    """Render the Figure-6 style grouped bar chart of variant finals."""
    plt = _matplotlib()
    if plt is None:
        return None

    tasks = list(summary.get("tasks", []))
    variants = list(summary.get("variants", []))
    variants_summary = summary.get("variants_summary", {})
    if not tasks or not variants:
        return None

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(tasks)), 4.2))
    width = 0.8 / max(len(variants), 1)
    import numpy as _np  # noqa: WPS433

    xs = _np.arange(len(tasks))
    for k, variant in enumerate(variants):
        values = []
        for task in tasks:
            entry = (variants_summary.get(task) or {}).get(variant) or {}
            value = _as_float(entry.get("final"))
            values.append(value if value == value else 0.0)
        ax.bar(xs + k * width, values, width=width, label=variant)
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel(summary.get("metric", "episode_return"))
    ax.set_title("SAPG ablations (Section 6.3 / Figure 6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_entropy_sweep(summary: Dict[str, Any], path: Optional[str] = None, show: bool = False) -> Any:
    """Render the entropy-coefficient sweep (Section 6.2 / Equation 10)."""
    plt = _matplotlib()
    if plt is None:
        return None

    sweep = summary.get("entropy_sweep") or {}
    if not sweep:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for task, entry in sweep.items():
        if not isinstance(entry, dict):
            continue
        sigmas = sorted(s for s in entry if isinstance(s, float))
        values = [max(_as_float((entry.get(s) or {}).get("final")), 1e-12) for s in sigmas]
        if not sigmas:
            continue
        ax.plot(sigmas, values, marker="o", label=task)
    ax.set_yscale("log")
    ax.set_xlabel("entropy coefficient sigma")
    ax.set_ylabel(summary.get("metric", "episode_return") + " (log scale)")
    ax.set_title("Per-task entropy coefficient sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def write_outputs(summary: Dict[str, Any], output_dir: str, make_plots: bool = True) -> Dict[str, str]:
    """Write the JSON payloads (and figures when matplotlib is available)."""
    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, str] = {}

    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    written["summary"] = summary_path

    curves = {
        "tasks": summary.get("tasks"),
        "variants": summary.get("variants"),
        "curves": {
            task: {
                variant: {
                    "samples": entry.get("samples"),
                    "mean": entry.get("mean"),
                    "band": entry.get("band"),
                }
                for variant, entry in (summary.get("variants_summary", {}).get(task) or {}).items()
            }
            for task in summary.get("tasks", [])
        },
        "entropy_sweep": summary.get("entropy_sweep", {}),
    }
    curves_path = os.path.join(output_dir, "ablation_curves.json")
    with open(curves_path, "w", encoding="utf-8") as handle:
        json.dump(curves, handle, indent=2, default=str)
    written["curves"] = curves_path

    if make_plots:
        for name, fn in (
            ("fig6_ablation.png", plot_ablation_bars),
            ("fig6b_entropy_sweep.png", plot_entropy_sweep),
        ):
            try:
                fig = fn(summary, path=os.path.join(output_dir, name))
            except Exception:
                fig = None
            if fig is not None:
                written[name] = os.path.join(output_dir, name)
                try:
                    import matplotlib.pyplot as plt  # noqa: WPS433

                    plt.close(fig)
                except Exception:
                    pass
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SAPG Section-6.3 ablations (Figure 6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", default=",".join(TASKS), help="comma separated task list")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="comma separated ablation variants ({})".format(", ".join(ALL_VARIANTS)),
    )
    parser.add_argument("--ablation-config", default=None, help="path to configs/ablations.yaml")
    parser.add_argument("--base-config", default=None, help="path to a task YAML config")
    parser.add_argument("--num-envs", type=int, default=None, help="override N (default 24576)")
    parser.add_argument("--num-policies", type=int, default=None, help="override M (default 6)")
    parser.add_argument("--seeds", default=None, help="comma separated seeds (default 0..4)")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS, help="seeds 0..n-1")
    parser.add_argument("--max-samples", type=float, default=None, help="transition budget per run")
    parser.add_argument("--iterations", type=int, default=None, help="outer iterations per run")
    parser.add_argument("--entropy-grid", default=None, help="comma separated sigma sweep values")
    parser.add_argument("--no-entropy-sweep", action="store_true", help="skip the sigma sweep")
    parser.add_argument("--metric", default=None, help="history key to report on")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--output-dir", default="runs/ablations", help="output directory")
    parser.add_argument("--synthetic", action="store_true", help="smoke test without training")
    parser.add_argument("--no-plot", action="store_true", help="disable figure rendering")
    parser.add_argument("--verbose", action="store_true", help="print per-cell progress")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    tasks = parse_str_list(args.tasks, TASKS)
    variants = parse_str_list(args.variants, DEFAULT_VARIANTS)
    seeds = parse_float_list(args.seeds)
    if seeds is None:
        seeds = list(range(int(args.num_seeds)))
    seeds = [int(s) for s in seeds]

    overrides: Dict[str, Any] = {}
    if args.num_envs:
        overrides["num_envs"] = int(args.num_envs)
    if args.num_policies:
        overrides["num_policies"] = int(args.num_policies)
    if args.device:
        overrides["device"] = str(args.device)
    if args.max_samples:
        overrides["max_samples"] = float(args.max_samples)

    ablation = load_ablation_config(args.ablation_config)
    if overrides:
        ablation.setdefault("base", {}).update(overrides)

    base_config = None
    if args.base_config and not args.synthetic:
        try:
            base_config = load_base_config(args.base_config, tasks[0], ablation)
        except Exception as exc:  # pragma: no cover - defensive
            print("[warn] could not load base config: {}".format(exc))
            base_config = None

    print(
        "[ablation] tasks={} variants={} seeds={} synthetic={}".format(
            tasks, variants, seeds, bool(args.synthetic)
        )
    )

    summary = run_ablation(
        tasks=tasks,
        variants=variants,
        seeds=seeds,
        ablation=ablation,
        base_config=base_config,
        num_iterations=args.iterations,
        max_samples=args.max_samples,
        verbose=args.verbose,
        log_dir=args.output_dir,
        metric=args.metric,
        synthetic=bool(args.synthetic),
        entropy_grid=parse_float_list(args.entropy_grid),
        run_entropy_sweep=not args.no_entropy_sweep,
    )

    print_summary(summary)
    written = write_outputs(summary, args.output_dir, make_plots=not args.no_plot)
    for name, path in written.items():
        print("[ablation] wrote {} -> {}".format(name, path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
