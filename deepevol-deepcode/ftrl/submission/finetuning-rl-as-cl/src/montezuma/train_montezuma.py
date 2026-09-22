"""Montezuma's Revenge pipeline orchestration (Section 3, Figure 3b / 6 / 17--19).

This module is the *integration* driver of the Montezuma track of

    "Fine-tuning Reinforcement Learning Models is Secretly a
     Forgetting Mitigation Problem" (Wolczyk et al., 2024).

It composes the already-implemented stages into the full two-stage pipeline of
Section 3 / Appendix B.2:

1. ``M1`` -- PPO + Random Network Distillation (``src/montezuma/ppo_rnd.py``,
   ``src/montezuma/m1_train.py``), trained from scratch until the windowed
   episode cumulative reward reaches ``~7000``.
2. ``M2`` -- behavioural cloning of ``M1`` on 500 trajectories collected from
   Room 7 onward (``src/montezuma/m2_bc.py``); ``M2`` is the pre-trained policy
   ``pi_*`` used for fine-tuning.
3. Fine-tuning on the whole game (rooms 1+: CLOSE and FAR) with one of the
   retention settings compared in the paper:

   * ``none``      -- vanilla fine-tuning (forgetting of Room 7+ capabilities),
   * ``bc``        -- actor-only behavioural-cloning retention loss,
   * ``ewc``       -- actor-only Elastic Weight Consolidation penalty,
   * ``scratch``   -- from-scratch baseline which *never* sees BC data.

The pipeline logs the total episode return and the Room-7 success rate every
``room7_every`` steps (5M by default, matching the paper), aggregates results
over at least ``num_seeds`` seeds with 90% confidence intervals, optionally
reproduces the Figure 13 KL-weight sweep, and serialises everything into
``montezuma_summary.json`` for the analysis modules
(``src/analysis/return_distribution.py``, ``src/analysis/plotting.py``).

All heavy imports (torch, sibling trainers) are performed defensively so that
this module can be imported -- and its bookkeeping/aggregation logic unit
tested -- in minimal environments.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_METHODS: Tuple[str, ...] = ("none", "bc", "ewc", "scratch")

METHOD_LABELS: Dict[str, str] = {
    "m1": "M1 (exploration agent)",
    "m2": r"M2 ($\pi_*$, BC pre-trained)",
    "none": "vanilla fine-tuning",
    "bc": "fine-tuning + BC",
    "ewc": "fine-tuning + EWC",
    "scratch": "from scratch",
}

METHOD_COLORS: Dict[str, str] = {
    "m2": "#1f77b4",
    "none": "#d62728",
    "bc": "#2ca02c",
    "ewc": "#ff7f0e",
    "scratch": "#7f7f7f",
    "m1": "#9467bd",
}

# Paper values (Appendix B.2 / Section 3) used for sanity checks.
DEFAULT_M1_TARGET_RETURN = 7000.0
DEFAULT_NUM_TRAJECTORIES = 500
DEFAULT_MIN_ROOM = 7
DEFAULT_ROOM7_EVERY = 5_000_000
DEFAULT_FINETUNE_STEPS = 200_000_000
DEFAULT_EVAL_EPISODES = 100
DEFAULT_CONFIDENCE = 0.90
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
KL_WEIGHT_GRID: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)

# Paper's qualitative expectations, used by ``summarize``/``check_ordering``.
PAPER_REFERENCE: Dict[str, Any] = {
    "m1_return": 7000.0,
    "ordering": ("bc", "ewc", "none"),  # best -> worst fine-tuning variants
    "room7_drops_under": ("none",),
    "room7_stable_for": ("bc", "ewc"),
    "description": (
        "BC separates from vanilla fine-tuning at ~20M steps and reaches a "
        "higher final return; EWC beats from scratch and vanilla but "
        "saturates lower; the Room-7 success rate collapses under vanilla "
        "fine-tuning while remaining stable for BC and EWC."
    ),
}


# ---------------------------------------------------------------------------
# Defensive import helpers
# ---------------------------------------------------------------------------


def _import_module(dotted: str, relative: Optional[str] = None):
    """Import ``dotted`` (e.g. ``src.montezuma.m2_bc``) or return ``None``."""
    try:  # pragma: no cover - depends on the runtime layout
        from importlib import import_module

        return import_module(dotted)
    except Exception:
        if relative:
            try:  # pragma: no cover
                from importlib import import_module

                return import_module(relative, package=__package__)
            except Exception:
                return None
        return None


def _m1_train():
    return _import_module("src.montezuma.m1_train", ".m1_train")


def _m2_bc():
    return _import_module("src.montezuma.m2_bc", ".m2_bc")


def _ppo_rnd():
    return _import_module("src.montezuma.ppo_rnd", ".ppo_rnd")


def _env_module():
    return _import_module("src.montezuma.env", ".env")


def _logger(name: str = "montezuma"):
    mod = _import_module("src.common.logging_utils", "..common.logging_utils")
    if mod is not None and hasattr(mod, "get_logger"):
        try:
            return mod.get_logger(name)
        except Exception:
            pass
    import logging

    return logging.getLogger(name)


def _set_seed(seed: int) -> None:
    mod = _import_module("src.common.seeding", "..common.seeding")
    if mod is not None and hasattr(mod, "set_seed"):
        try:
            mod.set_seed(seed)
            return
        except Exception:
            pass
    import random

    random.seed(seed)


def _ensure_dir(path: str) -> str:
    mod = _import_module("src.common.checkpointing", "..common.checkpointing")
    if mod is not None and hasattr(mod, "ensure_dir"):
        try:
            return mod.ensure_dir(path)
        except Exception:
            pass
    os.makedirs(path, exist_ok=True)
    return path


def _load_config(path: str, overrides: Optional[Sequence[str]] = None):
    mod = _import_module("src.common.config", "..common.config")
    if mod is None or not hasattr(mod, "load_config"):
        raise RuntimeError(
            "src.common.config is unavailable; cannot load configuration "
            f"'{path}'."
        )
    return mod.load_config(path, overrides)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the keywords accepted by ``fn`` (keeps callers tolerant)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call ``fn`` with whichever of ``kwargs`` it actually accepts."""
    return fn(**_filter_kwargs(fn, kwargs))


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Dotted-path lookup tolerant to ``Config``/dict/object containers."""
    if cfg is None:
        return default
    node = cfg
    for part in str(path).split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            if part in node:
                node = node[part]
                continue
            return default
        if hasattr(node, part):
            node = getattr(node, part)
            continue
        return default
    return default if node is None else node


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile (0.90 -> 1.6449) without SciPy."""
    confidence = float(confidence)
    table = {
        0.80: 1.2816,
        0.85: 1.4395,
        0.90: 1.6449,
        0.95: 1.9600,
        0.98: 2.3263,
        0.99: 2.5758,
    }
    key = round(confidence, 2)
    if key in table:
        return table[key]
    # Acklam's rational approximation
    p = 0.5 * (1.0 + confidence)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return float(z)


def summarize(values: Iterable[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean / std / 90% CI half width for a list of per-seed values."""
    vals = [v for v in (_to_float(x) for x in values) if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "half_width": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    half = z_for(confidence) * std / math.sqrt(n) if n > 0 else float("nan")
    return {
        "mean": float(mean),
        "std": float(std),
        "half_width": float(half),
        "lo": float(mean - half),
        "hi": float(mean + half),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MontezumaPipelineConfig:
    """Orchestration-level configuration for the Montezuma pipeline.

    Mirrors the ``montezuma`` section of ``configs/montezuma.yaml`` and the
    paper's Appendix B.2 (Table 2) plus the M1 -> M2 protocol of Section 3.
    """

    name: str = "montezuma"
    env_name: str = "montezuma"
    train_m1: bool = True
    m1_total_steps: int = 1_000_000_000
    m1_target_return: float = DEFAULT_M1_TARGET_RETURN
    m1_checkpoint: Optional[str] = None
    num_trajectories: int = DEFAULT_NUM_TRAJECTORIES
    min_room: int = DEFAULT_MIN_ROOM
    truncate_to_far: bool = True
    bc_epochs: int = 5
    bc_updates: int = 10_000
    bc_batch_size: int = 256
    bc_lr: float = 1e-4
    bc_loss_type: str = "kl"
    kl_weight: float = 1.0
    ewc_coef: float = 1.0
    total_steps: int = DEFAULT_FINETUNE_STEPS
    eval_every: int = DEFAULT_ROOM7_EVERY
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    room7_every: int = DEFAULT_ROOM7_EVERY
    save_every: int = 25_000_000
    methods: Tuple[str, ...] = DEFAULT_METHODS
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    confidence: float = DEFAULT_CONFIDENCE
    sweep_kl_weights: Tuple[float, ...] = KL_WEIGHT_GRID
    output_dir: str = "runs/montezuma"
    device: str = "cpu"
    stub: bool = False
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "MontezumaPipelineConfig":
        from dataclasses import replace

        known = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        known = {k: v for k, v in known.items() if v is not None}
        return replace(self, **known)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, tuple):
                value = list(value)
            out[key] = value
        return out

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "MontezumaPipelineConfig":
        """Build from a loaded YAML config (``configs/montezuma.yaml``)."""
        obj = cls()
        if cfg is not None:
            mapping = {
                "name": ("name", "env.name"),
                "env_name": ("env_name",),
                "train_m1": ("m1.train", "m1.enabled", "pipeline.train_m1"),
                "m1_total_steps": ("m1.total_steps", "m1.num_steps"),
                "m1_target_return": ("m1.target_return",),
                "m1_checkpoint": ("m1.checkpoint", "m1.path", "checkpoint"),
                "num_trajectories": ("bc.num_trajectories", "m2.num_trajectories",
                                     "collect.num_trajectories"),
                "min_room": ("bc.min_room", "m2.min_room"),
                "truncate_to_far": ("bc.truncate_to_far", "m2.truncate_to_far"),
                "bc_epochs": ("bc.epochs",),
                "bc_updates": ("bc.updates",),
                "bc_batch_size": ("bc.batch_size",),
                "bc_lr": ("bc.lr", "bc.learning_rate"),
                "bc_loss_type": ("bc.loss_type",),
                "kl_weight": ("bc.kl_weight", "finetune.kl_weight"),
                "ewc_coef": ("ewc.coef", "retention.ewc.actor_coef"),
                "total_steps": ("finetune.total_steps", "finetune.num_steps",
                                "ppo.total_steps"),
                "eval_every": ("eval.every", "finetune.eval_every"),
                "eval_episodes": ("eval.episodes", "eval.num_episodes"),
                "room7_every": ("eval.room7_every", "finetune.room7_every"),
                "save_every": ("finetune.save_every", "checkpoint.every"),
                "methods": ("methods", "finetune.methods"),
                "seeds": ("seeds", "eval.seeds", "eval.num_seeds"),
                "confidence": ("eval.confidence",),
                "output_dir": ("output_dir", "logging.output_dir", "paths.output_dir"),
                "device": ("device", "compute.device"),
                "stub": ("stub", "env.stub"),
            }
            for attr, paths in mapping.items():
                for path in paths:
                    value = _cfg_get(cfg, path, None)
                    if value is not None:
                        setattr(obj, attr, value)
                        break
            if isinstance(obj.methods, str):
                obj.methods = tuple(m.strip() for m in obj.methods.split(",") if m.strip())
            elif obj.methods is not None:
                obj.methods = tuple(obj.methods)
            if isinstance(obj.seeds, int):
                obj.seeds = tuple(range(int(obj.seeds)))
            elif obj.seeds is not None:
                obj.seeds = tuple(int(s) for s in obj.seeds)
            obj.extra = {"config": True}
        return obj.with_overrides(**overrides)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MethodRun:
    """Result of one (method, seed) fine-tuning run."""

    method: str
    seed: int
    steps: int = 0
    final_return: float = float("nan")
    best_return: float = float("nan")
    room7_success_rate: float = float("nan")
    max_room: float = float("nan")
    checkpoint: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0
    error: Optional[str] = None

    def as_dict(self, include_history: bool = True) -> Dict[str, Any]:
        out = {
            "method": self.method,
            "seed": self.seed,
            "steps": self.steps,
            "final_return": self.final_return,
            "best_return": self.best_return,
            "room7_success_rate": self.room7_success_rate,
            "max_room": self.max_room,
            "checkpoint": self.checkpoint,
            "elapsed": self.elapsed,
            "error": self.error,
        }
        if include_history:
            out["history"] = list(self.history)
        return out


@dataclass
class PipelineResult:
    """Aggregate result of the full Montezuma pipeline."""

    methods: Dict[str, List[MethodRun]] = field(default_factory=dict)
    aggregates: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    m1: Optional[Dict[str, Any]] = None
    m2: Optional[Dict[str, Any]] = None
    dataset: Optional[Dict[str, Any]] = None
    kl_sweep: Optional[Dict[str, Any]] = None
    ordering: Optional[Dict[str, Any]] = None
    output_dir: str = ""
    summary_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    errors: List[str] = field(default_factory=list)

    def as_dict(self, include_history: bool = True) -> Dict[str, Any]:
        return {
            "methods": {
                m: [r.as_dict(include_history=include_history) for r in runs]
                for m, runs in self.methods.items()
            },
            "aggregates": self.aggregates,
            "m1": self.m1,
            "m2": self.m2,
            "dataset": self.dataset,
            "kl_sweep": self.kl_sweep,
            "ordering": self.ordering,
            "output_dir": self.output_dir,
            "summary_path": self.summary_path,
            "config": self.config,
            "elapsed": self.elapsed,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Generic result/curve extraction (tolerant to sibling return types)
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Convert a result object (dataclass/model) to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    for attr in ("as_dict", "to_dict", "model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                continue
    out: Dict[str, Any] = {}
    for key in (
        "steps", "final_return", "best_return", "room7_success_rate", "max_room",
        "checkpoint", "history", "elapsed", "method", "kl_weight", "final_loss",
        "initial_loss", "dataset_size", "num_trajectories", "episode_return",
        "target_return", "target_reached", "updates", "epochs", "error",
    ):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def _first_key(payload: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def run_to_method_run(method: str, seed: int, result: Any, elapsed: float = 0.0) -> MethodRun:
    """Normalise a sibling fine-tuning result into a :class:`MethodRun`."""
    payload = _as_dict(result)
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    run = MethodRun(
        method=method,
        seed=seed,
        steps=int(payload.get("steps") or 0),
        final_return=_to_float(_first_key(payload, ("final_return", "episode_return",
                                                   "return_mean", "return"))),
        best_return=_to_float(_first_key(payload, ("best_return", "best_episode_return"))),
        room7_success_rate=_to_float(_first_key(payload, ("room7_success_rate",
                                                         "room7_success"))),
        max_room=_to_float(payload.get("max_room")),
        checkpoint=payload.get("checkpoint"),
        history=[dict(h) for h in history if isinstance(h, dict)],
        elapsed=float(payload.get("elapsed") or elapsed),
        error=payload.get("error"),
    )
    if math.isnan(run.final_return) and run.history:
        run.final_return = _extract_curve(run.history, ("return", "episode_return",
                                                        "return_mean"))[-1][1]
    if math.isnan(run.best_return) and run.history:
        vals = [v for _, v in _extract_curve(run.history, ("return", "episode_return",
                                                           "return_mean"))]
        if vals:
            run.best_return = max(vals)
    if math.isnan(run.room7_success_rate) and run.history:
        vals = _extract_curve(run.history, ("room7_success_rate", "room7_success",
                                            "success_rate"))
        if vals:
            run.room7_success_rate = vals[-1][1]
    if run.steps == 0 and run.history:
        run.steps = int(run.history[-1].get("step", 0) or 0)
    return run


def _extract_curve(history: Sequence[Any], keys: Sequence[str],
                   step_keys: Sequence[str] = ("step", "steps", "env_steps")) -> List[Tuple[float, float]]:
    """Extract ``[(step, value), ...]`` from a logged history."""
    out: List[Tuple[float, float]] = []
    for record in history:
        if isinstance(record, dict):
            step = _first_key(record, step_keys, len(out))
            value = _first_key(record, keys, None)
        else:
            step = getattr(record, "step", len(out))
            value = None
            for key in keys:
                if hasattr(record, key):
                    value = getattr(record, key)
                    break
        if value is None:
            continue
        val = _to_float(value)
        if math.isnan(val):
            continue
        out.append((float(step), val))
    return out


def room7_curve(runs: Sequence[MethodRun]) -> List[Tuple[float, float]]:
    """Concatenate the Room-7 success-rate curve of a method's seeds."""
    points: List[Tuple[float, float]] = []
    for run in runs:
        points.extend(_extract_curve(run.history, ("room7_success_rate", "room7_success",
                                                   "success_rate")))
    return sorted(points, key=lambda p: p[0])


# ---------------------------------------------------------------------------
# Stage 1: M1 (PPO + RND exploration agent)
# ---------------------------------------------------------------------------


def train_m1_stage(cfg: MontezumaPipelineConfig, *, seed: Optional[int] = None,
                   output_dir: Optional[str] = None, logger: Any = None,
                   verbose: bool = True) -> Dict[str, Any]:
    """Train ``M1`` from scratch until the episode return reaches ``~7000``."""
    mod = _m1_train()
    if mod is None:
        raise RuntimeError("src.montezuma.m1_train is unavailable; cannot train M1.")
    out_dir = _ensure_dir(output_dir or os.path.join(cfg.output_dir, "m1"))
    seed = cfg.seeds[0] if seed is None else int(seed)
    _set_seed(seed)
    logger = logger or _logger()
    fn = getattr(mod, "train_m1")
    agent, result = _call(
        fn,
        total_steps=cfg.m1_total_steps,
        target_return=cfg.m1_target_return,
        stub=cfg.stub,
        seed=seed,
        device=cfg.device,
        output_dir=out_dir,
        logger=logger,
        eval_every=cfg.room7_every,
        eval_episodes=cfg.eval_episodes,
        save_every=cfg.save_every,
        verbose=verbose,
    )
    payload = _as_dict(result)
    payload.setdefault("steps", 0)
    payload.setdefault("checkpoint", os.path.join(out_dir, "m1.pt"))
    if logger is not None:
        try:
            logger.info("M1 finished after %s steps (return %.1f)",
                        payload.get("steps"), _to_float(payload.get("episode_return")))
        except Exception:
            pass
    return payload


def load_m1(cfg: MontezumaPipelineConfig, checkpoint: Optional[str] = None,
            *, load_optimizer: bool = False) -> Any:
    """Load the ``M1`` agent from a checkpoint (helper for the BC stage)."""
    mod = _m1_train()
    if mod is None or not hasattr(mod, "load_m1_agent"):
        raise RuntimeError("src.montezuma.m1_train.load_m1_agent is unavailable.")
    path = checkpoint or cfg.m1_checkpoint or os.path.join(cfg.output_dir, "m1", "m1.pt")
    return _call(getattr(mod, "load_m1_agent"), path=path, stub=cfg.stub,
                 device=cfg.device, load_optimizer=load_optimizer)


# ---------------------------------------------------------------------------
# Stage 2: M2 (behavioural cloning on Room 7+ trajectories)
# ---------------------------------------------------------------------------


def collect_bc_dataset_stage(cfg: MontezumaPipelineConfig, *, agent: Any = None,
                             checkpoint: Optional[str] = None,
                             seed: Optional[int] = None,
                             num_trajectories: Optional[int] = None,
                             logger: Any = None, progress_fn: Any = None) -> Any:
    """Collect the Room-7-onward BC dataset (500 trajectories by default)."""
    mod = _m2_bc()
    if mod is None:
        raise RuntimeError("src.montezuma.m2_bc is unavailable; cannot collect BC data.")
    n_traj = int(num_trajectories or cfg.num_trajectories)
    seed = cfg.seeds[0] if seed is None else int(seed)
    if agent is not None and hasattr(mod, "build_bc_dataset"):
        return _call(
            getattr(mod, "build_bc_dataset"),
            agent=agent,
            num_trajectories=n_traj,
            min_room=cfg.min_room,
            truncate=cfg.truncate_to_far,
            seed=seed,
            stub=cfg.stub,
            progress_fn=progress_fn,
        )
    if hasattr(mod, "build_bc_dataset_from_m1"):
        dataset, _agent = _call(
            getattr(mod, "build_bc_dataset_from_m1"),
            checkpoint=checkpoint or cfg.m1_checkpoint,
            num_trajectories=n_traj,
            min_room=cfg.min_room,
            truncate=cfg.truncate_to_far,
            seed=seed,
            stub=cfg.stub,
            progress_fn=progress_fn,
        )
        return dataset
    raise RuntimeError("No BC dataset builder found in src.montezuma.m2_bc.")


def pretrain_m2_stage(cfg: MontezumaPipelineConfig, *, agent: Any = None,
                      dataset: Any = None, teacher: Any = None,
                      checkpoint: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      logger: Any = None) -> Tuple[Any, Dict[str, Any]]:
    """Behavioural-cloning pre-training of ``M2`` (= ``pi_*``)."""
    mod = _m2_bc()
    if mod is None or not hasattr(mod, "pretrain_m2"):
        raise RuntimeError("src.montezuma.m2_bc.pretrain_m2 is unavailable.")
    out_dir = _ensure_dir(output_dir or os.path.join(cfg.output_dir, "m2"))
    agent, result = _call(
        getattr(mod, "pretrain_m2"),
        agent=agent,
        dataset=dataset,
        teacher=teacher,
        checkpoint=checkpoint,
        updates=cfg.bc_updates,
        epochs=cfg.bc_epochs,
        batch_size=cfg.bc_batch_size,
        lr=cfg.bc_lr,
        output_dir=out_dir,
        logger=logger,
        verbose=cfg.verbose,
    )
    return agent, _as_dict(result)


# ---------------------------------------------------------------------------
# Stage 3: fine-tuning variants
# ---------------------------------------------------------------------------


def fine_tune_method(cfg: MontezumaPipelineConfig, method: str, seed: int, *,
                     dataset: Any = None, teacher: Any = None,
                     agent: Any = None, output_dir: Optional[str] = None,
                     logger: Any = None, total_steps: Optional[int] = None,
                     **kwargs: Any) -> MethodRun:
    """Fine-tune ``M2`` (or train from scratch) for one method/seed pair."""
    mod = _m2_bc()
    if mod is None:
        raise RuntimeError("src.montezuma.m2_bc is unavailable.")
    method = str(method).lower()
    if method in ("m2", "pistar", "pi_star", "pretrained"):
        # no fine-tuning: evaluate the BC pre-trained policy directly
        payload = {"checkpoint": None, "steps": 0}
        if agent is not None:
            rate = evaluate_room7(agent, cfg, seed=seed)
            payload["room7_success_rate"] = rate
        return run_to_method_run(method, seed, payload)
    seed = int(seed)
    _set_seed(seed)
    out_dir = _ensure_dir(output_dir or os.path.join(cfg.output_dir, method, f"seed_{seed}"))
    started = time.time()
    if method in ("scratch", "from_scratch", "baseline"):
        fn = getattr(mod, "train_from_scratch", None)
        if fn is None:
            raise RuntimeError("src.montezuma.m2_bc.train_from_scratch is unavailable.")
        _agent, result = _call(
            fn,
            total_steps=total_steps or cfg.total_steps,
            output_dir=out_dir,
            logger=logger,
            stub=cfg.stub,
            seed=seed,
            device=cfg.device,
        )
    else:
        fn = getattr(mod, "finetune_m2", None)
        if fn is None:
            raise RuntimeError("src.montezuma.m2_bc.finetune_m2 is unavailable.")
        _agent, result = _call(
            fn,
            agent=agent,
            dataset=dataset,
            teacher=teacher,
            method=method,
            kl_weight=cfg.kl_weight,
            total_steps=total_steps or cfg.total_steps,
            output_dir=out_dir,
            logger=logger,
            stub=cfg.stub,
            seed=seed,
            device=cfg.device,
            **kwargs,
        )
    run = run_to_method_run(method, seed, result, elapsed=time.time() - started)
    if math.isnan(run.room7_success_rate) and _agent is not None:
        try:
            run.room7_success_rate = evaluate_room7(_agent, cfg, seed=seed)
        except Exception:
            pass
    if run.checkpoint is None:
        candidate = os.path.join(out_dir, f"m2_{method}.pt")
        run.checkpoint = candidate if os.path.exists(candidate) else None
    return run


def evaluate_room7(agent: Any, cfg: MontezumaPipelineConfig, *, seed: int = 0,
                   num_episodes: Optional[int] = None, stub: Optional[bool] = None) -> float:
    """Room-7 success rate of a policy (computed every 5M steps in the paper)."""
    mod = _m2_bc()
    episodes = int(num_episodes or cfg.eval_episodes)
    if mod is not None and hasattr(mod, "evaluate_room7_success_rate"):
        try:
            return float(_call(
                getattr(mod, "evaluate_room7_success_rate"),
                agent=agent,
                num_episodes=episodes,
                seed=seed,
                stub=cfg.stub if stub is None else stub,
            ))
        except Exception:
            pass
    env_mod = _env_module()
    if env_mod is not None and hasattr(env_mod, "room7_success_rate"):
        try:
            return float(_call(
                getattr(env_mod, "room7_success_rate"),
                policy=agent,
                num_episodes=episodes,
                seed=seed,
                stub=cfg.stub if stub is None else stub,
            ))
        except Exception:
            pass
    return float("nan")


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------


def aggregate_seeds(runs: Dict[str, Sequence[MethodRun]],
                    confidence: float = DEFAULT_CONFIDENCE,
                    metrics: Sequence[str] = ("final_return", "best_return",
                                              "room7_success_rate")) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-method mean / 90% CI over seeds (paper reports >= 20 seeds)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, method_runs in runs.items():
        out[method] = {}
        for metric in metrics:
            values = [getattr(r, metric, float("nan")) for r in method_runs]
            out[method][metric] = summarize(values, confidence=confidence)
    return out


def check_ordering(aggregates: Dict[str, Dict[str, Dict[str, float]]],
                   reference: Dict[str, Any] = PAPER_REFERENCE) -> Dict[str, Any]:
    """Compare observed method ordering with the paper's qualitative claims."""
    means = {
        m: _to_float(vals.get("final_return", {}).get("mean"))
        for m, vals in (aggregates or {}).items()
    }
    ranked = sorted(
        [m for m in means if not math.isnan(means[m])],
        key=lambda m: means[m],
        reverse=True,
    )
    room7 = {
        m: _to_float(vals.get("room7_success_rate", {}).get("mean"))
        for m, vals in (aggregates or {}).items()
    }
    checks: Dict[str, Any] = {"ranked_returns": ranked, "return_means": means,
                              "room7_means": room7}
    expected_order = [m for m in reference.get("ordering", ()) if m in means]
    checks["matches_paper_ordering"] = bool(
        expected_order
        and len(expected_order) > 1
        and all(
            means.get(expected_order[i], float("-inf"))
            >= means.get(expected_order[i + 1], float("-inf"))
            for i in range(len(expected_order) - 1)
        )
    )
    scratch = means.get("scratch")
    checks["bc_beats_scratch"] = bool(
        scratch is not None and "bc" in means
        and not math.isnan(means["bc"]) and means["bc"] > scratch
    )
    checks["ewc_beats_scratch"] = bool(
        scratch is not None and "ewc" in means
        and not math.isnan(means["ewc"]) and means["ewc"] > scratch
    )
    return checks


def _curve_for_methods(runs: Dict[str, Sequence[MethodRun]],
                       keys: Sequence[str]) -> Dict[str, List[Tuple[float, float]]]:
    curves: Dict[str, List[Tuple[float, float]]] = {}
    for method, method_runs in runs.items():
        points: List[Tuple[float, float]] = []
        for run in method_runs:
            points.extend(_extract_curve(run.history, keys))
        if points:
            curves[method] = sorted(points, key=lambda p: p[0])
    return curves


def plot_pipeline_results(runs: Dict[str, Sequence[MethodRun]], output_dir: str,
                          confidence: float = DEFAULT_CONFIDENCE,
                          make_plots: bool = True) -> List[str]:
    """Render Figure 3b / 6 / 17--19 style curves next to the summary."""
    paths: List[str] = []
    if not make_plots:
        return paths
    plotting = _import_module("src.analysis.plotting", "..analysis.plotting")
    if plotting is None:
        return paths
    _ensure_dir(output_dir)
    try:
        curves = _curve_for_methods(runs, ("return", "episode_return", "return_mean"))
        if curves and hasattr(plotting, "plot_curves"):
            path = os.path.join(output_dir, "returns.png")
            plotting.plot_curves(
                curves,
                confidence=confidence,
                xlabel="environment steps",
                ylabel="episode return",
                title="Montezuma's Revenge fine-tuning",
                path=path,
            )
            paths.append(path)
        room_curves = _curve_for_methods(runs, ("room7_success_rate", "room7_success",
                                                "success_rate"))
        if room_curves and hasattr(plotting, "plot_curves"):
            path = os.path.join(output_dir, "room7_success_rate.png")
            plotting.plot_curves(
                room_curves,
                confidence=confidence,
                xlabel="environment steps",
                ylabel="Room-7 success rate",
                title="Room 7 completion (5M-step evaluation)",
                path=path,
            )
            paths.append(path)
        # Figure 3b style per-method return distribution
        return_dist = _import_module("src.analysis.return_distribution",
                                     "..analysis.return_distribution")
        if return_dist is not None and hasattr(return_dist, "plot_return_distribution"):
            records = []
            for method, method_runs in runs.items():
                values = [r.final_return for r in method_runs
                          if not math.isnan(_to_float(r.final_return))]
                if not values:
                    continue
                record = None
                cls = getattr(return_dist, "ReturnRecord", None)
                if cls is not None:
                    try:
                        record = cls(method=method, seed=0, step=0, returns=list(values))
                    except Exception:
                        record = None
                if record is not None:
                    records.append(record)
            aggregate = getattr(return_dist, "aggregate_methods", None)
            if records and callable(aggregate):
                dists = aggregate(records, confidence=confidence)
                path = os.path.join(output_dir, "return_distribution.png")
                return_dist.plot_return_distribution(dists, path=path,
                                                     confidence=confidence)
                paths.append(path)
    except Exception:
        # Plotting is a convenience, never a failure mode.
        pass
    return paths


def write_summary(result: PipelineResult, path: Optional[str] = None,
                  include_history: bool = True) -> str:
    """Persist the pipeline result as JSON."""
    path = path or result.summary_path or os.path.join(result.output_dir,
                                                       "montezuma_summary.json")
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.as_dict(include_history=include_history), handle,
                  indent=2, default=str)
    result.summary_path = path
    return path


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run_pipeline(cfg: Optional[MontezumaPipelineConfig] = None, *,
                 methods: Optional[Sequence[str]] = None,
                 seeds: Optional[Sequence[int]] = None,
                 train_m1: Optional[bool] = None,
                 m1_checkpoint: Optional[str] = None,
                 num_trajectories: Optional[int] = None,
                 total_steps: Optional[int] = None,
                 bc_updates: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 stub: Optional[bool] = None,
                 eval_episodes: Optional[int] = None,
                 run_kl_sweep: bool = False,
                 make_plots: bool = True,
                 num_seeds: Optional[int] = None,
                 logger: Any = None,
                 progress_fn: Any = None,
                 verbose: bool = True) -> PipelineResult:
    """Run the full M1 -> M2 -> fine-tuning Montezuma pipeline.

    Stages:

    1. ``M1`` (skipped when ``train_m1=False`` and a checkpoint is provided),
    2. BC dataset of 500 Room-7+ trajectories,
    3. ``M2`` (``pi_*``) by behavioural cloning on that dataset,
    4. fine-tuning variants ``{none, bc, ewc}`` plus the ``scratch`` baseline,
    5. aggregation over seeds + ordering checks + optional Figure 13 sweep.
    """
    cfg = cfg or MontezumaPipelineConfig()
    overrides: Dict[str, Any] = {
        "total_steps": total_steps,
        "bc_updates": bc_updates,
        "num_trajectories": num_trajectories,
        "output_dir": output_dir,
        "stub": stub,
        "eval_episodes": eval_episodes,
    }
    cfg = cfg.with_overrides(**{k: v for k, v in overrides.items() if v is not None})
    if methods is not None:
        cfg.methods = tuple(methods)
    if train_m1 is not None:
        cfg.train_m1 = bool(train_m1)
    if m1_checkpoint is not None:
        cfg.m1_checkpoint = m1_checkpoint
    if num_seeds is not None:
        cfg.seeds = tuple(range(int(num_seeds)))
    elif seeds is not None:
        cfg.seeds = tuple(int(s) for s in seeds)
    cfg.verbose = bool(verbose)

    logger = logger or _logger()
    started = time.time()
    _ensure_dir(cfg.output_dir)
    result = PipelineResult(output_dir=cfg.output_dir, config=cfg.to_dict())

    if logger is not None:
        try:
            logger.info("Montezuma pipeline: methods=%s seeds=%s steps=%s",
                        list(cfg.methods), list(cfg.seeds), cfg.total_steps)
        except Exception:
            pass

    # --- stage 1: M1 -----------------------------------------------------
    m1_agent = None
    checkpoint = cfg.m1_checkpoint
    if cfg.train_m1:
        try:
            m1_info = train_m1_stage(cfg, seed=cfg.seeds[0], logger=logger,
                                     verbose=cfg.verbose)
            result.m1 = m1_info
            checkpoint = m1_info.get("checkpoint", checkpoint)
        except Exception as exc:  # pragma: no cover - environment dependent
            message = f"M1 training failed: {exc}"
            result.errors.append(message)
            if logger is not None:
                logger.warning(message)
    if checkpoint and os.path.exists(str(checkpoint)):
        try:
            m1_agent = load_m1(cfg, checkpoint=str(checkpoint))
        except Exception as exc:  # pragma: no cover
            result.errors.append(f"Could not load M1 checkpoint: {exc}")

    # --- stage 2: BC dataset --------------------------------------------
    dataset = None
    try:
        dataset = collect_bc_dataset_stage(
            cfg, agent=m1_agent, checkpoint=checkpoint, seed=cfg.seeds[0],
            logger=logger, progress_fn=progress_fn,
        )
        try:
            size = len(dataset)  # type: ignore[arg-type]
        except Exception:
            size = None
        result.dataset = {
            "num_samples": size,
            "num_trajectories": cfg.num_trajectories,
            "min_room": cfg.min_room,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        message = f"BC dataset collection failed: {exc}"
        result.errors.append(message)
        if logger is not None:
            logger.warning(message)

    # --- stage 3: M2 (pi_*) ---------------------------------------------
    m2_agent = None
    try:
        m2_agent, m2_info = pretrain_m2_stage(
            cfg, agent=m1_agent, dataset=dataset, teacher=m1_agent,
            checkpoint=checkpoint,
            output_dir=os.path.join(cfg.output_dir, "m2"), logger=logger,
        )
        result.m2 = m2_info
        if m1_agent is not None and m2_agent is not None:
            # the BC loss should have decreased relative to the initial value
            initial = _to_float(m2_info.get("initial_loss"))
            final = _to_float(m2_info.get("final_loss"))
            if not math.isnan(initial) and not math.isnan(final):
                result.m2["loss_decreased"] = bool(final <= initial)
    except Exception as exc:  # pragma: no cover - environment dependent
        message = f"M2 BC pre-training failed: {exc}"
        result.errors.append(message)
        if logger is not None:
            logger.warning(message)

    # --- stage 4: fine-tuning variants ----------------------------------
    runs: Dict[str, List[MethodRun]] = {m: [] for m in cfg.methods}
    for method in cfg.methods:
        for seed in cfg.seeds:
            try:
                run = fine_tune_method(
                    cfg, method, seed,
                    dataset=dataset,
                    teacher=m1_agent,
                    agent=m2_agent,
                    output_dir=os.path.join(cfg.output_dir, method, f"seed_{seed}"),
                    logger=logger,
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                message = f"{method} seed {seed} failed: {exc}"
                result.errors.append(message)
                if logger is not None:
                    logger.warning(message)
                run = MethodRun(method=method, seed=int(seed), error=str(exc))
            runs[method].append(run)
            if logger is not None:
                try:
                    logger.info("%s seed %d: return=%.1f room7=%.3f",
                                method, seed, run.final_return, run.room7_success_rate)
                except Exception:
                    pass
            if progress_fn is not None:
                try:
                    progress_fn(method, seed, run)
                except Exception:
                    pass

    result.methods = runs
    result.aggregates = aggregate_seeds(runs, confidence=cfg.confidence)
    result.ordering = check_ordering(result.aggregates)

    # --- stage 5: optional Figure 13 KL-weight sweep --------------------
    if run_kl_sweep:
        try:
            result.kl_sweep = kl_weight_sweep(
                cfg, dataset=dataset, checkpoint=checkpoint, logger=logger,
                stub=cfg.stub,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            result.errors.append(f"KL-weight sweep failed: {exc}")

    result.elapsed = time.time() - started
    result.summary_path = os.path.join(cfg.output_dir, "montezuma_summary.json")
    try:
        write_summary(result)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        result.errors.append(f"Could not write summary: {exc}")
    if make_plots:
        plot_pipeline_results(runs, cfg.output_dir, confidence=cfg.confidence)
    return result


def kl_weight_sweep(cfg: Optional[MontezumaPipelineConfig] = None, *,
                    weights: Optional[Sequence[float]] = None,
                    dataset: Any = None, checkpoint: Optional[str] = None,
                    logger: Any = None, stub: Optional[bool] = None,
                    **kwargs: Any) -> Dict[str, Any]:
    """Reproduce Figure 13: sweep the BC KL weight during M2 fine-tuning.

    The paper does not state the chosen value, so the sweep
    (0.1, 0.5, 1.0, 2.0, 5.0) is run and the best-performing weight reported.
    """
    cfg = cfg or MontezumaPipelineConfig()
    mod = _m2_bc()
    if mod is None or not hasattr(mod, "sweep_kl_weight"):
        return {"error": "src.montezuma.m2_bc.sweep_kl_weight is unavailable"}
    out_dir = _ensure_dir(os.path.join(cfg.output_dir, "kl_sweep"))
    sweep = _call(
        getattr(mod, "sweep_kl_weight"),
        weights=tuple(weights or cfg.sweep_kl_weights),
        dataset=dataset,
        checkpoint=checkpoint or cfg.m1_checkpoint,
        seeds=cfg.seeds[:1],
        total_steps=cfg.total_steps,
        output_dir=out_dir,
        stub=cfg.stub if stub is None else stub,
        logger=logger,
        **kwargs,
    )
    payload = _as_dict(sweep) if not isinstance(sweep, dict) else dict(sweep)
    payload.setdefault("grid", list(weights or cfg.sweep_kl_weights))
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "kl_weight_sweep.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
    except Exception:
        pass
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Montezuma's Revenge M1 -> M2 -> fine-tuning pipeline "
            "(PPO+RND, behavioural cloning, EWC retention)."
        )
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config (default: configs/montezuma.yaml).")
    parser.add_argument("--set", action="append", default=None, dest="overrides",
                        help="Config overrides as key.subkey=value (repeatable).")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma separated retention methods (none,bc,ewc,scratch).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma separated seeds, e.g. '0,1,2'.")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Run seeds 0..num_seeds-1 (paper uses >= 20).")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Fine-tuning environment steps per run.")
    parser.add_argument("--bc-updates", type=int, default=None,
                        help="Number of BC updates for M2 pre-training.")
    parser.add_argument("--num-trajectories", type=int, default=None,
                        help="Room-7+ M1 trajectories for the BC dataset.")
    parser.add_argument("--kl-weight", type=float, default=None,
                        help="BC auxiliary loss weight during fine-tuning.")
    parser.add_argument("--train-m1", action="store_true", default=None,
                        help="Force re-training M1 (default: from config).")
    parser.add_argument("--no-train-m1", action="store_false", dest="train_m1",
                        help="Reuse the M1 checkpoint from the config/output dir.")
    parser.add_argument("--m1-checkpoint", type=str, default=None,
                        help="Path to an existing M1 checkpoint.")
    parser.add_argument("--kl-sweep", action="store_true",
                        help="Also run the Figure 13 KL-weight sweep.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed convenience flag.")
    parser.add_argument("--stub", action="store_true", default=None,
                        help="Use the dependency-free stub environment.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny run: 1 seed, few steps (no paper results).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = None
    config_path = args.config
    if config_path is None:
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "configs", "montezuma.yaml")
        if os.path.exists(candidate):
            config_path = candidate
    if config_path is not None and os.path.exists(config_path):
        try:
            cfg = _load_config(config_path, args.overrides)
        except Exception as exc:  # pragma: no cover - config dependent
            print(f"[warn] could not load config {config_path}: {exc}")
            cfg = None

    pipeline_cfg = MontezumaPipelineConfig.from_config(cfg)
    if args.output_dir:
        pipeline_cfg.output_dir = args.output_dir
    if args.device:
        pipeline_cfg.device = args.device
    if args.kl_weight is not None:
        pipeline_cfg.kl_weight = float(args.kl_weight)
    if args.stub:
        pipeline_cfg.stub = True

    methods: Optional[Sequence[str]] = None
    if args.methods:
        methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    seeds: Optional[Sequence[int]] = None
    if args.seeds:
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    elif args.seed is not None:
        seeds = (int(args.seed),)
    elif args.num_seeds is not None:
        seeds = tuple(range(int(args.num_seeds)))

    total_steps = args.total_steps
    bc_updates = args.bc_updates
    num_seeds = args.num_seeds
    if args.smoke_test:
        pipeline_cfg.stub = True
        methods = methods or ("none", "bc")
        seeds = seeds or (0,)
        total_steps = total_steps or 2_048
        bc_updates = bc_updates or 5
        pipeline_cfg.num_trajectories = args.num_trajectories or 2
        pipeline_cfg.eval_episodes = 2
        pipeline_cfg.eval_every = 1_024
        pipeline_cfg.train_m1 = False

    result = run_pipeline(
        pipeline_cfg,
        methods=methods,
        seeds=seeds,
        num_seeds=num_seeds,
        train_m1=args.train_m1,
        m1_checkpoint=args.m1_checkpoint,
        num_trajectories=args.num_trajectories,
        total_steps=total_steps,
        bc_updates=bc_updates,
        output_dir=args.output_dir,
        stub=True if args.stub else None,
        run_kl_sweep=bool(args.kl_sweep),
        make_plots=not args.no_plots,
        verbose=True,
    )
    print(json.dumps(result.aggregates, indent=2, default=str))
    if result.summary_path:
        print(f"summary written to {result.summary_path}")
    if result.errors:
        for message in result.errors:
            print(f"[warn] {message}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
