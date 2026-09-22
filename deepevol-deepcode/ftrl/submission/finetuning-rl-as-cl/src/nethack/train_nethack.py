"""NetHack Human Monk fine-tuning driver (Sections 3, 4, 5; Table 1; Appendix B.1).

This module is the orchestration entry point for the NetHack track of
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem" (Wolczyk et al., 2024).  It composes:

1.  **pi\* setup** -- the released 30M-step LSTM checkpoint is loaded and (for the
    retention variants) a *baseline head* is optionally pre-trained for 500M env
    steps with the encoders frozen (Appendix B.1).
2.  **Auxiliary-loss preparation** -- the diagonal Fisher of the actor at
    ``theta_*`` computed from 10000 NLD-AA batches (EWC), the 10000-state
    behavioral-cloning buffer built from expert data (BC), and the online
    ``B_theta`` rollout data used by kickstarting.
3.  **APPO fine-tuning** for the five variants
    ``scratch / none (vanilla) / ewc / bc / ks`` with the Table 1
    hyperparameters and the per-method retention coefficients::

        EWC  : coef 2e6           (no decay)
        BC   : coef 2.0, no decay (buffer of 10000 pre-training states)
        KS   : coef 0.5, exponential decay 0.99998 applied every train step

    All retention terms are attached to the **actor only** (critic coefficient is
    always 0) and entropy is disabled whenever a retention method is active.
4.  **Multi-seed aggregation** (>= 20 seeds for the RoboticSequence protocol;
    the NetHack pipeline typically uses 3 seeds) with 90% confidence intervals
    and the paper's qualitative ordering checks (KS/BC > EWC > vanilla, and
    scratch staying low at ~776).
5.  **Per-level evaluation** (levels 4 and Sokoban) via
    :mod:`src.nethack.per_level_eval`, every 25M environment steps, with 200
    AutoAscend saves per level.

The heavy lifting lives in :mod:`src.nethack.appo_runner`; this file owns the
pipeline, the variant wiring, checkpoint bookkeeping and result serialisation
(``nethack_summary.json``).

Usage
-----
::

    python -m src.nethack.train_nethack --method ks --total-steps 500000000
    python -m src.nethack.train_nethack --methods none bc ks --seeds 0 1 2
    python -m src.nethack.train_nethack --smoke-test          # CPU-only dry run
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (Table 1 / Appendix B.1)
# ---------------------------------------------------------------------------

TRAINING_METHODS: Tuple[str, ...] = ("scratch", "none", "ewc", "bc", "ks")

METHOD_ALIASES: Dict[str, str] = {
    "vanilla": "none",
    "finetune": "none",
    "fine-tuning": "none",
    "from_scratch": "scratch",
    "fromscratch": "scratch",
    "from-scratch": "scratch",
    "ewc": "ewc",
    "bc": "bc",
    "behavioral_cloning": "bc",
    "replay": "bc",
    "ks": "ks",
    "kickstarting": "ks",
}

METHOD_LABELS: Dict[str, str] = {
    "scratch": "from scratch",
    "none": "fine-tuning",
    "ewc": "fine-tuning + EWC",
    "bc": "fine-tuning + BC",
    "ks": "fine-tuning + KS",
}

METHOD_COLORS: Dict[str, str] = {
    "scratch": "#7f7f7f",
    "none": "#1f77b4",
    "ewc": "#2ca02c",
    "bc": "#ff7f0e",
    "ks": "#d62728",
}

#: Table 1 (Appendix B.1) APPO hyperparameters.
TABLE1: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "unroll_length": 32,
    "batch_size": 128,
    "num_envs": 32,
    "num_workers": 8,
    "discount": 0.999999,
    "entropy_cost": 0.001,
    "grad_clip": 4.0,
    "reward_clip": 10.0,
    "reward_scale": 1.0,
    "clip_ratio": 0.1,
    "clip_value": 1.0,
    "value_loss_coef": 0.5,
    "gae_lambda": 0.95,
    "num_epochs": 1,
    "max_grad_norm": 4.0,
    "normalize_advantage": True,
    "total_steps": 500_000_000,
}

#: Retention coefficients used by each variant (Table 1 / Appendix B.1).
RETENTION_CONFIG: Dict[str, Dict[str, Any]] = {
    "none": {"enabled": False},
    "ewc": {"enabled": True, "coef": 2e6, "critic_coef": 0.0, "num_batches": 10000, "batch_size": 128},
    "bc": {"enabled": True, "coef": 2.0, "critic_coef": 0.0, "decay": None, "memory_size": 10000, "batch_size": 128},
    "ks": {"enabled": True, "coef": 0.5, "critic_coef": 0.0, "decay": 0.99998, "decay_type": "exponential"},
    "scratch": {"enabled": False},
}

DEFAULT_METHODS: Tuple[str, ...] = ("none", "ewc", "bc", "ks")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_TOTAL_STEPS: int = 500_000_000
DEFAULT_EVAL_EVERY: int = 25_000_000
DEFAULT_EVAL_EPISODES: int = 1000
DEFAULT_SAVE_EVERY: int = 25_000_000
DEFAULT_CONFIDENCE: float = 0.90
BASELINE_PRETRAIN_STEPS: int = 500_000_000
FISHER_NUM_BATCHES: int = 10000
FISHER_BATCH_SIZE: int = 128
BC_BUFFER_SIZE: int = 10000

#: Paper's reported reference scores (Section 5 / Table 4 / Table 5).
PAPER_REFERENCE: Dict[str, Any] = {
    "scratch": 776.0,
    "none": {"far_degradation": True},
    "ewc": 3976.0,
    "bc": 7610.0,
    "ks": {"mean": 10588.0, "std": 672.0},
    "ordering": ("ks", "bc", "ewc", "none"),
    "per_level_ordering": ("ks", "bc", "ewc", "none"),
}

PER_LEVEL_TARGETS: Tuple[str, ...] = ("level_4", "sokoban")


# ---------------------------------------------------------------------------
# Safe attribute helpers (config / dataclass / dict tolerant)
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Dotted-path lookup tolerant to Config/dict/dataclass/object shapes."""
    if cfg is None:
        return default
    node: Any = cfg
    for key in str(path).split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            if key in node:
                node = node[key]
                continue
            return default
        if hasattr(node, key):
            node = getattr(node, key)
            continue
        return default
    return node


def _first(cfg: Any, paths: Sequence[str], default: Any = None) -> Any:
    """Return the first non-``None`` value found along ``paths``."""
    for path in paths:
        value = _cfg_get(cfg, path, None)
        if value is not None:
            return value
    return default


def _import(name: str, attr: Optional[str] = None) -> Any:
    """Import ``name`` (optionally ``name.attr``) returning ``None`` on failure."""
    candidates = [name]
    if name.startswith("src."):
        candidates.append(name[len("src."):])
    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
        except Exception:
            continue
        if attr is None:
            return module
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def _import_runner() -> Any:
    return _import("src.nethack.appo_runner") or _import(".appo_runner".lstrip("."))


def normalize_method(method: Optional[str]) -> str:
    """Map common aliases onto the canonical variant name."""
    if method is None:
        return "none"
    key = str(method).strip().lower().replace(" ", "_")
    if key in TRAINING_METHODS:
        return key
    return METHOD_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Statistics helpers (SciPy-free)
# ---------------------------------------------------------------------------


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided standard-normal quantile for a confidence level."""
    table = {0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    key = round(float(confidence), 4)
    if key in table:
        return table[key]
    p = 0.5 + 0.5 * float(confidence)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def summarize(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean / std / half-width of a normal-approximation confidence interval."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "half_width": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "half_width": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    half = z_for(confidence) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "half_width": half, "n": n}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NetHackPipelineConfig:
    """Orchestration config mirroring ``configs/nethack.yaml`` / Table 1."""

    name: str = "nethack"
    env_name: str = "nethack"
    # --- pi* ---------------------------------------------------------------
    pretrained_checkpoint: Optional[str] = None
    pretrain_baseline_head: bool = True
    baseline_pretrain_steps: int = BASELINE_PRETRAIN_STEPS
    freeze_encoders: bool = True
    # --- dataset -----------------------------------------------------------
    dataset_name: str = "nld-aa-v0"
    dataset_num_batches: int = FISHER_NUM_BATCHES
    dataset_batch_size: int = FISHER_BATCH_SIZE
    bc_buffer_size: int = BC_BUFFER_SIZE
    # --- APPO (Table 1) ----------------------------------------------------
    total_steps: int = DEFAULT_TOTAL_STEPS
    learning_rate: float = 1e-4
    unroll_length: int = 32
    batch_size: int = 128
    num_envs: int = 32
    num_workers: int = 8
    discount: float = 0.999999
    entropy_cost: float = 0.001
    grad_clip: float = 4.0
    reward_clip: float = 10.0
    clip_ratio: float = 0.1
    clip_value: float = 1.0
    value_loss_coef: float = 0.5
    gae_lambda: float = 0.95
    # --- retention ---------------------------------------------------------
    methods: Tuple[str, ...] = DEFAULT_METHODS
    ewc_coef: float = 2e6
    bc_coef: float = 2.0
    ks_coef: float = 0.5
    ks_decay: float = 0.99998
    actor_only: bool = True
    disable_entropy_with_retention: bool = True
    # --- evaluation --------------------------------------------------------
    eval_every: int = DEFAULT_EVAL_EVERY
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    save_every: int = DEFAULT_SAVE_EVERY
    per_level_eval: bool = True
    per_level_targets: Tuple[str, ...] = PER_LEVEL_TARGETS
    per_level_saves: int = 200
    # --- bookkeeping -------------------------------------------------------
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    confidence: float = DEFAULT_CONFIDENCE
    output_dir: str = "results/nethack"
    device: str = "cuda"
    stub: bool = False
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **overrides: Any) -> "NetHackPipelineConfig":
        data = dict(self.__dict__)
        for key, value in overrides.items():
            if value is not None and key in data:
                data[key] = value
        return NetHackPipelineConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["methods"] = list(self.methods)
        out["seeds"] = list(self.seeds)
        out["per_level_targets"] = list(self.per_level_targets)
        return out

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "NetHackPipelineConfig":
        """Build from a ``configs/nethack.yaml``-style object (tolerant)."""
        params: Dict[str, Any] = {}
        if cfg is not None:
            params["pretrained_checkpoint"] = _first(cfg, ["pretrained_checkpoint", "checkpoint", "model.checkpoint"])
            params["dataset_name"] = _cfg_get(cfg, "dataset.name", params.get("dataset_name", "nld-aa-v0"))
            params["dataset_num_batches"] = int(
                _first(cfg, ["dataset.num_batches", "fisher.num_batches", "fisher.num_batches_nethack"],
                       FISHER_NUM_BATCHES))
            params["total_steps"] = int(_first(cfg, ["total_steps", "appo.total_steps", "finetune.total_steps"],
                                                 DEFAULT_TOTAL_STEPS))
            params["learning_rate"] = float(_first(cfg, ["learning_rate", "appo.learning_rate", "appo.lr"],
                                                   TABLE1["learning_rate"]))
            params["unroll_length"] = int(_first(cfg, ["unroll_length", "appo.unroll_length", "appo.rollout_length"],
                                                 TABLE1["unroll_length"]))
            params["batch_size"] = int(_first(cfg, ["batch_size", "appo.batch_size"], TABLE1["batch_size"]))
            params["num_envs"] = int(_first(cfg, ["num_envs", "appo.num_envs", "env.num_envs"], TABLE1["num_envs"]))
            params["discount"] = float(_first(cfg, ["discount", "gamma", "appo.discount"], TABLE1["discount"]))
            params["entropy_cost"] = float(_first(cfg, ["entropy_cost", "appo.entropy_cost"], TABLE1["entropy_cost"]))
            params["grad_clip"] = float(_first(cfg, ["grad_clip", "appo.grad_clip"], TABLE1["grad_clip"]))
            params["reward_clip"] = float(_first(cfg, ["reward_clip", "appo.reward_clip"], TABLE1["reward_clip"]))
            params["clip_ratio"] = float(_first(cfg, ["clip_ratio", "appo.clip_ratio", "appo.ppo_clip"],
                                              TABLE1["clip_ratio"]))
            params["value_loss_coef"] = float(_first(cfg, ["value_loss_coef", "appo.value_loss_coef"],
                                                    TABLE1["value_loss_coef"]))
            params["gae_lambda"] = float(_first(cfg, ["gae_lambda", "lam", "appo.gae_lambda"], TABLE1["gae_lambda"]))
            params["eval_every"] = int(_first(cfg, ["eval_every", "finetune.eval_every"], DEFAULT_EVAL_EVERY))
            params["eval_episodes"] = int(_first(cfg, ["eval_episodes", "eval.episodes"], DEFAULT_EVAL_EPISODES))
            params["save_every"] = int(_first(cfg, ["save_every", "finetune.save_every"], DEFAULT_SAVE_EVERY))
            params["ewc_coef"] = float(_first(cfg, ["retention.ewc.actor_coef", "retention.ewc.coef"], 2e6))
            params["bc_coef"] = float(_first(cfg, ["retention.bc.actor_coef", "retention.bc.coef"], 2.0))
            params["bc_buffer_size"] = int(_first(cfg, ["retention.bc.memory_size", "retention.bc.buffer_size"],
                                                 BC_BUFFER_SIZE))
            params["ks_coef"] = float(_first(cfg, ["retention.ks.actor_coef", "retention.ks.coef"], 0.5))
            params["ks_decay"] = float(_first(cfg, ["retention.ks.decay"], 0.99998))
            params["per_level_eval"] = bool(_first(cfg, ["per_level_eval", "eval.per_level"], True))
            params["per_level_saves"] = int(_first(cfg, ["per_level_saves", "eval.per_level_saves"], 200))
            methods = _first(cfg, ["methods", "training_methods"], None)
            if methods:
                params["methods"] = tuple(normalize_method(m) for m in methods)
            seeds = _first(cfg, ["seeds"], None)
            if seeds:
                params["seeds"] = tuple(int(s) for s in seeds)
            params["output_dir"] = _first(cfg, ["output_dir", "logging.output_dir"], params.get("output_dir"))
            params["device"] = _first(cfg, ["device", "compute.device"], params.get("device"))
        params.update({k: v for k, v in overrides.items() if v is not None})
        base = cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})
        return base


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MethodRun:
    """Outcome of one ``(method, seed)`` fine-tuning run."""

    method: str
    seed: int
    steps: int = 0
    final_score: float = float("nan")
    best_score: float = float("nan")
    turns: float = float("nan")
    dlvl: float = float("nan")
    xplvl: float = float("nan")
    per_level: Dict[str, float] = field(default_factory=dict)
    checkpoint: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0
    error: Optional[str] = None

    def as_dict(self, include_history: bool = True) -> Dict[str, Any]:
        out = {
            "method": self.method,
            "seed": self.seed,
            "steps": self.steps,
            "final_score": self.final_score,
            "best_score": self.best_score,
            "turns": self.turns,
            "dlvl": self.dlvl,
            "xplvl": self.xplvl,
            "per_level": dict(self.per_level),
            "checkpoint": self.checkpoint,
            "elapsed": self.elapsed,
            "error": self.error,
        }
        out["history"] = list(self.history) if include_history else f"<{len(self.history)} records>"
        return out


@dataclass
class PipelineResult:
    """Aggregate result of the whole NetHack pipeline."""

    methods: Tuple[str, ...] = ()
    aggregates: Dict[str, Any] = field(default_factory=dict)
    per_level: Dict[str, Any] = field(default_factory=dict)
    ordering: Dict[str, Any] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    pi_star: Dict[str, Any] = field(default_factory=dict)
    fisher: Dict[str, Any] = field(default_factory=dict)
    bc_dataset: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None
    summary_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    errors: List[str] = field(default_factory=list)

    def as_dict(self, include_history: bool = True) -> Dict[str, Any]:
        return {
            "methods": list(self.methods),
            "aggregates": self.aggregates,
            "per_level": self.per_level,
            "ordering": self.ordering,
            "runs": [r if include_history else {k: v for k, v in r.items() if k != "history"}
                     for r in self.runs],
            "pi_star": self.pi_star,
            "fisher": self.fisher,
            "bc_dataset": self.bc_dataset,
            "output_dir": self.output_dir,
            "summary_path": self.summary_path,
            "config": self.config,
            "elapsed": self.elapsed,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# pi* / Fisher / BC buffer preparation
# ---------------------------------------------------------------------------


def verify_pistar(cfg: NetHackPipelineConfig, checkpoint: Optional[str] = None,
                  *, logger: Optional[Callable[[str], None]] = None,
                  num_episodes: int = 50, seed: int = 0) -> Dict[str, Any]:
    """Load pi* and check that it reproduces the ~5K Human Monk score.

    The paper reports that the released pre-trained checkpoint scores
    approximately 5000 on the Human Monk character ("pi* loads and scores ~5K").
    This function is a *verification* step: it loads the model (or falls back to
    the stub when NLE is unavailable) and evaluates it.
    """
    log = logger or (lambda message: None)
    runner = _import_runner()
    info: Dict[str, Any] = {
        "checkpoint": checkpoint or cfg.pretrained_checkpoint,
        "expected_score": 5000.0,
        "num_episodes": num_episodes,
    }
    if runner is None:
        info.update({"loaded": False, "score": float("nan"), "error": "appo_runner unavailable"})
        return info
    try:
        model = runner.build_nethack_model(cfg, checkpoint=checkpoint or cfg.pretrained_checkpoint)
        env = runner.build_env(cfg, seed=seed, stub=cfg.stub)
        eval_fn = getattr(runner, "evaluate", None)
        if eval_fn is not None:
            result = eval_fn(model, env=env, num_episodes=num_episodes, seed=seed, stub=cfg.stub)
            score = result.get("score", result.get("mean_score", result.get("return_mean")))
            info.update({"loaded": True, "score": score, "eval": result})
        else:
            info.update({"loaded": True, "score": float("nan")})
        log(f"[nethack] pi* verification score={info.get('score')}")
    except Exception as exc:  # pragma: no cover - environment dependent
        info.update({"loaded": False, "score": float("nan"), "error": f"{type(exc).__name__}: {exc}"})
    return info


def pretrain_baseline_head(cfg: NetHackPipelineConfig, model: Any = None, *,
                           steps: Optional[int] = None, logger: Optional[Callable[[str], None]] = None,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Pre-train the baseline (value) head with the rest of the network frozen.

    Appendix B.1: "the baseline head is pre-trained for 500M env steps with the
    rest frozen".  Delegates to :mod:`src.nethack.pretrain_baseline` when
    available; otherwise reports the intended configuration.
    """
    log = logger or (lambda message: None)
    steps = int(steps or cfg.baseline_pretrain_steps)
    module = _import("src.nethack.pretrain_baseline")
    info: Dict[str, Any] = {"steps": steps, "freeze_encoders": True, "freeze_actor": True}
    if module is None:
        info["status"] = "module-unavailable"
        return info
    train_fn = getattr(module, "train_baseline", None) or getattr(module, "pretrain_baseline", None)
    if train_fn is None:
        info["status"] = "no-train-fn"
        return info
    try:
        result = train_fn(model, steps=steps, config=cfg, output_dir=output_dir) \
            if model is not None else train_fn(cfg, steps=steps, output_dir=output_dir)
        info["status"] = "ok"
        info["result"] = result if isinstance(result, (dict, list, str, int, float)) else str(result)
    except Exception as exc:  # pragma: no cover
        info["status"] = "error"
        info["error"] = f"{type(exc).__name__}: {exc}"
    log(f"[nethack] baseline head pre-training: {info['status']} ({steps} steps)")
    return info


def compute_fisher(cfg: NetHackPipelineConfig, model: Any = None, *,
                   num_batches: Optional[int] = None, batch_size: Optional[int] = None,
                   logger: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Diagonal Fisher of the actor at ``theta_*`` from NLD-AA batches.

    Appendix C.1: the Fisher is accumulated over **10000 batches** of expert
    data (batch size 128) using the squared gradients of the expert
    log-likelihood w.r.t. the actor parameters.
    """
    log = logger or (lambda message: None)
    num_batches = int(num_batches or cfg.dataset_num_batches)
    batch_size = int(batch_size or cfg.dataset_batch_size)
    info: Dict[str, Any] = {
        "num_batches": num_batches,
        "batch_size": batch_size,
        "total_samples": num_batches * batch_size,
        "dataset": cfg.dataset_name,
    }
    fisher_cls = _import("src.retention.fisher", "FisherEstimator")
    if fisher_cls is None or model is None:
        info["status"] = "unavailable" if fisher_cls is None else "no-model"
        return info
    try:
        actor = getattr(model, "policy", None) or getattr(model, "actor", None) or model
        estimator = fisher_cls(actor, mode="expert", num_batches=num_batches, batch_size=batch_size)
        dataset_mod = _import("src.nethack.dataset")
        batches: Iterable[Any] = ()
        if dataset_mod is not None:
            loader_fn = getattr(dataset_mod, "fisher_batches", None) or getattr(dataset_mod, "iterate_batches", None)
            if loader_fn is not None:
                batches = loader_fn(num_batches=num_batches, batch_size=batch_size, config=cfg)
        diagonals = estimator.compute(batches, num_batches=num_batches)
        info["status"] = "ok"
        info["num_parameters"] = len(diagonals) if hasattr(diagonals, "__len__") else None
        log(f"[nethack] Fisher computed over {num_batches} batches")
    except Exception as exc:  # pragma: no cover
        info["status"] = "error"
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_bc_buffer(cfg: NetHackPipelineConfig, model: Any = None, *,
                    size: Optional[int] = None,
                    logger: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Build the ``B_BC`` buffer of 10000 pre-training states (Appendix C.2).

    The buffer holds ``(s, pi_*(s))`` pairs drawn from the NLD-AA expert states.
    """
    log = logger or (lambda message: None)
    size = int(size or cfg.bc_buffer_size)
    info: Dict[str, Any] = {"size": size, "source": cfg.dataset_name}
    builder = _import("src.retention.behavioral_cloning", "build_bc_buffer")
    dataset_mod = _import("src.nethack.dataset")
    if dataset_mod is None or model is None:
        info["status"] = "unavailable"
        return info
    try:
        obs_fn = getattr(dataset_mod, "load_states", None) or getattr(dataset_mod, "sample_states", None)
        states = obs_fn(size, config=cfg) if obs_fn is not None else None
        if builder is None or states is None:
            info["status"] = "partial"
            return info
        buffer = builder(states, teacher=model, capacity=size, batch_size=cfg.batch_size)
        info["status"] = "ok"
        info["buffer_size"] = len(buffer) if hasattr(buffer, "__len__") else size
    except Exception as exc:  # pragma: no cover
        info["status"] = "error"
        info["error"] = f"{type(exc).__name__}: {exc}"
    log(f"[nethack] BC buffer: {info['status']}")
    return info


def prepare_retention(cfg: NetHackPipelineConfig, method: str, *,
                      model: Any = None, teacher: Any = None, fisher: Any = None,
                      bc_dataset: Any = None, device: Optional[str] = None,
                      seed: int = 0, logger: Optional[Callable[[str], None]] = None) -> Any:
    """Instantiate the actor-only auxiliary loss for ``method``.

    Coefficients follow Appendix B.1: EWC ``2e6``, BC ``2.0`` with no decay,
    KS ``0.5`` with an exponential decay of ``0.99998`` per train step.
    """
    method = normalize_method(method)
    if method in ("none", "scratch"):
        return None
    runner = _import_runner()
    if runner is None:
        return None
    bundle_cls = getattr(runner, "RetentionBundle", None)
    if bundle_cls is None:  # pragma: no cover - fallback to individual classes
        return None
    kwargs: Dict[str, Any] = {
        "actor": teacher if teacher is not None else model,
        "method": method,
        "device": device or cfg.device,
        "seed": seed,
        "env_name": "nethack",
    }
    if method == "ewc":
        kwargs["fisher"] = fisher
    elif method == "bc":
        kwargs["bc_dataset"] = bc_dataset
    try:
        build = getattr(bundle_cls, "build")
        bundle = build(cfg, **kwargs)
        if cfg.verbose and logger is not None:
            logger(f"[nethack] retention {method}: {bundle.describe()}")
        return bundle
    except Exception as exc:  # pragma: no cover - environment dependent
        if logger is not None:
            logger(f"[nethack] retention {method} unavailable: {type(exc).__name__}: {exc}")
        return None


def prepare_agent_config(cfg: NetHackPipelineConfig, method: str) -> Any:
    """Build the APPO config for a variant (entropy disabled under retention)."""
    runner = _import_runner()
    if runner is None:
        return None
    config_cls = getattr(runner, "APPOConfig", None)
    if config_cls is None:
        return None
    base = None
    from_config = getattr(config_cls, "from_config", None)
    if from_config is not None:
        try:
            base = from_config(cfg, method=normalize_method(method))
            return base
        except Exception:
            base = None
    try:
        base = config_cls()
        with_ov = getattr(base, "with_overrides", None)
        if with_ov is not None:
            base = with_ov(method=normalize_method(method),
                           total_steps=cfg.total_steps,
                           learning_rate=cfg.learning_rate,
                           entropy_cost=(0.0 if (cfg.disable_entropy_with_retention and method in ("ewc", "bc", "ks"))
                                         else cfg.entropy_cost))
    except Exception:
        base = None
    return base


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def extract_scores(result: Any, keys: Sequence[str] = ("score", "mean_score", "return_mean", "episode_return")) -> float:
    """Best-effort extraction of a scalar score from an evaluation payload."""
    if result is None:
        return float("nan")
    if isinstance(result, (int, float)):
        return float(result)
    for key in keys:
        value = _cfg_get(result, key, None) if isinstance(result, (dict,)) or hasattr(result, key) else None
        if value is None and isinstance(result, dict):
            value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float("nan")


def per_level_evaluation(cfg: NetHackPipelineConfig, agent: Any, step: int, *,
                         targets: Optional[Sequence[str]] = None,
                         saves: Optional[int] = None, seed: int = 0,
                         logger: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Per-level evaluation using AutoAscend saves (Section 5).

    The agent is launched from the expert's finishing state at the target level
    and the score achieved *on top of* the expert score is reported.  200 saves
    per level are generated with AutoAscend (jt-nld branch) and evaluation runs
    every 25M environment steps.
    """
    targets = tuple(targets or cfg.per_level_targets)
    saves = int(saves or cfg.per_level_saves)
    module = _import("src.nethack.per_level_eval")
    out: Dict[str, Any] = {"step": int(step), "targets": list(targets), "num_saves": saves}
    if module is None:
        out["status"] = "module-unavailable"
        return out
    try:
        fn = getattr(module, "evaluate_per_level", None) or getattr(module, "run_per_level_eval", None)
        if fn is None:
            out["status"] = "no-eval-fn"
            return out
        out["results"] = fn(agent, targets=targets, num_saves=saves, seed=seed, config=cfg)
        out["status"] = "ok"
    except Exception as exc:  # pragma: no cover - environment dependent
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    if logger is not None:
        logger(f"[nethack] per-level eval @ {step}: {out.get('status')}")
    return out


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_method(cfg: NetHackPipelineConfig, method: str, seed: int, *,
               teacher: Any = None, fisher: Any = None, bc_dataset: Any = None,
               model: Any = None, env: Any = None,
               output_dir: Optional[str] = None,
               logger: Optional[Callable[[str], None]] = None,
               progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
               total_steps: Optional[int] = None,
               checkpoint: Optional[str] = None) -> MethodRun:
    """Fine-tune one variant for one seed and return a :class:`MethodRun`."""
    method = normalize_method(method)
    run = MethodRun(method=method, seed=int(seed))
    runner = _import_runner()
    if runner is None:
        run.error = "appo_runner unavailable"
        return run

    out_dir = os.path.join(output_dir or cfg.output_dir, method, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    retention = prepare_retention(cfg, method, model=model, teacher=teacher,
                                  fisher=fisher, bc_dataset=bc_dataset,
                                  device=cfg.device, seed=seed, logger=logger)
    appo_cfg = prepare_agent_config(cfg, method)

    start = time.time()
    try:
        fn = getattr(runner, "run_finetuning")
        kwargs: Dict[str, Any] = {
            "cfg": appo_cfg if appo_cfg is not None else cfg,
            "method": method,
            "total_steps": int(total_steps or cfg.total_steps),
            "seed": int(seed),
            "device": cfg.device,
            "output_dir": out_dir,
            "checkpoint": checkpoint or cfg.pretrained_checkpoint,
            "teacher": teacher,
            "fisher": fisher,
            "bc_dataset": bc_dataset,
            "stub": cfg.stub,
            "model": model,
            "env": env,
            "logger": logger,
            "progress_fn": _wrap_progress(progress_fn, method, seed),
        }
        if retention is not None:
            kwargs["retention"] = retention
        result = fn(**{k: v for k, v in kwargs.items() if v is not None or k in ("stub", "method")})
        run.steps = int(extract_attr(result, "steps", 0) or 0)
        run.checkpoint = extract_attr(result, "checkpoint", None)
        history = extract_attr(result, "history", None) or []
        run.history = list(history)
        final_eval = extract_attr(result, "final_eval", None)
        run.final_score = extract_scores(final_eval)
        if math.isnan(run.final_score):
            run.final_score = extract_scores(result)
        run.best_score = extract_attr(result, "best_return", run.final_score)
        run.turns = extract_scores(final_eval, ("turns_mean", "turns"))
        run.dlvl = extract_scores(final_eval, ("dlvl_mean", "dlvl"))
        run.xplvl = extract_scores(final_eval, ("xplvl_mean", "xplvl"))
    except Exception as exc:  # pragma: no cover - heavy/GPU path
        run.error = f"{type(exc).__name__}: {exc}"
    run.elapsed = time.time() - start

    if cfg.per_level_eval:
        try:
            per_level = per_level_evaluation(cfg, teacher if teacher is not None else model,
                                             run.steps, seed=seed, logger=logger)
            results = per_level.get("results") if isinstance(per_level, dict) else None
            if isinstance(results, dict):
                for key, value in results.items():
                    run.per_level[str(key)] = extract_scores(value)
        except Exception:  # pragma: no cover
            pass

    _write_json(os.path.join(out_dir, "summary.json"), run.as_dict(include_history=True))
    return run


def _wrap_progress(progress_fn: Optional[Callable[[Dict[str, Any]], None]],
                   method: str, seed: int) -> Optional[Callable[[Dict[str, Any]], None]]:
    if progress_fn is None:
        return None

    def hook(payload: Dict[str, Any]) -> None:
        payload = dict(payload or {})
        payload.setdefault("method", method)
        payload.setdefault("seed", seed)
        progress_fn(payload)

    return hook


def extract_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Fetch ``name`` from a mapping, dataclass field or plain attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    return default


def _write_json(path: str, payload: Any) -> Optional[str]:
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Aggregation / ordering checks
# ---------------------------------------------------------------------------


def aggregate_seeds(runs: Sequence[MethodRun], confidence: float = DEFAULT_CONFIDENCE,
                    metrics: Sequence[str] = ("final_score", "best_score", "turns", "dlvl", "xplvl")) -> Dict[str, Any]:
    """Per-method mean / 90% CI over seeds (Section 5, 1000-episode evaluation)."""
    aggregates: Dict[str, Any] = {}
    for method in TRAINING_METHODS:
        method_runs = [r for r in runs if r.method == method]
        if not method_runs:
            continue
        entry: Dict[str, Any] = {"n_seeds": len(method_runs)}
        for metric in metrics:
            values = [getattr(r, metric, float("nan")) for r in method_runs]
            entry[metric] = summarize(values, confidence)
        per_level_keys = sorted({k for r in method_runs for k in r.per_level})
        for key in per_level_keys:
            values = [r.per_level.get(key, float("nan")) for r in method_runs]
            entry.setdefault("per_level", {})[key] = summarize(values, confidence)
        aggregates[method] = entry
    return aggregates


def check_ordering(aggregates: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the observed method ordering with the paper's claims.

    Section 5 / Table 4: ``KS (10588) > BC (7610) > EWC (3976) > vanilla``, with
    the from-scratch baseline remaining essentially fixed at ~776.
    """
    means = {m: _cfg_get(v, "final_score.mean", float("nan")) for m, v in aggregates.items()}
    result: Dict[str, Any] = {"means": means}
    valid = {m: v for m, v in means.items() if v is not None and not math.isnan(v)}
    if len(valid) >= 2:
        result["ranking"] = tuple(sorted(valid, key=lambda m: valid[m], reverse=True))
    expected = tuple(m for m in PAPER_REFERENCE["ordering"] if m in valid)
    if expected:
        observed = tuple(m for m in result.get("ranking", ()) if m in expected)
        result["expected_ordering"] = expected
        result["observed_ordering"] = observed
        result["matches_ordering"] = observed == expected
    if "ks" in valid:
        ks = PAPER_REFERENCE["ks"]
        result["ks_within_reference"] = abs(valid["ks"] - ks["mean"]) <= 2.0 * ks["std"]
    if "scratch" in valid:
        result["scratch_near_776"] = abs(valid["scratch"] - PAPER_REFERENCE["scratch"]) <= 0.25 * PAPER_REFERENCE["scratch"]
    if "none" in valid and "bc" in valid:
        result["bc_above_vanilla"] = valid["bc"] > valid["none"]
    if "none" in valid and "ewc" in valid:
        result["ewc_above_vanilla"] = valid["ewc"] > valid["none"]
    return result


def aggregate_per_level(runs: Sequence[MethodRun], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, Any]:
    """Aggregate the per-level (level 4 / Sokoban) scores across seeds."""
    out: Dict[str, Any] = {}
    for method in TRAINING_METHODS:
        method_runs = [r for r in runs if r.method == method]
        if not method_runs:
            continue
        keys = sorted({k for r in method_runs for k in r.per_level})
        if not keys:
            continue
        out[method] = {k: summarize([r.per_level.get(k, float("nan")) for r in method_runs], confidence)
                       for k in keys}
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pipeline_results(runs: Sequence[MethodRun], output_dir: str,
                          confidence: float = DEFAULT_CONFIDENCE,
                          make_plots: bool = True) -> List[str]:
    """Render the evaluation-score curves and the return distributions."""
    paths: List[str] = []
    if not make_plots:
        return paths
    plotting = _import("src.analysis.plotting")
    return_dist = _import("src.analysis.return_distribution")
    os.makedirs(output_dir, exist_ok=True)

    curves: Dict[str, List[Tuple[float, float]]] = {}
    for run in runs:
        points: List[Tuple[float, float]] = []
        for record in run.history:
            step = _cfg_get(record, "step", None)
            value = _first(record, ["final_score", "score", "return", "mean_score", "return_mean"])
            if step is None or value is None:
                continue
            points.append((float(step), float(value)))
        if points:
            curves[run.method] = points

    if curves and plotting is not None:
        try:
            fig = plotting.plot_curves(curves, confidence=confidence,
                                       xlabel="environment steps", ylabel="score",
                                       title="NetHack Human Monk fine-tuning",
                                       path=os.path.join(output_dir, "nethack_curves.png"))
            if fig is not None:
                paths.append(os.path.join(output_dir, "nethack_curves.png"))
        except Exception:  # pragma: no cover
            pass

    if return_dist is not None and runs:
        try:
            records = []
            for run in runs:
                values = [v for _, v in [(0, run.final_score)] if v is not None and not math.isnan(v)]
                record = return_dist.ReturnRecord(method=run.method, seed=run.seed, step=run.steps,
                                                  returns=values, mean_return=run.final_score, n_episodes=1)
                records.append(record)
            dists = return_dist.aggregate_methods(records, confidence=confidence, step="final")
            path = os.path.join(output_dir, "nethack_return_distribution.png")
            return_dist.plot_return_distribution(dists, path=path, title="NetHack final scores")
            paths.append(path)
        except Exception:  # pragma: no cover
            pass
    return paths


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(cfg: Optional[NetHackPipelineConfig] = None, *,
                 methods: Optional[Sequence[str]] = None,
                 seeds: Optional[Sequence[int]] = None,
                 total_steps: Optional[int] = None,
                 checkpoint: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 stub: Optional[bool] = None,
                 eval_episodes: Optional[int] = None,
                 verify_pi_star: bool = True,
                 compute_fisher_diag: bool = True,
                 build_bc: bool = True,
                 make_plots: bool = True,
                 logger: Optional[Callable[[str], None]] = None,
                 progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
                 verbose: bool = True) -> PipelineResult:
    """Run the full NetHack Human Monk pipeline.

    Steps (Section 3 / 5, Appendix B.1):

    1. load and verify pi*  (``~5K`` on Human Monk),
    2. pre-train the baseline head for 500M steps with encoders frozen,
    3. compute the diagonal Fisher (10000 NLD-AA batches) and build the BC
       buffer (10000 pre-training states),
    4. fine-tune every requested variant (``scratch / none / ewc / bc / ks``),
    5. aggregate across seeds and check the paper's ordering,
    6. per-level evaluation at levels 4 and Sokoban (200 AutoAscend saves).
    """
    cfg = (cfg or NetHackPipelineConfig()).with_overrides(
        total_steps=total_steps,
        output_dir=output_dir,
        stub=stub,
        eval_episodes=eval_episodes,
        verbose=verbose,
    )
    methods = tuple(normalize_method(m) for m in (methods or cfg.methods))
    seeds = tuple(int(s) for s in (seeds or cfg.seeds))
    log = logger or ((lambda message: print(message)) if cfg.verbose else (lambda message: None))

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "config.json"), cfg.to_dict())

    result = PipelineResult(methods=methods, output_dir=out_dir, config=cfg.to_dict())
    start = time.time()

    # --- 1. pi* ------------------------------------------------------------
    model = None
    runner = _import_runner()
    if runner is not None:
        try:
            model = runner.build_nethack_model(cfg, checkpoint=checkpoint or cfg.pretrained_checkpoint)
        except Exception as exc:  # pragma: no cover
            result.errors.append(f"pi_star load failed: {type(exc).__name__}: {exc}")
    if verify_pi_star:
        try:
            result.pi_star = verify_pistar(cfg, checkpoint=checkpoint, logger=log)
        except Exception as exc:  # pragma: no cover
            result.pi_star = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}

    # --- 2. baseline head --------------------------------------------------
    if cfg.pretrain_baseline_head:
        try:
            result.pi_star.setdefault("baseline_head", pretrain_baseline_head(cfg, model, logger=log,
                                                                             output_dir=out_dir))
        except Exception as exc:  # pragma: no cover
            result.pi_star.setdefault("baseline_head", {"status": "error", "error": str(exc)})

    # --- 3. Fisher + BC buffer --------------------------------------------
    fisher = None
    if "ewc" in methods and compute_fisher_diag:
        result.fisher = compute_fisher(cfg, model, logger=log)
        fisher = _cfg_get(result.fisher, "diagonals", None)
    bc_dataset = None
    if "bc" in methods and build_bc:
        result.bc_dataset = build_bc_buffer(cfg, model, logger=log)
        bc_dataset = _cfg_get(result.bc_dataset, "buffer", None)

    teacher = model  # pi* is the distillation teacher for every retention method

    # --- 4. fine-tuning runs ----------------------------------------------
    runs: List[MethodRun] = []
    for seed in seeds:
        for method in methods:
            log(f"[nethack] === method={method} seed={seed} ===")
            run = run_method(cfg, method, seed, teacher=teacher, fisher=fisher,
                             bc_dataset=bc_dataset, model=model, output_dir=out_dir,
                             logger=log, progress_fn=progress_fn, total_steps=cfg.total_steps,
                             checkpoint=checkpoint)
            if run.error:
                result.errors.append(f"{method}/seed{seed}: {run.error}")
            runs.append(run)
            _write_json(os.path.join(out_dir, method, f"seed_{seed}", "summary.json"),
                        run.as_dict(include_history=True))

    result.runs = [r.as_dict(include_history=True) for r in runs]

    # --- 5. aggregation ---------------------------------------------------
    result.aggregates = aggregate_seeds(runs, confidence=cfg.confidence)
    result.ordering = check_ordering(result.aggregates)
    result.per_level = aggregate_per_level(runs, confidence=cfg.confidence)
    result.elapsed = time.time() - start

    # --- 6. figures + summary --------------------------------------------
    try:
        plot_pipeline_results(runs, out_dir, confidence=cfg.confidence, make_plots=make_plots)
    except Exception as exc:  # pragma: no cover
        result.errors.append(f"plotting failed: {type(exc).__name__}: {exc}")

    result.summary_path = _write_json(os.path.join(out_dir, "nethack_summary.json"),
                                      result.as_dict(include_history=True))
    log(f"[nethack] pipeline finished in {result.elapsed:.1f}s -> {result.summary_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.nethack.train_nethack",
        description="NetHack Human Monk APPO fine-tuning (Table 1 / Appendix B.1).",
    )
    parser.add_argument("--config", type=str, default=None, help="path to configs/nethack.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="config overrides, e.g. total_steps=500000000")
    parser.add_argument("--method", type=str, default=None, help="single variant to run")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="variants: scratch none ewc bc ks")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="pi* checkpoint path")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stub", action="store_true", help="CPU-only dry run")
    parser.add_argument("--no-pretrain-baseline", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny run: 2048 steps, 1 seed, stub environment")
    return parser


def _apply_cli_overrides(cfg: Any, overrides: Optional[Sequence[str]]) -> Any:
    """Apply ``a.b=c`` overrides through ``src.common.config.apply_overrides``."""
    if not overrides:
        return cfg
    fn = _import("src.common.config", "apply_overrides")
    if fn is None:
        return cfg
    try:
        return fn(cfg, list(overrides))
    except Exception:  # pragma: no cover
        return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    raw_cfg: Any = None
    if args.config:
        loader = _import("src.common.config", "load_config")
        if loader is not None:
            try:
                raw_cfg = loader(args.config)
            except Exception:  # pragma: no cover
                raw_cfg = None
    if raw_cfg is not None:
        raw_cfg = _apply_cli_overrides(raw_cfg, args.overrides)

    cfg = NetHackPipelineConfig.from_config(raw_cfg)
    if args.method:
        cfg = cfg.with_overrides(methods=(normalize_method(args.method),))
    if args.methods:
        cfg = cfg.with_overrides(methods=tuple(normalize_method(m) for m in args.methods))
    if args.seeds:
        cfg = cfg.with_overrides(seeds=tuple(int(s) for s in args.seeds))
    if args.total_steps:
        cfg = cfg.with_overrides(total_steps=int(args.total_steps))
    if args.output_dir:
        cfg = cfg.with_overrides(output_dir=args.output_dir)
    if args.eval_episodes:
        cfg = cfg.with_overrides(eval_episodes=int(args.eval_episodes))
    if args.confidence is not None:
        cfg = cfg.with_overrides(confidence=float(args.confidence))
    if args.device:
        cfg = cfg.with_overrides(device=args.device)
    if args.stub:
        cfg = cfg.with_overrides(stub=True)
    if args.no_pretrain_baseline:
        cfg = cfg.with_overrides(pretrain_baseline_head=False)
    if args.smoke_test:
        cfg = cfg.with_overrides(stub=True, total_steps=2048, seeds=(0,), eval_every=2048,
                                 eval_episodes=1, save_every=2048, per_level_eval=False,
                                 methods=("none", "bc"))

    result = run_pipeline(cfg, checkpoint=args.checkpoint, make_plots=not args.no_plots)
    print(json.dumps({k: v for k, v in result.as_dict(include_history=False).items()
                      if k not in ("config",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
