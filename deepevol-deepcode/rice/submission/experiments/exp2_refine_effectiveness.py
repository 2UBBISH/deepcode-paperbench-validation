"""Experiment II - Effectiveness of the RICE refining method (ICML 2024, PMLR 235).

This driver reproduces the paper's Experiment II (Section 4.2) and its two
companions inside Section 4.3 / Appendix C.4:

* **Experiment II (Table 1, left block + Figure 2 group (a))**
  *Fix the explanation method to ours (mask network) and vary the **refining**
  method*: PPO fine-tuning, JSRL, StateMask-R (StateMask's "start fine-tuning
  only from critical steps") and Ours.  We report the final reward of the
  refined agent, mean (std) over seeds, together with the "No Refine" value of
  the frozen pre-trained policy.

* **Experiment III style explanation ablation (Table 1, right block +
  Figure 2 group (b))**
  *Fix the refining method to Ours and vary the **explanation** method*:
  Random, StateMask and Ours.  All refiners consume the very same explanation
  objects so the comparison isolates the explanation quality.

* **Sparse MuJoCo refining (Figure 2)**
  The same comparison in SparseHopper / SparseHalfCheetah, where RICE is
  expected to show both the best final performance *and* the highest refining
  efficiency (i.e. it reaches its plateau in fewer samples).

Expected trends (validated, not numerically exact):

1. ``ours`` attains the largest improvement over "No Refine" in every
   application.
2. ``ppo_finetune`` only improves marginally (it cannot jump out of the local
   optimum).
3. ``statemask_r`` does **not** always help (always starting from critical
   states overfits and can even hurt, e.g. CAGE-2 / Auto Driving).
4. Explanation ablation: ``ours`` and ``statemask`` are comparable and both
   beat ``random``.
5. Sparse games: ``ours`` is best in final performance and refining efficiency.

Usage (CLI)::

    python experiments/exp2_refine_effectiveness.py --env hopper
    python experiments/exp2_refine_effectiveness.py --envs hopper walker2d reacher halfcheetah
    python experiments/exp2_refine_effectiveness.py --sparse
    python experiments/exp2_refine_effectiveness.py --env hopper --ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Repository import path (allow running as a plain script)
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --------------------------------------------------------------------------
# Defensive imports of the already implemented RICE modules
# --------------------------------------------------------------------------
try:  # utilities
    from rice.utils.io import ensure_dir, get_config, save_json
    from rice.utils.logging import Logger, format_mean_std, get_logger
    from rice.utils.seeding import seed_from, set_seed
except Exception:  # pragma: no cover - minimal fallbacks
    import logging as _logging

    def ensure_dir(path):  # type: ignore
        os.makedirs(path, exist_ok=True)
        return path

    def get_config(name="default", config_dir=None):  # type: ignore
        return {}

    def save_json(obj, path, indent=2):  # type: ignore
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=indent, default=str)
        return path

    def get_logger(name="rice", out_dir=None, level=None):  # type: ignore
        return _logging.getLogger(name)

    def format_mean_std(values, decimals=2):  # type: ignore
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        if arr.size == 0:
            return "n/a"
        if arr.size == 1:
            return f"{arr[0]:.{decimals}f}"
        return f"{arr.mean():.{decimals}f} +- {arr.std():.{decimals}f}"

    class Logger:  # type: ignore
        def __init__(self, out_dir=None, name="rice", config=None, verbose=True):
            self.out_dir = out_dir
            self.history = {}
            self.timers = {}

        def record(self, **kwargs):
            for k, v in kwargs.items():
                self.history.setdefault(k, []).append(v)

        def log_dict(self, data, prefix=""):
            return None

        def timer_start(self, name):
            self.timers[name] = {"start": time.time(), "total": 0.0}
            return self.timers[name]["start"]

        def timer_end(self, name, accumulate=True):
            t = self.timers.setdefault(name, {"start": time.time(), "total": 0.0})
            dt = time.time() - t["start"]
            t["total"] = t.get("total", 0.0) + dt if accumulate else dt
            return dt

        def dump(self, filename="progress.json"):
            return None

        def close(self):
            return None

    def set_seed(seed, deterministic=False):  # type: ignore
        np.random.seed(int(seed))
        return int(seed)

    def seed_from(base_seed, *offsets):  # type: ignore
        return int(base_seed) + sum(int(o) for o in offsets) * 7919


try:
    from rice.envs.make_env import (
        cage2_final_reward,
        d_max_for,
        env_backend,
        env_metadata,
        make_env,
        resolve_env_spec,
    )
except Exception:  # pragma: no cover
    make_env = None  # type: ignore
    d_max_for = None  # type: ignore
    env_metadata = None  # type: ignore
    env_backend = None  # type: ignore
    cage2_final_reward = None  # type: ignore
    resolve_env_spec = None  # type: ignore

try:
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        save_policy,
    )
except Exception:  # pragma: no cover
    build_policy = None  # type: ignore
    load_policy = None  # type: ignore
    save_policy = None  # type: ignore

    def normalize_env_key(env_id):  # type: ignore
        key = str(env_id).strip().lower().replace("-", "_")
        for suffix in ("_v3", "_v2", "_v1", "_v0", "_v4"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key


try:
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
    )
except Exception:  # pragma: no cover
    build_mask_network = None  # type: ignore
    load_mask_network = None  # type: ignore
    save_mask_network = None  # type: ignore

try:
    from rice.explanation.mask_trainer import DEFAULT_ALPHA, train_mask_network
except Exception:  # pragma: no cover
    DEFAULT_ALPHA = 1e-4  # type: ignore

    def train_mask_network(*args, **kwargs):  # type: ignore
        raise RuntimeError("rice.explanation.mask_trainer is not available")


try:
    from rice.refining.ppo_refine import (
        DEFAULT_LAMBDA,
        DEFAULT_P,
        evaluate_refined_policy,
        refine_policy,
        unpack_reset,
        unpack_step,
    )
except Exception:  # pragma: no cover
    DEFAULT_LAMBDA = 0.01  # type: ignore
    DEFAULT_P = 0.5  # type: ignore
    refine_policy = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore


# --- refining baselines ----------------------------------------------------
try:
    from rice.baselines.ppo_finetune import (
        DEFAULT_FINETUNE_LR,
        ppo_finetune_policy,
    )
except Exception:  # pragma: no cover
    DEFAULT_FINETUNE_LR = 1e-4  # type: ignore
    ppo_finetune_policy = None  # type: ignore

try:
    from rice.baselines.statemask_r import (
        DEFAULT_MASK_SAMPLES,
        refine_from_critical_state,
        samples_for,
        train_statemask_network,
    )
except Exception:  # pragma: no cover
    DEFAULT_MASK_SAMPLES = {}  # type: ignore
    train_statemask_network = None  # type: ignore
    refine_from_critical_state = None  # type: ignore

    def samples_for(env_id, default=300_000):  # type: ignore
        return default


try:
    from rice.baselines.jsrl import train_jsrl
except Exception:  # pragma: no cover
    train_jsrl = None  # type: ignore

try:
    from rice.baselines.sil import sil as sil_baseline
except Exception:  # pragma: no cover
    sil_baseline = None  # type: ignore

try:
    from rice.baselines.random_explanation import make_random_explanation
except Exception:  # pragma: no cover
    make_random_explanation = None  # type: ignore


# ==========================================================================
# Constants / paper reference values
# ==========================================================================

DEFAULT_ENVS: Tuple[str, ...] = (
    "hopper",
    "walker2d",
    "reacher",
    "halfcheetah",
    "selfish_mining",
    "cage2",
    "autodriving",
)

#: In-scope dense applications used by Experiment II (Malware Mutation excluded).
DENSE_ENVS: Tuple[str, ...] = DEFAULT_ENVS

#: Figure 2 environments.
SPARSE_ENVS: Tuple[str, ...] = ("sparse_hopper", "sparse_halfcheetah")

#: Refining methods compared in the left block of Table 1 / Figure 2(a).
REFINING_METHODS: Tuple[str, ...] = ("ours", "ppo_finetune", "statemask_r", "jsrl")

#: Explanation methods compared in the right block of Table 1 / Figure 2(b).
EXPLANATION_METHODS: Tuple[str, ...] = ("random", "statemask", "ours")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EVAL_EPISODES: int = 10
DEFAULT_REFINE_TIMESTEPS: int = 200_000
DEFAULT_MASK_TIMESTEPS: int = 300_000
DEFAULT_PRETRAIN_TIMESTEPS: int = 1_000_000

# --------------------------------------------------------------------------
# Table 1 reference values (trend validation only - NOT exact targets)
# "No Refine" == the frozen pre-trained target policy pi
# --------------------------------------------------------------------------
REFERENCE_NO_REFINE: Dict[str, float] = {
    "hopper": 3559.44,
    "walker2d": 3768.79,
    "reacher": -5.79,
    "halfcheetah": 2024.09,
    "selfish_mining": 14.36,
    "cage2": -23.64,
    "autodriving": 10.30,
    "malware_mutation": 42.20,
}

REFERENCE_OURS: Dict[str, float] = {
    "hopper": 3663.91,
    "walker2d": 3982.79,
    "reacher": -2.66,
    "halfcheetah": 2138.89,
    "selfish_mining": 16.56,
    "cage2": -20.02,
    "autodriving": 17.03,
    "malware_mutation": 57.53,
}

REFERENCE_PPO_FINETUNE: Dict[str, float] = {
    "hopper": 3638.75,
    "walker2d": 3965.63,
    "reacher": -3.04,
    "halfcheetah": 2133.31,
    "selfish_mining": 14.93,
    "cage2": -23.58,
    "autodriving": 13.37,
    "malware_mutation": 49.33,
}

REFERENCE_JSRL: Dict[str, float] = {
    "hopper": 3635.08,
    "walker2d": 3963.57,
    "reacher": -3.23,
    "halfcheetah": 2128.04,
    "selfish_mining": 14.88,
    "cage2": -22.97,
    "autodriving": 11.26,
    "malware_mutation": 43.10,
}

REFERENCE_STATEMASK_R: Dict[str, float] = {
    "hopper": 3652.06,
    "walker2d": 3966.96,
    "reacher": -3.45,
    "halfcheetah": 2085.28,
    "selfish_mining": 14.53,
    "cage2": -26.98,
    "autodriving": 7.62,
    "malware_mutation": 50.13,
}

# Right block of Table 1: fix refine = ours, vary explanation.
REFERENCE_RANDOM_EXPLANATION: Dict[str, float] = {
    "hopper": 3648.98,
    "walker2d": 3969.64,
    "reacher": -3.11,
    "halfcheetah": 2132.01,
    "selfish_mining": 15.09,
    "cage2": -25.94,
    "autodriving": 11.72,
}

REFERENCE_STATEMASK_EXPLANATION: Dict[str, float] = {
    "hopper": 3661.86,
    "walker2d": 3982.67,
    "reacher": -2.69,
    "halfcheetah": 2136.23,
    "selfish_mining": 16.49,
    "cage2": -20.07,
    "autodriving": 16.28,
}

# Table 5 (optional): SIL vs RICE on four MuJoCo games.
REFERENCE_SIL: Dict[str, float] = {
    "hopper": 3646.46,
    "walker2d": 3967.66,
    "reacher": -2.87,
    "halfcheetah": 2069.80,
}

#: CAGE Challenge 2 rewards are negative - "higher is better" still holds.
NEGATIVE_REWARD_ENVS: Tuple[str, ...] = ("cage2", "reacher")


# ==========================================================================
# Result containers
# ==========================================================================


@dataclass
class RefineResult:
    """Outcome of refining a frozen target policy with one method."""

    method: str
    env_id: str
    final_reward: float = float("nan")
    std: float = float("nan")
    eval_rewards: List[float] = field(default_factory=list)
    no_refine_reward: Optional[float] = None
    improvement: Optional[float] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    samples: int = 0
    explanation: str = "ours"
    seed: Optional[int] = None
    policy: Any = None
    refiner: Any = None
    checkpoint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "method": self.method,
            "env_id": self.env_id,
            "explanation": self.explanation,
            "final_reward": _safe_float(self.final_reward),
            "std": _safe_float(self.std),
            "no_refine_reward": _safe_float(self.no_refine_reward),
            "improvement": _safe_float(self.improvement),
            "wall_time": _safe_float(self.wall_time),
            "samples": int(self.samples),
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "eval_rewards": [_safe_float(v) for v in self.eval_rewards],
            "summary": _to_jsonable(self.summary),
            "extra": _to_jsonable(self.extra),
        }
        if include_history and self.history:
            out["history"] = _to_jsonable(self.history)
        return out

    def format(self, decimals: int = 2) -> str:
        return (
            f"{self.method:<14} reward={self.final_reward:.{decimals}f} "
            f"({self.std:.{decimals}f})"
        )


@dataclass
class ExplanationHandle:
    """A trained (or trivial) Stage-1 explanation ready for refining."""

    method: str
    env_id: str
    mask_net: Any = None
    trainer: Any = None
    train_time: float = 0.0
    samples: int = 0
    checkpoint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "env_id": self.env_id,
            "train_time": _safe_float(self.train_time),
            "samples": int(self.samples),
            "checkpoint": self.checkpoint,
            "has_mask_net": self.mask_net is not None,
            "extra": _to_jsonable(self.extra),
        }


# ==========================================================================
# Small helpers
# ==========================================================================


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std())


def reference_for(method: str, env_id: str) -> Optional[float]:
    """Paper reference final reward for ``(method, env_id)`` if available."""
    key = normalize_env_key(env_id)
    table = {
        "ours": REFERENCE_OURS,
        "ppo_finetune": REFERENCE_PPO_FINETUNE,
        "ppo": REFERENCE_PPO_FINETUNE,
        "jsrl": REFERENCE_JSRL,
        "statemask_r": REFERENCE_STATEMASK_R,
        "random": REFERENCE_RANDOM_EXPLANATION,
        "random_explanation": REFERENCE_RANDOM_EXPLANATION,
        "statemask": REFERENCE_STATEMASK_EXPLANATION,
        "statemask_explanation": REFERENCE_STATEMASK_EXPLANATION,
        "sil": REFERENCE_SIL,
    }.get(str(method).strip().lower())
    if table is None:
        return None
    return table.get(key)


# ==========================================================================
# Environment / policy preparation
# ==========================================================================


def build_experiment_env(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    mode: str = "train",
    **kwargs: Any,
) -> Any:
    """Create the Experiment-II environment through the RICE env factory."""
    if make_env is None:
        raise RuntimeError("rice.envs.make_env is unavailable")
    env_cfg = (cfg or {}).get("env", {}) or {}
    use_seed = seed if seed is not None else env_cfg.get("seed", 0)
    max_steps = kwargs.pop("max_episode_steps", env_cfg.get("max_episode_steps", None))
    normalize = kwargs.pop("normalize", env_cfg.get("normalize_obs", None))
    return make_env(
        env_id,
        seed=use_seed,
        normalize=normalize,
        mode=mode,
        max_episode_steps=max_steps,
        **kwargs,
    )


def build_or_load_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Load the frozen pre-trained target policy pi (checkpoint or fresh)."""
    target_cfg = (cfg or {}).get("target", {}) or {}
    ckpt = checkpoint or target_cfg.get("checkpoint")
    if ckpt and load_policy is not None and os.path.exists(str(ckpt)):
        try:
            policy = load_policy(ckpt, env_id=env_id, kind="policy", device=device, **kwargs)
            if logger is not None:
                logger.info("Loaded target policy pi from %s", ckpt)
            return policy
        except Exception as exc:  # pragma: no cover - defensive
            if logger is not None:
                logger.warning("Could not load %s (%s); building a fresh policy", ckpt, exc)
    return build_target_policy(env, env_id, cfg=cfg, device=device, **kwargs)


def build_target_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Instantiate a fresh target policy pi with the per-app architecture."""
    if build_policy is None:
        raise RuntimeError("rice.models.policies is unavailable")
    target_cfg = (cfg or {}).get("target", {}) or {}
    obs_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)
    return build_policy(
        env_id=env_id,
        observation_space=obs_space,
        action_space=action_space,
        kind="policy",
        device=device,
        hidden_sizes=target_cfg.get("hidden_sizes"),
        activation=target_cfg.get("activation"),
        **kwargs,
    )


def pretrain_target_policy(
    env: Any,
    env_id: str,
    total_timesteps: int = DEFAULT_PRETRAIN_TIMESTEPS,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Pre-train pi with plain PPO (no mixed init, no RND) -> "No Refine" regime."""
    if refine_policy is None:
        raise RuntimeError("rice.refining.ppo_refine is unavailable")
    if logger is not None:
        logger.info(
            "Pre-training target policy on %s for %s steps (plain PPO)", env_id, total_timesteps
        )
    policy, _refiner = refine_policy(
        env,
        policy=None,
        mask_net=None,
        total_timesteps=int(total_timesteps),
        env_id=env_id,
        config=cfg,
        seed=seed,
        device=device,
        use_mixed_init=False,
        use_rnd=False,
        **kwargs,
    )
    return policy


# ==========================================================================
# Stage 1 - explanations
# ==========================================================================


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Fixed mask-training sample budget (Table 4, Appendix C.3)."""
    explanation_cfg = (cfg or {}).get("explanation", {}) or {}
    configured = explanation_cfg.get("total_timesteps")
    if configured:
        return int(configured)
    return int(samples_for(env_id, DEFAULT_MASK_TIMESTEPS))


def train_ours_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> ExplanationHandle:
    """Algorithm 1: vanilla PPO + blinding bonus ``alpha * a_t^m``."""
    explanation_cfg = (cfg or {}).get("explanation", {}) or {}
    budget = int(total_timesteps or mask_budget_for(env_id, cfg))
    alpha = DEFAULT_ALPHA if alpha is None else float(alpha)
    if alpha is None:
        alpha = float(explanation_cfg.get("alpha", DEFAULT_ALPHA))
    t0 = time.time()
    mask_net, trainer = train_mask_network(
        env,
        policy,
        total_timesteps=budget,
        alpha=alpha,
        env_id=env_id,
        config=cfg,
        logger=logger,
        save_path=checkpoint,
        seed=seed,
        device=device,
        **kwargs,
    )
    elapsed = time.time() - t0
    if checkpoint and save_mask_network is not None:
        try:
            save_mask_network(mask_net, checkpoint, env_id=env_id, extra={"alpha": alpha})
        except Exception:  # pragma: no cover
            pass
    return ExplanationHandle(
        method="ours",
        env_id=env_id,
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=budget,
        checkpoint=checkpoint,
        extra={"alpha": alpha},
    )


def train_statemask_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: Optional[int] = None,
    alpha: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> ExplanationHandle:
    """StateMask's primal-dual mask training (explanation baseline)."""
    if train_statemask_network is None:
        raise RuntimeError("rice.baselines.statemask_r is unavailable")
    budget = int(total_timesteps or mask_budget_for(env_id, cfg))
    t0 = time.time()
    mask_net, trainer = train_statemask_network(
        env,
        policy,
        total_timesteps=budget,
        alpha=alpha,
        env_id=env_id,
        config=cfg,
        logger=logger,
        save_path=checkpoint,
        seed=seed,
        device=device,
        **kwargs,
    )
    elapsed = time.time() - t0
    return ExplanationHandle(
        method="statemask",
        env_id=env_id,
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=budget,
        checkpoint=checkpoint,
        extra={"alpha": alpha},
    )


def random_explanation(env_id: str, logger: Any = None, cfg: Optional[Dict[str, Any]] = None) -> ExplanationHandle:
    """Random explanation baseline (no mask net -> uninformative importance)."""
    handle = ExplanationHandle(method="random", env_id=env_id, mask_net=None, train_time=0.0)
    if make_random_explanation is not None:
        try:
            handle.extra["explainer"] = make_random_explanation(env_id=env_id)
        except Exception:  # pragma: no cover
            pass
    return handle


def train_explanation(
    method: str,
    env: Any,
    policy: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    mask_timesteps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    **kwargs: Any,
) -> ExplanationHandle:
    """Dispatch to the requested Stage-1 explanation method."""
    method = str(method).strip().lower()
    ckpt = None
    if checkpoint_dir:
        ensure_dir(checkpoint_dir)
        if method == "ours":
            ckpt = os.path.join(checkpoint_dir, f"{normalize_env_key(env_id)}_mask.pt")
        elif method == "statemask":
            ckpt = os.path.join(checkpoint_dir, f"{normalize_env_key(env_id)}_statemask_mask.pt")

    if method in ("ours", "rice", "mask", "mask_net"):
        return train_ours_explanation(
            env, policy, env_id, total_timesteps=mask_timesteps, cfg=cfg,
            seed=seed, device=device, logger=logger, checkpoint=ckpt, **kwargs,
        )
    if method in ("statemask", "state_mask"):
        return train_statemask_explanation(
            env, policy, env_id, total_timesteps=mask_timesteps, cfg=cfg,
            seed=seed, device=device, logger=logger, checkpoint=ckpt, **kwargs,
        )
    if method in ("random", "none"):
        return random_explanation(env_id, logger=logger, cfg=cfg)
    raise ValueError(f"Unknown explanation method: {method!r}")


# ==========================================================================
# Stage 2 - refining methods
# ==========================================================================


def evaluate_policy_return(
    env: Any,
    policy: Any,
    env_id: str,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Evaluate the (refined) policy's undiscounted episode return."""
    if evaluate_refined_policy is not None:
        try:
            res = evaluate_refined_policy(
                env,
                policy,
                env_id=env_id,
                n_episodes=int(n_episodes),
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
            )
            mean, std = _mean_std(res.get("rewards", []))
            res.setdefault("mean_reward", mean)
            res.setdefault("std_reward", std)
            res["mean"] = mean
            res["std"] = std
            return res
        except Exception as exc:  # pragma: no cover - fall back to local loop
            if logger is not None:
                logger.warning("evaluate_refined_policy failed (%s); using local loop", exc)
    rewards = _local_evaluate(
        env, policy, n_episodes=n_episodes, deterministic=deterministic,
        max_steps=max_steps, env_id=env_id, **kwargs,
    )
    mean, std = _mean_std(rewards)
    return {"rewards": rewards, "mean": mean, "std": std,
            "mean_reward": mean, "std_reward": std, "n_episodes": len(rewards)}


def _local_evaluate(
    env: Any,
    policy: Any,
    n_episodes: int = 10,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    env_id: str = "default",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> List[float]:
    """Minimal roll-out evaluator (used only when the shared one is unavailable)."""
    rewards: List[float] = []
    horizon = _resolve_horizon(env, default=1000)
    limit = int(max_steps or horizon)
    for ep in range(int(n_episodes)):
        if seed is not None:
            set_seed(seed_from(int(seed), ep))
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        total = 0.0
        for _ in range(limit):
            action = _policy_action(policy, obs, deterministic=deterministic)
            step = env.step(action)
            if len(step) == 5:
                obs, reward, terminated, truncated, _info = step
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, _info = step
            total += float(reward)
            if done:
                break
        rewards.append(total)
    return rewards


def _resolve_horizon(env: Any, default: int = 1000) -> int:
    node = env
    for _ in range(8):
        if node is None:
            break
        for attr in ("rice_max_episode_steps", "_max_episode_steps", "max_episode_steps"):
            value = getattr(node, attr, None)
            if isinstance(value, (int, float)) and value:
                return int(value)
        spec = getattr(node, "rice_env_spec", None)
        if spec is not None:
            v = getattr(spec, "max_episode_steps", None)
            if v:
                return int(v)
        node = getattr(node, "env", None)
    return int(default)


def _policy_action(policy: Any, observation: Any, deterministic: bool = False) -> Any:
    if policy is None:
        raise ValueError("policy is None")
    if hasattr(policy, "predict"):
        try:
            action, _ = policy.predict(observation, deterministic=deterministic)
            return action
        except TypeError:
            action, _ = policy.predict(observation)
            return action
    if hasattr(policy, "act"):
        try:
            return policy.act(observation, deterministic=deterministic)
        except TypeError:
            return policy.act(observation)
    if callable(policy):
        return policy(observation)
    raise TypeError(f"Cannot act with policy of type {type(policy)!r}")


def _run_refiner(
    method: str,
    env: Any,
    policy: Any,
    mask_net: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    total_timesteps: Optional[int] = None,
    evaluate: bool = True,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Run one refining method; returns ``(refined_policy, refiner, info)``."""
    method = str(method).strip().lower()
    refine_cfg = (cfg or {}).get("refine", {}) or {}
    budget = int(total_timesteps or refine_cfg.get("total_timesteps", DEFAULT_REFINE_TIMESTEPS))
    p = float(refine_cfg.get("p", DEFAULT_P))
    lam = float(refine_cfg.get("lam", DEFAULT_LAMBDA))
    info: Dict[str, Any] = {"method": method, "samples": budget, "p": p, "lam": lam}

    if method in ("ours", "rice"):
        if refine_policy is None:
            raise RuntimeError("rice.refining.ppo_refine is unavailable")
        policy_out, refiner = refine_policy(
            env, policy=policy, mask_net=mask_net, total_timesteps=budget,
            p=p, lam=lam, env_id=env_id, config=cfg, logger=logger,
            seed=seed, device=device, save_path=checkpoint, **kwargs,
        )
    elif method in ("ppo_finetune", "ppo", "finetune"):
        if ppo_finetune_policy is None:
            raise RuntimeError("rice.baselines.ppo_finetune is unavailable")
        policy_out, refiner = ppo_finetune_policy(
            env, policy=policy, total_timesteps=budget, env_id=env_id,
            config=cfg, logger=logger, save_path=checkpoint, seed=seed,
            device=device, **kwargs,
        )
    elif method in ("statemask_r", "statemask"):
        if refine_from_critical_state is None:
            raise RuntimeError("rice.baselines.statemask_r is unavailable")
        policy_out, refiner = refine_from_critical_state(
            env, policy=policy, mask_net=mask_net, total_timesteps=budget,
            env_id=env_id, config=cfg, logger=logger, save_path=checkpoint,
            seed=seed, device=device, **kwargs,
        )
    elif method in ("jsrl",):
        if train_jsrl is None:
            raise RuntimeError("rice.baselines.jsrl is unavailable")
        policy_out, refiner = train_jsrl(
            env, guided_policy=policy, policy=None, total_timesteps=budget,
            env_id=env_id, config=cfg, logger=logger, save_path=checkpoint,
            seed=seed, device=device, **kwargs,
        )
    elif method in ("sil",):
        if sil_baseline is None:
            raise RuntimeError("rice.baselines.sil is unavailable")
        res = sil_baseline(
            env, policy=policy, total_timesteps=budget, env_id=env_id,
            config=cfg, logger=logger, seed=seed, device=device, **kwargs,
        )
        policy_out = res.get("policy") if isinstance(res, dict) else res
        refiner = res if isinstance(res, dict) else None
        info["sil"] = _to_jsonable(res) if isinstance(res, dict) else None
    else:
        raise ValueError(f"Unknown refining method: {method!r}")

    info["eval"] = None
    if evaluate:
        ev = evaluate_policy_return(
            env, policy_out, env_id, n_episodes=eval_episodes, device=device, logger=logger
        )
        info["eval"] = ev
    return policy_out, refiner, info


def refine_with_method(
    method: str,
    env: Any,
    policy: Any,
    mask_net: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    total_timesteps: Optional[int] = None,
    no_refine_reward: Optional[float] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    explanation: str = "ours",
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> RefineResult:
    """Refine with ``method`` and package the outcome into a :class:`RefineResult`."""
    t0 = time.time()
    errors: List[str] = []
    policy_out, refiner, info = None, None, {}
    try:
        policy_out, refiner, info = _run_refiner(
            method, env, policy, mask_net, env_id, cfg=cfg, seed=seed, device=device,
            logger=logger, total_timesteps=total_timesteps, eval_episodes=eval_episodes,
            checkpoint=checkpoint, **kwargs,
        )
    except Exception as exc:  # pragma: no cover - keep the sweep alive
        errors.append(f"{type(exc).__name__}: {exc}")
        if logger is not None:
            logger.error("Refining method %r failed: %s", method, exc)
    elapsed = time.time() - t0

    ev = (info or {}).get("eval") or {}
    rewards = list(ev.get("rewards", []) or [])
    mean, std = _mean_std(rewards)
    if not rewards and policy_out is not None:
        try:
            ev = evaluate_policy_return(
                env, policy_out, env_id, n_episodes=eval_episodes, device=device, logger=logger
            )
            rewards = list(ev.get("rewards", []) or [])
            mean, std = _mean_std(rewards)
        except Exception as exc:  # pragma: no cover
            errors.append(f"eval: {exc}")

    improvement = None
    if no_refine_reward is not None and np.isfinite(mean):
        improvement = float(mean - float(no_refine_reward))

    summary: Dict[str, Any] = {}
    if refiner is not None and hasattr(refiner, "summary"):
        try:
            summary = _to_jsonable(refiner.summary())
        except Exception:  # pragma: no cover
            summary = {}

    return RefineResult(
        method=method,
        env_id=env_id,
        final_reward=mean,
        std=std,
        eval_rewards=rewards,
        no_refine_reward=no_refine_reward,
        improvement=improvement,
        history=_history_of(refiner),
        summary=summary if isinstance(summary, dict) else {"summary": summary},
        wall_time=elapsed,
        samples=int((info or {}).get("samples", 0) or 0),
        explanation=explanation,
        seed=seed,
        policy=policy_out,
        refiner=refiner,
        checkpoint=checkpoint,
        extra={"info": _to_jsonable(info), "errors": errors},
    )


def _history_of(refiner: Any) -> List[Dict[str, float]]:
    if refiner is None:
        return []
    for attr in ("history", "eval_history", "log_history"):
        hist = getattr(refiner, attr, None)
        if isinstance(hist, list) and hist:
            return [_to_jsonable(h) if isinstance(h, dict) else {"value": _safe_float(h)}
                    for h in hist]
    return []


# ==========================================================================
# Experiment II - main driver
# ==========================================================================


def run_experiment2(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    methods: Sequence[str] = REFINING_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    pretrain_timesteps: Optional[int] = None,
    policy_checkpoint: Optional[str] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    save_policies: bool = False,
    explanation: str = "ours",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fix the explanation (``ours`` by default) and vary the refining method."""
    log = logger or get_logger(f"exp2.{env_id}")
    cfg = cfg if cfg is not None else get_config(env_id)
    report: Dict[str, Any] = {
        "experiment": "II",
        "env_id": env_id,
        "methods": list(methods),
        "seeds": list(seeds),
        "results": {},
        "no_refine": None,
        "explanation": None,
        "reference": {
            "no_refine": REFERENCE_NO_REFINE.get(normalize_env_key(env_id)),
            "ours": REFERENCE_OURS.get(normalize_env_key(env_id)),
        },
    }
    log.info("=== Experiment II :: %s :: refining methods=%s ===", env_id, list(methods))

    per_method: Dict[str, List[RefineResult]] = {m: [] for m in methods}
    no_refine_values: List[float] = []
    explanation_handle: Optional[ExplanationHandle] = None

    for seed in seeds:
        set_seed(int(seed))
        env = build_experiment_env(env_id, cfg=cfg, seed=int(seed), mode="train")
        try:
            policy = build_or_load_policy(
                env, env_id, cfg=cfg, device=device, checkpoint=policy_checkpoint, logger=log
            )
            if policy_checkpoint is None and (cfg.get("target", {}) or {}).get("checkpoint") is None:
                policy = pretrain_target_policy(
                    env, env_id,
                    total_timesteps=int(pretrain_timesteps or DEFAULT_PRETRAIN_TIMESTEPS),
                    seed=int(seed), device=device, logger=log, cfg=cfg,
                )

            no_refine = evaluate_policy_return(
                env, policy, env_id, n_episodes=eval_episodes, device=device, logger=log
            )
            no_refine_values.append(float(no_refine.get("mean", float("nan"))))
            log.info("seed=%s No-Refine reward = %.3f", seed, no_refine_values[-1])

            # ---- Stage 1: explanation (shared by all refiners) -----------
            if explanation not in ("random",):
                if explanation_handle is None:
                    ckpt_dir = os.path.join(out_dir, "checkpoints") if out_dir else None
                    explanation_handle = train_explanation(
                        explanation, env, policy, env_id, cfg=cfg, seed=int(seed),
                        device=device, logger=log, mask_timesteps=mask_timesteps,
                        checkpoint_dir=ckpt_dir, progress=True,
                    )
            mask_net = explanation_handle.mask_net if explanation_handle else None

            # ---- Stage 2: each refining method ---------------------------
            for method in methods:
                ckpt = None
                if save_policies and out_dir:
                    ensure_dir(os.path.join(out_dir, "policies"))
                    ckpt = os.path.join(
                        out_dir, "policies",
                        f"{normalize_env_key(env_id)}_{method}_seed{seed}.zip",
                    )
                if progress:
                    log.info("seed=%s refining with %s ...", seed, method)
                res = refine_with_method(
                    method, env, policy, mask_net, env_id, cfg=cfg, seed=int(seed),
                    device=device, logger=log, total_timesteps=refine_timesteps,
                    no_refine_reward=no_refine_values[-1], eval_episodes=eval_episodes,
                    explanation=explanation, checkpoint=ckpt, **kwargs,
                )
                per_method[method].append(res)
                log.info("seed=%s %s -> %.3f", seed, method, res.final_reward)
        finally:
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass

    report["no_refine"] = _mean_std(no_refine_values)
    report["explanation"] = explanation_handle.to_dict() if explanation_handle else None
    no_refine_mean = report["no_refine"][0]

    for method, results in per_method.items():
        means = [r.final_reward for r in results]
        mean, std = _mean_std(means)
        entries = [r.to_dict() for r in results]
        all_eval = [v for r in results for v in r.eval_rewards]
        report["results"][method] = {
            "mean": _safe_float(mean),
            "std": _safe_float(std),
            "n_seeds": len(results),
            "per_seed": entries,
            "eval_rewards": [_safe_float(v) for v in all_eval],
            "wall_time": float(np.nansum([r.wall_time for r in results])),
            "samples": int(results[0].samples if results else 0),
            "improvement_over_no_refine": _safe_float(mean - no_refine_mean)
            if np.isfinite(mean) and np.isfinite(no_refine_mean) else None,
            "reference": reference_for(method, env_id),
        }

    report["trend_check"] = check_trends(report)
    if out_dir:
        ensure_dir(out_dir)
        save_json(report, os.path.join(out_dir, f"exp2_{normalize_env_key(env_id)}.json"))
    if progress:
        log.info("\n%s", format_report(report))
    return report


# ==========================================================================
# Experiment III style - vary the explanation, fix the refiner to ours
# ==========================================================================


def run_experiment2_explanations(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    explanations: Sequence[str] = EXPLANATION_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    pretrain_timesteps: Optional[int] = None,
    policy_checkpoint: Optional[str] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fix the refining method to ours and vary the explanation method."""
    log = logger or get_logger(f"exp2x.{env_id}")
    cfg = cfg if cfg is not None else get_config(env_id)
    report: Dict[str, Any] = {
        "experiment": "II-explanation-ablation",
        "env_id": env_id,
        "explanations": list(explanations),
        "seeds": list(seeds),
        "results": {},
        "no_refine": None,
        "reference": {
            "no_refine": REFERENCE_NO_REFINE.get(normalize_env_key(env_id)),
            "ours": REFERENCE_OURS.get(normalize_env_key(env_id)),
            "statemask": REFERENCE_STATEMASK_EXPLANATION.get(normalize_env_key(env_id)),
            "random": REFERENCE_RANDOM_EXPLANATION.get(normalize_env_key(env_id)),
        },
    }
    log.info("=== Experiment II :: %s :: explanations=%s ===", env_id, list(explanations))

    per_expl: Dict[str, List[RefineResult]] = {e: [] for e in explanations}
    no_refine_values: List[float] = []

    for seed in seeds:
        set_seed(int(seed))
        env = build_experiment_env(env_id, cfg=cfg, seed=int(seed), mode="train")
        try:
            policy = build_or_load_policy(
                env, env_id, cfg=cfg, device=device, checkpoint=policy_checkpoint, logger=log
            )
            if policy_checkpoint is None and (cfg.get("target", {}) or {}).get("checkpoint") is None:
                policy = pretrain_target_policy(
                    env, env_id,
                    total_timesteps=int(pretrain_timesteps or DEFAULT_PRETRAIN_TIMESTEPS),
                    seed=int(seed), device=device, logger=log, cfg=cfg,
                )
            no_refine = evaluate_policy_return(
                env, policy, env_id, n_episodes=eval_episodes, device=device, logger=log
            )
            no_refine_values.append(float(no_refine.get("mean", float("nan"))))

            for expl in explanations:
                ckpt_dir = os.path.join(out_dir, "checkpoints") if out_dir else None
                handle = train_explanation(
                    expl, env, policy, env_id, cfg=cfg, seed=int(seed), device=device,
                    logger=log, mask_timesteps=mask_timesteps, checkpoint_dir=ckpt_dir,
                    progress=progress,
                )
                res = refine_with_method(
                    "ours", env, policy, handle.mask_net, env_id, cfg=cfg, seed=int(seed),
                    device=device, logger=log, total_timesteps=refine_timesteps,
                    no_refine_reward=no_refine_values[-1], eval_episodes=eval_episodes,
                    explanation=expl, **kwargs,
                )
                res.extra["explanation_train_time"] = handle.train_time
                per_expl[expl].append(res)
                log.info("seed=%s explanation=%s -> %.3f", seed, expl, res.final_reward)
        finally:
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass

    report["no_refine"] = _mean_std(no_refine_values)
    for expl, results in per_expl.items():
        mean, std = _mean_std([r.final_reward for r in results])
        report["results"][expl] = {
            "mean": _safe_float(mean),
            "std": _safe_float(std),
            "n_seeds": len(results),
            "per_seed": [r.to_dict() for r in results],
            "reference": reference_for(expl, env_id),
            "improvement_over_no_refine": _safe_float(mean - report["no_refine"][0])
            if np.isfinite(mean) and np.isfinite(report["no_refine"][0]) else None,
        }
    report["trend_check"] = check_explanation_trends(report)
    if out_dir:
        ensure_dir(out_dir)
        save_json(report, os.path.join(out_dir, f"exp2_explanations_{normalize_env_key(env_id)}.json"))
    return report


# ==========================================================================
# Sparse MuJoCo refining (Figure 2)
# ==========================================================================


def run_experiment2_sparse(
    env_ids: Sequence[str] = SPARSE_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    methods: Sequence[str] = REFINING_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    refine_timesteps: Optional[int] = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    **kwargs: Any,
) -> Dict[str, Any]:
    """SparseHopper / SparseHalfCheetah refining comparison (Figure 2)."""
    log = logger or get_logger("exp2.sparse")
    out: Dict[str, Any] = {"experiment": "II-sparse", "envs": list(env_ids), "results": {}}
    for env_id in env_ids:
        env_cfg = cfg if cfg is not None else get_config(env_id)
        out["results"][env_id] = run_experiment2(
            env_id, cfg=env_cfg, methods=methods, seeds=seeds,
            refine_timesteps=refine_timesteps, device=device,
            out_dir=out_dir, logger=log, progress=progress,
            eval_episodes=eval_episodes, **kwargs,
        )
    if out_dir:
        ensure_dir(out_dir)
        save_json(out, os.path.join(out_dir, "exp2_sparse.json"))
    return out


def run_experiment2_multi(
    env_ids: Sequence[str] = DENSE_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Experiment II across several applications."""
    out_dir = kwargs.pop("out_dir", None)
    logger = kwargs.pop("logger", None)
    progress = kwargs.pop("progress", False)
    log = logger or get_logger("exp2.multi")
    combined: Dict[str, Any] = {"experiment": "II", "envs": list(env_ids), "results": {}}
    for env_id in env_ids:
        env_cfg = cfg if cfg is not None else None
        try:
            combined["results"][env_id] = run_experiment2(
                env_id, cfg=env_cfg, out_dir=out_dir, logger=log,
                progress=progress, **kwargs,
            )
        except Exception as exc:  # pragma: no cover - keep sweep alive
            log.error("Experiment II failed for %s: %s", env_id, exc)
            combined["results"][env_id] = {"error": f"{type(exc).__name__}: {exc}"}
    if out_dir:
        ensure_dir(out_dir)
        save_json(combined, os.path.join(out_dir, "exp2_all.json"))
    return combined


# ==========================================================================
# Trend validation (qualitative ordering - the reproduction target)
# ==========================================================================


def check_trends(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Table-1 style qualitative orderings for a single env."""
    results = report.get("results", {})
    no_refine = report.get("no_refine")
    no_refine_mean = no_refine[0] if isinstance(no_refine, (list, tuple)) else no_refine

    def _mean(method: str) -> Optional[float]:
        entry = results.get(method)
        if not entry:
            return None
        v = entry.get("mean")
        return None if v is None else float(v)

    ours = _mean("ours")
    ppo = _mean("ppo_finetune")
    smr = _mean("statemask_r")
    jsrl = _mean("jsrl")

    checks: Dict[str, Any] = {}
    checks["ours_beats_no_refine"] = (
        None if ours is None or no_refine_mean is None else bool(ours > no_refine_mean)
    )
    checks["ours_beats_ppo_finetune"] = (
        None if ours is None or ppo is None else bool(ours >= ppo)
    )
    checks["ours_beats_jsrl"] = None if ours is None or jsrl is None else bool(ours >= jsrl)
    checks["ours_beats_statemask_r"] = (
        None if ours is None or smr is None else bool(ours >= smr)
    )
    checks["ppo_finetune_marginal"] = (
        None if ppo is None or no_refine_mean is None else bool(0.0 <= ppo - no_refine_mean)
    )
    valid = [v for v in checks.values() if v is not None]
    checks["passed"] = int(sum(bool(v) for v in valid))
    checks["total"] = len(valid)
    return checks


def check_explanation_trends(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that Ours/StateMask explanations beat Random (Table 1 right block)."""
    results = report.get("results", {})

    def _mean(method: str) -> Optional[float]:
        entry = results.get(method)
        v = (entry or {}).get("mean")
        return None if v is None else float(v)

    ours, sm, rnd = _mean("ours"), _mean("statemask"), _mean("random")
    checks: Dict[str, Any] = {
        "ours_beats_random": None if ours is None or rnd is None else bool(ours > rnd),
        "statemask_beats_random": None if sm is None or rnd is None else bool(sm > rnd),
        # The paper only requires comparability between ours and StateMask.
        "ours_comparable_to_statemask": (
            None if ours is None or sm is None else bool(abs(ours - sm) <= 0.25 * max(1.0, abs(sm)))
        ),
    }
    valid = [v for v in checks.values() if v is not None]
    checks["passed"] = int(sum(bool(v) for v in valid))
    checks["total"] = len(valid)
    return checks


# ==========================================================================
# Reporting
# ==========================================================================


def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    """Render an Experiment-II report as a Table-1-like text block."""
    lines: List[str] = []
    env_id = report.get("env_id", "?")
    lines.append(f"Experiment II :: {env_id}")
    lines.append("-" * 72)
    no_refine = report.get("no_refine")
    if isinstance(no_refine, (list, tuple)):
        lines.append(f"{'No Refine':<22} {no_refine[0]:.{decimals}f} ({no_refine[1]:.{decimals}f})")
    elif no_refine is not None:
        lines.append(f"{'No Refine':<22} {no_refine:.{decimals}f}")
    lines.append("")
    lines.append(f"{'Method':<22} {'Reward':>18}   {'Ref':>10}  {'d':>10}")
    for method, entry in report.get("results", {}).items():
        mean, std = entry.get("mean", float("nan")), entry.get("std", float("nan"))
        ref = entry.get("reference")
        delta = entry.get("improvement_over_no_refine")
        mean_s = f"{mean:.{decimals}f}" if mean is not None else "n/a"
        std_s = f"{std:.{decimals}f}" if std is not None else "n/a"
        ref_s = f"{ref:.{decimals}f}" if ref is not None else "-"
        d_s = f"{delta:+.{decimals}f}" if delta is not None else "-"
        lines.append(f"{method:<22} {mean_s:>10} ({std_s:>6})   {ref_s:>10}  {d_s:>10}")
    checks = report.get("trend_check")
    if checks:
        lines.append("")
        lines.append(f"trend checks passed: {checks.get('passed')}/{checks.get('total')}")
        for key, value in checks.items():
            if key in ("passed", "total"):
                continue
            lines.append(f"  {key:<32} {value}")
    lines.append("-" * 72)
    return "\n".join(lines)


def format_explanation_report(report: Dict[str, Any], decimals: int = 2) -> str:
    lines = [f"Experiment II (explanation ablation) :: {report.get('env_id', '?')}", "-" * 72]
    no_refine = report.get("no_refine")
    if isinstance(no_refine, (list, tuple)):
        lines.append(f"{'No Refine':<18} {no_refine[0]:.{decimals}f} ({no_refine[1]:.{decimals}f})")
    for expl, entry in report.get("results", {}).items():
        mean, std = entry.get("mean", float("nan")), entry.get("std", float("nan"))
        ref = entry.get("reference")
        mean_s = f"{mean:.{decimals}f}" if mean is not None else "n/a"
        std_s = f"{std:.{decimals}f}" if std is not None else "n/a"
        ref_s = f"{ref:.{decimals}f}" if ref is not None else "-"
        lines.append(f"{expl:<18} {mean_s:>10} ({std_s:>6})   ref={ref_s:>10}")
    checks = report.get("trend_check")
    if checks:
        lines.append(f"trend checks passed: {checks.get('passed')}/{checks.get('total')}")
    lines.append("-" * 72)
    return "\n".join(lines)


# ==========================================================================
# CLI
# ==========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RICE Experiment II - effectiveness of the refining method (Table 1, Figure 2)."
    )
    p.add_argument("--env", "--env_id", dest="env", default=None, help="single application id")
    p.add_argument("--envs", nargs="+", default=None, help="several application ids")
    p.add_argument("--sparse", action="store_true", help="run the sparse MuJoCo comparison")
    p.add_argument("--ablation", action="store_true", help="vary the explanation, fix refine=ours")
    p.add_argument("--methods", nargs="+", default=list(REFINING_METHODS))
    p.add_argument("--explanations", nargs="+", default=list(EXPLANATION_METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--refine-timesteps", type=int, default=None)
    p.add_argument("--mask-timesteps", type=int, default=None)
    p.add_argument("--pretrain-timesteps", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "results", "exp2"))
    p.add_argument("--policy-checkpoint", default=None)
    p.add_argument("--save-policies", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = ensure_dir(args.out_dir) if args.out_dir else None
    logger = get_logger("exp2", out_dir=out_dir)
    progress = not args.quiet

    kwargs = dict(
        seeds=tuple(args.seeds),
        refine_timesteps=args.refine_timesteps,
        device=args.device,
        out_dir=out_dir,
        logger=logger,
        progress=progress,
        pretrain_timesteps=args.pretrain_timesteps,
        eval_episodes=args.eval_episodes,
    )

    if args.sparse:
        report = run_experiment2_sparse(
            env_ids=tuple(args.envs) if args.envs else SPARSE_ENVS,
            methods=tuple(args.methods),
            **kwargs,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, default=str))
        return 0

    env_ids = args.envs or ([args.env] if args.env else list(DENSE_ENVS))
    if args.ablation:
        reports = {}
        for env_id in env_ids:
            rep = run_experiment2_explanations(
                env_id, explanations=tuple(args.explanations), mask_timesteps=args.mask_timesteps,
                **kwargs,
            )
            reports[env_id] = rep
            if progress:
                print(format_explanation_report(rep))
        if out_dir:
            save_json(reports, os.path.join(out_dir, "exp2_explanations_all.json"))
        return 0

    if len(env_ids) == 1:
        rep = run_experiment2(
            env_ids[0], methods=tuple(args.methods), mask_timesteps=args.mask_timesteps,
            policy_checkpoint=args.policy_checkpoint, save_policies=args.save_policies,
            **kwargs,
        )
        if progress:
            print(format_report(rep))
        return 0

    combined = run_experiment2_multi(
        env_ids=tuple(env_ids), methods=tuple(args.methods), mask_timesteps=args.mask_timesteps,
        policy_checkpoint=args.policy_checkpoint, **kwargs,
    )
    if progress:
        for env_id, rep in combined.get("results", {}).items():
            if isinstance(rep, dict) and "results" in rep:
                print(format_report(rep))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
