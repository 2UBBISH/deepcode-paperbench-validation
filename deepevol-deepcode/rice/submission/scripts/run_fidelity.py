"""Experiment I entry point: fidelity score evaluation for RICE explanations.

This script is the CLI/glue layer for the paper's *fidelity* metric (Sec. 4.1
"Evaluation Metrics" and Sec. 4.2 "Experiment I").  It:

1. builds the environment + the frozen pre-trained target policy ``pi``,
2. trains (or loads) each requested Stage-1 explanation method
   (``ours`` = RICE mask net via Algorithm 1, ``statemask`` = primal-dual
   baseline, ``random`` = uniform uninformative baseline),
3. evaluates the sliding-window fidelity score

       ``fidelity = log(d / d_max) - log(l / L)``

   for ``K in {10%, 20%, 30%, 40%}`` (window width ``l = L * K``) over
   ``500`` trajectories x ``3`` seeds, reporting mean +- std, and
4. reports the mask-network *training-time* efficiency comparison against
   StateMask for a fixed sample budget (Table 4).

Heavy lifting is delegated to the already implemented ``rice`` modules; this
file only does CLI parsing, seed loops, checkpoint handling and reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Defensive imports (the CLI must stay introspectable without torch/SB3/MuJoCo)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float("nan")
    if np is not None:
        return float(np.mean(vals))
    return sum(vals) / len(vals)


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    if np is not None:
        return float(np.std(vals))
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


try:
    from rice.utils.io import ensure_dir, get_config, save_json
except Exception:  # pragma: no cover - fallbacks
    def ensure_dir(path):
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def get_config(name="default", config_dir=None):
        try:
            import yaml
        except Exception:
            return {}
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        base = os.path.join(root, "default.yaml")
        cfg = {}
        if os.path.exists(base):
            with open(base, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
        target = name if os.path.isabs(str(name)) else os.path.join(root, "%s.yaml" % name)
        if os.path.exists(target):
            with open(target, "r") as fh:
                override = yaml.safe_load(fh) or {}
            _merge(cfg, override)
        return cfg

    def _merge(base, override):
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _merge(base[key], value)
            else:
                base[key] = value
        return base

    def save_json(obj, path, indent=2):
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=indent, default=str)
        return path


try:
    from rice.utils.logging import Logger, format_mean_std, get_logger
except Exception:  # pragma: no cover
    import logging as _logging

    def get_logger(name="rice", out_dir=None, level=None):
        return _logging.getLogger(name)

    def format_mean_std(values, decimals=2):
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return "%.*f" % (decimals, vals[0])
        return "%.*f +- %.*f" % (decimals, _mean(vals), decimals, _std(vals))

    class Logger(object):  # noqa: D101 - minimal fallback
        def __init__(self, out_dir=None, name="rice", config=None, verbose=True):
            self.out_dir = out_dir
            self.history = {}
            self.timers = {}

        def record(self, **kwargs):
            for key, value in kwargs.items():
                try:
                    self.history.setdefault(key, []).append(float(value))
                except Exception:
                    continue

        def log_dict(self, data, prefix=""):
            return None

        def timer_start(self, name):
            self.timers["_start_" + name] = time.time()
            return self.timers["_start_" + name]

        def timer_end(self, name, accumulate=True):
            start = self.timers.pop("_start_" + name, time.time())
            elapsed = time.time() - start
            key = "time/" + name
            self.timers[key] = self.timers.get(key, 0.0) + elapsed
            return elapsed

        def dump(self, filename="progress.json"):
            if not self.out_dir:
                return None
            return save_json({"history": self.history, "timers": self.timers},
                             os.path.join(self.out_dir, filename))

        def close(self):
            return None


try:
    from rice.utils.seeding import seed_from, set_seed
except Exception:  # pragma: no cover
    import random as _random

    def set_seed(seed, deterministic=False):
        seed = int(seed)
        _random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        return seed

    def seed_from(base_seed, *offsets):
        value = int(base_seed)
        for offset in offsets:
            value = (value * 1000003 + int(offset) + 0x9E3779B9) & 0xFFFFFFFF
        return value


try:
    from rice.envs.make_env import (
        available_envs,
        d_max_for,
        env_backend,
        env_metadata,
        make_env,
        resolve_env_spec,
    )
except Exception:  # pragma: no cover
    make_env = None

    def d_max_for(env_id, default=None):
        return default

    def env_backend(env):
        return "fallback"

    def env_metadata(env_id, probe=False):
        return {}

    def available_envs():
        return []

    def resolve_env_spec(env_id):  # pragma: no cover
        raise NotImplementedError("rice.envs.make_env unavailable")


try:
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        save_policy,
    )
except Exception:  # pragma: no cover
    build_policy = None
    load_policy = None
    save_policy = None

    def normalize_env_key(env_id):
        key = str(env_id or "default").strip().lower().replace("-", "_")
        for suffix in ("_v2", "_v3", "_v4", "_csv"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key


try:
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
        state_importance,
    )
except Exception:  # pragma: no cover
    build_mask_network = None
    load_mask_network = None
    save_mask_network = None

    def state_importance(mask_net, observations, batch_size=4096):  # pragma: no cover
        if np is None:
            return [1.0] * len(observations)
        return np.ones(len(observations), dtype=np.float32)


try:
    from rice.explanation.mask_trainer import DEFAULT_ALPHA, train_mask_network
except Exception:  # pragma: no cover
    DEFAULT_ALPHA = 1e-4

    def train_mask_network(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("rice.explanation.mask_trainer unavailable")


try:
    from rice.explanation.fidelity import (
        DEFAULT_K_VALUES,
        DEFAULT_N_TRAJECTORIES,
        DEFAULT_SEEDS as FIDELITY_SEEDS,
        FidelityConfig,
        FidelityEvaluator,
        FidelityResult,
        evaluate_fidelity,
        evaluate_fidelity_multi_K,
        evaluate_methods,
        format_fidelity_table,
        training_time_reduction,
    )
except Exception:  # pragma: no cover - fidelity module missing
    DEFAULT_K_VALUES = (0.10, 0.20, 0.30, 0.40)
    DEFAULT_N_TRAJECTORIES = 500
    FIDELITY_SEEDS = (0, 1, 2)
    FidelityConfig = None
    FidelityEvaluator = None
    FidelityResult = None
    evaluate_fidelity = None
    evaluate_fidelity_multi_K = None
    evaluate_methods = None

    def format_fidelity_table(results, decimals=3):
        return {}

    def training_time_reduction(baseline_times, ours_times):
        ours = _mean(ours_times or [])
        base = _mean(baseline_times or [])
        reduction = (base - ours) / base if base else 0.0
        return {"ours": ours, "baseline": base, "reduction": reduction,
                "percent": 100.0 * reduction}


try:
    from rice.baselines.statemask_r import (
        DEFAULT_MASK_SAMPLES,
        samples_for,
        train_statemask_network,
    )
except Exception:  # pragma: no cover
    DEFAULT_MASK_SAMPLES = {}
    train_statemask_network = None

    def samples_for(env_id, default=300_000):
        return int(DEFAULT_MASK_SAMPLES.get(str(env_id), default))


try:
    from rice.baselines.random_explanation import make_random_explanation
except Exception:  # pragma: no cover
    make_random_explanation = None

try:
    from rice.refining.ppo_refine import (
        DEFAULT_GAMMA,
        DEFAULT_GAE_LAMBDA,
        DEFAULT_PPO_LR,
        evaluate_refined_policy,
        refine_policy,
    )
except Exception:  # pragma: no cover
    DEFAULT_PPO_LR = 3e-4
    DEFAULT_GAMMA = 0.99
    DEFAULT_GAE_LAMBDA = 0.95
    evaluate_refined_policy = None
    refine_policy = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
SPARSE_ENVS: Tuple[str, ...] = ("sparse_hopper", "sparse_halfcheetah")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_DEVICE = "cpu"
DEFAULT_OUT_DIR = "results/fidelity"
DEFAULT_CHECKPOINT_DIR = "policies"

EXPLANATION_METHODS: Tuple[str, ...] = ("ours", "statemask", "random")

#: Table-4 fixed mask-net sample budgets (per application).
TABLE4_SAMPLES: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "sparse_hopper": 300_000,
    "sparse_halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
}

#: Table-4 reference wall-clock times (seconds) for the efficiency comparison.
TABLE4_TIMES: Dict[str, Dict[str, float]] = {
    "hopper": {"ours": 12426.0, "statemask": 15393.0},
    "halfcheetah": {"ours": 1317.0, "statemask": 1579.0},
    "cage2": {"ours": 65400.0, "statemask": 79382.0},
}

PAPER_TIME_REDUCTION = 0.168  # ~16.8% faster mask training than StateMask

#: Table-3 per-application hyper-parameters (alpha is treated as insensitive).
TABLE3: Dict[str, Dict[str, float]] = {
    "hopper": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "walker2d": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "reacher": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "halfcheetah": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "sparse_hopper": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "sparse_halfcheetah": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "selfish_mining": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "cage2": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
    "autodriving": {"p": 0.5, "lam": 0.01, "alpha": 1e-4},
}

ALPHA_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)

#: Paper reference "No Refine" returns (used for sanity logging only).
REFERENCE_NO_REFINE: Dict[str, Optional[float]] = {
    "hopper": 3559.44,
    "walker2d": 3339.68,
    "reacher": -5.51,
    "halfcheetah": 4540.50,
    "selfish_mining": None,
    "cage2": -23.64,
    "autodriving": 10.30,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deep_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Nested lookup into dicts / objects with a default."""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    if not cfg:
        return default
    return deep_get(cfg, *keys, default=default)


def is_negative_env(env_id: str) -> bool:
    """True for applications whose returns are signed-negative."""
    return normalize_env_key(env_id) in ("reacher", "cage2")


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Fixed Table-4 mask-net training sample budget for an application."""
    key = normalize_env_key(env_id)
    value = _cfg_get(cfg, "explanation", "total_timesteps", default=None)
    if value:
        return int(value)
    try:
        return int(samples_for(key, default=TABLE4_SAMPLES.get(key, 300_000)))
    except Exception:
        return int(TABLE4_SAMPLES.get(key, 300_000))


def alpha_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Blinding-bonus coefficient alpha (Table 3 default, exposed for sweeps)."""
    key = normalize_env_key(env_id)
    value = _cfg_get(cfg, "explanation", "alpha", default=None)
    if value is not None:
        return float(value)
    default_alpha = float(getattr(sys.modules.get("rice.explanation.mask_trainer"), "DEFAULT_ALPHA", 1e-4) or 1e-4)
    return float(TABLE3.get(key, {}).get("alpha", default_alpha))


def checkpoint_path(env_id: str, seed: Optional[int] = None,
                    out_dir: str = DEFAULT_CHECKPOINT_DIR) -> str:
    """``policies/<env>_mask[_seedN].pt`` mask-network checkpoint path."""
    key = normalize_env_key(env_id)
    suffix = "" if seed is None else "_seed%d" % int(seed)
    return os.path.join(out_dir, "%s_mask%s.pt" % (key, suffix))


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def _policy_action(policy: Any, observation: Any, deterministic: bool = True) -> Any:
    if policy is None:
        return None
    for attr in ("predict", "act"):
        fn = getattr(policy, attr, None)
        if callable(fn):
            try:
                result = fn(observation, deterministic=deterministic)
            except TypeError:
                result = fn(observation)
            if isinstance(result, tuple):
                return result[0]
            return result
    if callable(policy):
        return policy(observation)
    return None


# ---------------------------------------------------------------------------
# Environment / policy plumbing
# ---------------------------------------------------------------------------

def build_experiment_env(env_id: str, cfg: Optional[Dict[str, Any]] = None,
                         seed: int = 0, mode: str = "eval", **kwargs: Any) -> Any:
    """Create the evaluation environment used by the fidelity sweep.

    ``mode="eval"`` freezes observation normalisation statistics
    (Walker2d / HalfCheetah) so repeated fidelity runs are comparable.
    """
    if make_env is None:
        raise RuntimeError("rice.envs.make_env is unavailable; cannot build environments")
    env_kwargs: Dict[str, Any] = {"mode": mode}
    max_steps = _cfg_get(cfg, "env", "max_episode_steps", default=None)
    if max_steps:
        env_kwargs["max_episode_steps"] = int(max_steps)
    env_kwargs.update(kwargs)
    return make_env(env_id, seed=seed, **env_kwargs)


def build_target_policy(env: Any, env_id: str, cfg: Optional[Dict[str, Any]] = None,
                        device: str = DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Build a fresh (untrained) per-application target policy."""
    if build_policy is None:
        raise RuntimeError("rice.models.policies.build_policy is unavailable")
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    return build_policy(
        env_id=env_id,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        **kwargs,
    )


def build_or_load_policy(env: Any, env_id: str, cfg: Optional[Dict[str, Any]] = None,
                         device: str = DEFAULT_DEVICE, checkpoint: Optional[str] = None,
                         logger: Any = None) -> Any:
    """Load the frozen pre-trained policy pi from a checkpoint, else build fresh."""
    key = normalize_env_key(env_id)
    candidates: List[str] = []
    if checkpoint:
        candidates.append(checkpoint)
    cfg_ckpt = _cfg_get(cfg, "target", "checkpoint", default=None)
    if cfg_ckpt:
        candidates.append(cfg_ckpt)
    candidates.extend([
        os.path.join(DEFAULT_CHECKPOINT_DIR, "%s_ppo.zip" % key),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "%s_ppo.pt" % key),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "%s_ppo" % key),
    ])
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            policy = load_policy(
                path,
                env_id=env_id,
                observation_space=getattr(env, "observation_space", None),
                action_space=getattr(env, "action_space", None),
                device=device,
            )
            if logger is not None:
                try:
                    logger.info("Loaded target policy from %s", path)
                except Exception:
                    pass
            return policy
        except Exception:
            continue
    if logger is not None:
        try:
            logger.info("No target-policy checkpoint found; building a fresh policy for %s", key)
        except Exception:
            pass
    return build_target_policy(env, env_id, cfg=cfg, device=device)


def pretrain_target_policy(env: Any, env_id: str, total_timesteps: int = 1_000_000,
                           seed: int = 0, device: str = DEFAULT_DEVICE,
                           logger: Any = None, cfg: Optional[Dict[str, Any]] = None,
                           **kwargs: Any) -> Any:
    """Pre-train pi with plain PPO (mixed-init and RND disabled)."""
    if refine_policy is None:
        raise RuntimeError("rice.refining.ppo_refine.refine_policy is unavailable")
    policy, _refiner = refine_policy(
        env,
        policy=None,
        total_timesteps=int(total_timesteps),
        env_id=env_id,
        config=cfg,
        seed=int(seed),
        device=device,
        logger=logger,
        progress=kwargs.get("progress", False),
        use_mixed_init=False,
        use_rnd=False,
        p=0.0,
        lam=0.0,
    )
    return policy


# ---------------------------------------------------------------------------
# Stage-1 explanation training / loading
# ---------------------------------------------------------------------------

@dataclass
class ExplanationArtifacts:
    """Trained (or trivially derived) Stage-1 explanation."""

    method: str
    env_id: str
    mask_net: Any = None
    trainer: Any = None
    train_time: Optional[float] = None
    samples: Optional[int] = None
    seconds_per_sample: Optional[float] = None
    checkpoint: Optional[str] = None
    scoring: str = "mask"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "env_id": self.env_id,
            "train_time": self.train_time,
            "samples": self.samples,
            "seconds_per_sample": self.seconds_per_sample,
            "checkpoint": self.checkpoint,
            "scoring": self.scoring,
            "extra": _jsonable(self.extra),
        }


def train_ours_explanation(env: Any, policy: Any, env_id: str,
                           total_timesteps: Optional[int] = None,
                           alpha: Optional[float] = None, seed: int = 0,
                           device: str = DEFAULT_DEVICE,
                           checkpoint: Optional[str] = None, logger: Any = None,
                           cfg: Optional[Dict[str, Any]] = None,
                           progress: bool = False, **kwargs: Any) -> ExplanationArtifacts:
    """Algorithm 1: vanilla PPO + blinding bonus ``alpha * a_t^m``."""
    key = normalize_env_key(env_id)
    budget = int(total_timesteps or mask_budget_for(key, cfg))
    alpha = alpha_for(key, cfg) if alpha is None else float(alpha)
    hidden = _cfg_get(cfg, "explanation", "hidden_sizes", default=None)

    start = time.time()
    mask_net, trainer = train_mask_network(
        env,
        policy,
        total_timesteps=budget,
        alpha=alpha,
        env_id=key,
        config=cfg,
        logger=logger,
        save_path=None,
        seed=int(seed),
        device=device,
        progress=progress,
        hidden_sizes=hidden,
    )
    elapsed = time.time() - start
    time_report = {}
    try:
        time_report = trainer.time_report() or {}
    except Exception:
        time_report = {}
    if time_report.get("total_time"):
        elapsed = float(time_report["total_time"])

    saved = None
    if save_mask_network is not None and checkpoint:
        try:
            save_mask_network(mask_net, checkpoint, env_id=key,
                              obs_dim=getattr(env, "observation_space", None) and
                              getattr(env.observation_space, "shape", [None])[0])
            saved = checkpoint
        except Exception:
            saved = None

    return ExplanationArtifacts(
        method="ours",
        env_id=key,
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=budget,
        seconds_per_sample=(elapsed / budget) if budget else None,
        checkpoint=saved,
        scoring="mask",
        extra={"alpha": alpha, "time_report": _jsonable(time_report)},
    )


def train_statemask_explanation(env: Any, policy: Any, env_id: str,
                                total_timesteps: Optional[int] = None,
                                alpha: float = 0.01, seed: int = 0,
                                device: str = DEFAULT_DEVICE,
                                checkpoint: Optional[str] = None,
                                logger: Any = None,
                                cfg: Optional[Dict[str, Any]] = None,
                                progress: bool = False,
                                **kwargs: Any) -> ExplanationArtifacts:
    """StateMask baseline: primal-dual ``min |eta(pi) - eta(pi_bar)|``."""
    key = normalize_env_key(env_id)
    budget = int(total_timesteps or mask_budget_for(key, cfg))
    if train_statemask_network is None:
        raise RuntimeError("rice.baselines.statemask_r.train_statemask_network is unavailable")

    start = time.time()
    mask_net, trainer = train_statemask_network(
        env,
        policy,
        total_timesteps=budget,
        alpha=float(alpha),
        env_id=key,
        config=cfg,
        logger=logger,
        save_path=None,
        seed=int(seed),
        device=device,
        progress=progress,
    )
    elapsed = time.time() - start
    time_report = {}
    try:
        time_report = trainer.time_report() or {}
    except Exception:
        time_report = {}
    if time_report.get("total_time"):
        elapsed = float(time_report["total_time"])

    return ExplanationArtifacts(
        method="statemask",
        env_id=key,
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=budget,
        seconds_per_sample=(elapsed / budget) if budget else None,
        checkpoint=None,
        scoring="mask",
        extra={"alpha_init": float(alpha), "time_report": _jsonable(time_report)},
    )


def random_explanation(env_id: str, logger: Any = None,
                       cfg: Optional[Dict[str, Any]] = None,
                       seed: int = 0, **_: Any) -> ExplanationArtifacts:
    """Random explanation baseline: no mask net, uninformative importance."""
    key = normalize_env_key(env_id)
    return ExplanationArtifacts(
        method="random",
        env_id=key,
        mask_net=None,
        trainer=None,
        train_time=0.0,
        samples=0,
        seconds_per_sample=0.0,
        checkpoint=None,
        scoring="random",
        extra={"seed": int(seed)},
    )


def train_explanation(method: str, env: Any, policy: Any, env_id: str,
                      cfg: Optional[Dict[str, Any]] = None, seed: int = 0,
                      device: str = DEFAULT_DEVICE, logger: Any = None,
                      mask_timesteps: Optional[int] = None,
                      checkpoint_dir: Optional[str] = None,
                      progress: bool = False, **kwargs: Any) -> ExplanationArtifacts:
    """Dispatch Stage-1 explanation construction by method name."""
    method = str(method).strip().lower()
    key = normalize_env_key(env_id)
    if method in ("ours", "rice", "mask"):
        ckpt = checkpoint_path(key, seed, checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        try:
            return train_ours_explanation(
                env, policy, key, total_timesteps=mask_timesteps, seed=seed,
                device=device, checkpoint=ckpt, logger=logger, cfg=cfg,
                progress=progress, **kwargs,
            )
        except Exception as exc:  # fall back to a frozen/loaded mask net
            if logger is not None:
                try:
                    logger.warning("Mask training failed for %s (%s)", key, exc)
                except Exception:
                    pass
            if load_mask_network is not None:
                for path in (ckpt, os.path.join(DEFAULT_CHECKPOINT_DIR, "%s_mask.pt" % key)):
                    if path and os.path.exists(path):
                        try:
                            mask_net = load_mask_network(path, env_id=key,
                                                         observation_space=getattr(env, "observation_space", None),
                                                         device=device)
                            return ExplanationArtifacts("ours", key, mask_net=mask_net)
                        except Exception:
                            continue
            raise
    if method in ("statemask", "state_mask", "statemask_r"):
        return train_statemask_explanation(
            env, policy, key, total_timesteps=mask_timesteps, alpha=0.01,
            seed=seed, device=device, logger=logger, cfg=cfg, progress=progress,
            **kwargs,
        )
    if method in ("random", "rand", "uniform"):
        return random_explanation(key, logger=logger, cfg=cfg, seed=seed)
    raise ValueError("Unknown explanation method: %r" % (method,))


# ---------------------------------------------------------------------------
# Fidelity evaluation
# ---------------------------------------------------------------------------

def evaluate_fidelity_for_method(env: Any, policy: Any, artifacts: ExplanationArtifacts,
                                 env_id: str,
                                 K_values: Sequence[float] = DEFAULT_K_VALUES,
                                 n_trajectories: int = DEFAULT_N_TRAJECTORIES,
                                 seeds: Sequence[int] = DEFAULT_SEEDS,
                                 d_max: Optional[float] = None,
                                 deterministic_policy: bool = True,
                                 store_details: bool = False,
                                 progress: bool = False,
                                 logger: Any = None,
                                 cfg: Optional[Dict[str, Any]] = None,
                                 **kwargs: Any) -> Dict[float, Any]:
    """Compute the fidelity score for every ``K`` in ``K_values``.

    Each entry of the returned dict maps ``K`` -> a ``FidelityResult``
    (mean/std across trajectories x seeds).  The sliding window width is
    ``l = L * K`` and the score is ``log(d/d_max) - log(l/L)``.
    """
    key = normalize_env_key(env_id)
    if d_max is None:
        d_max = _cfg_get(cfg, "env", "d_max", default=None)
    if d_max is None and d_max_for is not None:
        try:
            d_max = d_max_for(key)
        except Exception:
            d_max = None

    if evaluate_fidelity_multi_K is None and evaluate_fidelity is None:
        raise RuntimeError("rice.explanation.fidelity is unavailable")

    common = dict(
        env=env,
        policy=policy,
        mask_net=artifacts.mask_net,
        env_id=key,
        d_max=d_max,
        scoring=artifacts.scoring,
        deterministic_policy=deterministic_policy,
        n_trajectories=int(n_trajectories),
        seeds=tuple(int(s) for s in seeds),
        logger=logger,
        store_details=store_details,
        progress=progress,
    )
    common.update({k: v for k, v in kwargs.items()
                   if k in ("action_mode", "restore", "max_tail_steps", "scorer", "rng")})

    if evaluate_fidelity_multi_K is not None:
        return evaluate_fidelity_multi_K(K_values=tuple(K_values), **common)

    results: Dict[float, Any] = {}
    for k in K_values:
        results[float(k)] = evaluate_fidelity(K=k, **common)
    return results


def summarize_fidelity(results: Dict[Any, Any]) -> Dict[str, str]:
    """Format per-K fidelity as ``"mean +- std"`` strings."""
    summary: Dict[str, str] = {}
    for key, value in (results or {}).items():
        mean = getattr(value, "mean", None)
        std = getattr(value, "std", None)
        if mean is None and isinstance(value, dict):
            mean = value.get("mean")
            std = value.get("std")
        label = "K=%.2f" % float(key) if isinstance(key, (int, float)) else str(key)
        if mean is None:
            summary[label] = "n/a"
        elif std is None:
            summary[label] = "%.3f" % float(mean)
        else:
            summary[label] = "%.3f +- %.3f" % (float(mean), float(std))
    return summary


def efficiency_report(env_id: str, artifacts: Dict[str, ExplanationArtifacts],
                      cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mask-net training wall-clock comparison vs the StateMask baseline."""
    key = normalize_env_key(env_id)
    ours = artifacts.get("ours")
    statemask = artifacts.get("statemask")
    ours_time = getattr(ours, "train_time", None)
    baseline_time = getattr(statemask, "train_time", None)

    reduction = None
    if ours_time is not None and baseline_time:
        reduction = (float(baseline_time) - float(ours_time)) / float(baseline_time)

    reference = TABLE4_TIMES.get(key, {})
    return {
        "env_id": key,
        "ours_time": ours_time,
        "statemask_time": baseline_time,
        "ours_samples": getattr(ours, "samples", None),
        "statemask_samples": getattr(statemask, "samples", None),
        "ours_seconds_per_sample": getattr(ours, "seconds_per_sample", None),
        "statemask_seconds_per_sample": getattr(statemask, "seconds_per_sample", None),
        "time_reduction": reduction,
        "time_reduction_percent": (100.0 * reduction) if reduction is not None else None,
        "paper_time_reduction_percent": 100.0 * PAPER_TIME_REDUCTION,
        "reference": reference,
        "reference_reduction_percent": (
            100.0 * (reference["statemask"] - reference["ours"]) / reference["statemask"]
            if reference else None
        ),
    }


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------

def run_fidelity(env_id: str, cfg: Optional[Dict[str, Any]] = None,
                 methods: Sequence[str] = EXPLANATION_METHODS,
                 K_values: Sequence[float] = DEFAULT_K_VALUES,
                 seeds: Sequence[int] = DEFAULT_SEEDS,
                 n_trajectories: int = DEFAULT_N_TRAJECTORIES,
                 mask_timesteps: Optional[int] = None,
                 device: str = DEFAULT_DEVICE,
                 out_dir: Optional[str] = None,
                 logger: Any = None,
                 progress: bool = False,
                 pretrain_timesteps: Optional[int] = None,
                 checkpoint: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 d_max: Optional[float] = None,
                 store_details: bool = False,
                 **kwargs: Any) -> Dict[str, Any]:
    """Run Experiment I (fidelity + mask-training efficiency) for one application."""
    key = normalize_env_key(env_id)
    cfg = cfg or {}
    out_dir = ensure_dir(out_dir or DEFAULT_OUT_DIR)
    if logger is None:
        logger = get_logger("rice.run_fidelity", out_dir=out_dir)

    methods = [str(m).strip().lower() for m in methods]
    results: Dict[str, Any] = {
        "experiment": "fidelity",
        "env_id": key,
        "methods": methods,
        "K_values": [float(k) for k in K_values],
        "seeds": [int(s) for s in seeds],
        "n_trajectories": int(n_trajectories),
        "device": device,
        "d_max": d_max,
        "per_method": {},
        "efficiency": {},
        "errors": {},
    }

    # One environment instance shared by all methods so scoring is comparable.
    env = build_experiment_env(key, cfg=cfg, seed=int(seeds[0]) if seeds else 0, mode="eval")
    try:
        policy = build_or_load_policy(env, key, cfg=cfg, device=device,
                                      checkpoint=checkpoint, logger=logger)
        if pretrain_timesteps:
            try:
                policy = pretrain_target_policy(
                    env, key, total_timesteps=int(pretrain_timesteps),
                    seed=int(seeds[0]) if seeds else 0, device=device,
                    logger=logger, cfg=cfg, progress=progress,
                )
            except Exception as exc:
                results["errors"]["pretrain"] = str(exc)

        artifacts: Dict[str, ExplanationArtifacts] = {}
        for method in methods:
            payload: Dict[str, Any] = {"fidelity": {}, "by_K": {}, "summary": {}}
            try:
                handle = train_explanation(
                    method, env, policy, key, cfg=cfg,
                    seed=int(seeds[0]) if seeds else 0, device=device,
                    logger=logger, mask_timesteps=mask_timesteps,
                    checkpoint_dir=checkpoint_dir, progress=progress,
                )
                artifacts[method] = handle
                payload["explanation"] = handle.to_dict()

                per_K = evaluate_fidelity_for_method(
                    env, policy, handle, key,
                    K_values=K_values, n_trajectories=n_trajectories,
                    seeds=seeds, d_max=d_max, store_details=store_details,
                    progress=progress, logger=logger, cfg=cfg,
                )
                summary = summarize_fidelity(per_K)
                payload["summary"] = summary
                for K, value in per_K.items():
                    payload["by_K"]["%.2f" % float(K)] = {
                        "mean": getattr(value, "mean", None),
                        "std": getattr(value, "std", None),
                        "n_trajectories": getattr(value, "n_trajectories", None),
                        "n_seeds": getattr(value, "n_seeds", None),
                        "d_max": getattr(value, "d_max", None),
                        "per_seed": _jsonable(getattr(value, "per_seed", None)),
                    }
                    if hasattr(value, "to_dict"):
                        try:
                            payload["fidelity"]["%.2f" % float(K)] = _jsonable(
                                value.to_dict(include_details=store_details))
                            continue
                        except Exception:
                            pass
                    payload["fidelity"]["%.2f" % float(K)] = payload["by_K"]["%.2f" % float(K)]
            except Exception as exc:
                payload["error"] = str(exc)
                payload["traceback"] = traceback.format_exc(limit=4)
                results["errors"][method] = str(exc)
            results["per_method"][method] = payload

        if "ours" in artifacts or "statemask" in artifacts:
            results["efficiency"] = efficiency_report(key, artifacts, cfg=cfg)
            results["efficiency"]["paper_percent"] = 100.0 * PAPER_TIME_REDUCTION
    finally:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    # Trend validation: ours ~= statemask, both > random.
    try:
        results["trends"] = check_trends(results)
    except Exception:
        results["trends"] = {}

    report_path = os.path.join(out_dir, "fidelity_%s.json" % key)
    try:
        save_json(_jsonable(results), report_path)
        results["report_path"] = report_path
    except Exception:
        results["report_path"] = None

    try:
        text_path = os.path.join(out_dir, "fidelity_%s.txt" % key)
        with open(text_path, "w") as fh:
            fh.write(format_report(results))
        results["text_path"] = text_path
    except Exception:
        results["text_path"] = None

    return results


def run_fidelity_multi(env_ids: Sequence[str] = DEFAULT_ENVS,
                       cfg: Optional[Dict[str, Any]] = None,
                       out_dir: Optional[str] = None,
                       logger: Any = None,
                       progress: bool = False,
                       **kwargs: Any) -> Dict[str, Any]:
    """Run Experiment I across several applications."""
    out_dir = ensure_dir(out_dir or DEFAULT_OUT_DIR)
    if logger is None:
        logger = get_logger("rice.run_fidelity", out_dir=out_dir)
    combined: Dict[str, Any] = {"experiment": "fidelity", "envs": {}, "errors": {}}
    for env_id in env_ids:
        try:
            combined["envs"][normalize_env_key(env_id)] = run_fidelity(
                env_id, cfg=cfg, out_dir=out_dir, logger=logger,
                progress=progress, **kwargs,
            )
        except Exception as exc:
            combined["errors"][normalize_env_key(env_id)] = str(exc)
    try:
        combined["trends"] = check_trends(combined["envs"])
    except Exception:
        combined["trends"] = {}
    path = os.path.join(out_dir, "fidelity_all.json")
    try:
        save_json(_jsonable(combined), path)
        combined["report_path"] = path
    except Exception:
        combined["report_path"] = None
    return combined


def check_trends(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the paper's qualitative Experiment-I ordering.

    Expectation: our fidelity is comparable to StateMask and both are better
    than Random (the plan explicitly does *not* require Ours > StateMask).
    """
    per_method = report.get("per_method", report.get("envs", {})) or {}
    if not per_method:
        return {}

    def _means(method: str) -> List[float]:
        entry = per_method.get(method)
        if not entry:
            return []
        by_k = entry.get("by_K", {}) or {}
        vals = []
        for item in by_k.values():
            mean = item.get("mean") if isinstance(item, dict) else getattr(item, "mean", None)
            if mean is not None and not (isinstance(mean, float) and mean != mean):
                vals.append(float(mean))
        return vals

    ours = _mean(_means("ours")) if _means("ours") else None
    statemask = _mean(_means("statemask")) if _means("statemask") else None
    random = _mean(_means("random")) if _means("random") else None

    def _safe(value):
        try:
            return None if value is None or value != value else float(value)
        except Exception:
            return None

    ours, statemask, random = _safe(ours), _safe(statemask), _safe(random)
    trends: Dict[str, Any] = {
        "ours_mean": ours,
        "statemask_mean": statemask,
        "random_mean": random,
    }
    if ours is not None and random is not None:
        trends["ours_ge_random"] = ours >= random
    if statemask is not None and random is not None:
        trends["statemask_ge_random"] = statemask >= random
    if ours is not None and statemask is not None:
        trends["ours_comparable_to_statemask"] = abs(ours - statemask) <= max(
            0.25 * max(abs(statemask), abs(ours)), 1e-6)
    eff = report.get("efficiency", {}) or {}
    if eff.get("time_reduction_percent") is not None:
        trends["ours_trains_faster"] = eff["time_reduction_percent"] > 0.0

    # Higher fidelity is better; both comparisons are advisory only.
    trends["passed"] = bool(
        trends.get("ours_ge_random", True)
        and trends.get("statemask_ge_random", True)
    )
    return trends


def format_report(report: Dict[str, Any], decimals: int = 3) -> str:
    """Render a human-readable fidelity + efficiency report."""
    lines: List[str] = []
    env_id = report.get("env_id", report.get("envs", "all"))
    lines.append("=" * 72)
    lines.append("RICE Experiment I - Explanation fidelity & mask-training efficiency")
    lines.append("Env: %s | trajectories: %s | seeds: %s | K values: %s" % (
        env_id, report.get("n_trajectories"), report.get("seeds"), report.get("K_values")))
    lines.append("=" * 72)

    per_method = report.get("per_method") or {}
    if not per_method and isinstance(report.get("envs"), dict):
        for key, value in report["envs"].items():
            lines.append("")
            lines.append(format_report(value, decimals=decimals))
        return "\n".join(lines)

    header = "%-12s" % "method" + "".join("%-20s" % ("K=%.2f" % float(k))
                                          for k in report.get("K_values", []))
    lines.append(header)
    lines.append("-" * len(header))
    for method, payload in per_method.items():
        summary = payload.get("summary", {}) or {}
        row = "%-12s" % method
        for k in report.get("K_values", []):
            label = "K=%.2f" % float(k)
            row += "%-20s" % summary.get(label, "n/a")
        lines.append(row)

    eff = report.get("efficiency") or {}
    if eff:
        lines.append("")
        lines.append("Mask-net training efficiency (fixed sample budget):")
        lines.append("  ours      : %.1fs (%s samples)" % (
            eff.get("ours_time") or float("nan"), eff.get("ours_samples")))
        lines.append("  statemask : %.1fs (%s samples)" % (
            eff.get("statemask_time") or float("nan"), eff.get("statemask_samples")))
        if eff.get("time_reduction_percent") is not None:
            lines.append("  reduction : %.1f%% (paper: %.1f%%)" % (
                eff["time_reduction_percent"], eff.get("paper_time_reduction_percent", 16.8)))
        if eff.get("reference"):
            lines.append("  Table-4 reference: ours %.0fs / statemask %.0fs" % (
                eff["reference"].get("ours", float("nan")),
                eff["reference"].get("statemask", float("nan"))))

    trends = report.get("trends") or {}
    if trends:
        lines.append("")
        lines.append("Trend validation (higher fidelity is better):")
        for key in ("ours_ge_random", "statemask_ge_random",
                    "ours_comparable_to_statemask", "ours_trains_faster", "passed"):
            if key in trends:
                lines.append("  %-30s %s" % (key, trends[key]))

    errors = report.get("errors") or {}
    if errors:
        lines.append("")
        lines.append("Errors:")
        for key, value in errors.items():
            lines.append("  %-20s %s" % (key, value))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RICE Experiment I: fidelity score evaluation (500 trajectories x 3 seeds)."
    )
    parser.add_argument("--env", default=None, help="single application id")
    parser.add_argument("--envs", default=None,
                        help="comma-separated application ids (default: all)")
    parser.add_argument("--config", default=None, help="config name (default: per-env)")
    parser.add_argument("--methods", default="ours,statemask,random",
                        help="comma-separated explanation methods")
    parser.add_argument("--K", "--K-values", dest="K_values",
                        default="0.10,0.20,0.30,0.40",
                        help="comma-separated window fractions")
    parser.add_argument("--trajectories", type=int, default=DEFAULT_N_TRAJECTORIES,
                        help="number of trajectories per seed (default: 500)")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="mask-net training sample budget (default: Table 4)")
    parser.add_argument("--pretrain-timesteps", type=int, default=None,
                        help="optionally pre-train pi for this many steps if no checkpoint")
    parser.add_argument("--checkpoint", default=None, help="target-policy checkpoint")
    parser.add_argument("--checkpoint-dir", default=None, help="mask checkpoint directory")
    parser.add_argument("--d-max", type=float, default=None,
                        help="max single-episode reward for the fidelity normaliser")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--store-details", action="store_true",
                        help="keep per-trajectory fidelity details in the report")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--list-envs", action="store_true")
    return parser


def _split(value: Optional[str], cast=str) -> List[Any]:
    if not value:
        return []
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_envs:
        envs = list(available_envs() or DEFAULT_ENVS)
        print("Available environments: %s" % ", ".join(envs))
        return 0

    if args.env:
        env_ids = [args.env]
    elif args.envs:
        env_ids = _split(args.envs)
    else:
        env_ids = list(DEFAULT_ENVS)

    seeds = _split(args.seeds, int) or list(DEFAULT_SEEDS)
    K_values = _split(args.K_values, float) or list(DEFAULT_K_VALUES)
    methods = _split(args.methods, str) or list(EXPLANATION_METHODS)

    try:
        cfg = get_config(args.config) if args.config else {}
    except Exception:
        cfg = {}

    set_seed(seeds[0] if seeds else 0)

    out_dir = ensure_dir(args.out_dir or DEFAULT_OUT_DIR)
    logger = get_logger("rice.run_fidelity", out_dir=out_dir)

    common = dict(
        methods=methods,
        K_values=K_values,
        seeds=seeds,
        n_trajectories=int(args.trajectories),
        mask_timesteps=args.timesteps,
        device=args.device,
        out_dir=out_dir,
        logger=logger,
        progress=args.progress,
        pretrain_timesteps=args.pretrain_timesteps,
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        d_max=args.d_max,
        store_details=args.store_details,
    )

    if len(env_ids) == 1:
        report = run_fidelity(env_ids[0], cfg=cfg or None, **common)
        print(format_report(report))
        if report.get("report_path"):
            print("\nSaved report to %s" % report["report_path"])
    else:
        report = run_fidelity_multi(env_ids, cfg=cfg or None, **common)
        for key, value in (report.get("envs") or {}).items():
            print(format_report(value))
            print("")
        if report.get("report_path"):
            print("Saved combined report to %s" % report["report_path"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
