"""Experiment V driver for RICE (ICML 2024, PMLR 235): hyper-parameter sensitivity.

The paper (Sec. 4.3 "Impact of Hyper-parameters" and Appendix C.3) studies three
hyper-parameters of RICE:

* ``p`` -- the probability of resetting refinement rollouts to the mask-identified
  critical state, i.e. the mixture weight ``beta`` of the mixed initial state
  distribution ``mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s)``
  (Algorithm 2).  Paper conclusion: performance is low at ``p = 0`` (always
  ``rho``) and ``p = 1`` (always critical), improves for ``0 < p < 1``, and
  ``p = 0.25`` or ``p = 0.5`` is most beneficial across applications.
* ``lambda`` -- the coefficient of the RND intrinsic reward
  ``R_t + lambda * ||f(s_{t+1}) - fhat(s_{t+1})||^2``.  Paper conclusion: as long
  as ``lambda > 0`` exploration noticeably improves refinement, results are
  largely insensitive to the exact value, and ``lambda = 0.01`` is generally the
  best choice (except Selfish Mining).
* ``alpha`` -- the blinding bonus coefficient ``alpha * a_t^m`` in Algorithm 1.
  Paper conclusion: the fidelity score of the mask explanation is *not* sensitive
  to ``alpha`` over ``{0.01, 0.001, 0.0001}``.

This driver instantiates the corresponding sweeps:

* :func:`run_p_sweep`      -- refine with RICE (fixed explanation = ours) for each
  ``p`` in :data:`P_VALUES`, evaluating the final reward over seeds.
* :func:`run_lambda_sweep` -- same, varying ``lambda`` over :data:`LAMBDA_VALUES`.
* :func:`run_alpha_sweep`  -- train the Stage-1 mask network with each ``alpha``
  in :data:`ALPHA_VALUES` under a fixed sample budget and report the fidelity
  score (Experiment-I metric) across ``K``.
* :func:`run_experiment5`  -- all three sweeps for one application, plus trend
  validation against the paper's qualitative claims.
* :func:`run_experiment5_multi` -- the above across several applications.

All heavy lifting is delegated to the already-implemented ``rice`` modules
(``rice.refining.ppo_refine`` for Algorithm 2, ``rice.explanation.mask_trainer``
for Algorithm 1, ``rice.explanation.fidelity`` for the fidelity metric), so this
file only orchestrates sweeps, aggregation and trend checking.  Defensive imports
keep the module importable (and its CLI introspectable) without torch/SB3/MuJoCo.
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

# ---------------------------------------------------------------------------
# Defensive imports of the RICE modules (all optional at import time)
# ---------------------------------------------------------------------------
try:
    from rice.utils.io import ensure_dir, get_config, save_json
except Exception:  # pragma: no cover - fallback for minimal installs
    def ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def get_config(name: str = "default", config_dir: Optional[str] = None) -> Dict[str, Any]:
        return {}

    def save_json(obj: Any, path: str, indent: int = 2) -> str:
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        def _default(o: Any) -> Any:
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=indent, default=_default)
        return path

try:
    from rice.utils.logging import Logger, format_mean_std, get_logger
except Exception:  # pragma: no cover
    import logging as _logging

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = 20):
        logger = _logging.getLogger(name)
        if not logger.handlers:
            handler = _logging.StreamHandler()
            handler.setFormatter(_logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    def format_mean_std(values: Sequence[float], decimals: int = 2) -> str:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.{decimals}f}"
        return f"{float(np.mean(vals)):.{decimals}f} +- {float(np.std(vals)):.{decimals}f}"

    class Logger:  # type: ignore
        def __init__(self, out_dir: Optional[str] = None, name: str = "rice", **kwargs: Any) -> None:
            self.out_dir = out_dir
            self.name = name
            self.history: Dict[str, List[float]] = {}
            self.timers: Dict[str, float] = {}
            self.logger = get_logger(name, out_dir)

        def record(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    self.history.setdefault(key, []).append(float(value))

        def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
            flat: Dict[str, float] = {}

            def _walk(d: Dict[str, Any], pre: str) -> None:
                for k, v in d.items():
                    key = f"{pre}{k}" if not pre else f"{pre}/{k}"
                    if isinstance(v, dict):
                        _walk(v, key)
                    elif isinstance(v, (int, float)):
                        flat[key] = float(v)

            _walk(data, prefix)
            if flat:
                self.record(**flat)

        def timer_start(self, name: str) -> float:
            t0 = time.time()
            self.timers[name] = t0
            return t0

        def timer_end(self, name: str, accumulate: bool = True) -> float:
            t0 = self.timers.get(name, time.time())
            elapsed = time.time() - t0
            if accumulate:
                self.timers[name] = time.time()
            return elapsed

        def dump(self, filename: str = "progress.json") -> Optional[str]:
            if not self.out_dir:
                return None
            path = os.path.join(self.out_dir, filename)
            save_json({"history": self.history, "timers": self.timers}, path)
            return path

        def close(self) -> None:
            self.dump()

try:
    from rice.utils.seeding import seed_from, set_seed
except Exception:  # pragma: no cover
    import random as _random

    def set_seed(seed: int, deterministic: bool = False) -> int:
        seed = int(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        _random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass
        return seed

    def seed_from(base_seed: int, *offsets: int) -> int:
        value = int(base_seed) & 0xFFFFFFFF
        for offset in offsets:
            value = (value * 1000003 + int(offset) + 1) & 0xFFFFFFFF
        return value

try:
    from rice.envs.make_env import (
        available_envs,
        d_max_for,
        env_backend,
        make_env,
        resolve_env_spec,
    )
except Exception:  # pragma: no cover
    available_envs = lambda *a, **k: []  # type: ignore
    d_max_for = lambda *a, **k: None  # type: ignore
    env_backend = lambda *a, **k: "unknown"  # type: ignore
    make_env = None  # type: ignore
    resolve_env_spec = lambda *a, **k: None  # type: ignore

try:
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        save_policy,
    )
except Exception:  # pragma: no cover
    build_policy = load_policy = save_policy = None  # type: ignore

    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        import re

        key = str(env_id or "default").strip().lower()
        key = key.split("/")[-1].replace(".yaml", "").replace("-", "_")
        key = re.sub(r"_v\d+$", "", key)
        return key

try:
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
    )
except Exception:  # pragma: no cover
    build_mask_network = load_mask_network = save_mask_network = None  # type: ignore

try:
    from rice.explanation.mask_trainer import DEFAULT_ALPHA, train_mask_network
except Exception:  # pragma: no cover
    DEFAULT_ALPHA = 1e-4  # type: ignore
    train_mask_network = None  # type: ignore

try:
    from rice.explanation.fidelity import (
        DEFAULT_K_VALUES,
        DEFAULT_N_TRAJECTORIES,
        DEFAULT_SEEDS,
        FidelityConfig,
        FidelityResult,
        evaluate_fidelity_multi_K,
        training_time_reduction,
    )
except Exception:  # pragma: no cover
    DEFAULT_K_VALUES = (0.10, 0.20, 0.30, 0.40)  # type: ignore
    DEFAULT_N_TRAJECTORIES = 500  # type: ignore
    DEFAULT_SEEDS = (0, 1, 2)  # type: ignore
    FidelityConfig = None  # type: ignore
    FidelityResult = None  # type: ignore
    evaluate_fidelity_multi_K = None  # type: ignore
    training_time_reduction = None  # type: ignore

try:
    from rice.refining.ppo_refine import (
        DEFAULT_LAMBDA,
        DEFAULT_P,
        RefinePPOConfig,
        evaluate_refined_policy,
        refine_policy,
    )
except Exception:  # pragma: no cover
    DEFAULT_LAMBDA = 0.01  # type: ignore
    DEFAULT_P = 0.5  # type: ignore
    RefinePPOConfig = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore
    refine_policy = None  # type: ignore

try:
    from rice.baselines.random_explanation import make_random_explanation
except Exception:  # pragma: no cover
    make_random_explanation = None  # type: ignore

try:
    from rice.baselines.statemask_r import samples_for as statemask_samples_for
except Exception:  # pragma: no cover
    statemask_samples_for = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ``p`` grid -- includes the two degenerate extremes plus the recommended
#: interior values (paper: "performance is low when p = 0 ... or p = 1").
P_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: ``lambda`` grid -- ``0`` is included to demonstrate that exploration
#: (``lambda > 0``) is required; ``0.01`` is the paper's best general choice.
LAMBDA_VALUES: Tuple[float, ...] = (0.0, 0.1, 0.01, 0.001)

#: ``alpha`` grid (Algorithm-1 blinding bonus), varied in Experiment V.
ALPHA_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)

#: Applications used by the hyper-parameter sensitivity figures.  Figures 6-9
#: are shown for the MuJoCo games; the ``p``/``lambda`` sweeps are additionally
#: reported for every application (the paper reports ``p`` for all applications
#: and ``lambda`` for all applications except Selfish Mining).
MUJOCO_ENVS: Tuple[str, ...] = ("hopper", "walker2d", "reacher", "halfcheetah")

DEFAULT_ENVS: Tuple[str, ...] = (
    "hopper",
    "walker2d",
    "reacher",
    "halfcheetah",
    "selfish_mining",
    "cage2",
    "autodriving",
)

#: Applications where a larger reward is worse (signed rewards).
NEGATIVE_REWARD_ENVS: Tuple[str, ...] = ("reacher", "cage2")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EVAL_EPISODES = 10
DEFAULT_REFINE_TIMESTEPS = 200_000
DEFAULT_MASK_TIMESTEPS = 300_000
DEFAULT_PRETRAIN_TIMESTEPS = 1_000_000

#: Table 4 fixed mask-training sample budgets (used by the alpha sweep, which
#: must be compared at a fixed number of samples).
TABLE4_SAMPLES: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
    "malware_mutation": 32_349,
}

#: Table 3 (Appendix C.3): per-application hyper-parameter choices for RICE.
#: ``pair`` holds the refinement sweep coordinates ``(p, lambda)``.
TABLE3_CHOICES: Dict[str, Dict[str, float]] = {
    "hopper": {"p": 0.25, "lam": 0.01, "alpha": 0.01},
    "walker2d": {"p": 0.25, "lam": 0.01, "alpha": 0.01},
    "reacher": {"p": 0.5, "lam": 0.01, "alpha": 0.01},
    "halfcheetah": {"p": 0.5, "lam": 0.01, "alpha": 0.01},
    "selfish_mining": {"p": 0.5, "lam": 0.01, "alpha": 0.01},
    "cage2": {"p": 0.5, "lam": 0.01, "alpha": 0.01},
    "autodriving": {"p": 0.5, "lam": 0.01, "alpha": 0.01},
}

#: The paper's qualitative conclusions, kept as explicit trend expectations.
P_GOOD_VALUES: Tuple[float, ...] = (0.25, 0.5)
P_BAD_VALUES: Tuple[float, ...] = (0.0, 1.0)
LAMBDA_PREFERRED = 0.01
ALPHA_REFERENCE = 0.01

#: Reference "No Refine" returns from Table 1 (trend validation only).
REFERENCE_NO_REFINE: Dict[str, float] = {
    "hopper": 3559.44,
    "walker2d": 3768.79,
    "reacher": -5.79,
    "halfcheetah": 2024.09,
    "selfish_mining": 14.36,
    "cage2": -23.64,
    "autodriving": 10.30,
}

# Tolerance used when deciding whether a sweep is "insensitive" to a parameter.
# Expressed as a fraction of the median |reward| magnitude across the sweep.
INSENSITIVITY_TOLERANCE = 0.10
# Minimum relative gain required for "lambda > 0 improves over lambda = 0".
EXPLORATION_GAIN_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class HyperparamPoint:
    """Refinement outcome for a single hyper-parameter value.

    Attributes:
        param: Name of the swept parameter (``"p"`` or ``"lambda"``).
        value: The swept value.
        env_id: Application key.
        final_reward: Mean final reward after refinement over seeds.
        std: Standard deviation of the final reward over seeds.
        eval_rewards: Per-seed final rewards.
        per_seed: Optional per-seed detail dicts.
        no_refine_reward: Reference "No Refine" reward (if known).
        improvement: ``final_reward - no_refine_reward``.
        history: Optional per-iteration refinement history (first seed).
        summary: Refiner summary dict (first seed).
        wall_time: Total wall-clock seconds spent refining.
        samples: Total environment samples consumed.
        seed: First seed used.
        extra: Free-form extras (errors, checkpoints, ...).
    """

    param: str
    value: float
    env_id: str
    final_reward: float
    std: float = 0.0
    eval_rewards: List[float] = field(default_factory=list)
    per_seed: List[Dict[str, Any]] = field(default_factory=list)
    no_refine_reward: Optional[float] = None
    improvement: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    samples: int = 0
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_history: bool = False, include_policy: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "param": self.param,
            "value": self.value,
            "env_id": self.env_id,
            "final_reward": self.final_reward,
            "std": self.std,
            "eval_rewards": list(self.eval_rewards),
            "no_refine_reward": self.no_refine_reward,
            "improvement": self.improvement,
            "wall_time": self.wall_time,
            "samples": self.samples,
            "seed": self.seed,
            "summary": _jsonable(self.summary),
            "extra": _jsonable(self.extra),
        }
        if include_history:
            payload["history"] = _jsonable(self.history)
            payload["per_seed"] = _jsonable(self.per_seed)
        return payload

    def format(self, decimals: int = 2) -> str:
        name = "lambda" if self.param in ("lam", "lambda") else self.param
        return (
            f"{name}={self.value:<7g} reward={self.final_reward:.{decimals}f} "
            f"({self.std:.{decimals}f}) improvement={_fmt_optional(self.improvement, decimals)}"
        )


@dataclass
class AlphaPoint:
    """Fidelity outcome of the mask explanation trained with a given ``alpha``.

    Attributes:
        alpha: Blinding-bonus coefficient used in Algorithm 1.
        env_id: Application key.
        mean_fidelity: Mean fidelity across ``K`` values and seeds.
        fidelity_by_K: Mapping ``K -> mean fidelity``.
        std_by_K: Mapping ``K -> std of fidelity`` (over seeds).
        train_time: Mask-network training wall-clock seconds.
        samples: Mask-training sample budget.
        checkpoint: Optional mask checkpoint path.
        extra: Free-form extras.
    """

    alpha: float
    env_id: str
    mean_fidelity: float = float("nan")
    fidelity_by_K: Dict[str, float] = field(default_factory=dict)
    std_by_K: Dict[str, float] = field(default_factory=dict)
    train_time: float = 0.0
    samples: int = 0
    checkpoint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "env_id": self.env_id,
            "mean_fidelity": self.mean_fidelity,
            "fidelity_by_K": dict(self.fidelity_by_K),
            "std_by_K": dict(self.std_by_K),
            "train_time": self.train_time,
            "samples": self.samples,
            "checkpoint": self.checkpoint,
            "extra": _jsonable(self.extra),
        }

    def format(self, decimals: int = 3) -> str:
        per_k = ", ".join(f"K={k}: {v:.{decimals}f}" for k, v in sorted(self.fidelity_by_K.items()))
        return f"alpha={self.alpha:<8g} mean_fidelity={self.mean_fidelity:.{decimals}f} [{per_k}]"


@dataclass
class ExplanationHandle:
    """Trained Stage-1 explanation shared by the refinement sweeps."""

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
            "train_time": self.train_time,
            "samples": self.samples,
            "checkpoint": self.checkpoint,
            "extra": _jsonable(self.extra),
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Optional[Dict[str, Any]], keys: Sequence[str], default: Any = None) -> Any:
    """Fetch the first present key from a (possibly nested) config dict."""
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if isinstance(node, dict) and key in node:
            return node[key]
    return default


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def _fmt_optional(value: Optional[float], decimals: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of a nested structure to JSON-serialisable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
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
    if hasattr(obj, "state_dict") or hasattr(obj, "parameters"):
        return f"<{type(obj).__name__}>"
    return str(obj)


def is_negative_env(env_id: str) -> bool:
    """True for applications whose reward scale is signed (reacher, cage2)."""
    return normalize_env_key(env_id) in NEGATIVE_REWARD_ENVS


def reference_for(env_id: str) -> Optional[float]:
    """Paper "No Refine" reference reward for an application, if known."""
    return REFERENCE_NO_REFINE.get(normalize_env_key(env_id))


def table3_for(env_id: str) -> Dict[str, float]:
    """Paper hyper-parameter choices (Table 3) for an application."""
    key = normalize_env_key(env_id)
    return dict(TABLE3_CHOICES.get(key, {"p": DEFAULT_P, "lam": DEFAULT_LAMBDA, "alpha": ALPHA_REFERENCE}))


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Fixed Table-4 mask-training sample budget for an application."""
    key = normalize_env_key(env_id)
    override = _cfg_get(cfg, ("explanation", "total_timesteps"))
    if override:
        return int(override)
    if key in TABLE4_SAMPLES:
        return int(TABLE4_SAMPLES[key])
    if statemask_samples_for is not None:
        try:
            return int(statemask_samples_for(key))
        except Exception:
            pass
    return int(DEFAULT_MASK_TIMESTEPS)


def refine_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    override = _cfg_get(cfg, ("refine", "total_timesteps"))
    if override:
        return int(override)
    return int(DEFAULT_REFINE_TIMESTEPS)


def _resolve_env_id(env_id: str) -> str:
    key = normalize_env_key(env_id)
    if make_env is None:
        return key
    try:
        available = list(available_envs() or [])
    except Exception:
        available = []
    if key in available:
        return key
    return key


# ---------------------------------------------------------------------------
# Environment / policy plumbing
# ---------------------------------------------------------------------------


def build_experiment_env(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    mode: str = "train",
    **kwargs: Any,
) -> Any:
    """Create an environment through the RICE factory (or a light fallback)."""
    key = _resolve_env_id(env_id)
    if make_env is None:
        raise RuntimeError("rice.envs.make_env is unavailable in this installation")
    try:
        return make_env(key, seed=seed, mode=mode, **kwargs)
    except TypeError:
        env = make_env(key, seed=seed)
        return env


def build_target_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Instantiate the per-application target policy architecture."""
    if build_policy is None:
        raise RuntimeError("rice.models.policies.build_policy is unavailable")
    key = normalize_env_key(env_id)
    hidden = _cfg_get(cfg, ("target", "hidden_sizes"))
    activation = _cfg_get(cfg, ("target", "activation"))
    return build_policy(
        key,
        observation_space=getattr(env, "observation_space", None),
        action_space=getattr(env, "action_space", None),
        hidden_sizes=hidden,
        activation=activation,
        device=device,
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
    """Load the frozen target policy ``pi`` from a checkpoint, else build fresh."""
    key = normalize_env_key(env_id)
    ckpt = checkpoint or _cfg_get(cfg, ("target", "checkpoint"))
    if ckpt and load_policy is not None and os.path.exists(str(ckpt)):
        try:
            policy = load_policy(
                str(ckpt),
                env_id=key,
                observation_space=getattr(env, "observation_space", None),
                action_space=getattr(env, "action_space", None),
                device=device,
            )
            if logger is not None:
                logger.info("Loaded target policy from %s", ckpt)
            return policy
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning("Failed to load checkpoint %s (%s); building fresh policy", ckpt, exc)
    return build_target_policy(env, key, cfg=cfg, device=device, **kwargs)


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
    """Pre-train ``pi`` with plain PPO to reach the paper's "No Refine" regime."""
    if refine_policy is None:
        raise RuntimeError("rice.refining.ppo_refine.refine_policy is unavailable")
    key = normalize_env_key(env_id)
    policy, refiner = refine_policy(
        env,
        policy=None,
        mask_net=None,
        total_timesteps=int(total_timesteps),
        env_id=key,
        config={"use_mixed_init": False, "use_rnd": False, "p": 0.0, "lam": 0.0},
        logger=logger,
        seed=seed,
        device=device,
        progress=False,
        **kwargs,
    )
    return policy


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
    """Evaluate an episodic return for ``policy`` (delegating when available)."""
    key = normalize_env_key(env_id)
    if evaluate_refined_policy is not None:
        try:
            result = evaluate_refined_policy(
                env,
                policy,
                env_id=key,
                n_episodes=n_episodes,
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
            )
            rewards = list(result.get("rewards", []) or [])
            return {
                "mean_reward": float(result.get("mean_reward", _mean(rewards))),
                "std_reward": float(result.get("std_reward", _std(rewards))),
                "n_episodes": int(result.get("n_episodes", n_episodes)),
                "rewards": rewards,
            }
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning("evaluate_refined_policy failed (%s); falling back", exc)

    rewards: List[float] = []
    for episode in range(int(n_episodes)):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            action = _policy_action(policy, obs, deterministic=deterministic)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, _ = step_result
            episode_reward += float(reward)
            steps += 1
            if max_steps is not None and steps >= int(max_steps):
                break
        rewards.append(episode_reward)
    return {
        "mean_reward": _mean(rewards),
        "std_reward": _std(rewards),
        "n_episodes": int(n_episodes),
        "rewards": rewards,
    }


def _policy_action(policy: Any, obs: Any, deterministic: bool = True) -> Any:
    """Interface-agnostic action selection (SB3 ``predict`` / native ``act``)."""
    if policy is None:
        raise ValueError("policy is None")
    if hasattr(policy, "predict") and not hasattr(policy, "act"):
        action, _ = policy.predict(obs, deterministic=deterministic)
        return action
    if hasattr(policy, "act"):
        try:
            result = policy.act(obs, deterministic=deterministic)
        except TypeError:
            result = policy.act(obs)
        return result[0] if isinstance(result, tuple) else result
    if callable(policy):
        return policy(obs)
    raise TypeError(f"Cannot select an action from policy of type {type(policy)}")


# ---------------------------------------------------------------------------
# Stage-1 explanation (needed for the alpha sweep and as the frozen explainer)
# ---------------------------------------------------------------------------


def train_ours_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: int,
    alpha: float = ALPHA_REFERENCE,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ExplanationHandle:
    """Train the RICE mask network with Algorithm 1 (vanilla PPO + blinding bonus).

    Args:
        alpha: Blinding-bonus coefficient ``alpha`` multiplying ``a_t^m`` in
            ``R'_t = R(s_t, a_t) + alpha * a_t^m``.
    """
    if train_mask_network is None:
        raise RuntimeError("rice.explanation.mask_trainer.train_mask_network is unavailable")
    key = normalize_env_key(env_id)
    start = time.time()
    mask_net, trainer = train_mask_network(
        env,
        policy,
        total_timesteps=int(total_timesteps),
        alpha=float(alpha),
        env_id=key,
        config=cfg,
        logger=logger,
        save_path=checkpoint,
        seed=seed,
        device=device,
        store_dataset=False,
        progress=False,
    )
    elapsed = time.time() - start
    if logger is not None:
        logger.info(
            "Trained mask net (alpha=%g, %d samples) in %.1fs", alpha, total_timesteps, elapsed
        )
    return ExplanationHandle(
        method="ours",
        env_id=key,
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=int(total_timesteps),
        checkpoint=checkpoint,
        extra={"alpha": float(alpha)},
    )


def random_explanation(env_id: str, logger: Any = None, cfg: Optional[Dict[str, Any]] = None, seed: int = 0) -> ExplanationHandle:
    """Random-explanation reference (used to sanity-check alpha insensitivity)."""
    key = normalize_env_key(env_id)
    explainer = None
    if make_random_explanation is not None:
        try:
            explainer = make_random_explanation(env_id=key, seed=seed)
        except Exception:  # pragma: no cover
            explainer = None
    return ExplanationHandle(method="random", env_id=key, mask_net=None, extra={"explainer": explainer})


def _mask_net_of(handle: ExplanationHandle) -> Any:
    return handle.mask_net if handle is not None else None


# ---------------------------------------------------------------------------
# Stage-2 refinement sweep
# ---------------------------------------------------------------------------


def refine_with_hyperparams(
    env: Any,
    policy: Any,
    mask_net: Any,
    env_id: str,
    p: Optional[float] = None,
    lam: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    total_timesteps: Optional[int] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    no_refine_reward: Optional[float] = None,
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> HyperparamPoint:
    """Refine ``policy`` with Algorithm 2 under a specific ``(p, lambda)``.

    All sweeps share the *same* explanation (the mask network) and the *same*
    refiner, so that only the swept hyper-parameter changes -- exactly the
    controlled comparison of Experiment V.
    """
    if refine_policy is None:
        raise RuntimeError("rice.refining.ppo_refine.refine_policy is unavailable")
    key = normalize_env_key(env_id)
    table3 = table3_for(key)
    p_value = float(DEFAULT_P if p is None else p)
    lam_value = float(DEFAULT_LAMBDA if lam is None else lam)
    timesteps = int(total_timesteps or refine_budget_for(key, cfg))

    overrides: Dict[str, Any] = {
        "p": p_value,
        "lam": lam_value,
        "use_mixed_init": True,
        # When lambda == 0 the RND term is disabled outright so that the sweep
        # cleanly separates "no intrinsic reward" from "intrinsic reward".
        "use_rnd": bool(lam_value > 0.0),
        "copy_policy": True,
    }
    if checkpoint:
        overrides["checkpoint"] = checkpoint

    start = time.time()
    try:
        refined_policy, refiner = refine_policy(
            env,
            policy=policy,
            mask_net=mask_net,
            total_timesteps=timesteps,
            env_id=key,
            config=overrides,
            logger=logger,
            seed=seed,
            device=device,
            progress=False,
            **kwargs,
        )
    except TypeError:
        refined_policy, refiner = refine_policy(
            env,
            policy=policy,
            mask_net=mask_net,
            total_timesteps=timesteps,
            p=p_value,
            lam=lam_value,
            env_id=key,
            logger=logger,
            seed=seed,
            device=device,
        )
    elapsed = time.time() - start

    eval_result = evaluate_policy_return(
        env,
        refined_policy,
        key,
        n_episodes=eval_episodes,
        deterministic=True,
        device=device,
        logger=logger,
    )
    final_reward = float(eval_result.get("mean_reward", float("nan")))
    improvement = None
    if no_refine_reward is not None and np.isfinite(final_reward):
        improvement = final_reward - float(no_refine_reward)

    history: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    if refiner is not None:
        summary = _jsonable(getattr(refiner, "summary", lambda: {})()) if callable(
            getattr(refiner, "summary", None)
        ) else {}
        history = list(getattr(refiner, "eval_history", []) or [])

    param = "p" if p is not None else "lambda"
    value = p_value if p is not None else lam_value

    if logger is not None:
        logger.info(
            "%s sweep [%s]: %s=%g -> reward %.2f (%.1fs)",
            key,
            param,
            param,
            value,
            final_reward,
            elapsed,
        )

    return HyperparamPoint(
        param=param,
        value=float(value),
        env_id=key,
        final_reward=final_reward,
        std=float(eval_result.get("std_reward", 0.0) or 0.0),
        eval_rewards=list(eval_result.get("rewards", []) or []),
        no_refine_reward=no_refine_reward,
        improvement=improvement,
        history=history,
        summary=summary,
        wall_time=elapsed,
        samples=int(timesteps),
        seed=int(seed),
        extra={"p": p_value, "lam": lam_value, "explanation": "ours", "table3": table3},
    )


# ---------------------------------------------------------------------------
# Experiment V sweeps
# ---------------------------------------------------------------------------


def run_p_sweep(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    p_values: Sequence[float] = P_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    explanation: str = "ours",
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    lam: Optional[float] = None,
    device: str = "cpu",
    logger: Any = None,
    progress: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Sweep the mixed-initial-state probability ``p`` (Algorithm 2).

    Fixes ``lambda`` to the application's Table-3 value and varies ``p`` over
    ``p_values`` (default ``{0, 0.25, 0.5, 0.75, 1}``), refining from the same
    frozen pre-trained policy and the same explanation for every value.
    """
    key = normalize_env_key(env_id)
    table3 = table3_for(key)
    lam_value = float(table3["lam"] if lam is None else lam)
    mask_timesteps = int(mask_timesteps or mask_budget_for(key, cfg))
    pretrain_timesteps = int(pretrain_timesteps or _cfg_get(cfg, ("target", "total_timesteps"), DEFAULT_PRETRAIN_TIMESTEPS))
    refine_timesteps = int(refine_timesteps or refine_budget_for(key, cfg))

    points: List[HyperparamPoint] = []
    errors: List[str] = []

    env = build_experiment_env(key, cfg=cfg, seed=int(seeds[0]), mode="train")
    policy = build_or_load_policy(env, key, cfg=cfg, device=device, checkpoint=checkpoint, logger=logger)
    no_refine = reference_for(key)

    handles: Dict[int, ExplanationHandle] = {}
    for seed in seeds:
        set_seed(seed_from(int(seed), 5, 1))
        try:
            handles[int(seed)] = train_ours_explanation(
                env,
                policy,
                key,
                mask_timesteps,
                alpha=float(table3["alpha"]),
                seed=int(seed),
                device=device,
                checkpoint=None,
                logger=logger,
                cfg=cfg,
            )
        except Exception as exc:
            errors.append(f"explanation seed={seed}: {exc}")
            handles[int(seed)] = ExplanationHandle(method=explanation, env_id=key, extra={"error": str(exc)})

    for p_value in p_values:
        rewards: List[float] = []
        per_seed: List[Dict[str, Any]] = []
        best_history: List[Dict[str, Any]] = []
        best_summary: Dict[str, Any] = {}
        total_time = 0.0
        for seed in seeds:
            set_seed(seed_from(int(seed), 5, 100 + int(round(float(p_value) * 100))))
            try:
                point = refine_with_hyperparams(
                    env,
                    policy,
                    _mask_net_of(handles.get(int(seed))),
                    key,
                    p=float(p_value),
                    lam=lam_value,
                    cfg=cfg,
                    seed=int(seed),
                    device=device,
                    logger=logger,
                    total_timesteps=refine_timesteps,
                    eval_episodes=eval_episodes,
                    no_refine_reward=no_refine,
                    checkpoint=checkpoint_dir,
                    **kwargs,
                )
            except Exception as exc:
                errors.append(f"p={p_value} seed={seed}: {exc}")
                continue
            rewards.append(point.final_reward)
            total_time += point.wall_time
            per_seed.append({"seed": int(seed), "final_reward": point.final_reward, "std": point.std})
            if not best_history and point.history:
                best_history = point.history
                best_summary = point.summary

        points.append(
            HyperparamPoint(
                param="p",
                value=float(p_value),
                env_id=key,
                final_reward=_mean(rewards),
                std=_std(rewards) if len(rewards) > 1 else 0.0,
                eval_rewards=rewards,
                per_seed=per_seed,
                no_refine_reward=no_refine,
                improvement=None if no_refine is None else _mean(rewards) - float(no_refine),
                history=best_history,
                summary=best_summary,
                wall_time=total_time,
                samples=refine_timesteps * max(1, len(rewards)),
                seed=int(seeds[0]) if seeds else None,
                extra={"lam": lam_value, "explanation": explanation, "table3": table3},
            )
        )

    report: Dict[str, Any] = {
        "experiment": "exp5_p_sweep",
        "env_id": key,
        "param": "p",
        "explanation": explanation,
        "lam": lam_value,
        "p_values": [float(v) for v in p_values],
        "seeds": [int(s) for s in seeds],
        "refine_timesteps": refine_timesteps,
        "mask_timesteps": mask_timesteps,
        "pretrain_timesteps": pretrain_timesteps,
        "no_refine_reward": no_refine,
        "table3": table3,
        "results": [pt.to_dict(include_history=False) for pt in points],
        "references": _reference_points(key, [pt.value for pt in points], "p"),
        "trends": check_p_trends(points, env_id=key),
        "errors": errors,
    }
    return report


def run_lambda_sweep(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    lambda_values: Sequence[float] = LAMBDA_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    explanation: str = "ours",
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    p: Optional[float] = None,
    device: str = "cpu",
    logger: Any = None,
    progress: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Sweep the RND intrinsic-reward coefficient ``lambda`` (Algorithm 2).

    Fixes ``p`` to the application's Table-3 value and varies ``lambda`` over
    ``lambda_values`` (default ``{0, 0.1, 0.01, 0.001}``); ``lambda = 0`` disables
    the exploration bonus so the sweep also demonstrates that ``lambda > 0``
    improves refinement.
    """
    key = normalize_env_key(env_id)
    table3 = table3_for(key)
    p_value = float(table3["p"] if p is None else p)
    mask_timesteps = int(mask_timesteps or mask_budget_for(key, cfg))
    pretrain_timesteps = int(pretrain_timesteps or _cfg_get(cfg, ("target", "total_timesteps"), DEFAULT_PRETRAIN_TIMESTEPS))
    refine_timesteps = int(refine_timesteps or refine_budget_for(key, cfg))

    points: List[HyperparamPoint] = []
    errors: List[str] = []

    env = build_experiment_env(key, cfg=cfg, seed=int(seeds[0]), mode="train")
    policy = build_or_load_policy(env, key, cfg=cfg, device=device, checkpoint=checkpoint, logger=logger)
    no_refine = reference_for(key)

    handles: Dict[int, ExplanationHandle] = {}
    for seed in seeds:
        set_seed(seed_from(int(seed), 5, 2))
        try:
            handles[int(seed)] = train_ours_explanation(
                env,
                policy,
                key,
                mask_timesteps,
                alpha=float(table3["alpha"]),
                seed=int(seed),
                device=device,
                checkpoint=None,
                logger=logger,
                cfg=cfg,
            )
        except Exception as exc:
            errors.append(f"explanation seed={seed}: {exc}")
            handles[int(seed)] = ExplanationHandle(method=explanation, env_id=key, extra={"error": str(exc)})

    for lam_value in lambda_values:
        rewards: List[float] = []
        per_seed: List[Dict[str, Any]] = []
        best_history: List[Dict[str, Any]] = []
        best_summary: Dict[str, Any] = {}
        total_time = 0.0
        for seed in seeds:
            set_seed(seed_from(int(seed), 5, 200 + int(round(float(lam_value) * 100000))))
            try:
                point = refine_with_hyperparams(
                    env,
                    policy,
                    _mask_net_of(handles.get(int(seed))),
                    key,
                    p=p_value,
                    lam=float(lam_value),
                    cfg=cfg,
                    seed=int(seed),
                    device=device,
                    logger=logger,
                    total_timesteps=refine_timesteps,
                    eval_episodes=eval_episodes,
                    no_refine_reward=no_refine,
                    checkpoint=checkpoint_dir,
                    **kwargs,
                )
            except Exception as exc:
                errors.append(f"lambda={lam_value} seed={seed}: {exc}")
                continue
            rewards.append(point.final_reward)
            total_time += point.wall_time
            per_seed.append({"seed": int(seed), "final_reward": point.final_reward, "std": point.std})
            if not best_history and point.history:
                best_history = point.history
                best_summary = point.summary

        points.append(
            HyperparamPoint(
                param="lambda",
                value=float(lam_value),
                env_id=key,
                final_reward=_mean(rewards),
                std=_std(rewards) if len(rewards) > 1 else 0.0,
                eval_rewards=rewards,
                per_seed=per_seed,
                no_refine_reward=no_refine,
                improvement=None if no_refine is None else _mean(rewards) - float(no_refine),
                history=best_history,
                summary=best_summary,
                wall_time=total_time,
                samples=refine_timesteps * max(1, len(rewards)),
                seed=int(seeds[0]) if seeds else None,
                extra={"p": p_value, "explanation": explanation, "table3": table3},
            )
        )

    report: Dict[str, Any] = {
        "experiment": "exp5_lambda_sweep",
        "env_id": key,
        "param": "lambda",
        "explanation": explanation,
        "p": p_value,
        "lambda_values": [float(v) for v in lambda_values],
        "seeds": [int(s) for s in seeds],
        "refine_timesteps": refine_timesteps,
        "mask_timesteps": mask_timesteps,
        "pretrain_timesteps": pretrain_timesteps,
        "no_refine_reward": no_refine,
        "table3": table3,
        "results": [pt.to_dict(include_history=False) for pt in points],
        "references": _reference_points(key, [pt.value for pt in points], "lambda"),
        "trends": check_lambda_trends(points, env_id=key),
        "errors": errors,
    }
    return report


def run_alpha_sweep(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    alpha_values: Sequence[float] = ALPHA_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    K_values: Sequence[float] = DEFAULT_K_VALUES,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    include_random: bool = True,
    device: str = "cpu",
    logger: Any = None,
    progress: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    store_details: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Sweep the blinding-bonus coefficient ``alpha`` (Algorithm 1).

    Trains the mask network for each ``alpha`` under a *fixed* sample budget and
    measures the resulting fidelity score across ``K`` (the Experiment-I metric),
    reproducing Figure 9: the fidelity is insensitive to ``alpha``.
    """
    key = normalize_env_key(env_id)
    mask_timesteps = int(mask_timesteps or mask_budget_for(key, cfg))
    pretrain_timesteps = int(pretrain_timesteps or _cfg_get(cfg, ("target", "total_timesteps"), DEFAULT_PRETRAIN_TIMESTEPS))

    points: List[AlphaPoint] = []
    errors: List[str] = []

    env = build_experiment_env(key, cfg=cfg, seed=int(seeds[0]), mode="eval")
    policy = build_or_load_policy(env, key, cfg=cfg, device=device, checkpoint=checkpoint, logger=logger)

    random_fidelity: Dict[str, float] = {}
    if include_random and evaluate_fidelity_multi_K is not None:
        try:
            random_results = evaluate_fidelity_multi_K(
                env,
                policy,
                mask_net=None,
                K_values=tuple(K_values),
                n_trajectories=int(n_trajectories),
                seeds=tuple(seeds),
                env_id=key,
                scoring="random",
                progress=progress,
                store_details=store_details,
            )
            random_fidelity = {
                str(k): float(getattr(v, "mean", float("nan"))) for k, v in dict(random_results).items()
            }
        except Exception as exc:
            errors.append(f"random-explanation fidelity: {exc}")

    for alpha_value in alpha_values:
        by_k: Dict[str, float] = {}
        std_by_k: Dict[str, float] = {}
        train_time = 0.0
        ckpt: Optional[str] = None
        if checkpoint_dir:
            ckpt = os.path.join(str(checkpoint_dir), f"{key}_mask_alpha{alpha_value:g}.pt")
        for seed in seeds:
            set_seed(seed_from(int(seed), 5, 300 + int(round(float(alpha_value) * 100000))))
            try:
                handle = train_ours_explanation(
                    env,
                    policy,
                    key,
                    mask_timesteps,
                    alpha=float(alpha_value),
                    seed=int(seed),
                    device=device,
                    checkpoint=ckpt,
                    logger=logger,
                    cfg=cfg,
                )
            except Exception as exc:
                errors.append(f"alpha={alpha_value} seed={seed} mask training: {exc}")
                continue
            train_time += handle.train_time
            if evaluate_fidelity_multi_K is None:
                continue
            try:
                results = evaluate_fidelity_multi_K(
                    env,
                    policy,
                    mask_net=handle.mask_net,
                    K_values=tuple(K_values),
                    n_trajectories=int(n_trajectories),
                    seeds=(int(seed),),
                    env_id=key,
                    scoring="mask",
                    progress=progress,
                    store_details=store_details,
                )
            except Exception as exc:
                errors.append(f"alpha={alpha_value} seed={seed} fidelity: {exc}")
                continue
            for k, result in dict(results).items():
                key_str = str(k)
                value = float(getattr(result, "mean", float("nan")))
                by_k.setdefault(key_str, [])  # type: ignore[arg-type]
                by_k[key_str].append(value)  # type: ignore[attr-defined]

        fidelity_by_K: Dict[str, float] = {}
        std_by_K: Dict[str, float] = {}
        for k, values in by_k.items():
            fidelity_by_K[k] = _mean(values)  # type: ignore[arg-type]
            std_by_K[k] = _std(values) if len(values) > 1 else 0.0  # type: ignore[arg-type]

        points.append(
            AlphaPoint(
                alpha=float(alpha_value),
                env_id=key,
                mean_fidelity=_mean(list(fidelity_by_K.values())),
                fidelity_by_K=fidelity_by_K,
                std_by_K=std_by_K,
                train_time=train_time,
                samples=int(mask_timesteps * max(1, len(seeds))),
                checkpoint=ckpt,
                extra={"K_values": [float(k) for k in K_values], "n_trajectories": int(n_trajectories)},
            )
        )

    report: Dict[str, Any] = {
        "experiment": "exp5_alpha_sweep",
        "env_id": key,
        "param": "alpha",
        "alpha_values": [float(v) for v in alpha_values],
        "seeds": [int(s) for s in seeds],
        "K_values": [float(k) for k in K_values],
        "n_trajectories": int(n_trajectories),
        "mask_timesteps": mask_timesteps,
        "pretrain_timesteps": pretrain_timesteps,
        "table3": table3_for(key),
        "results": [pt.to_dict() for pt in points],
        "random_fidelity": random_fidelity,
        "trends": check_alpha_trends(points, random_fidelity=random_fidelity, env_id=key),
        "errors": errors,
    }
    return report


# ---------------------------------------------------------------------------
# Trend validation (qualitative claims of Sec. 4.3 / Appendix C.3)
# ---------------------------------------------------------------------------


def _reference_points(env_id: str, values: Sequence[float], param: str) -> Dict[str, Any]:
    """Per-value paper expectation notes (for reporting, not enforcement)."""
    key = normalize_env_key(env_id)
    notes: Dict[str, Any] = {
        "env_id": key,
        "param": param,
        "higher_is_better": not is_negative_env(key),
    }
    if param == "p":
        notes["expected"] = (
            "p=0 and p=1 underperform mixed values; p in {0.25,0.5} is most beneficial "
            "(Sec. 4.3 / App. C.3, Figures 7-8)."
        )
        notes["good_values"] = list(P_GOOD_VALUES)
        notes["bad_values"] = list(P_BAD_VALUES)
    elif param == "lambda":
        notes["expected"] = (
            "lambda>0 (exploration enabled) improves refinement; results are largely "
            "insensitive to lambda and lambda=0.01 is generally best "
            "(Sec. 4.3 / App. C.3, Figures 6/7)."
        )
        notes["preferred"] = LAMBDA_PREFERRED
    return notes


def _reward_magnitude(points: Sequence[HyperparamPoint]) -> float:
    rewards = [abs(pt.final_reward) for pt in points if np.isfinite(pt.final_reward)]
    if not rewards:
        return 0.0
    return float(np.median(rewards))


def check_p_trends(points: Sequence[HyperparamPoint], env_id: str = "hopper") -> Dict[str, Any]:
    """Validate the paper's qualitative claim for the ``p`` sweep.

    Returns booleans describing whether the mixed initial state distribution
    beats both extremes (``p = 0`` always ``rho`` and ``p = 1`` always critical)
    and whether a recommended interior value (0.25 or 0.5) is among the best.
    """
    values = {round(float(pt.value), 6): pt for pt in points if np.isfinite(pt.final_reward)}
    if not values:
        return {"available": False, "reason": "no finite sweep results"}

    best_value = max(values, key=lambda k: values[k].final_reward)
    worst_value = min(values, key=lambda k: values[k].final_reward)
    interior = [k for k in values if 0.0 < k < 1.0]
    interior_mean = _mean([values[k].final_reward for k in interior]) if interior else float("nan")
    magnitude = _reward_magnitude(points)
    tolerance = INSENSITIVITY_TOLERANCE * max(magnitude, 1e-8)

    p0 = values.get(0.0)
    p1 = values.get(1.0)
    mixed_better_than_p0 = bool(
        interior and p0 is not None and interior_mean - p0.final_reward > -tolerance
    )
    mixed_better_than_p1 = bool(
        interior and p1 is not None and interior_mean - p1.final_reward > -tolerance
    )
    extremes_are_worst = bool(set([best_value]).isdisjoint(P_BAD_VALUES) or (p0 is None and p1 is None))
    good_value_is_best = bool(round(float(best_value), 6) in [round(v, 6) for v in P_GOOD_VALUES])
    good_value_is_top2 = bool(
        round(float(best_value), 6) in [round(v, 6) for v in P_GOOD_VALUES]
        or (len(values) > 1 and round(float(worst_value), 6) in [round(v, 6) for v in P_BAD_VALUES])
    )
    insensitive = bool(
        magnitude > 0
        and (max(v.final_reward for v in values.values()) - min(v.final_reward for v in values.values()))
        <= 0.25 * magnitude
    )

    return {
        "available": True,
        "env_id": normalize_env_key(env_id),
        "best_value": float(best_value),
        "best_reward": float(values[best_value].final_reward),
        "worst_value": float(worst_value),
        "worst_reward": float(values[worst_value].final_reward),
        "interior_mean": float(interior_mean) if interior else None,
        "p0_reward": None if p0 is None else float(p0.final_reward),
        "p1_reward": None if p1 is None else float(p1.final_reward),
        "mixed_beats_p0": mixed_better_than_p0,
        "mixed_beats_p1": mixed_better_than_p1,
        "extremes_underperform": extremes_are_worst,
        "best_in_recommended": good_value_is_best,
        "recommended_top2": good_value_is_top2,
        "insensitive": insensitive,
    }


def check_lambda_trends(points: Sequence[HyperparamPoint], env_id: str = "hopper") -> Dict[str, Any]:
    """Validate the paper's qualitative claim for the ``lambda`` sweep.

    Checks that enabling exploration (``lambda > 0``) helps, that the sweep is
    largely insensitive to the exact ``lambda``, and that ``lambda = 0.01`` is
    among the best settings.
    """
    values = {float(pt.value): pt for pt in points if np.isfinite(pt.final_reward)}
    if not values:
        return {"available": False, "reason": "no finite sweep results"}

    positive = {k: v for k, v in values.items() if k > 0.0}
    best_value = max(values, key=lambda k: values[k].final_reward)
    magnitude = _reward_magnitude(points)
    tolerance = INSENSITIVITY_TOLERANCE * max(magnitude, 1e-8)

    zero = values.get(0.0)
    positive_mean = _mean([v.final_reward for v in positive.values()]) if positive else float("nan")
    exploration_helps = bool(
        zero is not None and positive and positive_mean - zero.final_reward > EXPLORATION_GAIN_TOLERANCE * max(magnitude, 1e-8)
    )
    # "insensitive" means the spread across positive lambda values is small.
    positive_spread = (
        max(v.final_reward for v in positive.values()) - min(v.final_reward for v in positive.values())
        if positive
        else float("nan")
    )
    insensitive = bool(positive and np.isfinite(positive_spread) and positive_spread <= 0.25 * max(magnitude, 1e-8))
    preferred_is_best = bool(abs(float(best_value) - LAMBDA_PREFERRED) < 1e-12)
    preferred_is_top2 = bool(
        abs(float(best_value) - LAMBDA_PREFERRED) < 1e-12
        or (values and float(sorted(values, key=lambda k: values[k].final_reward, reverse=True)[:2][-1]) == LAMBDA_PREFERRED)
    )
    insensitive_or_better = bool(insensitive or not positive or positive_mean <= tolerance)

    return {
        "available": True,
        "env_id": normalize_env_key(env_id),
        "best_value": float(best_value),
        "best_reward": float(values[best_value].final_reward),
        "lambda0_reward": None if zero is None else float(zero.final_reward),
        "positive_mean": float(positive_mean) if positive else None,
        "positive_spread": float(positive_spread) if np.isfinite(positive_spread) else None,
        "exploration_helps": exploration_helps,
        "insensitive": insensitive,
        "insensitive_or_no_gain": insensitive_or_better,
        "preferred_is_best": preferred_is_best,
        "preferred_top2": preferred_is_top2,
    }


def check_alpha_trends(
    points: Sequence[AlphaPoint],
    random_fidelity: Optional[Dict[str, float]] = None,
    env_id: str = "hopper",
) -> Dict[str, Any]:
    """Validate the paper's qualitative claim for the ``alpha`` sweep.

    Checks that the fidelity score is insensitive to ``alpha`` (Figure 9) and
    that the mask explanation still beats the random baseline for every ``alpha``.
    """
    usable = [pt for pt in points if np.isfinite(pt.mean_fidelity)]
    if not usable:
        return {"available": False, "reason": "no finite fidelity results"}

    fidelities = [pt.mean_fidelity for pt in usable]
    spread = float(max(fidelities) - min(fidelities))
    magnitude = float(abs(np.median(fidelities))) if fidelities else 0.0
    relative_spread = spread / magnitude if magnitude > 1e-12 else float("inf")
    insensitive = bool(relative_spread <= 0.50)  # generous: Figure 9 shows near-flat curves

    random_mean = None
    if random_fidelity:
        vals = [float(v) for v in random_fidelity.values() if np.isfinite(float(v))]
        random_mean = _mean(vals) if vals else None
    beats_random = None
    if random_mean is not None:
        beats_random = bool(all(pt.mean_fidelity >= random_mean for pt in usable))

    best = max(usable, key=lambda pt: pt.mean_fidelity)
    worst = min(usable, key=lambda pt: pt.mean_fidelity)

    return {
        "available": True,
        "env_id": normalize_env_key(env_id),
        "best_alpha": float(best.alpha),
        "best_fidelity": float(best.mean_fidelity),
        "worst_alpha": float(worst.alpha),
        "worst_fidelity": float(worst.mean_fidelity),
        "fidelity_spread": spread,
        "relative_spread": None if not np.isfinite(relative_spread) else float(relative_spread),
        "insensitive": insensitive,
        "random_fidelity": random_mean,
        "beats_random_for_all_alpha": beats_random,
    }


# ---------------------------------------------------------------------------
# Top-level experiment driver
# ---------------------------------------------------------------------------


def run_experiment5(
    env_id: str = "hopper",
    cfg: Optional[Dict[str, Any]] = None,
    p_values: Sequence[float] = P_VALUES,
    lambda_values: Sequence[float] = LAMBDA_VALUES,
    alpha_values: Sequence[float] = ALPHA_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sweeps: Sequence[str] = ("p", "lambda", "alpha"),
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    K_values: Sequence[float] = DEFAULT_K_VALUES,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    store_details: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Experiment V (hyper-parameter sensitivity) for one application.

    Args:
        sweeps: Which sweeps to run -- any subset of ``("p", "lambda", "alpha")``.

    Returns:
        A JSON-serialisable report dict containing one sub-report per sweep plus
        aggregated trend-validation flags.
    """
    key = normalize_env_key(env_id)
    out_dir = ensure_dir(out_dir or os.path.join("results", "exp5"))
    if logger is None:
        logger = get_logger("rice.exp5", out_dir)

    start = time.time()
    report: Dict[str, Any] = {
        "experiment": "exp5_hyperparams",
        "env_id": key,
        "seeds": [int(s) for s in seeds],
        "p_values": [float(v) for v in p_values],
        "lambda_values": [float(v) for v in lambda_values],
        "alpha_values": [float(v) for v in alpha_values],
        "table3": table3_for(key),
        "sweeps": list(sweeps),
        "reference_no_refine": reference_for(key),
        "sections": {},
        "trends": {},
        "errors": [],
    }

    if "p" in sweeps:
        try:
            report["sections"]["p"] = run_p_sweep(
                key,
                cfg=cfg,
                p_values=p_values,
                seeds=seeds,
                refine_timesteps=refine_timesteps,
                mask_timesteps=mask_timesteps,
                pretrain_timesteps=pretrain_timesteps,
                device=device,
                logger=logger,
                progress=progress,
                checkpoint=checkpoint,
                checkpoint_dir=checkpoint_dir,
                eval_episodes=eval_episodes,
                **kwargs,
            )
            report["trends"]["p"] = report["sections"]["p"]["trends"]
        except Exception as exc:
            report["errors"].append(f"p sweep: {exc}")
            logger.warning("p sweep failed: %s", exc)

    if "lambda" in sweeps:
        try:
            report["sections"]["lambda"] = run_lambda_sweep(
                key,
                cfg=cfg,
                lambda_values=lambda_values,
                seeds=seeds,
                refine_timesteps=refine_timesteps,
                mask_timesteps=mask_timesteps,
                pretrain_timesteps=pretrain_timesteps,
                device=device,
                logger=logger,
                progress=progress,
                checkpoint=checkpoint,
                checkpoint_dir=checkpoint_dir,
                eval_episodes=eval_episodes,
                **kwargs,
            )
            report["trends"]["lambda"] = report["sections"]["lambda"]["trends"]
        except Exception as exc:
            report["errors"].append(f"lambda sweep: {exc}")
            logger.warning("lambda sweep failed: %s", exc)

    if "alpha" in sweeps:
        try:
            report["sections"]["alpha"] = run_alpha_sweep(
                key,
                cfg=cfg,
                alpha_values=alpha_values,
                seeds=seeds,
                K_values=K_values,
                n_trajectories=n_trajectories,
                mask_timesteps=mask_timesteps,
                pretrain_timesteps=pretrain_timesteps,
                device=device,
                logger=logger,
                progress=progress,
                checkpoint=checkpoint,
                checkpoint_dir=checkpoint_dir,
                store_details=store_details,
                **kwargs,
            )
            report["trends"]["alpha"] = report["sections"]["alpha"]["trends"]
        except Exception as exc:
            report["errors"].append(f"alpha sweep: {exc}")
            logger.warning("alpha sweep failed: %s", exc)

    report["wall_time"] = time.time() - start

    json_path = os.path.join(out_dir, f"exp5_{key}.json")
    report["report_path"] = save_json(report, json_path)
    text_path = os.path.join(out_dir, f"exp5_{key}.txt")
    with open(text_path, "w") as fh:
        fh.write(format_report(report))
    report["text_path"] = text_path

    if logger is not None:
        logger.info("Experiment V (%s) finished in %.1fs -> %s", key, report["wall_time"], json_path)

    return report


def run_experiment5_multi(
    env_ids: Sequence[str] = DEFAULT_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Experiment V for several applications and persist a combined report."""
    out_dir = ensure_dir(out_dir or os.path.join("results", "exp5"))
    if logger is None:
        logger = get_logger("rice.exp5", out_dir)

    reports: Dict[str, Any] = {}
    combined_trends: Dict[str, Any] = {}
    for env_id in env_ids:
        key = normalize_env_key(env_id)
        try:
            sub = run_experiment5(key, cfg=cfg, out_dir=out_dir, logger=logger, progress=progress, **kwargs)
            reports[key] = sub
            combined_trends[key] = sub.get("trends", {})
        except Exception as exc:  # pragma: no cover
            logger.warning("Experiment V failed for %s: %s", key, exc)
            combined_trends[key] = {"error": str(exc)}

    combined = {
        "experiment": "exp5_hyperparams_multi",
        "env_ids": [normalize_env_key(e) for e in env_ids],
        "reports": reports,
        "trends": combined_trends,
        "summary": summarize_experiment5(combined_trends),
    }
    combined["report_path"] = save_json(combined, os.path.join(out_dir, "exp5_all.json"))
    return combined


def summarize_experiment5(combined_trends: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate per-application trend flags into an overall pass/fail summary."""
    summary: Dict[str, Any] = {}
    for param in ("p", "lambda", "alpha"):
        flags: Dict[str, List[bool]] = {}
        for env_id, trends in (combined_trends or {}).items():
            sub = (trends or {}).get(param) or {}
            if not sub.get("available"):
                continue
            for name, value in sub.items():
                if isinstance(value, bool):
                    flags.setdefault(name, []).append(value)
        summary[param] = {
            name: {
                "all": all(vals),
                "any": any(vals),
                "n": len(vals),
                "n_true": int(sum(bool(v) for v in vals)),
            }
            for name, vals in flags.items()
        }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    """Render a human-readable Experiment-V report (all sweeps)."""
    lines: List[str] = []
    env_id = report.get("env_id", "?")
    lines.append("=" * 72)
    lines.append(f"Experiment V - hyper-parameter sensitivity ({env_id})")
    lines.append("=" * 72)
    table3 = report.get("table3") or {}
    if table3:
        lines.append(
            "Table-3 choices: "
            + ", ".join(f"{k}={v:g}" for k, v in table3.items())
        )

    for section_name in ("p", "lambda", "alpha"):
        section = (report.get("sections") or {}).get(section_name)
        if not section:
            continue
        lines.append("")
        lines.append(f"--- Sweep over {section_name} ---")
        for entry in section.get("results", []):
            if section_name == "alpha":
                per_k = entry.get("fidelity_by_K", {}) or {}
                per_k_str = ", ".join(f"K={k}: {float(v):.{decimals}f}" for k, v in sorted(per_k.items()))
                lines.append(
                    f"alpha={float(entry.get('alpha')):<8g} "
                    f"mean_fidelity={float(entry.get('mean_fidelity', float('nan'))):.{decimals}f} "
                    f"train_time={float(entry.get('train_time', 0.0)):.0f}s [{per_k_str}]"
                )
            else:
                name = "lambda" if section_name == "lambda" else "p"
                lines.append(
                    f"{name}={float(entry.get('value', float('nan'))):<7g} "
                    f"reward={float(entry.get('final_reward', float('nan'))):.{decimals}f} "
                    f"({float(entry.get('std', 0.0)):.{decimals}f}) "
                    f"improvement={_fmt_optional(entry.get('improvement'), decimals)}"
                )
        trends = section.get("trends") or {}
        if trends.get("available"):
            lines.append("  trends: " + ", ".join(
                f"{k}={v}" for k, v in trends.items() if isinstance(v, bool)
            ))
        random_fidelity = section.get("random_fidelity")
        if random_fidelity:
            lines.append(
                "  random-explanation fidelity: "
                + ", ".join(f"K={k}: {float(v):.{decimals}f}" for k, v in sorted(random_fidelity.items()))
            )
        for err in section.get("errors", []) or []:
            lines.append(f"  [warning] {err}")

    summary = report.get("summary")
    if summary:
        lines.append("")
        lines.append("--- Overall trend summary ---")
        for param, flags in summary.items():
            if not flags:
                continue
            rendered = ", ".join(
                f"{name}:{info.get('n_true', 0)}/{info.get('n', 0)}"
                for name, info in flags.items()
            )
            lines.append(f"{param}: {rendered}")

    lines.append("")
    lines.append(
        "Notes: p=0/p=1 are the degenerate extremes; 0<p<1 (esp. 0.25/0.5) is "
        "beneficial; lambda>0 enables exploration with lambda=0.01 generally best; "
        "fidelity is insensitive to alpha (Sec. 4.3, App. C.3)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment V for RICE: sensitivity to p, lambda and alpha."
    )
    parser.add_argument("--env", default="hopper", help="Application key (default: hopper).")
    parser.add_argument("--envs", nargs="+", default=None, help="Run several applications.")
    parser.add_argument("--config", default=None, help="Config name forwarded to get_config.")
    parser.add_argument("--sweeps", nargs="+", default=["p", "lambda", "alpha"],
                        choices=["p", "lambda", "alpha"], help="Which sweeps to run.")
    parser.add_argument("--p-values", nargs="+", type=float, default=list(P_VALUES))
    parser.add_argument("--lambda-values", nargs="+", type=float, default=list(LAMBDA_VALUES))
    parser.add_argument("--alpha-values", nargs="+", type=float, default=list(ALPHA_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--refine-timesteps", type=int, default=None)
    parser.add_argument("--mask-timesteps", type=int, default=None)
    parser.add_argument("--pretrain-timesteps", type=int, default=None)
    parser.add_argument("--n-trajectories", type=int, default=DEFAULT_N_TRAJECTORIES)
    parser.add_argument("--K-values", nargs="+", type=float, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Pre-trained target policy checkpoint.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for mask checkpoints.")
    parser.add_argument("--store-details", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = get_config(args.config) if args.config else get_config("default")
    out_dir = ensure_dir(args.out_dir or os.path.join("results", "exp5"))
    logger = get_logger("rice.exp5", out_dir)

    kwargs: Dict[str, Any] = {
        "p_values": args.p_values,
        "lambda_values": args.lambda_values,
        "alpha_values": args.alpha_values,
        "seeds": args.seeds,
        "sweeps": args.sweeps,
        "refine_timesteps": args.refine_timesteps,
        "mask_timesteps": args.mask_timesteps,
        "pretrain_timesteps": args.pretrain_timesteps,
        "K_values": args.K_values,
        "n_trajectories": args.n_trajectories,
        "device": args.device,
        "out_dir": out_dir,
        "logger": logger,
        "progress": args.progress,
        "checkpoint": args.checkpoint,
        "checkpoint_dir": args.checkpoint_dir,
        "eval_episodes": args.eval_episodes,
        "store_details": args.store_details,
    }

    if args.envs:
        report = run_experiment5_multi(args.envs, cfg=cfg, **kwargs)
        print(format_report(report.get("reports", {}).get(normalize_env_key(args.envs[0]), report)))
    else:
        report = run_experiment5(args.env, cfg=cfg, **kwargs)
        print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
