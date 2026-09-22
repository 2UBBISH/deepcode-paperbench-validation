"""Experiment IV — Refining a pre-trained agent of *other* algorithms (RICE, ICML 2024).

Paper reference (Sec. 4.2, "Experiment IV")::

    "To show the versatility of our method, we examine the refining performance when the
     pre-trained agent was trained by other algorithms such as Soft Actor-Critic (SAC).
     First, we obtain a pre-trained SAC agent and then use Generative Adversarial Imitation
     Learning (GAIL) to learn an approximated policy network. We compare the refining
     performance using our method against baseline methods, i.e., PPO fine-tuning,
     StateMask's fine-tuning from critical steps, and Jump-Start Reinforcement Learning.
     In addition, we also include fine-tuning the pre-trained SAC agent with the SAC
     algorithm as a baseline."

Paper reference (Sec. 4.3, "Refining a Pre-trained Agent of Other Algorithms")::

    "we do experiments on refining a SAC agent in the Hopper game. Figure 3 demonstrates
     the advantage of our refining method against other baselines when refining a SAC agent.
     Additionally, we observe that fine-tuning the DRL agent with the SAC algorithm still
     suffers from the training bottleneck while switching to the PPO algorithm provides an
     opportunity to break through the bottleneck."

Pipeline implemented here (Figure 3 of the paper):

    1. ``pretrain_sac_agent``  — train a (deliberately sub-optimal / bottlenecked) SAC agent
       on Hopper and record its learning curve  -> Figure 3 (left).
    2. ``gail_approximate_policy`` — GAIL imitation of the frozen SAC expert to obtain an
       *approximated* policy network :math:`\\pi_G` that RICE/PPO can refine.
    3. ``refine_with_method``  — refine ``pi_G`` with

         * ``ours``            RICE Algorithm 2 (mixed initial distribution + RND bonus),
         * ``ppo_finetune``    plain PPO continuation,
         * ``statemask_r``     StateMask's "always reset to the critical state" fine-tuning,
         * ``jsrl``            Jump-Start RL guided curriculum,
         * ``sac_finetune``    continuing SAC on the pre-trained SAC agent (bottleneck probe).

       -> Figure 3 (right).

Qualitative trends validated by ``check_trends`` (per the reproduction plan, exact numbers
are not required — Figure 3 has no tabulated values):

    * ``ours`` achieves the best final reward of all methods;
    * ``ours`` (and the PPO switch in general) improves substantially over the SAC
      pre-trained agent — switching algorithms breaks the bottleneck;
    * ``sac_finetune`` (staying with SAC) only improves marginally / remains stuck;
    * ``ours`` is at least as good as ``ppo_finetune`` / ``jsrl`` / ``statemask_r``.

Only Hopper is used by the paper for Experiment IV (both because SAC requires continuous
actions and because Figure 3 is Hopper-only); ``DEFAULT_ENV`` reflects that and the driver
warns when another application is requested.

Usage::

    python -m experiments.exp4_sac_agent --env hopper --seeds 0 1 2 --out-dir results/exp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Defensive project imports (the driver must stay importable / introspectable even when
# optional RL backends, torch or MuJoCo are unavailable).
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the package is importable
    from rice.utils.io import ensure_dir, get_config, save_json
except Exception:  # pragma: no cover
    def ensure_dir(path: str) -> str:  # type: ignore[misc]
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def get_config(name: str = "default", **_: Any) -> Dict[str, Any]:  # type: ignore[misc]
        return {}

    def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore[misc]
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=indent, default=str)
        return path


try:  # pragma: no cover
    from rice.utils.logging import Logger, format_mean_std, get_logger
except Exception:  # pragma: no cover
    import logging as _logging

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, **_: Any):  # type: ignore[misc]
        logger = _logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(_logging.StreamHandler())
        logger.setLevel(_logging.INFO)
        return logger

    def format_mean_std(values: Sequence[float], decimals: int = 2) -> str:  # type: ignore[misc]
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.{decimals}f}"
        return f"{np.mean(vals):.{decimals}f} +- {np.std(vals):.{decimals}f}"

    class Logger:  # type: ignore[no-redef]
        def __init__(self, out_dir: Optional[str] = None, **_: Any) -> None:
            self.out_dir = out_dir
            self.history: Dict[str, List[Any]] = {}
            self.timers: Dict[str, float] = {}

        def record(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                self.history.setdefault(key, []).append(value)

        def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
            for key, value in (data or {}).items():
                self.history.setdefault(f"{prefix}{key}", []).append(value)

        def timer_start(self, name: str) -> float:
            start = time.time()
            self.timers[f"_{name}"] = start
            return start

        def timer_end(self, name: str, accumulate: bool = True) -> float:
            elapsed = time.time() - self.timers.get(f"_{name}", time.time())
            self.timers[name] = self.timers.get(name, 0.0) + elapsed if accumulate else elapsed
            return elapsed

        def dump(self, filename: str = "progress.json") -> Optional[str]:
            if not self.out_dir:
                return None
            return save_json({"history": self.history, "timers": self.timers},
                             os.path.join(self.out_dir, filename))

        def close(self) -> None:
            self.dump()


try:  # pragma: no cover
    from rice.utils.seeding import seed_from, set_seed
except Exception:  # pragma: no cover
    import random as _random

    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore[misc]
        _random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass
        return int(seed)

    def seed_from(base_seed: int, *offsets: int) -> int:  # type: ignore[misc]
        value = int(base_seed)
        for off in offsets:
            value = (value * 1000003 + int(off) + 7) % (2 ** 31 - 1)
        return value


try:  # pragma: no cover
    from rice.envs.make_env import (
        d_max_for,
        env_backend,
        env_metadata,
        make_env,
        resolve_env_spec,
    )
except Exception:  # pragma: no cover
    d_max_for = None  # type: ignore[assignment]
    env_backend = None  # type: ignore[assignment]
    env_metadata = None  # type: ignore[assignment]
    make_env = None  # type: ignore[assignment]
    resolve_env_spec = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        save_policy,
    )
except Exception:  # pragma: no cover
    build_policy = None  # type: ignore[assignment]
    load_policy = None  # type: ignore[assignment]
    save_policy = None  # type: ignore[assignment]

    def normalize_env_key(env_id: Any) -> str:  # type: ignore[misc]
        key = str(env_id or "default").strip().lower()
        for suffix in ("-v0", "-v1", "-v2", "-v3", "-v4", "-v5"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key.replace("-", "_").replace(" ", "_")


try:  # pragma: no cover
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
    )
except Exception:  # pragma: no cover
    build_mask_network = None  # type: ignore[assignment]
    load_mask_network = None  # type: ignore[assignment]
    save_mask_network = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.explanation.mask_trainer import DEFAULT_ALPHA, train_mask_network
except Exception:  # pragma: no cover
    DEFAULT_ALPHA = 1e-4  # type: ignore[assignment]
    train_mask_network = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.refining.ppo_refine import (
        DEFAULT_LAMBDA,
        DEFAULT_P,
        RefinePPOConfig,
        evaluate_refined_policy,
        refine_policy,
    )
except Exception:  # pragma: no cover
    DEFAULT_LAMBDA = 0.01  # type: ignore[assignment]
    DEFAULT_P = 0.5  # type: ignore[assignment]
    RefinePPOConfig = None  # type: ignore[assignment]
    evaluate_refined_policy = None  # type: ignore[assignment]
    refine_policy = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.baselines.sac_finetune import (
        SACFinetuneConfig,
        SACFinetuner,
        is_sb3_sac,
        make_sac_finetuner,
        sac_finetune_policy,
        train_sac_agent,
    )
except Exception:  # pragma: no cover
    SACFinetuneConfig = None  # type: ignore[assignment]
    SACFinetuner = None  # type: ignore[assignment]
    is_sb3_sac = None  # type: ignore[assignment]
    make_sac_finetuner = None  # type: ignore[assignment]
    sac_finetune_policy = None  # type: ignore[assignment]
    train_sac_agent = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.baselines.gail import (
        GAILConfig,
        GAILRefiner,
        GAILTrainer,
        approximate_policy_with_gail,
        collect_expert_demonstrations,
        make_gail,
        train_gail,
    )
except Exception:  # pragma: no cover
    GAILConfig = None  # type: ignore[assignment]
    GAILRefiner = None  # type: ignore[assignment]
    GAILTrainer = None  # type: ignore[assignment]
    approximate_policy_with_gail = None  # type: ignore[assignment]
    collect_expert_demonstrations = None  # type: ignore[assignment]
    make_gail = None  # type: ignore[assignment]
    train_gail = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.baselines.ppo_finetune import (
        DEFAULT_FINETUNE_LR,
        ppo_finetune_policy,
    )
except Exception:  # pragma: no cover
    DEFAULT_FINETUNE_LR = 1e-4  # type: ignore[assignment]
    ppo_finetune_policy = None  # type: ignore[assignment]


try:  # pragma: no cover
    from rice.baselines.statemask_r import (
        refine_from_critical_state,
        samples_for as statemask_samples_for,
        train_statemask_network,
    )
except Exception:  # pragma: no cover
    refine_from_critical_state = None  # type: ignore[assignment]
    train_statemask_network = None  # type: ignore[assignment]

    def statemask_samples_for(env_id: str, default: int = 300_000) -> int:  # type: ignore[misc]
        return int(default)


try:  # pragma: no cover
    from rice.baselines.jsrl import train_jsrl
except Exception:  # pragma: no cover
    train_jsrl = None  # type: ignore[assignment]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_ENV = "hopper"
DEFAULT_ENVS: Tuple[str, ...] = (DEFAULT_ENV,)

#: Experiment IV refines the *approximated* policy with these methods (Sec. 4.2).
REFINING_METHODS: Tuple[str, ...] = (
    "ours",
    "ppo_finetune",
    "statemask_r",
    "jsrl",
    "sac_finetune",
)
EXPLANATION_METHODS: Tuple[str, ...] = ("ours", "statemask", "random")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EVAL_EPISODES = 10

#: SAC pre-training budget (Figure 3 left).  The paper does not tabulate the number of SAC
#: steps; SB3-style defaults + the RICE refinement budget are used.
DEFAULT_SAC_TIMESTEPS = 300_000
#: GAIL imitation budget used to obtain the approximated policy network.
DEFAULT_GAIL_TIMESTEPS = 200_000
#: Refinement budget shared by every refining method (fair comparison).
DEFAULT_REFINE_TIMESTEPS = 200_000
#: Stage-1 mask-network training budget (Table 4 / StateMask budget for Hopper).
DEFAULT_MASK_TIMESTEPS = 300_000
#: Number of intermediate evaluation points of the SAC pre-training curve (Fig. 3 left).
DEFAULT_PRETRAIN_EVAL_POINTS = 5

TABLE4_SAMPLES: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
}

#: Figure 3 has no tabulated numbers, so trend validation is *ordinal* (the reproduction
#: plan validates trends rather than exact values for this experiment).
EXPECTED_ORDERING: Tuple[str, ...] = (
    "ours",
    "ppo_finetune",
    "jsrl",
    "statemask_r",
    "sac_finetune",
)

#: Qualitative reference notes from Sec. 4.3 ("Refining a Pre-trained Agent of Other
#: Algorithms") — used by ``check_trends`` to decide whether a run reproduces the paper.
REFERENCE_NOTES: Dict[str, str] = {
    "figure": "Figure 3 (SAC Agent Refining Performance in Hopper Game)",
    "bottleneck": (
        "fine-tuning the DRL agent with the SAC algorithm still suffers from the training "
        "bottleneck while switching to the PPO algorithm provides an opportunity to break "
        "through the bottleneck"
    ),
    "ordering": "ours is the best refining method when refining a SAC agent in Hopper",
    "notes": (
        "Figure 3 reports the SAC pre-training curve (left) and the refining curves of the "
        "different methods (right); no exact numeric values are tabulated, so only the "
        "qualitative ordering / effect directions are validated."
    ),
}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _cfg_get(cfg: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Nested ``"a.b.c"`` lookup into a (possibly ``None``) config dict."""
    if not cfg:
        return default
    node: Any = cfg
    for part in str(key).split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if vals else float("nan")


def is_negative_env(env_id: str) -> bool:
    """Signed-reward applications (Cage Challenge 2, Reacher) — 'improvement' semantics."""
    key = normalize_env_key(env_id)
    return key in ("reacher", "cage2", "cage")


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Fixed Stage-1 mask-training sample budget (Table 4) for an application."""
    key = normalize_env_key(env_id)
    if key.startswith("sparse_"):
        key = key[len("sparse_"):]
    configured = _cfg_get(cfg, "explanation.total_timesteps", None) or _cfg_get(
        cfg, "explanation.samples", None
    )
    if configured:
        return int(configured)
    if key in TABLE4_SAMPLES:
        return int(TABLE4_SAMPLES[key])
    return int(statemask_samples_for(env_id, default=DEFAULT_MASK_TIMESTEPS))


def reference_for(method: str, env_id: str) -> Optional[float]:
    """Paper reference value for ``(method, env_id)``.

    Experiment IV is reported graphically (Figure 3) and the paper does not tabulate its
    values, so this always returns ``None`` — trend validation is ordinal here.
    """
    return None


# --------------------------------------------------------------------------------------
# Environment / policy construction
# --------------------------------------------------------------------------------------
def build_experiment_env(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    mode: str = "train",
    **kwargs: Any,
) -> Any:
    """Create the Experiment-IV environment through the RICE env factory.

    Observation normalization follows the config (Walker2d / HalfCheetah per App. C.2);
    SAC/GAIL/refinement all see the *same* wrapper stack, keeping the comparison fair.
    """
    if make_env is None:  # pragma: no cover
        raise RuntimeError("rice.envs.make_env.make_env is unavailable")
    normalize = kwargs.pop("normalize", _cfg_get(cfg, "env.normalize_obs", None))
    max_steps = kwargs.pop(
        "max_episode_steps",
        _cfg_get(cfg, "env.max_episode_steps", None),
    )
    env = make_env(
        env_id,
        seed=seed,
        normalize=normalize,
        mode=mode,
        max_episode_steps=max_steps,
        **kwargs,
    )
    return env


def build_target_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Instantiate the per-application policy architecture (SB3 MlpPolicy by default)."""
    if build_policy is None:  # pragma: no cover
        raise RuntimeError("rice.models.policies.build_policy is unavailable")
    hidden = kwargs.pop("hidden_sizes", _cfg_get(cfg, "target.hidden_sizes", None))
    try:
        return build_policy(
            env_id=env_id,
            observation_space=getattr(env, "observation_space", None),
            action_space=getattr(env, "action_space", None),
            kind="policy",
            hidden_sizes=hidden,
            device=device,
            **kwargs,
        )
    except Exception:  # pragma: no cover - fall back to a bare policy construction
        return build_policy(env_id=env_id, kind="policy", device=device, **kwargs)


def build_or_load_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Load a saved policy (checkpoint argument or config default), else build a fresh one."""
    path = checkpoint or _cfg_get(cfg, "target.checkpoint", None)
    if path and load_policy is not None and os.path.exists(str(path)):
        try:
            policy = load_policy(
                str(path),
                env_id=env_id,
                observation_space=getattr(env, "observation_space", None),
                action_space=getattr(env, "action_space", None),
                kind="policy",
                device=device,
            )
            if logger is not None:
                logger.info("[exp4] loaded policy from %s", path)
            return policy
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning("[exp4] failed to load policy from %s: %s", path, exc)
    return build_target_policy(env, env_id, cfg=cfg, device=device, **kwargs)


# --------------------------------------------------------------------------------------
# Step 1 — pre-train the SAC agent (Figure 3, left)
# --------------------------------------------------------------------------------------
@dataclass
class SACPretrainResult:
    """Outcome of pre-training the SAC agent that Experiment IV starts from."""

    env_id: str
    total_timesteps: int = 0
    policy: Any = None
    trainer: Any = None
    eval_history: List[Dict[str, Any]] = field(default_factory=list)
    final_reward: float = float("nan")
    std: float = float("nan")
    wall_time: float = 0.0
    checkpoint: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_history: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "env_id": self.env_id,
            "total_timesteps": int(self.total_timesteps),
            "final_reward": None if not np.isfinite(self.final_reward) else float(self.final_reward),
            "std": None if not np.isfinite(self.std) else float(self.std),
            "wall_time": float(self.wall_time),
            "checkpoint": self.checkpoint,
            "summary": _jsonable(self.summary),
            "extra": _jsonable(self.extra),
        }
        if include_history:
            data["eval_history"] = _jsonable(self.eval_history)
        return data

    def format(self, decimals: int = 2) -> str:
        return (
            f"SAC pre-training ({self.env_id}): {self.total_timesteps} steps, "
            f"final reward {_safe_fmt(self.final_reward, decimals)} "
            f"(+- {_safe_fmt(self.std, decimals)}), {self.wall_time:.1f}s"
        )


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of metrics/results into JSON-serialisable objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return str(obj)


def _safe_fmt(value: Any, decimals: int = 2) -> str:
    try:
        fval = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(fval):
        return "n/a"
    return f"{fval:.{decimals}f}"


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
    """Evaluate a policy's undiscounted episodic return (Table 1 quantity)."""
    if evaluate_refined_policy is not None:
        try:
            result = evaluate_refined_policy(
                env,
                policy,
                env_id=env_id,
                n_episodes=n_episodes,
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
            )
            if isinstance(result, dict) and result.get("rewards") is not None:
                rewards = list(result.get("rewards") or [])
                return {
                    "mean_reward": float(result.get("mean_reward", _mean(rewards))),
                    "std_reward": float(result.get("std_reward", _std(rewards))),
                    "n_episodes": int(result.get("n_episodes", len(rewards)) or len(rewards)),
                    "rewards": rewards,
                }
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning("[exp4] evaluate_refined_policy failed (%s); local eval", exc)

    rewards: List[float] = []
    for episode in range(int(n_episodes)):
        obs, _ = _reset(env, seed=kwargs.get("seed", episode))
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = _policy_action(policy, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = _step(env, action)
            total += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if max_steps is not None and steps >= int(max_steps):
                break
        rewards.append(total)
    return {
        "mean_reward": _mean(rewards),
        "std_reward": _std(rewards),
        "n_episodes": len(rewards),
        "rewards": rewards,
    }


def _reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], result[1] or {}
        return result[0], {}
    return result, {}


def _step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    result = env.step(action)
    if isinstance(result, tuple):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated), bool(truncated), info or {}
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), False, info or {}
    raise RuntimeError("unsupported env.step return signature")


def _policy_action(policy: Any, observation: Any, deterministic: bool = True) -> Any:
    """Interface-agnostic action selection (SB3 ``predict`` / native ``act`` / callable)."""
    if policy is None:
        raise RuntimeError("policy is None")
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
    raise RuntimeError(f"cannot obtain an action from policy of type {type(policy)!r}")


def pretrain_sac_agent(
    env: Any,
    env_id: str = DEFAULT_ENV,
    total_timesteps: int = DEFAULT_SAC_TIMESTEPS,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    eval_points: int = DEFAULT_PRETRAIN_EVAL_POINTS,
    eval_episodes: int = 3,
    resolve: bool = True,
    **kwargs: Any,
) -> SACPretrainResult:
    """Pre-train a SAC agent and record its learning curve (Figure 3, left half).

    Training is performed in ``eval_points`` chunks so intermediate evaluations can be
    recorded; if the trainer does not support incremental ``train`` calls the whole budget
    is run at once (the curve then holds a single point, which is still reportable).

    With ``resolve=True`` the checkpoints maximize the *bottleneck* aspect of the paper's
    setup: refinement has to break out of a locally-optimal (sub-optimal) pre-trained
    policy.  The paper's Figure 3 (left) shows a SAC training curve that plateaus below the
    refined performance, which is exactly what a fixed-budget SAC run produces here.
    """
    if train_sac_agent is None:  # pragma: no cover
        raise RuntimeError("rice.baselines.sac_finetune.train_sac_agent is unavailable")

    if logger is not None:
        logger.info(
            "[exp4] pre-training SAC on %s for %d timesteps (seed=%d)",
            env_id,
            int(total_timesteps),
            int(seed),
        )

    set_seed(seed)
    start = time.time()
    policy: Any = None
    trainer: Any = None
    history: List[Dict[str, Any]] = []
    extra: Dict[str, Any] = {}

    sac_kwargs = dict(kwargs)
    sac_kwargs.setdefault("env_id", env_id)
    sac_kwargs.setdefault("seed", seed)
    sac_kwargs.setdefault("device", device)
    sac_kwargs.setdefault("logger", logger)
    sac_kwargs.setdefault("progress", False)

    n_chunks = max(1, int(eval_points))
    chunk = max(1, int(total_timesteps) // n_chunks)
    trained = 0

    try:
        policy, trainer = train_sac_agent(env, total_timesteps=chunk, **sac_kwargs)
        trained = chunk
        eval_hist = evaluate_policy_return(
            env,
            getattr(trainer, "policy", policy),
            env_id,
            n_episodes=eval_episodes,
            deterministic=True,
            device=device,
            logger=logger,
        )
        history.append({"timesteps": trained, "mean_reward": eval_hist["mean_reward"],
                        "std_reward": eval_hist["std_reward"]})
        for _ in range(n_chunks - 1):
            trainer.train(total_timesteps=chunk)
            trained += chunk
            eval_hist = evaluate_policy_return(
                env,
                getattr(trainer, "policy", policy),
                env_id,
                n_episodes=eval_episodes,
                deterministic=True,
                device=device,
                logger=logger,
            )
            history.append({"timesteps": trained, "mean_reward": eval_hist["mean_reward"],
                            "std_reward": eval_hist["std_reward"]})
            if logger is not None:
                logger.info(
                    "[exp4] SAC %d/%d steps, reward %.2f",
                    trained,
                    int(total_timesteps),
                    eval_hist["mean_reward"],
                )
    except Exception as exc:  # pragma: no cover - incremental path unavailable
        extra["incremental_error"] = str(exc)
        if logger is not None:
            logger.warning("[exp4] incremental SAC training failed (%s); single-shot run", exc)
        policy, trainer = train_sac_agent(env, total_timesteps=total_timesteps, **sac_kwargs)
        trained = int(total_timesteps)
        eval_hist = evaluate_policy_return(
            env,
            getattr(trainer, "policy", policy),
            env_id,
            n_episodes=eval_episodes,
            deterministic=True,
            device=device,
            logger=logger,
        )
        history.append({"timesteps": trained, "mean_reward": eval_hist["mean_reward"],
                        "std_reward": eval_hist["std_reward"]})

    wall = time.time() - start
    eval_policy = getattr(trainer, "policy", policy)
    final_eval = evaluate_policy_return(
        env,
        eval_policy,
        env_id,
        n_episodes=max(eval_episodes, 5),
        deterministic=True,
        device=device,
        logger=logger,
    )

    saved: Optional[str] = None
    if checkpoint and save_policy is not None:
        try:
            ensure_dir(os.path.dirname(os.path.abspath(str(checkpoint))))
            saved = save_policy(policy, str(checkpoint), env_id=env_id, kind="sac_expert")
        except Exception as exc:  # pragma: no cover
            extra["save_error"] = str(exc)

    summary: Dict[str, Any] = {}
    if trainer is not None and hasattr(trainer, "summary"):
        try:
            summary = dict(trainer.summary())
        except Exception:  # pragma: no cover
            summary = {}

    result = SACPretrainResult(
        env_id=env_id,
        total_timesteps=int(trained),
        policy=eval_policy,
        trainer=trainer,
        eval_history=history,
        final_reward=float(final_eval["mean_reward"]),
        std=float(final_eval["std_reward"]),
        wall_time=wall,
        checkpoint=saved,
        summary=summary,
        extra=extra,
    )
    if logger is not None:
        logger.info("[exp4] %s", result.format())
    return result


# --------------------------------------------------------------------------------------
# Step 2 — GAIL approximation of the SAC expert (Sec. 4.2, Experiment IV)
# --------------------------------------------------------------------------------------
@dataclass
class GAILApproximationResult:
    """The approximated policy network π_G learnt from the frozen SAC expert via GAIL."""

    env_id: str
    policy: Any = None
    trainer: Any = None
    expert_samples: int = 0
    total_timesteps: int = 0
    eval_reward: float = float("nan")
    std: float = float("nan")
    wall_time: float = 0.0
    checkpoint: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "expert_samples": int(self.expert_samples),
            "total_timesteps": int(self.total_timesteps),
            "eval_reward": None if not np.isfinite(self.eval_reward) else float(self.eval_reward),
            "std": None if not np.isfinite(self.std) else float(self.std),
            "wall_time": float(self.wall_time),
            "checkpoint": self.checkpoint,
            "summary": _jsonable(self.summary),
            "extra": _jsonable(self.extra),
        }

    def format(self, decimals: int = 2) -> str:
        return (
            f"GAIL approximation ({self.env_id}): {self.total_timesteps} steps on "
            f"{self.expert_samples} expert transitions, reward "
            f"{_safe_fmt(self.eval_reward, decimals)} (+- {_safe_fmt(self.std, decimals)}), "
            f"{self.wall_time:.1f}s"
        )


def _build_approx_policy(env: Any, env_id: str, device: str = "cpu") -> Any:
    """Fresh policy network used as GAIL's initial π_G (falls back to the SAC actor shape)."""
    if build_policy is None:  # pragma: no cover
        raise RuntimeError("rice.models.policies.build_policy is unavailable")
    try:
        return build_policy(
            env_id=env_id,
            observation_space=getattr(env, "observation_space", None),
            action_space=getattr(env, "action_space", None),
            kind="policy",
            device=device,
        )
    except Exception:  # pragma: no cover
        return build_policy(env_id=env_id, kind="policy", device=device)


def gail_approximate_policy(
    env: Any,
    expert_policy: Any,
    env_id: str = DEFAULT_ENV,
    total_timesteps: int = DEFAULT_GAIL_TIMESTEPS,
    expert_timesteps: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    eval_episodes: int = 5,
    **kwargs: Any,
) -> GAILApproximationResult:
    """Learn the approximated policy network π_G from the pre-trained SAC agent via GAIL.

    The SAC agent is treated as a *black box*: GAIL first collects demonstrations from it
    and then trains π_G adversarially.  π_G is the policy that every refining method then
    starts from, ensuring the only difference between refiners is the refining method.
    """
    if approximate_policy_with_gail is None and train_gail is None:  # pragma: no cover
        raise RuntimeError("rice.baselines.gail is unavailable")

    if logger is not None:
        logger.info("[exp4] approximating the SAC expert with GAIL (budget=%d)", int(total_timesteps))

    set_seed(seed)
    start = time.time()
    extra: Dict[str, Any] = {}
    policy: Any = None
    trainer: Any = None
    expert_samples = 0

    gail_kwargs: Dict[str, Any] = dict(kwargs)
    gail_kwargs.setdefault("env_id", env_id)
    gail_kwargs.setdefault("seed", seed)
    gail_kwargs.setdefault("device", device)
    gail_kwargs.setdefault("logger", logger)
    if expert_timesteps is not None:
        gail_kwargs.setdefault("expert_timesteps", int(expert_timesteps))

    try:
        if approximate_policy_with_gail is not None:
            policy = approximate_policy_with_gail(
                env,
                expert_policy=expert_policy,
                total_timesteps=int(total_timesteps),
                **gail_kwargs,
            )
        else:  # pragma: no cover - fallback path
            policy, trainer = train_gail(
                env,
                expert_policy=expert_policy,
                total_timesteps=int(total_timesteps),
                **gail_kwargs,
            )
            if hasattr(trainer, "expert_data") and trainer.expert_data is not None:
                obs_arr = np.asarray(trainer.expert_data[0])
                expert_samples = int(obs_arr.shape[0])
    except Exception as exc:  # pragma: no cover - GAIL unavailable ⇒ degrade gracefully
        extra["gail_error"] = str(exc)
        if logger is not None:
            logger.warning("[exp4] GAIL approximation failed (%s); using a fresh π_G", exc)
        policy = _build_approx_policy(env, env_id, device=device)

    if trainer is None and GAILRefiner is not None:
        try:  # attach a trainer object for bookkeeping when only a policy is returned
            trainer = make_gail(env=env, expert_policy=expert_policy, env_id=env_id,
                                seed=seed, device=device, logger=logger)
        except Exception:  # pragma: no cover
            trainer = None

    if expert_samples == 0 and collect_expert_demonstrations is not None:
        try:
            obs, act, _info = collect_expert_demonstrations(
                env,
                expert_policy,
                n_timesteps=int(expert_timesteps or DEFAULT_GAIL_TIMESTEPS // 2),
                seed=seed,
                deterministic=True,
                device=device,
            )
            expert_samples = int(np.asarray(obs).shape[0])
        except Exception:  # pragma: no cover
            expert_samples = 0

    wall = time.time() - start
    eval_hist = evaluate_policy_return(
        env, policy, env_id, n_episodes=eval_episodes, deterministic=True,
        device=device, logger=logger,
    )

    saved: Optional[str] = None
    if checkpoint and save_policy is not None:
        try:
            ensure_dir(os.path.dirname(os.path.abspath(str(checkpoint))))
            saved = save_policy(policy, str(checkpoint), env_id=env_id, kind="gail_policy")
        except Exception as exc:  # pragma: no cover
            extra["save_error"] = str(exc)

    summary: Dict[str, Any] = {}
    if trainer is not None and hasattr(trainer, "summary"):
        try:
            summary = dict(trainer.summary())
        except Exception:  # pragma: no cover
            summary = {}

    result = GAILApproximationResult(
        env_id=env_id,
        policy=policy,
        trainer=trainer,
        expert_samples=expert_samples,
        total_timesteps=int(total_timesteps),
        eval_reward=float(eval_hist["mean_reward"]),
        std=float(eval_hist["std_reward"]),
        wall_time=wall,
        checkpoint=saved,
        summary=summary,
        extra=extra,
    )
    if logger is not None:
        logger.info("[exp4] %s", result.format())
    return result


# --------------------------------------------------------------------------------------
# Stage 1 — explanation (mask network) used by every refiner
# --------------------------------------------------------------------------------------
@dataclass
class ExplanationHandle:
    """A trained Stage-1 explanation (mask network) shared by all refining methods."""

    method: str = "ours"
    env_id: str = DEFAULT_ENV
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
            "train_time": float(self.train_time),
            "samples": int(self.samples),
            "checkpoint": self.checkpoint,
            "extra": _jsonable(self.extra),
        }


def train_ours_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: int,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ExplanationHandle:
    """Stage 1 / Algorithm 1: vanilina-PPO mask training with the blinding bonus α·a_t^m."""
    if train_mask_network is None:  # pragma: no cover
        raise RuntimeError("rice.explanation.mask_trainer.train_mask_network is unavailable")
    if logger is not None:
        logger.info("[exp4] training mask network (ours) for %d samples", int(total_timesteps))
    set_seed(seed)
    start = time.time()
    extra: Dict[str, Any] = {}
    try:
        mask_net, trainer = train_mask_network(
            env,
            policy,
            total_timesteps=int(total_timesteps),
            alpha=float(alpha),
            env_id=env_id,
            logger=logger,
            seed=seed,
            device=device,
            store_dataset=False,
        )
    except TypeError:  # pragma: no cover - older signature without some kwargs
        mask_net, trainer = train_mask_network(
            env, policy, total_timesteps=int(total_timesteps), alpha=float(alpha),
            env_id=env_id, logger=logger,
        )
    wall = time.time() - start

    saved: Optional[str] = None
    if checkpoint and save_mask_network is not None:
        try:
            ensure_dir(os.path.dirname(os.path.abspath(str(checkpoint))))
            saved = save_mask_network(mask_net, str(checkpoint), env_id=env_id)
        except Exception as exc:  # pragma: no cover
            extra["save_error"] = str(exc)

    return ExplanationHandle(
        method="ours",
        env_id=env_id,
        mask_net=mask_net,
        trainer=trainer,
        train_time=wall,
        samples=int(total_timesteps),
        checkpoint=saved,
        extra=extra,
    )


def train_statemask_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: int,
    alpha: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ExplanationHandle:
    """StateMask's primal-dual mask training (comparison explanation for Exp III-style runs)."""
    if train_statemask_network is None:  # pragma: no cover
        raise RuntimeError("rice.baselines.statemask_r.train_statemask_network is unavailable")
    if logger is not None:
        logger.info("[exp4] training mask network (StateMask) for %d samples", int(total_timesteps))
    set_seed(seed)
    start = time.time()
    extra: Dict[str, Any] = {}
    mask_net, trainer = train_statemask_network(
        env,
        policy,
        total_timesteps=int(total_timesteps),
        alpha=float(alpha),
        env_id=env_id,
        logger=logger,
        seed=seed,
        device=device,
        store_dataset=False,
    )
    wall = time.time() - start

    saved: Optional[str] = None
    if checkpoint and save_mask_network is not None:
        try:
            ensure_dir(os.path.dirname(os.path.abspath(str(checkpoint))))
            saved = save_mask_network(mask_net, str(checkpoint), env_id=env_id)
        except Exception as exc:  # pragma: no cover
            extra["save_error"] = str(exc)

    return ExplanationHandle(
        method="statemask",
        env_id=env_id,
        mask_net=mask_net,
        trainer=trainer,
        train_time=wall,
        samples=int(total_timesteps),
        checkpoint=saved,
        extra=extra,
    )


def random_explanation(env_id: str, logger: Any = None, cfg: Optional[Dict[str, Any]] = None,
                       seed: int = 0, **_: Any) -> ExplanationHandle:
    """Random explanation baseline (no mask network ⇒ uninformative importance scores)."""
    return ExplanationHandle(
        method="random",
        env_id=env_id,
        mask_net=None,
        trainer=None,
        train_time=0.0,
        samples=0,
        checkpoint=None,
        extra={"note": "Random explanation: uniformly random critical states"},
    )


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
    """Dispatch Stage-1 explanation training by method name."""
    name = str(method).strip().lower()
    budget = int(mask_timesteps or mask_budget_for(env_id, cfg))
    ckpt = None
    if checkpoint_dir:
        ckpt = os.path.join(str(checkpoint_dir), f"{normalize_env_key(env_id)}_{name}_mask.pt")

    if name in ("ours", "rice", "mask", "mask_network"):
        return train_ours_explanation(
            env, policy, env_id, budget,
            alpha=float(_cfg_get(cfg, "explanation.alpha", DEFAULT_ALPHA) or DEFAULT_ALPHA),
            seed=seed, device=device, checkpoint=ckpt, logger=logger, cfg=cfg,
        )
    if name in ("statemask", "state_mask", "statemask_explanation"):
        return train_statemask_explanation(
            env, policy, env_id, budget,
            alpha=float(_cfg_get(cfg, "baselines.statemask.alpha_init", 0.01) or 0.01),
            seed=seed, device=device, checkpoint=ckpt, logger=logger, cfg=cfg,
        )
    if name in ("random", "none"):
        return random_explanation(env_id, logger=logger, cfg=cfg, seed=seed)
    raise ValueError(f"unknown explanation method: {method!r}")


# --------------------------------------------------------------------------------------
# Step 3 — refinement (Figure 3, right)
# --------------------------------------------------------------------------------------
@dataclass
class SACRefineResult:
    """Outcome of refining the GAIL-approximated SAC policy with one refining method."""

    method: str
    env_id: str = DEFAULT_ENV
    explanation: str = "ours"
    final_reward: float = float("nan")
    std: float = float("nan")
    eval_rewards: List[float] = field(default_factory=list)
    no_refine_reward: float = float("nan")
    improvement: float = float("nan")
    history: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    samples: int = 0
    seed: int = 0
    policy: Any = None
    refiner: Any = None
    checkpoint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def format(self, decimals: int = 2) -> str:
        return (
            f"{self.method:<14} reward {_safe_fmt(self.final_reward, decimals)} "
            f"(+- {_safe_fmt(self.std, decimals)})  "
            f"improvement {_safe_fmt(self.improvement, decimals)}  "
            f"[{self.wall_time:.1f}s]"
        )

    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "method": self.method,
            "env_id": self.env_id,
            "explanation": self.explanation,
            "final_reward": None if not np.isfinite(self.final_reward) else float(self.final_reward),
            "std": None if not np.isfinite(self.std) else float(self.std),
            "eval_rewards": _jsonable(self.eval_rewards),
            "no_refine_reward": (
                None if not np.isfinite(self.no_refine_reward) else float(self.no_refine_reward)
            ),
            "improvement": None if not np.isfinite(self.improvement) else float(self.improvement),
            "summary": _jsonable(self.summary),
            "wall_time": float(self.wall_time),
            "samples": int(self.samples),
            "seed": int(self.seed),
            "checkpoint": self.checkpoint,
            "extra": _jsonable(self.extra),
        }
        if include_history:
            data["history"] = _jsonable(self.history)
        return data


def _refine_config_for(env_id: str, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> Any:
    """Build a ``RefinePPOConfig`` (Algorithm 2) from the merged YAML config."""
    if RefinePPOConfig is None:  # pragma: no cover
        return None
    params: Dict[str, Any] = {
        "p": _cfg_get(cfg, "refine.p", DEFAULT_P),
        "lam": _cfg_get(cfg, "refine.lam", DEFAULT_LAMBDA),
        "lr": _cfg_get(cfg, "refine.lr", None),
        "gamma": _cfg_get(cfg, "refine.gamma", None),
        "gae_lambda": _cfg_get(cfg, "refine.gae_lambda", None),
        "clip_range": _cfg_get(cfg, "refine.clip_range", None),
        "n_epochs": _cfg_get(cfg, "refine.n_epochs", None),
        "batch_size": _cfg_get(cfg, "refine.batch_size", None),
        "env_id": env_id,
    }
    params = {k: v for k, v in params.items() if v is not None}
    params.update(overrides)
    try:
        return RefinePPOConfig.from_dict(params)
    except Exception:  # pragma: no cover
        try:
            return RefinePPOConfig(**params)
        except Exception:
            return None


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
    p: Optional[float] = None,
    lam: Optional[float] = None,
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Refine ``policy`` with one method; returns ``(refined_policy, trainer, info)``."""
    name = str(method).strip().lower()
    budget = int(total_timesteps or _cfg_get(cfg, "refine.total_timesteps", None)
                 or DEFAULT_REFINE_TIMESTEPS)
    info: Dict[str, Any] = {"method": name, "samples": budget}
    trainer: Any = None
    refined: Any = policy

    if name in ("ours", "rice", "ppo_refine"):
        if refine_policy is None:  # pragma: no cover
            raise RuntimeError("rice.refining.ppo_refine.refine_policy is unavailable")
        ref_cfg = _refine_config_for(
            env_id,
            cfg,
            p=float(p if p is not None else (_cfg_get(cfg, "refine.p", DEFAULT_P) or DEFAULT_P)),
            lam=float(lam if lam is not None else (_cfg_get(cfg, "refine.lam", DEFAULT_LAMBDA) or DEFAULT_LAMBDA)),
        )
        refined, trainer = refine_policy(
            env,
            policy=policy,
            mask_net=mask_net,
            total_timesteps=budget,
            env_id=env_id,
            config=ref_cfg,
            logger=logger,
            seed=seed,
            device=device,
            progress=False,
            evaluate=False,
        )
        info["config"] = _jsonable(ref_cfg.to_dict()) if hasattr(ref_cfg, "to_dict") else {}
        if checkpoint and save_policy is not None:
            try:
                ensure_dir(os.path.dirname(os.path.abspath(str(checkpoint))))
                refined = save_policy(refined, str(checkpoint), env_id=env_id,
                                      kind="refined") and refined or refined
            except Exception as exc:  # pragma: no cover
                info["save_error"] = str(exc)

    elif name in ("ppo_finetune", "ppo", "finetune"):
        if ppo_finetune_policy is None:  # pragma: no cover
            raise RuntimeError("rice.baselines.ppo_finetune.ppo_finetune_policy is unavailable")
        refined, trainer = ppo_finetune_policy(
            env,
            policy=policy,
            total_timesteps=budget,
            env_id=env_id,
            config=None,
            logger=logger,
            seed=seed,
            device=device,
            progress=False,
            evaluate=False,
            lr=float(_cfg_get(cfg, "baselines.ppo_finetune.lr", DEFAULT_FINETUNE_LR)
                     or DEFAULT_FINETUNE_LR),
        )
        info["lr"] = float(_cfg_get(cfg, "baselines.ppo_finetune.lr", DEFAULT_FINETUNE_LR)
                           or DEFAULT_FINETUNE_LR)

    elif name in ("statemask_r", "state_mask_r", "statemask"):
        if refine_from_critical_state is None:  # pragma: no cover
            raise RuntimeError("rice.baselines.statemask_r.refine_from_critical_state is unavailable")
        refined, trainer = refine_from_critical_state(
            env,
            policy=policy,
            mask_net=mask_net,
            total_timesteps=budget,
            env_id=env_id,
            config=None,
            logger=logger,
            seed=seed,
            device=device,
            progress=False,
            evaluate=False,
            use_rnd=False,
        )

    elif name in ("jsrl", "jumpstart", "jump_start"):
        if train_jsrl is None:  # pragma: no cover
            raise RuntimeError("rice.baselines.jsrl.train_jsrl is unavailable")
        refined, trainer = train_jsrl(
            env,
            guided_policy=policy,
            policy=None,
            total_timesteps=budget,
            env_id=env_id,
            config=None,
            logger=logger,
            seed=seed,
            device=device,
            progress=False,
            evaluate=False,
        )

    elif name in ("sac_finetune", "sac", "sacft"):
        if sac_finetune_policy is None:  # pragma: no cover
            raise RuntimeError("rice.baselines.sac_finetune.sac_finetune_policy is unavailable")
        refined, trainer = sac_finetune_policy(
            env,
            policy=policy,
            total_timesteps=budget,
            env_id=env_id,
            config=None,
            logger=logger,
            seed=seed,
            device=device,
            progress=False,
            evaluate=False,
            lr=float(_cfg_get(cfg, "baselines.sac_finetune.lr", None) or None) or None,
        )

    else:
        raise ValueError(f"unknown refining method: {method!r}")

    return refined, trainer, info


def refine_with_method(
    method: str,
    env: Any,
    policy: Any,
    mask_net: Any,
    env_id: str = DEFAULT_ENV,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    total_timesteps: Optional[int] = None,
    no_refine_reward: Optional[float] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    explanation: str = "ours",
    checkpoint: Optional[str] = None,
    p: Optional[float] = None,
    lam: Optional[float] = None,
    **kwargs: Any,
) -> SACRefineResult:
    """Refine ``policy`` with ``method`` and package the evaluation into a result object.

    Every refining method starts from the *same* π_G obtained by GAIL (Experiment IV) and,
    where needed, uses the *same* explanation (mask network), fulfilling the paper's
    fairness requirement ("all the refining methods use the same explanation ... to ensure
    a fair comparison", Sec. 4.2).
    """
    if logger is not None:
        logger.info("[exp4] refining with %s (env=%s, seed=%d)", method, env_id, seed)
    set_seed(seed)
    start = time.time()
    extra: Dict[str, Any] = {}

    if method == "sac_finetune" and refiner_startswith_sac(policy):
        extra["backend"] = "sb3_sac"

    refined, trainer, info = _run_refiner(
        method,
        env,
        policy,
        mask_net,
        env_id,
        cfg=cfg,
        seed=seed,
        device=device,
        logger=logger,
        total_timesteps=total_timesteps,
        p=p,
        lam=lam,
        checkpoint=checkpoint,
        **kwargs,
    )
    wall = time.time() - start
    extra.update(info)

    eval_hist = evaluate_policy_return(
        env,
        refined,
        env_id,
        n_episodes=eval_episodes,
        deterministic=True,
        device=device,
        logger=logger,
    )

    history: List[Dict[str, Any]] = []
    if trainer is not None:
        for attr in ("eval_history", "history", "train_history"):
            cand = getattr(trainer, attr, None)
            if isinstance(cand, list) and cand:
                history = _jsonable(cand)
                break

    summary: Dict[str, Any] = {}
    if trainer is not None and hasattr(trainer, "summary"):
        try:
            summary = dict(trainer.summary())
        except Exception:  # pragma: no cover
            summary = {}

    final_reward = float(eval_hist["mean_reward"])
    no_refine = float(no_refine_reward) if no_refine_reward is not None else float("nan")
    improvement = (
        final_reward - no_refine if np.isfinite(no_refine) else float("nan")
    )

    result = SACRefineResult(
        method=str(method),
        env_id=env_id,
        explanation=str(explanation),
        final_reward=final_reward,
        std=float(eval_hist["std_reward"]),
        eval_rewards=list(eval_hist["rewards"]),
        no_refine_reward=no_refine,
        improvement=improvement,
        history=history,
        summary=summary,
        wall_time=wall,
        samples=int(total_timesteps or _cfg_get(cfg, "refine.total_timesteps", None)
                    or DEFAULT_REFINE_TIMESTEPS),
        seed=int(seed),
        policy=refined,
        refiner=trainer,
        checkpoint=checkpoint,
        extra=extra,
    )
    if logger is not None:
        logger.info("[exp4] %s", result.format())
    return result


def refiner_startswith_sac(policy: Any) -> bool:
    """True when the given object looks like an SB3 SAC model (for backend bookkeeping)."""
    if policy is None:
        return False
    if is_sb3_sac is not None:
        try:
            return bool(is_sb3_sac(policy))
        except Exception:  # pragma: no cover
            pass
    return type(policy).__name__.upper().startswith("SAC")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def _aggregate(results: Sequence[SACRefineResult]) -> Dict[str, Any]:
    """Aggregate per-seed refine results into mean/std per method."""
    by_method: Dict[str, List[SACRefineResult]] = {}
    for res in results:
        by_method.setdefault(res.method, []).append(res)

    aggregated: Dict[str, Any] = {}
    for method, runs in by_method.items():
        rewards = [r.final_reward for r in runs]
        improvements = [r.improvement for r in runs]
        aggregated[method] = {
            "n_seeds": len(runs),
            "mean_reward": _mean(rewards),
            "std_reward": _std(rewards),
            "mean_improvement": _mean(improvements),
            "std_improvement": _std(improvements),
            "final_rewards": rewards,
            "wall_time": float(sum(r.wall_time for r in runs)),
            "formatted": format_mean_std(rewards, decimals=2),
            "per_seed": [r.to_dict() for r in runs],
        }
    return aggregated


def run_experiment4(
    env_id: str = DEFAULT_ENV,
    cfg: Optional[Dict[str, Any]] = None,
    methods: Sequence[str] = REFINING_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sac_timesteps: Optional[int] = None,
    gail_timesteps: Optional[int] = None,
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    use_gail: bool = True,
    pretrain_sac: bool = True,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    mask_seed: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Experiment IV end-to-end for one application (Hopper by the paper's design).

    Steps: (1) pre-train SAC (Figure 3 left), (2) GAIL-approximate a policy network,
    (3) train the Stage-1 mask explanation once per seed, (4) refine π_G with each method
    and evaluate (Figure 3 right).
    """
    if normalize_env_key(env_id) != "hopper":
        if logger is not None:
            logger.warning(
                "[exp4] Experiment IV is reported on Hopper in the paper (Figure 3); "
                "running it on %s is an extension.",
                env_id,
            )
    if cfg is None:
        try:
            cfg = get_config(env_id)
        except Exception:  # pragma: no cover
            cfg = {}

    logger = logger or get_logger("rice.exp4", out_dir=out_dir)
    out_dir = ensure_dir(out_dir or _cfg_get(cfg, "results_dir", "results") + "/exp4")
    chk_dir = ensure_dir(checkpoint_dir or os.path.join(out_dir, "checkpoints"))

    sac_steps = int(sac_timesteps or _cfg_get(cfg, "exp4.sac_timesteps", None) or DEFAULT_SAC_TIMESTEPS)
    gail_steps = int(gail_timesteps or _cfg_get(cfg, "exp4.gail_timesteps", None) or DEFAULT_GAIL_TIMESTEPS)
    refine_steps = int(refine_timesteps or _cfg_get(cfg, "refine.total_timesteps", None)
                       or DEFAULT_REFINE_TIMESTEPS)
    mask_steps = int(mask_timesteps or mask_budget_for(env_id, cfg))

    env = build_experiment_env(env_id, cfg=cfg, seed=int(seeds[0]) if seeds else 0, mode="train")

    report: Dict[str, Any] = {
        "experiment": "exp4_sac_agent",
        "env_id": env_id,
        "methods": list(methods),
        "seeds": list(seeds),
        "budgets": {
            "sac_pretrain": sac_steps,
            "gail": gail_steps,
            "mask": mask_steps,
            "refine": refine_steps,
        },
        "use_gail": bool(use_gail),
        "backend": env_backend(env) if env_backend is not None else "unknown",
        "metadata": env_metadata(env_id, probe=False) if env_metadata is not None else {},
        "reference": dict(REFERENCE_NOTES),
        "expected_ordering": list(EXPECTED_ORDERING),
        "sac_pretrain": {},
        "gail": {},
        "results": [],
        "aggregated": {},
        "started_at": time.time(),
    }

    # -- Step 1/2 -------------------------------------------------------------------
    base_seed = int(seeds[0]) if seeds else 0
    sac_result: Optional[SACPretrainResult] = None
    if sac_result is None and (pretrain_sac or checkpoint or _cfg_get(cfg, "target.checkpoint", None)):
        try:
            sac_ckpt = checkpoint or os.path.join(chk_dir, f"{normalize_env_key(env_id)}_sac_expert.pt")
            sac_result = pretrain_sac_agent(
                env,
                env_id=env_id,
                total_timesteps=sac_steps,
                seed=seed_from(base_seed, 11),
                device=device,
                logger=logger,
                cfg=cfg,
                checkpoint=sac_ckpt,
                eval_episodes=max(1, min(eval_episodes, 5)),
            )
            report["sac_pretrain"] = sac_result.to_dict()
            if logger is not None and progress:
                for point in sac_result.eval_history:
                    logger.info("[exp4] SAC curve %s", point)
        except Exception as exc:  # pragma: no cover - SAC unavailable
            report["sac_pretrain"] = {"error": str(exc)}
            if logger is not None:
                logger.warning("[exp4] SAC pre-training failed: %s", exc)

    expert_policy = sac_result.policy if sac_result is not None else None

    approx_policy: Any = None
    if use_gail and expert_policy is not None:
        try:
            gail_result = gail_approximate_policy(
                env,
                expert_policy,
                env_id=env_id,
                total_timesteps=gail_steps,
                seed=seed_from(base_seed, 13),
                device=device,
                logger=logger,
                cfg=cfg,
                checkpoint=os.path.join(chk_dir, f"{normalize_env_key(env_id)}_gail_policy.pt"),
                eval_episodes=max(1, min(eval_episodes, 5)),
            )
            report["gail"] = gail_result.to_dict()
            approx_policy = gail_result.policy
        except Exception as exc:  # pragma: no cover - GAIL unavailable
            report["gail"] = {"error": str(exc)}
            if logger is not None:
                logger.warning("[exp4] GAIL approximation failed: %s", exc)

    if approx_policy is None:
        # Fallback: refine the SAC expert's actor directly (still "another algorithm").
        approx_policy = expert_policy
        if approx_policy is None:
            approx_policy = build_or_load_policy(env, env_id, cfg=cfg, device=device, logger=logger)
        report["gail"].setdefault("note", "GAIL unavailable ⇒ refining the SAC/PPO policy directly")

    no_refine = evaluate_policy_return(
        env, approx_policy, env_id, n_episodes=eval_episodes, deterministic=True,
        device=device, logger=logger,
    )
    report["no_refine"] = {
        "mean_reward": float(no_refine["mean_reward"]),
        "std_reward": float(no_refine["std_reward"]),
        "eval_rewards": list(no_refine["rewards"]),
    }
    if logger is not None:
        logger.info(
            "[exp4] no-refine (pre-trained SAC/GAIL policy) reward %.2f +- %.2f",
            no_refine["mean_reward"],
            no_refine["std_reward"],
        )

    # -- Step 3/4 -------------------------------------------------------------------
    results: List[SACRefineResult] = []
    errors: List[Dict[str, str]] = []
    for seed in seeds:
        try:
            set_seed(seed)
            explanation = train_explanation(
                "ours",
                env,
                approx_policy,
                env_id,
                cfg=cfg,
                seed=mask_seed,
                device=device,
                logger=logger,
                mask_timesteps=mask_steps,
                checkpoint_dir=chk_dir,
            )
        except Exception as exc:  # pragma: no cover - no mask ⇒ ours degrades to random-like
            errors.append({"stage": "explanation", "seed": str(seed), "error": str(exc)})
            if logger is not None:
                logger.warning("[exp4] mask training failed for seed %s: %s", seed, exc)
            continue

        for method in methods:
            try:
                result = refine_with_method(
                    method,
                    env,
                    approx_policy,
                    explanation.mask_net,
                    env_id=env_id,
                    cfg=cfg,
                    seed=seed,
                    device=device,
                    logger=logger,
                    total_timesteps=refine_steps,
                    no_refine_reward=float(no_refine["mean_reward"]),
                    eval_episodes=eval_episodes,
                    explanation="ours",
                    checkpoint=os.path.join(
                        chk_dir, f"{normalize_env_key(env_id)}_{method}_seed{seed}.pt"
                    ),
                )
                result.extra["explanation_train_time"] = explanation.train_time
                result.extra["mask_samples"] = explanation.samples
                results.append(result)
                if logger is not None:
                    logger.info("[exp4] seed %d %s", seed, result.format())
            except Exception as exc:  # pragma: no cover - keep the sweep alive
                errors.append({"stage": f"refine:{method}", "seed": str(seed), "error": str(exc)})
                if logger is not None:
                    logger.warning("[exp4] refining with %s failed (seed %s): %s", method, seed, exc)

    report["results"] = [r.to_dict(include_history=True) for r in results]
    report["aggregated"] = _aggregate(results)
    report["errors"] = errors
    report["finished_at"] = time.time()
    report["wall_time"] = report["finished_at"] - report["started_at"]
    report["trends"] = check_trends(report)

    # -- Persist --------------------------------------------------------------------
    try:
        save_json(report, os.path.join(out_dir, f"exp4_{normalize_env_key(env_id)}.json"))
        save_json({"experiment": "exp4", "env_ids": [env_id], "report": report},
                  os.path.join(out_dir, "exp4_all.json"))
        with open(os.path.join(out_dir, f"exp4_{normalize_env_key(env_id)}.txt"), "w") as fh:
            fh.write(format_report(report))
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("[exp4] failed to write results: %s", exc)

    try:
        env.close()
    except Exception:  # pragma: no cover
        pass

    return report


def run_experiment4_multi(
    env_ids: Sequence[str] = DEFAULT_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Experiment IV for several applications (the paper reports Hopper only)."""
    reports: Dict[str, Any] = {}
    for env_id in env_ids:
        env_cfg = cfg
        if env_cfg is None:
            try:
                env_cfg = get_config(env_id)
            except Exception:  # pragma: no cover
                env_cfg = {}
        reports[env_id] = run_experiment4(env_id, cfg=env_cfg, **kwargs)
    return reports


# --------------------------------------------------------------------------------------
# Trend validation / reporting
# --------------------------------------------------------------------------------------
def check_trends(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the qualitative Figure-3 trends (ordering only; no numeric targets)."""
    agg: Dict[str, Any] = report.get("aggregated") or {}
    means = {
        method: float(agg[method]["mean_reward"])
        for method in agg
        if np.isfinite(agg[method].get("mean_reward", float("nan")))
    }
    no_refine = float((report.get("no_refine") or {}).get("mean_reward", float("nan")))

    checks: List[Dict[str, Any]] = []
    notes: List[str] = []

    if not means:
        return {"available": False, "reason": "no refine results recorded", "checks": []}

    best_method = max(means, key=means.get)
    checks.append({
        "check": "ours_is_best",
        "passed": best_method == "ours",
        "detail": f"best={best_method} ({means[best_method]:.2f})",
    })

    if np.isfinite(no_refine):
        checks.append({
            "check": "ours_beats_no_refine",
            "passed": means.get("ours", float("-inf")) > no_refine,
            "detail": f"ours={means.get('ours', float('nan')):.2f} vs no_refine={no_refine:.2f}",
        })
        if "sac_finetune" in means:
            sac_impr = means["sac_finetune"] - no_refine
            ours_impr = means.get("ours", float("-inf")) - no_refine
            checks.append({
                "check": "ppo_switch_breaks_bottleneck",
                "passed": ours_impr > sac_impr,
                "detail": (
                    f"ours improvement={ours_impr:.2f} vs sac_finetune improvement={sac_impr:.2f}"
                ),
            })
            notes.append(
                "Paper: 'fine-tuning ... with the SAC algorithm still suffers from the training "
                "bottleneck while switching to the PPO algorithm provides an opportunity to "
                "break through the bottleneck.'"
            )

    for rival in ("ppo_finetune", "jsrl", "statemask_r", "sac_finetune"):
        if rival in means and "ours" in means:
            checks.append({
                "check": f"ours_ge_{rival}",
                "passed": means["ours"] >= means[rival],
                "detail": f"ours={means['ours']:.2f} vs {rival}={means[rival]:.2f}",
            })

    if "ours" in means and "statemask_r" in means:
        checks.append({
            "check": "ours_gt_statemask_r",
            "passed": means["ours"] > means["statemask_r"],
            "detail": (
                "Paper notes StateMask-R (refining only from critical steps) can overfit and "
                "harm performance."
            ),
        })

    return {
        "available": True,
        "env_ids": [report.get("env_id")],
        "no_refine": None if not np.isfinite(no_refine) else no_refine,
        "mean_rewards": means,
        "best_method": best_method,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "reference_notes": REFERENCE_NOTES,
        "notes": notes,
    }


def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    """Render a human-readable Experiment-IV report (Figure-3 trend summary)."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Experiment IV — Refining a Pre-trained SAC Agent (RICE, ICML 2024)")
    lines.append("=" * 78)
    lines.append(f"application        : {report.get('env_id')}")
    lines.append(f"backend            : {report.get('backend')}")
    lines.append(f"methods            : {', '.join(report.get('methods', []))}")
    lines.append(f"seeds              : {report.get('seeds')}")
    budgets = report.get("budgets") or {}
    lines.append(
        "budgets (steps)    : "
        f"SAC={budgets.get('sac_pretrain')}, GAIL={budgets.get('gail')}, "
        f"mask={budgets.get('mask')}, refine={budgets.get('refine')}"
    )

    sac = report.get("sac_pretrain") or {}
    if sac and "error" not in sac:
        lines.append("-" * 78)
        lines.append("SAC pre-training (Figure 3, left)")
        lines.append(
            f"  final reward: {_safe_fmt(sac.get('final_reward'), decimals)} "
            f"(+- {_safe_fmt(sac.get('std'), decimals)}), "
            f"{int(sac.get('total_timesteps') or 0)} steps, {float(sac.get('wall_time') or 0):.1f}s"
        )
        for point in sac.get("eval_history") or []:
            lines.append(
                f"    step {int(point.get('timesteps', 0)):>8d}: "
                f"{_safe_fmt(point.get('mean_reward'), decimals)}"
            )
    elif sac:
        lines.append(f"SAC pre-training failed: {sac.get('error')}")

    gail = report.get("gail") or {}
    if gail:
        lines.append("-" * 78)
        lines.append("GAIL approximation of the SAC expert (π_G)")
        if "error" in gail:
            lines.append(f"  failed: {gail.get('error')}")
        else:
            lines.append(
                f"  reward {_safe_fmt(gail.get('eval_reward'), decimals)} "
                f"(+- {_safe_fmt(gail.get('std'), decimals)}), "
                f"{int(gail.get('total_timesteps') or 0)} steps, "
                f"{float(gail.get('wall_time') or 0):.1f}s"
            )

    lines.append("-" * 78)
    lines.append("Refining performance (Figure 3, right) — higher is better")
    no_refine = (report.get("no_refine") or {}).get("mean_reward")
    lines.append(f"  {'No Refine':<14} {_safe_fmt(no_refine, decimals)}")

    agg = report.get("aggregated") or {}
    for method in report.get("methods", list(agg.keys())):
        if method not in agg:
            lines.append(f"  {method:<14} n/a")
            continue
        entry = agg[method]
        lines.append(
            f"  {method:<14} {float(entry['mean_reward']):.{decimals}f} "
            f"(+- {float(entry['std_reward']):.{decimals}f})  "
            f"improvement {_safe_fmt(entry.get('mean_improvement'), decimals)}  "
            f"[{float(entry.get('wall_time') or 0):.1f}s]"
        )

    trends = report.get("trends") or {}
    if trends.get("available"):
        lines.append("-" * 78)
        lines.append("Trend validation (qualitative, Figure 3 has no tabulated values)")
        for check in trends.get("checks", []):
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(f"  [{status}] {check['check']}: {check['detail']}")
        lines.append(f"  best method: {trends.get('best_method')}")
        lines.append(f"  overall    : {'all checks passed' if trends.get('all_passed') else 'some checks failed'}")
    else:
        lines.append(f"Trend validation unavailable: {trends.get('reason', 'n/a')}")

    errors = report.get("errors") or []
    if errors:
        lines.append("-" * 78)
        lines.append(f"{len(errors)} error(s) recorded during the sweep:")
        for err in errors[:10]:
            lines.append(f"  {err.get('stage')} (seed {err.get('seed')}): {err.get('error')}")

    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp4_sac_agent",
        description=(
            "RICE Experiment IV (ICML 2024): pre-train SAC, approximate a policy with GAIL, "
            "and compare RICE refining against PPO fine-tuning / StateMask-R / JSRL / SAC "
            "fine-tuning on Hopper (Figure 3)."
        ),
    )
    parser.add_argument("--env", default=DEFAULT_ENV, help="application id (default: hopper)")
    parser.add_argument("--envs", nargs="+", default=None, help="several applications (multi-run)")
    parser.add_argument("--config", default=None, help="config name/path (defaults to the env id)")
    parser.add_argument("--methods", nargs="+", default=list(REFINING_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--sac-timesteps", type=int, default=None)
    parser.add_argument("--gail-timesteps", type=int, default=None)
    parser.add_argument("--refine-timesteps", type=int, default=None)
    parser.add_argument("--mask-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="pre-trained SAC checkpoint to reuse")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--no-gail", action="store_true", help="skip GAIL and refine the SAC actor")
    parser.add_argument("--no-sac-pretrain", action="store_true", help="reuse an existing SAC policy")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = get_config(args.config) if args.config else None
    out_dir = args.out_dir

    kwargs = dict(
        methods=tuple(args.methods),
        seeds=tuple(int(s) for s in args.seeds),
        sac_timesteps=args.sac_timesteps,
        gail_timesteps=args.gail_timesteps,
        refine_timesteps=args.refine_timesteps,
        mask_timesteps=args.mask_timesteps,
        use_gail=not args.no_gail,
        pretrain_sac=not args.no_sac_pretrain,
        device=args.device,
        out_dir=out_dir,
        progress=bool(args.progress),
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        eval_episodes=int(args.eval_episodes),
    )

    logger = get_logger("rice.exp4", out_dir=out_dir)
    if args.envs:
        reports = run_experiment4_multi(tuple(args.envs), cfg=cfg, logger=logger, **kwargs)
        for env_id, report in reports.items():
            print(format_report(report))
        if args.json:
            print(json.dumps(_jsonable(reports), indent=2))
        return 0

    report = run_experiment4(args.env, cfg=cfg, logger=logger, **kwargs)
    if args.json:
        print(json.dumps(_jsonable(report), indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
