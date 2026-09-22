#!/usr/bin/env python
"""Train the pre-trained (sub-optimal / bottlenecked) target policy ``pi`` for RICE.

This script produces the frozen policy ``pi`` that Stage 1 (``rice/explanation``)
explains and Stage 2 (``rice/refining``) refines.  It is the "No Refine" regime of
Table 1 (e.g. Hopper ~3559.44, Walker2d ~3339.68, HalfCheetah ~4540.50, CAGE-2
~-23.64, Auto Driving ~10.30).

Paper notes (Appendix C.1 / Addendum "Architectures"):
    * dense / sparse MuJoCo: Stable-Baselines3 default ``MlpPolicy`` (2x64 tanh).
    * Selfish Mining: MLP with hidden sizes [128, 128, 128, 128].
    * CAGE Challenge 2: MLP with hidden sizes [64, 64, 64].
    * MetaDrive Macro-v1: DI-engine default ``VAC`` network.

The addendum ("Focus on overall results") states the exact architecture is not
checked -- only trends are -- so this script always tries the best available
backend (SB3 PPO -> native PyTorch PPO from :mod:`rice.refining.ppo_refine`) and
falls back gracefully.

Usage::

    python scripts/train_target.py --env hopper --timesteps 1000000
    python scripts/train_target.py --envs hopper walker2d --seeds 0 1 2
    python -m scripts.train_target --env cage2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Defensive imports: the script must stay importable (and CLI introspectable)
# even when torch / SB3 / MuJoCo / DI-drive are not installed.
# ---------------------------------------------------------------------------

try:  # numpy is a hard dependency of the whole project
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from rice.utils.io import ensure_dir, get_config, save_json
    from rice.utils.logging import Logger, format_mean_std, get_logger
    from rice.utils.seeding import set_seed
except Exception:  # pragma: no cover - local fallbacks
    import logging as _logging

    def ensure_dir(path: str) -> str:  # type: ignore
        os.makedirs(path, exist_ok=True)
        return path

    def get_config(name: str = "default", config_dir: Optional[str] = None) -> Dict[str, Any]:  # type: ignore
        try:
            import yaml  # type: ignore

            root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
            path = os.path.join(config_dir or root, f"{name}.yaml")
            if not os.path.exists(path):
                path = os.path.join(config_dir or root, name)
            with open(path, "r") as handle:
                return yaml.safe_load(handle) or {}
        except Exception:
            return {}

    def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w") as handle:
            json.dump(obj, handle, indent=indent, default=str)
        return path

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = 20):  # type: ignore
        logger = _logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(_logging.StreamHandler())
            logger.setLevel(level)
        return logger

    class Logger:  # type: ignore
        def __init__(self, out_dir: Optional[str] = None, name: str = "rice", **kwargs: Any) -> None:
            self.out_dir = out_dir
            self.history: Dict[str, List[float]] = {}

        def record(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                try:
                    self.history.setdefault(key, []).append(float(value))
                except Exception:
                    pass

        def timer_start(self, name: str) -> float:
            self._t = (name, time.time())
            return self._t[1]

        def timer_end(self, name: str = "", accumulate: bool = True) -> float:
            if getattr(self, "_t", None) is None:
                return 0.0
            return time.time() - self._t[1]

        def dump(self, filename: str = "progress.json") -> Optional[str]:
            return None

        def close(self) -> None:
            return None

    def format_mean_std(values: Sequence[float], decimals: int = 2) -> str:  # type: ignore
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.{decimals}f}"
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        return f"{mean:.{decimals}f} +- {var ** 0.5:.{decimals}f}"

    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        import random as _random

        os.environ["PYTHONHASHSEED"] = str(int(seed))
        _random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        return int(seed)


try:
    from rice.envs.make_env import (
        available_envs,
        d_max_for,
        env_backend,
        env_metadata,
        make_env,
        resolve_env_spec,
    )

    _HAS_ENVS = True
except Exception:  # pragma: no cover
    _HAS_ENVS = False

    available_envs = lambda: []  # type: ignore
    resolve_env_spec = None  # type: ignore
    env_metadata = None  # type: ignore
    env_backend = lambda env=None: "unknown"  # type: ignore
    d_max_for = lambda env_id, default=None: default  # type: ignore
    make_env = None  # type: ignore

try:
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        policy_arch,
        save_policy,
        sb3_policy_kwargs,
    )

    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    _HAS_POLICIES = False

    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        text = str(env_id or "default").strip().lower()
        for token in ("-v0", "-v1", "-v2", "-v3", "-v4", "-v5"):
            text = text.replace(token, "")
        return text.replace("-", "_").replace(" ", "_") or "default"

    build_policy = None  # type: ignore
    load_policy = None  # type: ignore
    save_policy = None  # type: ignore
    policy_arch = None  # type: ignore
    sb3_policy_kwargs = None  # type: ignore

try:
    from rice.refining.ppo_refine import (
        DEFAULT_GAMMA,
        DEFAULT_GAE_LAMBDA,
        DEFAULT_PPO_LR,
        evaluate_refined_policy,
        refine_policy,
        unpack_reset,
        unpack_step,
    )

    _HAS_REFINER = True
except Exception:  # pragma: no cover
    _HAS_REFINER = False

    DEFAULT_PPO_LR = 3e-4
    DEFAULT_GAMMA = 0.99
    DEFAULT_GAE_LAMBDA = 0.95
    refine_policy = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore

    def unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:  # type: ignore
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return result, {}

    def unpack_step(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:  # type: ignore
        if isinstance(result, tuple) and len(result) == 5:
            return result
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, bool(done), False, info or {}
        raise ValueError("unrecognised env.step() return signature")


# ---------------------------------------------------------------------------
# Paper constants (Table 3 / Table 1 references, training budgets)
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

#: Pre-training budgets (paper trains agents once; 1M is the SB3-standard budget
#: for dense MuJoCo, the applications need more).  These are not stated in the
#: paper -> sensible defaults that reach the "No Refine" regime.
DEFAULT_TIMESTEPS: Dict[str, int] = {
    "hopper": 1_000_000,
    "walker2d": 1_000_000,
    "reacher": 1_000_000,
    "halfcheetah": 1_000_000,
    "sparse_hopper": 1_000_000,
    "sparse_halfcheetah": 1_000_000,
    "selfish_mining": 2_000_000,
    "cage2": 1_000_000,
    "autodriving": 2_000_000,
    "default": 1_000_000,
}

#: Table 1 "No Refine" reference returns (trend validation only).
REFERENCE_NO_REFINE: Dict[str, float] = {
    "hopper": 3559.44,
    "walker2d": 3339.68,
    "reacher": -5.51,
    "halfcheetah": 4540.50,
    "selfish_mining": 0.0,
    "cage2": -23.64,
    "autodriving": 10.30,
}

#: Reward magnitudes that are signed / negative (handled specially in reporting).
NEGATIVE_REWARD_ENVS: Tuple[str, ...] = ("reacher", "cage2")

#: Per-application architecture (Appendix C.1 / Addendum "Architectures").
TARGET_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "hopper": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
    "walker2d": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
    "reacher": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
    "halfcheetah": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
    "sparse_hopper": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
    "sparse_halfcheetah": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
    "selfish_mining": {"policy": "MlpPolicy", "hidden_sizes": (128, 128, 128, 128), "activation": "tanh"},
    "cage2": {"policy": "MlpPolicy", "hidden_sizes": (64, 64, 64), "activation": "tanh"},
    "autodriving": {"policy": "VAC", "hidden_sizes": (64, 64), "activation": "relu", "backend": "di-engine"},
    "default": {"policy": "MlpPolicy", "hidden_sizes": (64, 64), "activation": "tanh"},
}

DEFAULT_OUT_DIR = "results/target"
DEFAULT_EVAL_EPISODES = 10


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def target_architecture(env_id: str) -> Dict[str, Any]:
    """Return the paper's target-policy architecture spec for ``env_id``."""
    key = normalize_env_key(env_id)
    spec = TARGET_ARCHITECTURES.get(key)
    if spec is None:
        spec = dict(TARGET_ARCHITECTURES["default"])
    spec = dict(spec)
    spec["env_key"] = key
    return spec


def timesteps_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Resolve the pre-training budget for an application."""
    if cfg:
        value = deep_get(cfg, "target", "total_timesteps")
        if value:
            return int(value)
    key = normalize_env_key(env_id)
    return int(DEFAULT_TIMESTEPS.get(key, DEFAULT_TIMESTEPS["default"]))


def reference_for(env_id: str) -> Optional[float]:
    """Paper "No Refine" reference return (trend validation only)."""
    return REFERENCE_NO_REFINE.get(normalize_env_key(env_id))


def is_negative_env(env_id: str) -> bool:
    return normalize_env_key(env_id) in NEGATIVE_REWARD_ENVS


def deep_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Nested dict/list lookup tolerant of missing keys."""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, None)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(key)]
            except Exception:
                return default
            if isinstance(current, (list, tuple)) and len(current) == 1:
                current = current[0]
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    return deep_get(cfg, *keys, default=default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float("nan")
    if np is not None:
        return float(np.mean(vals))
    return sum(vals) / len(vals)


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    if np is not None:
        return float(np.std(vals))
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def _policy_action(policy: Any, observation: Any, deterministic: bool = False) -> Any:
    """Interface-agnostic action selection (SB3 ``predict`` / native ``act``)."""
    if policy is None:
        raise ValueError("policy is required to select an action")
    if hasattr(policy, "predict"):
        try:
            action = policy.predict(observation, deterministic=deterministic)
        except TypeError:
            action = policy.predict(observation)
        if isinstance(action, tuple):
            action = action[0]
        return action
    if hasattr(policy, "act"):
        try:
            action = policy.act(observation, deterministic=deterministic)
        except TypeError:
            action = policy.act(observation)
        if isinstance(action, tuple):
            action = action[0]
        return action
    if callable(policy):
        return policy(observation)
    raise TypeError("unsupported policy interface")


def _make_obs(obs: Any) -> Any:
    """Flatten/normalise an observation for a torch policy."""
    if np is None:
        return obs
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    try:
        import torch

        return torch.as_tensor(flat, dtype=torch.float32)
    except Exception:
        return flat


# ---------------------------------------------------------------------------
# Environment / policy construction
# ---------------------------------------------------------------------------


def build_experiment_env(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    mode: str = "train",
    **kwargs: Any,
) -> Any:
    """Create an environment via the RICE factory."""
    if make_env is None:
        raise RuntimeError("rice.envs.make_env is unavailable; cannot create environments")
    normalize = _cfg_get(cfg, "env", "normalize_obs")
    normalize = kwargs.pop("normalize", normalize)
    max_steps = kwargs.pop("max_episode_steps", _cfg_get(cfg, "env", "max_episode_steps"))
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
    """Instantiate a fresh target policy using the paper's per-app architecture."""
    spec = target_architecture(env_id)
    seed = kwargs.pop("seed", None)
    if build_policy is not None:
        try:
            policy = build_policy(
                env_id,
                observation_space=getattr(env, "observation_space", None),
                action_space=getattr(env, "action_space", None),
                kind="policy",
                hidden_sizes=spec.get("hidden_sizes"),
                activation=spec.get("activation"),
                device=device,
                seed=seed,
                **kwargs,
            )
            return policy
        except TypeError:
            pass
        except Exception:
            pass

    # SB3 fallback (default MlpPolicy, 2x64 tanh).
    try:
        from stable_baselines3 import PPO  # type: ignore

        policy_kwargs = None
        if sb3_policy_kwargs is not None:
            try:
                policy_kwargs = sb3_policy_kwargs(env_id, activation=spec.get("activation"))
            except Exception:
                policy_kwargs = None
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=_cfg_get(cfg, "target", "lr", default=DEFAULT_PPO_LR),
            gamma=_cfg_get(cfg, "target", "gamma", default=DEFAULT_GAMMA),
            gae_lambda=_cfg_get(cfg, "target", "gae_lambda", default=DEFAULT_GAE_LAMBDA),
            clip_range=_cfg_get(cfg, "target", "clip_range", default=0.2),
            n_epochs=_cfg_get(cfg, "target", "n_epochs", default=10),
            batch_size=_cfg_get(cfg, "target", "batch_size", default=64),
            n_steps=_cfg_get(cfg, "target", "n_steps", default=2048),
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            verbose=0,
        )
        return model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"could not construct a target policy for {env_id!r}: {exc}") from exc


def build_or_load_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Load the frozen target policy from a checkpoint, else build a fresh one."""
    checkpoint = checkpoint or _cfg_get(cfg, "target", "checkpoint")
    for candidate in [checkpoint, f"policies/{normalize_env_key(env_id)}_ppo.zip"]:
        if not candidate or load_policy is None:
            continue
        path = str(candidate)
        if not os.path.exists(path):
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
                logger.info("loaded pre-trained policy from %s", path)
            return policy
        except Exception:
            continue
    if load_policy is not None:
        try:
            policy = load_policy(env_id, device=device)
            if policy is not None and logger is not None:
                logger.info("loaded pre-trained policy via registry for %s", env_id)
            if policy is not None:
                return policy
        except Exception:
            pass
    return build_target_policy(env, env_id, cfg=cfg, device=device, **kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_target_policy(
    env: Any,
    env_id: str,
    total_timesteps: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    policy: Any = None,
    eval_points: int = 0,
    eval_episodes: int = 3,
    progress: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Pre-train ``pi`` on the environment.

    Tries, in order:
      1. Stable-Baselines3 ``PPO`` (the backend the paper used for MuJoCo),
      2. the native PyTorch PPO engine in :mod:`rice.refining.ppo_refine` with
         RICE's contributions disabled (``use_mixed_init=False``, ``use_rnd=False``),
         which reproduces the "No Refine" regime.

    Returns ``(policy, info)``.
    """
    total_timesteps = int(total_timesteps or timesteps_for(env_id, cfg))
    info: Dict[str, Any] = {
        "env_id": normalize_env_key(env_id),
        "total_timesteps": total_timesteps,
        "seed": int(seed),
        "backend": None,
        "curve": [],
        "errors": [],
    }
    set_seed(seed)
    start = time.time()

    # ---- 1) Stable-Baselines3 path ---------------------------------------
    try:
        from stable_baselines3 import PPO  # type: ignore

        spec = target_architecture(env_id)
        policy_kwargs = None
        if sb3_policy_kwargs is not None:
            try:
                policy_kwargs = sb3_policy_kwargs(env_id, activation=spec.get("activation"))
            except Exception:
                policy_kwargs = None
        if policy is not None and hasattr(policy, "learn"):
            model = policy
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=_cfg_get(cfg, "target", "lr", default=DEFAULT_PPO_LR),
                gamma=_cfg_get(cfg, "target", "gamma", default=DEFAULT_GAMMA),
                gae_lambda=_cfg_get(cfg, "target", "gae_lambda", default=DEFAULT_GAE_LAMBDA),
                clip_range=_cfg_get(cfg, "target", "clip_range", default=0.2),
                n_epochs=_cfg_get(cfg, "target", "n_epochs", default=10),
                batch_size=_cfg_get(cfg, "target", "batch_size", default=64),
                n_steps=_cfg_get(cfg, "target", "n_steps", default=2048),
                policy_kwargs=policy_kwargs,
                device=device,
                seed=seed,
                verbose=1 if progress else 0,
            )
        chunk = max(1, total_timesteps // max(1, eval_points)) if eval_points else total_timesteps
        trained = 0
        while trained < total_timesteps:
            step = min(chunk, total_timesteps - trained)
            model.learn(total_timesteps=step, reset_num_timesteps=False, progress_bar=False)
            trained += step
            if eval_points:
                try:
                    result = evaluate_policy_return(
                        env, model, env_id, n_episodes=eval_episodes, device=device
                    )
                    info["curve"].append({"timesteps": trained, **result})
                    if logger is not None:
                        logger.info(
                            "target %s: %d/%d timesteps, return=%.2f",
                            env_id,
                            trained,
                            total_timesteps,
                            result.get("mean_reward", float("nan")),
                        )
                except Exception:
                    pass
        info["backend"] = "sb3"
        info["wall_time"] = time.time() - start
        return model, info
    except Exception as exc:
        info["errors"].append(f"sb3: {exc}")

    # ---- 2) Native PyTorch PPO (RICE refiner with RICE parts off) ---------
    if refine_policy is not None:
        try:
            config = {
                "use_mixed_init": False,
                "use_rnd": False,
                "p": 0.0,
                "lam": 0.0,
                "lr": _cfg_get(cfg, "target", "lr", default=DEFAULT_PPO_LR),
                "gamma": _cfg_get(cfg, "target", "gamma", default=DEFAULT_GAMMA),
                "gae_lambda": _cfg_get(cfg, "target", "gae_lambda", default=DEFAULT_GAE_LAMBDA),
                "clip_range": _cfg_get(cfg, "target", "clip_range", default=0.2),
                "n_epochs": _cfg_get(cfg, "target", "n_epochs", default=10),
                "batch_size": _cfg_get(cfg, "target", "batch_size", default=64),
            }
            trained_policy, refiner = refine_policy(
                env,
                policy=policy,
                total_timesteps=total_timesteps,
                env_id=env_id,
                config=config,
                seed=seed,
                device=device,
                logger=logger,
                progress=progress,
            )
            info["backend"] = "native_ppo"
            info["wall_time"] = time.time() - start
            info["trainer"] = refiner
            return trained_policy, info
        except Exception as exc:
            info["errors"].append(f"native_ppo: {exc}\\n{traceback.format_exc(limit=2)}")

    info["wall_time"] = time.time() - start
    if policy is None:
        policy = build_target_policy(env, env_id, cfg=cfg, device=device, seed=seed)
    return policy, info


def pretrain_target_policy(
    env: Any,
    env_id: str,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    device: str = "cpu",
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper returning only the pre-trained ``pi``."""
    policy, _ = train_target_policy(
        env,
        env_id,
        total_timesteps=total_timesteps,
        seed=seed,
        device=device,
        logger=logger,
        cfg=cfg,
        **kwargs,
    )
    return policy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


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
    """Evaluate a policy from the default initial state distribution ``rho``."""
    if evaluate_refined_policy is not None:
        try:
            return evaluate_refined_policy(
                env,
                policy,
                env_id=env_id,
                n_episodes=n_episodes,
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
            )
        except TypeError:
            try:
                return evaluate_refined_policy(
                    env, policy, env_id=env_id, n_episodes=n_episodes, deterministic=deterministic
                )
            except Exception:
                pass
        except Exception:
            pass

    rewards: List[float] = []
    max_steps = int(max_steps or _cfg_get({}, "env", "max_episode_steps", default=0) or 0)
    for _ in range(max(1, int(n_episodes))):
        obs, _info = unpack_reset(env.reset())
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            action = _policy_action(policy, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = unpack_step(env.step(action))
            episode_reward += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        rewards.append(episode_reward)
    return {
        "mean_reward": _mean(rewards),
        "std_reward": _std(rewards),
        "n_episodes": len(rewards),
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def checkpoint_path(env_id: str, seed: Optional[int] = None, out_dir: str = "policies") -> str:
    key = normalize_env_key(env_id)
    ensure_dir(out_dir)
    if seed is None:
        return os.path.join(out_dir, f"{key}_ppo.zip")
    return os.path.join(out_dir, f"{key}_ppo_seed{seed}.zip")


def save_target_checkpoint(
    policy: Any,
    env: Any,
    env_id: str,
    path: Optional[str] = None,
    seed: Optional[int] = None,
    total_timesteps: Optional[int] = None,
    logger: Any = None,
    **extra: Any,
) -> Optional[str]:
    """Persist ``pi`` (SB3 ``.zip`` or torch checkpoint) for later stages."""
    key = normalize_env_key(env_id)
    path = path or checkpoint_path(env_id, seed=seed)
    # SB3 model -> use its native save so the checkpoint stays loadable by SB3.
    if hasattr(policy, "save"):
        try:
            ensure_dir(os.path.dirname(os.path.abspath(path)))
            policy.save(path)
            if logger is not None:
                logger.info("saved SB3 target policy to %s", path)
            return path
        except Exception:
            pass
    if save_policy is not None:
        try:
            return save_policy(
                policy,
                path,
                env_id=key,
                kind="policy",
                seed=seed,
                total_timesteps=total_timesteps,
                **extra,
            )
        except TypeError:
            return save_policy(policy, path, env_id=key, kind="policy")
        except Exception:
            pass
    if hasattr(policy, "state_dict"):
        try:
            import torch

            import copy as _copy

            ensure_dir(os.path.dirname(os.path.abspath(path)))
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "env_key": key,
                    "kind": "policy",
                    "hidden_sizes": list(policy_arch(env_id)) if policy_arch is not None else None,
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                },
                path,
            )
            _ = _copy  # keep import explicit for clarity
            if logger is not None:
                logger.info("saved torch target policy to %s", path)
            return path
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Single / multi-run drivers
# ---------------------------------------------------------------------------


@dataclass
class TargetResult:
    """Outcome of training / evaluating the pre-trained target policy ``pi``."""

    env_id: str
    seed: int
    total_timesteps: int
    mean_reward: float
    std_reward: float
    eval_rewards: List[float] = field(default_factory=list)
    reference: Optional[float] = None
    backend: Optional[str] = None
    wall_time: float = 0.0
    curve: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Optional[str] = None
    policy: Any = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_policy: bool = False) -> Dict[str, Any]:
        payload = {
            "env_id": self.env_id,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "eval_rewards": list(self.eval_rewards),
            "reference": self.reference,
            "backend": self.backend,
            "wall_time": self.wall_time,
            "curve": list(self.curve),
            "checkpoint": self.checkpoint,
            "info": {k: v for k, v in self.info.items() if k != "trainer"},
        }
        if include_policy:
            payload["policy"] = repr(self.policy)
        return payload

    def format(self, decimals: int = 2) -> str:
        ref = "n/a" if self.reference is None else f"{self.reference:.{decimals}f}"
        return (
            f"{self.env_id:<18} seed={self.seed}  "
            f"return={self.mean_reward:.{decimals}f} +- {self.std_reward:.{decimals}f}  "
            f"(paper No-Refine: {ref})  backend={self.backend}"
        )


def run_target(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seeds: Sequence[int] = (0,),
    total_timesteps: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    progress: bool = False,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    eval_points: int = 0,
    out_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    save: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Pre-train and evaluate ``pi`` for one application over one or more seeds."""
    env_id = normalize_env_key(env_id)
    budget = int(total_timesteps or timesteps_for(env_id, cfg))
    results: List[TargetResult] = []
    errors: List[str] = []

    for seed in seeds:
        try:
            env = build_experiment_env(env_id, cfg=cfg, seed=int(seed), mode="train")
            policy, info = train_target_policy(
                env,
                env_id,
                total_timesteps=budget,
                seed=int(seed),
                device=device,
                logger=logger,
                cfg=cfg,
                eval_points=eval_points,
                eval_episodes=max(1, min(eval_episodes, 3)),
                progress=progress,
            )
            eval_result = evaluate_policy_return(
                env,
                policy,
                env_id,
                n_episodes=eval_episodes,
                device=device,
                logger=logger,
            )
            ckpt = None
            if save:
                target_dir = os.path.join(out_dir or _results_dir(cfg), "checkpoints")
                ckpt = save_target_checkpoint(
                    policy,
                    env,
                    env_id,
                    path=os.path.join(target_dir, f"{env_id}_ppo_seed{seed}.zip")
                    if checkpoint is None
                    else checkpoint,
                    seed=int(seed),
                    total_timesteps=budget,
                    logger=logger,
                )
            results.append(
                TargetResult(
                    env_id=env_id,
                    seed=int(seed),
                    total_timesteps=budget,
                    mean_reward=float(eval_result.get("mean_reward", float("nan"))),
                    std_reward=float(eval_result.get("std_reward", 0.0)),
                    eval_rewards=list(eval_result.get("rewards", [])),
                    reference=reference_for(env_id),
                    backend=info.get("backend"),
                    wall_time=float(info.get("wall_time", 0.0)),
                    curve=list(info.get("curve", [])),
                    checkpoint=ckpt,
                    policy=policy,
                    info=info,
                )
            )
            if logger is not None:
                logger.info("target %s %s", env_id, results[-1].format())
        except Exception as exc:
            errors.append(f"{env_id} seed={seed}: {exc}")
            if logger is not None:
                logger.warning("target training failed for %s seed=%s: %s", env_id, seed, exc)

    report = {
        "env_id": env_id,
        "seeds": list(seeds),
        "total_timesteps": budget,
        "architecture": target_architecture(env_id),
        "results": [r.to_dict() for r in results],
        "mean_reward": _mean([r.mean_reward for r in results]),
        "std_reward": _std([r.mean_reward for r in results]),
        "reference": reference_for(env_id),
        "errors": errors,
    }
    if out_dir:
        report["report_json"] = _write_report(
            os.path.join(out_dir, f"target_{env_id}.json"), report
        )
    return report


def run_target_multi(
    env_ids: Sequence[str] = DEFAULT_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run :func:`run_target` over several applications."""
    combined: Dict[str, Any] = {"envs": {}, "errors": []}
    for env_id in env_ids:
        try:
            combined["envs"][normalize_env_key(env_id)] = run_target(
                env_id, cfg=cfg, out_dir=out_dir, logger=logger, progress=progress, **kwargs
            )
        except Exception as exc:
            combined["errors"].append(f"{env_id}: {exc}")
    if out_dir:
        combined["report_json"] = _write_report(os.path.join(out_dir, "target_all.json"), combined)
    return combined


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _results_dir(cfg: Optional[Dict[str, Any]] = None) -> str:
    return str(_cfg_get(cfg, "results_dir", default="results") or "results")


def _write_report(path: str, payload: Dict[str, Any]) -> Optional[str]:
    try:
        return save_json(_jsonable(payload), path)
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k not in ("policy", "trainer")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if np is not None and isinstance(value, (np.integer, np.floating)):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    lines: List[str] = []
    envs = report.get("envs") if "envs" in report else {report.get("env_id", "?"): report}
    for env_id, entry in (envs or {}).items():
        if not isinstance(entry, dict):
            continue
        values = [r.get("mean_reward") for r in entry.get("results", []) if r.get("mean_reward") is not None]
        ref = entry.get("reference")
        ref_s = "n/a" if ref is None else f"{ref:.{decimals}f}"
        lines.append(
            f"{env_id:<18} pre-trained pi = {format_mean_std(values, decimals)}   "
            f"(paper No-Refine: {ref_s})"
        )
    errors = report.get("errors") or []
    if errors:
        lines.append("")
        lines.append("errors:")
        for err in errors:
            lines.append(f"  - {err}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the pre-trained (sub-optimal) target policy pi for RICE."
    )
    parser.add_argument("--env", "--env-id", dest="env", default=None, help="Application / env id.")
    parser.add_argument("--envs", nargs="*", default=None, help="Several applications to pre-train.")
    parser.add_argument("--config", default="default", help="Config stem in configs/ (default: default).")
    parser.add_argument("--timesteps", type=int, default=None, help="Pre-training budget.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0], help="Random seeds.")
    parser.add_argument("--device", default="cpu", help="torch device (cpu/cuda).")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Results directory.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path to write.")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-points", type=int, default=0, help="Record N points along the curve.")
    parser.add_argument("--no-save", action="store_true", help="Do not save checkpoints.")
    parser.add_argument("--progress", action="store_true", help="Verbose SB3 progress.")
    parser.add_argument("--list-envs", action="store_true", help="List known applications and exit.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_envs:
        for env_id in DEFAULT_ENVS + SPARSE_ENVS:
            spec = target_architecture(env_id)
            print(f"{env_id:<18} {spec['policy']:<10} hidden={tuple(spec['hidden_sizes'])}")
        return 0

    env_ids: List[str] = list(args.envs or ([args.env] if args.env else ["hopper"]))
    cfg = get_config(args.config) if args.config else {}
    out_dir = args.out_dir or os.path.join(_results_dir(cfg), "target")
    ensure_dir(out_dir)
    logger = get_logger("rice.train_target", out_dir=out_dir)

    if len(env_ids) == 1:
        report = run_target(
            env_ids[0],
            cfg=cfg,
            seeds=args.seeds,
            total_timesteps=args.timesteps,
            device=args.device,
            logger=logger,
            progress=args.progress,
            eval_episodes=args.eval_episodes,
            eval_points=args.eval_points,
            out_dir=out_dir,
            checkpoint=args.checkpoint,
            save=not args.no_save,
        )
    else:
        report = run_target_multi(
            env_ids,
            cfg=cfg,
            seeds=args.seeds,
            total_timesteps=args.timesteps,
            device=args.device,
            logger=logger,
            progress=args.progress,
            eval_episodes=args.eval_episodes,
            eval_points=args.eval_points,
            out_dir=out_dir,
            checkpoint=args.checkpoint,
            save=not args.no_save,
        )

    text = format_report(report)
    if text:
        print(text)
    try:
        save_json(_jsonable(report), os.path.join(out_dir, "target_report.json"))
    except Exception:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
