#!/usr/bin/env python
"""Obtain the warm-start (bottlenecked) policies ``pi`` used by RICE.

The RICE pipeline (Algorithm 2) does *not* train an agent from scratch: it refines a
pre-trained policy that has reached a plateau ("bottlenecked but reasonable" -- see
Phase 1 of the reproduction plan, and Assumption 3.2 of the paper).  This script
produces those warm-start policies.

Two back-ends are supported:

* **Stable-Baselines3** (``--backend sb3``, the default when SB3 is importable).  The
  paper used the SB3 default ``MlpPolicy`` for the MuJoCo games, so this is the most
  faithful route.
* **RICE's own PPO** (``--backend rice``), which reuses ``rice.algorithms.ppo`` so the
  script keeps working on CPU-only / SB3-less machines.

For ``Experiment IV`` (refining a non-PPO agent) the ``--backend sac`` mode pre-trains
the self-contained SAC agent of :mod:`rice.baselines.sac_gail`.

Bottleneck detection (paper silent -> documented default).  Training stops early when
the evaluation return has stopped improving.  Two criteria are used, both configurable:

1. ``--target-reward``: stop once the mean evaluation return reaches a fraction
   ``--target-ratio`` (default ``0.99``) of the Table 1 "No Refine" reference value for
   the task.  This reproduces the paper's *bottlenecked -- but reasonable* policies.
2. ``--patience``: stop after ``patience`` consecutive evaluations without an
   improvement larger than ``--min-delta``.

Usage
-----
    python -m scripts.pretrain_agent --task Hopper-v3 --timesteps 1000000 --seeds 0 1 2
    python scripts/pretrain_agent.py --all --timesteps 1000000 --out-dir runs/pretrain

Every run writes

* ``{out_dir}/weights/{task}_seed{seed}.pt``  -- plain ``ActorCritic`` state dict
  (``torch.save``), loadable with ``rice.algorithms.refine.load_policy_weights``,
* ``{out_dir}/logs/{task}_seed{seed}.json``  -- training history (returns, timesteps,
  plateau flag, bottleneck reason),
* ``{out_dir}/summary.json``                  -- aggregate over seeds.
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
# Path bootstrap: the repository may be launched from its root (``rice/``) or from the
# inner package directory (``rice/rice``), so make both import layouts work.
# --------------------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)                      # .../rice
_PKG = os.path.join(_ROOT, "rice")                 # .../rice/rice
for _p in (_ROOT, _PKG, os.path.dirname(_ROOT)):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def import_first(module_names: Sequence[str]) -> Optional[Any]:
    """Import and return the first importable dotted module of ``module_names``."""
    import importlib

    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:  # pragma: no cover - environment dependent
            continue
    return None


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return default


# --------------------------------------------------------------------------------------
# Reference constants (Table 1 "No Refine" column, paper Sec. 4.3)
# --------------------------------------------------------------------------------------
TABLE1_NO_REFINE: Dict[str, float] = {
    "Hopper-v3": 3559.44,
    "Walker2d-v3": 3768.79,
    "Reacher-v2": -5.79,
    "HalfCheetah-v3": 2024.09,
    "SelfishMining": 14.36,
    "CageChallenge2": -23.64,
    "Macro-v1": 10.30,
    "SparseHopper": 0.0,
    "SparseHalfCheetah": 0.0,
    "MalwareMutation": 42.20,
}

# Tasks used by Experiment II/III (dense) and the sparse MuJoCo games.  SparseWalker2d
# and MalwareMutation are OUT OF SCOPE per the addendum.
DENSE_TASKS: Tuple[str, ...] = (
    "Hopper-v3",
    "Walker2d-v3",
    "Reacher-v2",
    "HalfCheetah-v3",
    "SelfishMining",
    "CageChallenge2",
    "Macro-v1",
)
SPARSE_TASKS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah")
ALL_TASKS: Tuple[str, ...] = DENSE_TASKS + SPARSE_TASKS
OUT_OF_SCOPE_TASKS: Tuple[str, ...] = ("SparseWalker2d", "MalwareMutation")

TASK_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3", "hopper-v3": "Hopper-v3",
    "walker2d": "Walker2d-v3", "walker2d-v3": "Walker2d-v3", "walker": "Walker2d-v3",
    "reacher": "Reacher-v2", "reacher-v2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3", "half-cheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3",
    "selfish": "SelfishMining", "selfishmining": "SelfishMining",
    "selfish_mining": "SelfishMining", "mining": "SelfishMining",
    "cage": "CageChallenge2", "cagechallenge2": "CageChallenge2",
    "cage_challenge2": "CageChallenge2", "cage-2": "CageChallenge2",
    "auto": "Macro-v1", "autodriving": "Macro-v1", "macro": "Macro-v1",
    "macro-v1": "Macro-v1", "metadrive": "Macro-v1",
    "sparsehopper": "SparseHopper", "sparse-hopper": "SparseHopper",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "sparse-halfcheetah": "SparseHalfCheetah",
    "sparsewalker2d": "SparseWalker2d",
    "malware": "MalwareMutation", "malwaremutation": "MalwareMutation",
}


def canonical_task(name: str) -> str:
    """Resolve a task spelling/alias to a canonical RICE task name."""
    if name is None:
        return "Hopper-v3"
    key = str(name).strip()
    if key in TABLE1_NO_REFINE:
        return key
    lowered = key.lower().replace(" ", "").replace("_", "")
    if lowered in TASK_ALIASES:
        return TASK_ALIASES[lowered]
    for canonical in ALL_TASKS:
        if canonical.lower().replace("-", "") == lowered.replace("-", ""):
            return canonical
    # Unknown names are returned as-is so custom environments keep working.
    return key


def is_sparse(task: str) -> bool:
    return canonical_task(task) in SPARSE_TASKS


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when torch + CUDA are available."""
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [json_default(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_default(v) for k, v in obj.items()}
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except Exception:  # pragma: no cover
        pass
    return str(obj)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class PretrainConfig:
    """Hyper-parameters of the warm-start pre-training stage.

    The paper does not specify the pre-training budget/algorithm details, so the
    defaults follow Stable-Baselines3 (Raffin et al. 2021) and are documented in the
    README.
    """

    task: str = "Hopper-v3"
    backend: str = "auto"                  # auto | sb3 | rice | sac
    total_timesteps: int = 1_000_000
    n_envs: int = 1
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: Optional[Tuple[int, ...]] = None
    activation: str = "tanh"
    eval_every: int = 20_000
    eval_episodes: int = 5
    target_reward: Optional[float] = None
    target_ratio: float = 0.99
    patience: int = 5
    min_delta: float = 25.0
    normalize_target: bool = True        # normalize by |reference| when detecting plateau
    n_seeds: int = 3
    seeds: Optional[List[int]] = None
    device: str = "auto"
    out_dir: str = "runs/pretrain"
    weights_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    save_every_eval: bool = False
    verbose: int = 1
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    sac_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def clone(self, **overrides: Any) -> "PretrainConfig":
        import copy

        new = copy.deepcopy(self)
        for key, value in overrides.items():
            if value is not None and hasattr(new, key):
                setattr(new, key, value)
        return new

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "PretrainConfig":
        mapping = dict(mapping or {})
        aliases = {
            "task": "task",
            "steps": "total_timesteps",
            "timesteps": "total_timesteps",
            "total_steps": "total_timesteps",
            "lr": "learning_rate",
            "n_eval_episodes": "eval_episodes",
            "K": "eval_every",
        }
        cfg = cls()
        for key, value in mapping.items():
            key = aliases.get(key, key)
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                cfg.extra[key] = value
        return cfg.clone(**overrides)

    def seed_list(self) -> List[int]:
        if self.seeds:
            return [int(s) for s in self.seeds]
        return list(range(int(self.n_seeds)))

    def resolved_dirs(self) -> Tuple[str, str]:
        weights_dir = self.weights_dir or os.path.join(self.out_dir, "weights")
        logs_dir = self.logs_dir or os.path.join(self.out_dir, "logs")
        return ensure_dir(weights_dir), ensure_dir(logs_dir)

    def resolved_target(self, task: Optional[str] = None) -> Optional[float]:
        """Return the plateau threshold for a task (``None`` = no early stop)."""
        if self.target_reward is not None:
            return float(self.target_reward)
        task = canonical_task(task or self.task)
        ref = TABLE1_NO_REFINE.get(task)
        if ref is None:
            return None
        if ref <= 0:
            # Tasks with negative rewards (Reacher, Cage Challenge 2): the paper's
            # bottleneck value may be worse than a randomly initialised policy, so a
            # fraction of it is not a useful stopping rule.  The reference itself is
            # used as an upper target only when it is positive; otherwise plateau
            # detection relies on --patience alone.
            return None
        return float(self.target_ratio * ref)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["seeds"] = self.seed_list()
        return out


# --------------------------------------------------------------------------------------
# Pre-trainer
# --------------------------------------------------------------------------------------
class Pretrainer:
    """Trains one warm-start policy per seed for a single task."""

    def __init__(
        self,
        config: Optional[PretrainConfig] = None,
        env: Any = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config.clone(**kwargs) if config is not None else PretrainConfig(**kwargs)
        if task:
            self.config.task = task
        self.task = canonical_task(self.config.task)
        self.env = env
        self.device = resolve_device(self.config.device)
        self.notes: List[str] = []

    # ---------------------------------------------------------------- env / policy
    def _env_module(self, task: str) -> Optional[Any]:
        if is_sparse(task):
            return import_first(
                ["rice.environments.mujoco_sparse", "rice.rice.environments.mujoco_sparse",
                 "environments.mujoco_sparse", "mujoco_sparse"]
            )
        registry = import_first(["rice.environments", "rice.rice.environments", "environments"])
        return registry

    def build_env(self, seed: Optional[int] = None) -> Any:
        if self.env is not None:
            return self.env
        registry = import_first(["rice.environments", "rice.rice.environments", "environments"])
        if registry is not None and hasattr(registry, "make_env"):
            try:
                return registry.make_env(self.task, seed=seed, **dict(self.config.env_kwargs))
            except Exception as exc:  # pragma: no cover - env dependent
                self.notes.append(f"make_env({self.task}) failed: {exc}")
        module = self._env_module(self.task)
        if module is not None and hasattr(module, "make_env"):
            return module.make_env(self.task, seed=seed, **dict(self.config.env_kwargs))
        raise ImportError(
            f"Could not build environment {self.task}: neither 'rice.environments' nor a "
            "task-specific module exposing 'make_env' could be imported."
        )

    def net_arch(self) -> Tuple[int, ...]:
        if self.config.net_arch:
            return tuple(int(x) for x in self.config.net_arch)
        registry = import_first(["rice.environments", "rice.rice.environments", "environments"])
        for name in ("default_net_arch", "net_arch_for", "get_net_arch"):
            fn = _get(registry, name)
            if callable(fn):
                try:
                    return tuple(int(x) for x in fn(self.task))
                except Exception:
                    continue
        # Fallbacks from Appendix C.2 (must mirror the mask network architecture).
        if self.task == "SelfishMining":
            return (128, 128, 128, 128)
        if self.task == "CageChallenge2":
            return (64, 64, 64)
        if self.task == "Macro-v1":
            return (256, 256)
        return (64, 64)

    def build_policy(self, env: Any = None) -> Any:
        """Build an ``ActorCritic`` matching the environment (used by backend='rice')."""
        ppo = import_first(["rice.algorithms.ppo", "rice.rice.algorithms.ppo", "algorithms.ppo"])
        if ppo is None:
            raise ImportError("rice.algorithms.ppo is required for backend='rice'")
        env = env if env is not None else self.build_env(seed=0)
        obs_space = _get(env, "observation_space")
        act_space = _get(env, "action_space")
        return ppo.ActorCritic(
            observation_space=obs_space,
            action_space=act_space,
            net_arch=self.net_arch(),
            activation=self.config.activation,
            device=self.device,
        )

    def ppo_config(self, ppo_module: Any) -> Any:
        cfg = ppo_module.PPOConfig()
        cfg.learning_rate = float(self.config.learning_rate)
        cfg.n_steps = int(self.config.n_steps)
        cfg.batch_size = int(self.config.batch_size)
        cfg.n_epochs = int(self.config.n_epochs)
        cfg.gamma = float(self.config.gamma)
        cfg.gae_lambda = float(self.config.gae_lambda)
        cfg.clip_range = float(self.config.clip_range)
        cfg.ent_coef = float(self.config.ent_coef)
        cfg.vf_coef = float(self.config.vf_coef)
        cfg.max_grad_norm = float(self.config.max_grad_norm)
        return cfg

    # ------------------------------------------------------------------- backends
    def _backend(self) -> str:
        backend = (self.config.backend or "auto").lower()
        if backend == "auto":
            if import_first(["stable_baselines3"]) is not None:
                return "sb3"
            return "rice"
        return backend

    def train(self, seed: int = 0, **overrides: Any) -> Dict[str, Any]:
        backend = self._backend()
        if backend == "sb3":
            result = self._train_sb3(seed=seed, **overrides)
        elif backend == "sac":
            result = self._train_sac(seed=seed, **overrides)
        else:
            result = self._train_rice(seed=seed, **overrides)
        return result

    # --- Stable-Baselines3 PPO ---------------------------------------------------
    def _train_sb3(self, seed: int = 0, **overrides: Any) -> Dict[str, Any]:
        sb3 = import_first(["stable_baselines3"])
        if sb3 is None:  # pragma: no cover - guarded by _backend
            raise ImportError("stable_baselines3 is not available; use --backend rice")
        PPO = _get(sb3, "PPO")
        DummyVecEnv = _get(import_first(["stable_baselines3.common.vec_env"]), "DummyVecEnv")
        Monitor = _get(import_first(["stable_baselines3.common.monitor"]), "Monitor")

        env = self.build_env(seed=seed)
        self._seed_env(env, seed)

        def make_env():
            def _init():
                return Monitor(self.build_env(seed=seed))

            return _init

        if DummyVecEnv is not None and self.config.n_envs > 1:
            vec = DummyVecEnv([make_env() for _ in range(int(self.config.n_envs))])
        else:
            vec = DummyVecEnv([make_env()]) if DummyVecEnv is not None else env

        policy_kwargs = {
            "net_arch": list(self.net_arch()),
            "activation_fn": _sb3_activation(self.config.activation),
        }
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=float(self.config.learning_rate),
            n_steps=int(self.config.n_steps),
            batch_size=int(self.config.batch_size),
            n_epochs=int(self.config.n_epochs),
            gamma=float(self.config.gamma),
            gae_lambda=float(self.config.gae_lambda),
            clip_range=float(self.config.clip_range),
            ent_coef=float(self.config.ent_coef),
            vf_coef=float(self.config.vf_coef),
            max_grad_norm=float(self.config.max_grad_norm),
            policy_kwargs=policy_kwargs,
            device=self.device,
            seed=seed,
            verbose=0,
        )

        history: List[Dict[str, Any]] = []
        target = self.config.resolved_target(self.task)
        best = -np.inf
        stale = 0
        plateau_reason = "budget"
        steps_done = 0

        while steps_done < int(self.config.total_timesteps):
            chunk = int(min(self.config.eval_every, int(self.config.total_timesteps) - steps_done))
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            steps_done += chunk
            stats = self._evaluate_sb3(model, self.build_env(seed=seed + 10_000), seed=seed)
            history.append({"timesteps": steps_done, **stats})
            if self.config.verbose:
                print(f"  [{self.task} seed={seed}] step {steps_done}: "
                      f"eval {stats['mean_return']:.3f}")
            improved = stats["mean_return"] > best + self._delta_scale(target) * float(self.config.min_delta)
            if improved:
                best = stats["mean_return"]
                stale = 0
            else:
                stale += 1
            if target is not None and stats["mean_return"] >= target:
                plateau_reason = "target_reward"
                break
            if stale >= int(self.config.patience):
                plateau_reason = "patience"
                break

        weights_path = self._save_weights(self._sb3_state_dict(model), seed)
        log = {
            "task": self.task,
            "seed": int(seed),
            "backend": "sb3",
            "timesteps": int(steps_done),
            "history": history,
            "final_eval": history[-1] if history else {},
            "best_eval": best if np.isfinite(best) else None,
            "plateau_reason": plateau_reason,
            "target_reward": target,
            "reference_no_refine": TABLE1_NO_REFINE.get(self.task),
            "net_arch": list(self.net_arch()),
            "weights": weights_path,
            "notes": list(self.notes),
            "seconds": None,
        }
        return log

    # --- RICE's own PPO ---------------------------------------------------------
    def _train_rice(self, seed: int = 0, **overrides: Any) -> Dict[str, Any]:
        ppo = import_first(["rice.algorithms.ppo", "rice.rice.algorithms.ppo", "algorithms.ppo"])
        if ppo is None:
            raise ImportError("rice.algorithms.ppo is unavailable")
        refine = import_first(["rice.algorithms.refine", "rice.rice.algorithms.refine",
                               "algorithms.refine"])
        cfg = self.ppo_config(ppo)

        env = self.build_env(seed=seed)
        self._seed_env(env, seed)
        policy = ppo.ActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=self.net_arch(),
            activation=self.config.activation,
            device=self.device,
        )
        optimizer = ppo.PPO(policy, config=cfg, device=self.device)

        history: List[Dict[str, Any]] = []
        target = self.config.resolved_target(self.task)
        best = -np.inf
        stale = 0
        plateau_reason = "budget"
        total_steps = 0
        start = time.time()

        while total_steps < int(self.config.total_timesteps):
            buffer = ppo.RolloutBuffer()
            obs, _ = _env_reset(env, seed=None if total_steps else seed)
            ep_returns, ep_return = [], 0.0
            steps = 0
            while steps < int(self.config.n_steps):
                action, value, log_prob = policy.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = _env_step(env, action)
                done = bool(terminated or truncated)
                buffer.add(obs, action, float(reward), next_obs, done, value, log_prob)
                ep_return += float(reward)
                obs = next_obs
                steps += 1
                total_steps += 1
                if done:
                    ep_returns.append(ep_return)
                    ep_return = 0.0
                    obs, _ = _env_reset(env)
            if ppo.flatten_obs is not None:
                last_obs = ppo.flatten_obs(obs)
            else:  # pragma: no cover
                last_obs = np.asarray(obs, dtype=np.float32).ravel()
            with torch.no_grad() if _torch() else _nullcontext():
                last_values = float(policy.predict_values(last_obs).item()) if _torch() else 0.0
            optimizer.update(buffer, last_values=last_values)

            if total_steps % int(self.config.eval_every) < int(self.config.n_steps):
                stats = self._evaluate_rice(policy, self.build_env(seed=seed + 10_000), seed=seed)
                history.append({"timesteps": total_steps, **stats})
                if self.config.verbose:
                    print(f"  [{self.task} seed={seed}] step {total_steps}: "
                          f"eval {stats['mean_return']:.3f}")
                improved = (stats["mean_return"]
                            > best + self._delta_scale(target) * float(self.config.min_delta))
                if improved:
                    best = stats["mean_return"]
                    stale = 0
                else:
                    stale += 1
                if target is not None and stats["mean_return"] >= target:
                    plateau_reason = "target_reward"
                    break
                if stale >= int(self.config.patience):
                    plateau_reason = "patience"
                    break

        weights_path = self._save_weights(getattr(policy, "state_dict")(), seed)
        return {
            "task": self.task,
            "seed": int(seed),
            "backend": "rice",
            "timesteps": int(total_steps),
            "history": history,
            "final_eval": history[-1] if history else {},
            "best_eval": best if np.isfinite(best) else None,
            "plateau_reason": plateau_reason,
            "target_reward": target,
            "reference_no_refine": TABLE1_NO_REFINE.get(self.task),
            "net_arch": list(self.net_arch()),
            "weights": weights_path,
            "notes": list(self.notes),
            "seconds": time.time() - start,
        }

    # --- SAC pre-training (Experiment IV) ---------------------------------------
    def _train_sac(self, seed: int = 0, **overrides: Any) -> Dict[str, Any]:
        sac_mod = import_first(["rice.baselines.sac_gail", "rice.rice.baselines.sac_gail",
                                "baselines.sac_gail"])
        if sac_mod is None:
            raise ImportError("rice.baselines.sac_gail is unavailable (SAC pre-training)")
        env = self.build_env(seed=seed)
        self._seed_env(env, seed)
        sac_cfg = sac_mod.SACConfig(task=self.task, seed=seed, device=self.device,
                                    **dict(self.config.sac_kwargs))
        agent = sac_mod.SACAgent(env=env, config=sac_cfg, device=self.device, seed=seed)
        stats = agent.pretrain(total_timesteps=int(self.config.total_timesteps), seed=seed)
        weights_path = self._save_weights(agent.state_dict(), seed, suffix="_sac")
        return {
            "task": self.task,
            "seed": int(seed),
            "backend": "sac",
            "timesteps": int(self.config.total_timesteps),
            "history": _get(stats, "history", default=[]) or [],
            "final_eval": _get(stats, "eval", default={}) or {},
            "plateau_reason": "budget",
            "reference_no_refine": TABLE1_NO_REFINE.get(self.task),
            "weights": weights_path,
            "notes": list(self.notes),
        }

    # ------------------------------------------------------------------ evaluation
    def _evaluate_rice(self, policy: Any, env: Any, seed: int = 0) -> Dict[str, float]:
        returns = []
        for ep in range(int(self.config.eval_episodes)):
            obs, _ = _env_reset(env, seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                action = policy.predict(obs, deterministic=True) if hasattr(policy, "predict") \
                    else policy.act(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, _ = _env_step(env, action)
                total += float(reward)
                done = bool(terminated or truncated)
            returns.append(total)
        return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
                "episodes": int(len(returns))}

    def _evaluate_sb3(self, model: Any, env: Any, seed: int = 0) -> Dict[str, float]:
        returns = []
        for ep in range(int(self.config.eval_episodes)):
            obs, _ = _env_reset(env, seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = _env_step(env, action)
                total += float(reward)
                done = bool(terminated or truncated)
            returns.append(total)
        return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
                "episodes": int(len(returns))}

    def evaluate(self, policy: Any, n_episodes: Optional[int] = None,
                 seed: int = 0, env: Any = None) -> Dict[str, float]:
        """Public evaluation helper (delegates to ``refine.evaluate_policy`` if possible)."""
        refine = import_first(["rice.algorithms.refine", "rice.rice.algorithms.refine",
                               "algorithms.refine"])
        env = env if env is not None else self.build_env(seed=seed)
        if refine is not None and hasattr(refine, "evaluate_policy"):
            try:
                return refine.evaluate_policy(
                    env, policy, n_episodes=n_episodes or self.config.eval_episodes,
                    seed=seed, deterministic=True,
                )
            except Exception as exc:  # pragma: no cover
                self.notes.append(f"evaluate_policy failed: {exc}")
        return self._evaluate_rice(policy, env, seed=seed)

    # ------------------------------------------------------------------- utilities
    def _delta_scale(self, target: Optional[float]) -> float:
        """Scale the improvement threshold so plateau detection works across tasks."""
        if not self.config.normalize_target:
            return 1.0
        if target:
            return max(abs(float(target)) / 100.0, 1e-6)
        ref = TABLE1_NO_REFINE.get(self.task)
        if ref:
            return max(abs(float(ref)) / 100.0, 1e-6)
        return 1.0

    def _seed_env(self, env: Any, seed: int) -> None:
        try:
            seeding = import_first(["rice.utils.seeding", "rice.rice.utils.seeding", "utils.seeding"])
            if seeding is not None and hasattr(seeding, "seed_env"):
                seeding.seed_env(env, seed)
                return
        except Exception:  # pragma: no cover
            pass
        try:
            env.reset(seed=seed)
        except Exception:
            pass

    def _save_weights(self, state: Any, seed: int, suffix: str = "") -> Optional[str]:
        """Persist a plain ``ActorCritic`` state dict so ``load_policy_weights`` works."""
        weights_dir, _ = self.config.resolved_dirs()
        path = os.path.join(weights_dir, f"{self.task}_seed{int(seed)}{suffix}.pt")
        try:
            import torch

            if isinstance(state, dict) and "policy" in state and "optimizer" in state:
                # RICE ActorCritic/PPO state dict -> keep the policy weights only so the
                # file is a drop-in for load_policy_weights().
                state = state.get("policy", state)
            torch.save(state, path)
            return path
        except Exception as exc:  # pragma: no cover
            self.notes.append(f"could not save weights to {path}: {exc}")
            return None

    def _sb3_state_dict(self, model: Any) -> Dict[str, Any]:
        """Extract the SB3 policy weights as a plain state dict."""
        try:
            import torch

            policy = model.policy
            sd = policy.state_dict()
            # Keep only the actor network so the checkpoint mirrors ActorCritic.
            if hasattr(policy, "mlp_extractor") and hasattr(policy, "action_net"):
                return {k: v for k, v in sd.items()}
            return sd
        except Exception:  # pragma: no cover
            return {}

    # ------------------------------------------------------------------------ run
    def run(self, seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        seeds = list(seeds) if seeds is not None else self.config.seed_list()
        _, logs_dir = self.config.resolved_dirs()
        results: List[Dict[str, Any]] = []
        for seed in seeds:
            print(f"[pretrain] task={self.task} seed={seed} backend={self._backend()}")
            try:
                log = self.train(seed=int(seed))
            except Exception as exc:  # pragma: no cover - keep multi-seed runs alive
                log = {"task": self.task, "seed": int(seed), "error": repr(exc),
                       "notes": list(self.notes)}
            results.append(log)
            with open(os.path.join(logs_dir, f"{self.task}_seed{int(seed)}.json"), "w") as fh:
                json.dump(log, fh, indent=2, default=json_default)

        finals = [r["final_eval"].get("mean_return") for r in results
                  if isinstance(r.get("final_eval"), dict) and r["final_eval"]]
        summary = {
            "task": self.task,
            "backend": self._backend(),
            "seeds": [int(s) for s in seeds],
            "reference_no_refine": TABLE1_NO_REFINE.get(self.task),
            "final_returns": finals,
            "mean_final_return": float(np.mean(finals)) if finals else None,
            "std_final_return": float(np.std(finals)) if finals else None,
            "results": results,
            "notes": list(self.notes),
        }
        ensure_dir(self.config.out_dir)
        with open(os.path.join(self.config.out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=json_default)
        return summary


# --------------------------------------------------------------------------------------
# Small helpers shared with the rest of the repo
# --------------------------------------------------------------------------------------
def _torch() -> Any:  # pragma: no cover - trivial
    try:
        import torch

        return torch
    except Exception:
        return None


class _nullcontext:  # noqa: N801 - tiny stand-in for contextlib.nullcontext
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _env_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if len(out) == 5:
        return out  # type: ignore[return-value]
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, dict(info or {})


def _sb3_activation(name: str) -> Any:
    try:
        import torch.nn as nn

        return {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }.get(str(name).lower(), nn.Tanh)
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-train warm-start (bottlenecked) policies for RICE refining."
    )
    parser.add_argument("--task", type=str, default="Hopper-v3",
                        help="Environment name or alias (default: Hopper-v3).")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Several tasks in one invocation.")
    parser.add_argument("--all", action="store_true",
                        help="Pre-train every in-scope task (dense + sparse).")
    parser.add_argument("--dense", action="store_true", help="Pre-train the dense tasks only.")
    parser.add_argument("--sparse", action="store_true", help="Pre-train the sparse tasks only.")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "sb3", "rice", "sac"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--net-arch", type=int, nargs="+", default=None)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--target-reward", type=float, default=None,
                        help="Absolute plateau threshold; default = target_ratio * No-Refine ref.")
    parser.add_argument("--target-ratio", type=float, default=0.99)
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after this many evaluations without improvement.")
    parser.add_argument("--min-delta", type=float, default=25.0,
                        help="Normalized improvement required to reset patience.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/pretrain")
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.all:
        tasks = list(ALL_TASKS)
    elif args.sparse:
        tasks = list(SPARSE_TASKS)
    elif args.dense:
        tasks = list(DENSE_TASKS)
    elif args.tasks:
        tasks = [canonical_task(t) for t in args.tasks]
    else:
        tasks = [canonical_task(args.task)]

    seeding = import_first(["rice.utils.seeding", "rice.rice.utils.seeding", "utils.seeding"])
    summaries: Dict[str, Any] = {}
    for task in tasks:
        if task in OUT_OF_SCOPE_TASKS:
            print(f"[pretrain] skipping {task} (out of scope per the addendum)")
            continue
        cfg = PretrainConfig(
            task=task,
            backend=args.backend,
            total_timesteps=int(args.timesteps),
            n_envs=int(args.n_envs),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.lr),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            net_arch=tuple(args.net_arch) if args.net_arch else None,
            activation=args.activation,
            eval_every=int(args.eval_every),
            eval_episodes=int(args.eval_episodes),
            target_reward=args.target_reward,
            target_ratio=float(args.target_ratio),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            seeds=list(args.seeds) if args.seeds else None,
            n_seeds=int(args.n_seeds),
            device=args.device,
            out_dir=os.path.join(args.out_dir, task),
            weights_dir=args.weights_dir,
            verbose=0 if args.quiet else 1,
        )
        seeds = cfg.seed_list()
        if seeding is not None and hasattr(seeding, "set_global_seeds"):
            for seed in seeds:
                seeding.set_global_seeds(seed)
        trainer = Pretrainer(config=cfg)
        summaries[task] = trainer.run(seeds)

    ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "all_summaries.json"), "w") as fh:
        json.dump(summaries, fh, indent=2, default=json_default)
    for task, summary in summaries.items():
        print(f"{task}: mean final return "
              f"{summary.get('mean_final_return')} (ref {summary.get('reference_no_refine')})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
