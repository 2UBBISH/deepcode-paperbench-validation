"""Stage-1 mask-network training entry point for RICE (Algorithm 1).

This script trains the *mask network* ``~pi_theta(a_t^m | s_t)`` that RICE uses as a
step-level explanation of a frozen, pre-trained (sub-optimal / bottlenecked) target
policy ``pi``.  It implements Algorithm 1 of Cheng et al. (ICML 2024):

    initialise theta, theta_old <- theta
    for iteration = 1 .. N do
        s_0 ~ rho
        for t = 0 .. T-1 do
            a_t   ~ pi(. | s_t)                # frozen target policy
            a_t^m ~ ~pi_{theta_old}(. | s_t)   # mask action (0 = keep, 1 = blind)
            a     = a_t (.) a_t^m              # masked action operator
            s_{t+1}, R_t ~ env.step(a)
            store (s_t, s_{t+1}, a_t^m, R'_t) with R'_t = R_t + alpha * a_t^m
        end for
        update theta with VANILLA PPO on the collected dataset D   (no primal-dual)
    end for

The reformulation ``max eta(pi_bar)`` (instead of StateMask's primal-dual
``min |eta(pi) - eta(pi_bar)|``) is justified by Theorem 3.3 of the paper and is the
source of the reported ~16.8% training-time reduction over StateMask (Table 4).  The
wall-clock timers exposed by :class:`rice.explanation.mask_trainer.MaskTrainer` are
therefore persisted in the report so Experiment I can reproduce the efficiency claim.

The heavy lifting is delegated to the already-implemented modules:

* ``rice.explanation.mask_trainer`` - Algorithm 1 trainer (vanilla PPO + blinding bonus)
* ``rice.explanation.mask_network`` - mask network module + masked-action operator
* ``rice.envs.make_env``            - environment factory (dense/sparse/applications)
* ``rice.models.policies``          - per-application policy architectures & checkpoints
* ``rice.baselines.statemask_r``    - StateMask (primal-dual) baseline for the timing
                                      comparison of Table 4

Everything imports defensively so the CLI stays introspectable (``--help``,
``--list-envs``) even when torch / SB3 / MuJoCo are unavailable.
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

# --------------------------------------------------------------------------------------
# Project imports (defensive: keep the CLI usable on minimal installs)
# --------------------------------------------------------------------------------------
_HAS_IO = _HAS_LOGGING = _HAS_SEEDING = False
_HAS_ENVS = _HAS_POLICIES = _HAS_MASK_NETWORK = _HAS_MASK_TRAINER = False
_HAS_STATEMASK = _HAS_REFINER = False

try:  # pragma: no cover - trivial availability probing
    from rice.utils.io import ensure_dir, get_config, save_json

    _HAS_IO = True
except Exception:  # pragma: no cover
    def ensure_dir(path: str) -> str:  # type: ignore
        os.makedirs(path, exist_ok=True)
        return path

    def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(obj), handle, indent=indent)
        return path

    def get_config(name: str = "default", config_dir: Optional[str] = None) -> Dict[str, Any]:  # type: ignore
        try:
            import yaml  # type: ignore

            path = name if str(name).endswith((".yaml", ".yml")) else os.path.join("configs", f"{name}.yaml")
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception:
            return {}

try:  # pragma: no cover
    from rice.utils.logging import Logger, format_mean_std, get_logger

    _HAS_LOGGING = True
except Exception:  # pragma: no cover
    import logging as _logging

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = 20):  # type: ignore
        logger = _logging.getLogger(name)
        if not logger.handlers:
            handler = _logging.StreamHandler()
            handler.setFormatter(_logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    class Logger:  # type: ignore
        def __init__(self, out_dir: Optional[str] = None, name: str = "rice", config: Optional[Dict] = None, **_: Any) -> None:
            self.out_dir = out_dir
            self.history: Dict[str, List[float]] = {}
            self.timers: Dict[str, float] = {}

        def record(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                try:
                    self.history.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue

        def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
            flat: Dict[str, Any] = {}
            for key, value in (data or {}).items():
                flat[f"{prefix}{key}" if prefix else key] = value
            self.record(**flat)

        def timer_start(self, name: str) -> float:
            self.timers.setdefault(f"{name}:start", time.time())
            self.timers[name] = time.time()
            return self.timers[name]

        def timer_end(self, name: str, accumulate: bool = True) -> float:
            start = self.timers.get(name, time.time())
            elapsed = time.time() - start
            if accumulate:
                self.timers[f"{name}:total"] = self.timers.get(f"{name}:total", 0.0) + elapsed
            return elapsed

        def dump(self, filename: str = "progress.json") -> Optional[str]:
            if not self.out_dir:
                return None
            path = os.path.join(self.out_dir, filename)
            save_json({"history": self.history, "timers": self.timers}, path)
            return path

        def close(self) -> None:
            self.dump()

    def format_mean_std(values: Sequence[float], decimals: int = 2) -> str:  # type: ignore
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.{decimals}f}"
        try:
            import numpy as _np

            return f"{_np.mean(vals):.{decimals}f} +- {_np.std(vals):.{decimals}f}"
        except Exception:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            return f"{mean:.{decimals}f} +- {var ** 0.5:.{decimals}f}"

try:  # pragma: no cover
    from rice.utils.seeding import seed_from, set_seed

    _HAS_SEEDING = True
except Exception:  # pragma: no cover
    import random as _random

    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        _random.seed(int(seed))
        try:
            import numpy as _np

            _np.random.seed(int(seed))
        except Exception:
            pass
        try:
            import torch as _torch

            _torch.manual_seed(int(seed))
        except Exception:
            pass
        return int(seed)

    def seed_from(base_seed: int, *offsets: int) -> int:  # type: ignore
        value = int(base_seed) & 0xFFFFFFFF
        for offset in offsets:
            value = (value * 2654435761 + int(offset) + 12345) & 0xFFFFFFFF
        return int(value)

try:  # pragma: no cover
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
    def make_env(env_id: str, **kwargs: Any):  # type: ignore
        raise ImportError("rice.envs.make_env is unavailable (install gym / MuJoCo).")

    def available_envs() -> List[str]:  # type: ignore
        return []

    def env_metadata(env_id: str, probe: bool = True) -> Dict[str, Any]:  # type: ignore
        return {}

    def env_backend(env: Any) -> str:  # type: ignore
        return "unknown"

    def d_max_for(env_id: str, default: Optional[float] = None) -> Optional[float]:  # type: ignore
        return default

    def resolve_env_spec(name: str):  # type: ignore
        return None

try:  # pragma: no cover
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
    import re as _re

    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        if env_id is None:
            return "default"
        key = str(env_id).strip().lower()
        key = key.split("/")[-1]
        key = _re.sub(r"\.ya?ml$", "", key)
        key = _re.sub(r"-v\d+$", "", key)
        key = key.replace("-", "_")
        return key or "default"

    def build_policy(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.models.policies is unavailable (install torch).")

    def load_policy(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.models.policies is unavailable (install torch).")

    def save_policy(*args: Any, **kwargs: Any):  # type: ignore
        return None

    def policy_arch(env_id: str) -> Tuple[int, ...]:  # type: ignore
        return (64, 64)

    def sb3_policy_kwargs(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore
        return {}

try:  # pragma: no cover
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
        state_importance,
    )

    _HAS_MASK_NETWORK = True
except Exception:  # pragma: no cover
    def build_mask_network(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.explanation.mask_network is unavailable (install torch).")

    def load_mask_network(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.explanation.mask_network is unavailable (install torch).")

    def save_mask_network(*args: Any, **kwargs: Any):  # type: ignore
        return None

    def state_importance(*args: Any, **kwargs: Any):  # type: ignore
        import numpy as _np

        return _np.ones(0, dtype="float32")

try:  # pragma: no cover
    from rice.explanation.mask_trainer import (
        DEFAULT_ALPHA,
        MaskEnv,
        MaskPPOConfig,
        MaskTrainer,
        RolloutBatch,
        compute_gae,
        make_mask_env,
        train_mask_network,
    )

    _HAS_MASK_TRAINER = True
except Exception:  # pragma: no cover
    DEFAULT_ALPHA = 1e-4

    def train_mask_network(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.explanation.mask_trainer is unavailable (install torch).")

    class MaskTrainer:  # type: ignore
        pass

    class MaskPPOConfig:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class MaskEnv:  # type: ignore
        pass

    class RolloutBatch:  # type: ignore
        pass

    def compute_gae(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.explanation.mask_trainer is unavailable.")

    def make_mask_env(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.explanation.mask_trainer is unavailable.")

try:  # pragma: no cover
    from rice.baselines.statemask_r import (
        DEFAULT_MASK_SAMPLES,
        samples_for as statemask_samples_for,
        train_statemask_network,
    )

    _HAS_STATEMASK = True
except Exception:  # pragma: no cover
    DEFAULT_MASK_SAMPLES: Dict[str, int] = {}

    def statemask_samples_for(env_id: str, default: int = 300_000) -> int:  # type: ignore
        return int(default)

    def train_statemask_network(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.baselines.statemask_r is unavailable (install torch).")

try:  # pragma: no cover
    from rice.refining.ppo_refine import (
        DEFAULT_GAE_LAMBDA,
        DEFAULT_GAMMA,
        DEFAULT_PPO_LR,
        evaluate_refined_policy,
        refine_policy,
        unpack_reset,
        unpack_step,
    )

    _HAS_REFINER = True
except Exception:  # pragma: no cover
    DEFAULT_PPO_LR = 3e-4
    DEFAULT_GAMMA = 0.99
    DEFAULT_GAE_LAMBDA = 0.95

    def refine_policy(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.refining.ppo_refine is unavailable (install torch).")

    def evaluate_refined_policy(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("rice.refining.ppo_refine is unavailable (install torch).")

    def unpack_reset(result: Any):  # type: ignore
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return result, {}

    def unpack_step(result: Any):  # type: ignore
        if isinstance(result, tuple) and len(result) == 5:
            return result
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, bool(done), False, info or {}
        return result, 0.0, False, False, {}


# --------------------------------------------------------------------------------------
# Paper constants / reference values (trend validation only)
# --------------------------------------------------------------------------------------
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
DEFAULT_OUT_DIR = os.path.join("results", "mask")
DEFAULT_CHECKPOINT_DIR = "policies"

#: Fixed mask-training sample budgets from Table 4 (identical for Ours and StateMask).
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

#: Wall-clock reference times (seconds) for the two mask trainers, Table 4.
TABLE4_TIMES: Dict[str, Dict[str, float]] = {
    "hopper": {"ours": 12_426.0, "statemask": 15_393.0},
    "halfcheetah": {"ours": 1_317.0, "statemask": 1_579.0},
    "cage2": {"ours": 65_400.0, "statemask": 79_382.0},
}

#: Reported training-time reduction of Algorithm 1 over StateMask (16.8%).
PAPER_TIME_REDUCTION = 0.168

#: Per-application hyper-parameters from Table 3 (alpha treated as insensitive ->
#: Table 3 default 1e-4 is used, the conflicting §4.3 prose value 0.01 is sweep-only).
TABLE3: Dict[str, Dict[str, float]] = {
    "hopper": {"alpha": 1e-4},
    "walker2d": {"alpha": 1e-4},
    "reacher": {"alpha": 1e-4},
    "halfcheetah": {"alpha": 1e-4},
    "sparse_hopper": {"alpha": 1e-4},
    "sparse_halfcheetah": {"alpha": 1e-4},
    "selfish_mining": {"alpha": 1e-4},
    "cage2": {"alpha": 1e-4},
    "autodriving": {"alpha": 1e-4},
}

ALPHA_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)
MASK_METHODS: Tuple[str, ...] = ("ours", "statemask")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def deep_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Nested ``dict`` / attribute lookup tolerating missing keys."""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return default if current is None else current


def is_negative_env(env_id: str) -> bool:
    """True for applications whose reward scale is signed (reacher, cage2)."""
    return normalize_env_key(env_id) in ("reacher", "cage2")


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Fixed mask-training sample budget for an application (Table 4)."""
    key = normalize_env_key(env_id)
    cfg_budget = deep_get(cfg or {}, "explanation", "total_timesteps", default=None)
    if cfg_budget:
        try:
            return int(cfg_budget)
        except (TypeError, ValueError):
            pass
    if key in TABLE4_SAMPLES:
        return int(TABLE4_SAMPLES[key])
    if _HAS_STATEMASK:
        try:
            return int(statemask_samples_for(key))
        except Exception:
            pass
    return 300_000


def alpha_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Blinding-bonus coefficient alpha (Table 3 default; exposed for the sweep)."""
    cfg_alpha = deep_get(cfg or {}, "explanation", "alpha", default=None)
    if cfg_alpha is not None:
        try:
            return float(cfg_alpha)
        except (TypeError, ValueError):
            pass
    return float(TABLE3.get(normalize_env_key(env_id), {}).get("alpha", DEFAULT_ALPHA))


def checkpoint_path(env_id: str, seed: Optional[int] = None, out_dir: str = DEFAULT_CHECKPOINT_DIR, method: str = "ours") -> str:
    """``policies/<key>_mask[_seedN].pt`` (StateMask checkpoints get a suffix)."""
    key = normalize_env_key(env_id)
    suffix = "" if normalize_env_key(method) in ("ours", "rice") else f"_{normalize_env_key(method)}"
    seed_suffix = "" if seed is None else f"_seed{int(seed)}"
    return os.path.join(out_dir, f"{key}_mask{suffix}{seed_suffix}.pt")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of numpy / torch / dataclass payloads to JSON types."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def _cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    return deep_get(cfg or {}, *keys, default=default)


# --------------------------------------------------------------------------------------
# Environment / policy construction
# --------------------------------------------------------------------------------------
def build_experiment_env(env_id: str, cfg: Optional[Dict[str, Any]] = None, seed: int = 0, mode: str = "train", **kwargs: Any) -> Any:
    """Create the environment used for Stage-1 mask training."""
    env_kwargs: Dict[str, Any] = {}
    if cfg:
        env_kwargs["normalize"] = _cfg_get(cfg, "env", "normalize_obs", default=None)
        max_steps = _cfg_get(cfg, "env", "max_episode_steps", default=None)
        if max_steps:
            env_kwargs["max_episode_steps"] = int(max_steps)
    env_kwargs.update(kwargs)
    env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
    return make_env(env_id, seed=seed, mode=mode, **env_kwargs)


def build_target_policy(env: Any, env_id: str, cfg: Optional[Dict[str, Any]] = None, device: str = DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Instantiate the frozen target policy pi (per-application architecture)."""
    hidden = _cfg_get(cfg, "target", "hidden_sizes", default=None)
    activation = _cfg_get(cfg, "target", "activation", default=None)
    backend = _cfg_get(cfg, "target", "backend", default="auto")
    obs_dim = deep_get(env_metadata(env_id), "obs_dim", default=None) if _HAS_ENVS else None
    action_dim = deep_get(env_metadata(env_id), "action_dim", default=None) if _HAS_ENVS else None
    try:
        return build_policy(
            env_id=env_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            observation_space=getattr(env, "observation_space", None),
            action_space=getattr(env, "action_space", None),
            kind="policy",
            backend=backend or "auto",
            hidden_sizes=hidden,
            activation=activation,
            device=device,
            **kwargs,
        )
    except Exception:
        try:  # SB3 fallback keeps the pipeline usable without a native builder
            from stable_baselines3 import PPO  # type: ignore

            return PPO("MlpPolicy", env, device=device, policy_kwargs=sb3_policy_kwargs(env_id=env_id))
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                f"Unable to build target policy for '{env_id}': {exc}"
            ) from exc


def build_or_load_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = DEFAULT_DEVICE,
    checkpoint: Optional[str] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Load the frozen pre-trained pi from a checkpoint, else build a fresh one."""
    path = checkpoint or _cfg_get(cfg, "target", "checkpoint", default=None) or checkpoint_path(env_id, seed=None, method="policy")
    candidates: List[str] = []
    if checkpoint:
        candidates.append(checkpoint)
    if path:
        candidates.append(path)
    key = normalize_env_key(env_id)
    candidates.extend(
        [
            os.path.join(DEFAULT_CHECKPOINT_DIR, f"{key}_ppo.zip"),
            os.path.join(DEFAULT_CHECKPOINT_DIR, f"{key}_ppo.pt"),
            os.path.join(DEFAULT_CHECKPOINT_DIR, f"{key}_policy.pt"),
        ]
    )
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            policy = load_policy(
                candidate,
                env_id=env_id,
                observation_space=getattr(env, "observation_space", None),
                action_space=getattr(env, "action_space", None),
                device=device,
            )
            if logger is not None:
                logger.info(f"Loaded frozen target policy from {candidate}")
            return policy
        except Exception:
            try:
                from stable_baselines3 import PPO  # type: ignore

                model = PPO.load(candidate, env=env, device=device)
                if logger is not None:
                    logger.info(f"Loaded SB3 PPO checkpoint from {candidate}")
                return model
            except Exception:
                continue
    if logger is not None:
        logger.warning(
            f"No target-policy checkpoint found for '{env_id}'; building an untrained pi. "
            "Run scripts/train_target.py first for meaningful explanation scores."
        )
    return build_target_policy(env, env_id, cfg=cfg, device=device, **kwargs)


def pretrain_target_policy(
    env: Any,
    env_id: str,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Pre-train pi with plain PPO (RICE contributions disabled) - the No-Refine regime."""
    set_seed(seed)
    try:
        policy, _ = refine_policy(
            env,
            policy=None,
            total_timesteps=int(total_timesteps),
            env_id=env_id,
            device=device,
            seed=seed,
            use_mixed_init=False,
            use_rnd=False,
            p=0.0,
            lam=0.0,
            logger=logger,
            **kwargs,
        )
        return policy
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Native pre-training unavailable ({exc}); trying SB3 PPO.")
        from stable_baselines3 import PPO  # type: ignore

        model = PPO("MlpPolicy", env, device=device, seed=seed, policy_kwargs=sb3_policy_kwargs(env_id=env_id))
        model.learn(total_timesteps=int(total_timesteps))
        return model


# --------------------------------------------------------------------------------------
# Stage-1 trainers (Ours = Algorithm 1 vanilla PPO + blinding bonus)
# --------------------------------------------------------------------------------------
@dataclass
class MaskArtifacts:
    """Trained Stage-1 explanation artefacts for one (env, method, seed)."""

    method: str
    env_id: str
    seed: int
    mask_net: Any = None
    trainer: Any = None
    train_time: float = 0.0
    samples: int = 0
    seconds_per_sample: float = 0.0
    checkpoint: Optional[str] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)
    time_report: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_net: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "method": self.method,
            "env_id": self.env_id,
            "seed": int(self.seed),
            "train_time": float(self.train_time),
            "samples": int(self.samples),
            "seconds_per_sample": float(self.seconds_per_sample),
            "checkpoint": self.checkpoint,
            "final_metrics": _jsonable(self.final_metrics),
            "time_report": _jsonable(self.time_report),
            "extra": _jsonable(self.extra),
        }
        if include_net and self.mask_net is not None:
            payload["mask_net"] = self.mask_net
        return payload

    def format(self, decimals: int = 2) -> str:
        return (
            f"{self.env_id:<16} {self.method:<10} seed={self.seed} "
            f"samples={self.samples:>9,} time={self.train_time:>9.{decimals}f}s "
            f"({self.seconds_per_sample * 1e3:.3f} ms/sample)"
        )


def train_ours_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    checkpoint: Optional[str] = None,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    progress: bool = False,
    **kwargs: Any,
) -> MaskArtifacts:
    """Algorithm 1: train the mask network with VANILLA PPO + blinding bonus alpha*a_t^m."""
    key = normalize_env_key(env_id)
    samples = int(total_timesteps or mask_budget_for(key, cfg))
    alpha_value = float(alpha if alpha is not None else alpha_for(key, cfg))
    save_path = checkpoint if checkpoint is not None else checkpoint_path(key, seed=seed, method="ours")

    start = time.time()
    mask_net, trainer = train_mask_network(
        env,
        policy,
        total_timesteps=samples,
        alpha=alpha_value,
        env_id=key,
        config=cfg.get("explanation") if isinstance(cfg, dict) else None,
        logger=logger,
        save_path=save_path,
        seed=seed,
        device=device,
        progress=progress,
        store_dataset=False,
        **kwargs,
    )
    elapsed = float(getattr(trainer, "total_time", 0.0) or (time.time() - start))
    per_sample = float(getattr(trainer, "seconds_per_sample", 0.0) or (elapsed / max(samples, 1)))

    final_metrics: Dict[str, float] = {}
    try:
        summary = trainer.summary() if hasattr(trainer, "summary") else {}
        history = getattr(trainer, "history", None) or summary.get("history", {})
        for name in ("loss", "policy_loss", "value_loss", "entropy", "mean_bonus", "mean_reward", "blind_fraction"):
            series = (history or {}).get(name) if isinstance(history, dict) else None
            if series:
                final_metrics[f"final_{name}"] = float(series[-1]) if not isinstance(series[-1], (list, tuple)) else float(series[-1][-1])
    except Exception:
        pass

    return MaskArtifacts(
        method="ours",
        env_id=key,
        seed=int(seed),
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=samples,
        seconds_per_sample=per_sample,
        checkpoint=save_path if save_path and os.path.exists(save_path) else None,
        final_metrics=final_metrics,
        time_report=_jsonable(getattr(trainer, "time_report", lambda: {})()) if hasattr(trainer, "time_report") else {},
        extra={"alpha": alpha_value, "backend": env_backend(env) if _HAS_ENVS else "unknown"},
    )


def train_statemask_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: Optional[int] = None,
    alpha: float = 0.01,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    checkpoint: Optional[str] = None,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    progress: bool = False,
    **kwargs: Any,
) -> MaskArtifacts:
    """StateMask (primal-dual) mask trainer - the Table-4 efficiency comparison point."""
    key = normalize_env_key(env_id)
    samples = int(total_timesteps or mask_budget_for(key, cfg))
    save_path = checkpoint if checkpoint is not None else checkpoint_path(key, seed=seed, method="statemask")

    start = time.time()
    try:
        mask_net, trainer = train_statemask_network(
            env,
            policy,
            total_timesteps=samples,
            alpha=alpha,
            env_id=key,
            config=cfg.get("baselines", {}).get("statemask") if isinstance(cfg, dict) else None,
            logger=logger,
            save_path=save_path,
            seed=seed,
            device=device,
            progress=progress,
            store_dataset=False,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"StateMask baseline training unavailable: {exc}") from exc
    elapsed = float(getattr(trainer, "total_time", 0.0) or (time.time() - start))
    per_sample = float(getattr(trainer, "seconds_per_sample", 0.0) or (elapsed / max(samples, 1)))

    dual_report = {}
    if hasattr(trainer, "dual_report"):
        try:
            dual_report = _jsonable(trainer.dual_report())
        except Exception:
            dual_report = {}

    return MaskArtifacts(
        method="statemask",
        env_id=key,
        seed=int(seed),
        mask_net=mask_net,
        trainer=trainer,
        train_time=elapsed,
        samples=samples,
        seconds_per_sample=per_sample,
        checkpoint=save_path if save_path and os.path.exists(save_path) else None,
        final_metrics={"final_alpha": float(deep_get(dual_report, "alpha", default=alpha))},
        time_report=_jsonable(getattr(trainer, "time_report", lambda: {})()) if hasattr(trainer, "time_report") else {},
        extra={"alpha_init": float(alpha), "dual_report": dual_report, "backend": env_backend(env) if _HAS_ENVS else "unknown"},
    )


def train_explanation(
    method: str,
    env: Any,
    policy: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    logger: Any = None,
    mask_timesteps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    progress: bool = False,
    alpha: Optional[float] = None,
    **kwargs: Any,
) -> MaskArtifacts:
    """Dispatch Stage-1 explanation training for ``method`` in {ours, statemask}."""
    key = normalize_env_key(method)
    checkpoint = None
    if checkpoint_dir:
        checkpoint = checkpoint_path(env_id, seed=seed, out_dir=checkpoint_dir, method=key)
    if key in ("ours", "rice"):
        return train_ours_explanation(
            env,
            policy,
            env_id,
            total_timesteps=mask_timesteps,
            alpha=alpha,
            seed=seed,
            device=device,
            checkpoint=checkpoint,
            logger=logger,
            cfg=cfg,
            progress=progress,
            **kwargs,
        )
    if key in ("statemask", "state_mask", "statemask_r"):
        return train_statemask_explanation(
            env,
            policy,
            env_id,
            total_timesteps=mask_timesteps,
            alpha=float(alpha if alpha is not None else _cfg_get(cfg, "baselines", "statemask", "alpha_init", default=0.01)),
            seed=seed,
            device=device,
            checkpoint=checkpoint,
            logger=logger,
            cfg=cfg,
            progress=progress,
            **kwargs,
        )
    raise ValueError(f"Unknown explanation method '{method}'. Expected one of {MASK_METHODS}.")


# --------------------------------------------------------------------------------------
# Evaluation / reporting
# --------------------------------------------------------------------------------------
def blob_time_reduction(artifacts: Dict[str, MaskArtifacts]) -> Dict[str, Any]:
    """Ours vs StateMask mask-training wall-clock comparison (Table 4 / 16.8% claim)."""
    ours = artifacts.get("ours")
    baseline = artifacts.get("statemask")
    report: Dict[str, Any] = {"ours_time": None, "statemask_time": None, "reduction": None, "paper_reduction": PAPER_TIME_REDUCTION}
    if ours is not None:
        report["ours_time"] = float(ours.train_time)
        report["ours_seconds_per_sample"] = float(ours.seconds_per_sample)
    if baseline is not None:
        report["statemask_time"] = float(baseline.train_time)
        report["statemask_seconds_per_sample"] = float(baseline.seconds_per_sample)
    if ours is not None and baseline is not None and baseline.train_time > 0:
        reduction = 1.0 - (ours.train_time / baseline.train_time)
        report["reduction"] = float(reduction)
        report["ours_faster"] = bool(reduction > 0.0)
    return report


def evaluate_mask_scores(env: Any, policy: Any, mask_net: Any, env_id: str, n_episodes: int = 3, max_steps: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Sanity check: roll pi, score every visited state, report blind-fraction stats."""
    if mask_net is None or policy is None:
        return {}
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return {}
    rng = np.random.RandomState(seed)
    import numpy as _np

    horizon = int(max_steps or deep_get(env_metadata(env_id) if _HAS_ENVS else {}, "max_episode_steps", default=1000) or 1000)
    all_scores: List[float] = []
    for episode in range(int(n_episodes)):
        obs, _ = unpack_reset(env.reset())
        observations = [_np.asarray(obs, dtype=_np.float32).ravel()]
        for _ in range(horizon):
            action = _policy_action(policy, obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = unpack_step(env.step(action))
            observations.append(_np.asarray(obs, dtype=_np.float32).ravel())
            if terminated or truncated:
                break
        try:
            scores = _np.asarray(state_importance(mask_net, _np.stack(observations)), dtype=_np.float64).ravel()
            all_scores.extend(scores.tolist())
        except Exception:
            continue
    if not all_scores:
        return {}
    scores = _np.asarray(all_scores, dtype=_np.float64)
    return {
        "n_states_scored": int(scores.size),
        "mean_importance": float(_np.mean(scores)),
        "std_importance": float(_np.std(scores)),
        "blind_fraction": float(_np.mean(scores < 0.5)),
    }


def _policy_action(policy: Any, observation: Any, deterministic: bool = True) -> Any:
    """Interface-agnostic action selection (SB3 ``predict`` / native ``act`` / callable)."""
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
    raise TypeError(f"Unsupported policy object of type {type(policy)!r}")


# --------------------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------------------
def run_mask(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    methods: Sequence[str] = MASK_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    mask_timesteps: Optional[int] = None,
    alpha: Optional[float] = None,
    device: str = DEFAULT_DEVICE,
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    pretrain_timesteps: Optional[int] = None,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    save: bool = True,
    eval_episodes: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train the Stage-1 mask network(s) for one application and report timings."""
    key = normalize_env_key(env_id)
    out_dir = out_dir or DEFAULT_OUT_DIR
    ensure_dir(out_dir)
    log = logger or get_logger("rice.train_mask", out_dir=out_dir)

    report: Dict[str, Any] = {
        "experiment": "stage1_mask_training",
        "env_id": key,
        "methods": list(methods),
        "seeds": list(seeds),
        "mask_timesteps": mask_timesteps or mask_budget_for(key, cfg),
        "alpha": alpha if alpha is not None else alpha_for(key, cfg),
        "results": {},
        "timings": {},
        "efficiency": {},
        "errors": {},
        "checkpoints": {},
    }

    for method in methods:
        records: List[Dict[str, Any]] = []
        retrain: List[MaskArtifacts] = []
        for seed in seeds:
            run_seed = seed_from(int(seed), 1) if _HAS_SEEDING else int(seed)
            set_seed(run_seed)
            env = None
            try:
                env = build_experiment_env(key, cfg=cfg, seed=run_seed, mode="train")
                if method in ("ours", "rice") and pretrain_timesteps:
                    policy = pretrain_target_policy(
                        env,
                        key,
                        total_timesteps=int(pretrain_timesteps),
                        seed=run_seed,
                        device=device,
                        logger=log,
                        cfg=cfg,
                    )
                    policy = _maybe_serialise_policy(policy, key, seed, out_dir, method="policy")
                else:
                    policy = build_or_load_policy(env, key, cfg=cfg, device=device, checkpoint=checkpoint, logger=log)

                artifacts = train_explanation(
                    method,
                    env,
                    policy,
                    key,
                    cfg=cfg,
                    seed=run_seed,
                    device=device,
                    logger=log,
                    mask_timesteps=mask_timesteps,
                    checkpoint_dir=checkpoint_dir,
                    progress=progress,
                    alpha=alpha,
                    **kwargs,
                )
                record = artifacts.to_dict()
                if method in ("ours", "rice"):
                    record["importance_stats"] = evaluate_mask_scores(
                        env, policy, artifacts.mask_net, key, n_episodes=eval_episodes, seed=run_seed
                    )
                records.append(record)
                retrain.append(artifacts)
                log.info(f"[{key}/{method}] seed={seed} " + artifacts.format())
            except Exception as exc:  # keep the sweep alive
                tb = traceback.format_exc(limit=3)
                report["errors"].setdefault(method, []).append({"seed": int(seed), "error": str(exc), "traceback": tb})
                log.warning(f"[{key}/{method}] seed={seed} failed: {exc}")
            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass

        report["results"][method] = records
        times = [r["train_time"] for r in records if r.get("train_time")]
        report["timings"][method] = {
            "mean_time": _mean(times),
            "std_time": _std(times),
            "mean_seconds_per_sample": _mean([r.get("seconds_per_sample", 0.0) for r in records]),
            "n_runs": len(times),
        }
        report["checkpoints"][method] = [r.get("checkpoint") for r in records]

    # Table-4 style efficiency comparison (Ours vs StateMask) using mean timings.
    ours_time = deep_get(report["timings"], "ours", "mean_time", default=None)
    sm_time = deep_get(report["timings"], "statemask", "mean_time", default=None)
    efficiency: Dict[str, Any] = {
        "ours_time": ours_time,
        "statemask_time": sm_time,
        "reduction": None,
        "paper_reduction": PAPER_TIME_REDUCTION,
        "reference": TABLE4_TIMES.get(key, {}),
    }
    if ours_time and sm_time and sm_time > 0:
        reduction = 1.0 - (float(ours_time) / float(sm_time))
        efficiency["reduction"] = float(reduction)
        efficiency["ours_faster"] = bool(reduction > 0.0)
        efficiency["matches_paper_direction"] = bool(reduction > 0.0)
    report["efficiency"] = efficiency

    if save:
        path = os.path.join(out_dir, f"mask_{key}.json")
        save_json(report, path)
        log.info(f"Wrote Stage-1 report to {path}")
    return report


def _maybe_serialise_policy(policy: Any, env_id: str, seed: int, out_dir: str, method: str = "policy") -> Any:
    """Persist a freshly pre-trained pi so later stages can reload the frozen weights."""
    try:
        ckpt_dir = os.path.join(out_dir, "policies")
        ensure_dir(ckpt_dir)
        path = os.path.join(ckpt_dir, f"{normalize_env_key(env_id)}_{method}_seed{int(seed)}.pt")
        if hasattr(policy, "save"):
            try:
                policy.save(path.replace(".pt", ".zip"))
                return policy
            except Exception:
                pass
        save_policy(policy, path, env_id=env_id, kind="policy", seed=int(seed))
    except Exception:
        pass
    return policy


def run_mask_multi(
    env_ids: Sequence[str] = DEFAULT_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train Stage-1 explanations for several applications."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    ensure_dir(out_dir)
    log = logger or get_logger("rice.train_mask", out_dir=out_dir)
    combined: Dict[str, Any] = {"experiment": "stage1_mask_training", "envs": list(env_ids), "reports": {}, "errors": {}}
    for env_id in env_ids:
        try:
            combined["reports"][normalize_env_key(env_id)] = run_mask(
                env_id, cfg=cfg, out_dir=out_dir, logger=log, progress=progress, **kwargs
            )
        except Exception as exc:
            combined["errors"][normalize_env_key(env_id)] = str(exc)
            log.warning(f"Env '{env_id}' mask training failed: {exc}")
    path = os.path.join(out_dir, "mask_all.json")
    save_json(combined, path)
    return combined


def efficiency_report(env_id: str, artifacts: Dict[str, MaskArtifacts], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Public helper mirroring `experiments/exp1_fidelity_efficiency.efficiency_report`."""
    report = blob_time_reduction(artifacts)
    reference = TABLE4_TIMES.get(normalize_env_key(env_id), {})
    report["reference"] = reference
    if reference.get("ours") and reference.get("statemask"):
        reference_reduction = 1.0 - (reference["ours"] / reference["statemask"])
        report["reference_reduction"] = float(reference_reduction)
    report["matches_paper_direction"] = bool(report.get("reduction") is not None and report["reduction"] > 0.0)
    return report


# --------------------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------------------
def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    """Human-readable summary of a Stage-1 mask-training run."""
    lines: List[str] = []
    env_id = report.get("env_id") or report.get("envs") or "?"
    lines.append("=" * 78)
    lines.append(f"Stage-1 mask-network training (Algorithm 1) - env: {env_id}")
    lines.append("=" * 78)
    lines.append(f"mask sample budget : {report.get('mask_timesteps')}")
    lines.append(f"alpha (blinding)   : {report.get('alpha')}")
    lines.append("-" * 78)
    lines.append(f"{'method':<12} {'mean time (s)':>16} {'ms/sample':>12} {'runs':>6}")
    lines.append("-" * 78)
    timings = report.get("timings", {}) or {}
    for method, timing in timings.items():
        mean_time = timing.get("mean_time", float("nan"))
        per_sample = timing.get("mean_seconds_per_sample", 0.0) * 1e3
        lines.append(f"{method:<12} {mean_time:>16.{decimals}f} {per_sample:>12.3f} {timing.get('n_runs', 0):>6}")

    efficiency = report.get("efficiency", {}) or {}
    if efficiency.get("reduction") is not None:
        lines.append("-" * 78)
        lines.append(
            "Table-4 efficiency : ours=%.{0}f s vs statemask=%.{0}f s -> reduction=%.2f%% "
            "(paper %.1f%% reduction)".format(decimals)
            % (
                efficiency.get("ours_time") or float("nan"),
                efficiency.get("statemask_time") or float("nan"),
                100.0 * efficiency["reduction"],
                100.0 * PAPER_TIME_REDUCTION,
            )
        )
        lines.append(f"                     ours faster than StateMask: {efficiency.get('ours_faster')}")
    for method, records in (report.get("results", {}) or {}).items():
        for record in records:
            stats = record.get("importance_stats") or {}
            if stats:
                lines.append(
                    f"  [{env_id}/{method}/seed={record.get('seed')}] "
                    f"mean importance={stats.get('mean_importance', float('nan')):.4f} "
                    f"blind fraction={stats.get('blind_fraction', float('nan')):.4f}"
                )
    errors = report.get("errors", {}) or {}
    if errors:
        lines.append("-" * 78)
        for method, items in errors.items():
            for item in items:
                lines.append(f"  ERROR [{env_id}/{method}/seed={item.get('seed')}]: {item.get('error')}")
    lines.append("=" * 78)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_mask",
        description="RICE Stage 1: train the step-level mask-network explanation (Algorithm 1).",
    )
    parser.add_argument("--env", "--env-id", dest="env", default=None, help="Application key (e.g. hopper).")
    parser.add_argument("--envs", default=None, help="Comma-separated list of application keys.")
    parser.add_argument("--config", default="default", help="Config name/path merged over configs/default.yaml.")
    parser.add_argument("--methods", default=",".join(MASK_METHODS), help="Explanation methods (ours,statemask).")
    parser.add_argument("--timesteps", type=int, default=None, help="Mask-training sample budget (Table 4 default).")
    parser.add_argument("--alpha", type=float, default=None, help="Blinding-bonus coefficient (Table 3 default).")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated seeds.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device (cpu/cuda).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for JSON reports.")
    parser.add_argument("--checkpoint", default=None, help="Pre-trained target-policy checkpoint for pi.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for mask-network checkpoints.")
    parser.add_argument("--pretrain-timesteps", type=int, default=None, help="If set, pre-train pi before explaining it.")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Rollouts used for the importance sanity check.")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not write JSON reports.")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar during training.")
    parser.add_argument("--list-envs", action="store_true", help="List available environments and exit.")
    parser.set_defaults(save=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list_envs:
        envs = available_envs() if _HAS_ENVS else list(DEFAULT_ENVS)
        print("Available RICE environments:", ", ".join(envs) if envs else "(none - gym/MuJoCo missing)")
        return 0

    cfg = get_config(args.config) if args.config else {}
    device = args.device
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if str(s).strip())
    methods = tuple(m.strip().lower() for m in str(args.methods).split(",") if m.strip())

    if args.envs:
        env_ids = [e.strip() for e in str(args.envs).split(",") if e.strip()]
    elif args.env:
        env_ids = [args.env]
    else:
        env_ids = list(DEFAULT_ENVS)

    ensure_dir(args.out_dir)
    logger = get_logger("rice.train_mask", out_dir=args.out_dir)

    last_report: Optional[Dict[str, Any]] = None
    for env_id in env_ids:
        try:
            last_report = run_mask(
                env_id,
                cfg=cfg,
                methods=methods,
                seeds=seeds,
                mask_timesteps=args.timesteps,
                alpha=args.alpha,
                device=device,
                out_dir=args.out_dir,
                logger=logger,
                progress=args.progress,
                pretrain_timesteps=args.pretrain_timesteps,
                checkpoint=args.checkpoint,
                checkpoint_dir=args.checkpoint_dir,
                save=args.save,
                eval_episodes=args.eval_episodes,
            )
            print(format_report(last_report))
        except Exception as exc:
            logger.error(f"Stage-1 training failed for '{env_id}': {exc}")
            if not env_ids[1:]:
                return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
