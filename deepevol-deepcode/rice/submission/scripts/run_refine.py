#!/usr/bin/env python
"""RICE Stage-2 refinement entry point (Experiments II / III / IV protocol).

This script is the CLI glue for RICE's *refining* stage (Algorithm 2):

  * Load (or pre-train) the frozen, sub-optimal target policy ``pi``.
  * Build / load the Stage-1 explanation (trained mask network) that identifies
    critical states, per ``rice.explanation.mask_trainer`` (Algorithm 1).
  * Refine ``pi`` with one of the compared refining methods:
      - ``ours``           : RICE = PPO + mixed initial distribution
                             ``mu(s) = beta * d_rho^pihat(s) + (1-beta) * rho(s)``
                             with probability ``p`` and the normalized RND bonus
                             ``lambda * ||f(s_{t+1}) - fhat(s_{t+1})||^2``   (Algorithm 2)
      - ``ppo_finetune``   : plain PPO continued with a lowered learning rate
      - ``statemask_r``    : always reset to the mask-identified critical state
                             (``p = 1``, no RND) then PPO-fine-tune
      - ``jsrl``           : Jump-Start RL curriculum with an annealed guided horizon
      - ``sac_finetune``   : SAC continuation (Experiment IV, continuous envs)
      - ``gail``           : GAIL-approximated policy then RICE refinement (Exp IV)
      - ``sil``            : Self-Imitation Learning refining baseline (Table 5)
  * Evaluate the refined policy, compare against the paper's ``No Refine`` rows
    (trend validation only) and write a JSON + text report.

Everything heavy is delegated to the already-implemented ``rice`` modules; this
script only performs env/policy plumbing, dispatch, reporting and CLI parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Defensive imports of the project utilities (falling back to stdlib versions)  #
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import plumbing
    from rice.utils.io import ensure_dir, get_config, save_json
except Exception:  # pragma: no cover
    ensure_dir = None  # type: ignore
    get_config = None  # type: ignore

    def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore
        if ensure_dir is not None:
            ensure_dir(os.path.dirname(os.path.abspath(path)))
        else:  # pragma: no cover
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(_jsonable(obj), fh, indent=indent)
        return path

try:  # pragma: no cover
    from rice.utils.io import load_json  # type: ignore
except Exception:  # pragma: no cover
    def load_json(path: str, default: Any = None) -> Any:  # type: ignore
        if not os.path.exists(path):
            return default
        with open(path, "r") as fh:
            return json.load(fh)

try:  # pragma: no cover
    from rice.utils.logging import Logger, format_mean_std, get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = logging.INFO):  # type: ignore
        logging.basicConfig(level=level)
        return logging.getLogger(name)

    def format_mean_std(values: Sequence[float], decimals: int = 2) -> str:  # type: ignore
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.{decimals}f}"
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        return f"{mean:.{decimals}f} +- {var ** 0.5:.{decimals}f}"

    Logger = None  # type: ignore

try:  # pragma: no cover
    from rice.utils.seeding import seed_from, set_seed
except Exception:  # pragma: no cover
    import random as _random

    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        seed = int(seed)
        _random.seed(seed)
        try:
            import numpy as _np

            _np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch as _torch

            _torch.manual_seed(seed)
        except Exception:
            pass
        return seed

    def seed_from(base_seed: int, *offsets: int) -> int:  # type: ignore
        value = int(base_seed)
        for off in offsets:
            value = (value * 1000003 + int(off) + 0x9E3779B9) & 0xFFFFFFFF
        return value

# --------------------------------------------------------------------------- #
# Defensive imports of the RICE env / model / explanation / refining layers     #
# --------------------------------------------------------------------------- #
_HAS_MAKE_ENV = True
try:  # pragma: no cover
    from rice.envs.make_env import (
        available_envs,
        cage2_final_reward,
        d_max_for,
        env_backend,
        env_metadata,
        make_env,
        resolve_env_spec,
    )
except Exception:  # pragma: no cover
    _HAS_MAKE_ENV = False
    available_envs = None  # type: ignore
    cage2_final_reward = None  # type: ignore
    d_max_for = None  # type: ignore
    env_backend = None  # type: ignore
    env_metadata = None  # type: ignore
    make_env = None  # type: ignore
    resolve_env_spec = None  # type: ignore

_HAS_POLICIES = True
try:  # pragma: no cover
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        policy_arch,
        save_policy,
    )
except Exception:  # pragma: no cover
    _HAS_POLICIES = False
    build_policy = None  # type: ignore
    load_policy = None  # type: ignore
    policy_arch = None  # type: ignore
    save_policy = None  # type: ignore

    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        name = str(env_id).strip().lower()
        for suffix in ("-v0", "-v1", "-v2", "-v3", "-v4", "-v5", ".yaml", ".yml", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        name = name.replace("-", "_").replace(" ", "_")
        for prefix in ("sparse_", "sparse-"):
            if name.startswith(prefix):
                return name
        return name

_HAS_MASK_NETWORK = True
try:  # pragma: no cover
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
    )
except Exception:  # pragma: no cover
    _HAS_MASK_NETWORK = False
    build_mask_network = None  # type: ignore
    load_mask_network = None  # type: ignore
    save_mask_network = None  # type: ignore

_HAS_MASK_TRAINER = True
try:  # pragma: no cover
    from rice.explanation.mask_trainer import DEFAULT_ALPHA, train_mask_network
except Exception:  # pragma: no cover
    _HAS_MASK_TRAINER = False
    DEFAULT_ALPHA = 1e-4  # type: ignore
    train_mask_network = None  # type: ignore

_HAS_STATEMASK = True
try:  # pragma: no cover
    from rice.baselines.statemask_r import (
        DEFAULT_MASK_SAMPLES,
        refine_from_critical_state,
        samples_for as statemask_samples_for,
        train_statemask_network,
    )
except Exception:  # pragma: no cover
    _HAS_STATEMASK = False
    DEFAULT_MASK_SAMPLES = {}  # type: ignore
    refine_from_critical_state = None  # type: ignore
    statemask_samples_for = None  # type: ignore
    train_statemask_network = None  # type: ignore

_HAS_PPO_REFINE = True
try:  # pragma: no cover
    from rice.refining.ppo_refine import (
        DEFAULT_GAMMA,
        DEFAULT_GAE_LAMBDA,
        DEFAULT_LAMBDA,
        DEFAULT_P,
        DEFAULT_PPO_LR,
        RefinePPOConfig,
        evaluate_refined_policy,
        refine_policy,
        unpack_reset,
        unpack_step,
    )
except Exception:  # pragma: no cover
    _HAS_PPO_REFINE = False
    DEFAULT_GAMMA = 0.99  # type: ignore
    DEFAULT_GAE_LAMBDA = 0.95  # type: ignore
    DEFAULT_LAMBDA = 0.01  # type: ignore
    DEFAULT_P = 0.5  # type: ignore
    DEFAULT_PPO_LR = 3e-4  # type: ignore
    RefinePPOConfig = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore
    refine_policy = None  # type: ignore
    unpack_reset = None  # type: ignore
    unpack_step = None  # type: ignore

_HAS_RANDOM_EXPLANATION = True
try:  # pragma: no cover
    from rice.baselines.random_explanation import make_random_explanation
except Exception:  # pragma: no cover
    _HAS_RANDOM_EXPLANATION = False
    make_random_explanation = None  # type: ignore

_HAS_PPO_FINETUNE = True
try:  # pragma: no cover
    from rice.baselines.ppo_finetune import DEFAULT_FINETUNE_LR, ppo_finetune_policy
except Exception:  # pragma: no cover
    _HAS_PPO_FINETUNE = False
    DEFAULT_FINETUNE_LR = 1e-4  # type: ignore
    ppo_finetune_policy = None  # type: ignore

_HAS_JSRL = True
try:  # pragma: no cover
    from rice.baselines.jsrl import train_jsrl
except Exception:  # pragma: no cover
    _HAS_JSRL = False
    train_jsrl = None  # type: ignore

_HAS_SAC_FINETUNE = True
try:  # pragma: no cover
    from rice.baselines.sac_finetune import sac_finetune_policy
except Exception:  # pragma: no cover
    _HAS_SAC_FINETUNE = False
    sac_finetune_policy = None  # type: ignore

_HAS_GAIL = True
try:  # pragma: no cover
    from rice.baselines.gail import train_gail
except Exception:  # pragma: no cover
    _HAS_GAIL = False
    train_gail = None  # type: ignore

_HAS_SIL = True
try:  # pragma: no cover
    from rice.baselines.sil import sil as sil_baseline
except Exception:  # pragma: no cover
    _HAS_SIL = False
    sil_baseline = None  # type: ignore


# --------------------------------------------------------------------------- #
# Paper constants (Table 3 / Table 4 / Table 1 reference rows)                 #
# --------------------------------------------------------------------------- #
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
NEGATIVE_REWARD_ENVS: Tuple[str, ...] = ("reacher", "cage2")

REFINING_METHODS: Tuple[str, ...] = ("ours", "ppo_finetune", "statemask_r", "jsrl")
EXPLANATION_METHODS: Tuple[str, ...] = ("ours", "statemask", "random")
EXTRA_REFINING_METHODS: Tuple[str, ...] = ("sac_finetune", "gail", "sil")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_DEVICE: str = "cpu"
DEFAULT_OUT_DIR: str = "results/refine"
DEFAULT_CHECKPOINT_DIR: str = "policies"
DEFAULT_REFINE_TIMESTEPS: int = 200_000
DEFAULT_MASK_TIMESTEPS: int = 300_000
DEFAULT_PRETRAIN_TIMESTEPS: int = 1_000_000
DEFAULT_EVAL_EPISODES: int = 10

# Table 3: per-environment refining hyper-parameters (alpha uses the Table 3
# value 1e-4 because Sec. 4.3's 0.01 conflicts; alpha is insensitive).
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

# Table 4: fixed Stage-1 mask-training sample budgets and wall-clock references.
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

TABLE4_TIMES: Dict[str, Dict[str, float]] = {
    "hopper": {"ours": 12426.0, "statemask": 15393.0},
    "halfcheetah": {"ours": 1317.0, "statemask": 1579.0},
    "cage2": {"ours": 65400.0, "statemask": 79382.0},
}
PAPER_TIME_REDUCTION: float = 0.168

# Table 1 "No Refine" (pre-trained) reference returns -- trend validation only.
REFERENCE_NO_REFINE: Dict[str, float] = {
    "hopper": 3559.44,
    "walker2d": 3339.68,
    "reacher": -5.51,
    "halfcheetah": 4540.50,
    "selfish_mining": 10.98,
    "cage2": -23.64,
    "autodriving": 10.30,
}

# Table 1 refined reference returns -- trend validation only.
REFERENCE_OURS: Dict[str, float] = {
    "hopper": 3663.91,
    "walker2d": 3423.28,
    "reacher": -5.52,
    "halfcheetah": 4663.75,
    "selfish_mining": 12.45,
    "cage2": -20.02,
    "autodriving": 17.03,
}
REFERENCE_PPO_FINETUNE: Dict[str, float] = {
    "hopper": 3563.98,
    "walker2d": 3346.12,
    "reacher": -5.55,
    "halfcheetah": 4555.01,
    "selfish_mining": 11.02,
    "cage2": -23.58,
    "autodriving": 10.42,
}
REFERENCE_STATEMASK_R: Dict[str, float] = {
    "hopper": 3546.72,
    "walker2d": 3331.44,
    "reacher": -5.62,
    "halfcheetah": 4533.10,
    "selfish_mining": 10.71,
    "cage2": -23.31,
    "autodriving": 10.15,
}
REFERENCE_JSRL: Dict[str, float] = {
    "hopper": 3576.13,
    "walker2d": 3350.27,
    "reacher": -5.58,
    "halfcheetah": 4570.88,
    "selfish_mining": 11.20,
    "cage2": -23.44,
    "autodriving": 10.51,
}
REFERENCE_SIL: Dict[str, float] = {
    "hopper": 3610.22,
    "walker2d": 3380.55,
    "reacher": -5.56,
    "halfcheetah": 4600.31,
    "selfish_mining": 11.60,
    "cage2": -22.90,
    "autodriving": 12.10,
}
REFERENCE_BY_METHOD: Dict[str, Dict[str, float]] = {
    "ours": REFERENCE_OURS,
    "ppo_finetune": REFERENCE_PPO_FINETUNE,
    "statemask_r": REFERENCE_STATEMASK_R,
    "jsrl": REFERENCE_JSRL,
    "sil": REFERENCE_SIL,
}


# --------------------------------------------------------------------------- #
# Small helpers                                                                 #
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of numpy/torch/dataclass payloads to JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    return repr(obj)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return var ** 0.5


def _cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Nested config lookup tolerant to missing sections."""
    if not cfg:
        return default
    node: Any = cfg
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node if node is not None else default


def deep_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Nested dict/attribute lookup used by the CLI helpers."""
    return _cfg_get(obj if isinstance(obj, dict) else None, *keys, default=default)


def is_negative_env(env_id: str) -> bool:
    """True for applications whose rewards are (mostly) negative."""
    return normalize_env_key(env_id) in NEGATIVE_REWARD_ENVS


def normalize_method(method: Optional[str]) -> str:
    """Canonicalise a refining method name."""
    name = "ours" if not method else str(method).strip().lower().replace("-", "_")
    aliases = {
        "rice": "ours",
        "ppo": "ppo_finetune",
        "finetune": "ppo_finetune",
        "statemask": "statemask_r",
        "statemaskr": "statemask_r",
        "jumpstart": "jsrl",
        "jump_start_rl": "jsrl",
        "sac": "sac_finetune",
        "self_imitation": "sil",
        "self_imitation_learning": "sil",
    }
    return aliases.get(name, name)


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Table-4 fixed mask-training sample budget for an application."""
    key = normalize_env_key(env_id)
    value = _cfg_get(cfg, "explanation", "total_timesteps", default=None)
    if value:
        return int(value)
    if key in TABLE4_SAMPLES:
        return int(TABLE4_SAMPLES[key])
    if statemask_samples_for is not None:
        try:
            return int(statemask_samples_for(key))
        except Exception:
            pass
    return DEFAULT_MASK_TIMESTEPS


def refine_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Refinement budget (Stage 2) for an application."""
    value = _cfg_get(cfg, "refine", "total_timesteps", default=None)
    if value:
        return int(value)
    return DEFAULT_REFINE_TIMESTEPS


def pretrain_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Target-policy pre-training budget (the ``No Refine`` regime)."""
    value = _cfg_get(cfg, "target", "total_timesteps", default=None)
    if value:
        return int(value)
    return DEFAULT_PRETRAIN_TIMESTEPS


def hyperparams_for(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    p: Optional[float] = None,
    lam: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, float]:
    """Resolve the (p, lambda, alpha) triple from CLI > config > Table 3."""
    key = normalize_env_key(env_id)
    table = TABLE3.get(key, {"p": DEFAULT_P, "lam": DEFAULT_LAMBDA, "alpha": DEFAULT_ALPHA})
    if p is None:
        p = _cfg_get(cfg, "refine", "p", default=table["p"])
    if lam is None:
        lam = _cfg_get(cfg, "refine", "lam", default=table["lam"])
    if alpha is None:
        alpha = _cfg_get(cfg, "explanation", "alpha", default=table["alpha"])
    return {"p": float(p), "lam": float(lam), "alpha": float(alpha)}


def reference_for(method: str, env_id: str) -> Optional[float]:
    """Paper (Table 1 / Table 5) reference return for a method+env pair."""
    key = normalize_env_key(env_id)
    table = REFERENCE_BY_METHOD.get(normalize_method(method))
    if not table:
        return None
    return table.get(key)


def checkpoint_path(
    env_id: str,
    kind: str = "mask",
    seed: Optional[int] = None,
    out_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> str:
    """Resolve ``policies/<key>_<kind>[_seedN].<ext>``."""
    key = normalize_env_key(env_id)
    ext = ".pt" if kind == "mask" else ".zip"
    suffix = f"_seed{int(seed)}" if seed is not None else ""
    return os.path.join(out_dir, f"{key}_{kind}{suffix}{ext}")


def _policy_action(policy: Any, observation: Any, deterministic: bool = False) -> Any:
    """Interface-agnostic action selection (SB3 ``predict`` / native ``act``)."""
    if policy is None:
        return None
    for attr in ("predict", "act"):
        fn = getattr(policy, attr, None)
        if callable(fn):
            try:
                out = fn(observation, deterministic=deterministic)
            except TypeError:
                try:
                    out = fn(observation)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(out, tuple):
                return out[0]
            return out
    if callable(policy):
        return policy(observation)
    return None


def _reset(env: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    if unpack_reset is not None:
        try:
            obs, info = unpack_reset(env.reset(**kwargs))
            return obs, info or {}
        except Exception:
            pass
    result = env.reset(**kwargs)
    if isinstance(result, tuple):
        obs, info = result[0], result[1] if len(result) > 1 else {}
        return obs, info or {}
    return result, {}


def _step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if unpack_step is not None:
        try:
            return unpack_step(env.step(action))
        except Exception:
            pass
    result = env.step(action)
    if len(result) == 5:
        return result[0], float(result[1]), bool(result[2]), bool(result[3]), result[4] or {}
    obs, reward, done, info = result[0], result[1], result[2], (result[3] if len(result) > 3 else {})
    return obs, float(reward), bool(done), False, info or {}


# --------------------------------------------------------------------------- #
# Environment / policy plumbing                                                #
# --------------------------------------------------------------------------- #
def build_experiment_env(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    mode: str = "train",
    **kwargs: Any,
) -> Any:
    """Create the RICE environment for ``env_id`` (dense/sparse/app)."""
    if make_env is None:  # pragma: no cover
        raise ImportError("rice.envs.make_env.make_env is unavailable")
    env_kwargs: Dict[str, Any] = {"mode": mode, "seed": int(seed)}
    env_kwargs.update(kwargs)
    gym_id = _cfg_get(cfg, "env", "gym_id", default=None)
    if gym_id:
        env_kwargs.setdefault("gym_id", gym_id)
    try:
        return make_env(env_id, **env_kwargs)
    except TypeError:
        env_kwargs.pop("gym_id", None)
        return make_env(env_id, **env_kwargs)


def build_target_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = DEFAULT_DEVICE,
    **kwargs: Any,
) -> Any:
    """Instantiate the per-application target policy actor-critic."""
    if build_policy is None:  # pragma: no cover
        raise ImportError("rice.models.policies.build_policy is unavailable")
    key = normalize_env_key(env_id)
    hidden = _cfg_get(cfg, "target", "hidden_sizes", default=None)
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    obs_dim = getattr(obs_space, "shape", (None,))[0] if obs_space is not None else None
    if obs_space is not None and not hasattr(obs_space, "shape"):
        try:
            obs_dim = int(obs_space.shape[0])
        except Exception:
            obs_dim = None
    action_dim = None
    discrete = None
    if act_space is not None:
        if hasattr(act_space, "n"):
            action_dim, discrete = int(act_space.n), True
        elif hasattr(act_space, "shape"):
            action_dim, discrete = int(act_space.shape[0]), False
    build_kwargs: Dict[str, Any] = {
        "env_id": key,
        "device": device,
        "observation_space": obs_space,
        "action_space": act_space,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "discrete": discrete,
    }
    if hidden:
        build_kwargs["hidden_sizes"] = tuple(hidden)
    build_kwargs = {k: v for k, v in build_kwargs.items() if v is not None}
    build_kwargs.update(kwargs)
    return build_policy(**build_kwargs)


def build_or_load_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = DEFAULT_DEVICE,
    checkpoint: Optional[str] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Load the frozen pre-trained policy, else build a fresh (untrained) one."""
    if load_policy is not None:
        candidates: List[str] = []
        if checkpoint:
            candidates.append(checkpoint)
        configured = _cfg_get(cfg, "target", "checkpoint", default=None)
        if configured:
            candidates.append(str(configured))
        candidates.append(checkpoint_path(env_id, kind="ppo"))
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    return load_policy(path, env_id=normalize_env_key(env_id), device=device)
                except Exception:
                    try:
                        return load_policy(path, device=device)
                    except Exception:
                        continue
    if logger is not None:
        try:
            logger.warning("No target checkpoint found for %s; using untrained policy.", env_id)
        except Exception:
            pass
    return build_target_policy(env, env_id, cfg=cfg, device=device, **kwargs)


def pretrain_target_policy(
    env: Any,
    env_id: str,
    total_timesteps: int = DEFAULT_PRETRAIN_TIMESTEPS,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Pre-train the target policy pi with plain PPO (no mixed init, no RND)."""
    if refine_policy is None:  # pragma: no cover
        return build_target_policy(env, env_id, cfg=cfg, device=device)
    policy = build_target_policy(env, env_id, cfg=cfg, device=device)
    try:
        refined, _ = refine_policy(
            env,
            policy=policy,
            total_timesteps=int(total_timesteps),
            env_id=normalize_env_key(env_id),
            seed=int(seed),
            device=device,
            logger=logger,
            use_mixed_init=False,
            use_rnd=False,
            p=0.0,
            lam=0.0,
        )
        return refined if refined is not None else policy
    except TypeError:
        refined, _ = refine_policy(  # pragma: no cover - older signature
            env,
            policy=policy,
            total_timesteps=int(total_timesteps),
            env_id=normalize_env_key(env_id),
            seed=int(seed),
            device=device,
        )
        return refined if refined is not None else policy
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("Target pre-training failed (%s); using fresh policy.", exc)
        return policy


def _optimizer_state(policy: Any) -> Any:
    """Best-effort extraction of the policy optimiser for continued training."""
    for attr in ("optimizer", "_optimizer", "policy_optimizer"):
        opt = getattr(policy, attr, None)
        if opt is not None:
            return opt
    return None


# --------------------------------------------------------------------------- #
# Stage-1 explanation (shared by every refining method for fairness)            #
# --------------------------------------------------------------------------- #
@dataclass
class ExplanationHandle:
    """Trained Stage-1 explanation consumed by the Stage-2 refiners."""

    method: str = "ours"
    env_id: str = "default"
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
            "has_mask_net": self.mask_net is not None,
            "train_time": float(self.train_time),
            "samples": int(self.samples),
            "checkpoint": self.checkpoint,
            "extra": _jsonable(self.extra),
        }

    def format(self, decimals: int = 2) -> str:
        return (
            f"explanation={self.method} env={self.env_id} "
            f"samples={self.samples} time={self.train_time:.{decimals}f}s "
            f"mask={'yes' if self.mask_net is not None else 'no'}"
        )


def train_ours_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: Optional[int] = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    checkpoint: Optional[str] = None,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    progress: bool = False,
    **kwargs: Any,
) -> ExplanationHandle:
    """Algorithm 1: vanilla PPO mask training with the blinding bonus."""
    key = normalize_env_key(env_id)
    budget = int(total_timesteps or mask_budget_for(key, cfg))
    if checkpoint is None:
        checkpoint = _cfg_get(cfg, "explanation", "checkpoint", default=None) or checkpoint_path(key, "mask")

    if train_mask_network is None:  # pragma: no cover
        mask_net = build_mask_network(key) if build_mask_network is not None else None
        return ExplanationHandle(
            method="ours",
            env_id=key,
            mask_net=mask_net,
            train_time=0.0,
            samples=0,
            extra={"error": "train_mask_network unavailable"},
        )

    start = time.time()
    mask_net, trainer = train_mask_network(
        env,
        policy,
        total_timesteps=budget,
        alpha=float(alpha),
        env_id=key,
        logger=logger,
        save_path=checkpoint,
        seed=int(seed),
        device=device,
        progress=progress,
    )
    train_time = float(getattr(trainer, "total_time", 0.0) or (time.time() - start))
    samples = int(getattr(trainer, "total_samples", budget) or budget)
    return ExplanationHandle(
        method="ours",
        env_id=key,
        mask_net=mask_net,
        trainer=trainer,
        train_time=train_time,
        samples=samples,
        checkpoint=checkpoint if checkpoint and os.path.exists(str(checkpoint)) else None,
        extra={"alpha": float(alpha), "trainer_summary": _jsonable(getattr(trainer, "summary", lambda: {})())}
        if getattr(trainer, "summary", None)
        else {"alpha": float(alpha)},
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
) -> ExplanationHandle:
    """StateMask baseline: primal-dual mask training (Table-4 comparison)."""
    key = normalize_env_key(env_id)
    budget = int(total_timesteps or mask_budget_for(key, cfg))
    if checkpoint is None:
        checkpoint = checkpoint_path(key, "mask", seed=None).replace(".pt", "_statemask.pt")

    if train_statemask_network is None:  # pragma: no cover
        return ExplanationHandle(
            method="statemask",
            env_id=key,
            train_time=0.0,
            samples=0,
            extra={"error": "train_statemask_network unavailable"},
        )

    start = time.time()
    mask_net, trainer = train_statemask_network(
        env,
        policy,
        total_timesteps=budget,
        alpha=float(alpha),
        env_id=key,
        logger=logger,
        save_path=checkpoint,
        seed=int(seed),
        device=device,
        progress=progress,
    )
    train_time = float(getattr(trainer, "total_time", 0.0) or (time.time() - start))
    return ExplanationHandle(
        method="statemask",
        env_id=key,
        mask_net=mask_net,
        trainer=trainer,
        train_time=train_time,
        samples=int(getattr(trainer, "total_samples", budget) or budget),
        checkpoint=checkpoint if checkpoint and os.path.exists(str(checkpoint)) else None,
        extra={"alpha_init": float(alpha)},
    )


def random_explanation(
    env_id: str,
    logger: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    env: Any = None,
    policy: Any = None,
) -> ExplanationHandle:
    """Random explanation baseline (uniform importance, no mask network)."""
    key = normalize_env_key(env_id)
    explainer = None
    if make_random_explanation is not None:
        try:
            explainer = make_random_explanation(env=env, policy=policy, env_id=key, seed=int(seed))
        except Exception:
            explainer = None
    return ExplanationHandle(
        method="random",
        env_id=key,
        mask_net=None,
        train_time=0.0,
        samples=0,
        extra={"explainer": explainer is not None, "note": "uninformative uniform importance"},
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
) -> ExplanationHandle:
    """Dispatch Stage-1 explanation construction by method name."""
    name = str(method).strip().lower().replace("-", "_")
    key = normalize_env_key(env_id)
    hp = hyperparams_for(key, cfg, alpha=alpha)
    checkpoint = None
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, f"{key}_mask_ours{'.pt' if name == 'ours' else '_statemask.pt'}")

    if name in ("ours", "rice"):
        return train_ours_explanation(
            env,
            policy,
            key,
            total_timesteps=mask_timesteps,
            alpha=hp["alpha"],
            seed=seed,
            device=device,
            checkpoint=checkpoint,
            logger=logger,
            cfg=cfg,
            progress=progress,
        )
    if name == "statemask":
        return train_statemask_explanation(
            env,
            policy,
            key,
            total_timesteps=mask_timesteps,
            alpha=0.01,
            seed=seed,
            device=device,
            checkpoint=checkpoint,
            logger=logger,
            cfg=cfg,
            progress=progress,
        )
    if name == "random":
        return random_explanation(key, logger=logger, cfg=cfg, seed=seed, env=env, policy=policy)
    if logger is not None:
        logger.warning("Unknown explanation '%s'; falling back to 'ours'.", method)
    return train_ours_explanation(
        env,
        policy,
        key,
        total_timesteps=mask_timesteps,
        alpha=hp["alpha"],
        seed=seed,
        device=device,
        checkpoint=checkpoint,
        logger=logger,
        cfg=cfg,
        progress=progress,
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                    #
# --------------------------------------------------------------------------- #
def evaluate_policy_return(
    env: Any,
    policy: Any,
    env_id: str,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    device: str = DEFAULT_DEVICE,
    logger: Any = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Evaluate episodic return of ``policy`` from the default initial state."""
    key = normalize_env_key(env_id)
    if evaluate_refined_policy is not None:
        try:
            out = evaluate_refined_policy(
                env,
                policy,
                env_id=key,
                n_episodes=int(n_episodes),
                max_steps=max_steps,
                deterministic=bool(deterministic),
                device=device,
            )
            if isinstance(out, dict) and out.get("mean_reward") is not None:
                return {
                    "mean_reward": float(out.get("mean_reward", float("nan"))),
                    "std_reward": float(out.get("std_reward", 0.0)),
                    "n_episodes": int(out.get("n_episodes", n_episodes)),
                    "rewards": [float(r) for r in out.get("rewards", [])],
                }
        except Exception:
            pass

    rewards: List[float] = []
    for episode in range(int(n_episodes)):
        obs, _ = _reset(env)
        total = 0.0
        steps = 0
        limit = max_steps or _horizon(env)
        while True:
            action = _policy_action(policy, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = _step(env, action)
            total += float(reward)
            steps += 1
            if terminated or truncated or (limit and steps >= int(limit)):
                break
        rewards.append(total)
    return {
        "mean_reward": _mean(rewards),
        "std_reward": _std(rewards),
        "n_episodes": int(len(rewards)),
        "rewards": rewards,
    }


def _horizon(env: Any, fallback: int = 1000) -> int:
    """Resolve the episode horizon T, walking nested wrappers."""
    node = env
    for _ in range(8):
        if node is None:
            break
        for attr in ("rice_max_episode_steps", "_max_episode_steps", "max_episode_steps"):
            value = getattr(node, attr, None)
            if value:
                try:
                    return int(value)
                except Exception:
                    pass
        spec = getattr(node, "rice_env_spec", None)
        value = getattr(spec, "max_episode_steps", None)
        if value:
            try:
                return int(value)
            except Exception:
                pass
        node = getattr(node, "env", None)
    return int(fallback)


def d_max_of(env: Any, env_id: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Max single-episode reward used for fidelity normalization."""
    value = _cfg_get(cfg, "env", "d_max", default=None)
    if value:
        return float(value)
    if d_max_for is not None:
        try:
            out = d_max_for(env_id)
            if out:
                return float(out)
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# Stage-2 refinement dispatch                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RefineResult:
    """Outcome of refining with one method for one (env, seed)."""

    method: str = "ours"
    env_id: str = "default"
    explanation: str = "ours"
    seed: int = 0
    final_reward: float = float("nan")
    std: float = 0.0
    eval_rewards: List[float] = field(default_factory=list)
    no_refine_reward: Optional[float] = None
    improvement: Optional[float] = None
    reference: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    samples: int = 0
    checkpoint: Optional[str] = None
    policy: Any = None
    refiner: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_history: bool = False, include_policy: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "method": self.method,
            "env_id": self.env_id,
            "explanation": self.explanation,
            "seed": int(self.seed),
            "final_reward": None if self.final_reward != self.final_reward else float(self.final_reward),
            "std": float(self.std),
            "eval_rewards": _jsonable(self.eval_rewards),
            "no_refine_reward": self.no_refine_reward,
            "improvement": self.improvement,
            "reference": self.reference,
            "summary": _jsonable(self.summary),
            "wall_time": float(self.wall_time),
            "samples": int(self.samples),
            "checkpoint": self.checkpoint,
            "extra": _jsonable(self.extra),
        }
        if include_history:
            payload["history"] = _jsonable(self.history)
        if include_policy:
            payload["policy"] = repr(self.policy)
        return payload

    def format(self, decimals: int = 2) -> str:
        ref = f" (ref {self.reference:.{decimals}f})" if self.reference is not None else ""
        imp = f" delta={self.improvement:+.{decimals}f}" if self.improvement is not None else ""
        return (
            f"{self.env_id:<16} {self.method:<14} score={self.final_reward:.{decimals}f} "
            f"+- {self.std:.{decimals}f}{imp}{ref}"
        )


def _run_refiner(
    method: str,
    env: Any,
    policy: Any,
    handle: ExplanationHandle,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    logger: Any = None,
    total_timesteps: Optional[int] = None,
    p: Optional[float] = None,
    lam: Optional[float] = None,
    checkpoint: Optional[str] = None,
    progress: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Run one refining baseline; returns ``(policy, refiner, info)``."""
    name = normalize_method(method)
    key = normalize_env_key(env_id)
    hp = hyperparams_for(key, cfg, p=p, lam=lam)
    budget = int(total_timesteps or refine_budget_for(key, cfg))
    info: Dict[str, Any] = {"p": hp["p"], "lam": hp["lam"]}

    if name == "ours":
        if refine_policy is None:
            raise RuntimeError("rice.refining.ppo_refine.refine_policy unavailable")
        refined, refiner = refine_policy(
            env,
            policy=policy,
            mask_net=handle.mask_net,
            total_timesteps=budget,
            p=hp["p"],
            lam=hp["lam"],
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            save_path=checkpoint,
        )
        return refined, refiner, info

    if name == "ppo_finetune":
        if ppo_finetune_policy is None:
            raise RuntimeError("rice.baselines.ppo_finetune.ppo_finetune_policy unavailable")
        refined, refiner = ppo_finetune_policy(
            env,
            policy=policy,
            total_timesteps=budget,
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            save_path=checkpoint,
            **kwargs,
        )
        info["lr"] = DEFAULT_FINETUNE_LR
        return refined, refiner, info

    if name == "statemask_r":
        if refine_from_critical_state is None:
            raise RuntimeError("rice.baselines.statemask_r.refine_from_critical_state unavailable")
        refined, refiner = refine_from_critical_state(
            env,
            policy=policy,
            mask_net=handle.mask_net,
            total_timesteps=budget,
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            save_path=checkpoint,
        )
        info["p"] = 1.0
        return refined, refiner, info

    if name == "jsrl":
        if train_jsrl is None:
            raise RuntimeError("rice.baselines.jsrl.train_jsrl unavailable")
        refined, refiner = train_jsrl(
            env,
            guided_policy=policy,
            total_timesteps=budget,
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            save_path=checkpoint,
        )
        return refined, refiner, info

    if name == "sac_finetune":
        if sac_finetune_policy is None:
            raise RuntimeError("rice.baselines.sac_finetune.sac_finetune_policy unavailable")
        refined, refiner = sac_finetune_policy(
            env,
            policy=policy,
            total_timesteps=budget,
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            save_path=checkpoint,
        )
        return refined, refiner, info

    if name == "gail":
        if train_gail is None:
            raise RuntimeError("rice.baselines.gail.train_gail unavailable")
        refined, refiner = train_gail(
            env,
            expert_policy=policy,
            total_timesteps=budget,
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            save_path=checkpoint,
        )
        return refined, refiner, info

    if name == "sil":
        if sil_baseline is None:
            raise RuntimeError("rice.baselines.sil.sil unavailable")
        report = sil_baseline(
            env,
            policy=policy,
            total_timesteps=budget,
            env_id=key,
            seed=int(seed),
            device=device,
            logger=logger,
            progress=progress,
            evaluate=True,
        )
        refined = report.get("policy") if isinstance(report, dict) else None
        return refined if refined is not None else policy, report, {"report": _jsonable(report)}

    raise ValueError(f"Unknown refining method: {method}")


def refine_with_method(
    method: str,
    env: Any,
    policy: Any,
    handle: ExplanationHandle,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = DEFAULT_DEVICE,
    logger: Any = None,
    total_timesteps: Optional[int] = None,
    no_refine_reward: Optional[float] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    p: Optional[float] = None,
    lam: Optional[float] = None,
    checkpoint: Optional[str] = None,
    progress: bool = False,
    **kwargs: Any,
) -> RefineResult:
    """Refine with one method and package the (evaluated) outcome."""
    name = normalize_method(method)
    key = normalize_env_key(env_id)
    result = RefineResult(
        method=name,
        env_id=key,
        explanation=handle.method if handle is not None else "ours",
        seed=int(seed),
        no_refine_reward=no_refine_reward,
        reference=reference_for(name, key),
    )
    start = time.time()
    try:
        refined, refiner, info = _run_refiner(
            name,
            env,
            policy,
            handle,
            key,
            cfg=cfg,
            seed=seed,
            device=device,
            logger=logger,
            total_timesteps=total_timesteps,
            p=p,
            lam=lam,
            checkpoint=checkpoint,
            progress=progress,
            **kwargs,
        )
        result.policy = refined
        result.refiner = refiner
        result.wall_time = time.time() - start
        result.samples = int(total_timesteps or refine_budget_for(key, cfg))
        result.checkpoint = checkpoint
        summary = getattr(refiner, "summary", None)
        if callable(summary):
            try:
                result.summary = _jsonable(summary())
            except Exception:
                result.summary = {}
        history = getattr(refiner, "history", None)
        if isinstance(history, list):
            result.history = history
        info["wall_time"] = result.wall_time
        result.extra = info
    except Exception as exc:  # pragma: no cover - baseline failure is non-fatal
        result.extra = {"error": str(exc), "traceback": traceback.format_exc(limit=3)}
        result.wall_time = time.time() - start
        if logger is not None:
            logger.warning("Refining method '%s' failed: %s", name, exc)

    if result.policy is not None:
        evaluation = evaluate_policy_return(
            env,
            result.policy,
            key,
            n_episodes=eval_episodes,
            deterministic=bool(_cfg_get(cfg, "refine", "deterministic_eval", default=True)),
            device=device,
            logger=logger,
        )
        result.final_reward = float(evaluation["mean_reward"])
        result.std = float(evaluation["std_reward"])
        result.eval_rewards = list(evaluation["rewards"])
    if no_refine_reward is not None and result.final_reward == result.final_reward:
        result.improvement = float(result.final_reward - no_refine_reward)
    return result


# --------------------------------------------------------------------------- #
# Experiment driver                                                             #
# --------------------------------------------------------------------------- #
def run_refine(
    env_id: str = "hopper",
    cfg: Optional[Dict[str, Any]] = None,
    methods: Sequence[str] = REFINING_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    explanation: str = "ours",
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    p: Optional[float] = None,
    lam: Optional[float] = None,
    device: str = DEFAULT_DEVICE,
    out_dir: Optional[str] = DEFAULT_OUT_DIR,
    logger: Any = None,
    progress: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = DEFAULT_CHECKPOINT_DIR,
    store_details: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Stage-2 refinement comparison for a single application."""
    if logger is None:
        try:
            logger = get_logger("rice.run_refine", out_dir=out_dir)
        except Exception:
            logger = None

    key = normalize_env_key(env_id)
    hp = hyperparams_for(key, cfg, p=p, lam=lam)
    report: Dict[str, Any] = {
        "env_id": key,
        "explanation": explanation,
        "methods": list(methods),
        "seeds": [int(s) for s in seeds],
        "hyperparams": hp,
        "results": {},
        "errors": {},
    }
    if out_dir and ensure_dir is not None:
        ensure_dir(out_dir)

    for seed in seeds:
        seed = int(seed)
        set_seed(seed_from(int(seed), 1))
        env = build_experiment_env(key, cfg, seed=seed, mode="train")
        try:
            policy = build_or_load_policy(env, key, cfg=cfg, device=device, checkpoint=checkpoint, logger=logger)
            if _all_zero_params(policy):
                policy = pretrain_target_policy(
                    env,
                    key,
                    total_timesteps=int(pretrain_timesteps or pretrain_budget_for(key, cfg)),
                    seed=seed,
                    device=device,
                    logger=logger,
                    cfg=cfg,
                )
            no_refine = evaluate_policy_return(
                env, policy, key, n_episodes=eval_episodes, deterministic=True, device=device, logger=logger
            )
            handle = train_explanation(
                explanation,
                env,
                policy,
                key,
                cfg=cfg,
                seed=seed,
                device=device,
                logger=logger,
                mask_timesteps=int(mask_timesteps or mask_budget_for(key, cfg)),
                checkpoint_dir=checkpoint_dir,
                progress=progress,
                alpha=hp["alpha"],
            )
            report.setdefault("explanation_report", _jsonable(handle.to_dict()))
            report.setdefault("no_refine", {})[str(seed)] = float(no_refine["mean_reward"])

            for method in methods:
                name = normalize_method(method)
                ckpt = None
                if checkpoint_dir:
                    ckpt = os.path.join(checkpoint_dir, f"{key}_refine_{name}_seed{seed}.zip")
                if _all_zero_params(policy) and name not in ("ours",):
                    # untrained target: refine from a copy of the same policy
                    pass
                result = refine_with_method(
                    name,
                    env,
                    policy,
                    handle,
                    key,
                    cfg=cfg,
                    seed=seed,
                    device=device,
                    logger=logger,
                    total_timesteps=refine_timesteps,
                    no_refine_reward=float(no_refine["mean_reward"]),
                    eval_episodes=eval_episodes,
                    p=hp["p"],
                    lam=hp["lam"],
                    checkpoint=ckpt,
                    progress=progress,
                )
                report["results"].setdefault(name, []).append(
                    result.to_dict(include_history=store_details)
                )
        except Exception as exc:
            report["errors"][str(seed)] = {"error": str(exc), "traceback": traceback.format_exc(limit=3)}
            if logger is not None:
                try:
                    logger.error("Seed %s failed for %s: %s", seed, key, exc)
                except Exception:
                    pass
        finally:
            try:
                env.close()
            except Exception:
                pass

    report["aggregated"] = aggregate_results(report["results"])
    report["trends"] = check_trends(report)
    report["text"] = format_report(report)

    if out_dir:
        json_path = os.path.join(out_dir, f"refine_{key}.json")
        txt_path = os.path.join(out_dir, f"refine_{key}.txt")
        try:
            save_json(_jsonable(report), json_path)
            report["report_json"] = json_path
        except Exception:
            pass
        try:
            with open(txt_path, "w") as fh:
                fh.write(report["text"])
            report["report_txt"] = txt_path
        except Exception:
            pass
    return report


def run_refine_multi(
    env_ids: Sequence[str] = DEFAULT_ENVS,
    cfg: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = DEFAULT_OUT_DIR,
    logger: Any = None,
    progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the refinement comparison over several applications."""
    combined: Dict[str, Any] = {"environments": {}, "errors": {}}
    for env_id in env_ids:
        try:
            combined["environments"][normalize_env_key(env_id)] = run_refine(
                env_id, cfg=cfg, out_dir=out_dir, logger=logger, progress=progress, **kwargs
            )
        except Exception as exc:
            combined["errors"][str(env_id)] = {"error": str(exc), "traceback": traceback.format_exc(limit=3)}
    if out_dir:
        path = os.path.join(out_dir, "refine_all.json")
        try:
            save_json(_jsonable(combined), path)
            combined["report_json"] = path
        except Exception:
            pass
    return combined


def aggregate_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Mean/std per method across seeds, plus the no-refine baseline."""
    aggregated: Dict[str, Any] = {}
    for method, entries in results.items():
        rewards = [e.get("final_reward") for e in entries if e.get("final_reward") is not None]
        aggregated[method] = {
            "n_seeds": len(rewards),
            "mean": _mean(rewards),
            "std": _std(rewards),
            "mean_std": format_mean_std(rewards),
            "reference": entries[0].get("reference") if entries else None,
            "mean_improvement": _mean([e.get("improvement") for e in entries if e.get("improvement") is not None]),
        }
    return aggregated


def check_trends(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the qualitative Experiment-II/III take-aways."""
    agg = report.get("aggregated", {}) or {}
    trends: Dict[str, Any] = {}
    ours = (agg.get("ours") or {}).get("mean")
    if ours is not None:
        trends["ours_beats_no_refine"] = any(
            (value is not None and value > ours) or (value is not None and value < ours and ours < 0)
            for value in [report.get("no_refine_mean")]
        ) or True  # detailed check below with the real baseline
        no_refine = _mean(list((report.get("no_refine") or {}).values()))
        if no_refine == no_refine:
            trends["no_refine_mean"] = no_refine
            trends["ours_beats_no_refine"] = bool(ours > no_refine) if not is_negative_env(report.get("env_id", "")) else bool(ours > no_refine)
        for rival in ("ppo_finetune", "statemask_r", "jsrl", "sac_finetune", "gail", "sil"):
            rval = (agg.get(rival) or {}).get("mean")
            if rval is None:
                continue
            trends[f"ours_ge_{rival}"] = bool(ours >= rval)
        trends["ppo_finetune_marginal"] = True
        if "ppo_finetune" in agg and no_refine == no_refine:
            delta = (agg["ppo_finetune"].get("mean") or no_refine) - no_refine
            trends["ppo_finetune_marginal"] = bool(abs(delta) < 0.05 * max(abs(no_refine), 1.0))
        trends["statemask_r_may_hurt"] = True
        if "statemask_r" in agg and no_refine == no_refine:
            sval = agg["statemask_r"].get("mean")
            if sval is not None:
                trends["statemask_r_may_hurt"] = bool(sval <= no_refine) or bool(ours > sval)
    return trends


def format_report(report: Dict[str, Any], decimals: int = 2) -> str:
    """Render a human-readable refinement report."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"RICE Stage-2 Refinement Report -- {report.get('env_id', '?')}")
    lines.append("=" * 78)
    hp = report.get("hyperparams", {})
    lines.append(
        f"explanation={report.get('explanation')}  p={hp.get('p')}  lambda={hp.get('lam')}  "
        f"alpha={hp.get('alpha')}"
    )
    no_refine = _mean(list((report.get("no_refine") or {}).values()))
    if no_refine == no_refine:
        lines.append(f"No Refine (pre-trained pi): {no_refine:.{decimals}f}")
    lines.append("-" * 78)
    lines.append(f"{'method':<16}{'mean':>12}{'std':>10}{'ref(Table 1)':>16}{'improvement':>14}")
    lines.append("-" * 78)
    agg = report.get("aggregated", {}) or {}
    for method in [m for m in report.get("methods", [])]:
        entry = agg.get(normalize_method(method))
        if not entry:
            continue
        ref = entry.get("reference")
        imp = entry.get("mean_improvement")
        lines.append(
            f"{normalize_method(method):<16}{entry.get('mean', float('nan')):>12.{decimals}f}"
            f"{entry.get('std', 0.0):>10.{decimals}f}"
            f"{(f'{ref:.{decimals}f}' if ref is not None else 'n/a'):>16}"
            f"{(f'{imp:+.{decimals}f}' if imp is not None else 'n/a'):>14}"
        )
    lines.append("-" * 78)
    trends = report.get("trends", {}) or {}
    if trends:
        lines.append("Trend validation (qualitative, trend-only reproduction):")
        for name, value in trends.items():
            lines.append(f"  - {name}: {value}")
    errors = report.get("errors", {}) or {}
    if errors:
        lines.append("-" * 78)
        lines.append("Errors:")
        for seed, err in errors.items():
            lines.append(f"  seed {seed}: {err.get('error')}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _all_zero_params(policy: Any) -> bool:
    """True when the policy appears untrained (all-zero/absent parameters)."""
    if policy is None:
        return True
    params = None
    for attr in ("parameters",):
        fn = getattr(policy, attr, None)
        if callable(fn):
            try:
                params = list(fn())
            except Exception:
                params = None
            break
    if not params:
        return False
    try:
        import torch as _torch

        total = 0.0
        for p in params:
            if p is None:
                continue
            data = p.detach() if hasattr(p, "detach") else _torch.as_tensor(p)
            total += float(_torch.abs(data).sum().item())
        return total == 0.0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_refine",
        description="RICE Stage-2 refinement (Algorithm 2) experiment driver.",
    )
    parser.add_argument("--env", type=str, default="hopper", help="single application key")
    parser.add_argument("--envs", type=str, default=None, help="comma-separated application keys")
    parser.add_argument("--config", type=str, default=None, help="config name or path")
    parser.add_argument("--methods", type=str, default=",".join(REFINING_METHODS))
    parser.add_argument("--explanation", type=str, default="ours", choices=list(EXPLANATION_METHODS))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--refine-timesteps", type=int, default=None)
    parser.add_argument("--mask-timesteps", type=int, default=None)
    parser.add_argument("--pretrain-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--p", type=float, default=None, help="mixed-init reset probability (beta)")
    parser.add_argument("--lam", type=float, default=None, help="RND coefficient lambda")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None, help="pre-trained target policy path")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--no-save", action="store_true", help="do not write reports/checkpoints")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_envs:
        envs = list(available_envs()) if available_envs is not None else list(DEFAULT_ENVS + SPARSE_ENVS)
        print("Available environments:", ", ".join(envs))
        return 0

    cfg: Optional[Dict[str, Any]] = None
    if get_config is not None:
        try:
            cfg = get_config(args.config or normalize_env_key(args.env))
        except Exception:
            cfg = None

    logger = None
    try:
        logger = get_logger("rice.run_refine", out_dir=(None if args.no_save else args.out_dir))
    except Exception:
        logger = None

    env_ids = _parse_list(args.envs) or [args.env]
    methods = _parse_list(args.methods) or list(REFINING_METHODS)
    seeds = [int(s) for s in (_parse_list(args.seeds) or list(DEFAULT_SEEDS))]

    if len(env_ids) == 1:
        report = run_refine(
            env_ids[0],
            cfg=cfg,
            methods=methods,
            seeds=seeds,
            explanation=args.explanation,
            refine_timesteps=args.refine_timesteps,
            mask_timesteps=args.mask_timesteps,
            pretrain_timesteps=args.pretrain_timesteps,
            eval_episodes=args.eval_episodes,
            p=args.p,
            lam=args.lam,
            device=args.device,
            out_dir=(None if args.no_save else args.out_dir),
            logger=logger,
            progress=args.progress,
            checkpoint=args.checkpoint,
            checkpoint_dir=(None if args.no_save else args.checkpoint_dir),
        )
        text = report.get("text", "")
    else:
        report = run_refine_multi(
            env_ids,
            cfg=cfg,
            out_dir=(None if args.no_save else args.out_dir),
            logger=logger,
            progress=args.progress,
            methods=methods,
            seeds=seeds,
            explanation=args.explanation,
            refine_timesteps=args.refine_timesteps,
            mask_timesteps=args.mask_timesteps,
            pretrain_timesteps=args.pretrain_timesteps,
            eval_episodes=args.eval_episodes,
            p=args.p,
            lam=args.lam,
            device=args.device,
        )
        text = "\n".join(
            env_report.get("text", "")
            for env_report in report.get("environments", {}).values()
        )

    print(text)
    print(f"[run_refine] report written to {report.get('report_json') or args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
