"""RICE Stage 2 -- Algorithm 2: refining the pre-trained (bottlenecked) policy.

This module implements the second stage of RICE ("Refining the DRL Agent",
Algorithm 2 of Cheng et al., ICML 2024).  Conceptually, each iteration

1. builds the mixed initial state distribution
   ``mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s)``

   by drawing ``RAND_NUM ~ U(0,1)``; if ``RAND_NUM < p`` the frozen pre-trained
   policy ``pi`` is rolled for ``K`` steps, the mask network identifies the most
   critical state ``s_t`` and the environment is reset to it (``s_0 ~
   d_rho^pihat``); otherwise a regular environment reset is used (``s_0 ~ rho``).
   The mixture weight ``beta`` is realised by the reset probability ``p``.

2. rolls the *trainable* policy ``pi_theta`` for ``T`` steps starting from
   ``s_0``, sampling ``a_t ~ pi(theta)(.|s_t)``, stepping the environment and
   storing ``(s_t, s_{t+1}, a_t, R_t + lambda * R_t^RND)`` in the dataset ``D``,
   where ``R_t^RND = ||f(s_{t+1}) - fhat(s_{t+1})||^2`` is the (normalised)
   Random Network Distillation intrinsic reward.

3. optimises ``pi_theta`` with the standard (clipped) PPO loss on ``D`` and
   updates the RND predictor ``fhat`` with an MSE loss using Adam.

After the loop the refined policy ``pi' <- pi_theta`` is returned.

The target policy is *frozen*: by default the incoming pre-trained policy is
deep-copied, so ``pi`` is never mutated while ``pi_theta`` is trained.
Unspecified PPO hyper-parameters follow the plan/addendum defaults
(gamma=0.99, clip=0.2, lr=3e-4, n_epochs=10, GAE lambda=0.95).
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Optional / defensive intra-package imports
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - torch is a hard runtime dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

try:
    from rice.utils.seeding import get_rng, seed_env, set_seed
except Exception:  # pragma: no cover
    def get_rng(seed: Optional[int] = None):  # type: ignore
        return np.random.RandomState(seed)

    def seed_env(env, seed=None, rank: int = 0):  # type: ignore
        try:
            if seed is not None and hasattr(env, "seed"):
                env.seed(int(seed) + int(rank))
        except Exception:
            pass
        return seed

    def set_seed(seed):  # type: ignore
        try:
            np.random.seed(int(seed))
        except Exception:
            pass
        return seed

try:
    from rice.utils.logging import get_logger
except Exception:  # pragma: no cover
    import logging as _logging

    def get_logger(name="rice", out_dir=None, level=None):  # type: ignore
        logger = _logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(_logging.StreamHandler())
        logger.setLevel(level or _logging.INFO)
        return logger

try:
    from rice.utils.io import ensure_dir
except Exception:  # pragma: no cover
    def ensure_dir(path):  # type: ignore
        if path:
            os.makedirs(path, exist_ok=True)
        return path

try:
    from rice.utils.metrics import compute_returns, discounted_return, mean_std
except Exception:  # pragma: no cover
    def compute_returns(rewards, gamma=0.99, normalize_by_discount=False):  # type: ignore
        out = np.zeros(len(rewards), dtype=np.float64)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            out[t] = running
        return out

    def discounted_return(rewards, gamma=0.99, normalize_by_discount=False):  # type: ignore
        return float(compute_returns(rewards, gamma)[0]) if len(rewards) else 0.0

    def mean_std(values):  # type: ignore
        arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std())

try:
    from rice.refining.rnd import DEFAULT_LAMBDA, RNDModule, build_rnd
except Exception:  # pragma: no cover
    DEFAULT_LAMBDA = 0.01
    RNDModule = None  # type: ignore
    build_rnd = None  # type: ignore

try:
    from rice.refining.mixed_init import DEFAULT_P, MixedInitialStateSampler, make_mixed_init_sampler
except Exception:  # pragma: no cover
    DEFAULT_P = 0.5
    MixedInitialStateSampler = None  # type: ignore
    make_mixed_init_sampler = None  # type: ignore

try:
    from rice.models.policies import build_policy, normalize_env_key, sb3_policy_kwargs
except Exception:  # pragma: no cover
    build_policy = None  # type: ignore
    normalize_env_key = None  # type: ignore
    sb3_policy_kwargs = None  # type: ignore

try:  # optional: only used when no pretrained policy object is supplied
    from rice.explanation.mask_network import flatten_observation
except Exception:  # pragma: no cover
    def flatten_observation(observation):  # type: ignore
        if isinstance(observation, dict):
            parts = []
            for key in sorted(observation.keys()):
                parts.append(np.asarray(observation[key], dtype=np.float32).ravel())
            return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return np.asarray(observation, dtype=np.float32).ravel()


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_HORIZON = 1000
DEFAULT_PPO_LR = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_N_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_VF_COEF = 0.5
DEFAULT_ENT_COEF = 0.0
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_TOTAL_TIMESTEPS = 100_000
_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RefinePPOConfig:
    """Hyper-parameters for Algorithm 2 (PPO refinement + mixed init + RND).

    Only ``p``, ``lambda`` and ``alpha``-like knobs are paper-specified
    (Table 3); everything else defaults to the standard PPO settings because
    the paper does not state them (see the reproduction plan's
    "missing_but_critical defaults").
    """

    # optimisation
    lr: float = DEFAULT_PPO_LR
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_range: float = DEFAULT_CLIP_RANGE
    clip_range_vf: Optional[float] = None
    n_epochs: int = DEFAULT_N_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    vf_coef: float = DEFAULT_VF_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    normalize_advantage: bool = True
    target_kl: Optional[float] = None
    clip_actions: bool = True

    # rollouts
    n_steps: Optional[int] = None          # T: trajectory length per iteration
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    n_iterations: Optional[int] = None
    resample_on_done: bool = True          # re-draw s_0 when an episode ends early

    # mixed initial state distribution (Algorithm 2 roll-in)
    use_mixed_init: bool = True
    p: float = DEFAULT_P                   # reset probability threshold (== beta)
    K: Optional[int] = None                # length-K trajectory for critical state

    # exploration bonus (RND)
    use_rnd: bool = True
    lam: float = DEFAULT_LAMBDA            # lambda in R_t + lambda * R_t^RND
    rnd_update_epochs: int = 1
    rnd_batch_size: int = 256
    rnd_hidden_sizes: Optional[Sequence[int]] = None
    rnd_lr: Optional[float] = None

    # misc
    device: str = "cpu"
    seed: Optional[int] = None
    deterministic_policy: bool = False
    copy_policy: bool = True               # keep the pre-trained pi frozen
    log_interval: int = 1
    eval_interval: Optional[int] = None
    eval_episodes: int = 5
    store_dataset: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides) -> "RefinePPOConfig":
        """Build a config from a (possibly nested) YAML-like dict."""
        params: Dict[str, Any] = {}
        if isinstance(cfg, RefinePPOConfig):
            return cfg
        if isinstance(cfg, dict):
            # accept nested sections produced by the configs/*.yaml files
            for key in ("refine", "refining", "ppo", "refiner", "ppo_refine", "rice_refine"):
                section = cfg.get(key)
                if isinstance(section, dict):
                    params.update(section)
            for key, value in cfg.items():
                if key in ("refine", "refining", "ppo", "refiner", "ppo_refine", "rice_refine"):
                    continue
                if not isinstance(value, dict):
                    params.setdefault(key, value)
                else:
                    # mixed-init / rnd sub-sections
                    if key in ("mixed_init", "mixed_initial_state", "roll_in"):
                        params.setdefault("p", value.get("p", value.get("beta")))
                        if value.get("K") is not None:
                            params.setdefault("K", value["K"])
                    if key in ("rnd", "exploration", "intrinsic"):
                        params.setdefault("lam", value.get("lam", value.get("lambda")))
                        if value.get("learning_rate") is not None:
                            params.setdefault("rnd_lr", value.get("learning_rate"))
                        if value.get("hidden_sizes") is not None:
                            params.setdefault("rnd_hidden_sizes", value.get("hidden_sizes"))
        params.update(overrides)
        params = {k: v for k, v in params.items() if v is not None or k in ("K", "n_steps", "n_iterations", "target_kl", "clip_range_vf", "eval_interval", "seed")}
        # alias handling
        aliases = {
            "lambda": "lam",
            "lambda_": "lam",
            "rnd_lambda": "lam",
            "beta": "p",
            "reset_prob": "p",
            "clip": "clip_range",
            "learning_rate": "lr",
            "num_epochs": "n_epochs",
            "eps": None,
        }
        for src, dst in aliases.items():
            if dst and src in params and dst not in params:
                params[dst] = params.pop(src)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in params.items() if k in known}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, name)
            if isinstance(value, np.generic):
                value = value.item()
            out[name] = value
        return out

    @property
    def beta(self) -> float:
        """The mixture weight ``beta`` of ``mu(s)`` (realised through ``p``)."""
        return float(self.p)


# --------------------------------------------------------------------------- #
# Rollout container (dataset D)
# --------------------------------------------------------------------------- #
@dataclass
class RefineRollout:
    """Dataset ``D`` of one Algorithm-2 iteration plus PPO bookkeeping."""

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray                  # R_t + lambda * R_t^RND
    task_rewards: np.ndarray             # R_t
    rnd_bonuses: np.ndarray              # normalised ||f - fhat||^2
    log_probs: np.ndarray
    values: np.ndarray
    dones: np.ndarray                    # 1.0 for terminated or truncated
    truncated: np.ndarray
    advantages: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    episode_task_returns: List[float] = field(default_factory=list)
    episode_augmented_returns: List[float] = field(default_factory=list)
    start_modes: List[str] = field(default_factory=list)
    p_used: float = DEFAULT_P
    lam_used: float = DEFAULT_LAMBDA
    wall_time: float = 0.0

    def __len__(self) -> int:
        return int(len(self.actions))

    @property
    def critical_fraction(self) -> float:
        if not self.start_modes:
            return 0.0
        n_crit = sum(1 for m in self.start_modes if m == "critical")
        return float(n_crit) / float(len(self.start_modes))

    @property
    def mean_intrinsic_bonus(self) -> float:
        if self.rnd_bonuses.size == 0:
            return 0.0
        return float(np.mean(self.rnd_bonuses))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": len(self),
            "mean_task_reward": float(np.mean(self.task_rewards)) if len(self.task_rewards) else 0.0,
            "mean_rnd_bonus": self.mean_intrinsic_bonus,
            "mean_reward": float(np.mean(self.rewards)) if len(self.rewards) else 0.0,
            "critical_fraction": self.critical_fraction,
            "n_episodes": len(self.episode_task_returns),
            "episode_task_returns": list(self.episode_task_returns),
            "episode_augmented_returns": list(self.episode_augmented_returns),
            "p": float(self.p_used),
            "lambda": float(self.lam_used),
            "wall_time": float(self.wall_time),
        }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _resolve_horizon(env=None, fallback: int = DEFAULT_HORIZON) -> int:
    """Resolve the episode horizon ``T`` (Algorithm 2's inner loop length)."""
    candidates: List[Any] = []
    if env is not None:
        candidates.extend(
            [
                getattr(env, "rice_max_episode_steps", None),
                getattr(getattr(env, "spec", None), "max_episode_steps", None),
                getattr(env, "max_episode_steps", None),
                getattr(env, "_max_episode_steps", None),
                getattr(env, "_max_episode_steps", None),
            ]
        )
        inner = getattr(env, "env", None)
        depth = 0
        while inner is not None and depth < 6:
            candidates.append(getattr(inner, "_max_episode_steps", None))
            candidates.append(getattr(inner, "max_episode_steps", None))
            candidates.append(getattr(getattr(inner, "spec", None), "max_episode_steps", None))
            inner = getattr(inner, "env", None)
            depth += 1
    for value in candidates:
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            continue
    return int(fallback)


def _space_dims(env) -> Tuple[Optional[int], Optional[int], Optional[Tuple[float, float]]]:
    """Best-effort (obs_dim, action_dim, action_bounds) extraction."""
    obs_dim = act_dim = None
    bounds = None
    if env is None:
        return obs_dim, act_dim, bounds
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    try:
        shape = getattr(obs_space, "shape", None)
        if shape is not None:
            obs_dim = int(np.prod(shape))
    except Exception:
        obs_dim = None
    try:
        if hasattr(act_space, "n"):
            act_dim = int(act_space.n)
        else:
            shape = getattr(act_space, "shape", None)
            if shape is not None:
                act_dim = int(np.prod(shape))
    except Exception:
        act_dim = None
    try:
        low = np.asarray(getattr(act_space, "low", []), dtype=np.float64).ravel()
        high = np.asarray(getattr(act_space, "high", []), dtype=np.float64).ravel()
        if low.size and high.size:
            bounds = (float(low.min()), float(high.max()))
    except Exception:
        bounds = None
    return obs_dim, act_dim, bounds


def _is_discrete(policy=None, action_space=None, fallback: Optional[bool] = None) -> bool:
    """Decide whether the action space is discrete."""
    if action_space is not None:
        if hasattr(action_space, "n"):
            return True
        if hasattr(action_space, "shape") and not hasattr(action_space, "low"):
            return False
    if policy is not None:
        if getattr(policy, "discrete", None) is not None:
            return bool(getattr(policy, "discrete"))
        space = getattr(policy, "action_space", None)
        if space is not None and hasattr(space, "n"):
            return True
    return bool(fallback) if fallback is not None else False


def _action_bounds(policy=None, action_space=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    space = action_space if action_space is not None else getattr(policy, "action_space", None)
    low = high = None
    try:
        low = np.asarray(getattr(space, "low", []), dtype=np.float32).ravel()
        high = np.asarray(getattr(space, "high", []), dtype=np.float32).ravel()
        if low.size == 0 or high.size == 0:
            low = high = None
    except Exception:
        low = high = None
    return low, high


def _obs_tensor(observation, device: str = "cpu"):
    """Convert a single observation (or a batch) to a float32 tensor."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for PPO refinement.")
    if isinstance(observation, torch.Tensor):
        tensor = observation.detach().float()
        return tensor if tensor.dim() > 1 else tensor.unsqueeze(0)
    flat = flatten_observation(observation)
    tensor = torch.as_tensor(np.asarray(flat, dtype=np.float32), device=device)
    return tensor.unsqueeze(0)


def _batch_obs_tensor(observations, device: str = "cpu"):
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for PPO refinement.")
    arr = np.asarray(observations, dtype=np.float32)
    return torch.as_tensor(arr, device=device)


def _flatten_sequence(sequence, dtype=np.float32) -> np.ndarray:
    """Flatten a list of observations into a 2D array (dict observations ok)."""
    if sequence is None:
        return np.zeros((0, 1), dtype=dtype)
    out = []
    for item in sequence:
        flat = flatten_observation(item)
        out.append(np.asarray(flat, dtype=dtype).ravel())
    if not out:
        return np.zeros((0, 1), dtype=dtype)
    width = max(item.size for item in out)
    padded = np.zeros((len(out), width), dtype=dtype)
    for i, item in enumerate(out):
        padded[i, : item.size] = item
    return padded


def unpack_step(result) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise 4-tuple (classic) and 5-tuple (terminated/truncated) steps."""
    if not isinstance(result, (tuple, list)):
        return result, 0.0, False, False, {}
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
    if len(result) == 4:
        obs, reward, done, info = result
        info = dict(info or {})
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return obs, float(reward), terminated, truncated, info
    if len(result) == 3:
        obs, reward, done = result
        return obs, float(reward), bool(done), False, {}
    return result[0], float(result[1] if len(result) > 1 else 0.0), False, False, {}


def unpack_reset(result) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(result, (tuple, list)) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[float],
    last_value: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (Schulman et al., 2016)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = float(last_value)
            next_nonterminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


# --------------------------------------------------------------------------- #
# Policy access helpers (native ActorCritic *and* SB3 ActorCriticPolicy)
# --------------------------------------------------------------------------- #
def policy_distribution_and_values(policy, obs_tensor):
    """Return ``(distribution, values)`` for a native or SB3-style policy."""
    dist = None
    values = None
    if hasattr(policy, "get_distribution"):
        dist = policy.get_distribution(obs_tensor)
    elif hasattr(policy, "_get_action_dist_from_latent"):
        try:  # pragma: no cover - SB3 internal
            latent_pi, _ = policy._get_latent(obs_tensor)
            dist = policy._get_action_dist_from_latent(latent_pi)
        except Exception:
            dist = None
    if hasattr(policy, "predict_values"):
        try:
            values = policy.predict_values(obs_tensor)
        except Exception:
            values = None
    if values is None and hasattr(policy, "value_net"):
        try:  # pragma: no cover
            values = policy.value_net(policy.extract_features(obs_tensor))
        except Exception:
            values = None
    return dist, values


def policy_action(policy, observation, deterministic: bool = False) -> np.ndarray:
    """Sample an action from a native/SB3 policy (no grad)."""
    if policy is None:
        raise ValueError("policy_action requires a policy object.")
    if hasattr(policy, "predict") and not hasattr(policy, "get_distribution"):
        action, _ = policy.predict(observation, deterministic=deterministic)
        return np.asarray(action)
    if hasattr(policy, "predict"):
        try:
            action, _ = policy.predict(observation, deterministic=deterministic)
            return np.asarray(action)
        except Exception:
            pass
    if hasattr(policy, "act"):
        action = policy.act(observation, deterministic=deterministic)
        return np.asarray(action)
    if callable(policy):
        return np.asarray(policy(observation))
    raise TypeError("Unsupported policy object for policy_action().")


def eval_policy_action(policy, observation, deterministic: bool = True) -> np.ndarray:
    return policy_action(policy, observation, deterministic=deterministic)


def policy_values(policy, observation) -> float:
    """Value estimate V(s) for a single observation."""
    if policy is None or not _HAS_TORCH:
        return 0.0
    try:
        obs_t = _obs_tensor(observation)
        if hasattr(policy, "predict_values"):
            with torch.no_grad():
                return float(np.asarray(policy.predict_values(obs_t).detach().cpu()).ravel()[0])
    except Exception:
        pass
    return 0.0


def _sample_from_policy(policy, observation, device: str = "cpu", deterministic: bool = False):
    """Sample action + log-prob + value from a native/SB3 policy.

    Returns ``(action, log_prob, value)`` where ``action`` is a numpy array
    (int64 scalar-array for discrete spaces).
    """
    if not _HAS_TORCH or policy is None:
        raise RuntimeError("PyTorch and a policy object are required.")
    obs_t = _obs_tensor(observation, device=device)
    with torch.no_grad():
        dist, values = policy_distribution_and_values(policy, obs_t)
        if dist is None:
            action = policy_action(policy, observation, deterministic=deterministic)
            return np.asarray(action), 0.0, float(np.asarray(values).ravel()[0]) if values is not None else 0.0
        if deterministic:
            if hasattr(dist, "mode"):
                try:
                    action_t = dist.mode
                except Exception:
                    action_t = dist.mean
            else:
                action_t = dist.mean
        else:
            action_t = dist.sample()
        try:
            log_prob = float(dist.log_prob(action_t).detach().cpu().reshape(-1)[0])
        except Exception:
            log_prob = 0.0
        value = float(values.detach().cpu().reshape(-1)[0]) if values is not None else 0.0
    action = action_t.detach().cpu().numpy()
    return action, log_prob, value


def prepare_action_for_env(action, discrete: bool, low=None, high=None, policy=None, action_space=None, clip: bool = True):
    """Convert a policy output into a valid environment action."""
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action)
    if discrete or (arr.ndim <= 1 and arr.dtype.kind in "iu"):
        space = action_space if action_space is not None else getattr(policy, "action_space", None)
        n = getattr(space, "n", None)
        idx = int(np.asarray(arr).ravel()[0])
        if n is not None:
            idx = int(np.clip(idx, 0, int(n) - 1))
        return idx
    arr = arr.astype(np.float32).ravel()
    if clip:
        if low is None or high is None:
            low, high = _action_bounds(policy, action_space)
        if low is not None and high is not None and low.size == arr.size:
            arr = np.clip(arr, low, high)
    return arr.astype(np.float32)


def resolve_obs_act_dims(env, policy=None) -> Tuple[int, Optional[int], bool]:
    """Resolve (obs_dim, action_dim, discrete) for policy/RND construction."""
    obs_dim, act_dim, _ = _space_dims(env)
    if obs_dim is None and policy is not None:
        obs_space = getattr(policy, "observation_space", None)
        try:
            obs_dim = int(np.prod(obs_space.shape))
        except Exception:
            obs_dim = None
    discrete = _is_discrete(policy=policy, action_space=getattr(env, "action_space", None))
    if act_dim is None and not discrete:
        low, high = _action_bounds(policy, getattr(env, "action_space", None))
        act_dim = int(low.size) if low is not None else None
    if obs_dim is None:
        obs_dim = int(getattr(policy, "obs_dim", 0) or 0) or 1
    return int(obs_dim), (int(act_dim) if act_dim is not None else None), bool(discrete)


def ensure_trainable_policy(
    policy=None,
    env=None,
    env_id: str = "default",
    device: str = "cpu",
    copy_policy: bool = True,
    **kwargs,
):
    """Return a trainable policy object.

    If a policy is supplied it is (optionally deep-copied and) returned -- both
    the native ``ActorCritic`` and SB3's ``ActorCriticPolicy`` are nn.Modules
    exposing ``get_distribution`` / ``predict_values``, so the PPO update below
    works uniformly.  When no policy is supplied a fresh native policy is built
    through :func:`rice.models.policies.build_policy`.
    """
    if policy is None:
        if build_policy is None:
            raise ValueError("No policy supplied and rice.models.policies is unavailable.")
        obs_dim, act_dim, discrete = resolve_obs_act_dims(env)
        policy = build_policy(
            env_id=env_id,
            obs_dim=obs_dim,
            action_dim=act_dim,
            action_space=getattr(env, "action_space", None),
            observation_space=getattr(env, "observation_space", None),
            kind="policy",
            backend=kwargs.get("backend", "native"),
            device=device,
            **{k: v for k, v in kwargs.items() if k in ("hidden_sizes", "activation")},
        )
    else:
        if copy_policy and _HAS_TORCH and isinstance(policy, nn.Module):
            try:
                policy = copy.deepcopy(policy)
            except Exception:
                policy = policy
    if _HAS_TORCH and isinstance(policy, nn.Module):
        try:
            policy.to(device)
        except Exception:
            pass
    return policy


def policy_parameters(policy) -> List[Any]:
    if policy is None:
        return []
    if hasattr(policy, "parameters"):
        params = [p for p in policy.parameters() if getattr(p, "requires_grad", True)]
        if params:
            return params
    return []


# --------------------------------------------------------------------------- #
# Algorithm 2 -- PPO refinement engine
# --------------------------------------------------------------------------- #
class PPORefiner:
    """Algorithm 2: refine a pre-trained policy with PPO + mixed init + RND.

    Parameters
    ----------
    env:
        Environment (optionally a :class:`rice.envs.reset_wrapper.ResetWrapper`).
    policy:
        The frozen pre-trained policy ``pi``.  Used by the mixed-initial-state
        sampler and (deep-copied) as the initialisation of ``pi_theta``.
    mask_net:
        The trained mask network ``pi~`` from Stage 1 (may be ``None``, in which
        case importance is uniform and the critical-state branch degenerates to
        a random visited state -- the Random-explanation baseline behaviour).
    rnd:
        Optional pre-built :class:`rice.refining.rnd.RNDModule`.  Built
        automatically when ``use_rnd`` and none is given.
    config:
        :class:`RefinePPOConfig`, dict or ``None`` (defaults).
    """

    def __init__(
        self,
        env,
        policy=None,
        mask_net=None,
        rnd=None,
        config: Optional[Any] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger=None,
        observation_space=None,
        action_space=None,
        discrete: Optional[bool] = None,
        sampler=None,
        rng=None,
        seed: Optional[int] = None,
        copy_policy: Optional[bool] = None,
        store_dataset: Optional[bool] = None,
        optimizer=None,
        **kwargs,
    ) -> None:
        self.cfg = RefinePPOConfig.from_dict(config, **{k: v for k, v in kwargs.items() if k in RefinePPOConfig.__dataclass_fields__})  # type: ignore[attr-defined]
        if seed is not None:
            self.cfg.seed = seed
        self.device = str(device or self.cfg.device or "cpu")
        self.env = env
        if normalize_env_key is not None:
            try:
                env_id = normalize_env_key(env_id)
            except Exception:
                pass
        self.env_id = env_id or "default"
        self.logger = logger
        self.rng = rng if rng is not None else get_rng(self.cfg.seed)

        if self.cfg.seed is not None:
            try:
                set_seed(self.cfg.seed)
            except Exception:
                pass
            try:
                seed_env(env, self.cfg.seed)
            except Exception:
                pass

        self.observation_space = observation_space if observation_space is not None else getattr(env, "observation_space", None)
        self.action_space = action_space if action_space is not None else getattr(env, "action_space", None)

        # -- frozen pre-trained policy pi -----------------------------------
        if copy_policy is not None:
            self.cfg.copy_policy = bool(copy_policy)
        self.pretrained_policy = self._snapshot_policy(policy)

        # -- trainable pi_theta (initialised from pi) -----------------------
        self.policy = ensure_trainable_policy(
            policy=policy,
            env=env,
            env_id=self.env_id,
            device=self.device,
            copy_policy=self.cfg.copy_policy,
            **kwargs,
        )
        if policy is None and self.pretrained_policy is None:
            self.pretrained_policy = self._snapshot_policy(self.policy)

        self.obs_dim, self.action_dim, self.discrete = resolve_obs_act_dims(env, self.policy)
        if discrete is not None:
            self.discrete = bool(discrete)
        self.low, self.high = _action_bounds(self.policy, self.action_space)

        # -- trajectory length T -------------------------------------------
        if self.cfg.n_steps is None:
            self.cfg.n_steps = _resolve_horizon(env, DEFAULT_HORIZON)
        self.horizon = int(self.cfg.n_steps)

        # -- optimiser for pi_theta ----------------------------------------
        self.optimizer = optimizer
        if self.optimizer is None:
            params = policy_parameters(self.policy)
            if params and _HAS_TORCH:
                self.optimizer = torch.optim.Adam(params, lr=float(self.cfg.lr), eps=1e-5)

        # -- RND exploration bonus -----------------------------------------
        self.rnd = rnd
        if self.rnd is None and self.cfg.use_rnd and build_rnd is not None:
            try:
                rnd_kwargs: Dict[str, Any] = dict(
                    obs_dim=self.obs_dim,
                    observation_space=self.observation_space,
                    lam=float(self.cfg.lam),
                    learning_rate=float(self.cfg.rnd_lr if self.cfg.rnd_lr is not None else self.cfg.lr),
                    device=self.device,
                    seed=self.cfg.seed,
                )
                if self.cfg.rnd_hidden_sizes is not None:
                    rnd_kwargs["hidden_sizes"] = tuple(self.cfg.rnd_hidden_sizes)
                self.rnd = build_rnd(**rnd_kwargs)
            except Exception as exc:  # pragma: no cover
                self._log("warning", f"RND construction failed ({exc}); continuing without intrinsic reward.")
                self.rnd = None
        if self.rnd is not None:
            try:
                self.rnd.set_lambda(float(self.cfg.lam))
            except Exception:
                pass

        # -- mixed initial state distribution (Algorithm 2 roll-in) --------
        self.sampler = sampler
        if self.sampler is None and self.cfg.use_mixed_init:
            self.sampler = self._build_sampler(mask_net)

        self.store_dataset = self.cfg.store_dataset if store_dataset is None else bool(store_dataset)
        self.dataset: Optional[RefineRollout] = None
        self.history: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []
        self._timers: Dict[str, float] = {}
        self.total_timesteps = 0
        self.n_iterations = 0
        self._last_obs: Optional[Any] = None
        self._last_info: Dict[str, Any] = {}
        self._log("info", self.describe())

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _snapshot_policy(self, policy):
        """Keep a frozen copy of the pre-trained policy (used for roll-ins)."""
        if policy is None:
            return None
        try:
            return copy.deepcopy(policy)
        except Exception:
            return policy

    def _build_sampler(self, mask_net):
        if make_mixed_init_sampler is None:
            return None
        try:
            return make_mixed_init_sampler(
                env=self.env,
                policy=self.pretrained_policy if self.pretrained_policy is not None else self.policy,
                mask_net=mask_net,
                p=float(self.cfg.p),
                K=self.cfg.K,
                env_id=self.env_id,
                seed=self.cfg.seed,
                deterministic_policy=self.cfg.deterministic_policy,
            )
        except Exception as exc:  # pragma: no cover
            self._log("warning", f"mixed-init sampler construction failed ({exc}).")
            return None

    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        try:
            getattr(self.logger, level, self.logger.info)(message)
        except Exception:
            pass

    def describe(self) -> str:
        return (
            f"PPORefiner(env={self.env_id}, T={self.horizon}, p={self.cfg.p}, "
            f"lambda={self.cfg.lam if self.cfg.use_rnd else 0.0}, lr={self.cfg.lr}, "
            f"n_epochs={self.cfg.n_epochs}, discrete={self.discrete}, device={self.device})"
        )

    def timer_start(self, name: str) -> float:
        self._timers[f"__start__{name}"] = time.time()
        return self._timers[f"__start__{name}"]

    def timer_end(self, name: str, accumulate: bool = True) -> float:
        start = self._timers.get(f"__start__{name}", time.time())
        elapsed = time.time() - start
        if accumulate:
            self._timers[name] = self._timers.get(name, 0.0) + elapsed
        else:
            self._timers[name] = elapsed
        return elapsed

    # ------------------------------------------------------------------ #
    # Algorithm 2 -- roll-in (mixed initial state distribution)
    # ------------------------------------------------------------------ #
    def sample_initial_state(self, policy=None, reset_kwargs: Optional[Dict[str, Any]] = None):
        """Draw ``s_0 ~ mu(s)`` (Algorithm 2, lines 3-8).

        Returns ``(observation, info, mode)`` where ``mode`` is one of
        ``"critical"``, ``"default"`` or ``"fallback"``.
        """
        if self.sampler is None:
            obs, info = unpack_reset(self.env.reset())
            return obs, info, "default"
        sample = self.sampler.sample(
            policy=policy if policy is not None else self.pretrained_policy,
            reset_kwargs=reset_kwargs,
        )
        try:
            mode = str(getattr(sample, "mode", "default"))
        except Exception:
            mode = "default"
        obs = getattr(sample, "observation", sample)
        info = getattr(sample, "info", {}) or {}
        if isinstance(info, dict) and getattr(sample, "from_critical", False):
            info = dict(info)
            info["critical_start"] = True
        return obs, dict(info), mode

    def will_reset_to_critical(self) -> bool:
        """Algorithm 2 line 4: ``RAND_NUM ~ U(0,1)``; branch on ``RAND_NUM < p``."""
        if not self.cfg.use_mixed_init:
            return False
        rand_num = float(self.rng.uniform(0.0, 1.0))
        return rand_num < float(self.cfg.p)

    # ------------------------------------------------------------------ #
    # Algorithm 2 -- rollout collection
    # ------------------------------------------------------------------ #
    def collect_rollout(self, policy=None) -> RefineRollout:
        """Roll ``pi_theta`` for ``T`` steps from ``s_0 ~ mu(s)`` (lines 3-15)."""
        policy = policy if policy is not None else self.policy
        start = time.time()
        observations: List[Any] = []
        next_observations: List[Any] = []
        actions: List[Any] = []
        rewards: List[float] = []
        task_rewards: List[float] = []
        rnd_bonuses: List[float] = []
        log_probs: List[float] = []
        values: List[float] = []
        dones: List[float] = []
        truncated_flags: List[float] = []
        start_modes: List[str] = []
        episode_task_returns: List[float] = []
        episode_augmented_returns: List[float] = []
        running_task = 0.0
        running_aug = 0.0

        obs, info, mode = self.sample_initial_state()
        start_modes.append(mode)
        self._last_obs = obs
        self._last_info = info

        for t in range(self.horizon):
            action_raw, log_prob, value = _sample_from_policy(
                policy, obs, device=self.device, deterministic=self.cfg.deterministic_policy
            )
            action = prepare_action_for_env(
                action_raw,
                discrete=self.discrete,
                low=self.low,
                high=self.high,
                policy=policy,
                action_space=self.action_space,
                clip=self.cfg.clip_actions,
            )
            result = self.env.step(action)
            next_obs, task_reward, terminated, truncated, step_info = unpack_step(result)

            # intrinsic reward RND bonus for s_{t+1} (Algorithm 2 line 12)
            bonus = 0.0
            if self.rnd is not None and self.cfg.use_rnd:
                try:
                    raw_bonus = self.rnd.bonus(next_obs, update_stats=True)
                    bonus = float(np.asarray(raw_bonus).ravel()[0])
                except TypeError:
                    raw_bonus = self.rnd.bonus(next_obs)  # type: ignore[misc]
                    bonus = float(np.asarray(raw_bonus).ravel()[0])
                except Exception:
                    bonus = 0.0
                if not np.isfinite(bonus):
                    bonus = 0.0
            augmented_reward = float(task_reward) + float(self.cfg.lam) * float(bonus)

            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(action)
            rewards.append(augmented_reward)
            task_rewards.append(float(task_reward))
            rnd_bonuses.append(float(bonus))
            log_probs.append(float(log_prob))
            values.append(float(value))
            dones.append(1.0 if (terminated or truncated) else 0.0)
            truncated_flags.append(1.0 if truncated else 0.0)

            running_task += float(task_reward)
            running_aug += augmented_reward

            if terminated or truncated:
                episode_task_returns.append(float(running_task))
                episode_augmented_returns.append(float(running_aug))
                running_task = 0.0
                running_aug = 0.0
                if self.cfg.resample_on_done and t < self.horizon - 1:
                    obs, info, mode = self.sample_initial_state()
                    start_modes.append(mode)
                    self._last_obs = obs
                    self._last_info = info
                else:
                    obs = next_obs
            else:
                obs = next_obs

        # bootstrap value for a truncated final transition
        last_value = 0.0
        if truncated_flags and truncated_flags[-1] > 0.5:
            last_value = policy_values(policy, next_observations[-1])

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            last_value=last_value,
            gamma=float(self.cfg.gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        if self.cfg.normalize_advantage and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + _EPS)
        if running_task != 0.0 or running_aug != 0.0:
            episode_task_returns.append(float(running_task))
            episode_augmented_returns.append(float(running_aug))

        batch = RefineRollout(
            observations=_flatten_sequence(observations),
            next_observations=_flatten_sequence(next_observations),
            actions=(np.asarray(actions, dtype=np.int64) if self.discrete else np.asarray(actions, dtype=np.float32)),
            rewards=np.asarray(rewards, dtype=np.float32),
            task_rewards=np.asarray(task_rewards, dtype=np.float32),
            rnd_bonuses=np.asarray(rnd_bonuses, dtype=np.float32),
            log_probs=np.asarray(log_probs, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            truncated=np.asarray(truncated_flags, dtype=np.float32),
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
            episode_task_returns=episode_task_returns,
            episode_augmented_returns=episode_augmented_returns,
            start_modes=start_modes,
            p_used=float(self.cfg.p),
            lam_used=float(self.cfg.lam) if self.cfg.use_rnd else 0.0,
            wall_time=time.time() - start,
        )
        self.total_timesteps += len(batch)
        if self.store_dataset:
            self.dataset = batch
        return batch

    # ------------------------------------------------------------------ #
    # Algorithm 2 -- PPO update of pi_theta (+ RND predictor update)
    # ------------------------------------------------------------------ #
    def _parse_evaluate_actions(self, output) -> Tuple[Any, Any, Any]:
        """Tolerantly parse ``evaluate_actions`` output; returns (logp, entropy, values)."""
        if not isinstance(output, (tuple, list)):
            return None, None, None
        if len(output) == 4:  # SB3: (loss, log_prob, entropy, values)
            _, log_prob, entropy, values = output
            return log_prob, entropy, values
        if len(output) == 3:
            first, second, third = output
            # native ActorCritic convention used in this project: (log_prob, entropy, values)
            return first, second, third
        return None, None, None

    def update(self, batch: Optional[RefineRollout] = None) -> Dict[str, float]:
        """PPO clipped update on ``D`` and the RND predictor MSE update."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for PPO refinement.")
        batch = batch if batch is not None else self.dataset
        if batch is None or len(batch) == 0:
            return {}
        policy = self.policy
        if self.optimizer is None:
            params = policy_parameters(policy)
            self.optimizer = torch.optim.Adam(params, lr=float(self.cfg.lr), eps=1e-5)

        obs_t = _batch_obs_tensor(batch.observations, self.device)
        if self.discrete:
            actions_t = torch.as_tensor(np.asarray(batch.actions, dtype=np.int64), device=self.device)
        else:
            actions_t = torch.as_tensor(np.asarray(batch.actions, dtype=np.float32), device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)

        n = int(obs_t.shape[0])
        batch_size = int(max(1, min(self.cfg.batch_size, n)))
        indices = np.arange(n)
        stats: Dict[str, float] = {}
        clip_fracs: List[float] = []
        approx_kls: List[float] = []
        pg_losses: List[float] = []
        vf_losses: List[float] = []
        entropies: List[float] = []
        early_stop = False

        policy.train() if hasattr(policy, "train") else None
        for epoch in range(int(self.cfg.n_epochs)):
            self.rng.shuffle(indices)
            for start in range(0, n, batch_size):
                mb = indices[start : start + batch_size]
                mb_obs = obs_t[mb]
                mb_actions = actions_t[mb]
                mb_old_logp = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]
                mb_old_values = old_values[mb]

                log_prob = entropy = values = None
                if hasattr(policy, "evaluate_actions"):
                    try:
                        out = policy.evaluate_actions(mb_obs, mb_actions)
                        log_prob, entropy, values = self._parse_evaluate_actions(out)
                    except Exception:
                        log_prob, values, entropy = None, None, None
                if log_prob is None:
                    dist, values = policy_distribution_and_values(policy, mb_obs)
                    if dist is None:
                        continue
                    try:
                        log_prob = dist.log_prob(mb_actions)
                    except Exception:
                        continue
                    try:
                        entropy = dist.entropy()
                    except Exception:
                        entropy = torch.zeros_like(log_prob)
                if values is None:
                    _, values = policy_distribution_and_values(policy, mb_obs)
                if values is None:
                    continue
                values = values.reshape(-1)
                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=-1)
                if entropy is None:
                    entropy = torch.zeros_like(log_prob)
                if entropy.dim() > 1:
                    entropy = entropy.sum(dim=-1)

                # ---- clipped PPO policy loss -----------------------------
                ratio = torch.exp(log_prob - mb_old_logp)
                clip_range = float(self.cfg.clip_range)
                pg_loss_unclipped = -mb_adv * ratio
                pg_loss_clipped = -mb_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

                # ---- value loss ------------------------------------------
                if self.cfg.clip_range_vf is not None:
                    vf_clip = float(self.cfg.clip_range_vf)
                    values_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values, -vf_clip, vf_clip
                    )
                    vf_loss_unclipped = (values - mb_returns) ** 2
                    vf_loss_clipped = (values_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + float(self.cfg.vf_coef) * value_loss - float(self.cfg.ent_coef) * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad], float(self.cfg.max_grad_norm)
                    )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_prob - mb_old_logp
                    approx_kl = float(((torch.exp(log_ratio) - 1.0) - log_ratio).mean().cpu())
                    clip_frac = float(
                        (torch.abs(ratio - 1.0) > clip_range).float().mean().cpu()
                    )
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                pg_losses.append(float(policy_loss.detach().cpu()))
                vf_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy_loss.detach().cpu()))

                if self.cfg.target_kl is not None and approx_kl > 1.5 * float(self.cfg.target_kl):
                    early_stop = True
                    break
            if early_stop:
                break

        # ---- RND predictor update (MSE / Adam), Algorithm 2 line 17 ------
        rnd_stats: Dict[str, float] = {}
        if self.rnd is not None and self.cfg.use_rnd:
            try:
                update_batch = list(zip(batch.observations, batch.next_observations))
                rnd_stats = dict(
                    self.rnd.update_from_dataset(update_batch) or {}
                )
            except Exception:
                try:
                    rnd_stats = dict(self.rnd.update(batch.next_observations, epochs=self.cfg.rnd_update_epochs) or {})
                except Exception:
                    rnd_stats = {}

        self.n_iterations += 1
        stats.update(
            {
                "iteration": float(self.n_iterations),
                "timesteps": float(self.total_timesteps),
                "rollout/length": float(len(batch)),
                "rollout/mean_task_reward": float(np.mean(batch.task_rewards)) if len(batch) else 0.0,
                "rollout/mean_rnd_bonus": batch.mean_intrinsic_bonus,
                "rollout/mean_reward": float(np.mean(batch.rewards)) if len(batch) else 0.0,
                "rollout/critical_fraction": batch.critical_fraction,
                "rollout/n_episodes": float(len(batch.episode_task_returns)),
                "rollout/mean_episode_task_return": float(np.mean(batch.episode_task_returns))
                if batch.episode_task_returns
                else 0.0,
            }
        )
        if pg_losses:
            for key, values_list in (
                ("train/policy_loss", pg_losses),
                ("train/value_loss", vf_losses),
                ("train/entropy", entropies),
                ("train/clip_fraction", clip_fracs),
                ("train/approx_kl", approx_kls),
            ):
                stats[key] = float(np.mean(values_list))
            stats["train/early_stop"] = float(early_stop)
        for key, value in rnd_stats.items():
            try:
                stats[f"rnd/{key}"] = float(value)
            except Exception:
                continue
        if self.rnd is not None:
            try:
                rnd_std = float(np.asarray(self.rnd.statistics().get("rms_std", 0.0)))
            except Exception:
                rnd_std = 0.0
            stats["rnd/rms_std"] = rnd_std
            stats["rnd/lambda"] = float(self.cfg.lam)
        stats["p"] = float(self.cfg.p)
        return stats

    # ------------------------------------------------------------------ #
    # Algorithm 2 -- main loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        logger=None,
        progress: bool = False,
        callback: Optional[Callable[["PPORefiner"], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run Algorithm 2 iterations; returns the per-iteration metric history."""
        logger = logger if logger is not None else self.logger
        budget = int(total_timesteps if total_timesteps is not None else self.cfg.total_timesteps)
        n_iters = total_iterations if total_iterations is not None else self.cfg.n_iterations
        if n_iters is None:
            n_iters = max(1, int(np.ceil(budget / max(1, self.horizon))))

        self.timer_start("refine")
        for it in range(int(n_iters)):
            self.timer_start("rollout")
            batch = self.collect_rollout()
            roll_time = self.timer_end("rollout")
            self.timer_start("update")
            stats = self.update(batch)
            upd_time = self.timer_end("update")
            stats["time/rollout"] = roll_time
            stats["time/update"] = upd_time
            stats["time/total"] = self._timers.get("refine", 0.0)
            self.history.append(stats)

            if progress and logger is not None:
                try:
                    logger.info(
                        "[refine %d] steps=%d task_R=%.2f bonus=%.4f crit_frac=%.2f pi_loss=%.4f v_loss=%.4f kl=%.5f",
                        it + 1,
                        self.total_timesteps,
                        stats.get("rollout/mean_episode_task_return", 0.0),
                        stats.get("rollout/mean_rnd_bonus", 0.0),
                        stats.get("rollout/critical_fraction", 0.0),
                        stats.get("train/policy_loss", float("nan")),
                        stats.get("train/value_loss", float("nan")),
                        stats.get("train/approx_kl", float("nan")),
                    )
                except Exception:
                    pass
            if callback is not None:
                try:
                    callback(self)
                except Exception:
                    pass
            # optional intermediate evaluation
            if self.cfg.eval_interval and (it + 1) % int(self.cfg.eval_interval) == 0:
                eval_stats = self.evaluate(n_episodes=int(self.cfg.eval_episodes))
                eval_stats["iteration"] = float(it + 1)
                self.eval_history.append(eval_stats)
                if logger is not None:
                    try:
                        logger.info(
                            "[eval after iter %d] return=%.2f +- %.2f (n=%d)",
                            it + 1,
                            eval_stats.get("eval/mean_return", float("nan")),
                            eval_stats.get("eval/std_return", float("nan")),
                            eval_stats.get("eval/n_episodes", 0),
                        )
                    except Exception:
                        pass
            if self.total_timesteps >= budget and total_iterations is None:
                break
        self.timer_end("refine")
        return self.history

    fit = train

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        policy=None,
        max_steps: Optional[int] = None,
        use_task_reward: bool = True,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate the refined policy from the default initial distribution rho."""
        policy = policy if policy is not None else self.policy
        max_steps = int(max_steps if max_steps is not None else self.horizon)
        returns: List[float] = []
        lengths: List[int] = []
        for _ in range(int(n_episodes)):
            obs, info = unpack_reset(self.env.reset(**(reset_kwargs or {})))
            total = 0.0
            steps = 0
            while steps < max_steps:
                action_raw, _, _ = _sample_from_policy(
                    policy, obs, device=self.device, deterministic=deterministic
                )
                action = prepare_action_for_env(
                    action_raw,
                    discrete=self.discrete,
                    low=self.low,
                    high=self.high,
                    policy=policy,
                    action_space=self.action_space,
                    clip=self.cfg.clip_actions,
                )
                result = self.env.step(action)
                obs, reward, terminated, truncated, info = unpack_step(result)
                if use_task_reward:
                    task_reward = info.get("dense_reward", reward) if isinstance(info, dict) else reward
                    if not isinstance(info, dict) or "dense_reward" not in info:
                        task_reward = reward
                else:
                    task_reward = reward
                total += float(task_reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(float(total))
            lengths.append(int(steps))
        mean, std = mean_std(returns)
        stats = {
            "eval/mean_return": float(mean),
            "eval/std_return": float(std),
            "eval/n_episodes": float(len(returns)),
            "eval/mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval/returns": returns,
        }
        return stats

    # ------------------------------------------------------------------ #
    # misc API
    # ------------------------------------------------------------------ #
    def set_p(self, p: float) -> float:
        """Set the reset probability threshold ``p`` (== beta) at runtime."""
        p = float(np.clip(p, 0.0, 1.0))
        self.cfg.p = p
        if self.sampler is not None:
            try:
                self.sampler.set_p(p)
            except Exception:
                try:
                    self.sampler.p = p
                except Exception:
                    pass
        return p

    def set_lambda(self, lam: float) -> float:
        """Set the RND trade-off ``lambda`` at runtime."""
        lam = float(lam)
        self.cfg.lam = lam
        if self.rnd is not None:
            try:
                self.rnd.set_lambda(lam)
            except Exception:
                pass
        return lam

    def set_alpha(self, alpha: float) -> float:
        """Accepted for API parity with the mask trainer (alpha is Stage 1 only)."""
        self.cfg.__dict__["alpha"] = float(alpha)
        return float(alpha)

    @property
    def total_time(self) -> float:
        return float(self._timers.get("refine", 0.0))

    @property
    def seconds_per_sample(self) -> float:
        if self.total_timesteps <= 0:
            return 0.0
        return float(self.total_time / max(1, self.total_timesteps))

    def time_report(self) -> Dict[str, float]:
        return {
            "refine_time": float(self._timers.get("refine", 0.0)),
            "rollout_time": float(self._timers.get("rollout", 0.0)),
            "update_time": float(self._timers.get("update", 0.0)),
            "total_timesteps": float(self.total_timesteps),
            "seconds_per_sample": float(self.seconds_per_sample),
            "n_iterations": float(self.n_iterations),
        }

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "env_id": self.env_id,
            "method": "RICE",
            "n_iterations": self.n_iterations,
            "total_timesteps": int(self.total_timesteps),
            "p": float(self.cfg.p),
            "lambda": float(self.cfg.lam) if self.cfg.use_rnd else 0.0,
            "horizon": int(self.horizon),
            "time": self.time_report(),
            "config": self.cfg.to_dict(),
        }
        if self.history:
            last = self.history[-1]
            summary["last"] = {
                "mean_task_reward": last.get("rollout/mean_task_reward", 0.0),
                "mean_rnd_bonus": last.get("rollout/mean_rnd_bonus", 0.0),
                "critical_fraction": last.get("rollout/critical_fraction", 0.0),
                "policy_loss": last.get("train/policy_loss", 0.0),
                "value_loss": last.get("train/value_loss", 0.0),
            }
            summary["mean_critical_fraction"] = float(
                np.mean([h.get("rollout/critical_fraction", 0.0) for h in self.history])
            )
            summary["mean_rnd_bonus"] = float(
                np.mean([h.get("rollout/mean_rnd_bonus", 0.0) for h in self.history])
            )
            epi = [h.get("rollout/mean_episode_task_return", 0.0) for h in self.history]
            summary["mean_episode_task_return"] = float(np.mean(epi))
        if self.eval_history:
            summary["eval"] = self.eval_history[-1]
        if self.sampler is not None:
            try:
                summary["mixed_init"] = self.sampler.statistics()
            except Exception:
                pass
        return summary

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Persist the refined policy ``pi'`` plus metadata/config."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to save a refiner.")
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            ensure_dir(directory)
        payload: Dict[str, Any] = {
            "kind": "refiner",
            "env_id": self.env_id,
            "obs_dim": int(self.obs_dim),
            "action_dim": self.action_dim,
            "discrete": bool(self.discrete),
            "config": self.cfg.to_dict(),
            "summary": self.summary(),
        }
        try:
            payload["policy_state_dict"] = self.policy.state_dict()
            payload["policy_class"] = type(self.policy).__name__
        except Exception:
            payload["policy"] = self.policy
        if self.rnd is not None:
            try:
                payload["rnd_state_dict"] = self.rnd.state_dict()
            except Exception:
                pass
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str, env, **kwargs):
        """Rebuild a :class:`PPORefiner` from a checkpoint written by :meth:`save`."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to load a refiner.")
        payload = torch.load(path, map_location=kwargs.get("device", "cpu"))
        cfg = RefinePPOConfig.from_dict(payload.get("config"))
        policy = kwargs.pop("policy", None)
        if policy is None and "policy" in payload:
            policy = payload["policy"]
        if policy is None and build_policy is not None and "policy_state_dict" in payload:
            policy = build_policy(
                env_id=payload.get("env_id", "default"),
                obs_dim=payload.get("obs_dim"),
                action_dim=payload.get("action_dim"),
                action_space=getattr(env, "action_space", None),
                observation_space=getattr(env, "observation_space", None),
                kind="policy",
                backend="native",
                device=cfg.device,
            )
            try:
                policy.load_state_dict(payload["policy_state_dict"])
            except Exception:
                pass
        trainer = cls(env=env, policy=policy, config=cfg, **kwargs)
        return trainer


# --------------------------------------------------------------------------- #
# Module-level convenience API
# --------------------------------------------------------------------------- #
def make_refiner(env, policy=None, **kwargs) -> PPORefiner:
    """Factory: build a :class:`PPORefiner` (alias of the constructor)."""
    return PPORefiner(env=env, policy=policy, **kwargs)


build_refiner = make_refiner


def refine_policy(
    env,
    policy=None,
    mask_net=None,
    total_timesteps: Optional[int] = None,
    total_iterations: Optional[int] = None,
    p: float = DEFAULT_P,
    lam: float = DEFAULT_LAMBDA,
    K: Optional[int] = None,
    env_id: str = "default",
    config: Optional[Any] = None,
    logger=None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress: bool = False,
    evaluate: bool = False,
    eval_episodes: int = 10,
    **kwargs,
) -> Tuple[Any, PPORefiner]:
    """Algorithm 2 end-to-end: refine ``pi`` into ``pi'``.

    Returns ``(refined_policy, refiner)``.  The input policy is not modified
    (the trainable copy is ``refiner.policy == pi'``).
    """
    trainer = PPORefiner(
        env=env,
        policy=policy,
        mask_net=mask_net,
        config=config,
        env_id=env_id,
        device=device,
        logger=logger,
        seed=seed,
        **kwargs,
    )
    trainer.set_p(p)
    trainer.set_lambda(lam)
    if K is not None:
        trainer.cfg.K = K
    trainer.train(total_timesteps=total_timesteps, total_iterations=total_iterations, progress=progress)
    if evaluate:
        eval_stats = trainer.evaluate(n_episodes=eval_episodes)
        trainer.eval_history.append(eval_stats)
        if logger is not None:
            try:
                logger.info(
                    "refined policy return: %.2f +- %.2f",
                    eval_stats.get("eval/mean_return", float("nan")),
                    eval_stats.get("eval/std_return", float("nan")),
                )
            except Exception:
                pass
    if save_path:
        trainer.save(save_path)
    return trainer.policy, trainer


train_ppo_refine = refine_policy
refine = refine_policy


def evaluate_refined_policy(
    env,
    policy,
    env_id: str = "default",
    n_episodes: int = 10,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    discrete: Optional[bool] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Standalone evaluation of a (refined) policy from ``rho``.

    Used by Experiment II/IV to report the final refined reward, optionally via
    the CAGE-2 aggregate rule when the environment exposes it.
    """
    discrete = _is_discrete(policy=policy, action_space=getattr(env, "action_space", None), fallback=discrete)
    low, high = _action_bounds(policy, getattr(env, "action_space", None))
    if max_steps is None:
        max_steps = _resolve_horizon(env, DEFAULT_HORIZON)
    returns: List[float] = []
    for _ in range(int(n_episodes)):
        obs, _info = unpack_reset(env.reset())
        total = 0.0
        steps = 0
        while steps < int(max_steps):
            if _HAS_TORCH:
                try:
                    action_raw, _, _ = _sample_from_policy(policy, obs, device=device, deterministic=deterministic)
                except Exception:
                    action_raw = policy_action(policy, obs, deterministic=deterministic)
            else:  # pragma: no cover
                action_raw = policy_action(policy, obs, deterministic=deterministic)
            action = prepare_action_for_env(
                action_raw, discrete=discrete, low=low, high=high, policy=policy, action_space=getattr(env, "action_space", None)
            )
            result = env.step(action)
            obs, reward, terminated, truncated, info = unpack_step(result)
            task_reward = reward
            if isinstance(info, dict) and "dense_reward" in info:
                task_reward = info["dense_reward"]
            total += float(task_reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(float(total))
    mean, std = mean_std(returns)
    return {
        "mean_return": float(mean),
        "std_return": float(std),
        "n_episodes": float(len(returns)),
        "returns": returns,
    }


__all__ = [
    "RefinePPOConfig",
    "RefineRollout",
    "PPORefiner",
    "compute_gae",
    "make_refiner",
    "build_refiner",
    "refine_policy",
    "train_ppo_refine",
    "refine",
    "evaluate_refined_policy",
    "policy_action",
    "eval_policy_action",
    "policy_values",
    "prepare_action_for_env",
    "ensure_trainable_policy",
    "resolve_obs_act_dims",
    "unpack_step",
    "unpack_reset",
    "DEFAULT_HORIZON",
    "DEFAULT_PPO_LR",
    "DEFAULT_GAMMA",
    "DEFAULT_GAE_LAMBDA",
    "DEFAULT_CLIP_RANGE",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_P",
    "DEFAULT_LAMBDA",
]
