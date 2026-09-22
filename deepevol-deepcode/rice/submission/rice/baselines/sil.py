"""Self-Imitation Learning (SIL) baseline for RICE (Oh et al., 2018).

Paper context (RICE, Cheng et al., ICML 2024, Appendix C.3 "Comparison with
Self-Imitation Learning" / Table 5)::

    "We compare RICE against the self-imitation learning (SIL) approach
     (Oh et al., 2018) across four MuJoCo games. ... While self-imitation
     learning has the advantage of encouraging the agent to imitate past
     successful experiences by prioritizing them in the replay buffer, it
     cannot address scenarios where the agent (and its past experience) has
     errors or sub-optimal actions. In contrast, RICE constructs a mixed
     initial distribution based on the identified critical states (using
     explanation methods) and encourages the agent to explore the new initial
     states."

So this module implements SIL as an *additional refining baseline*: starting
from the same frozen pre-trained (bottlenecked) policy ``pi`` that RICE refines,
SIL continues on-policy actor-critic training while maintaining a self-imitation
replay buffer storing transitions whose empirical return exceeds the current
value estimate (i.e. past "successful experiences"), and additionally optimises
the SIL objective on batches drawn from that buffer::

    L_sil(theta) = E_{(s,a,R) ~ D_sil} [
        - log pi_theta(a|s) * (R - V_theta(s))_+
        + beta * (R - V_theta(s))^2 * 1{R > V_theta(s)} ]

which encourages the policy to imitate the (relatively) good behaviour already
present in its own replay buffer without any explicit exploration mechanism
(no mixed initial state distribution and no RND bonus -- those are RICE's
contributions, exactly the gap the paper highlights).

The module is defensive: every ``rice`` import is wrapped in ``try/except`` so
the file is importable even when torch/SB3 or sibling modules are missing.  The
public surface intentionally mirrors :mod:`rice.baselines.gail` and
:mod:`rice.baselines.ppo_finetune` (config dataclass with ``from_dict``/
``to_dict``, a trainer with ``train``/``evaluate``/``save``/``summary``, a plain
functional driver ``train_sil``, and factories ``make_sil``/``build_sil``) so
experiment drivers can swap refining baselines transparently.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Optional / defensive imports
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - torch is expected to exist in real runs
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    Adam = None
    _HAS_TORCH = False

try:  # pragma: no cover
    from ..utils.seeding import get_rng, set_seed
except Exception:  # pragma: no cover
    def get_rng(seed: Optional[int] = None):  # type: ignore
        return np.random.RandomState(seed)

    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        np.random.seed(int(seed) % (2 ** 32))
        return int(seed)

try:  # pragma: no cover
    from ..utils.logging import get_logger
except Exception:  # pragma: no cover
    def get_logger(name="rice", out_dir=None, level=None, **kwargs):  # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(level or logging.INFO)
        return logger

try:  # pragma: no cover
    from ..utils.io import ensure_dir
except Exception:  # pragma: no cover
    def ensure_dir(path):  # type: ignore
        if path:
            os.makedirs(path, exist_ok=True)
        return path

_HAS_POLICIES = False
try:  # pragma: no cover
    from ..models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        sample_random_action,
        save_policy,
    )

    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        key = str(env_id or "default").strip().lower()
        for suffix in ("-v0", "-v1", "-v2", "-v3", "-v4", "-v5"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key.replace("-", "_").replace(" ", "_")

    def build_policy(*args, **kwargs):  # type: ignore
        raise ImportError("rice.models.policies.build_policy is unavailable")

    def save_policy(*args, **kwargs):  # type: ignore
        raise ImportError("rice.models.policies.save_policy is unavailable")

    def load_policy(*args, **kwargs):  # type: ignore
        raise ImportError("rice.models.policies.load_policy is unavailable")

    def sample_random_action(*args, **kwargs):  # type: ignore
        return np.zeros(1, dtype=np.float32)


try:  # pragma: no cover
    from ..refining.ppo_refine import DEFAULT_HORIZON
except Exception:  # pragma: no cover
    DEFAULT_HORIZON = 1000

_HAS_REFINER = False
try:  # pragma: no cover
    from ..refining.ppo_refine import PPORefiner, RefinePPOConfig, refine_policy

    _HAS_REFINER = True
except Exception:  # pragma: no cover
    PPORefiner = None  # type: ignore
    RefinePPOConfig = None  # type: ignore

    def refine_policy(*args, **kwargs):  # type: ignore
        raise ImportError("rice.refining.ppo_refine is unavailable")


__all__ = [
    # config
    "SILConfig",
    # buffer
    "SelfImitationBuffer",
    "SILBuffer",
    # rollout container
    "SILRollout",
    # trainer / refiner
    "SILTrainer",
    "SILRefiner",
    # losses
    "sil_policy_loss",
    "sil_value_loss",
    "self_imitation_loss",
    "sil_advantages",
    "compute_gae",
    # drivers
    "train_sil",
    "sil",
    "sil_baseline",
    "run_sil",
    "approximate_policy_with_sil",
    "collect_successful_experiences",
    "sil_update_from_buffer",
    # factories
    "make_sil",
    "build_sil",
    "make_sil_refiner",
    "sil_for",
    "describe_sil",
    # helpers
    "unpack_step",
    "unpack_reset",
    "policy_action",
    "prepare_action_for_env",
    # constants
    "DEFAULT_LR",
    "DEFAULT_GAMMA",
    "DEFAULT_GAE_LAMBDA",
    "DEFAULT_CLIP_RANGE",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_VF_COEF",
    "DEFAULT_ENT_COEF",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_N_STEPS",
    "DEFAULT_TOTAL_TIMESTEPS",
    "DEFAULT_SIL_COEF",
    "DEFAULT_SIL_BATCH_SIZE",
    "DEFAULT_BUFFER_SIZE",
    "SIL_REWARD_MODES",
]


# --------------------------------------------------------------------------- #
# Constants (SB3 defaults where the paper is silent)
# --------------------------------------------------------------------------- #
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_N_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_VF_COEF = 0.5
DEFAULT_ENT_COEF = 0.0
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_N_STEPS = 2048
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_SIL_COEF = 1.0
DEFAULT_SIL_BATCH_SIZE = 128
DEFAULT_SIL_EPOCHS = 1
DEFAULT_BUFFER_SIZE = 100_000
DEFAULT_MIN_BUFFER_SIZE = 1000
DEFAULT_SUCCESS_QUANTILE = 0.0  # keep all positive-advantage transitions
SIL_REWARD_MODES = ("sil", "on_policy", "task", "hybrid")
DEFAULT_REWARD_MODE = "sil"


# --------------------------------------------------------------------------- #
# Small gym-API helpers
# --------------------------------------------------------------------------- #
def unpack_step(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise a ``env.step`` result to ``(obs, reward, terminated, truncated, info)``."""
    if isinstance(result, tuple):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
        if len(result) == 4:
            obs, reward, done, info = result
            done = bool(done)
            return obs, float(reward), done, False, dict(info or {})
        if len(result) == 3:
            obs, reward, info = result
            return obs, float(reward), False, False, dict(info or {})
    raise ValueError(f"Unsupported step result of type {type(result)}")


def unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:
    """Normalise a ``env.reset`` result to ``(obs, info)``."""
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def _flat(observation: Any) -> np.ndarray:
    """Flatten an observation (array / dict / scalar) to a 1-D float32 array."""
    if observation is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(observation, dict):
        parts = [np.asarray(observation[k], dtype=np.float32).ravel() for k in sorted(observation)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if isinstance(observation, (list, tuple)) and observation and isinstance(observation[0], dict):
        return _flat(observation[0])
    arr = np.asarray(observation, dtype=np.float32)
    return arr.ravel()


def _mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def _resolve_horizon(env: Any, default: int = DEFAULT_HORIZON) -> int:
    """Best-effort resolution of the episode horizon T of a (possibly wrapped) env."""
    if env is None:
        return int(default)
    for attr in ("rice_max_episode_steps", "_max_episode_steps", "max_episode_steps"):
        try:
            value = getattr(env, attr, None)
            if value:
                return int(value)
        except Exception:
            continue
    spec = getattr(env, "rice_env_spec", None)
    if spec is not None:
        try:
            steps = getattr(spec, "max_episode_steps", None)
            if steps:
                return int(steps)
        except Exception:
            pass
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return _resolve_horizon(inner, default)
    return int(default)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SILConfig:
    """Hyper-parameters for the Self-Imitation Learning baseline.

    Unspecified PPO hyper-parameters follow the Stable-Baselines3 defaults used
    everywhere else in the RICE codebase (``gamma=0.99``, ``clip_range=0.2``,
    ``lr=3e-4``, ``n_epochs=10``, GAE ``lambda=0.95``).
    """

    env_id: str = "default"
    # optim / PPO
    lr: float = DEFAULT_LR
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_range: float = DEFAULT_CLIP_RANGE
    clip_range_vf: Optional[float] = None
    n_epochs: int = DEFAULT_N_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    vf_coef: float = DEFAULT_VF_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    target_kl: Optional[float] = None
    normalize_advantage: bool = True
    n_steps: Optional[int] = None
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    total_iterations: Optional[int] = None
    # SIL specific
    sil_coef: float = DEFAULT_SIL_COEF
    sil_batch_size: int = DEFAULT_SIL_BATCH_SIZE
    sil_update_epochs: int = DEFAULT_SIL_EPOCHS
    buffer_size: int = DEFAULT_BUFFER_SIZE
    min_buffer_size: int = DEFAULT_MIN_BUFFER_SIZE
    use_positive_only: bool = True
    positive_quantile: float = DEFAULT_SUCCESS_QUANTILE
    sil_vf_coef: float = 0.0
    use_onpolicy_ppo: bool = True
    reward_mode: str = DEFAULT_REWARD_MODE
    # misc
    device: str = "cpu"
    seed: Optional[int] = None
    log_interval: int = 1
    eval_episodes: int = 10
    deterministic_eval: bool = True
    copy_policy: bool = True
    hidden_sizes: Optional[Sequence[int]] = None
    activation: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- aliases ---------------------------------------------------------- #
    @property
    def learning_rate(self) -> float:
        return float(self.lr)

    @property
    def timesteps(self) -> int:
        return int(self.total_timesteps)

    @property
    def beta(self) -> float:
        """SIL value-loss weight (Oh et al. name the coefficient ``beta``)."""
        return float(self.sil_coef)

    @property
    def lam(self) -> float:
        return float(self.gae_lambda)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("extra", None)
        return data

    @classmethod
    def from_dict(cls, cfg: Optional[Any] = None, **overrides: Any) -> "SILConfig":
        """Build a config from a (possibly nested) YAML/dict config.

        Accepts nested sections (``sil``/``self_imitation``/``refine``/``ppo``/
        ``finetune``/``baseline``) and common alias keys.
        """
        if isinstance(cfg, SILConfig):
            base = cfg.to_dict()
        else:
            data: Dict[str, Any] = {}
            if isinstance(cfg, dict):
                data = dict(cfg)
                for section in (
                    "sil",
                    "self_imitation",
                    "self_imitation_learning",
                    "refine",
                    "ppo",
                    "finetune",
                    "ppo_finetune",
                    "baseline",
                    "default",
                ):
                    sub = data.get(section)
                    if isinstance(sub, dict):
                        merged = {k: v for k, v in data.items() if not isinstance(v, dict)}
                        merged.update(sub)
                        data = merged
                        break
            base = {}
            known = set(getattr(cls, "__dataclass_fields__", {}).keys())
            aliases = {
                "learning_rate": "lr",
                "lr_actor": "lr",
                "gamma_discount": "gamma",
                "lam": "gae_lambda",
                "lambda": "gae_lambda",
                "gae": "gae_lambda",
                "clip": "clip_range",
                "clip_ratio": "clip_range",
                "n_epoch": "n_epochs",
                "epochs": "n_epochs",
                "bs": "batch_size",
                "n_step": "n_steps",
                "total_samples": "total_timesteps",
                "timesteps": "total_timesteps",
                "num_timesteps": "total_timesteps",
                "sil_lambda": "sil_coef",
                "sil_alpha": "sil_coef",
                "self_imitation_coef": "sil_coef",
                "beta": "sil_coef",
                "buf_size": "buffer_size",
                "replay_size": "buffer_size",
                "min_buffer": "min_buffer_size",
                "sil_bs": "sil_batch_size",
                "sil_epochs": "sil_update_epochs",
                "positive_only": "use_positive_only",
                "quantile": "positive_quantile",
                "env": "env_id",
            }
            for key, value in data.items():
                key_norm = str(key).strip().lower()
                target = aliases.get(key_norm, key_norm)
                if target in known:
                    base[target] = value
                elif key_norm in known:
                    base[key_norm] = value
        for key, value in overrides.items():
            if value is not None:
                base[key] = value
        known = set(getattr(cls, "__dataclass_fields__", {}).keys())
        return cls(**{k: v for k, v in base.items() if k in known})

    def __post_init__(self) -> None:
        if self.reward_mode not in SIL_REWARD_MODES:
            self.reward_mode = DEFAULT_REWARD_MODE
        if self.n_steps is not None:
            self.n_steps = int(self.n_steps)
        self.batch_size = int(self.batch_size)
        self.sil_batch_size = int(self.sil_batch_size)
        self.buffer_size = int(self.buffer_size)


# --------------------------------------------------------------------------- #
# Losses (Oh et al., 2018 Eq. 4-6)
# --------------------------------------------------------------------------- #
def sil_policy_loss(
    log_probs: Any,
    advantages: Any,
    mask: Any = None,
    coef: float = 1.0,
) -> Any:
    """SIL policy loss ``-log pi(a|s) * (R - V)_+`` (advantage clipped at 0)."""
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for sil_policy_loss")
    positive = torch.clamp(advantages, min=0.0)
    if mask is not None:
        positive = positive * mask
    if log_probs.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=log_probs.device) * float(coef)
    return -(log_probs * positive.detach()).mean() * float(coef)


def sil_value_loss(
    values: Any,
    returns: Any,
    advantages: Any = None,
    mask: Any = None,
    positive_only: bool = True,
    coef: float = 0.5,
) -> Any:
    """SIL value loss ``(R - V)^2`` restricted to the successful (positive) set."""
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for sil_value_loss")
    if mask is None and positive_only:
        mask = ((returns - values.detach()) > 0.0).to(values.dtype)
    diff = returns - values
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return (diff ** 2).sum() / denom * float(coef)
    return (diff ** 2).mean() * float(coef)


def self_imitation_loss(
    log_probs: Any,
    values: Any,
    returns: Any,
    advantages: Any = None,
    policy_coef: float = 1.0,
    value_coef: float = 0.0,
    positive_only: bool = True,
    mask: Any = None,
) -> Dict[str, Any]:
    """Combined SIL objective (policy term + optional value term)."""
    if advantages is None:
        advantages = returns - values.detach()
    if mask is None and positive_only:
        mask = (advantages > 0.0).to(values.dtype)
    policy_term = sil_policy_loss(log_probs, advantages, mask=mask, coef=policy_coef)
    value_term = sil_value_loss(
        values, returns, advantages=advantages, mask=mask, positive_only=positive_only, coef=value_coef
    )
    return {
        "policy_loss": policy_term,
        "value_loss": value_term,
        "loss": policy_term + value_term,
    }


def sil_advantages(returns: Any, values: Any, positive_only: bool = True) -> Any:
    """``(R - V)_+`` advantages used by the SIL objective."""
    if torch is None:  # pragma: no cover
        returns = np.asarray(returns, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        adv = returns - values
        return np.clip(adv, 0.0, None) if positive_only else adv
    adv = returns - values
    return torch.clamp(adv, min=0.0) if positive_only else adv


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[float],
    last_value: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (same convention as the RICE refiner)."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = float(last_value) if t == n - 1 else float(values[t + 1])
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# --------------------------------------------------------------------------- #
# Self-imitation replay buffer
# --------------------------------------------------------------------------- #
class SelfImitationBuffer:
    """Ring buffer of transitions whose return exceeded the value estimate.

    Stores ``(s, a, R_t, A_t, log_prob)`` for "successful" steps, prioritising
    them by ``A_t = R_t - V(s_t) > 0`` (Oh et al., 2018).  Torch tensors are
    stored as numpy arrays so the buffer stays pickle-friendly.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_BUFFER_SIZE,
        min_size: int = DEFAULT_MIN_BUFFER_SIZE,
        positive_only: bool = True,
        positive_quantile: float = DEFAULT_SUCCESS_QUANTILE,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        discrete: bool = False,
        seed: Optional[int] = None,
        rng: Any = None,
    ) -> None:
        self.capacity = int(capacity)
        self.min_size = int(min_size)
        self.positive_only = bool(positive_only)
        self.positive_quantile = float(positive_quantile)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = bool(discrete)
        self.rng = rng or get_rng(seed)
        self._observations: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._returns: List[float] = []
        self._advs: List[float] = []
        self._log_probs: List[float] = []
        self.total_added = 0
        self.total_rejected = 0

    # -- introspection ---------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._returns)

    @property
    def ready(self) -> bool:
        return len(self) >= max(1, self.min_size)

    def clear(self) -> None:
        self._observations.clear()
        self._actions.clear()
        self._returns.clear()
        self._advs.clear()
        self._log_probs.clear()

    def statistics(self) -> Dict[str, float]:
        advs = np.asarray(self._advs, dtype=np.float32) if self._advs else np.zeros(0)
        rets = np.asarray(self._returns, dtype=np.float32) if self._returns else np.zeros(0)
        return {
            "buffer/size": float(len(self)),
            "buffer/ready": float(self.ready),
            "buffer/total_added": float(self.total_added),
            "buffer/total_rejected": float(self.total_rejected),
            "buffer/mean_advantage": float(advs.mean()) if advs.size else 0.0,
            "buffer/mean_return": float(rets.mean()) if rets.size else 0.0,
            "buffer/max_return": float(rets.max()) if rets.size else 0.0,
        }

    # -- insertion -------------------------------------------------------- #
    def add(
        self,
        observations: Any,
        actions: Any,
        returns: Any,
        advantages: Any = None,
        log_probs: Any = None,
    ) -> int:
        """Add a batch of transitions; returns the number actually stored."""
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations[None, :]
        returns = np.asarray(returns, dtype=np.float32).reshape(-1)
        advantages = (
            np.asarray(advantages, dtype=np.float32).reshape(-1)
            if advantages is not None
            else returns.copy()
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        log_probs_arr = (
            np.asarray(log_probs, dtype=np.float32).reshape(-1)
            if log_probs is not None
            else np.zeros(len(returns), dtype=np.float32)
        )

        kept = 0
        threshold = self._threshold(advantages)
        for i in range(len(returns)):
            adv = float(advantages[i])
            if self.positive_only and adv <= 0.0:
                self.total_rejected += 1
                continue
            if np.isfinite(threshold) and adv < threshold:
                self.total_rejected += 1
                continue
            self._push(
                observations[i], actions[i], float(returns[i]), adv, float(log_probs_arr[i])
            )
            kept += 1
        return kept

    def _threshold(self, advantages: np.ndarray) -> float:
        if self.positive_quantile <= 0.0 or advantages.size == 0:
            return float("-inf")
        return float(np.quantile(advantages, self.positive_quantile))

    def _push(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        ret: float,
        advantage: float,
        log_prob: float,
    ) -> None:
        if self.obs_dim is None:
            self.obs_dim = int(np.asarray(observation).size)
        if self.action_dim is None:
            self.action_dim = int(np.asarray(action).size)
        if len(self) >= self.capacity:
            self._observations.pop(0)
            self._actions.pop(0)
            self._returns.pop(0)
            self._advs.pop(0)
            self._log_probs.pop(0)
        self._observations.append(np.asarray(observation, dtype=np.float32).reshape(-1))
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._returns.append(float(ret))
        self._advs.append(float(advantage))
        self._log_probs.append(float(log_prob))
        self.total_added += 1

    def add_trajectory(
        self,
        observations: Any,
        actions: Any,
        rewards: Any,
        values: Any = None,
        gamma: float = DEFAULT_GAMMA,
        last_value: float = 0.0,
        log_probs: Any = None,
    ) -> int:
        """Insert a whole trajectory, using the discounted return-to-go as ``R_t``."""
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        n = len(rewards)
        returns = np.zeros(n, dtype=np.float32)
        running = float(last_value)
        for t in reversed(range(n)):
            running = float(rewards[t]) + gamma * running
            returns[t] = running
        advantages = None
        if values is not None:
            advantages = returns - np.asarray(values, dtype=np.float32).reshape(-1)
        return self.add(observations, actions, returns, advantages=advantages, log_probs=log_probs)

    # -- sampling --------------------------------------------------------- #
    def sample(self, batch_size: int, rng: Any = None) -> Dict[str, np.ndarray]:
        size = len(self)
        if size == 0:
            raise RuntimeError("Cannot sample from an empty SelfImitationBuffer")
        n = int(min(batch_size, size))
        rng = rng or self.rng
        if hasattr(rng, "choice"):
            idx = rng.choice(size, size=n, replace=(n < size))
        else:  # pragma: no cover
            idx = np.random.RandomState().choice(size, size=n, replace=(n < size))
        obs = np.stack([self._observations[i] for i in idx], axis=0)
        act = np.stack([self._actions[i] for i in idx], axis=0)
        returns = np.asarray([self._returns[i] for i in idx], dtype=np.float32)
        advs = np.asarray([self._advs[i] for i in idx], dtype=np.float32)
        log_probs = np.asarray([self._log_probs[i] for i in idx], dtype=np.float32)
        return {
            "observations": obs,
            "actions": act,
            "returns": returns,
            "advantages": advs,
            "log_probs": log_probs,
        }

    def all_returns(self) -> np.ndarray:
        return np.asarray(self._returns, dtype=np.float32)


SILBuffer = SelfImitationBuffer


# --------------------------------------------------------------------------- #
# Rollout container
# --------------------------------------------------------------------------- #
@dataclass
class SILRollout:
    """Dataset collected during one SIL/PPO iteration."""

    observations: List[Any] = field(default_factory=list)
    next_observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    task_rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    truncated: List[float] = field(default_factory=list)
    advantages: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    episode_task_returns: List[float] = field(default_factory=list)
    start_modes: List[str] = field(default_factory=list)
    sil_added: int = 0

    def __len__(self) -> int:
        return len(self.rewards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_steps": len(self),
            "mean_reward": _mean(self.task_rewards),
            "mean_episode_reward": _mean(self.episode_task_returns),
            "n_episodes": len(self.episode_task_returns),
            "sil_added": int(self.sil_added),
        }


# --------------------------------------------------------------------------- #
# Policy helpers
# --------------------------------------------------------------------------- #
def _policy_observation(policy: Any, observation: Any) -> Any:
    """Build a batched torch observation tensor for a policy."""
    obs = _flat(observation)
    if torch is not None:
        return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    return obs[None, :]  # pragma: no cover


def policy_distribution_and_values(policy: Any, observations: Any) -> Tuple[Any, Any]:
    """Return ``(distribution, values)`` for a native or SB3 actor-critic policy."""
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for policy_distribution_and_values")
    obs_tensor = observations
    if not isinstance(obs_tensor, torch.Tensor):
        obs_tensor = torch.as_tensor(np.asarray(obs_tensor, dtype=np.float32))
    distribution = policy.get_distribution(obs_tensor)
    if hasattr(policy, "predict_values"):
        values = policy.predict_values(obs_tensor)
    else:  # pragma: no cover - very defensive
        values = torch.zeros(obs_tensor.shape[0], 1)
    return distribution, values


def policy_action(policy: Any, observation: Any, deterministic: bool = False) -> np.ndarray:
    """Sample (or greedily pick) an action from a policy (native / SB3 / callable)."""
    if policy is None or observation is None:
        return np.zeros(1, dtype=np.float32)
    if hasattr(policy, "predict") and not hasattr(policy, "get_distribution"):
        action, _ = policy.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)
    if hasattr(policy, "act"):
        try:
            action = policy.act(observation, deterministic=deterministic)
            return np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            pass
    if hasattr(policy, "predict"):
        action, _ = policy.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)
    if callable(policy):
        return np.asarray(policy(observation), dtype=np.float32).reshape(-1)
    raise TypeError(f"Cannot extract an action from policy {type(policy)}")


def prepare_action_for_env(
    action: Any, action_space: Any = None, discrete: Optional[bool] = None
) -> Any:
    """Convert a policy action into a valid environment action."""
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if discrete is None:
        discrete = bool(getattr(action_space, "n", None) is not None)
    if discrete:
        n = max(1, int(getattr(action_space, "n", 1) or 1))
        return int(np.clip(round(float(arr.reshape(-1)[0])), 0, n - 1))
    if action_space is not None and hasattr(action_space, "low") and hasattr(action_space, "high"):
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if low.size == arr.size:
            arr = np.clip(arr, low, high)
    return arr.astype(np.float32)


def resolve_obs_act_dims(env: Any = None, policy: Any = None) -> Tuple[int, int]:
    """Best-effort ``(obs_dim, action_dim)`` resolution."""
    obs_dim, action_dim = 0, 0
    space = getattr(env, "observation_space", None)
    if space is not None:
        try:
            obs_dim = int(np.prod(getattr(space, "shape", (0,))))
        except Exception:
            obs_dim = 0
    space = getattr(env, "action_space", None)
    if space is not None:
        n = getattr(space, "n", None)
        if n is not None:
            action_dim = int(n)
        else:
            try:
                action_dim = int(np.prod(getattr(space, "shape", (0,))))
            except Exception:
                action_dim = 0
    if policy is not None:
        if obs_dim == 0:
            for attr in ("obs_dim", "observation_dim", "input_dim"):
                value = getattr(policy, attr, None)
                if isinstance(value, int) and value > 0:
                    obs_dim = value
                    break
        if action_dim == 0:
            value = getattr(policy, "action_dim", None)
            if isinstance(value, int) and value > 0:
                action_dim = value
    return obs_dim, action_dim


def ensure_trainable_policy(
    policy: Any = None,
    env: Any = None,
    env_id: str = "default",
    device: str = "cpu",
    copy_policy: bool = True,
    **kwargs: Any,
) -> Any:
    """Clone (or build) a trainable policy initialised from the frozen ``pi``."""
    if policy is None:
        if not _HAS_POLICIES:
            raise ImportError("A policy is required (rice.models.policies unavailable)")
        obs_dim, action_dim = resolve_obs_act_dims(env, None)
        space = getattr(env, "action_space", None)
        discrete = space is not None and getattr(space, "n", None) is not None
        return build_policy(
            env_id,
            obs_dim=obs_dim or None,
            action_dim=action_dim or None,
            action_space=space,
            observation_space=getattr(env, "observation_space", None),
            discrete=discrete,
            device=device,
        )
    if copy_policy and hasattr(policy, "state_dict"):
        try:
            cloned = copy.deepcopy(policy)
            if hasattr(cloned, "to"):
                cloned.to(device)
            return cloned
        except Exception:
            pass
    if hasattr(policy, "to"):
        try:
            policy.to(device)
        except Exception:
            pass
    return policy


def _policy_parameters(policy: Any) -> List[Any]:
    if hasattr(policy, "parameters"):
        params = [p for p in policy.parameters() if getattr(p, "requires_grad", True)]
        if params:
            return params
    if nn is not None and isinstance(policy, nn.Module):  # pragma: no cover
        return list(policy.parameters())
    raise TypeError(f"Policy {type(policy)} exposes no trainable parameters")


def _policy_log_prob_values(policy: Any, observations: Any, actions: Any) -> Tuple[Any, Any, Any]:
    """Return ``(log_prob, entropy, values)`` for a batch, handling both APIs."""
    distribution, values = policy_distribution_and_values(policy, observations)
    log_prob = distribution.log_prob(actions)
    if log_prob.ndim > 1:
        log_prob = log_prob.sum(dim=-1)
    entropy = distribution.entropy()
    if entropy.ndim > 1:
        entropy = entropy.sum(dim=-1)
    return log_prob, entropy, values.reshape(-1)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class SILTrainer:
    """Self-Imitation Learning refining baseline (Oh et al., 2018).

    Continues PPO training from the frozen pre-trained policy ``pi`` while
    maintaining a self-imitation buffer of "successful" past transitions and
    optimising the SIL objective on batches drawn from it.  It intentionally
    contains none of RICE's contributions (no mixed initial state distribution,
    no critical-state reset, no RND bonus) -- matching the paper's Table 5
    comparison.
    """

    name = "SIL"

    def __init__(
        self,
        env: Any,
        policy: Any = None,
        config: Optional[Any] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        buffer: Optional[SelfImitationBuffer] = None,
        observation_space: Any = None,
        action_space: Any = None,
        discrete: Optional[bool] = None,
        rng: Any = None,
        seed: Optional[int] = None,
        copy_policy: Optional[bool] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch is required for the SIL baseline")

        if isinstance(config, str):
            env_id, config = config, None
        config_kwargs = {k: v for k, v in kwargs.items() if k in getattr(SILConfig, "__dataclass_fields__", {})}
        self.config = SILConfig.from_dict(config, env_id=env_id, **config_kwargs)
        self.env = env
        self.env_id = normalize_env_key(self.config.env_id or env_id)
        self.device = str(self.config.device or device or "cpu")
        self.logger = logger or get_logger("rice.baselines.sil")
        self.seed = self.config.seed if self.config.seed is not None else seed
        self.rng = rng or get_rng(self.seed)
        self.discrete = discrete
        if self.discrete is None:
            space = action_space if action_space is not None else getattr(env, "action_space", None)
            self.discrete = bool(space is not None and getattr(space, "n", None) is not None)
        self.observation_space = (
            observation_space if observation_space is not None else getattr(env, "observation_space", None)
        )
        self.action_space = action_space if action_space is not None else getattr(env, "action_space", None)
        self.copy_policy = self.config.copy_policy if copy_policy is None else bool(copy_policy)

        self.pretrained_policy = policy  # kept frozen for reference
        self.policy = ensure_trainable_policy(
            policy, env=env, env_id=self.env_id, device=self.device, copy_policy=self.copy_policy
        )
        if self.policy is not None and hasattr(self.policy, "set_action_space"):
            low = getattr(self.action_space, "low", None)
            high = getattr(self.action_space, "high", None)
            if low is not None and high is not None:
                try:
                    self.policy.set_action_space(low, high)
                except Exception:
                    pass

        obs_dim, action_dim = resolve_obs_act_dims(env, self.policy)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.buffer = buffer or SelfImitationBuffer(
            capacity=self.config.buffer_size,
            min_size=self.config.min_buffer_size,
            positive_only=self.config.use_positive_only,
            positive_quantile=self.config.positive_quantile,
            obs_dim=obs_dim or None,
            action_dim=action_dim or None,
            discrete=bool(self.discrete),
            seed=self.seed,
        )
        self.optimizer = Adam(_policy_parameters(self.policy), lr=float(self.config.lr))
        self.n_steps = int(self.config.n_steps or _resolve_horizon(env))
        self.total_steps = 0
        self.iteration = 0
        self.history: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []
        self._timers: Dict[str, float] = {}
        self._timer_starts: Dict[str, float] = {}
        self._current_obs: Any = None
        self._current_episode_task_return = 0.0
        if verbose:
            self.logger.info(describe_sil(self))

    # -- timers ----------------------------------------------------------- #
    def timer_start(self, name: str) -> float:
        self._timer_starts[name] = time.time()
        return self._timer_starts[name]

    def timer_end(self, name: str, accumulate: bool = True) -> float:
        start = self._timer_starts.pop(name, None)
        elapsed = (time.time() - start) if start is not None else 0.0
        self._timers[name] = (self._timers.get(name, 0.0) + elapsed) if accumulate else elapsed
        return elapsed

    @property
    def total_time(self) -> float:
        return float(sum(self._timers.values()))

    @property
    def seconds_per_sample(self) -> float:
        return self.total_time / max(1, self.total_steps)

    def time_report(self) -> Dict[str, float]:
        report = dict(self._timers)
        report["total_time"] = self.total_time
        report["seconds_per_sample"] = self.seconds_per_sample
        return report

    # -- rollouts --------------------------------------------------------- #
    def _maybe_reset(self) -> Any:
        if self._current_obs is None:
            obs, _info = unpack_reset(self.env.reset())
            self._current_obs = obs
            self._current_episode_task_return = 0.0
        return self._current_obs

    def collect_rollout(self, n_steps: Optional[int] = None) -> SILRollout:
        """Collect ``n_steps`` transitions with the current policy (on-policy)."""
        n_steps = int(n_steps or self.n_steps)
        rollout = SILRollout()
        obs = self._maybe_reset()
        for _ in range(n_steps):
            obs_tensor = _policy_observation(self.policy, obs)
            with torch.no_grad():
                distribution, values = policy_distribution_and_values(self.policy, obs_tensor)
                action_tensor = distribution.sample()
                log_prob = distribution.log_prob(action_tensor)
                if log_prob.ndim > 1:
                    log_prob = log_prob.sum(dim=-1)
            action_np = action_tensor.detach().cpu().numpy().reshape(-1)
            env_action = prepare_action_for_env(action_np, self.action_space, self.discrete)
            next_obs, reward, terminated, truncated, _info = unpack_step(self.env.step(env_action))

            rollout.observations.append(_flat(obs))
            rollout.next_observations.append(_flat(next_obs))
            rollout.actions.append(np.asarray(action_np, dtype=np.float32).reshape(-1))
            rollout.rewards.append(float(reward))
            rollout.task_rewards.append(float(reward))
            rollout.log_probs.append(float(log_prob.reshape(-1)[0].item()))
            rollout.values.append(float(np.asarray(values.detach().cpu()).reshape(-1)[0]))
            rollout.dones.append(1.0 if terminated else 0.0)
            rollout.truncated.append(1.0 if truncated else 0.0)
            rollout.start_modes.append("default")

            self._current_episode_task_return += float(reward)
            self.total_steps += 1
            done = bool(terminated or truncated)
            if done:
                rollout.episode_task_returns.append(self._current_episode_task_return)
                self._current_episode_task_return = 0.0
                obs, _info = unpack_reset(self.env.reset())
            else:
                obs = next_obs
        self._current_obs = obs

        # Bootstrapped value for GAE on time-limit truncation.
        last_value = 0.0
        if rollout.truncated and rollout.truncated[-1] > 0.0 and not rollout.dones[-1]:
            with torch.no_grad():
                _dist, value_tensor = policy_distribution_and_values(
                    self.policy, _policy_observation(self.policy, obs)
                )
            last_value = float(np.asarray(value_tensor.detach().cpu()).reshape(-1)[0])
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            last_value=last_value,
            gamma=float(self.config.gamma),
            gae_lambda=float(self.config.gae_lambda),
        )
        rollout.advantages = advantages
        rollout.returns = returns

        # Populate the self-imitation buffer with "successful" transitions.
        if self.config.reward_mode != "task":
            added = self.buffer.add(
                np.asarray(rollout.observations, dtype=np.float32),
                np.asarray(rollout.actions, dtype=np.float32),
                returns,
                advantages=advantages,
                log_probs=np.asarray(rollout.log_probs, dtype=np.float32),
            )
            rollout.sil_added = int(added)
        return rollout

    # -- updates ---------------------------------------------------------- #
    def _to_tensor(self, array: Any) -> Any:
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device)

    def update_policy(self, rollout: SILRollout) -> Dict[str, float]:
        """Standard PPO clipped update on the freshly collected on-policy batch."""
        if not self.config.use_onpolicy_ppo or len(rollout) == 0:
            return {}
        obs = self._to_tensor(rollout.observations)
        actions = self._to_tensor(rollout.actions)
        old_log_probs = self._to_tensor(rollout.log_probs)
        returns = self._to_tensor(rollout.returns)
        advantages = self._to_tensor(rollout.advantages)
        if self.config.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        batch_size = max(1, min(int(self.config.batch_size), n))
        idx = np.arange(n)
        stats: Dict[str, float] = {}
        clip_range = float(self.config.clip_range)
        for _epoch in range(int(self.config.n_epochs)):
            self.rng.shuffle(idx)
            for start in range(0, n, batch_size):
                mb = idx[start : start + batch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long)
                log_prob, entropy, values = _policy_log_prob_values(self.policy, obs[mb_t], actions[mb_t])
                ratio = torch.exp(log_prob - old_log_probs[mb_t])
                adv = advantages[mb_t]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[mb_t])
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + float(self.config.vf_coef) * value_loss
                    - float(self.config.ent_coef) * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    _policy_parameters(self.policy), float(self.config.max_grad_norm)
                )
                self.optimizer.step()
                stats = {
                    "ppo/policy_loss": float(policy_loss.item()),
                    "ppo/value_loss": float(value_loss.item()),
                    "ppo/entropy": float(entropy_loss.item()),
                    "ppo/approx_kl": float(
                        ((ratio - 1.0) - (log_prob - old_log_probs[mb_t])).mean().item()
                    ),
                    "ppo/clip_fraction": float(
                        (torch.abs(ratio - 1.0) > clip_range).float().mean().item()
                    ),
                }
        return stats

    def update_self_imitation(self) -> Dict[str, float]:
        """SIL off-policy updates drawn from the self-imitation buffer."""
        if len(self.buffer) == 0 or not self.buffer.ready:
            return {"sil/skipped": 1.0, "sil/buffer_size": float(len(self.buffer))}
        stats: Dict[str, float] = {}
        for _ in range(max(1, int(self.config.sil_update_epochs))):
            batch = self.buffer.sample(int(self.config.sil_batch_size), rng=self.rng)
            obs = self._to_tensor(batch["observations"])
            actions = self._to_tensor(batch["actions"])
            returns = self._to_tensor(batch["returns"])
            advantages = self._to_tensor(batch["advantages"])
            log_prob, _entropy, values = _policy_log_prob_values(self.policy, obs, actions)
            terms = self_imitation_loss(
                log_prob,
                values,
                returns,
                advantages=advantages,
                policy_coef=float(self.config.sil_coef),
                value_coef=float(self.config.sil_vf_coef),
                positive_only=bool(self.config.use_positive_only),
            )
            self.optimizer.zero_grad()
            terms["loss"].backward()
            nn.utils.clip_grad_norm_(_policy_parameters(self.policy), float(self.config.max_grad_norm))
            self.optimizer.step()
            stats = {
                "sil/policy_loss": float(terms["policy_loss"].item()),
                "sil/value_loss": float(terms["value_loss"].item()),
                "sil/loss": float(terms["loss"].item()),
                "sil/buffer_size": float(len(self.buffer)),
                "sil/mean_advantage": float(advantages.mean().item()),
            }
        return stats

    def update(self, rollout: Optional[SILRollout] = None) -> Dict[str, float]:
        """One training update: PPO (on-policy) + SIL (off-policy from buffer)."""
        stats: Dict[str, float] = {}
        if rollout is not None:
            stats.update(self.update_policy(rollout))
        stats.update(self.update_self_imitation())
        return stats

    # -- training loop ---------------------------------------------------- #
    def train(
        self,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        logger: Any = None,
        progress: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the SIL training loop and return the per-iteration history."""
        logger = logger or self.logger
        total_timesteps = int(total_timesteps or self.config.total_timesteps)
        if total_iterations is not None:
            total_iterations = int(total_iterations)
        else:
            total_iterations = max(1, int(math.ceil(total_timesteps / max(1, self.n_steps))))
        total_timesteps = total_iterations * self.n_steps

        started_steps = self.total_steps
        for iteration in range(total_iterations):
            self.iteration = iteration
            self.timer_start("collect")
            rollout = self.collect_rollout()
            collect_time = self.timer_end("collect", accumulate=False)
            self.timer_start("update")
            stats = self.update(rollout)
            update_time = self.timer_end("update", accumulate=False)
            record: Dict[str, Any] = {
                "iteration": iteration,
                "total_timesteps": self.total_steps,
                "env/reward": _mean(rollout.episode_task_returns),
                "env/mean_step_reward": _mean(rollout.task_rewards),
                "env/n_episodes": len(rollout.episode_task_returns),
                "sil/added": float(rollout.sil_added),
                "time/collect": collect_time,
                "time/update": update_time,
            }
            record.update(stats)
            record.update(self.buffer.statistics())
            self.history.append(record)
            if progress or (
                self.config.log_interval and iteration % max(1, int(self.config.log_interval)) == 0
            ):
                logger.info(
                    "[SIL] iter %d | steps %d/%d | reward %s | buffer %d | sil_added %d",
                    iteration,
                    self.total_steps,
                    started_steps + total_timesteps,
                    record.get("env/reward"),
                    len(self.buffer),
                    rollout.sil_added,
                )
            if callback is not None:
                try:
                    callback(record)
                except Exception:
                    pass
        return self.history

    fit = train

    # -- evaluation ------------------------------------------------------- #
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        policy: Any = None,
        max_steps: Optional[int] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate a policy from the default initial state distribution ``rho``."""
        policy = policy or self.policy
        max_steps = int(max_steps or _resolve_horizon(self.env))
        returns, lengths = [], []
        for _ in range(int(n_episodes)):
            kwargs = dict(reset_kwargs or {})
            obs, _info = unpack_reset(self.env.reset(**kwargs) if kwargs else self.env.reset())
            ep_return, steps = 0.0, 0
            for _step in range(max_steps):
                action = policy_action(policy, obs, deterministic=deterministic)
                env_action = prepare_action_for_env(action, self.action_space, self.discrete)
                obs, reward, terminated, truncated, _info = unpack_step(self.env.step(env_action))
                ep_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(ep_return)
            lengths.append(float(steps))
        result = {
            "return_mean": _mean(returns),
            "return_std": float(np.std(returns)) if returns else float("nan"),
            "episode_length": _mean(lengths),
            "n_episodes": float(len(returns)),
        }
        self.eval_history.append(result)
        return result

    # -- bookkeeping ------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else {}
        return {
            "name": self.name,
            "env_id": self.env_id,
            "total_timesteps": int(self.total_steps),
            "iterations": len(self.history),
            "buffer_size": len(self.buffer),
            "sil_coef": float(self.config.sil_coef),
            "timesteps": int(self.total_steps),
            "final_reward": last.get("env/reward", float("nan")),
            "total_time": self.total_time,
            "config": self.config.to_dict(),
        }

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        payload = {
            "name": self.name,
            "env_id": self.env_id,
            "config": self.config.to_dict(),
            "history": self.history,
            "eval_history": self.eval_history,
            "summary": self.summary(),
            "state_dict": getattr(self.policy, "state_dict", lambda: {})(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def save_policy(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        if not _HAS_POLICIES:
            return self.save(path, extra=extra)
        return save_policy(
            self.policy,
            path,
            env_id=self.env_id,
            kind="policy",
            extra={"baseline": self.name, **(extra or {})},
        )


class SILRefiner(SILTrainer):
    """SIL as a *refining* baseline (Table 5: RICE vs SIL on four MuJoCo games).

    Optionally also exposes :meth:`refine`, which delegates to RICE's Stage-2
    PPO refinement so that SIL-approximated policies can be plugged into the
    RICE pipeline (mirrors :class:`rice.baselines.gail.GAILRefiner`).
    """

    name = "SIL"

    def approximate_policy(
        self, total_timesteps: Optional[int] = None, progress: bool = False, **kwargs: Any
    ) -> Any:
        self.train(total_timesteps=total_timesteps, progress=progress, **kwargs)
        return self.policy

    @property
    def policy_network(self) -> Any:
        return self.policy

    @property
    def approximated_policy(self) -> Any:
        return self.policy

    def refine(
        self,
        env: Any = None,
        mask_net: Any = None,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        config: Optional[Any] = None,
        logger: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Refine the (SIL-trained) policy with RICE's Stage-2 engine."""
        if not _HAS_REFINER:
            raise ImportError("rice.refining.ppo_refine is required for SILRefiner.refine")
        env = env if env is not None else self.env
        return refine_policy(
            env,
            policy=self.policy,
            mask_net=mask_net,
            total_timesteps=total_timesteps,
            total_iterations=total_iterations,
            env_id=self.env_id,
            config=config,
            logger=logger or self.logger,
            seed=seed if seed is not None else self.seed,
            device=self.device,
            **kwargs,
        )

    def run(
        self,
        env: Any = None,
        sil_timesteps: Optional[int] = None,
        progress: bool = False,
        evaluate: bool = True,
        eval_episodes: int = 10,
        save_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run SIL from the pre-trained policy and (optionally) evaluate it."""
        self.train(total_timesteps=sil_timesteps, progress=progress, **kwargs)
        result: Dict[str, Any] = {"baseline": self.name, "summary": self.summary()}
        if evaluate:
            metrics = self.evaluate(
                n_episodes=eval_episodes, deterministic=self.config.deterministic_eval
            )
            result["eval"] = metrics
            result["final_reward"] = metrics.get("return_mean")
        if save_dir:
            ensure_dir(save_dir)
            self.save(os.path.join(save_dir, f"sil_{self.env_id}.pt"))
        return result


# --------------------------------------------------------------------------- #
# Functional API
# --------------------------------------------------------------------------- #
def collect_successful_experiences(
    env: Any,
    policy: Any,
    n_timesteps: int = 10_000,
    gamma: float = DEFAULT_GAMMA,
    positive_only: bool = True,
    device: str = "cpu",
    seed: Optional[int] = None,
    discrete: Optional[bool] = None,
    action_space: Any = None,
    buffer: Optional[SelfImitationBuffer] = None,
    progress: bool = False,
) -> SelfImitationBuffer:
    """Warm-start a :class:`SelfImitationBuffer` by rolling ``policy``.

    Seeds SIL's buffer with the pre-trained agent's successful past experience
    (the premise of Oh et al., 2018) before the first update.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for collect_successful_experiences")
    if seed is not None:
        set_seed(seed)
    action_space = action_space if action_space is not None else getattr(env, "action_space", None)
    buffer = buffer or SelfImitationBuffer(positive_only=positive_only, seed=seed)
    obs, _info = unpack_reset(env.reset())
    observations: List[Any] = []
    actions: List[Any] = []
    rewards: List[float] = []
    log_probs: List[float] = []
    collected = 0
    while collected < int(n_timesteps):
        with torch.no_grad():
            obs_tensor = _policy_observation(policy, obs)
            distribution, _values = policy_distribution_and_values(policy, obs_tensor)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
            if log_prob.ndim > 1:
                log_prob = log_prob.sum(dim=-1)
        action_np = action_tensor.detach().cpu().numpy().reshape(-1)
        env_action = prepare_action_for_env(action_np, action_space, discrete)
        next_obs, reward, terminated, truncated, _info = unpack_step(env.step(env_action))
        observations.append(_flat(obs))
        actions.append(np.asarray(action_np, dtype=np.float32).reshape(-1))
        rewards.append(float(reward))
        log_probs.append(float(log_prob.reshape(-1)[0].item()))
        collected += 1
        done = bool(terminated or truncated)
        if done:
            buffer.add_trajectory(observations, actions, rewards, gamma=gamma, log_probs=log_probs)
            observations, actions, rewards, log_probs = [], [], [], []
            obs, _info = unpack_reset(env.reset())
        else:
            obs = next_obs
    if observations:
        buffer.add_trajectory(observations, actions, rewards, gamma=gamma, log_probs=log_probs)
    return buffer


def sil_update_from_buffer(
    policy: Any,
    optimizer: Any,
    buffer: SelfImitationBuffer,
    batch_size: int = DEFAULT_SIL_BATCH_SIZE,
    epochs: int = DEFAULT_SIL_EPOCHS,
    coef: float = DEFAULT_SIL_COEF,
    value_coef: float = 0.0,
    positive_only: bool = True,
    device: str = "cpu",
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    rng: Any = None,
) -> Dict[str, float]:
    """Standalone SIL off-policy update loop (used by SILTrainer and tests)."""
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for sil_update_from_buffer")
    if len(buffer) == 0 or not buffer.ready:
        return {"sil/skipped": 1.0, "sil/buffer_size": float(len(buffer))}
    stats: Dict[str, float] = {}
    for _ in range(max(1, int(epochs))):
        batch = buffer.sample(int(batch_size), rng=rng)
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        act = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
        advs = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
        log_prob, _entropy, values = _policy_log_prob_values(policy, obs, act)
        terms = self_imitation_loss(
            log_prob,
            values,
            returns,
            advantages=advs,
            policy_coef=float(coef),
            value_coef=float(value_coef),
            positive_only=bool(positive_only),
        )
        optimizer.zero_grad()
        terms["loss"].backward()
        nn.utils.clip_grad_norm_(_policy_parameters(policy), float(max_grad_norm))
        optimizer.step()
        stats = {
            "sil/policy_loss": float(terms["policy_loss"].item()),
            "sil/value_loss": float(terms["value_loss"].item()),
            "sil/loss": float(terms["loss"].item()),
            "sil/buffer_size": float(len(buffer)),
        }
    return stats


def train_sil(
    env: Any,
    policy: Any = None,
    total_timesteps: Optional[int] = None,
    total_iterations: Optional[int] = None,
    env_id: str = "default",
    config: Optional[Any] = None,
    logger: Any = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    warmstart_timesteps: int = 0,
    progress: bool = False,
    evaluate: bool = False,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Tuple[Any, SILTrainer]:
    """High-level SIL training entry point; returns ``(refined_policy, trainer)``."""
    if seed is not None:
        set_seed(seed)
    trainer = SILTrainer(
        env, policy=policy, config=config, env_id=env_id, device=device, logger=logger, seed=seed, **kwargs
    )
    if warmstart_timesteps and int(warmstart_timesteps) > 0:
        collect_successful_experiences(
            env,
            trainer.policy,
            n_timesteps=int(warmstart_timesteps),
            gamma=float(trainer.config.gamma),
            positive_only=bool(trainer.config.use_positive_only),
            device=device,
            seed=seed,
            discrete=trainer.discrete,
            action_space=trainer.action_space,
            buffer=trainer.buffer,
            progress=progress,
        )
    trainer.train(
        total_timesteps=total_timesteps,
        total_iterations=total_iterations,
        logger=logger,
        progress=progress,
    )
    if evaluate:
        trainer.evaluate(n_episodes=eval_episodes, deterministic=trainer.config.deterministic_eval)
    if save_path:
        trainer.save(save_path)
    return trainer.policy, trainer


def approximate_policy_with_sil(env: Any, policy: Any = None, **kwargs: Any) -> Any:
    """Return only the SIL-refined policy (used by Exp IV-style pipelines)."""
    refined, _trainer = train_sil(env, policy=policy, **kwargs)
    return refined


def make_sil(
    env: Any = None,
    policy: Any = None,
    env_id: str = "default",
    config: Optional[Any] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    **kwargs: Any,
) -> SILRefiner:
    """Factory for registry-based dispatch."""
    return SILRefiner(
        env, policy=policy, config=config, env_id=env_id, seed=seed, device=device, logger=logger, **kwargs
    )


build_sil = make_sil
make_sil_refiner = make_sil


def sil_for(env_id: str = "default", **kwargs: Any) -> SILConfig:
    """Return a config pre-configured for the given application."""
    return SILConfig.from_dict({"env_id": env_id}, **kwargs)


def sil(
    env: Any,
    policy: Any = None,
    env_id: str = "default",
    config: Optional[Any] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress: bool = False,
    evaluate: bool = True,
    eval_episodes: int = 10,
    save_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plain-dict SIL pipeline helper (Table 5 comparison)."""
    known = set(getattr(SILConfig, "__dataclass_fields__", {}).keys())
    if isinstance(config, dict) and "env_id" in config:
        env_id = config.get("env_id", env_id)
        config = {k: v for k, v in config.items() if k != "env_id"}
    refiner = SILRefiner(
        env,
        policy=policy,
        config=config,
        env_id=env_id,
        seed=seed,
        device=device,
        **{k: v for k, v in kwargs.items() if k in known},
    )
    result = refiner.run(
        env=env,
        progress=progress,
        evaluate=evaluate,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        **{k: v for k, v in kwargs.items() if k not in known},
    )
    return result


sil_baseline = sil
run_sil = sil


def describe_sil(trainer: Any = None) -> str:
    """One-line human-readable summary for logging/tables."""
    if trainer is None:
        return (
            "SIL (Self-Imitation Learning, Oh et al. 2018): continues PPO training from the "
            "pre-trained policy while prioritising past successful experience in a replay buffer."
        )
    cfg = getattr(trainer, "config", None)
    env_id = getattr(trainer, "env_id", "default")
    coef = getattr(cfg, "sil_coef", DEFAULT_SIL_COEF) if cfg is not None else DEFAULT_SIL_COEF
    ts = (
        getattr(cfg, "total_timesteps", DEFAULT_TOTAL_TIMESTEPS)
        if cfg is not None
        else DEFAULT_TOTAL_TIMESTEPS
    )
    buf = len(getattr(trainer, "buffer", [])) if hasattr(trainer, "buffer") else 0
    reward_mode = (
        getattr(cfg, "reward_mode", DEFAULT_REWARD_MODE) if cfg is not None else DEFAULT_REWARD_MODE
    )
    return (
        f"SIL(env={env_id}, sil_coef={coef}, total_timesteps={ts}, buffer={buf}, "
        f"reward_mode={reward_mode})"
    )
