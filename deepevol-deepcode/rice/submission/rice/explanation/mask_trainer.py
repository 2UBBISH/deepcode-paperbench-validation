"""Stage-1 mask-network trainer for RICE -- Algorithm 1 of the paper.

Paper (Cheng et al., ICML 2024), Sec. 3.3 "Step-level Explanation":

    The mask net takes the state ``s_t`` and outputs a binary action
    ``a_t^m in {0, 1}``.  The executed action is

        a_t (x) a_t^m = a_t            if a_t^m = 0      (Eq. 1)
                        a_random       if a_t^m = 1

    StateMask trains the mask by minimising ``J(theta) = min |eta(pi) - eta(pi_bar)|``
    with a primal-dual method.  Theorem 3.3 shows ``eta(pi_bar) <= eta(pi)`` under
    Assumption 3.1, so RICE instead maximises ``J(theta) = max eta(pi_bar)`` and
    "can utilize the vanilla PPO algorithm to train the state mask without
    sacrificing the theoretical guarantee".

    To avoid the trivial solution "never blind", an extra bonus is added when the
    mask outputs "1":

        R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m

Algorithm 1 (verbatim structure implemented here)::

    Input: target agent's policy pi
    Output: mask network pi_tilde_theta
    theta_old <- theta
    for iteration = 1, 2, ... do
        Set the initial state s_0 ~ rho
        D <- empty
        for t = 0 to T do
            Sample a_t        ~ pi(a_t | s_t)
            Sample a_t^m      ~ pi_tilde_{theta_old}(a_t^m | s_t)
            Compute the actual taken action a <- a_t (x) a_t^m
            (s_{t+1}, R'_t) <- env.step(a) and record (s_t, s_{t+1}, a_t^m, R'_t) in D
        end for
        update theta_old <- theta using D by PPO algorithm
    end for

This module provides:

* :class:`MaskPPOConfig`   -- hyper-parameters (SB3 defaults for anything the paper
  does not specify: gamma=0.99, clip=0.2, lr=3e-4, n_epochs=10, GAE lambda=0.95).
* :class:`MaskEnv`         -- couples the *frozen* target policy ``pi`` with the real
  environment; exposes the binary mask as its action space and returns the
  augmented reward ``R + alpha * a_t^m``.
* :class:`MaskTrainer`     -- Algorithm 1 with a self-contained PPO implementation
  (native PyTorch, no SB3 weight copying), plus wall-clock timers so that the
  paper's "fixed sample budget" training-time comparison (Table 4) can be made.
* :func:`train_mask_network` -- convenience end-to-end helper.

The reformulation ``max eta(pi_bar)`` is what yields RICE's ~16.8% mask-training
time gain over StateMask (no primal-dual loop, no dual variable updates).
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - environment dependent
    import gym  # type: ignore

    _HAS_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    _HAS_GYM = False


# --------------------------------------------------------------------------------------
# rice imports
# --------------------------------------------------------------------------------------
from rice.models import normalize_env_key, sample_random_action  # noqa: E402
from rice.models import sb3_policy_kwargs  # noqa: E402  (kept for API parity)
from rice.explanation.mask_network import (  # noqa: E402
    BLIND_INDEX,
    KEEP_INDEX,
    MASK_BLIND,
    MASK_KEEP,
    NUM_MASK_ACTIONS,
    MaskCritic,
    MaskNetwork,
    augmented_reward,
    blinding_bonus,
    build_mask_critic,
    build_mask_network,
    flatten_observation,
    masked_action,
    save_mask_network,
    load_mask_network,
    state_importance,
)
from rice.utils.io import ensure_dir  # noqa: E402
from rice.utils.logging import get_logger  # noqa: E402
from rice.utils.seeding import get_rng, seed_env, set_seed  # noqa: E402

__all__ = [
    "MaskPPOConfig",
    "MaskEnv",
    "RolloutBatch",
    "MaskTrainer",
    "train_mask_network",
    "compute_gae",
    "make_mask_env",
    "DEFAULT_ALPHA",
]


# Paper Table 3 lists alpha = 0.0001 (Sec. 4.3 text uses 0.01 -- alpha is reported to
# be insensitive to the fidelity score; the Table 3 value is the default and the
# hyper-parameter is exposed for the Experiment-V sweep).
DEFAULT_ALPHA = 1e-4

# Number of logits of the mask network ("keep" / "blind").
MASK_ACTION_DIM = 2


# --------------------------------------------------------------------------------------
# Fallback Discrete space (tiny, dependency free)
# --------------------------------------------------------------------------------------
class _DiscreteSpace:
    """Minimal ``Discrete`` stand-in used when gym is unavailable."""

    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(self.n))

    def contains(self, x: Any) -> bool:
        try:
            xi = int(x)
        except Exception:
            return False
        return 0 <= xi < self.n

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Discrete({})".format(self.n)

    def __eq__(self, other: Any) -> bool:
        return getattr(other, "n", None) == self.n


def _make_discrete(n: int = MASK_ACTION_DIM):
    if _HAS_GYM:
        try:
            return gym.spaces.Discrete(n)
        except Exception:  # pragma: no cover
            pass
    return _DiscreteSpace(n)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class MaskPPOConfig:
    """Hyper-parameters for Algorithm 1.

    Unspecified-in-paper values default to Stable-Baselines3's PPO defaults
    (gamma=0.99, GAE lambda=0.95, clip=0.2, lr=3e-4, n_epochs=10), as required by
    the reproduction plan.
    """

    alpha: float = DEFAULT_ALPHA           # blinding bonus  R' = R + alpha * a_t^m
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    n_epochs: int = 10
    batch_size: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: Optional[float] = None
    n_steps: Optional[int] = None          # trajectory length T (defaults to env horizon)
    total_timesteps: int = 100_000         # sample budget for a full mask training run
    device: str = "cpu"
    seed: Optional[int] = None
    deterministic_policy: bool = False     # sample a_t ~ pi (algorithm samples stochastic a_t)
    reward_mode: str = "augmented"         # "augmented" (R + alpha*a^m) or "task"
    use_blinding_bonus: bool = True        # Algorithm-1 reformulation vs. plain |eta diff|
    log_interval: int = 1

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None) -> "MaskPPOConfig":
        cfg = dict(cfg or {})
        # allow nested {"mask": {...}} / {"explanation": {...}} configs
        for key in ("mask", "explanation", "mask_trainer"):
            sub = cfg.pop(key, None)
            if isinstance(sub, dict):
                cfg.update(sub)
        # common per-app keys
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in cfg.items() if k in known and v is not None}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Mask environment: frozen target policy + binary mask action
# --------------------------------------------------------------------------------------
_MaskEnvBase = gym.Env if (_HAS_GYM and hasattr(gym, "Env")) else object


class MaskEnv(_MaskEnvBase):  # type: ignore[misc, valid-type]
    """Environment whose action is the binary mask ``a_t^m``.

    ``observation_space`` == the real environment's observation space.
    ``action_space``      == ``Discrete(2)`` (0 = keep the target action, 1 = blind).

    Inside :meth:`step` the target action ``a_t ~ pi(.|s_t)`` is sampled from the frozen
    policy, the executed action is ``a_t (x) a_t^m`` (Eq. 1), and the reward returned to
    the mask learner is ``R + alpha * a_t^m`` (Algorithm 1's ``R'_t``).
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        env: Any,
        policy: Any,
        alpha: float = DEFAULT_ALPHA,
        deterministic_policy: bool = False,
        reward_mode: str = "augmented",
        use_blinding_bonus: bool = True,
        action_space: Any = None,
        discrete: Optional[bool] = None,
        env_id: str = "default",
        seed: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        try:  # gym.Env.__init__ in some versions requires nothing; be tolerant.
            _MaskEnvBase.__init__(self)
        except Exception:  # pragma: no cover
            pass

        self.env = env
        self.policy = policy
        self.alpha = float(alpha)
        self.deterministic_policy = bool(deterministic_policy)
        self.reward_mode = str(reward_mode)
        self.use_blinding_bonus = bool(use_blinding_bonus)
        self.env_id = normalize_env_key(env_id) if env_id else "default"
        self.rng = rng if rng is not None else get_rng(seed)

        self.target_action_space = (
            action_space if action_space is not None else getattr(env, "action_space", None)
        )
        if discrete is None:
            discrete = bool(getattr(self.target_action_space, "n", None) is not None) or (
                "Discrete" in type(self.target_action_space).__name__
            )
        self.discrete = bool(discrete)

        self.observation_space = getattr(env, "observation_space", None)
        self.mask_action_space = _make_discrete(MASK_ACTION_DIM)
        self.action_space = self.mask_action_space
        self.reward_range = (-np.inf, np.inf)

        # internal counters / diagnostics
        self._step_arity = 4
        self.step_count = 0
        self.episode_count = 0
        self.blind_count = 0
        self.last_target_action = None
        self.last_executed_action = None
        self.episode_task_return = 0.0
        self.episode_augmented_return = 0.0
        self.episode_blind_steps = 0
        self.history: List[Dict[str, float]] = []

        if seed is not None:
            self.seed(seed)

    # ------------------------------------------------------------------ spaces
    @property
    def target_action_dim(self) -> Optional[int]:
        sp = self.target_action_space
        if sp is None:
            return None
        n = getattr(sp, "n", None)
        if n is not None:
            return 1
        shape = getattr(sp, "shape", None)
        if shape is None:
            return None
        return int(np.prod(shape))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        if isinstance(result, tuple) and len(result) >= 1:
            info = result[-1] if isinstance(result[-1], dict) else {}
            return result[0], info
        return result, {}

    @staticmethod
    def _unpack_step(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Normalise 4-tuple / 5-tuple gym step results to (obs, r, term, trunc, info)."""
        if not isinstance(result, tuple):  # pragma: no cover - defensive
            raise TypeError("env.step must return a tuple, got {}".format(type(result)))
        if len(result) == 5:
            obs, r, terminated, truncated, info = result
            return obs, float(r), bool(terminated), bool(truncated), dict(info or {})
        obs, r, done, info = result
        return obs, float(r), bool(done), False, dict(info or {})

    def _policy_action(self, obs: Any) -> np.ndarray:
        pol = self.policy
        action = None
        if hasattr(pol, "predict"):
            out = pol.predict(obs, deterministic=self.deterministic_policy)
            action = out[0] if isinstance(out, tuple) else out
        elif hasattr(pol, "act"):
            out = pol.act(obs, deterministic=self.deterministic_policy)
            action = out[0] if isinstance(out, tuple) else out
        elif callable(pol):
            out = pol(obs)
            action = out[0] if isinstance(out, tuple) else out
        else:  # pragma: no cover - defensive
            raise TypeError(
                "target policy must expose .predict(), .act() or be callable; got {}".format(type(pol))
            )
        if _HAS_TORCH and isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        if self.discrete:
            return np.asarray(int(np.argmax(action)) if action.size > 1 else int(action.reshape(-1)[0]))
        return action.astype(np.float32)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        sp = self.target_action_space
        low = getattr(sp, "low", None)
        high = getattr(sp, "high", None)
        if self.discrete or low is None or high is None:
            return action
        try:
            return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
        except Exception:  # pragma: no cover - defensive
            return action

    def _random_action(self) -> np.ndarray:
        a = sample_random_action(
            action_space=self.target_action_space,
            rng=self.rng,
            discrete=self.discrete,
        )
        return np.asarray(a)

    # ------------------------------------------------------------------ gym API
    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = int(self.rng.randint(0, 2 ** 31 - 1))
        seed = int(seed)
        seed_env(self.env, seed)
        self.rng = get_rng(seed)
        if _HAS_GYM:
            try:
                self.action_space.seed(seed)
            except Exception:
                pass
        return [seed]

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        obs, info = self._unpack_reset(result)
        self._finalize_episode()
        return result

    def _finalize_episode(self) -> None:
        if self.step_count > 0:
            self.history.append(
                {
                    "episode": self.episode_count,
                    "task_return": float(self.episode_task_return),
                    "augmented_return": float(self.episode_augmented_return),
                    "blind_steps": int(self.episode_blind_steps),
                    "length": int(self.step_count),
                }
            )
        self.step_count = 0
        self.episode_count += 1
        self.episode_task_return = 0.0
        self.episode_augmented_return = 0.0
        self.episode_blind_steps = 0

    def step(self, mask_action: Any):
        """Execute one masked step (Algorithm 1, inner loop)."""
        mask = int(np.asarray(mask_action).reshape(-1)[0])
        mask = MASK_BLIND if mask == MASK_BLIND else MASK_KEEP

        # current observation: cached from the last reset/step
        obs = self._current_obs
        target_action = self._policy_action(obs)
        executed = masked_action(
            target_action,
            mask,
            action_space=self.target_action_space,
            rng=self.rng,
            discrete=self.discrete,
        )
        executed = np.asarray(self._clip_action(np.asarray(executed)))

        result = self.env.step(executed)
        next_obs, task_reward, terminated, truncated, info = self._unpack_step(result)

        bonus = self.alpha * float(mask) if self.use_blinding_bonus else 0.0
        augmented = float(task_reward) + bonus
        reward = augmented if self.reward_mode == "augmented" else float(task_reward)

        info = dict(info or {})
        info["mask"] = mask
        info["target_action"] = target_action
        info["executed_action"] = executed
        info["task_reward"] = float(task_reward)
        info["blinding_bonus"] = float(bonus)
        info["augmented_reward"] = float(augmented)

        self.step_count += 1
        self.blind_count += int(mask == MASK_BLIND)
        self.episode_task_return += float(task_reward)
        self.episode_augmented_return += float(augmented)
        self.episode_blind_steps += int(mask == MASK_BLIND)
        self.last_target_action = target_action
        self.last_executed_action = executed
        self._current_obs = next_obs

        done = bool(terminated or truncated)
        if len(result) == 5:
            return next_obs, reward, bool(terminated), bool(truncated), info
        return next_obs, reward, done, info

    def render(self, *args, **kwargs):  # pragma: no cover - passthrough
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    # ------------------------------------------------------------------ extras
    def set_policy(self, policy: Any) -> None:
        self.policy = policy

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    @property
    def blind_fraction(self) -> float:
        total = self.step_count or 1
        return float(self.episode_blind_steps) / float(total)

    @property
    def stats(self) -> Dict[str, float]:
        if not self.history:
            return {"episodes": 0.0}
        tasks = [h["task_return"] for h in self.history]
        augs = [h["augmented_return"] for h in self.history]
        blind = [h["blind_steps"] / max(1, h["length"]) for h in self.history]
        return {
            "episodes": float(len(self.history)),
            "task_return_mean": float(np.mean(tasks)),
            "augmented_return_mean": float(np.mean(augs)),
            "blind_fraction_mean": float(np.mean(blind)),
            "blind_fraction_std": float(np.std(blind)),
        }

    def __getattr__(self, item: str) -> Any:
        # delegate unknown attributes to the wrapped environment
        if item.startswith("__") or item in {"env", "policy"}:
            raise AttributeError(item)
        return getattr(self.__dict__["env"], item)


# --------------------------------------------------------------------------------------
# Rollout container
# --------------------------------------------------------------------------------------
@dataclass
class RolloutBatch:
    """Dataset ``D`` of Algorithm 1: ``(s_t, s_{t+1}, a_t^m, R'_t)`` (+ PPO bookkeeping)."""

    observations: Any
    next_observations: Any
    masks: Any
    rewards: Any                 # augmented reward R' recorded in D
    task_rewards: Any
    bonuses: Any
    log_probs: Any
    values: Any
    dones: Any
    truncated: Any
    advantages: Any = None
    returns: Any = None
    episode_task_returns: List[float] = field(default_factory=list)
    episode_augmented_returns: List[float] = field(default_factory=list)
    dataset: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return int(len(self.masks))

    @property
    def blind_fraction(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(np.mean(np.asarray(self.masks, dtype=np.float64) > 0.5))

    def as_dataset_tuples(self) -> List[Tuple[Any, Any, int, float]]:
        """``[(s_t, s_{t+1}, a_t^m, R'_t), ...]`` exactly as recorded in D."""
        return [
            (self.observations[i], self.next_observations[i], int(self.masks[i]), float(self.rewards[i]))
            for i in range(len(self))
        ]


# --------------------------------------------------------------------------------------
# Torch helpers
# --------------------------------------------------------------------------------------
def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover - environment dependent
        raise ImportError(
            "MaskTrainer (Algorithm 1) requires PyTorch. Install torch to train the mask network."
        )


def _as_1d_float(values: Any) -> Any:
    if values is None:
        return None
    if _HAS_TORCH and isinstance(values, torch.Tensor):
        values = values.reshape(-1)
        return values.float()
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return torch.as_tensor(arr) if _HAS_TORCH else arr


def _mask_logits(net: Any, observations: Any):
    """Return raw 2-class logits from a MaskNetwork / ActorCritic-like mask module."""
    out = None
    logits_fn = getattr(net, "logits", None)
    if callable(logits_fn):
        try:
            out = logits_fn(observations)
        except Exception:
            out = None
    if out is None:
        out = net(observations)
        if isinstance(out, tuple):
            out = out[0]
    if isinstance(out, dict):
        out = out.get("logits", out.get("action_logits"))
    if _HAS_TORCH and isinstance(out, torch.Tensor) and out.dim() == 1:
        out = out.reshape(1, -1)
    return out


def _critic_values(critic: Any, observations: Any):
    """Return a 1-D tensor of state values from a MaskCritic-like module."""
    out = None
    pv = getattr(critic, "predict_values", None)
    if callable(pv):
        try:
            out = pv(observations)
        except Exception:
            out = None
    if out is None:
        out = critic(observations)
        if isinstance(out, tuple):
            out = out[-1] if isinstance(out[-1], (torch.Tensor, np.ndarray)) else out[0]
    if isinstance(out, dict):
        out = out.get("values", out.get("value"))
    if _HAS_TORCH and isinstance(out, torch.Tensor):
        out = out.reshape(-1)
    return out


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[float],
    last_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation (SB3 defaults gamma=0.99, lambda=0.95)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    dones = np.asarray(dones, dtype=np.float64).reshape(-1)
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# --------------------------------------------------------------------------------------
# Trainer (Algorithm 1)
# --------------------------------------------------------------------------------------
class MaskTrainer:
    """Vanilla-PPO trainer for the mask network -- Algorithm 1.

    Parameters
    ----------
    env : gym-like env
        The *real* environment (already wrapped with the RICE wrappers).
    target_policy : any
        The frozen pre-trained policy ``pi`` (SB3 model/policy, a native
        :class:`rice.models.ActorCritic`, or a callable).  It is never modified.
    mask_net : :class:`rice.explanation.MaskNetwork`, optional
        Pre-built mask network; otherwise built from the per-env architecture.
    config : :class:`MaskPPOConfig` or dict, optional
        Hyper-parameters (``alpha`` in particular).
    """

    def __init__(
        self,
        env: Any,
        target_policy: Any,
        mask_net: Optional[Any] = None,
        critic: Optional[Any] = None,
        config: Optional[Union[MaskPPOConfig, Dict[str, Any]]] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        observation_space: Any = None,
        action_space: Any = None,
        discrete: Optional[bool] = None,
        seed: Optional[int] = None,
        store_dataset: bool = True,
        **kwargs: Any,
    ):
        _require_torch()

        if isinstance(config, dict):
            cfg = MaskPPOConfig.from_dict(config)
        elif config is None:
            cfg = MaskPPOConfig.from_dict(kwargs)
        else:
            cfg = copy.deepcopy(config)
        # allow explicit kwargs to override individual fields
        for key, val in kwargs.items():
            if hasattr(cfg, key) and val is not None:
                setattr(cfg, key, val)
        self.config = cfg

        self.env_id = normalize_env_key(env_id) if env_id else "default"
        self.device = torch.device(cfg.device if device is None else device)
        self.logger = logger
        self.seed = cfg.seed if seed is None else seed
        if self.seed is not None:
            set_seed(int(self.seed))

        self.env = env
        self.target_policy = target_policy
        self.observation_space = observation_space or getattr(env, "observation_space", None)
        self.action_space = action_space or getattr(env, "action_space", None)

        self.obs_dim = self._infer_obs_dim()
        if discrete is None:
            discrete = bool(getattr(self.action_space, "n", None) is not None) or (
                "Discrete" in type(self.action_space).__name__
            )
        self.discrete = bool(discrete)
        self.action_dim = self._infer_action_dim()

        # ---- mask network + critic ------------------------------------------------
        self.mask_net = mask_net if mask_net is not None else build_mask_network(
            env_id=self.env_id,
            obs_dim=self.obs_dim,
            action_space=self.action_space,
            observation_space=self.observation_space,
            device=str(self.device),
        )
        self.mask_net.to(self.device)
        self.critic = critic if critic is not None else build_mask_critic(
            env_id=self.env_id,
            obs_dim=self.obs_dim,
            observation_space=self.observation_space,
            device=str(self.device),
        )
        self.critic.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.mask_net.parameters()) + list(self.critic.parameters()),
            lr=float(cfg.lr),
            eps=1e-5,
        )

        # ---- environment that turns the mask into the action ----------------------
        self.mask_env = MaskEnv(
            env=env,
            policy=target_policy,
            alpha=cfg.alpha,
            deterministic_policy=cfg.deterministic_policy,
            reward_mode=cfg.reward_mode,
            use_blinding_bonus=cfg.use_blinding_bonus,
            action_space=self.action_space,
            discrete=self.discrete,
            env_id=self.env_id,
            seed=self.seed,
            rng=get_rng(self.seed),
        )

        # ---- trajectory length T ---------------------------------------------------
        self.n_steps = int(cfg.n_steps) if cfg.n_steps else self._infer_horizon()

        # ---- bookkeeping -----------------------------------------------------------
        self.rng = get_rng(self.seed)
        self.history: List[Dict[str, float]] = []
        self.timings: Dict[str, float] = {"rollout": 0.0, "update": 0.0, "total": 0.0}
        self.timer_history: List[Dict[str, float]] = []
        self.total_samples = 0
        self.num_updates = 0
        self.iterations_done = 0
        self.last_batch: Optional[RolloutBatch] = None
        self.store_dataset = bool(store_dataset)
        self._current_obs: Optional[np.ndarray] = None
        self._episode_task_return = 0.0
        self._episode_augmented_return = 0.0

    # ------------------------------------------------------------------ inference
    def _infer_obs_dim(self) -> int:
        sp = self.observation_space
        if sp is not None and hasattr(sp, "shape") and sp.shape is not None:
            try:
                return int(np.prod(sp.shape))
            except Exception:  # pragma: no cover
                pass
        obs, _ = MaskEnv._unpack_reset(self.env.reset())
        self._probe_obs = obs
        return int(np.asarray(flatten_observation(obs)).shape[0])

    def _infer_action_dim(self) -> int:
        if self.discrete:
            return int(getattr(self.action_space, "n", 1) or 1)
        sp = self.action_space
        if sp is not None and hasattr(sp, "shape") and sp.shape is not None:
            try:
                return int(np.prod(sp.shape))
            except Exception:  # pragma: no cover
                pass
        return 1

    def _infer_horizon(self) -> int:
        for attr in ("rice_max_episode_steps",):
            val = getattr(self.env, attr, None)
            if val:
                return int(val)
        spec = getattr(self.env, "spec", None)
        if spec is not None and getattr(spec, "max_episode_steps", None):
            return int(spec.max_episode_steps)
        time_limit = getattr(self.env, "_max_episode_steps", None)
        if time_limit:
            return int(time_limit)
        return 1000  # MuJoCo default horizon

    # ------------------------------------------------------------------ torch utils
    def _obs_tensor(self, obs: Any):
        flat = np.atleast_2d(np.asarray(flatten_observation(obs), dtype=np.float32))
        return torch.as_tensor(flat, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ rollout
    def collect_rollout(self) -> RolloutBatch:
        """One iteration of Algorithm 1: roll out ``T`` steps with ``pi_tilde_{theta_old}``."""
        assert self.mask_net is not None and torch is not None
        mask_net, critic = self.mask_net, self.critic

        if self._current_obs is None:
            self._current_obs, _ = MaskEnv._unpack_reset(self.mask_env.reset())
        obs = self._current_obs

        obs_list, next_obs_list, mask_list, rew_list = [], [], [], []
        task_rew_list, bonus_list = [], []
        logp_list, val_list, done_list, trunc_list = [], [], [], []
        dataset: List[Dict[str, Any]] = []
        ep_task, ep_aug = [], []

        with torch.no_grad():
            for _ in range(self.n_steps):
                obs_arr = np.asarray(flatten_observation(obs), dtype=np.float32)
                obs_t = torch.as_tensor(
                    obs_arr.reshape(1, -1), dtype=torch.float32, device=self.device
                )
                logits = _mask_logits(mask_net, obs_t)
                dist = torch.distributions.Categorical(logits=logits.reshape(1, -1))
                mask_t = dist.sample()
                log_prob_t = dist.log_prob(mask_t)
                value_t = _critic_values(critic, obs_t)
                value_t = torch.as_tensor(value_t, dtype=torch.float32, device=self.device).reshape(-1)

                mask = int(mask_t.reshape(-1)[0].item())
                result = self.mask_env.step(mask)
                if len(result) == 5:
                    next_obs, reward, terminated, truncated, info = result
                else:
                    next_obs, reward, done, info = result
                    terminated, truncated = bool(done), False

                obs_list.append(obs_arr)
                next_obs_list.append(np.asarray(flatten_observation(next_obs), dtype=np.float32))
                mask_list.append(mask)
                rew_list.append(float(reward))
                task_rew_list.append(float(info.get("task_reward", reward)))
                bonus_list.append(float(info.get("blinding_bonus", 0.0)))
                logp_list.append(float(log_prob_t.reshape(-1)[0].item()))
                val_list.append(float(value_t.reshape(-1)[0].item()))
                done_list.append(1.0 if terminated else 0.0)
                trunc_list.append(1.0 if truncated else 0.0)

                self._episode_task_return += float(info.get("task_reward", reward))
                self._episode_augmented_return += float(reward)

                if self.store_dataset:
                    dataset.append(
                        {
                            "s_t": obs_arr,
                            "s_t1": np.asarray(flatten_observation(next_obs), dtype=np.float32),
                            "a_t_m": mask,
                            "R_prime": float(reward),
                            "task_reward": float(info.get("task_reward", reward)),
                            "blinding_bonus": float(info.get("blinding_bonus", 0.0)),
                        }
                    )

                obs = next_obs
                if terminated or truncated:
                    ep_task.append(self._episode_task_return)
                    ep_aug.append(self._episode_augmented_return)
                    self._episode_task_return = 0.0
                    self._episode_augmented_return = 0.0
                    obs, _ = MaskEnv._unpack_reset(self.mask_env.reset())

            # bootstrap value
            last_obs_arr = np.asarray(flatten_observation(obs), dtype=np.float32)
            last_obs_t = torch.as_tensor(
                last_obs_arr.reshape(1, -1), dtype=torch.float32, device=self.device
            )
            last_value = float(
                torch.as_tensor(
                    _critic_values(critic, last_obs_t), dtype=torch.float32, device=self.device
                )
                .reshape(-1)[0]
                .item()
            )

        self._current_obs = obs

        # GAE with correct bootstrap on time-limit truncation
        advantages, returns = compute_gae(
            rewards=rew_list,
            values=val_list,
            dones=done_list,
            last_value=last_value,
            gamma=float(self.config.gamma),
            gae_lambda=float(self.config.gae_lambda),
        )

        batch = RolloutBatch(
            observations=np.asarray(obs_list, dtype=np.float32),
            next_observations=np.asarray(next_obs_list, dtype=np.float32),
            masks=np.asarray(mask_list, dtype=np.int64),
            rewards=np.asarray(rew_list, dtype=np.float32),
            task_rewards=np.asarray(task_rew_list, dtype=np.float32),
            bonuses=np.asarray(bonus_list, dtype=np.float32),
            log_probs=np.asarray(logp_list, dtype=np.float32),
            values=np.asarray(val_list, dtype=np.float32),
            dones=np.asarray(done_list, dtype=np.float32),
            truncated=np.asarray(trunc_list, dtype=np.float32),
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
            episode_task_returns=ep_task,
            episode_augmented_returns=ep_aug,
            dataset=dataset,
        )
        # Algorithm 1 sets theta_old <- theta at this point (the sampled log-probs are
        # the ones of theta_old and are kept in `batch.log_probs`).
        self.last_batch = batch
        self.total_samples += len(batch)
        return batch

    # ------------------------------------------------------------------ PPO update
    def update(self, batch: Optional[RolloutBatch] = None) -> Dict[str, float]:
        """Vanilla PPO update of ``theta`` on ``D`` (Algorithm 1, last line)."""
        if batch is None:
            batch = self.last_batch.or_else if False else self.last_batch  # type: ignore[attr-defined]
        if batch is None:
            raise ValueError("no rollout available: call collect_rollout() first")
        assert torch is not None

        cfg = self.config
        mask_net, critic, opt = self.mask_net, self.critic, self.optimizer

        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(batch.masks, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.asarray(batch.advantages), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(np.asarray(batch.returns), dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

        n = len(batch)
        batch_size = max(1, min(int(cfg.batch_size), n))
        stats: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "n_updates": 0.0,
        }
        stop_early = False

        for _epoch in range(int(cfg.n_epochs)):
            if stop_early:
                break
            perm = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                mb = perm[start : start + batch_size]
                mb_idx = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                logits = _mask_logits(mask_net, obs[mb_idx])
                dist = torch.distributions.Categorical(logits=logits.reshape(len(mb), -1))
                new_log_probs = dist.log_prob(masks[mb_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)
                mb_adv = advantages[mb_idx]
                if cfg.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

                clip = float(cfg.clip_range)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = _critic_values(critic, obs[mb_idx])
                values = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1)
                if cfg.clip_range_vf is not None:
                    v_clipped = old_values[mb_idx] + torch.clamp(
                        values - old_values[mb_idx], -float(cfg.clip_range_vf), float(cfg.clip_range_vf)
                    )
                    value_loss = torch.max(
                        F.mse_loss(values, returns[mb_idx]),
                        F.mse_loss(v_clipped, returns[mb_idx]),
                    )
                else:
                    value_loss = F.mse_loss(values, returns[mb_idx])

                loss = policy_loss + float(cfg.vf_coef) * value_loss - float(cfg.ent_coef) * entropy
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(mask_net.parameters()) + list(critic.parameters()),
                    float(cfg.max_grad_norm),
                )
                opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()  # Schulman k3 estimator
                    clip_frac = (torch.abs(ratio - 1.0) > clip).float().mean()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_fraction"] += float(clip_frac.item())
                stats["grad_norm"] += float(
                    grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                )
                stats["n_updates"] += 1.0
                self.num_updates += 1

                if cfg.target_kl is not None and approx_kl.item() > 1.5 * float(cfg.target_kl):
                    stop_early = True
                    break

        denom = max(1.0, stats["n_updates"])
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"):
            stats[key] /= denom
        stats["stopped_early"] = 1.0 if stop_early else 0.0
        return stats

    # ------------------------------------------------------------------ train loop
    def train(
        self,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        logger: Any = None,
        progress: bool = False,
    ) -> List[Dict[str, float]]:
        """Run Algorithm 1 for a sample budget (or a fixed number of iterations)."""
        logger = logger or self.logger
        if total_iterations is None:
            budget = int(total_timesteps if total_timesteps is not None else self.config.total_timesteps)
            total_iterations = max(1, int(np.ceil(budget / float(self.n_steps))))
        total_iterations = int(total_iterations)

        t_start = time.perf_counter()
        for it in range(total_iterations):
            t0 = time.perf_counter()
            batch = self.collect_rollout()
            t1 = time.perf_counter()
            stats = self.update(batch)
            t2 = time.perf_counter()

            self.timings["rollout"] += t1 - t0
            self.timings["update"] += t2 - t1
            self.iterations_done += 1

            record: Dict[str, float] = {
                "iteration": float(self.iterations_done),
                "samples": float(self.total_samples),
                "mean_reward": float(np.mean(batch.task_rewards)),
                "mean_augmented_reward": float(np.mean(batch.rewards)),
                "mean_blinding_bonus": float(np.mean(batch.bonuses)),
                "blind_fraction": float(batch.blind_fraction),
                "rollout_time": float(t1 - t0),
                "update_time": float(t2 - t1),
            }
            if batch.episode_task_returns:
                record["episode_task_return"] = float(np.mean(batch.episode_task_returns))
                record["episode_augmented_return"] = float(
                    np.mean(batch.episode_augmented_returns)
                )
            record.update(stats)
            self.history.append(record)
            self.timer_history.append(
                {"iteration": float(self.iterations_done), "rollout": t1 - t0, "update": t2 - t1}
            )

            if logger is not None and (self.iterations_done % max(1, int(self.config.log_interval)) == 0):
                try:
                    logger.record(**{("mask/" + k): v for k, v in record.items()})
                except Exception:  # pragma: no cover - logger is best effort
                    pass
            if progress and logger is not None and hasattr(logger, "info"):  # pragma: no cover
                try:
                    logger.info(
                        "mask iter %d | samples %d | blind %.3f | pi_loss %.4f",
                        self.iterations_done,
                        self.total_samples,
                        record["blind_fraction"],
                        record["policy_loss"],
                    )
                except Exception:
                    pass

        self.timings["total"] += time.perf_counter() - t_start
        return self.history

    # ------------------------------------------------------------------ evaluation API
    def importance(self, observations: Any, batch_size: int = 4096) -> np.ndarray:
        """State importance = ``P(mask = 0 | s)`` ("keep") under the trained mask."""
        return state_importance(self.mask_net, observations, batch_size=batch_size)

    def keep_probability(self, observations: Any) -> np.ndarray:
        return self.importance(observations)

    def blind_probability(self, observations: Any) -> np.ndarray:
        return 1.0 - self.importance(observations)

    # ------------------------------------------------------------------ reporting
    @property
    def total_time(self) -> float:
        return float(self.timings.get("total", 0.0))

    @property
    def seconds_per_sample(self) -> float:
        return float(self.total_time) / float(self.total_samples) if self.total_samples else 0.0

    def time_report(self) -> Dict[str, float]:
        """Wall-clock statistics used for the Table-4 training-time comparison."""
        return {
            "total_time": self.total_time,
            "rollout_time": float(self.timings.get("rollout", 0.0)),
            "update_time": float(self.timings.get("update", 0.0)),
            "total_samples": float(self.total_samples),
            "iterations": float(self.iterations_done),
            "num_updates": float(self.num_updates),
            "seconds_per_sample": self.seconds_per_sample,
            "time_per_1k_samples": 1000.0 * self.seconds_per_sample,
            "algorithm": "rice_vanilla_ppo",
            "objective": "max eta(pi_bar) + alpha*a_t^m",
        }

    def summary(self) -> Dict[str, Any]:
        latest = self.history[-1] if self.history else {}
        return {
            "env": self.env_id,
            "config": self.config.to_dict(),
            "n_steps": self.n_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "samples": self.total_samples,
            "iterations": self.iterations_done,
            "latest": latest,
            "time": self.time_report(),
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        return save_mask_network(
            self.mask_net,
            path,
            env_id=self.env_id,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            extra={
                "critic": (self.critic.state_dict() if hasattr(self.critic, "state_dict") else None),
                "config": self.config.to_dict(),
                "history": self.history,
                "time": self.time_report(),
                **(extra or {}),
            },
        )

    @classmethod
    def load_trainer(
        cls,
        path: str,
        env: Any,
        target_policy: Any,
        **kwargs: Any,
    ) -> "MaskTrainer":
        mask_net = load_mask_network(path, **{k: v for k, v in kwargs.items() if k in {"device"}})
        return cls(env=env, target_policy=target_policy, mask_net=mask_net, **kwargs)


# --------------------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------------------
def make_mask_env(env: Any, policy: Any, **kwargs: Any) -> MaskEnv:
    """Build a :class:`MaskEnv` (frozen ``pi`` + binary mask action)."""
    return MaskEnv(env=env, policy=policy, **kwargs)


def train_mask_network(
    env: Any,
    target_policy: Any,
    total_timesteps: Optional[int] = None,
    alpha: float = DEFAULT_ALPHA,
    env_id: str = "default",
    config: Optional[Union[MaskPPOConfig, Dict[str, Any]]] = None,
    logger: Any = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    store_dataset: bool = False,
    progress: bool = False,
    **kwargs: Any,
) -> Tuple[Any, MaskTrainer]:
    """Build + train the Stage-1 mask network (Algorithm 1) end to end.

    Returns ``(mask_net, trainer)`` where ``trainer.time_report()`` provides the
    wall-clock numbers needed for the paper's Table-4 efficiency comparison.
    """
    trainer = MaskTrainer(
        env=env,
        target_policy=target_policy,
        config=config,
        env_id=env_id,
        seed=seed,
        device=device,
        logger=logger,
        store_dataset=store_dataset,
        alpha=alpha,
        **kwargs,
    )
    trainer.train(total_timesteps=total_timesteps, logger=logger, progress=progress)
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        trainer.save(save_path)
    return trainer.mask_net, trainer
