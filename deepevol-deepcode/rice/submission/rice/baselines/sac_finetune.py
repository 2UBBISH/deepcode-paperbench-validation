"""SAC fine-tuning baseline for RICE (ICML 2024, PMLR 235).

Paper references
----------------
* §4.2 (Experiment IV): *"Finally, we examine the refining performance when the
  pre-trained agent was trained by other algorithms such as Soft Actor-Critic
  (SAC) (Haarnoja et al., 2018). First, we obtain a pre-trained SAC agent and
  then use Generative Adversarial Imitation Learning (GAIL) ... We compare the
  refining performance using our method against baseline methods ... In
  addition, we also include fine-tuning the pre-trained SAC agent with the SAC
  algorithm as a baseline."*
* §4.1 (Baseline Refining Methods) / Appendix C.1: baseline implementations use
  the authors' released code or an own version, and all refining baselines use
  the *same explanation* as RICE whenever one is needed.

This module therefore provides the **"SAC fine-tuning"** refining baseline: an
off-policy SAC v2 (Haarnoja et al. 2018) trainer that starts from a pre-trained
SAC agent and continues training it with SAC.  It deliberately contains **none**
of RICE's contributions (no mixed initial state distribution ``mu(s)``, no
critical-state reset, no RND intrinsic bonus) — the comparison is fair because
the underlying algorithm is identical to the one used for pre-training.

Because the paper reports that the refining baselines must reuse RICE's
explanation when one is required, :class:`SACRefiner` optionally chains a SAC
fine-tuned policy into RICE's Stage-2 refinement (``rice.refining.ppo_refine``)
so Experiment IV can compose ``pre-trained SAC -> GAIL approximation -> RICE``.

Two execution paths are supported:

1. **SB3 path** (:class:`SB3SACFineTuner`) — if the caller supplies a Stable-
   Baselines3 ``SAC`` model (the natural artifact of pre-training with SB3), we
   simply continue ``model.learn(total_timesteps, reset_num_timesteps=False)``.
   Unspecified SAC hyper-parameters follow SB3 defaults.
2. **Native path** (:class:`SACFinetuner`) — a self-contained PyTorch SAC
   implementation (twin critics, target networks, automatic entropy tuning,
   replay buffer) used when no SB3 model is available and/or only a policy
   network (e.g. the GAIL-approximated ``pi_G``) is provided.

Both paths expose the same surface as the other refining baselines
(``train``/``evaluate``/``summary``/``save``/``save_policy``) so experiment
drivers can swap baselines transparently.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Optional dependencies (kept defensive so this module always imports)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on environment
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Adam = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover
    from ..utils.seeding import get_rng, set_seed  # type: ignore
except Exception:  # pragma: no cover
    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        try:
            np.random.seed(int(seed))
        except Exception:
            pass
        return int(seed)

    def get_rng(seed: Optional[int] = None):  # type: ignore
        return np.random.RandomState(seed)

try:  # pragma: no cover
    from ..utils.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = logging.INFO):  # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

try:  # pragma: no cover
    from ..utils.io import ensure_dir  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(path: str) -> str:  # type: ignore
        if path:
            os.makedirs(path, exist_ok=True)
        return path

try:  # pragma: no cover
    from ..models.policies import (  # type: ignore
        build_policy,
        load_policy,
        normalize_env_key,
        sample_random_action,
        save_policy,
    )

    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    _HAS_POLICIES = False
    build_policy = load_policy = save_policy = None  # type: ignore
    sample_random_action = None  # type: ignore

    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        if env_id is None:
            return "default"
        key = str(env_id).strip().lower()
        for prefix in ("sparse_", "sparse-", "sparse"):
            if key.startswith(prefix):
                key = key[len(prefix):]
        key = key.split("/")[-1].split(".yaml")[0].split(".yml")[0]
        key = key.replace("-", "_").strip("_")
        if "_v" in key:
            head, _, tail = key.rpartition("_v")
            if tail.isdigit():
                key = head
        return key or "default"

try:  # pragma: no cover
    from ..refining.ppo_refine import (  # type: ignore
        DEFAULT_ENT_COEF,
        DEFAULT_GAMMA,
        DEFAULT_GAE_LAMBDA,
        DEFAULT_HORIZON,
        DEFAULT_MAX_GRAD_NORM,
        PPORefiner,
        RefinePPOConfig,
        evaluate_refined_policy,
        prepare_action_for_env,
        refine_policy,
        unpack_reset,
        unpack_step,
    )

    _HAS_REFINER = True
except Exception:  # pragma: no cover
    _HAS_REFINER = False
    PPORefiner = None  # type: ignore
    RefinePPOConfig = None  # type: ignore
    refine_policy = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore
    DEFAULT_GAMMA = 0.99
    DEFAULT_HORIZON = 1000

    def unpack_reset(result):  # type: ignore
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return result, {}

    def unpack_step(result):  # type: ignore
        if isinstance(result, tuple):
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return obs, float(reward), bool(terminated), bool(truncated), info or {}
            if len(result) == 4:
                obs, reward, done, info = result
                return obs, float(reward), bool(done), False, info or {}
        return result, 0.0, False, False, {}

    def prepare_action_for_env(action, action_space=None, discrete=None):  # type: ignore
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_space is not None and hasattr(action_space, "low") and hasattr(action_space, "high"):
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            if low.shape == action.shape:
                action = np.clip(action, low, high)
        return action


# --------------------------------------------------------------------------- #
# Defaults (SB3 SAC defaults where the paper does not specify / Appendix C.1)
# --------------------------------------------------------------------------- #
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (256, 256)      # SB3 SAC default net_arch
DEFAULT_ACTIVATION = "relu"                              # SB3 SAC default
DEFAULT_LR = 3e-4                                        # SB3 SAC default lr
DEFAULT_BATCH_SIZE = 256                                 # SB3 SAC default
DEFAULT_BUFFER_SIZE = 1_000_000                          # SB3 SAC default
DEFAULT_LEARNING_STARTS = 100                            # SB3 SAC default
DEFAULT_TAU = 0.005                                      # SB3 SAC default
DEFAULT_GAMMA_SAC = 0.99                                 # SB3 SAC default
DEFAULT_TRAIN_FREQ = 1                                   # SB3 SAC default (env steps)
DEFAULT_GRADIENT_STEPS = 1                               # SB3 SAC default
DEFAULT_ENT_COEF = "auto"                                # SB3 SAC default (automatic temperature)
DEFAULT_LOG_STD_INIT = -3.0
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_FINETUNE_LR = 3e-4
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
SAC_ENT_COEF_MODES = ("auto", "fixed")
SAC_REWARD_MODES = ("task",)

_LOGGER = logging.getLogger("rice.baselines.sac_finetune")


def _has_torch() -> bool:
    return _HAS_TORCH


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError(
            "rice.baselines.sac_finetune requires PyTorch to train/refine with SAC. "
            "Install torch or pass a Stable-Baselines3 SAC model to SB3SACFineTuner."
        )


def _module_base() -> type:
    """``nn.Module`` when torch is present, otherwise ``object`` (import safety)."""
    return nn.Module if _HAS_TORCH else object


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SACFinetuneConfig:
    """Hyper-parameters of the SAC fine-tuning baseline.

    The paper does not state SAC hyper-parameters (§4.1/§4.2 are silent), so we
    follow Stable-Baselines3's SAC defaults, which is also the library the
    authors use for the MuJoCo experiments (Appendix C.1).
    """

    env_id: str = "default"

    # --- optimisation -----------------------------------------------------
    lr: float = DEFAULT_FINETUNE_LR                       # actor & critic learning rate
    learning_rate: Optional[float] = None                 # alias of ``lr``
    lr_alpha: float = DEFAULT_FINETUNE_LR                 # temperature learning rate
    gamma: float = DEFAULT_GAMMA_SAC
    tau: float = DEFAULT_TAU
    batch_size: int = DEFAULT_BATCH_SIZE

    # --- update schedule --------------------------------------------------
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    total_iterations: Optional[int] = None
    train_freq: int = DEFAULT_TRAIN_FREQ                  # env steps per gradient update
    gradient_steps: int = DEFAULT_GRADIENT_STEPS
    learning_starts: int = DEFAULT_LEARNING_STARTS        # random-action warmup
    n_steps: Optional[int] = None                         # rollout length per iteration

    # --- replay / exploration --------------------------------------------
    buffer_size: int = DEFAULT_BUFFER_SIZE
    ent_coef: Union[str, float] = DEFAULT_ENT_COEF
    target_entropy: Optional[float] = None                # default: -action_dim
    action_noise: float = 0.0                             # extra Gaussian action noise
    reward_scale: float = 1.0

    # --- networks ---------------------------------------------------------
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES
    activation: str = DEFAULT_ACTIVATION
    policy_net_arch: Optional[Sequence[int]] = None       # alias of ``hidden_sizes``
    log_std_init: float = DEFAULT_LOG_STD_INIT

    # --- bookkeeping ------------------------------------------------------
    device: str = "cpu"
    seed: Optional[int] = None
    log_interval: int = 1
    eval_episodes: int = 10
    deterministic_eval: bool = True
    copy_policy: bool = True
    verbose: bool = True

    # --- RICE components: always OFF for this baseline --------------------
    use_mixed_init: bool = False
    use_rnd: bool = False
    p: float = 0.0
    lam: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate is not None:
            self.lr = float(self.learning_rate)
        if self.policy_net_arch is not None:
            self.hidden_sizes = tuple(int(h) for h in self.policy_net_arch)
        elif not isinstance(self.hidden_sizes, tuple):
            self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
        if isinstance(self.ent_coef, str) and self.ent_coef.lower() != "auto":
            # allow numeric strings from YAML ("0.2")
            try:
                self.ent_coef = float(self.ent_coef)
            except ValueError:
                self.ent_coef = "auto"
        # This baseline must never silently become RICE (see module docstring).
        self.use_mixed_init = False
        self.use_rnd = False
        self.p = 0.0
        self.lam = 0.0

    # -- convenience -------------------------------------------------------
    @property
    def lam(self) -> float:  # type: ignore[override]
        return float(getattr(self, "_lam", 0.0) or 0.0)

    @lam.setter
    def lam(self, value: float) -> None:
        self._lam = 0.0  # forced off

    @property
    def timesteps(self) -> int:
        return int(self.total_timesteps or 0)

    @property
    def auto_entropy(self) -> bool:
        return isinstance(self.ent_coef, str) and self.ent_coef.lower() == "auto"

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data.pop("_lam", None)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "SACFinetuneConfig":
        """Build a config from a (possibly nested) YAML dictionary."""
        cfg = dict(cfg or {})
        for section in ("sac_finetune", "sac", "baseline", "finetune", "refine", "refining"):
            section_cfg = cfg.pop(section, None)
            if isinstance(section_cfg, dict):
                merged = dict(cfg)
                merged.update(section_cfg)
                cfg = merged
        aliases = {
            "learning_rate": "learning_rate",
            "actor_lr": "lr",
            "critic_lr": "lr",
            "alpha_lr": "lr_alpha",
            "hidden": "hidden_sizes",
            "net_arch": "hidden_sizes",
            "hidden_sizes": "hidden_sizes",
            "bs": "batch_size",
            "lambda": "lam",
            "lambda_": "lam",
            "beta": "p",
            "prob": "p",
            "timesteps": "total_timesteps",
        }
        known = set(getattr(cls, "__dataclass_fields__", {}).keys())
        kwargs: Dict[str, Any] = {}
        for key, value in cfg.items():
            key = aliases.get(key, key)
            if key in known or key in ("learning_rate", "policy_net_arch"):
                kwargs[key] = value
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Native SAC networks
# --------------------------------------------------------------------------- #
def _build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: str = DEFAULT_ACTIVATION,
) -> Any:
    """Small MLP builder (SB3-style activation defaults)."""
    _require_torch()
    acts = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    act_cls = acts.get(str(activation).lower(), nn.ReLU)
    layers: List[Any] = []
    last = int(input_dim)
    for hidden in hidden_sizes:
        layer = nn.Linear(last, int(hidden))
        try:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        except Exception:
            pass
        layers.append(layer)
        layers.append(act_cls())
        last = int(hidden)
    out = nn.Linear(last, int(output_dim))
    try:
        nn.init.orthogonal_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
    except Exception:
        pass
    layers.append(out)
    return nn.Sequential(*layers)


class SACGaussianActor(_module_base()):  # type: ignore[misc]
    """Squashed-Gaussian actor for SAC (Haarnoja et al., 2018)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = DEFAULT_ACTIVATION,
        log_std_init: float = DEFAULT_LOG_STD_INIT,
        squash: bool = True,
        action_low: Optional[Sequence[float]] = None,
        action_high: Optional[Sequence[float]] = None,
    ) -> None:
        if _HAS_TORCH:
            super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.squash = bool(squash)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = activation
        self.body = _build_mlp(self.obs_dim, self.hidden_sizes, self.action_dim, activation)
        self.log_std_layer = _build_mlp(self.obs_dim, self.hidden_sizes, self.action_dim, activation)
        if _HAS_TORCH:
            try:
                nn.init.constant_(self.log_std_layer[-1].bias, float(log_std_init))
            except Exception:
                pass
        self._low = np.asarray(action_low, dtype=np.float32) if action_low is not None else None
        self._high = np.asarray(action_high, dtype=np.float32) if action_high is not None else None

    # -- core -------------------------------------------------------------
    def forward(self, obs: Any) -> Tuple[Any, Any]:
        if not _HAS_TORCH:
            raise ImportError("SACGaussianActor requires PyTorch.")
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self._device())
        mu = self.body(obs)
        log_std = self.log_std_layer(obs)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def _device(self) -> Any:
        try:
            return next(self.parameters()).device
        except Exception:
            return torch.device("cpu")

    def distribution(self, obs: Any) -> Any:
        mu, log_std = self.forward(obs)
        return torch.distributions.Normal(mu, log_std.exp())

    def sample(
        self,
        obs: Any,
        deterministic: bool = False,
        reparameterize: bool = True,
    ) -> Tuple[Any, Any, Any, Any]:
        """Return ``(action, log_prob, mu, log_std)`` with tanh squashing."""
        mu, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mu, log_std.exp())
        if deterministic:
            u = mu
        elif reparameterize:
            u = dist.rsample()
        else:
            u = dist.sample()
        if self.squash:
            action = torch.tanh(u)
            log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        else:
            action = u
            log_prob = dist.log_prob(u)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu, log_std

    def action_log_prob(self, obs: Any, actions: Any) -> Any:
        """Log-density of (already squashed) actions — needed for SB3 import."""
        if not _HAS_TORCH:
            raise ImportError("SACGaussianActor requires PyTorch.")
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self._device())
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        if self.squash:
            actions = torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6)
            u = 0.5 * torch.log((1.0 + actions) / (1.0 - actions))
            log_prob = torch.distributions.Normal(mu, std).log_prob(u)
            log_prob = log_prob - torch.log(1.0 - actions.pow(2) + 1e-6)
        else:
            log_prob = torch.distributions.Normal(mu, std).log_prob(actions)
        return log_prob.sum(dim=-1, keepdim=True)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map in [-1, 1] to the environment's action range when known."""
        action = np.asarray(action, dtype=np.float32)
        if self._low is not None and self._high is not None:
            low, high = self._low, self._high
            if low.shape == action.shape:
                return low + 0.5 * (action + 1.0) * (high - low)
        return action

    @property
    def net_arch(self) -> List[int]:
        return list(self.hidden_sizes)


class SACCritic(_module_base()):  # type: ignore[misc]
    """Twin Q-network ``(Q1, Q2)`` for SAC."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        if _HAS_TORCH:
            super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.q1 = _build_mlp(self.obs_dim + self.action_dim, hidden_sizes, 1, activation)
        self.q2 = _build_mlp(self.obs_dim + self.action_dim, hidden_sizes, 1, activation)

    def forward(self, obs: Any, action: Any) -> Tuple[Any, Any]:
        if not _HAS_TORCH:
            raise ImportError("SACCritic requires PyTorch.")
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: Any, action: Any) -> Any:
        if not _HAS_TORCH:
            raise ImportError("SACCritic requires PyTorch.")
        return self.q1(torch.cat([obs, action], dim=-1))


class SACTemperature(_module_base()):  # type: ignore[misc]
    """Learnable log-temperature for automatic entropy tuning (SAC v2)."""

    def __init__(self, init_value: float = 1.0) -> None:
        if _HAS_TORCH:
            super().__init__()
            self.log_alpha = nn.Parameter(torch.tensor(float(np.log(init_value)), dtype=torch.float32))
        else:  # pragma: no cover
            self.log_alpha = None

    @property
    def alpha(self) -> Any:
        return self.log_alpha.exp()


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class SACReplayBuffer:
    """Simple ring buffer of ``(s, a, r, s', done)`` transitions."""

    def __init__(
        self,
        capacity: int = DEFAULT_BUFFER_SIZE,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        discrete: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(max(1, capacity))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = bool(discrete)
        self._observations: List[np.ndarray] = [None] * self.capacity  # type: ignore
        self._next_observations: List[np.ndarray] = [None] * self.capacity  # type: ignore
        self._actions: List[Any] = [None] * self.capacity
        self._rewards: np.ndarray = np.zeros(self.capacity, dtype=np.float32)
        self._dones: np.ndarray = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = get_rng(seed)
        self.n_added = 0

    # -- writing -----------------------------------------------------------
    def add(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self._observations[self.pos] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._next_observations[self.pos] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        if self.discrete:
            self._actions[self.pos] = int(np.asarray(action).reshape(-1)[0]) if np.ndim(action) else int(action)
        else:
            self._actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(-1)
        self._rewards[self.pos] = float(reward)
        self._dones[self.pos] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.n_added += 1

    def add_batch(self, transitions: Sequence[Tuple[Any, Any, float, Any, bool]]) -> int:
        for transition in transitions:
            self.add(*transition)
        return len(transitions)

    def extend(self, observations, actions, rewards, next_observations, dones) -> int:
        n = len(np.asarray(rewards).reshape(-1))
        for i in range(n):
            self.add(observations[i], actions[i], float(rewards[i]), next_observations[i], bool(dones[i]))
        return n

    # -- reading -----------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self.rng.randint(0, self.size, size=int(batch_size))
        return {
            "observations": np.asarray([self._observations[i] for i in idx], dtype=np.float32),
            "next_observations": np.asarray([self._next_observations[i] for i in idx], dtype=np.float32),
            "actions": np.asarray([self._actions[i] for i in idx]),
            "rewards": self._rewards[idx].reshape(-1, 1),
            "dones": self._dones[idx].reshape(-1, 1),
        }

    def all_transitions(self) -> Dict[str, np.ndarray]:
        if self.size == 0:
            return {}
        idx = list(range(self.size))
        return {
            "observations": np.asarray([self._observations[i] for i in idx], dtype=np.float32),
            "next_observations": np.asarray([self._next_observations[i] for i in idx], dtype=np.float32),
            "actions": np.asarray([self._actions[i] for i in idx]),
            "rewards": self._rewards[idx].reshape(-1, 1),
            "dones": self._dones[idx].reshape(-1, 1),
        }

    @property
    def ready(self) -> bool:
        return self.size > 0

    def clear(self) -> None:
        self._observations = [None] * self.capacity  # type: ignore
        self._next_observations = [None] * self.capacity  # type: ignore
        self._actions = [None] * self.capacity
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def statistics(self) -> Dict[str, float]:
        if self.size == 0:
            return {"size": 0.0, "mean_reward": 0.0}
        rewards = np.asarray([self._rewards[i] for i in range(self.size)], dtype=np.float32)
        return {
            "size": float(self.size),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }

    def __len__(self) -> int:
        return int(self.size)


# --------------------------------------------------------------------------- #
# Rollout container
# --------------------------------------------------------------------------- #
@dataclass
class SACRollout:
    """Transitions gathered during one SAC iteration."""

    observations: List[np.ndarray] = field(default_factory=list)
    next_observations: List[np.ndarray] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    n_random_actions: int = 0
    n_policy_actions: int = 0
    n_env_steps: int = 0

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def mean_episode_reward(self) -> float:
        if not self.episode_rewards:
            return float("nan")
        return float(np.mean(self.episode_rewards))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_env_steps": int(self.n_env_steps),
            "n_episodes": len(self.episode_rewards),
            "mean_episode_reward": self.mean_episode_reward,
            "n_random_actions": int(self.n_random_actions),
            "n_policy_actions": int(self.n_policy_actions),
        }


# --------------------------------------------------------------------------- #
# Helpers shared with the other baselines
# --------------------------------------------------------------------------- #
def _resolve_horizon(env: Any, default: int = DEFAULT_HORIZON) -> int:
    """Best-effort episode horizon lookup (used for ``n_steps`` defaulting)."""
    for attr in ("rice_max_episode_steps", "_max_episode_steps", "max_episode_steps"):
        value = getattr(env, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    spec = getattr(env, "rice_env_spec", None)
    if spec is not None:
        value = getattr(spec, "max_episode_steps", None) or (spec.get("max_episode_steps") if isinstance(spec, dict) else None)
        if value:
            return int(value)
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return _resolve_horizon(inner, default=default)
    for attr in ("spec",):
        spec = getattr(env, attr, None)
        if spec is not None and getattr(spec, "max_episode_steps", None):
            return int(spec.max_episode_steps)
    return int(default)


def resolve_obs_act_dims(env: Any, policy: Any = None) -> Tuple[int, int, bool]:
    """Return ``(obs_dim, action_dim, discrete)`` for env (falling back to the policy)."""
    obs_dim = action_dim = None
    discrete = False
    space = getattr(env, "observation_space", None)
    if space is not None and hasattr(space, "shape") and space.shape:
        obs_dim = int(np.prod(space.shape))
    act_space = getattr(env, "action_space", None)
    if act_space is not None:
        if hasattr(act_space, "shape") and act_space.shape:
            action_dim = int(np.prod(act_space.shape))
        elif hasattr(act_space, "n"):
            action_dim = int(act_space.n)
            discrete = True
    if policy is not None:
        obs_dim = obs_dim or getattr(policy, "obs_dim", None) or getattr(policy, "observation_space", None) and None
        action_dim = action_dim or getattr(policy, "action_dim", None)
        for attr in ("observation_space",):
            space = getattr(policy, attr, None)
            if obs_dim is None and space is not None and getattr(space, "shape", None):
                obs_dim = int(np.prod(space.shape))
    if obs_dim is None:
        obs_dim = int(getattr(policy, "obs_dim", 0) or 0)
    if action_dim is None:
        action_dim = int(getattr(policy, "action_dim", 0) or 0)
    return int(obs_dim or 0), int(action_dim or 0), bool(discrete)


def policy_action(policy: Any, observation: Any, deterministic: bool = False) -> np.ndarray:
    """Interface-agnostic action selection (native torch policy, SB3, or callable)."""
    if policy is None:
        raise ValueError("policy_action() requires a policy.")
    if hasattr(policy, "predict"):  # SB3
        action, _ = policy.predict(observation, deterministic=deterministic)
        return np.asarray(action)
    if hasattr(policy, "act"):
        try:
            result = policy.act(observation, deterministic=deterministic)
        except TypeError:
            result = policy.act(observation)
        if isinstance(result, tuple):
            result = result[0]
        return np.asarray(result)
    if callable(policy):
        result = policy(observation)
        if isinstance(result, tuple):
            result = result[0]
        return np.asarray(result)
    if _HAS_TORCH and isinstance(policy, torch.nn.Module):
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1))
            mu = policy.body(obs) if hasattr(policy, "body") else policy(obs)
            mu = mu[0] if isinstance(mu, tuple) else mu
            return np.asarray(torch.tanh(mu).cpu().numpy().reshape(-1))
    raise TypeError(f"Cannot obtain an action from policy of type {type(policy)!r}.")


def rollout_episode(
    env: Any,
    policy: Any = None,
    deterministic: bool = False,
    max_steps: Optional[int] = None,
    reset_kwargs: Optional[Dict[str, Any]] = None,
    rng: Any = None,
    random_action_fn: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Roll one episode; returns rewards/steps/observations (used for evaluation)."""
    reset_kwargs = dict(reset_kwargs or {})
    try:
        obs, info = unpack_reset(env.reset(**reset_kwargs))
    except TypeError:
        obs, info = unpack_reset(env.reset())
    max_steps = int(max_steps or _resolve_horizon(env))
    total = 0.0
    steps = 0
    observations: List[np.ndarray] = []
    for _ in range(max_steps):
        if policy is None:
            action = random_action_fn() if random_action_fn is not None else env.action_space.sample()
        else:
            action = policy_action(policy, obs, deterministic=deterministic)
        action = prepare_action_for_env(action, getattr(env, "action_space", None), None)
        observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        obs, reward, terminated, truncated, info = unpack_step(env.step(action))
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return {"reward": float(total), "steps": steps, "observations": observations}


def evaluate_policy(
    env: Any,
    policy: Any,
    env_id: str = "default",
    n_episodes: int = 10,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a policy over ``n_episodes`` from the default start distribution."""
    if _HAS_REFINER and evaluate_refined_policy is not None:
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
        except Exception:  # pragma: no cover - fall through to local eval
            pass
    returns: List[float] = []
    steps_list: List[int] = []
    for _ in range(int(max(1, n_episodes))):
        result = rollout_episode(env, policy=policy, deterministic=deterministic, max_steps=max_steps)
        returns.append(result["reward"])
        steps_list.append(result["steps"])
    return {
        "mean_reward": float(np.mean(returns)) if returns else float("nan"),
        "std_reward": float(np.std(returns)) if returns else float("nan"),
        "mean_length": float(np.mean(steps_list)) if steps_list else float("nan"),
        "n_episodes": float(len(returns)),
    }


def is_sb3_sac(model: Any) -> bool:
    """Duck-typed check for a Stable-Baselines3 SAC-like model."""
    if model is None:
        return False
    if not hasattr(model, "learn"):
        return False
    name = type(model).__name__.lower()
    if "sac" in name:
        return True
    return hasattr(model, "actor") and hasattr(model, "critic")


# --------------------------------------------------------------------------- #
# SB3 execution path
# --------------------------------------------------------------------------- #
class SB3SACFineTuner:
    """Continue training a Stable-Baselines3 SAC model (SAC fine-tuning baseline).

    This is the path taken when a pre-trained SB3 ``SAC`` agent is available:
    Experiment IV's baseline is *"fine-tuning the pre-trained SAC agent with the
    SAC algorithm"*, i.e. calling ``model.learn(..., reset_num_timesteps=False)``
    on the pre-trained agent.  ``model`` keeps its (possibly changed) learning
    rate, matching SB3 semantics.
    """

    name = "SAC fine-tuning (SB3)"

    def __init__(
        self,
        env: Any,
        model: Any,
        config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        lr: Optional[float] = None,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.env = env
        self.model = model
        self.env_id = normalize_env_key(env_id)
        self.device = device
        self.logger = logger or _LOGGER
        self.config = (
            config
            if isinstance(config, SACFinetuneConfig)
            else SACFinetuneConfig.from_dict(config if isinstance(config, dict) else None, env_id=self.env_id, **(kwargs or {}))
        )
        if lr is not None:
            self.config.lr = float(lr)
            try:  # SB3 keeps a separate actor/critic lr; set the common one too
                self.model.learning_rate = float(lr)
            except Exception:
                pass
        if total_timesteps is not None:
            self.config.total_timesteps = int(total_timesteps)
        if seed is not None:
            self.config.seed = int(seed)
            set_seed(int(seed))
            try:
                self.model.set_random_seed(int(seed))
            except Exception:
                pass
        self.history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        self._timers: Dict[str, float] = {}
        self._timer_starts: Dict[str, float] = {}

    # -- configuration -----------------------------------------------------
    @property
    def policy(self) -> Any:
        return self.model

    @property
    def policy_network(self) -> Any:
        return getattr(self.model, "actor", self.model)

    @property
    def sac_model(self) -> Any:
        return self.model

    @property
    def lr(self) -> float:
        return float(getattr(self.model, "learning_rate", self.config.lr) or self.config.lr)

    # -- timers ------------------------------------------------------------
    def timer_start(self, name: str = "train") -> float:
        self._timer_starts[name] = time.time()
        return self._timer_starts[name]

    def timer_end(self, name: str = "train", accumulate: bool = True) -> float:
        start = self._timer_starts.pop(name, None)
        if start is None:
            return 0.0
        elapsed = float(time.time() - start)
        self._timers[name] = self._timers.get(name, 0.0) + elapsed if accumulate else elapsed
        return elapsed

    @property
    def total_time(self) -> float:
        return float(self._timers.get("train", 0.0))

    # -- training ----------------------------------------------------------
    def train(
        self,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        logger: Any = None,
        progress: bool = False,
        callback: Any = None,
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        del total_iterations  # SAC is step-budgeted
        logger = logger or self.logger
        timesteps = int(total_timesteps or self.config.total_timesteps)
        before = self._model_timesteps()
        self.timer_start("train")
        try:
            self.model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,
                callback=callback,
                progress_bar=bool(progress),
            )
        except TypeError:  # older SB3 without progress_bar
            self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)
        elapsed = self.timer_end("train")
        after = self._model_timesteps()
        record = {
            "timesteps": float(after),
            "env_steps": float(max(0, after - before)),
            "train_seconds": float(elapsed),
            "mean_reward": float(np.mean(self._last_episode_rewards()) if self._last_episode_rewards() else float("nan")),
        }
        self.history.append(record)
        if logger is not None:
            try:
                logger.info(
                    "SAC fine-tuning (SB3): %d env steps in %.1fs (lr=%s)",
                    int(record["env_steps"]),
                    elapsed,
                    self.lr,
                )
            except Exception:
                pass
        self.save = self.save  # noqa: B018 - keep attribute documented
        return self.history

    fit = train

    def _model_timesteps(self) -> int:
        value = getattr(self.model, "num_timesteps", None)
        if value is None:
            value = getattr(getattr(self.model, "_num_timesteps", None), "data", 0)
        try:
            return int(value)
        except Exception:
            return 0

    def _last_episode_rewards(self) -> List[float]:
        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer:
            return []
        rewards: List[float] = []
        for item in buffer:
            if isinstance(item, dict) and "r" in item:
                rewards.append(float(item["r"]))
        return rewards

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        policy: Any = None,
        max_steps: Optional[int] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        del reset_kwargs
        target = policy if policy is not None else self.model
        result = evaluate_policy(
            self.env,
            target,
            env_id=self.env_id,
            n_episodes=n_episodes,
            max_steps=max_steps,
            deterministic=deterministic,
            device=self.device,
        )
        self.eval_history.append(result)
        return result

    # -- persistence -------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "baseline": "sac_finetune",
            "backend": "sb3",
            "env_id": self.env_id,
            "lr": self.lr,
            "total_time": self.total_time,
            "timesteps": float(self._model_timesteps()),
            "n_updates": float(len(self.history)),
            "final_eval": self.eval_history[-1] if self.eval_history else None,
            "sac_config": self.config.to_dict(),
        }

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:  # type: ignore[no-redef]
        path = ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        target = path if path.endswith(".zip") else path + ".zip"
        try:
            self.model.save(target)
        except Exception:
            import pickle

            with open(target, "wb") as handle:
                pickle.dump(self.model, handle)
        return target

    def save_policy(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        return self.save(path, extra=extra)

    def time_report(self) -> Dict[str, float]:
        return dict(self._timers)


# --------------------------------------------------------------------------- #
# Native SAC trainer
# --------------------------------------------------------------------------- #
class SACFinetuner:
    """Native PyTorch SAC v2 trainer used as the "SAC fine-tuning" baseline.

    The paper does not specify SAC hyper-parameters for the fine-tuning
    baseline, so we use Stable-Baselines3's SAC defaults (net ``[256, 256]``
    ReLU, ``lr=3e-4``, ``gamma=0.99``, ``tau=0.005``, batch ``256``,
    ``learning_starts=100``, automatic entropy tuning).
    """

    name = "SAC fine-tuning"

    def __init__(
        self,
        env: Any,
        policy: Any = None,
        config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        observation_space: Any = None,
        action_space: Any = None,
        discrete: Optional[bool] = None,
        rng: Any = None,
        seed: Optional[int] = None,
        replay_buffer: Optional[SACReplayBuffer] = None,
        lr: Optional[float] = None,
        total_timesteps: Optional[int] = None,
        copy_policy: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        _require_torch()
        self.env = env
        self.env_id = normalize_env_key(env_id)
        self.device = torch.device(device if torch is not None else "cpu")
        self.logger = logger or _LOGGER

        if isinstance(config, SACFinetuneConfig):
            self.config = config
        else:
            base = dict(config) if isinstance(config, dict) else {}
            base.update({k: v for k, v in kwargs.items() if v is not None})
            self.config = SACFinetuneConfig.from_dict(base, env_id=self.env_id)
        if lr is not None:
            self.config.lr = float(lr)
        if total_timesteps is not None:
            self.config.total_timesteps = int(total_timesteps)
        if copy_policy is not None:
            self.config.copy_policy = bool(copy_policy)
        if seed is not None:
            self.config.seed = int(seed)
        if self.config.seed is not None:
            set_seed(int(self.config.seed))
            torch.manual_seed(int(self.config.seed))

        self.rng = rng if rng is not None else get_rng(self.config.seed)

        self.observation_space = observation_space if observation_space is not None else getattr(env, "observation_space", None)
        self.action_space = action_space if action_space is not None else getattr(env, "action_space", None)
        self.obs_dim, self.action_dim, inferred_discrete = self._resolve_dims(policy)
        self.discrete = inferred_discrete if discrete is None else bool(discrete)
        self.horizon = _resolve_horizon(env)
        if self.config.n_steps is None:
            self.config.n_steps = self.horizon

        self.actor = self._build_actor(policy)
        self.critic = SACCritic(self.obs_dim, self.action_dim, self.config.hidden_sizes, self.config.activation).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for param in self.target_critic.parameters():
            param.requires_grad_(False)
        self.temperature: Optional[SACTemperature] = None
        self.log_alpha = None
        self.target_entropy = (
            float(self.config.target_entropy)
            if self.config.target_entropy is not None
            else -float(max(1, self.action_dim))
        )
        if self.config.auto_entropy:
            self.temperature = SACTemperature(1.0).to(self.device)
            self.log_alpha = self.temperature.log_alpha

        self.actor_optimizer = Adam(self.actor.parameters(), lr=float(self.config.lr))
        self.critic_optimizer = Adam(self.critic.parameters(), lr=float(self.config.lr))
        self.alpha_optimizer = (
            Adam([self.log_alpha], lr=float(self.config.lr_alpha)) if self.log_alpha is not None else None
        )

        self.buffer = (
            replay_buffer
            if replay_buffer is not None
            else SACReplayBuffer(self.config.buffer_size, self.obs_dim, self.action_dim, self.discrete, seed=self.config.seed)
        )

        self.observation: Optional[np.ndarray] = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_count = 0
        self.env_steps = 0            # steps taken *inside this trainer*
        self.gradient_updates = 0

        self.history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        self._timers: Dict[str, float] = {}
        self._timer_starts: Dict[str, float] = {}
        self.last_rollout: Optional[SACRollout] = None

    # -- setup helpers -----------------------------------------------------
    def _resolve_dims(self, policy: Any) -> Tuple[int, int, bool]:
        obs_dim, action_dim, discrete = resolve_obs_act_dims(self.env, policy)
        if obs_dim <= 0 and self.observation_space is not None and getattr(self.observation_space, "shape", None):
            obs_dim = int(np.prod(self.observation_space.shape))
        if action_dim <= 0 and self.action_space is not None:
            if getattr(self.action_space, "shape", None):
                action_dim = int(np.prod(self.action_space.shape))
            elif hasattr(self.action_space, "n"):
                action_dim = int(self.action_space.n)
                discrete = True
        if obs_dim <= 0:
            obs_dim = int(getattr(policy, "obs_dim", 0) or 0)
        if action_dim <= 0:
            action_dim = int(getattr(policy, "action_dim", 0) or 0)
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError(
                "SACFinetuner could not infer obs/action dimensions from the environment or policy; "
                "pass observation_space=/action_space= explicitly."
            )
        return obs_dim, action_dim, discrete

    def _action_bounds(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        space = self.action_space
        if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
            return None, None
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        if low.shape != (self.action_dim,) or high.shape != (self.action_dim,):
            return None, None
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            return None, None
        return low, high

    def _build_actor(self, policy: Any) -> Any:
        """Initialise the SAC actor from the pre-trained agent when possible."""
        low, high = self._action_bounds()
        # (a) an existing native SACGaussianActor (e.g. a pre-trained SAC agent)
        if isinstance(policy, SACGaussianActor):
            actor = copy.deepcopy(policy) if self.config.copy_policy else policy
            return actor.to(self.device)
        # (b) a Stable-Baselines3 SAC model: copy its actor weights
        if is_sb3_sac(policy):
            actor = SACGaussianActor(
                self.obs_dim,
                self.action_dim,
                self.config.hidden_sizes,
                self.config.activation,
                self.config.log_std_init,
                action_low=low,
                action_high=high,
            ).to(self.device)
            try:
                sb3_actor = policy.actor
                state = sb3_actor.state_dict()
                keys = [k for k in state.keys() if k.startswith("latent_pi") or k.startswith("mu") or k.startswith("log_std")]
                own = actor.state_dict()
                for key in keys:
                    if key in own and own[key].shape == state[key].shape:
                        own[key].copy_(state[key])
                actor.load_state_dict(own)
                self.logger.info("SAC fine-tuning: copied %d weight tensors from the SB3 SAC actor.", len(keys))
            except Exception as exc:  # pragma: no cover - architecture mismatch
                self.logger.warning("Could not transfer SB3 SAC actor weights (%s); training from scratch.", exc)
            return actor
        # (c) a generic native torch policy: reuse its body if shapes match
        actor = SACGaussianActor(
            self.obs_dim,
            self.action_dim,
            self.config.hidden_sizes,
            self.config.activation,
            self.config.log_std_init,
            action_low=low,
            action_high=high,
        ).to(self.device)
        if policy is not None and _HAS_TORCH and isinstance(policy, torch.nn.Module):
            try:
                actor_state = actor.state_dict()
                policy_state = policy.state_dict()
                copied = 0
                for key, value in policy_state.items():
                    if key in actor_state and actor_state[key].shape == value.shape:
                        actor_state[key].copy_(value)
                        copied += 1
                if copied:
                    actor.load_state_dict(actor_state)
                    self.logger.info("SAC fine-tuning: initialised actor with %d tensors from the provided policy.", copied)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Could not transfer policy weights to the SAC actor (%s).", exc)
        return actor

    # -- timers ------------------------------------------------------------
    def timer_start(self, name: str = "train") -> float:
        self._timer_starts[name] = time.time()
        return self._timer_starts[name]

    def timer_end(self, name: str = "train", accumulate: bool = True) -> float:
        start = self._timer_starts.pop(name, None)
        if start is None:
            return 0.0
        elapsed = float(time.time() - start)
        self._timers[name] = self._timers.get(name, 0.0) + elapsed if accumulate else elapsed
        return elapsed

    @property
    def total_time(self) -> float:
        return float(self._timers.get("train", 0.0))

    @property
    def seconds_per_sample(self) -> float:
        return self.total_time / float(max(1, self.env_steps))

    # -- action selection --------------------------------------------------
    def _select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.env_steps < int(self.config.learning_starts):
            action = self._random_action()
        else:
            with torch.no_grad():
                obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1), device=self.device)
                action_t, _, _, _ = self.actor.sample(obs, deterministic=deterministic)
                action = action_t.cpu().numpy().reshape(-1)
            action = self.actor.scale_action(action)
        if self.config.action_noise > 0:
            action = action + self.rng.normal(0.0, float(self.config.action_noise), size=np.shape(action))
        return prepare_action_for_env(action, self.action_space, self.discrete)

    def _random_action(self) -> Any:
        if sample_random_action is not None:
            try:
                return sample_random_action(self.action_space, rng=self.rng, dim=self.action_dim, discrete=self.discrete)
            except Exception:
                pass
        if self.discrete:
            return int(self.rng.randint(0, max(1, self.action_dim)))
        if self.action_space is not None and hasattr(self.action_space, "sample"):
            try:
                return self.action_space.sample()
            except Exception:
                pass
        low, high = self._action_bounds()
        if low is not None and high is not None:
            return self.rng.uniform(low, high).astype(np.float32)
        return self.rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    # -- rollouts ----------------------------------------------------------
    def reset_episode(self, reset_kwargs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        reset_kwargs = dict(reset_kwargs or {})
        try:
            obs, info = unpack_reset(self.env.reset(**reset_kwargs))
        except TypeError:
            obs, info = unpack_reset(self.env.reset())
        self.observation = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.episode_reward = 0.0
        self.episode_steps = 0
        return self.observation

    def collect_rollout(self, n_steps: Optional[int] = None, policy: Any = None) -> SACRollout:
        """Collect ``n_steps`` environment transitions (SAC is off-policy)."""
        del policy  # this baseline always samples from its own SAC actor
        n_steps = int(n_steps or self.config.n_steps or self.horizon)
        rollout = SACRollout()
        if self.observation is None:
            self.reset_episode()
        for _ in range(n_steps):
            observation = self.observation
            deterministic = False
            if self.env_steps < int(self.config.learning_starts):
                rollout.n_random_actions += 1
            else:
                rollout.n_policy_actions += 1
            action = self._select_action(observation, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = unpack_step(self.env.step(action))
            done = bool(terminated or truncated)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            reward = float(reward) * float(self.config.reward_scale)

            rollout.observations.append(observation)
            rollout.actions.append(action)
            rollout.rewards.append(reward)
            rollout.next_observations.append(next_obs)
            rollout.dones.append(done)
            rollout.n_env_steps += 1
            self.buffer.add(observation, action, reward, next_obs, terminated)
            self.episode_reward += reward
            self.episode_steps += 1
            self.env_steps += 1

            if done:
                rollout.episode_rewards.append(self.episode_reward)
                self.episode_count += 1
                self.observation = self.reset_episode()
            else:
                self.observation = next_obs
        self.last_rollout = rollout
        return rollout

    # -- SAC updates -------------------------------------------------------
    def update(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Run ``train_freq * gradient_steps`` SAC gradient updates."""
        if self.buffer.size < max(int(self.config.batch_size), 1):
            return {"actor_loss": float("nan"), "critic_loss": float("nan"), "alpha_loss": float("nan")}
        updates = max(1, int(self.config.gradient_steps))
        logs: Dict[str, float] = {}
        for _ in range(updates):
            logs = self._update_step(batch=None)
        return logs

    def _to_tensor(self, array: np.ndarray) -> Any:
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device)

    def _update_step(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        if batch is None:
            batch = self.buffer.sample(int(self.config.batch_size))
        observations = self._to_tensor(batch["observations"])
        next_observations = self._to_tensor(batch["next_observations"])
        rewards = self._to_tensor(batch["rewards"]).reshape(-1, 1)
        dones = self._to_tensor(batch["dones"]).reshape(-1, 1)
        if self.discrete:
            actions = torch.as_tensor(np.asarray(batch["actions"], dtype=np.int64), device=self.device).reshape(-1, 1)
        else:
            actions = self._to_tensor(batch["actions"])

        # -- critic ------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_prob, _, _ = self.actor.sample(next_observations)
            target_q1, target_q2 = self.target_critic(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha_value * next_log_prob
            backup = rewards + float(self.config.gamma) * (1.0 - dones) * target_q
        q1, q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.max_grad_norm if hasattr(self.config, "max_grad_norm") else True:
            try:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
            except Exception:
                pass
        self.critic_optimizer.step()

        # -- actor -------------------------------------------------------
        new_actions, log_prob, _, _ = self.actor.sample(observations)
        q1_pi, q2_pi = self.critic(observations, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha_value * log_prob - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        except Exception:
            pass
        self.actor_optimizer.step()

        # -- temperature (SAC v2 automatic entropy tuning) ---------------
        alpha_loss_value = float("nan")
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().cpu().item())

        # -- target networks (soft update) -------------------------------
        self._soft_update(self.critic, self.target_critic, float(self.config.tau))

        with torch.no_grad():
            q_values = q_pi.mean()
            log_prob_mean = log_prob.mean()
        self.gradient_updates += 1
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha_value.detach().cpu().item()),
            "entropy": float(-log_prob_mean.detach().cpu().item()),
            "q_value": float(q_values.detach().cpu().item()),
        }

    @property
    def alpha_value(self) -> Any:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        coef = self.config.ent_coef
        value = float(coef) if not isinstance(coef, str) else 0.2
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    @property
    def alpha(self) -> float:
        return float(self.alpha_value.detach().cpu().item())

    def _soft_update(self, source: Any, target: Any, tau: float) -> None:
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)

    # -- training loop -----------------------------------------------------
    def train(
        self,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        logger: Any = None,
        progress: bool = False,
        callback: Any = None,
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Fine-tune with SAC for ``total_timesteps`` environment steps."""
        logger = logger or self.logger
        total_timesteps = int(total_timesteps or self.config.total_timesteps)
        n_steps = int(self.config.n_steps or self.horizon)
        if total_iterations is not None:
            total_timesteps = int(total_iterations) * n_steps
        n_iterations = max(1, int(math.ceil(total_timesteps / max(1, n_steps))))
        if self.observation is None:
            self.reset_episode()

        self.timer_start("train")
        collected = 0
        for iteration in range(n_iterations):
            rollout = self.collect_rollout(n_steps=n_steps)
            collected += rollout.n_env_steps
            # SAC updates every train_freq env steps (SB3 semantics).
            n_updates = max(1, int(np.ceil(max(1, rollout.n_env_steps) / max(1, int(self.config.train_freq)))))
            record: Dict[str, float] = {}
            for _ in range(n_updates):
                record = self.update()
            record = dict(record)
            record.update(
                {
                    "iteration": float(iteration),
                    "env_steps": float(self.env_steps),
                    "collected": float(collected),
                    "buffer_size": float(len(self.buffer)),
                    "episodes": float(self.episode_count),
                    "mean_episode_reward": rollout.mean_episode_reward,
                    "learning_rate": float(self.config.lr),
                }
            )
            self.history.append(record)
            should_log = (iteration % max(1, int(self.config.log_interval)) == 0) or iteration == n_iterations - 1
            if should_log and logger is not None:
                try:
                    logger.info(
                        "SAC fine-tuning iter %d/%d | env_steps=%d | mean_ep_reward=%.2f | "
                        "critic=%.3f actor=%.3f alpha=%.4f",
                        iteration + 1,
                        n_iterations,
                        self.env_steps,
                        record["mean_episode_reward"],
                        record.get("critic_loss", float("nan")),
                        record.get("actor_loss", float("nan")),
                        record.get("alpha", float("nan")),
                    )
                except Exception:
                    pass
            if callback is not None:
                try:
                    callback(self, iteration, record)
                except TypeError:
                    callback(record)
            if self.env_steps >= total_timesteps:
                break
        self.timer_end("train")
        return self.history

    fit = train

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        policy: Any = None,
        max_steps: Optional[int] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        del reset_kwargs
        if policy is not None and policy is not self.actor:
            result = evaluate_policy(
                self.env,
                policy,
                env_id=self.env_id,
                n_episodes=n_episodes,
                max_steps=max_steps,
                deterministic=deterministic,
                device=str(self.device),
            )
            self.eval_history.append(result)
            return result
        returns: List[float] = []
        steps_list: List[int] = []
        max_steps = int(max_steps or self.horizon)
        for _ in range(int(max(1, n_episodes))):
            obs = self.reset_episode()
            total = 0.0
            steps = 0
            for _ in range(max_steps):
                action = self._select_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = unpack_step(self.env.step(action))
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)
            steps_list.append(steps)
        self.observation = None
        result = {
            "mean_reward": float(np.mean(returns)) if returns else float("nan"),
            "std_reward": float(np.std(returns)) if returns else float("nan"),
            "mean_length": float(np.mean(steps_list)) if steps_list else float("nan"),
            "n_episodes": float(len(returns)),
        }
        self.eval_history.append(result)
        return result

    # -- persistence / introspection ---------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "baseline": "sac_finetune",
            "backend": "native",
            "env_id": self.env_id,
            "lr": float(self.config.lr),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": list(self.config.hidden_sizes),
            "total_time": self.total_time,
            "env_steps": float(self.env_steps),
            "gradient_updates": float(self.gradient_updates),
            "episodes": float(self.episode_count),
            "buffer": self.buffer.statistics(),
            "alpha": self.alpha,
            "target_entropy": self.target_entropy,
            "history": self.history[-1] if self.history else None,
            "final_eval": self.eval_history[-1] if self.eval_history else None,
            "sac_config": self.config.to_dict(),
        }

    def time_report(self) -> Dict[str, float]:
        return dict(self._timers)

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        _require_torch()
        directory = ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        del directory
        payload = {
            "kind": "sac_finetune",
            "env_id": self.env_id,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.config.hidden_sizes),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "config": self.config.to_dict(),
            "summary": self.summary(),
            "history": self.history,
            "eval_history": self.eval_history,
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
        self.logger.info("Saved SAC fine-tuning checkpoint to %s", path)
        return path

    def save_policy(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Save only the actor (the refined policy network)."""
        _require_torch()
        return self.save(path, extra={"actor_only": True, **(extra or {})})

    # -- exposing the refined policy ---------------------------------------
    @property
    def policy(self) -> Any:
        return self.actor

    @property
    def policy_network(self) -> Any:
        return self.actor

    @property
    def refined_policy(self) -> Any:
        return self.actor

    @classmethod
    def load(cls, path: str, env: Any = None, device: str = "cpu", **kwargs: Any) -> "SACFinetuner":
        _require_torch()
        payload = torch.load(path, map_location=device)
        trainer = cls(
            env,
            policy=None,
            config=SACFinetuneConfig.from_dict(payload.get("config", {})),
            env_id=payload.get("env_id", "default"),
            device=device,
            **kwargs,
        )
        trainer.actor.load_state_dict(payload["actor"])
        if "critic" in payload:
            trainer.critic.load_state_dict(payload["critic"])
        if "target_critic" in payload:
            trainer.target_critic.load_state_dict(payload["target_critic"])
        if payload.get("log_alpha") is not None and trainer.log_alpha is not None:
            trainer.log_alpha.data.copy_(payload["log_alpha"].to(trainer.device))
        trainer.history = list(payload.get("history", []))
        trainer.eval_history = list(payload.get("eval_history", []))
        return trainer


# --------------------------------------------------------------------------- #
# Functional entry points
# --------------------------------------------------------------------------- #
def make_sac_finetuner(
    env: Any,
    policy: Any = None,
    config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
    env_id: str = "default",
    seed: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Factory: dispatch to the SB3 or the native SAC fine-tuner.

    ``config`` may also be a string, which is then interpreted as ``env_id``
    (mirroring ``make_finetuner`` in ``rice.baselines.ppo_finetune``).
    """
    if isinstance(config, str):
        env_id = config
        config = None
    if is_sb3_sac(policy):
        return SB3SACFineTuner(
            env,
            policy,
            config=config,
            env_id=env_id,
            device=device,
            logger=logger,
            seed=seed,
            **kwargs,
        )
    return SACFinetuner(
        env,
        policy=policy,
        config=config,
        env_id=env_id,
        device=device,
        logger=logger,
        seed=seed,
        **kwargs,
    )


build_sac_finetuner = make_sac_finetuner
build_sac_finetune = make_sac_finetuner


def train_sac_agent(
    env: Any,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    env_id: str = "default",
    config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
    policy: Any = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    progress: bool = False,
    evaluate: bool = False,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Pre-train (or continue training) an SAC agent — Experiment IV's expert.

    The paper *"first obtain[s] a pre-trained SAC agent"*; this helper produces
    that agent so the pipeline ``SAC (pre-trained) -> GAIL approximation ->
    RICE refinement`` of Experiment IV can be assembled end-to-end.
    """
    trainer = make_sac_finetuner(
        env,
        policy=policy,
        config=config,
        env_id=env_id,
        seed=seed,
        device=device,
        logger=logger,
        total_timesteps=total_timesteps,
        **kwargs,
    )
    trainer.train(total_timesteps=total_timesteps, logger=logger, progress=progress)
    if evaluate:
        trainer.evaluate(n_episodes=eval_episodes)
    return trainer.policy, trainer


def sac_finetune_policy(
    env: Any,
    policy: Any = None,
    total_timesteps: Optional[int] = None,
    total_iterations: Optional[int] = None,
    env_id: str = "default",
    config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
    logger: Any = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress: bool = False,
    evaluate: bool = False,
    eval_episodes: int = 10,
    lr: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """"SAC fine-tuning" refining baseline (§4.2 Experiment IV).

    Parameters
    ----------
    env : gym environment (MuJoCo ``Hopper-v3`` etc.).
    policy : the pre-trained agent — an SB3 ``SAC`` model, a native
        ``SACGaussianActor``, or any torch policy network (e.g. the GAIL
        approximation ``pi_G`` used to transfer a SAC agent's behaviour).
    total_timesteps : SAC environment steps used for fine-tuning.

    Returns
    -------
    ``(refined_policy, finetuner)`` — the fine-tuned SAC actor and the trainer
    object (which also exposes ``evaluate``/``summary``/``save``).
    """
    finetuner = make_sac_finetuner(
        env,
        policy=policy,
        config=config,
        env_id=env_id,
        seed=seed,
        device=device,
        logger=logger,
        lr=lr,
        total_timesteps=total_timesteps,
        total_iterations=total_iterations,
        **kwargs,
    )
    finetuner.train(
        total_timesteps=total_timesteps,
        total_iterations=total_iterations,
        logger=logger,
        progress=progress,
    )
    if evaluate:
        finetuner.evaluate(n_episodes=eval_episodes, deterministic=True)
    if save_path is not None:
        try:
            finetuner.save_policy(save_path, extra={"env_id": env_id, "baseline": "sac_finetune"})
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("Could not save SAC fine-tuned policy (%s).", exc)
    return finetuner.policy, finetuner


# Aliases matching the naming style of the other baseline modules.
sac_finetune = sac_finetune_policy
train_sac_finetune = sac_finetune_policy
finetune_sac = sac_finetune_policy
SACFinetuneBaseline = SACFinetuner


def describe_sac_finetune(finetuner: Any = None) -> str:
    """One-line human-readable summary (logging/tables)."""
    if finetuner is None:
        return (
            "SAC fine-tuning baseline: continue SAC training from a pre-trained SAC agent "
            "(SB3/native), no mixed initial distribution, no RND bonus."
        )
    backend = "SB3" if isinstance(finetuner, SB3SACFineTuner) else "native"
    env_id = getattr(finetuner, "env_id", "default")
    return (
        f"SAC fine-tuning [{backend}] on '{env_id}': env_steps={getattr(finetuner, 'env_steps', 0)}, "
        f"updates={getattr(finetuner, 'gradient_updates', 'n/a')}, "
        f"mean_eval_reward={getattr(finetuner, 'eval_history', [])[-1].get('mean_reward', float('nan')) if getattr(finetuner, 'eval_history', None) else float('nan')}"
    )


def sac_for(env_id: str = "default", **kwargs: Any) -> SACFinetuneConfig:
    """Per-environment default configuration for the SAC fine-tuning baseline."""
    return SACFinetuneConfig.from_dict({"env_id": normalize_env_key(env_id)}, **kwargs)


# --------------------------------------------------------------------------- #
# RICE-compatible refining facade (Experiment IV composition)
# --------------------------------------------------------------------------- #
class SACRefiner:
    """SAC fine-tuning exposed with the refining-baseline interface.

    ``SACRefiner`` mirrors ``rice.baselines.gail.GAILRefiner`` /
    ``rice.baselines.jsrl.JSRLRefiner``: it can (a) train a plain SAC
    fine-tuning baseline, and (b) chain the resulting policy into RICE's
    Stage-2 PPO refinement (``rice.refining.ppo_refine.refine_policy``) using
    the *same* mask-network explanation as RICE — as required by §4.2
    (*"all the refining methods use the same explanation generated by our
    explanation method if needed"*).
    """

    name = "SAC fine-tuning"

    def __init__(
        self,
        env: Any = None,
        sac: Any = None,
        policy: Any = None,
        config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.env = env
        self.env_id = normalize_env_key(env_id)
        self.device = device
        self.logger = logger or _LOGGER
        self.config = (
            config if isinstance(config, SACFinetuneConfig) else SACFinetuneConfig.from_dict(config if isinstance(config, dict) else None)
        )
        self.seed = seed if seed is not None else self.config.seed
        self.sac = sac if sac is not None else policy
        self.finetuner: Optional[Any] = None
        self.refiner: Optional[Any] = None
        self.refined_policy: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []
        self.info: Dict[str, Any] = {}

    # -- Stage 2 (baseline) -------------------------------------------------
    def approximate_policy(
        self,
        env: Any = None,
        total_timesteps: Optional[int] = None,
        progress: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the SAC fine-tuning baseline and return its policy."""
        env = env if env is not None else self.env
        policy, finetuner = sac_finetune_policy(
            env,
            policy=self.sac,
            total_timesteps=total_timesteps or self.config.total_timesteps,
            env_id=self.env_id,
            config=self.config,
            logger=self.logger,
            seed=self.seed,
            device=self.device,
            progress=progress,
            **kwargs,
        )
        self.finetuner = finetuner
        self.sac = policy
        return policy

    @property
    def policy_network(self) -> Any:
        return self.sac

    @property
    def approximated_policy(self) -> Any:
        return self.sac

    # -- optional RICE Stage-2 refinement ----------------------------------
    def refine(
        self,
        env: Any = None,
        mask_net: Any = None,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Refine the SAC-fine-tuned policy with RICE's Stage-2 PPO engine."""
        if not _HAS_REFINER:  # pragma: no cover
            raise ImportError("RICE refinement (rice.refining.ppo_refine) is unavailable.")
        env = env if env is not None else self.env
        policy, refiner = refine_policy(
            env,
            policy=self.sac,
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
        self.refiner = refiner
        self.refined_policy = policy
        return policy, refiner

    def run(
        self,
        env: Any = None,
        sac: Any = None,
        mask_net: Any = None,
        finetune_timesteps: Optional[int] = None,
        refine_timesteps: Optional[int] = None,
        refine_iterations: Optional[int] = None,
        refine_config: Optional[Dict[str, Any]] = None,
        evaluate: bool = True,
        eval_episodes: int = 10,
        save_dir: Optional[str] = None,
        progress: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Full Experiment-IV entry point for this baseline.

        ``pre-trained SAC -> SAC fine-tuning (baseline) -> [optional RICE
        Stage-2 refinement with the shared explanation] -> evaluation``.
        """
        env = env if env is not None else self.env
        if sac is not None:
            self.sac = sac
        policy = self.approximate_policy(
            env=env, total_timesteps=finetune_timesteps, progress=progress, **kwargs
        )
        result: Dict[str, Any] = {
            "baseline": self.name,
            "env_id": self.env_id,
            "sac_finetune_summary": self.finetuner.summary() if self.finetuner is not None else None,
        }
        if evaluate:
            try:
                result["eval_sac_finetune"] = self.finetuner.evaluate(
                    n_episodes=eval_episodes, deterministic=True
                )
            except Exception as exc:  # pragma: no cover
                result["eval_sac_finetune"] = {"error": str(exc)}
        if refine_timesteps or refine_iterations:
            refined, refiner = self.refine(
                env=env,
                mask_net=mask_net,
                total_timesteps=refine_timesteps,
                total_iterations=refine_iterations,
                config=refine_config,
                progress=progress,
            )
            result["refine_summary"] = refiner.summary() if hasattr(refiner, "summary") else None
            if evaluate:
                result["eval_refined"] = evaluate_policy(
                    env,
                    refined,
                    env_id=self.env_id,
                    n_episodes=eval_episodes,
                    deterministic=True,
                    device=self.device,
                )
        if save_dir is not None:
            try:
                ensure_dir(save_dir)
                path = os.path.join(save_dir, f"sac_finetune_{self.env_id}.pt")
                self.finetuner.save(path)
                result["checkpoint"] = path
            except Exception as exc:  # pragma: no cover
                result["save_error"] = str(exc)
        self.info = result
        self.history.append(result)
        return result

    def summary(self) -> Dict[str, Any]:
        return {
            "baseline": self.name,
            "env_id": self.env_id,
            "finetuner": self.finetuner.summary() if self.finetuner is not None else None,
            "refiner": (self.refiner.summary() if self.refiner is not None and hasattr(self.refiner, "summary") else None),
        }

    def describe(self) -> str:
        return describe_sac_finetune(self.finetuner)


def make_sac_refiner(
    env: Any = None,
    sac: Any = None,
    policy: Any = None,
    env_id: str = "default",
    config: Optional[Union[SACFinetuneConfig, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    **kwargs: Any,
) -> SACRefiner:
    return SACRefiner(
        env=env,
        sac=sac,
        policy=policy,
        config=config,
        env_id=env_id,
        seed=seed,
        device=device,
        logger=logger,
        **kwargs,
    )


build_sac_refiner = make_sac_refiner


__all__ = [
    "SACFinetuneConfig",
    "SACReplayBuffer",
    "SACRollout",
    "SACGaussianActor",
    "SACCritic",
    "SACTemperature",
    "SACFinetuner",
    "SB3SACFineTuner",
    "SACRefiner",
    "SACFinetuneBaseline",
    "make_sac_finetuner",
    "build_sac_finetuner",
    "build_sac_finetune",
    "make_sac_refiner",
    "build_sac_refiner",
    "sac_finetune_policy",
    "sac_finetune",
    "train_sac_finetune",
    "finetune_sac",
    "train_sac_agent",
    "describe_sac_finetune",
    "sac_for",
    "is_sb3_sac",
    "evaluate_policy",
    "rollout_episode",
    "policy_action",
    "resolve_obs_act_dims",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_LEARNING_STARTS",
    "DEFAULT_TAU",
    "DEFAULT_GAMMA_SAC",
    "DEFAULT_TRAIN_FREQ",
    "DEFAULT_GRADIENT_STEPS",
    "DEFAULT_ENT_COEF",
    "DEFAULT_TOTAL_TIMESTEPS",
    "DEFAULT_FINETUNE_LR",
]
