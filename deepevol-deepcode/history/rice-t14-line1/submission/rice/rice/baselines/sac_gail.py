"""SAC pre-training + GAIL imitation baseline for RICE (Experiment IV).

Paper reference
---------------
RICE, Proc. 41st ICML (PMLR 235), 2024.

* §4.2 Experiment IV (verbatim): "To show the versatility of our method, we examine the
  refining performance when the pre-trained agent was trained by other algorithms such as
  Soft Actor-Critic (SAC) (Haarnoja et al., 2018). First, we obtain a pre-trained SAC agent
  and then use Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) to learn
  an approximated policy network. We compare the refining performance using our method
  against baseline methods, i.e., PPO fine-tuning (Schulman et al., 2017), StateMask's
  fine-tuning from critical steps (Cheng et al., 2023), and Jump-Start Reinforcement Learning
  (Uchendu et al., 2023). In addition, we also include fine-tuning the pre-trained SAC agent
  with the SAC algorithm as a baseline."
* §4.1 (baseline refining methods, verbatim): "The first baseline is 'PPO fine-tuning'
  (Schulman et al., 2017), i.e., lowering the learning rate and continuing training with the
  PPO algorithm. The second baseline is a refining method introduced by StateMask
  (Cheng et al., 2023), i.e., resetting to the critical state and continuing training from
  the critical state. The third baseline is Jump-Start Reinforcement Learning (referred to as
  'JSRL') (Uchendu et al., 2023)."
* §4.3 (takeaway, verbatim): "we observe that our refining method can bring the largest
  improvement for the target agent in all applications."

Design
------
The pipeline has three stages:

1. ``pretrain_sac()``   – warm-start a SAC agent (Haarnoja et al. 2018) on the task.  The
  agent is *bottlenecked* (local optimum) in the sense of the paper; its replay buffer is
  kept because it doubles as the expert demonstration set for GAIL.
2. ``imitate_with_gail()`` – learn an *approximated policy network* from the SAC expert using
  Generative Adversarial Imitation Learning (Ho & Ermon 2016).  The approximated policy is a
  ``rice.algorithms.ppo.ActorCritic`` Gaussian policy so that it can be handed, unchanged, to
  ``rice.algorithms.refine.RICERefiner`` (and to the PPO / StateMask-R / JSRL baselines).
  GAIL maximizes ``E[log D(s, a)]`` for expert data and ``E[log(1 - D(s, a))]`` for policy
  data, and the policy is optimized with the shared PPO clipped surrogate on the
  discriminator-style external reward ``r(s, a) = -log(1 - D(s, a))``.
3. ``refine()``          – compare refining methods: ours (RICE), PPO fine-tuning, StateMask-R,
  JSRL and SAC fine-tuning (continuing the pre-trained SAC agent).

All of the RICE-specific components are reached through the same entry points that the other
baselines use (``RICERefiner`` / ``RefineConfig`` / ``evaluate_policy``), so that RICE vs.
baseline differences come *only* from the refining strategy (mixed initial state distribution
``mu(s) = beta d_rho^{pi_hat}(s) + (1 - beta) rho(s)`` with ``beta = p`` and the RND bonus
``R + lambda ||f(s') - f_hat(s')||^2``).

Deviations from the paper (unspecified details, documented for the README):
* SAC hyper-parameters follow the SAC defaults (Haarnoja et al. 2018) / Stable-Baselines3
  ``SAC`` defaults; the paper does not specify them.
* GAIL hyper-parameters (discriminator width, gradient penalty, reward normalization,
  imitation budget) are not specified either; sensible defaults are used.
* ``K = 10%`` of the episode length is the plan's default roll-in length for critical-state
  identification (the paper only specifies ``K`` for the fidelity window in Experiment I).
"""

from __future__ import annotations

import copy
import importlib
import os
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# torch (optional at import time so that `import rice` never hard-fails)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:  # pragma: no cover
    _ModuleBase = nn.Module
else:  # pragma: no cover
    _ModuleBase = type("_DummyModule", (object,), {"__init__": lambda self, *a, **k: None})


# ---------------------------------------------------------------------------
# tolerant imports (the repository can be launched from several sys.path roots)
# ---------------------------------------------------------------------------
def _import_first(module_names: Sequence[str]) -> Optional[Any]:
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _get(module: Optional[Any], name: str, default: Any = None) -> Any:
    if module is None:
        return default
    return getattr(module, name, default)


_PPO_MODULE = _import_first(
    ["rice.algorithms.ppo", "rice.rice.algorithms.ppo", "algorithms.ppo"]
)
_REFINE_MODULE = _import_first(
    ["rice.algorithms.refine", "rice.rice.algorithms.refine", "algorithms.refine"]
)
_ENVS_MODULE = _import_first(["rice.environments", "rice.rice.environments"])
_RND_MODULE = _import_first(["rice.algorithms.rnd", "rice.rice.algorithms.rnd"])

PPOConfig = _get(_PPO_MODULE, "PPOConfig")
ActorCritic = _get(_PPO_MODULE, "ActorCritic")
PPO = _get(_PPO_MODULE, "PPO")
RolloutBuffer = _get(_PPO_MODULE, "RolloutBuffer")
flatten_obs = _get(_PPO_MODULE, "flatten_obs", lambda x: np.asarray(x, dtype=np.float32).reshape(-1))
observation_size = _get(_PPO_MODULE, "observation_size", None)
action_size = _get(_PPO_MODULE, "action_size", None)
resolve_device = _get(_PPO_MODULE, "resolve_device", lambda d="auto": "cpu")
make_target_policy_callable = _get(_PPO_MODULE, "make_target_policy_callable", None)

RefineConfig = _get(_REFINE_MODULE, "RefineConfig")
RefineResult = _get(_REFINE_MODULE, "RefineResult")
refine_policy = _get(_REFINE_MODULE, "refine_policy")
evaluate_policy = _get(_REFINE_MODULE, "evaluate_policy")
load_policy_weights = _get(_REFINE_MODULE, "load_policy_weights")
RICERefiner = _get(_REFINE_MODULE, "RICERefiner")

make_env = _get(_ENVS_MODULE, "make_env")
default_net_arch = _get(_ENVS_MODULE, "default_net_arch")

RunningMeanStd = _get(_RND_MODULE, "RunningMeanStd")


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
METHOD_NAME = "sac_gail"
METHOD_ALIASES: Tuple[str, ...] = (
    "sac_gail",
    "sac-gail",
    "sacgail",
    "sac_plus_gail",
    "gail",
)

#: Experiment IV runs on Hopper (§4.2 / Figure 3); Table 3 hyper-parameters for Hopper.
DEFAULT_TASK = "Hopper-v3"
DEFAULT_PRETRAIN_STEPS = 1_000_000
DEFAULT_GAIL_STEPS = 300_000

#: Table 3 (verbatim) for Hopper: {p: 0.25, lambda: 0.001, alpha: 0.0001}
TABLE3_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SelfishMining": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "CageChallenge2": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "Macro-v1": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
}

#: Reference trend values for the Hopper row of Table 1 (used for trend checks only).
TABLE1_HOPPER_REFERENCE: Dict[str, float] = {
    "no_refine": 3559.44,
    "ppo": 3561.0,
    "jsrl": 3621.0,
    "statemask_r": 3627.0,
    "ours": 3663.91,
}

SAC_METHODS: Tuple[str, ...] = ("ours", "ppo", "statemask_r", "jsrl", "sac")


def sac_enabled() -> bool:
    """Experiment IV is gated behind ``RICE_ENABLE_SAC=1`` (heavy SAC/GAIL pre-training)."""
    return str(os.environ.get("RICE_ENABLE_SAC", "0")).lower() in {"1", "true", "yes", "on"}


# ===========================================================================
# small numerical helpers
# ===========================================================================
def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _obs_vector(obs: Any) -> np.ndarray:
    try:
        return np.asarray(flatten_obs(obs), dtype=np.float32).reshape(-1)
    except Exception:
        return np.asarray(obs, dtype=np.float32).reshape(-1)


def _box_dims(space: Any) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(dim, low, high)`` for a Box space (low/high may be None)."""
    shape = getattr(space, "shape", None)
    if shape is None:
        dim = int(np.prod(getattr(space, "shape", (1,))))
    else:
        dim = int(np.prod(shape))
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is not None:
        low = np.asarray(low, dtype=np.float32).reshape(-1)
    if high is not None:
        high = np.asarray(high, dtype=np.float32).reshape(-1)
    return dim, low, high


def _env_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    """gym / gymnasium tolerant ``reset``."""
    if seed is not None:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            try:
                if hasattr(env, "seed"):
                    env.seed(seed)
            except Exception:
                pass
            out = env.reset()
    else:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], (out[1] or {})
    return out, {}


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalize ``step`` to ``(obs, reward, terminated, truncated, info)``."""
    out = env.step(action)
    if not isinstance(out, tuple):  # pragma: no cover - exotic env
        return out, 0.0, False, False, {}
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, float(rew), bool(term), bool(trunc), dict(info or {})
    obs, rew, done, info = out  # type: ignore[misc]
    return obs, float(rew), bool(done), False, dict(info or {})


def _env_episode_length(env: Any, default: int = 1000) -> int:
    probe = env
    for _ in range(10):
        if probe is None:
            break
        for attr in ("max_episode_steps", "_max_episode_steps"):
            val = getattr(probe, attr, None)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
        spec = getattr(probe, "spec", None)
        if spec is not None and getattr(spec, "max_episode_steps", None):
            return int(spec.max_episode_steps)
        probe = getattr(probe, "env", None)
    return int(default)


def _mlp(
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> Any:
    """Plain MLP builder (torch required)."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("sac_gail requires PyTorch.")
    acts = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
    }
    act_cls = acts.get(str(activation).lower(), nn.ReLU)
    layers: List[Any] = []
    last = int(input_dim)
    for h in hidden:
        layers.append(nn.Linear(last, int(h)))
        layers.append(act_cls())
        last = int(h)
    layers.append(nn.Linear(last, int(output_dim)))
    if output_activation is not None:
        out_act = acts.get(str(output_activation).lower(), None)
        if out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


def _soft_update(target: Any, source: Any, tau: float) -> None:
    with torch.no_grad():  # pragma: no cover - torch required
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


# ===========================================================================
# SAC (Haarnoja et al. 2018) — soft actor-critic, self-contained
# ===========================================================================
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(_ModuleBase):
    """tanh-squashed Gaussian policy (SAC actor)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        net_arch: Sequence[int] = (256, 256),
        activation: str = "relu",
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__() if _TORCH_AVAILABLE else None
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.trunk = _mlp(self.obs_dim, net_arch, int(net_arch[-1]) if len(net_arch) else 64, activation)
        last = int(net_arch[-1]) if len(net_arch) else 64
        self.mean_layer = nn.Linear(last, self.act_dim)
        self.log_std_layer = nn.Linear(last, self.act_dim)
        if action_low is None:
            action_low = -np.ones(self.act_dim, dtype=np.float32)
        if action_high is None:
            action_high = np.ones(self.act_dim, dtype=np.float32)
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        # map the tanh output [-1, 1] onto the (possibly unbounded) action range
        finite = np.isfinite(low) & np.isfinite(high)
        scale = np.ones_like(low)
        bias = np.zeros_like(low)
        scale[finite] = (high[finite] - low[finite]) / 2.0
        bias[finite] = (high[finite] + low[finite]) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))
        self._finite_mask = finite

    # -- forward ---------------------------------------------------------
    def forward(self, obs: Any) -> Tuple[Any, Any]:
        h = self.trunk(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: Any, deterministic: bool = False, with_log_prob: bool = True):
        mean, log_std = self.forward(obs)
        if deterministic:
            u = torch.tanh(mean)
            action = u * self.action_scale + self.action_bias
            log_prob = torch.zeros(obs.shape[0], device=mean.device)
            return action, log_prob
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        u = torch.tanh(x)
        action = u * self.action_scale + self.action_bias
        log_prob = None
        if with_log_prob:
            # change of variables (Haarnoja et al. 2018, Appendix C)
            log_prob = dist.log_prob(x) - torch.log(1.0 - u.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, obs: Any) -> Any:
        with torch.no_grad():
            action, _ = self.sample(obs, deterministic=True)
        return action


class SoftQNetwork(_ModuleBase):
    """Twin soft Q-value network ``Q(s, a)``."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        net_arch: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__() if _TORCH_AVAILABLE else None
        self.net = _mlp(int(obs_dim) + int(act_dim), net_arch, 1, activation)

    def forward(self, obs: Any, action: Any) -> Any:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


@dataclass
class SACConfig:
    """SAC hyper-parameters (Haarnoja et al. 2018 / SB3 ``SAC`` defaults).

    The paper does not specify SAC hyper-parameters (§4.2 only says a pre-trained SAC agent
    is obtained), so the values below follow the reference implementation defaults.
    """

    task: str = DEFAULT_TASK
    total_timesteps: int = DEFAULT_PRETRAIN_STEPS
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: Any = "auto"          # "auto" | float
    target_entropy: Any = "auto"    # "auto" | float
    learning_starts: int = 10_000
    gradient_steps: int = 1
    net_arch: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    normalize_obs: bool = False
    train_freq: int = 1
    n_eval_episodes: int = 5
    eval_every: int = 25_000
    seeds: Tuple[int, ...] = (0, 1, 2)
    seed: Optional[int] = None
    device: str = "auto"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    verbose: int = 1
    log_every: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self, **overrides: Any) -> "SACConfig":
        data = asdict(self)
        data.update(overrides)
        known = {f for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = dict(data.pop("extra", {}) or {})
        for key in list(data):
            if key not in known:
                extra[key] = data.pop(key)
        cfg = SACConfig(**data)
        cfg.extra = extra
        return cfg

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "SACConfig":
        cfg = cls()
        if mapping:
            merged = dict(mapping)
            if "lambda" in merged:
                merged.setdefault("extra", {})["lambda"] = merged.pop("lambda")
            cfg = cfg.clone(**merged)
        if overrides:
            cfg = cfg.clone(**overrides)
        return cfg


class ReplayBuffer:
    """Simple ring buffer of ``(obs, action, reward, next_obs, done)`` transitions."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, int(obs_dim)), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, int(obs_dim)), dtype=np.float32)
        self.actions = np.zeros((self.capacity, int(act_dim)), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        i = self.ptr
        self.obs[i] = _obs_vector(obs)
        self.next_obs[i] = _obs_vector(next_obs)
        self.actions[i] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, transitions: Iterable[Tuple[Any, Any, float, Any, bool]]) -> None:
        for t in transitions:
            self.add(*t)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done": self.dones[idx],
        }

    def all_transitions(self) -> Dict[str, np.ndarray]:
        n = self.size
        idx = np.arange(n)
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size


class SACAgent:
    """Self-contained SAC agent used to produce the pre-trained (bottlenecked) expert."""

    def __init__(
        self,
        env: Any = None,
        config: Optional[Any] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("SACAgent requires PyTorch.")
        if isinstance(config, SACConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SACConfig.from_mapping(config)
        else:
            self.config = SACConfig()
        if kwargs:
            self.config = self.config.clone(**kwargs)

        self.env = env
        if env is not None:
            obs_space = getattr(env, "observation_space", None)
            act_space = getattr(env, "action_space", None)
            if obs_dim is None and obs_space is not None:
                obs_dim = int(np.prod(getattr(obs_space, "shape", (obs_space.shape if hasattr(obs_space, "shape") else 1,))))
            if act_dim is None and act_space is not None:
                act_dim = int(np.prod(getattr(act_space, "shape", (1,))))
            if action_low is None and act_space is not None and hasattr(act_space, "low"):
                action_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
            if action_high is None and act_space is not None and hasattr(act_space, "high"):
                action_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)

        self.obs_dim = int(obs_dim if obs_dim is not None else 11)
        self.act_dim = int(act_dim if act_dim is not None else 3)
        self.action_low = action_low
        self.action_high = action_high

        if isinstance(device, str):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = torch.device(str(device))

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        arch = tuple(self.config.net_arch)
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.act_dim, arch, self.config.activation, action_low, action_high
        ).to(self.device)
        self.q1 = SoftQNetwork(self.obs_dim, self.act_dim, arch, self.config.activation).to(self.device)
        self.q2 = SoftQNetwork(self.obs_dim, self.act_dim, arch, self.config.activation).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.config.learning_rate
        )

        if str(self.config.ent_coef) == "auto":
            self.target_entropy = (
                -float(self.act_dim)
                if str(self.config.target_entropy) == "auto"
                else float(self.config.target_entropy)
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
            self.autotune_alpha = True
        else:
            self.target_entropy = (
                -float(self.act_dim)
                if str(self.config.target_entropy) == "auto"
                else float(self.config.target_entropy)
            )
            self.log_alpha = torch.tensor(
                [float(np.log(max(float(self.config.ent_coef), 1e-8)))], device=self.device
            )
            self.alpha_optimizer = None
            self.autotune_alpha = False

        self.buffer = ReplayBuffer(self.config.buffer_size, self.obs_dim, self.act_dim)
        self.env_steps = 0
        self.gradient_updates = 0
        self.episode_returns: List[float] = []
        self.log: List[Dict[str, float]] = []

    # -- property --------------------------------------------------------
    @property
    def alpha(self) -> Any:
        return self.log_alpha.detach().exp()

    @property
    def policy(self) -> "SACAgent":
        """Self-reference so ``make_target_policy_callable`` can wrap the agent."""
        return self

    # -- acting ----------------------------------------------------------
    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(_obs_vector(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(x, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def predict(self, obs: Any, deterministic: bool = True, **_: Any) -> np.ndarray:
        """SB3-like ``predict`` (returns the action only)."""
        return self.act(obs, deterministic=deterministic)

    def policy_callable(self, deterministic: bool = False) -> Callable[[Any], np.ndarray]:
        return lambda obs: self.act(obs, deterministic=deterministic)

    # -- replay / updates ------------------------------------------------
    def store(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self, batch: Optional[Dict[str, np.ndarray]] = None,
               batch_size: Optional[int] = None) -> Dict[str, float]:
        if len(self.buffer) < max(int(self.config.batch_size), 1):
            return {}
        if batch is None:
            batch = self.buffer.sample(batch_size or self.config.batch_size, self.rng)

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1.0 - done) * self.config.gamma * q_next

        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_action, log_prob = self.actor.sample(obs, with_log_prob=True)
        q1_new = self.q1(obs, new_action)
        q2_new = self.q2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_val = 0.0
        if self.autotune_alpha and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = float(alpha_loss.item())

        _soft_update(self.q1_target, self.q1, self.config.tau)
        _soft_update(self.q2_target, self.q2, self.config.tau)
        self.gradient_updates += 1

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_val,
            "alpha": float(self.alpha.item()),
        }

    # -- training --------------------------------------------------------
    def pretrain(
        self,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        learning_starts: Optional[int] = None,
        eval_env: Any = None,
        eval_callback: Optional[Callable[[int, "SACAgent"], None]] = None,
        progress: bool = False,
    ) -> Dict[str, Any]:
        """Run SAC until ``total_timesteps`` (or the configured budget)."""
        if self.env is None:
            raise ValueError("SACAgent.pretrain requires an environment.")
        budget = int(total_timesteps if total_timesteps is not None else self.config.total_timesteps)
        start_learning = int(learning_starts if learning_starts is not None
                             else self.config.learning_starts)
        t0 = time.time()
        obs, _ = _env_reset(self.env, seed=seed)
        ep_return = 0.0
        ep_len = 0
        while self.env_steps < budget:
            if self.env_steps < start_learning:
                action = np.asarray(self.env.action_space.sample(), dtype=np.float32)
            else:
                action = self.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = _env_step(self.env, action)
            done = bool(terminated or truncated)
            self.store(obs, action, reward, next_obs, float(terminated))
            self.env_steps += 1
            ep_return += float(reward)
            ep_len += 1
            obs = next_obs
            if done:
                self.episode_returns.append(ep_return)
                obs, _ = _env_reset(self.env)
                ep_return, ep_len = 0.0, 0
            if self.env_steps >= start_learning:
                for _ in range(int(self.config.gradient_steps)):
                    self.update()
            if eval_callback is not None and self.config.eval_every > 0 and \
                    self.env_steps % int(self.config.eval_every) == 0:
                eval_callback(self.env_steps, self)

        metrics = {
            "env_steps": self.env_steps,
            "seconds": time.time() - t0,
            "episodes": len(self.episode_returns),
            "mean_episode_return": float(np.mean(self.episode_returns[-20:]))
            if self.episode_returns else float("nan"),
            "final_episode_return": float(self.episode_returns[-1]) if self.episode_returns else float("nan"),
        }
        if eval_env is not None:
            metrics["eval_reward"] = self.evaluate(eval_env)
        return metrics

    # -- evaluation / io -------------------------------------------------
    def evaluate(self, env: Any, n_episodes: int = 5, seed: Optional[int] = None,
                 deterministic: bool = True, max_steps: Optional[int] = None) -> Dict[str, float]:
        returns: List[float] = []
        lengths: List[int] = []
        for i in range(int(n_episodes)):
            obs, _ = _env_reset(env, seed=None if seed is None else seed + i)
            done = False
            ep_ret, ep_len = 0.0, 0
            limit = max_steps or _env_episode_length(env)
            while not done and ep_len < limit:
                action = self.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = _env_step(env, action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1
            returns.append(ep_ret)
            lengths.append(ep_len)
        arr = np.asarray(returns, dtype=np.float64)
        return {
            "mean_return": float(arr.mean()) if arr.size else float("nan"),
            "std_return": float(arr.std()) if arr.size else float("nan"),
            "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        }

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "config": self.config.to_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "env_steps": self.env_steps,
        }
        if self.alpha_optimizer is not None:
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any], load_optimizers: bool = True) -> "SACAgent":
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state.get("q1_target", state["q1"]))
        self.q2_target.load_state_dict(state.get("q2_target", state["q2"]))
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device).reshape(1))
        if load_optimizers:
            try:
                self.actor_optimizer.load_state_dict(state["actor_optimizer"])
                self.q_optimizer.load_state_dict(state["q_optimizer"])
                if self.alpha_optimizer is not None and "alpha_optimizer" in state:
                    self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            except Exception:
                pass
        self.env_steps = int(state.get("env_steps", self.env_steps))
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, map_location: Optional[str] = None) -> "SACAgent":
        state = torch.load(path, map_location=map_location or self.device)
        return self.load_state_dict(state)


class SACFineTuner:
    """SAC fine-tuning baseline: keep training the pre-trained SAC agent (§4.2 Experiment IV)."""

    def __init__(
        self,
        env: Any = None,
        agent: Optional[SACAgent] = None,
        config: Optional[Any] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, SACGAILConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SACGAILConfig.from_mapping(config)
        else:
            self.config = SACGAILConfig()
        if kwargs:
            self.config = self.config.clone(**kwargs)
        self.env = env
        if agent is None:
            agent = SACAgent(env=env, config=self.config.sac_config().clone(seed=seed), seed=seed)
        self.agent = agent
        self.seed = seed
        self.notes: List[str] = []

    def baseline_reward(self, n_episodes: Optional[int] = None) -> float:
        if self.env is None:
            return float("nan")
        return self.agent.evaluate(
            self.env, n_episodes or self.config.n_eval_episodes, seed=self.seed
        )["mean_return"]

    def refine(self, seed: Optional[int] = None, n_iterations: Optional[int] = None) -> Any:
        if self.env is None:
            raise ValueError("SACFineTuner.refine requires an environment.")
        seed = self.seed if seed is None else seed
        t0 = time.time()
        before = self.baseline_reward()
        curve: List[Tuple[int, float]] = []
        n_iters = int(n_iterations if n_iterations is not None else self.config.sac_finetune_iterations)
        steps_per_iter = int(self.config.sac_finetune_steps_per_iter)
        obs, _ = _env_reset(self.env, seed=seed)
        ep_return = 0.0
        total = 0
        for it in range(n_iters):
            for _ in range(steps_per_iter):
                action = self.agent.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = _env_step(self.env, action)
                done = bool(terminated or truncated)
                self.agent.store(obs, action, reward, next_obs, float(terminated))
                self.agent.env_steps += 1
                total += 1
                ep_return += float(reward)
                obs = next_obs
                if done:
                    self.agent.episode_returns.append(ep_return)
                    obs, _ = _env_reset(self.env)
                    ep_return = 0.0
                for _ in range(int(self.config.sac_config().gradient_steps)):
                    self.agent.update()
            eval_info = self.agent.evaluate(self.env, self.config.n_eval_episodes, seed=seed)
            curve.append((total, float(eval_info["mean_return"])))
            self.agent.log.append({"env_steps": total, **eval_info})
        after = curve[-1][1] if curve else before
        return _make_refine_result(
            task=self.config.task,
            method="sac",
            explanation="sac_gail",
            baseline_reward=before,
            final_reward=after,
            curve=curve,
            env_steps=total,
            seconds=time.time() - t0,
            seed=seed,
            notes=["SAC fine-tuning baseline (§4.2 Experiment IV)."],
        )


# ===========================================================================
# GAIL (Ho & Ermon 2016)
# ===========================================================================
class GAILDiscriminator(_ModuleBase):
    """Discriminator ``D(s, a) -> P(expert)`` with a sigmoid output head."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        net_arch: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__() if _TORCH_AVAILABLE else None
        self.net = _mlp(int(obs_dim) + int(act_dim), net_arch, 1, activation, output_activation="sigmoid")

    def forward(self, obs: Any, action: Any) -> Any:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    def prob_expert(self, obs: Any, action: Any) -> Any:
        return self.forward(obs, action)

    def logits(self, obs: Any, action: Any) -> Any:
        probs = torch.clamp(self.forward(obs, action), 1e-6, 1.0 - 1e-6)
        return torch.log(probs) - torch.log1p(-probs)

    @staticmethod
    def grad_penalty(disc: "GAILDiscriminator", expert_obs: Any, expert_act: Any,
                     policy_obs: Any, policy_act: Any, coef: float = 10.0) -> Any:
        """WGAN-GP style gradient penalty (Gulrajani et al. 2017); optional regularizer."""
        if coef <= 0:
            return torch.zeros((), device=expert_obs.device)
        alpha = torch.rand(expert_obs.size(0), 1, device=expert_obs.device)
        inter_obs = (alpha * expert_obs + (1 - alpha) * policy_obs).requires_grad_(True)
        inter_act = (alpha * expert_act + (1 - alpha) * policy_act).requires_grad_(True)
        scores = disc(inter_obs, inter_act)
        grads = torch.autograd.grad(
            outputs=scores.sum(), inputs=[inter_obs, inter_act],
            create_graph=True, retain_graph=True, allow_unused=True,
        )
        norm = torch.zeros_like(scores)
        for g in grads:
            if g is not None:
                norm = norm + g.reshape(g.size(0), -1).pow(2).sum(dim=1, keepdim=True)
        return coef * ((norm.sqrt() - 1.0) ** 2).mean()


@dataclass
class GAILConfig:
    """GAIL hyper-parameters (unspecified in the paper; documented defaults)."""

    total_timesteps: int = DEFAULT_GAIL_STEPS
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    disc_learning_rate: float = 3e-4
    disc_net_arch: Tuple[int, ...] = (256, 256)
    disc_activation: str = "relu"
    disc_updates_per_iter: int = 1
    disc_batch_size: int = 256
    grad_penalty_coef: float = 0.0
    reward_clip: float = 10.0
    normalize_reward: bool = True
    net_arch: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    eval_every: int = 0
    n_eval_episodes: int = 5
    device: str = "auto"
    seed: Optional[int] = None
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self, **overrides: Any) -> "GAILConfig":
        data = asdict(self)
        data.update(overrides)
        known = {f for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: data.pop(k) for k in list(data) if k not in known}
        cfg = GAILConfig(**data)
        if extra:
            setattr(cfg, "extra", extra)
        return cfg


class GAILTrainer:
    """Learn an approximated policy network from SAC expert transitions.

    The policy is an ``ActorCritic`` (the same network class RICE uses for refining) trained
    with the shared PPO clipped surrogate on the GAIL external reward
    ``r(s, a) = -log(1 - D(s, a))``.
    """

    def __init__(
        self,
        env: Any,
        expert: Any,
        config: Optional[Any] = None,
        policy: Optional[Any] = None,
        expert_buffer: Optional[ReplayBuffer] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("GAILTrainer requires PyTorch.")
        if isinstance(config, GAILConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = GAILConfig(**{k: v for k, v in config.items()
                                        if k in GAILConfig.__dataclass_fields__})
        else:
            self.config = GAILConfig()
        if kwargs:
            self.config = self.config.clone(**kwargs)

        self.env = env
        self.expert = expert
        self.seed = seed
        self.rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if str(device) == "auto"
            else torch.device(str(device))
        )

        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        self.obs_dim = int(np.prod(getattr(obs_space, "shape", (expert.obs_dim if expert is not None else 11,))))
        self.act_dim = int(np.prod(getattr(act_space, "shape", (expert.act_dim if expert is not None else 3,))))

        # policy: reuse RICE's ActorCritic so the imitation output is RICE-ready
        if policy is not None:
            self.policy = policy
        elif ActorCritic is not None:
            self.policy = ActorCritic(
                observation_space=obs_space,
                action_space=act_space,
                net_arch=tuple(self.config.net_arch),
                activation=self.config.activation,
                obs_dim=self.obs_dim,
            )
        else:  # pragma: no cover - only if algorithms.ppo is missing
            raise ImportError("GAILTrainer requires rice.algorithms.ppo.ActorCritic.")

        if PPOConfig is not None and PPO is not None:
            ppo_cfg = PPOConfig(
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                seed=self.config.seed,
            )
            self.ppo = PPO(self.policy, ppo_cfg, device=device)
        else:  # pragma: no cover
            self.ppo = None

        self.discriminator = GAILDiscriminator(
            self.obs_dim, self.act_dim, self.config.disc_net_arch, self.config.disc_activation
        ).to(self.device)
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config.disc_learning_rate
        )

        self.expert_buffer = expert_buffer if expert_buffer is not None else getattr(expert, "buffer", None)
        self.env_steps = 0
        self.log: List[Dict[str, float]] = []
        self.reward_normalizer = self._make_normalizer()
        self.buffer = RolloutBuffer() if RolloutBuffer is not None else None

    # ------------------------------------------------------------------
    def _make_normalizer(self) -> Optional[Any]:
        if not self.config.normalize_reward:
            return None
        if RunningMeanStd is not None:
            return RunningMeanStd(shape=())
        return None

    def _normalize_reward(self, r: float, update: bool = True) -> float:
        r = float(np.clip(r, -self.config.reward_clip, self.config.reward_clip))
        if self.reward_normalizer is None:
            return r
        self.reward_normalizer.update(np.asarray([r], dtype=np.float64))
        try:
            return float(self.reward_normalizer.normalize(np.asarray([r], dtype=np.float64))[0])
        except Exception:
            return r

    # -- discriminator reward / losses -----------------------------------
    def discriminator_reward(self, obs: Any, action: Any, update_stats: bool = True) -> float:
        """GAIL external reward ``r(s, a) = -log(1 - D(s, a))`` (Ho & Ermon 2016, Eq. 2)."""
        o = torch.as_tensor(_obs_vector(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            d = torch.clamp(self.discriminator(o, a), 1e-6, 1.0 - 1e-6)
            r = -torch.log1p(-d)
        return self._normalize_reward(float(r.item()), update=update_stats)

    def discriminator_loss(
        self, expert_obs: Any, expert_act: Any, policy_obs: Any, policy_act: Any
    ) -> Tuple[Any, Dict[str, float]]:
        expert_logits = self.discriminator.logits(expert_obs, expert_act)
        policy_logits = self.discriminator.logits(policy_obs, policy_act)
        # BCE: expert labels 1, policy labels 0
        loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        ) + F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
        loss = 0.5 * loss
        if self.config.grad_penalty_coef > 0:
            loss = loss + GAILDiscriminator.grad_penalty(
                self.discriminator, expert_obs, expert_act, policy_obs, policy_act,
                self.config.grad_penalty_coef,
            )
        with torch.no_grad():
            acc = 0.5 * (
                (expert_logits > 0).float().mean() + (policy_logits < 0).float().mean()
            )
        return loss, {
            "disc_loss": float(loss.item()),
            "disc_acc": float(acc.item()),
        }

    # -- rollout ---------------------------------------------------------
    def collect_iteration(self, seed: Optional[int] = None, obs: Any = None,
                          context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        if self.buffer is None or self.ppo is None:  # pragma: no cover
            raise ImportError("GAILTrainer requires rice.algorithms.ppo.")
        self.buffer.clear()
        if context is None or "obs" not in context:
            obs, _ = _env_reset(self.env, seed=seed)
        else:
            obs = context["obs"]
        n_steps = int(self.config.n_steps)
        last_obs = obs
        last_done = False
        task_returns: List[float] = []
        ep_return = 0.0
        while len(self.buffer) < n_steps:
            action, value, log_prob = self.policy.act(obs, deterministic=False)
            next_obs, _task_reward, terminated, truncated, _info = _env_step(self.env, action)
            r_gail = self.discriminator_reward(obs, action)
            self.buffer.add(obs, action, r_gail, next_obs, float(terminated), value, log_prob)
            self.env_steps += 1
            ep_return += float(_task_reward)
            last_obs, last_done = next_obs, bool(terminated)
            if terminated or truncated:
                task_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = _env_reset(self.env)
            else:
                obs = next_obs
        if context is not None:
            context["obs"] = obs if not (last_done or truncated) else obs
        last_values = 0.0
        try:
            last_values = float(self.policy.predict_values(last_obs))
        except Exception:
            pass
        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=np.asarray([float(last_done)]),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        info = {
            "env_steps": self.env_steps,
            "task_returns": task_returns,
            "mean_task_return": float(np.mean(task_returns)) if task_returns else float("nan"),
        }
        return self.buffer, info

    def update_discriminator(self, policy_batch: Dict[str, Any]) -> Dict[str, float]:
        if self.expert_buffer is None or len(self.expert_buffer) == 0:
            return {}
        n = int(self.config.disc_batch_size)
        expert = self.expert_buffer.sample(n, self.rng)
        idx = self.rng.integers(0, len(policy_batch["obs"]), size=n)
        expert_obs = torch.as_tensor(expert["obs"], dtype=torch.float32, device=self.device)
        expert_act = torch.as_tensor(expert["action"], dtype=torch.float32, device=self.device)
        policy_obs = torch.as_tensor(np.asarray(policy_batch["obs"])[idx], dtype=torch.float32, device=self.device)
        policy_act = torch.as_tensor(np.asarray(policy_batch["action"])[idx], dtype=torch.float32, device=self.device)

        stats: Dict[str, float] = {}
        for _ in range(max(int(self.config.disc_updates_per_iter), 1)):
            loss, stats = self.discriminator_loss(expert_obs, expert_act, policy_obs, policy_act)
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
        return stats

    # -- main loop -------------------------------------------------------
    def run(
        self,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        eval_env: Any = None,
        eval_callback: Optional[Callable[[int, "GAILTrainer"], None]] = None,
    ) -> Dict[str, Any]:
        budget = int(total_timesteps if total_timesteps is not None
                     else self.config.total_timesteps)
        t0 = time.time()
        context: Dict[str, Any] = {}
        n_updates = 0
        while self.env_steps < budget:
            buffer, info = self.collect_iteration(seed=seed, context=context)
            arrays = buffer.as_arrays()
            stats = self.update_discriminator(arrays)
            update_stats: Dict[str, float] = {}
            try:
                update_stats = self.ppo.update(
                    buffer, last_values=0.0, progress=self.env_steps / max(budget, 1)
                )
            except Exception as exc:  # pragma: no cover - defensive
                update_stats = {"ppo_error": float(0)}
                info["error"] = repr(exc)
            n_updates += 1
            record = {
                "iteration": n_updates,
                "env_steps": self.env_steps,
                **{k: v for k, v in info.items() if isinstance(v, (int, float))},
                **stats,
                **{k: v for k, v in update_stats.items() if isinstance(v, (int, float))},
            }
            self.log.append(record)
            if eval_callback is not None and self.config.eval_every > 0 and \
                    self.env_steps % int(self.config.eval_every) < self.config.n_steps:
                eval_callback(self.env_steps, self)
        metrics = {
            "env_steps": self.env_steps,
            "seconds": time.time() - t0,
            "updates": n_updates,
            "mean_task_return": float(np.mean([r["mean_task_return"] for r in self.log[-5:]]))
            if self.log else float("nan"),
        }
        if eval_env is not None:
            metrics["eval_reward"] = self.evaluate(eval_env)
        return metrics

    def evaluate(self, env: Any, n_episodes: int = 5, seed: Optional[int] = None) -> Dict[str, float]:
        returns: List[float] = []
        for i in range(int(n_episodes)):
            obs, _ = _env_reset(env, seed=None if seed is None else seed + i)
            done, ep_ret, ep_len = False, 0.0, 0
            limit = _env_episode_length(env)
            while not done and ep_len < limit:
                action = self.policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = _env_step(env, action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1
            returns.append(ep_ret)
        arr = np.asarray(returns, dtype=np.float64)
        return {
            "mean_return": float(arr.mean()) if arr.size else float("nan"),
            "std_return": float(arr.std()) if arr.size else float("nan"),
        }

    # -- export ----------------------------------------------------------
    def export_policy(self) -> Any:
        """Return the approximated policy network (RICE-ready ``ActorCritic``)."""
        return self.policy

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "env_steps": self.env_steps,
            "config": self.config.to_dict(),
        }


# ===========================================================================
# results / pipeline
# ===========================================================================
def _make_refine_result(
    task: str,
    method: str,
    explanation: str,
    baseline_reward: float,
    final_reward: float,
    curve: Sequence[Tuple[int, float]],
    env_steps: int,
    seconds: float,
    seed: Optional[int] = None,
    notes: Optional[Sequence[str]] = None,
) -> Any:
    """Build a ``RefineResult`` when available, else a duck-typed stand-in."""
    seeds = [seed] if seed is not None else []
    if RefineResult is not None:
        try:
            log = [{"iteration": i, "env_steps": s, "eval_reward": r} for i, (s, r) in enumerate(curve)]
            return RefineResult(
                policy=None,
                iterations=len(curve),
                env_steps=int(env_steps),
                seconds=float(seconds),
                episode_returns=[float(r) for _, r in curve],
                mean_episode_return=float(final_reward),
                final_eval_reward=float(final_reward),
                critical_fraction=float("nan"),
                log=log,
                stopping_reason="completed",
            )
        except Exception:
            pass
    return _StandaloneRefineResult(
        task=task, method=method, explanation=explanation,
        baseline_reward=baseline_reward, final_reward=final_reward,
        curve=list(curve), env_steps=int(env_steps), seconds=float(seconds),
        seeds=seeds, notes=list(notes or []),
    )


@dataclass
class _StandaloneRefineResult:
    """Lightweight ``RefineResult``-compatible container (used when algorithms.refine is absent)."""

    task: str = ""
    method: str = ""
    explanation: str = ""
    baseline_reward: float = float("nan")
    final_reward: float = float("nan")
    curve: List[Tuple[int, float]] = field(default_factory=list)
    env_steps: int = 0
    seconds: float = 0.0
    seeds: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- RefineResult-compatible surface ---------------------------------
    @property
    def final_eval_reward(self) -> float:
        return float(self.final_reward)

    @property
    def mean_episode_return(self) -> float:
        return float(self.final_reward)

    @property
    def iterations(self) -> int:
        return len(self.curve)

    @property
    def improvement(self) -> float:
        return float(self.final_reward - self.baseline_reward)

    def refining_curve(self, window: int = 1) -> np.ndarray:
        rewards = np.asarray([r for _, r in self.curve], dtype=np.float64)
        if window and window > 1 and rewards.size >= window:
            kernel = np.ones(int(window)) / float(window)
            return np.convolve(rewards, kernel, mode="valid")
        return rewards

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "explanation": self.explanation,
            "baseline_reward": self.baseline_reward,
            "final_reward": self.final_reward,
            "improvement": self.improvement,
            "env_steps": self.env_steps,
            "seconds": self.seconds,
            "notes": list(self.notes),
            "error": self.error,
        }


@dataclass
class SACGAILSummary:
    """Aggregate over seeds; duck-types ``RefiningResult`` for downstream evaluation code."""

    task: str = ""
    method: str = METHOD_NAME
    explanation: str = "sac_gail"
    seeds: List[int] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    final_rewards: List[float] = field(default_factory=list)
    baseline_rewards: List[float] = field(default_factory=list)
    curves: List[np.ndarray] = field(default_factory=list)
    methods: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    env_steps: int = 0
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def final_reward(self) -> float:
        return float(np.nanmean(self.final_rewards)) if self.final_rewards else float("nan")

    @property
    def final_std(self) -> float:
        return float(np.nanstd(self.final_rewards)) if len(self.final_rewards) > 1 else 0.0

    @property
    def baseline_reward(self) -> float:
        return float(np.nanmean(self.baseline_rewards)) if self.baseline_rewards else float("nan")

    @property
    def baseline_std(self) -> float:
        return float(np.nanstd(self.baseline_rewards)) if len(self.baseline_rewards) > 1 else 0.0

    @property
    def improvement(self) -> float:
        return self.final_reward - self.baseline_reward

    @property
    def iterations(self) -> int:
        return max((len(c) for c in self.curves), default=0)

    def mean_curve(self, window: int = 1) -> np.ndarray:
        if not self.curves:
            return np.asarray([])
        n = min(len(c) for c in self.curves)
        stacked = np.stack([np.asarray(c[:n], dtype=np.float64) for c in self.curves])
        mean = stacked.mean(axis=0)
        if window and window > 1 and mean.size >= window:
            kernel = np.ones(int(window)) / float(window)
            return np.convolve(mean, kernel, mode="valid")
        return mean

    def curve_std(self, window: int = 1) -> np.ndarray:
        if len(self.curves) < 2:
            return np.zeros_like(self.mean_curve(window))
        n = min(len(c) for c in self.curves)
        stacked = np.stack([np.asarray(c[:n], dtype=np.float64) for c in self.curves])
        std = stacked.std(axis=0)
        if window and window > 1 and std.size >= window:
            kernel = np.ones(int(window)) / float(window)
            return np.convolve(std, kernel, mode="valid")
        return std

    def refining_curve(self, window: int = 1) -> np.ndarray:
        return self.mean_curve(window)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "explanation": self.explanation,
            "final_reward": self.final_reward,
            "final_std": self.final_std,
            "baseline_reward": self.baseline_reward,
            "baseline_std": self.baseline_std,
            "improvement": self.improvement,
            "seeds": list(self.seeds),
            "seconds": self.seconds,
            "env_steps": self.env_steps,
            "methods": {k: (v.as_dict() if hasattr(v, "as_dict") else v) for k, v in self.methods.items()},
            "notes": list(self.notes),
            "error": self.error,
        }

    def __len__(self) -> int:
        return len(self.final_rewards)


@dataclass
class SACGAILConfig:
    """Full Experiment IV configuration (SAC pre-train -> GAIL imitation -> refining)."""

    task: str = DEFAULT_TASK
    sac: Dict[str, Any] = field(default_factory=dict)
    gail: Dict[str, Any] = field(default_factory=dict)

    # refining budget (unspecified by the paper -> saturating curves)
    p: float = 0.25
    lam: float = 0.001
    alpha: float = 1e-4
    n_iterations: int = 100
    steps_per_iter: Optional[int] = None
    total_env_steps: Optional[float] = None
    rollin_length: Optional[int] = None
    sac_finetune_iterations: int = 100
    sac_finetune_steps_per_iter: int = 1000

    methods: Tuple[str, ...] = SAC_METHODS
    n_seeds: int = 3
    seeds: Optional[Tuple[int, ...]] = None
    seed: Optional[int] = None
    n_eval_episodes: int = 5
    deterministic_eval: bool = True
    policy_config: Optional[Any] = None
    net_arch: Optional[Tuple[int, ...]] = None
    device: str = "auto"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    weights: Optional[str] = None
    mask_weights: Optional[str] = None
    verbose: int = 1
    log_every: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # fill p/lambda/alpha from Table 3 when the task is known and defaults were kept
        hp = TABLE3_HYPERPARAMS.get(self.task)
        if hp is not None:
            if self.p == 0.25 and hp["p"] != 0.25:
                self.p = hp["p"]
            if self.lam == 0.001 and hp["lambda"] != 0.001:
                self.lam = hp["lambda"]
            if self.alpha == 1e-4 and hp["alpha"] != 1e-4:
                self.alpha = hp["alpha"]

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self, **overrides: Any) -> "SACGAILConfig":
        data = asdict(self)
        data.update(overrides)
        known = {f for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = dict(data.pop("extra", {}) or {})
        for key in list(data):
            if key not in known:
                extra[key] = data.pop(key)
        cfg = SACGAILConfig(**data)
        cfg.extra = extra
        return cfg

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "SACGAILConfig":
        cfg = cls()
        if mapping:
            merged = dict(mapping)
            if "lambda" in merged:
                merged["lam"] = merged.pop("lambda")
            if "beta" in merged:
                merged["p"] = merged.pop("beta")
            if "K" in merged:
                merged["rollin_length"] = merged.pop("K")
            cfg = cfg.clone(**merged)
        if overrides:
            cfg = cfg.clone(**overrides)
        return cfg

    def seed_list(self) -> List[int]:
        if self.seeds:
            return [int(s) for s in self.seeds]
        if self.seed is not None:
            return [int(self.seed)]
        return list(range(int(self.n_seeds)))

    def budget(self) -> Optional[float]:
        return self.total_env_steps

    def sac_config(self, seed: Optional[int] = None, **overrides: Any) -> SACConfig:
        base = SACConfig.from_mapping({"task": self.task, "seed": seed, **(self.sac or {})})
        if overrides:
            base = base.clone(**overrides)
        return base

    def gail_config(self, seed: Optional[int] = None, **overrides: Any) -> GAILConfig:
        data = dict(self.gail or {})
        if self.net_arch and "net_arch" not in data:
            data["net_arch"] = tuple(self.net_arch)
        cfg = GAILConfig()
        if data:
            cfg = cfg.clone(**{k: v for k, v in data.items()
                               if k in GAILConfig.__dataclass_fields__})
        if seed is not None:
            cfg = cfg.clone(seed=seed)
        if overrides:
            cfg = cfg.clone(**overrides)
        return cfg

    def to_refine_config(self, seed: Optional[int] = None, p: Optional[float] = None,
                         lam: Optional[float] = None, **overrides: Any) -> Any:
        """Build a ``RefineConfig`` for RICE refining / PPO-FT / StateMask-R on this task."""
        if RefineConfig is None:  # pragma: no cover
            return None
        defaults = {
            "p": self.p if p is None else p,
            "lam": self.lam if lam is None else lam,
            "n_iterations": self.n_iterations,
            "seed": seed,
            "device": self.device,
            "verbose": self.verbose,
        }
        if self.steps_per_iter is not None:
            defaults["steps_per_iter"] = self.steps_per_iter
        if self.total_env_steps is not None:
            defaults["total_env_steps"] = self.total_env_steps
        if self.rollin_length is not None:
            defaults["rollin_length"] = self.rollin_length
        if self.policy_config is not None:
            defaults["policy_config"] = self.policy_config
        defaults.update(overrides)
        try:
            if hasattr(RefineConfig, "from_mapping"):
                return RefineConfig.from_mapping(defaults)
            return RefineConfig(**{k: v for k, v in defaults.items()
                                   if k in RefineConfig.__dataclass_fields__})
        except Exception:
            return RefineConfig()


class SACGAILPipeline:
    """End-to-end Experiment IV pipeline: SAC pre-train -> GAIL imitation -> refining."""

    def __init__(
        self,
        env: Any = None,
        config: Optional[Any] = None,
        evaluation_env: Any = None,
        mask_network: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, SACGAILConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SACGAILConfig.from_mapping(config)
        else:
            self.config = SACGAILConfig()
        if task:
            self.config = self.config.clone(task=task)
        if kwargs:
            self.config = self.config.clone(**kwargs)

        self.env = env
        self.evaluation_env = evaluation_env
        self.mask_network = mask_network
        self.rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        self.sac_agent: Optional[SACAgent] = None
        self.gail_trainer: Optional[GAILTrainer] = None
        self.policy: Optional[Any] = None  # approximated (GAIL) policy network
        self.notes: List[str] = []
        self.log: List[Dict[str, Any]] = []

    # -- construction helpers -------------------------------------------
    def build_env(self, seed: Optional[int] = None) -> Any:
        if self.env is not None:
            return self.env
        if make_env is None:  # pragma: no cover
            raise ImportError("rice.environments.make_env is unavailable.")
        kwargs = dict(self.config.env_kwargs)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        self.env = make_env(self.config.task, **kwargs)
        return self.env

    def build_eval_env(self, seed: Optional[int] = None) -> Any:
        if self.evaluation_env is not None:
            return self.evaluation_env
        try:
            kwargs = dict(self.config.eval_env_kwargs)
            if seed is not None:
                kwargs.setdefault("seed", seed)
            self.evaluation_env = make_env(self.config.task, **kwargs) if make_env else None
        except Exception:
            self.evaluation_env = None
        return self.evaluation_env if self.evaluation_env is not None else self.build_env(seed)

    def resolve_task(self) -> str:
        return self.config.task

    # -- stage 1: SAC pre-training ---------------------------------------
    def pretrain_sac(self, seed: Optional[int] = None, total_timesteps: Optional[int] = None,
                     **overrides: Any) -> SACAgent:
        env = self.build_env(seed)
        cfg = self.config.sac_config(seed=seed, **overrides)
        self.sac_agent = SACAgent(env=env, config=cfg, seed=seed, device=self.config.device)
        metrics = self.sac_agent.pretrain(total_timesteps=total_timesteps, seed=seed)
        self.log.append({"stage": "sac_pretrain", **metrics})
        if self.config.verbose:
            print(f"[SAC] task={self.config.task} seed={seed} {metrics}")
        return self.sac_agent

    def load_sac(self, path: str, seed: Optional[int] = None) -> SACAgent:
        env = self.build_env(seed)
        self.sac_agent = SACAgent(env=env, config=self.config.sac_config(seed=seed),
                                  seed=seed, device=self.config.device)
        self.sac_agent.load(path)
        return self.sac_agent

    # -- stage 2: GAIL imitation -----------------------------------------
    def imitate_with_gail(self, seed: Optional[int] = None, total_timesteps: Optional[int] = None,
                          policy: Optional[Any] = None, **overrides: Any) -> Any:
        env = self.build_env(seed)
        if self.sac_agent is None:
            raise ValueError("imitate_with_gail requires a pre-trained SAC agent "
                             "(call pretrain_sac() or load_sac() first).")
        cfg = self.config.gail_config(seed=seed, **overrides)
        self.gail_trainer = GAILTrainer(
            env=env,
            expert=self.sac_agent,
            config=cfg,
            policy=policy,
            expert_buffer=self.sac_agent.buffer,
            device=self.config.device,
            seed=seed,
        )
        metrics = self.gail_trainer.run(total_timesteps=total_timesteps, seed=seed,
                                        eval_env=self.evaluation_env)
        self.policy = self.gail_trainer.export_policy()
        self.log.append({"stage": "gail", **metrics})
        if self.config.verbose:
            print(f"[GAIL] task={self.config.task} seed={seed} {metrics}")
        return self.policy

    # -- stage 3: refining ------------------------------------------------
    def _mask_for(self, seed: Optional[int]) -> Optional[Any]:
        if self.mask_network is not None:
            return self.mask_network
        cfg = self.config
        if not cfg.mask_weights:
            self.notes.append("No mask network provided; StateMask-R falls back to RICE's PPO-FT path.")
            return None
        try:  # pragma: no cover - depends on checkpoint availability
            from rice.algorithms.mask_network import MaskNetwork  # type: ignore

            env = self.build_env(seed)
            net = MaskNetwork(
                observation_space=env.observation_space,
                net_arch=tuple(cfg.net_arch) if cfg.net_arch else None,
            )
            net.load_policy_state_dict(torch.load(cfg.mask_weights, map_location="cpu"))
            self.mask_network = net
        except Exception as exc:
            self.notes.append(f"Failed to load mask network from {cfg.mask_weights}: {exc!r}")
            self.mask_network = None
        return self.mask_network

    def _baseline_reward(self, seed: Optional[int], policy: Any) -> float:
        env = self.build_eval_env(seed)
        if evaluate_policy is not None:
            try:
                info = evaluate_policy(env, policy, n_episodes=self.config.n_eval_episodes,
                                       seed=seed, deterministic=self.config.deterministic_eval)
                return float(info.get("mean_return", np.nan))
            except Exception:
                pass
        if policy is not None and hasattr(policy, "predict"):
            returns = []
            for i in range(self.config.n_eval_episodes):
                obs, _ = _env_reset(env, seed=None if seed is None else seed + i)
                done, ep_ret, ep_len = False, 0.0, 0
                limit = _env_episode_length(env)
                while not done and ep_len < limit:
                    obs, r, term, trunc, _ = _env_step(env, policy.predict(obs, deterministic=True))
                    done = bool(term or trunc)
                    ep_ret += float(r)
                    ep_len += 1
                returns.append(ep_ret)
            return float(np.mean(returns)) if returns else float("nan")
        return float("nan")

    def refine_method(
        self,
        method: str = "ours",
        seed: Optional[int] = None,
        policy: Optional[Any] = None,
        n_iterations: Optional[int] = None,
        **overrides: Any,
    ) -> Any:
        """Run one refining method and return a ``RefineResult``-compatible object."""
        method = str(method).lower()
        env = self.build_env(seed)
        eval_env = self.build_eval_env(seed)
        warm_policy = policy if policy is not None else self.policy
        mask_net = self._mask_for(seed)

        if method in {"sac", "sac_finetune", "sac-finetune"}:
            fine = SACFineTuner(env=self.build_env(seed), agent=self.sac_agent,
                                config=self.config, seed=seed)
            return fine.refine(seed=seed, n_iterations=n_iterations)

        if method in {"jsrl", "jumpstart", "jump_start_rl"}:
            try:
                from rice.baselines.jsrl import jsrl_refine  # type: ignore
                return jsrl_refine(env=env, policy=warm_policy, mask_network=mask_net,
                                   config={"task": self.config.task,
                                           "n_iterations": n_iterations or self.config.n_iterations},
                                   seeds=[seed] if seed is not None else None,
                                   evaluation_env=eval_env, task=self.config.task)
            except Exception as exc:
                self.notes.append(f"JSRL unavailable ({exc!r}); falling back to PPO-FT path.")
                method = "ppo"

        if method in {"ppo", "ppo_finetune", "ppo-finetune", "finetune"}:
            p_val, lam_val = 0.0, 0.0
        elif method in {"statemask_r", "statemask-r", "statemask"}:
            p_val, lam_val = 1.0, 0.0
        else:  # "ours" / "rice"
            p_val, lam_val = float(self.config.p), float(self.config.lam)

        if refine_policy is None:  # pragma: no cover
            raise ImportError("rice.algorithms.refine.refine_policy is unavailable.")

        cfg = self.config.to_refine_config(seed=seed, p=p_val, lam=lam_val,
                                           n_iterations=n_iterations or self.config.n_iterations,
                                           **overrides)
        t0 = time.time()
        baseline = self._baseline_reward(seed, warm_policy)
        result = refine_policy(
            env=env,
            policy=warm_policy,
            mask_network=mask_net,
            config=cfg,
            evaluation_env=eval_env,
            rng=self.rng,
        )
        if not hasattr(result, "baseline_reward") or result.baseline_reward is None or \
                (isinstance(getattr(result, "baseline_reward", None), float) and
                 np.isnan(getattr(result, "baseline_reward"))):
            try:
                result.baseline_reward = baseline  # type: ignore[attr-defined]
            except Exception:
                pass
        if self.config.verbose:
            print(f"[refine:{method}] task={self.config.task} seed={seed} "
                  f"final={getattr(result, 'final_reward', float('nan')):.3f} "
                  f"({time.time() - t0:.1f}s)")
        return result

    # -- full experiment --------------------------------------------------
    def run(
        self,
        seeds: Optional[Sequence[int]] = None,
        methods: Optional[Sequence[str]] = None,
        sac_timesteps: Optional[int] = None,
        gail_timesteps: Optional[int] = None,
        **kwargs: Any,
    ) -> SACGAILSummary:
        """Run Experiment IV over ``seeds`` and ``methods`` (defaults: all SAC baselines)."""
        seed_list = [int(s) for s in (seeds if seeds is not None else self.config.seed_list())]
        method_list = [str(m).lower() for m in (methods if methods is not None else self.config.methods)]

        summary = SACGAILSummary(
            task=self.config.task,
            method=METHOD_NAME,
            explanation="sac_gail",
            seeds=seed_list,
            notes=list(self.notes),
        )
        t0 = time.time()
        per_method: Dict[str, List[Any]] = {m: [] for m in method_list}

        for seed in seed_list:
            try:
                sac = self.pretrain_sac(seed=seed, total_timesteps=sac_timesteps)
                policy = self.imitate_with_gail(seed=seed, total_timesteps=gail_timesteps)
                summary.baseline_rewards.append(self._baseline_reward(seed, policy))
                if sac is not None:
                    summary.notes.append(
                        f"seed={seed}: SAC pre-train mean episode return "
                        f"{np.nanmean(sac.episode_returns[-20:]) if sac.episode_returns else float('nan'):.2f}"
                    )
            except Exception as exc:
                summary.error = repr(exc)
                summary.notes.append(f"seed={seed}: pre-training/imitation failed: {exc!r}")
                continue

            for method in method_list:
                try:
                    result = self.refine_method(method=method, seed=seed, policy=policy)
                    per_method[method].append(result)
                except Exception as exc:
                    summary.notes.append(f"seed={seed} method={method}: {exc!r}")

            ours_all = per_method.get("ours") or []
            if ours_all:
                last = ours_all[-1]
                final = getattr(last, "final_reward", getattr(last, "final_eval_reward", np.nan))
                curve = getattr(last, "refining_curve", None)
                if callable(curve):
                    try:
                        summary.curves.append(np.asarray(curve(1), dtype=np.float64))
                    except Exception:
                        pass
                summary.final_rewards.append(float(final))
                summary.results.append(last)
                summary.env_steps += int(getattr(last, "env_steps", 0) or 0)

        summary.seconds = time.time() - t0
        summary.methods = {
            m: _aggregate_method(results) for m, results in per_method.items() if results
        }
        summary.notes.extend(self.notes)
        return summary

    def describe(self) -> Dict[str, Any]:
        return {
            "pipeline": "SAC pre-train -> GAIL imitation -> refining (Experiment IV)",
            "task": self.config.task,
            "config": self.config.to_dict(),
            "notes": list(self.notes),
        }


def _aggregate_method(results: Sequence[Any]) -> Dict[str, Any]:
    finals = [float(getattr(r, "final_reward", getattr(r, "final_eval_reward", np.nan)))
              for r in results]
    baselines = [float(getattr(r, "baseline_reward", np.nan)) for r in results]
    return {
        "n": len(results),
        "final_reward": float(np.nanmean(finals)) if finals else float("nan"),
        "final_std": float(np.nanstd(finals)) if len(finals) > 1 else 0.0,
        "baseline_reward": float(np.nanmean(baselines)) if baselines else float("nan"),
        "improvement": (float(np.nanmean(finals)) - float(np.nanmean(baselines)))
        if finals and baselines else float("nan"),
        "results": [r.as_dict() if hasattr(r, "as_dict") else r for r in results],
    }


# ===========================================================================
# factories / functional entry points
# ===========================================================================
def make_sac_gail_pipeline(env: Any = None, config: Any = None, **kwargs: Any) -> SACGAILPipeline:
    """Factory mirroring ``rice.algorithms.refine.make_refiner``."""
    return SACGAILPipeline(env=env, config=config, **kwargs)


def sac_gail_refine(
    env: Any = None,
    policy: Any = None,
    mask_network: Optional[Any] = None,
    config: Any = None,
    seeds: Optional[Sequence[int]] = None,
    n_iterations: Optional[int] = None,
    evaluation_env: Any = None,
    state_manager: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
    task: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Any:
    """Experiment IV entry point (SAC pre-train + GAIL imitation + refining comparison).

    ``env`` / ``policy`` are optional: when omitted, the pipeline builds the environment from
    ``config.task`` via ``rice.environments.make_env`` and produces the warm-start policy by
    pre-training SAC and imitating it with GAIL -- exactly as described in §4.2 Experiment IV.
    Passing a pre-trained ``policy`` skips stage 1/2 and refines it directly.
    """
    pipeline = SACGAILPipeline(
        env=env, config=config, evaluation_env=evaluation_env,
        mask_network=mask_network, rng=rng, task=task,
    )
    if n_iterations is not None:
        pipeline.config = pipeline.config.clone(n_iterations=int(n_iterations))
    if not sac_enabled():
        pipeline.notes.append(
            "RICE_ENABLE_SAC is not set; Experiment IV is heavy (SAC pre-training + GAIL). "
            "Set RICE_ENABLE_SAC=1 to run end to end."
        )

    if policy is not None:
        pipeline.policy = policy
        return pipeline.run(seeds=seeds, methods=methods, **kwargs)
    return pipeline.run(seeds=seeds, methods=methods, **kwargs)


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------
SACGAILRunner = SACGAILPipeline
SACGAILRefiner = SACGAILPipeline
SACGAILPipelineBuilder = make_sac_gail_pipeline
sac_gail_baseline = sac_gail_refine
make_sac_gail_runner = make_sac_gail_pipeline
SoftActorCritic = SACAgent
SoftActorCriticConfig = SACConfig
GenerativeAdversarialImitationLearning = GAILTrainer
GAILConfigDataclass = GAILConfig
SACFineTuneBaseline = SACFineTuner


__all__ = [
    # configs
    "SACConfig",
    "GAILConfig",
    "SACGAILConfig",
    # agents
    "SACAgent",
    "SACFineTuner",
    "GAILTrainer",
    "GAILDiscriminator",
    "SquashedGaussianActor",
    "SoftQNetwork",
    "ReplayBuffer",
    # pipeline
    "SACGAILPipeline",
    "SACGAILSummary",
    "make_sac_gail_pipeline",
    "sac_gail_refine",
    "sac_gail_baseline",
    "sac_enabled",
    # aliases
    "SACGAILRunner",
    "SACGAILRefiner",
    "SoftActorCritic",
    "SoftActorCriticConfig",
    "GenerativeAdversarialImitationLearning",
    "SACFineTuneBaseline",
    # constants
    "METHOD_NAME",
    "METHOD_ALIASES",
    "SAC_METHODS",
    "DEFAULT_TASK",
    "TABLE3_HYPERPARAMS",
    "TABLE1_HOPPER_REFERENCE",
]
