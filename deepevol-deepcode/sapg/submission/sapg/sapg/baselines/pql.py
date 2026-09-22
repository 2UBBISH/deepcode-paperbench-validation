"""Parallel Q-Learning (PQL) baseline -- Section 5.2 of *SAPG: Split and Aggregate
Policy Gradients*.

The paper describes the baseline as::

    Parallel Q-Learning (Li et al., 2023) A parallelized version of DDPG with
    different mixed exploration i.e. varying exploration noise across
    environments to further aid exploration. We use this baseline to compare if
    off-policy methods can outperform on-policy methods when the data collection
    capacity is high.

So the baseline is a massively parallel DDPG:
  * a deterministic actor ``mu(s)`` and a Q-critic ``Q(s, a)`` (+ Polyak-averaged
    target networks), trained off-policy from a large replay buffer;
  * data collected from ``N = 24576`` vectorized environments in parallel;
  * *mixed exploration*: each environment is assigned a different exploration
    noise process / noise magnitude (Gaussian, Ornstein-Uhlenbeck, pink, uniform,
    ...) instead of one global noise level.

Implementation notes / paper ambiguities
----------------------------------------
* The paper gives no hyperparameters for PQL, so this module uses conventional
  DDPG defaults (Adam, lr 1e-4, replay 1e6, batch 4096, Polyak ``tau = 0.005``)
  and exposes every one of them through :class:`PQLConfig` so runs are
  reproducible and tunable. ``tau`` here is the *Polyak* coefficient; the SAPG
  PPO-side ``tau = 0.95`` (GAE lambda) is unrelated.
* Sample-based reporting follows Section 5.2: curves are indexed by the number
  of environment transitions collected (``steps * num_envs``), and multi-seed
  aggregation uses the paper's shaded-band rule
  ``(2 / sqrt(n)) * sum_i (y(t) - y_i(t))^2``.
* All heavy imports (``torch``, project envs/models) are deferred so the module
  can be imported -- and its pure-python configuration validated -- without a
  GPU, torch or IsaacGym installation.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PQL_METHODS",
    "PQLConfig",
    "PQLResult",
    "ReplayBuffer",
    "MixedExplorationNoise",
    "DeterministicActor",
    "QNetwork",
    "PQLTrainer",
    "make_pql_config",
    "make_pql_policy",
    "make_pql_env",
    "train_pql",
    "run_pql_seeds",
    "paper_standard_error",
    "aggregate_seed_histories",
    "main",
]

# Method tags accepted by ``sapg.baselines`` / ``sapg.__init__.train``.
PQL_METHODS: Tuple[str, ...] = ("pql", "apql", "parallel_q_learning", "parallel-q-learning")

# Noise families used for "mixed exploration" (varying noise across envs).
NOISE_TYPES: Tuple[str, ...] = ("gaussian", "ou", "pink", "uniform", "gaussian_decay")

DEFAULT_ENV_COUNTS: Tuple[int, ...] = (128, 512, 2048, 8192, 24576)


# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PQLConfig:
    """Hyperparameters of the PQL (parallelized DDPG) baseline.

    Defaults are conventional DDPG values scaled to a very large parallel
    setting; every field can be overridden from a YAML/dict/config object via
    :meth:`from_any`.
    """

    task: str = "regrasping"
    method: str = "pql"
    num_envs: int = 24576
    horizon_length: int = 16

    # Networks
    actor_mlp_units: Tuple[int, ...] = (768, 512, 256)
    critic_mlp_units: Tuple[int, ...] = (768, 512, 256)
    activation: str = "elu"
    use_lstm: bool = False
    lstm_hidden_size: int = 768
    obs_dim: int = 60
    action_dim: int = 23

    # Optimisation
    learning_rate: float = 1e-4
    critic_learning_rate: Optional[float] = None
    optimizer: str = "adam"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    grad_norm: float = 1.0
    gamma: float = 0.99
    polyak_tau: float = 0.005          # target-network soft-update coefficient
    batch_size: int = 4096
    gradient_steps: int = 16           # updates per collected iteration
    warmup_transitions: int = 100_000
    replay_size: int = 1_000_000
    update_after_transitions: Optional[int] = None  # defaults to warmup

    # Exploration ("mixed exploration noise across environments")
    noise_types: Tuple[str, ...] = NOISE_TYPES
    noise_scale: float = 0.1
    noise_scale_min: float = 0.005
    noise_scale_max: float = 0.4
    noise_decay: float = 1.0           # 1.0 = no decay
    ou_theta: float = 0.15
    ou_dt: float = 0.01
    intra_noise_std: float = 0.0       # additional action noise at update time
    action_clip: float = 1.0
    deterministic_eval: bool = True

    # Book-keeping
    seed: int = 0
    device: str = "cuda:0"
    log_dir: str = "runs"
    log_interval: int = 1
    save_interval: int = 0
    aggregation: str = "none"

    # ---- convenience -----------------------------------------------------
    def __post_init__(self) -> None:
        if isinstance(self.actor_mlp_units, list):
            self.actor_mlp_units = tuple(self.actor_mlp_units)
        if isinstance(self.critic_mlp_units, list):
            self.critic_mlp_units = tuple(self.critic_mlp_units)
        if isinstance(self.noise_types, list):
            self.noise_types = tuple(self.noise_types)
        if self.critic_learning_rate is None:
            self.critic_learning_rate = self.learning_rate
        if self.update_after_transitions is None:
            self.update_after_transitions = self.warmup_transitions
        if self.batch_size <= 0:
            self.batch_size = max(1, int(self.num_envs))
        # Actor and critic share the same trunk width by default (DDPG style);
        # keep them independent if the user asked for different widths.
        if not self.critic_mlp_units:
            self.critic_mlp_units = tuple(self.actor_mlp_units)

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    def as_dict(self) -> Dict[str, Any]:  # alias
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "PQLConfig":
        if not d:
            return cls()
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in dict(d).items() if k in valid}
        # Normalise alternative spellings coming from other configs.
        aliases = {
            "lr": "learning_rate",
            "actor_lr": "learning_rate",
            "critic_lr": "critic_learning_rate",
            "tau_polyak": "polyak_tau",
            "soft_tau": "polyak_tau",
            "replay_buffer_size": "replay_size",
            "buffer_size": "replay_size",
            "min_buffer_size": "warmup_transitions",
            "num_updates": "gradient_steps",
        }
        for src, dst in aliases.items():
            if src in d and dst not in kwargs:
                kwargs[dst] = d[src]
        # Tuples that may have been serialised as lists.
        for key in ("actor_mlp_units", "critic_mlp_units", "adam_betas", "noise_types"):
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    @classmethod
    def from_any(cls, config: Any = None, **overrides: Any) -> "PQLConfig":
        """Build a config from ``None``, a dict, a ``PQLConfig`` or a foreign
        config object (e.g. :class:`sapg.utils.config.SAPGConfig`)."""
        if config is None:
            base = cls()
        elif isinstance(config, PQLConfig):
            base = config
        elif isinstance(config, dict):
            base = cls.from_dict(config)
        else:
            data = {}
            try:
                from dataclasses import asdict  # local import, cheap

                if hasattr(config, "__dataclass_fields__"):
                    data = asdict(config)
                else:
                    data = dict(getattr(config, "__dict__", {}))
            except Exception:
                data = {}
            base = cls.from_dict(data)
        if overrides:
            valid = {f.name for f in fields(cls)}
            clean = {k: v for k, v in overrides.items() if k in valid and v is not None}
            if clean:
                base = replace(base, **clean)
        base.__post_init__()
        return base


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity replay buffer of environment transitions.

    Tensors are kept time-major/flat of shape ``[capacity, dim]`` and the
    sampling/insertion logic is pure ``torch``.  ``add`` accepts either batched
    tensors (``[num_envs, ...]``) or unbatched ones.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: Any = "cpu",
        num_envs: int = 1,
        use_replace_mask: bool = False,
    ) -> None:
        if not HAS_TORCH:  # pragma: no cover
            raise RuntimeError("ReplayBuffer requires torch to be installed")
        self.capacity = int(max(1, capacity))
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_envs = int(max(1, num_envs))
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.use_replace_mask = bool(use_replace_mask)

        self.obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.next_obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.capacity, self.action_dim, device=self.device)
        self.rewards = torch.zeros(self.capacity, 1, device=self.device)
        self.dones = torch.zeros(self.capacity, 1, device=self.device)
        self.ptr = 0
        self.size = 0
        self.total_added = 0
        # Optional per-transition "valid" mask (used by the block/parallel
        # variants where only a subset of envs contributes on a given step).
        self.mask = torch.ones(self.capacity, 1, device=self.device, dtype=torch.bool)

    # ---- insertion -------------------------------------------------------
    def add(
        self,
        obs: Any,
        actions: Any,
        rewards: Any,
        next_obs: Any,
        dones: Any,
        mask: Any = None,
    ) -> int:
        """Insert a batch of transitions. Returns the number inserted."""
        obs_t = self._as_2d(obs, self.obs_dim)
        act_t = self._as_2d(actions, self.action_dim)
        next_t = self._as_2d(next_obs, self.obs_dim)
        rew_t = self._as_2d(rewards, 1)
        done_t = self._as_2d(dones, 1)
        mask_t = self._as_2d(mask, 1) if mask is not None else None

        n = int(obs_t.shape[0])
        if n == 0:
            return 0
        if n >= self.capacity:
            obs_t, act_t, next_t, rew_t, done_t = (
                obs_t[-self.capacity :],
                act_t[-self.capacity :],
                next_t[-self.capacity :],
                rew_t[-self.capacity :],
                done_t[-self.capacity :],
            )
            if mask_t is not None:
                mask_t = mask_t[-self.capacity :]
            n = self.capacity

        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.obs[sl] = obs_t
            self.actions[sl] = act_t
            self.next_obs[sl] = next_t
            self.rewards[sl] = rew_t
            self.dones[sl] = done_t
            if mask_t is not None:
                self.mask[sl] = mask_t.bool().view(-1, 1)
            elif self.use_replace_mask:
                self.mask[sl] = True
        else:
            first = self.capacity - self.ptr
            self._write(slice(self.ptr, self.capacity), obs_t[:first], act_t[:first],
                        next_t[:first], rew_t[:first], done_t[:first],
                        None if mask_t is None else mask_t[:first])
            rest = n - first
            self._write(slice(0, rest), obs_t[first:], act_t[first:],
                        next_t[first:], rew_t[first:], done_t[first:],
                        None if mask_t is None else mask_t[first:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n
        return n

    def _write(self, sl, obs, act, next_obs, rew, done, mask) -> None:
        self.obs[sl] = obs
        self.actions[sl] = act
        self.next_obs[sl] = next_obs
        self.rewards[sl] = rew
        self.dones[sl] = done
        if mask is not None:
            self.mask[sl] = mask.bool().view(-1, 1)
        elif self.use_replace_mask:
            self.mask[sl] = True

    def _as_2d(self, x: Any, dim: int) -> Any:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        t = t.detach().to(self.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.view(-1, dim) if dim > 1 else t.view(-1, 1)
        elif t.dim() > 2:
            t = t.reshape(-1, dim)
        if t.shape[-1] != dim:
            t = t.reshape(-1, dim)
        return t

    # ---- sampling --------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        generator: Any = None,
        device: Any = None,
        replace: bool = True,
    ) -> Dict[str, Any]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        n = min(int(batch_size), self.size)
        idx = torch.randint(0, self.size, (n,), device=self.device, generator=generator)
        out: Dict[str, Any] = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "indices": idx,
        }
        if device is not None:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev != self.device:
                out = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        return out

    def __len__(self) -> int:
        return int(self.size)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "ptr": self.ptr,
            "size": self.size,
            "total_added": self.total_added,
            "obs": self.obs.cpu(),
            "actions": self.actions.cpu(),
            "rewards": self.rewards.cpu(),
            "next_obs": self.next_obs.cpu(),
            "dones": self.dones.cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ptr = int(state.get("ptr", 0))
        self.size = int(state.get("size", 0))
        self.total_added = int(state.get("total_added", self.size))
        for key in ("obs", "actions", "rewards", "next_obs", "dones"):
            if key in state:
                src = state[key].to(self.device)
                n = min(src.shape[0], self.capacity)
                getattr(self, key)[:n] = src[:n]


# ---------------------------------------------------------------------------
# Mixed exploration noise (varying noise across environments)
# ---------------------------------------------------------------------------
class MixedExplorationNoise:
    """Assigns a *different* exploration noise process to each environment.

    The paper's PQL baseline is "a parallelized version of DDPG with different
    mixed exploration i.e. varying exploration noise across environments to
    further aid exploration".  This class implements exactly that:

    * environments are partitioned into ``len(noise_types)`` contiguous groups,
      each with a distinct noise family;
    * within a family the noise magnitude is spread geometrically between
      ``noise_scale_min`` and ``noise_scale_max`` so neighbouring environments
      also differ;
    * optionally the magnitudes decay over training (``noise_decay < 1``).
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        config: Optional[PQLConfig] = None,
        device: Any = "cpu",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else PQLConfig.from_any(**kwargs)
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.config = cfg
        self.device = device
        self.seed = int(seed)
        self.types: Tuple[str, ...] = tuple(cfg.noise_types) or ("gaussian",)
        self.decay = float(cfg.noise_decay)
        self._step = 0

        # Per-environment noise-type index and magnitude ladder.
        idx = torch.arange(self.num_envs, dtype=torch.float32)
        type_idx = (idx * len(self.types) / max(1, self.num_envs)).long()
        self.noise_type_idx = type_idx.clamp(max=len(self.types) - 1)
        frac = (idx % max(1, math.ceil(self.num_envs / max(1, len(self.types))))) / max(
            1.0, float(math.ceil(self.num_envs / max(1, len(self.types)))) - 1.0
        )
        scale = cfg.noise_scale_min * (cfg.noise_scale_max / max(1e-8, cfg.noise_scale_min)) ** frac
        self.base_scale = scale.to(torch.float32)
        self.mask_gaussian = (self.noise_type_idx % 4 == 0).float()
        self.mask_ou = (self.noise_type_idx % 4 == 1).float()
        self.mask_pink = (self.noise_type_idx % 4 == 2).float()
        self.mask_uniform = (self.noise_type_idx % 4 == 3).float()

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        self.generator = gen
        self.ou_state = torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32)
        self.reset()

    # ---- helpers ---------------------------------------------------------
    def reset(self, env_ids: Any = None) -> None:
        if env_ids is None:
            self.ou_state.zero_()
            return
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
        if ids.numel():
            self.ou_state[ids.long()] = 0.0

    @property
    def current_scale(self) -> Any:
        factor = self.decay ** float(self._step)
        return self.base_scale * factor

    def scale_for(self, env_ids: Any = None) -> Any:
        if env_ids is None:
            return self.current_scale
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
        return self.current_scale[ids.long()]

    def type_for(self, env_id: int) -> str:
        return self.types[int(self.noise_type_idx[int(env_id)]) % len(self.types)]

    # ---- main API --------------------------------------------------------
    def sample(self, dones: Any = None) -> Any:
        """Return an exploration-noise tensor of shape ``[num_envs, action_dim]``."""
        if not HAS_TORCH:  # pragma: no cover
            raise RuntimeError("MixedExplorationNoise requires torch")
        n, a = self.num_envs, self.action_dim
        scale = self.current_scale.view(n, 1)

        if dones is not None:
            d = dones if isinstance(dones, torch.Tensor) else torch.as_tensor(dones)
            d = d.detach().reshape(-1).bool()
            if d.numel() == n and bool(d.any()):
                # Reset the temporally-correlated components of finished envs.
                self.ou_state[d] = 0.0

        noise = torch.zeros(n, a, dtype=torch.float32)

        # Gaussian
        gauss = torch.randn(n, a, generator=self.generator) * scale
        noise += gauss * self.mask_gaussian.view(n, 1)
        # Uniform
        unif = (torch.rand(n, a, generator=self.generator) * 2.0 - 1.0) * scale
        noise += unif * self.mask_uniform.view(n, 1)
        # Ornstein-Uhlenbeck
        theta, dt = float(self.config.ou_theta), float(self.config.ou_dt)
        drift = -theta * self.ou_state * dt
        diffusion = math.sqrt(dt) * torch.randn(n, a, generator=self.generator)
        new_ou = self.ou_state + drift + diffusion
        self.ou_state = new_ou.clamp(-2.0, 2.0)
        noise += self.ou_state * scale * self.mask_ou.view(n, 1)
        # Pink / temporally-correlated 1/f-ish noise (random-walk low-pass)
        if not hasattr(self, "_pink_state"):
            self._pink_state = torch.zeros(n, a, dtype=torch.float32)
        self._pink_state = 0.98 * self._pink_state + 0.02 * torch.randn(
            n, a, generator=self.generator
        )
        noise += self._pink_state * scale * self.mask_pink.view(n, 1)

        # Additive global noise floor so that every env still explores a little.
        if float(self.config.intra_noise_std) > 0.0:
            noise += float(self.config.intra_noise_std) * torch.randn(
                n, a, generator=self.generator
            )

        self._step += 1
        return noise

    def __call__(self, dones: Any = None) -> Any:
        return self.sample(dones)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ou_state": self.ou_state.clone(),
            "pink_state": getattr(self, "_pink_state", torch.zeros_like(self.ou_state)).clone(),
            "step": self._step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("ou_state") is not None:
            self.ou_state = state["ou_state"].clone()
        if state.get("pink_state") is not None:
            self._pink_state = state["pink_state"].clone()
        self._step = int(state.get("step", 0))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def _build_mlp(input_dim: int, units: Sequence[int], output_dim: int, activation: str = "elu"):
    """Reuse the SAPG network builder when available, else build a local MLP."""
    try:  # pragma: no cover - depends on import graph
        from ..models.networks import MLP as _MLP
        from ..models.networks import make_backbone as _make_backbone

        trunk = _make_backbone(
            input_dim=int(input_dim),
            mlp_units=tuple(int(u) for u in units),
            activation=activation,
            use_lstm=False,
            output_dim=None,
        )
        head = _MLP(
            int(trunk.out_features),
            units=(),
            output_dim=int(output_dim),
            activation="identity",
        )
        return _Sequential(trunk, head)
    except Exception:
        pass
    return _LocalMLP(int(input_dim), tuple(int(u) for u in units), int(output_dim), activation)


if HAS_TORCH:

    class _Sequential(nn.Module):
        """Minimal trunk+head composition (kept local to avoid API surprises)."""

        def __init__(self, trunk: Any, head: Any) -> None:
            super().__init__()
            self.trunk = trunk
            self.head = head
            self.input_dim = getattr(trunk, "input_dim", None)
            self.out_features = getattr(head, "out_features", head)

        def forward(self, x: Any) -> Any:
            out = self.trunk(x)
            if isinstance(out, tuple):
                out = out[0]
            return self.head(out)

    class _LocalMLP(nn.Module):
        """Fallback MLP: Linear -> activation -> ... -> Linear(output_dim)."""

        ACTS = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "leaky_relu": nn.LeakyReLU,
            "identity": nn.Identity,
        }

        def __init__(
            self,
            input_dim: int,
            units: Sequence[int],
            output_dim: int,
            activation: str = "elu",
        ) -> None:
            super().__init__()
            act_cls = self.ACTS.get(str(activation).lower(), nn.ELU)
            layers: List[nn.Module] = []
            prev = int(input_dim)
            for u in units:
                layers.append(nn.Linear(prev, int(u)))
                layers.append(act_cls())
                prev = int(u)
            layers.append(nn.Linear(prev, int(output_dim)))
            self.net = nn.Sequential(*layers)
            self.input_dim = int(input_dim)
            self.out_features = int(output_dim)
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last.weight, gain=0.01)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class DeterministicActor(nn.Module):
        """DDPG deterministic actor ``mu(s)`` with ``tanh``-bounded output."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            mlp_units: Sequence[int] = (768, 512, 256),
            activation: str = "elu",
            action_scale: float = 1.0,
            output_activation: str = "tanh",
            config: Optional[PQLConfig] = None,
        ) -> None:
            super().__init__()
            if config is not None:
                obs_dim = int(getattr(config, "obs_dim", obs_dim))
                action_dim = int(getattr(config, "action_dim", action_dim))
                mlp_units = tuple(getattr(config, "actor_mlp_units", mlp_units))
                activation = str(getattr(config, "activation", activation))
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.action_scale = float(action_scale)
            self.output_activation = str(output_activation).lower()
            self.net = _build_mlp(self.obs_dim, mlp_units, self.action_dim, activation)

        def forward(self, obs: Any) -> Any:
            raw = self.net(obs)
            if self.output_activation == "tanh":
                return self.action_scale * torch.tanh(raw)
            if self.output_activation == "identity":
                return raw * self.action_scale
            return self.action_scale * torch.tanh(raw)

        @property
        def actor_parameters(self) -> Iterable[Any]:
            return self.parameters()

    class QNetwork(nn.Module):
        """DDPG critic ``Q(s, a)`` (state-action input, scalar output)."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            mlp_units: Sequence[int] = (768, 512, 256),
            activation: str = "elu",
            config: Optional[PQLConfig] = None,
            action_in_first_layer: bool = False,
        ) -> None:
            super().__init__()
            if config is not None:
                obs_dim = int(getattr(config, "obs_dim", obs_dim))
                action_dim = int(getattr(config, "action_dim", action_dim))
                mlp_units = tuple(getattr(config, "critic_mlp_units", mlp_units))
                activation = str(getattr(config, "activation", activation))
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.action_in_first_layer = bool(action_in_first_layer)
            self.net = _build_mlp(self.obs_dim + self.action_dim, mlp_units, 1, activation)

        def forward(self, obs: Any, actions: Any) -> Any:
            x = torch.cat([obs, actions], dim=-1)
            return self.net(x)

        def q1(self, obs: Any, actions: Any) -> Any:
            return self.forward(obs, actions)

else:  # pragma: no cover - torch missing

    class DeterministicActor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DeterministicActor requires torch to be installed")

    class QNetwork:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("QNetwork requires torch to be installed")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PQLResult:
    """Result of one PQL training run (one seed)."""

    history: List[Dict[str, float]] = field(default_factory=list)
    samples: Any = None
    num_envs: int = 0
    seed: int = 0
    trainer: Any = None

    def final(self, key: str = "episode_return", default: float = float("nan")) -> float:
        for rec in reversed(self.history):
            if key in rec and rec[key] is not None:
                try:
                    return float(rec[key])
                except (TypeError, ValueError):
                    continue
        return float(default)

    def curve(self, key: str = "episode_return") -> List[float]:
        return [float(rec.get(key, float("nan"))) for rec in self.history]

    def sample_curve(self, key: str = "episode_return") -> Tuple[List[float], List[float]]:
        xs = [float(rec.get("samples", i)) for i, rec in enumerate(self.history)]
        ys = self.curve(key)
        return xs, ys

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "num_envs": self.num_envs,
            "samples": self.samples,
            "final": self.final(),
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PQLTrainer:
    """Parallel Q-Learning: massively parallel DDPG with mixed exploration noise.

    Parameters
    ----------
    config:
        A :class:`PQLConfig`, a dict, an ``SAPGConfig``-like object, or ``None``.
    env:
        Vectorized environment exposing ``reset()`` / ``step(actions)``.  When
        omitted, one is built lazily through ``sapg.envs.make_env``.
    actor, critic, critic_target:
        Optional pre-built modules (target defaults to a copy of ``critic``).
    noise:
        Optional :class:`MixedExplorationNoise` instance.
    """

    def __init__(
        self,
        config: Any = None,
        env: Any = None,
        actor: Any = None,
        critic: Any = None,
        critic_target: Any = None,
        actor_target: Any = None,
        noise: Optional[MixedExplorationNoise] = None,
        optimizers: Optional[Tuple[Any, Any]] = None,
        logger: Any = None,
        device: Any = None,
        replay: Optional[ReplayBuffer] = None,
        seed: Optional[int] = None,
        **overrides: Any,
    ) -> None:
        self.config = PQLConfig.from_any(config, **overrides)
        self.logger = logger
        if seed is not None:
            self.config.seed = int(seed)
        self.seed = int(self.config.seed)
        self._set_global_seed(self.seed)

        self.device = self._resolve_device(device or self.config.device)
        self.history: List[Dict[str, float]] = []
        self.total_samples = 0
        self.total_env_steps = 0
        self.update_count = 0
        self._last_obs: Any = None
        self._ep_returns: Any = None
        self._ep_lengths: Any = None
        self._done_counts: Any = None
        self._episode_return_stats: Dict[str, Any] = {}

        self.env = env if env is not None else self._build_env()
        self.num_envs = int(self._infer_num_envs(self.env))
        self.obs_dim = int(self.config.obs_dim or self._infer_dim(getattr(self.env, "obs_dim", 0)))
        self.action_dim = int(
            self.config.action_dim or self._infer_dim(getattr(self.env, "action_dim", 0))
        )
        if not self.obs_dim:
            raise ValueError("Unable to determine obs_dim for the PQL baseline")
        if not self.action_dim:
            raise ValueError("Unable to determine action_dim for the PQL baseline")

        # Networks ---------------------------------------------------------
        self.actor = actor if actor is not None else DeterministicActor(
            self.obs_dim,
            self.action_dim,
            mlp_units=self.config.actor_mlp_units,
            activation=self.config.activation,
            config=self.config,
        )
        self.critic = critic if critic is not None else QNetwork(
            self.obs_dim,
            self.action_dim,
            mlp_units=self.config.critic_mlp_units,
            activation=self.config.activation,
            config=self.config,
        )
        self.actor_target = (
            actor_target if actor_target is not None else copy.deepcopy(self.actor)
        )
        self.critic_target = (
            critic_target if critic_target is not None else copy.deepcopy(self.critic)
        )
        for net in (self.actor, self.critic, self.actor_target, self.critic_target):
            try:
                net.to(self.device)
            except Exception:
                pass
            for p in net.parameters():
                p.requires_grad_(net in (self.actor, self.critic))

        # Optimisers -------------------------------------------------------
        if optimizers is not None:
            self.optimizer_actor, self.optimizer_critic = optimizers
        else:
            self.optimizer_actor = self._make_optimizer(
                self.actor.parameters(), self.config.learning_rate
            )
            self.optimizer_critic = self._make_optimizer(
                self.critic.parameters(), self.config.critic_learning_rate
            )

        # Replay + exploration --------------------------------------------
        self.replay = replay if replay is not None else ReplayBuffer(
            capacity=self.config.replay_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            num_envs=self.num_envs,
        )
        self.noise = noise if noise is not None else MixedExplorationNoise(
            self.num_envs,
            self.action_dim,
            config=self.config,
            device=self.device,
            seed=self.seed,
        )
        self._generator = self._make_generator(self.seed)

    # ---- setup helpers ---------------------------------------------------
    @staticmethod
    def _set_global_seed(seed: int) -> None:
        random.seed(seed)
        try:  # pragma: no cover
            import numpy as _np

            _np.random.seed(seed)
        except Exception:
            pass
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _make_generator(seed: int) -> Any:
        if not HAS_TORCH:
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        return gen

    @staticmethod
    def _make_optimizer(params: Iterable[Any], lr: float) -> Any:
        if not HAS_TORCH:  # pragma: no cover
            raise RuntimeError("PQLTrainer requires torch")
        return torch.optim.Adam(list(params), lr=float(lr), eps=1e-8)

    @staticmethod
    def _resolve_device(device: Any) -> Any:
        if not HAS_TORCH:
            return device
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        return dev

    @staticmethod
    def _infer_num_envs(env: Any) -> int:
        for attr in ("num_envs", "n_envs"):
            value = getattr(env, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if value:
                return int(value)
        cfg = getattr(env, "cfg", None) or getattr(env, "config", None)
        value = getattr(cfg, "num_envs", None) if cfg is not None else None
        if value:
            return int(value)
        try:
            return int(len(env))
        except Exception:
            return 1

    @staticmethod
    def _infer_dim(value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    def _build_env(self) -> Any:
        try:  # pragma: no cover - depends on import graph
            from ..envs import make_env

            return make_env(
                self.config.task,
                num_envs=self.num_envs if hasattr(self, "num_envs") else self.config.num_envs,
                device=str(self.device),
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "No environment supplied and one could not be constructed for the PQL "
                f"baseline (task={self.config.task!r}): {exc}"
            ) from exc

    # ---- rollout ---------------------------------------------------------
    def reset(self) -> Any:
        obs = self.env.reset()
        self._last_obs = self._to_tensor(obs)
        n = int(self._last_obs.shape[0]) if HAS_TORCH else self.num_envs
        if HAS_TORCH:
            self._ep_returns = torch.zeros(n, device=self.device)
            self._ep_lengths = torch.zeros(n, device=self.device)
            self._done_counts = torch.zeros(n, device=self.device)
        else:  # pragma: no cover
            self._ep_returns = [0.0] * n
            self._ep_lengths = [0.0] * n
            self._done_counts = [0.0] * n
        self.noise.reset()
        return self._last_obs

    def _to_tensor(self, x: Any) -> Any:
        if not HAS_TORCH:  # pragma: no cover
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def select_actions(self, obs: Any, deterministic: bool = False) -> Any:
        """Deterministic policy output (+ mixed exploration noise when training)."""
        obs_t = self._to_tensor(obs)
        with torch.no_grad():
            actions = self.actor(obs_t)
        if not deterministic:
            noise = self.noise.sample()
            noise = noise.to(actions.device) if hasattr(noise, "to") else noise
            actions = actions + noise
        if float(self.config.action_clip) > 0:
            actions = actions.clamp(-float(self.config.action_clip), float(self.config.action_clip))
        return actions

    def collect(self, horizon_length: Optional[int] = None, deterministic: bool = False) -> Dict[str, Any]:
        """Collect ``horizon_length`` steps from every parallel environment."""
        if self._last_obs is None:
            self.reset()
        horizon = int(horizon_length or self.config.horizon_length)
        obs = self._last_obs
        if not HAS_TORCH:  # pragma: no cover
            return {"samples": 0, "steps": 0}

        totals = {
            "samples": 0,
            "steps": 0,
            "reward_sum": 0.0,
            "reward_count": 0,
            "done_count": 0.0,
        }
        ep_returns_list: List[Any] = []
        ep_lengths_list: List[Any] = []

        for _ in range(horizon):
            actions = self.select_actions(obs, deterministic=deterministic)
            step_out = self.env.step(actions)
            if not isinstance(step_out, (tuple, list)) or len(step_out) < 4:
                raise RuntimeError(
                    "PQLTrainer expects env.step(actions) -> (obs, reward, done, info)"
                )
            next_obs, rewards, dones, infos = step_out[:4]
            next_obs = self._to_tensor(next_obs)
            rewards_t = self._to_tensor(rewards).reshape(-1)
            dones_t = self._to_tensor(dones).reshape(-1).float()

            self.replay.add(obs, actions, rewards_t, next_obs, dones_t)

            self._ep_returns = self._ep_returns + rewards_t
            self._ep_lengths = self._ep_lengths + 1.0
            n = min(self._ep_returns.shape[0], dones_t.shape[0])
            done_mask = dones_t[:n].bool()
            if bool(done_mask.any()):
                finished = done_mask.nonzero(as_tuple=False).reshape(-1)
                ep_returns_list.append(self._ep_returns[:n][finished].clone())
                ep_lengths_list.append(self._ep_lengths[:n][finished].clone())
                self._ep_returns[:n][finished] = 0.0
                self._ep_lengths[:n][finished] = 0.0
                self.noise.reset(finished)

            tot_r = float(rewards_t.sum().item())
            info_rewards, info_count = self._info_rewards(infos)
            if info_count:
                tot_r = info_rewards
                count = info_count
            else:
                count = int(rewards_t.numel())

            totals["samples"] += int(rewards_t.numel())
            totals["steps"] += 1
            totals["reward_sum"] += tot_r
            totals["reward_count"] += count
            totals["done_count"] += float(dones_t.sum().item())
            obs = next_obs

        self._last_obs = obs
        self.total_samples += totals["samples"]
        self.total_env_steps += totals["steps"]

        stats: Dict[str, float] = {
            "samples": float(self.total_samples),
            "steps": float(self.total_env_steps),
        }
        if totals["reward_count"]:
            stats["reward_mean"] = totals["reward_sum"] / float(totals["reward_count"])
        if ep_returns_list:
            cat = torch.cat(ep_returns_list)
            stats["episode_return"] = float(cat.mean().item())
            stats["episode_return_max"] = float(cat.max().item())
            if ep_lengths_list:
                stats["episode_length"] = float(torch.cat(ep_lengths_list).float().mean().item())
            try:  # success metrics forwarded by the task wrappers, if any
                success = self._info_successes(infos)
                if success is not None:
                    stats["successes"] = success
            except Exception:
                pass
        return stats

    @staticmethod
    def _info_rewards(infos: Any) -> Tuple[float, int]:
        """Extract a mean episode-return style quantity from ``infos`` if present."""
        if isinstance(infos, dict):
            for key in (
                "episode_return",
                "episode_reward",
                "episode_stats_return",
                "return",
                "returns",
            ):
                if key in infos:
                    val = infos[key]
                    arr = val if isinstance(val, (list, tuple)) else None
                    if arr is not None:
                        if len(arr) == 0:
                            continue
                        return float(sum(arr) / len(arr)), len(arr)
                    try:
                        return float(val), 1
                    except (TypeError, ValueError):
                        continue
        return 0.0, 0

    @staticmethod
    def _info_successes(infos: Any) -> Optional[float]:
        if isinstance(infos, dict):
            for key in ("successes", "episode_successes", "success"):
                if key in infos:
                    val = infos[key]
                    if isinstance(val, (list, tuple)):
                        if not val:
                            continue
                        return float(sum(val) / len(val))
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
        return None

    # ---- update ----------------------------------------------------------
    def update(self, gradient_steps: Optional[int] = None) -> Dict[str, float]:
        """Run ``gradient_steps`` DDPG updates. Returns aggregate statistics."""
        if not HAS_TORCH:  # pragma: no cover
            raise RuntimeError("PQLTrainer.update requires torch")
        steps = int(gradient_steps or self.config.gradient_steps)
        stats_accum: Dict[str, float] = {}
        n_used = 0

        if self.total_samples < int(self.config.update_after_transitions):
            return {"updates": 0.0, "replay_size": float(len(self.replay))}

        for _ in range(max(0, steps)):
            batch = self.replay.sample(self.config.batch_size, generator=self._generator)
            obs = batch["obs"].to(self.device)
            actions = batch["actions"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            next_obs = batch["next_obs"].to(self.device)
            dones = batch["dones"].to(self.device)

            stats = self._update_step(obs, actions, rewards, next_obs, dones)
            for k, v in stats.items():
                stats_accum[k] = stats_accum.get(k, 0.0) + float(v)
            n_used += 1

        if n_used:
            for k in list(stats_accum.keys()):
                stats_accum[k] /= float(n_used)
        stats_accum["updates"] = float(n_used)
        stats_accum["replay_size"] = float(len(self.replay))
        self.update_count += n_used
        return stats_accum

    def _update_step(self, obs: Any, actions: Any, rewards: Any, next_obs: Any, dones: Any) -> Dict[str, float]:
        gamma = float(self.config.gamma)

        # --- critic -------------------------------------------------------
        with torch.no_grad():
            target_actions = self.actor_target(next_obs)
            if float(self.config.intra_noise_std) > 0.0:
                target_actions = target_actions + float(self.config.intra_noise_std) * torch.randn_like(
                    target_actions
                )
                target_actions = target_actions.clamp(
                    -float(self.config.action_clip), float(self.config.action_clip)
                )
            q_target_next = self.critic_target(next_obs, target_actions)
            target_q = rewards + gamma * (1.0 - dones) * q_target_next

        q_pred = self.critic(obs, actions)
        critic_loss = F.mse_loss(q_pred, target_q)

        self.optimizer_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if float(self.config.grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config.grad_norm))
        self.optimizer_critic.step()

        # --- actor --------------------------------------------------------
        actor_actions = self.actor(obs)
        actor_loss = -self.critic(obs, actor_actions).mean()

        self.optimizer_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if float(self.config.grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.grad_norm))
        self.optimizer_actor.step()

        # --- target networks ---------------------------------------------
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
        }

    def soft_update(self, source: Any, target: Any) -> None:
        tau = float(self.config.polyak_tau)
        with torch.no_grad():
            for p_s, p_t in zip(source.parameters(), target.parameters()):
                p_t.mul_(1.0 - tau).add_(tau * p_s)

    # ---- outer loop ------------------------------------------------------
    def learn(
        self,
        num_iterations: Optional[int] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, float]]:
        """Train until ``max_samples`` transitions (or ``num_iterations``)."""
        if self._last_obs is None:
            self.reset()

        horizon = max(1, int(self.config.horizon_length))
        num_envs = max(1, self.num_envs)
        per_iter = horizon * num_envs

        if max_samples is not None and num_iterations is None:
            num_iterations = int(math.ceil(float(max_samples) / per_iter))
        if num_iterations is None:
            num_iterations = 1
        num_iterations = max(1, int(num_iterations))

        start = time.time()
        for it in range(1, num_iterations + 1):
            rollout_stats = self.collect(horizon_length=horizon)
            update_stats = self.update()

            record: Dict[str, float] = {}
            record.update(rollout_stats)
            record.update(update_stats)
            record["iteration"] = float(it)
            record["samples"] = float(self.total_samples)
            record["samples_target"] = float(max_samples or 0)
            record["wall_time"] = time.time() - start
            record["num_envs"] = float(num_envs)
            self._log(record, it)
            self.history.append(record)

            if verbose and (it == 1 or it % max(1, int(self.config.log_interval)) == 0):
                msg = (
                    f"[PQL] iter {it}/{num_iterations} samples={self.total_samples:.3e} "
                    f"replay={len(self.replay)}"
                )
                for key in ("episode_return", "critic_loss", "actor_loss", "q_mean"):
                    if key in record:
                        msg += f" {key}={record[key]:.4g}"
                print(msg, flush=True)

            if max_samples is not None and self.total_samples >= int(max_samples):
                break

        return self.history

    # Aliases mirroring PPOBaselineTrainer's API.
    def train(self, num_iterations: Optional[int] = None, max_samples: Optional[int] = None,
              verbose: bool = False) -> List[Dict[str, float]]:
        return self.learn(num_iterations=num_iterations, max_samples=max_samples, verbose=verbose)

    def train_samples(self, max_samples: int, verbose: bool = False) -> List[Dict[str, float]]:
        return self.learn(max_samples=int(max_samples), verbose=verbose)

    def run(self, max_samples: Optional[int] = None, num_iterations: Optional[int] = None,
            verbose: bool = False) -> List[Dict[str, float]]:
        return self.learn(num_iterations=num_iterations, max_samples=max_samples, verbose=verbose)

    def _log(self, record: Dict[str, float], iteration: int) -> None:
        if self.logger is None:
            return
        for hook in ("log", "add_scalars", "log_metrics", "record"):
            fn = getattr(self.logger, hook, None)
            if callable(fn):
                try:
                    fn(record, iteration) if hook != "record" else fn(record)
                    return
                except TypeError:
                    try:
                        fn(record)
                        return
                    except Exception:
                        continue
                except Exception:
                    continue

    # ---- evaluation / persistence ---------------------------------------
    def evaluate(self, num_episodes: int = 1, deterministic: bool = True, **kwargs: Any) -> Dict[str, float]:
        """Greedy evaluation rollouts on the training environment."""
        if not HAS_TORCH:  # pragma: no cover
            return {}
        returns: List[float] = []
        lengths: List[int] = []
        obs = self._to_tensor(self.env.reset())
        ep_ret = torch.zeros(obs.shape[0], device=self.device)
        ep_len = torch.zeros(obs.shape[0], device=self.device)
        max_steps = int(kwargs.get("max_steps", 1000 * max(1, num_episodes)))
        for _ in range(max_steps):
            actions = self.select_actions(obs, deterministic=deterministic)
            step_out = self.env.step(actions)
            next_obs, _rewards, dones, _infos = step_out[:4]
            obs = self._to_tensor(next_obs)
            rewards_t = self._to_tensor(_rewards).reshape(-1)
            dones_t = self._to_tensor(dones).reshape(-1).float()
            ep_ret = ep_ret + rewards_t
            ep_len = ep_len + 1.0
            done = dones_t.bool()
            if bool(done.any()):
                idx = done.nonzero(as_tuple=False).reshape(-1)
                returns.extend([float(v) for v in ep_ret[idx].tolist()])
                lengths.extend([int(v) for v in ep_len[idx].tolist()])
                ep_ret[idx] = 0.0
                ep_len[idx] = 0.0
            if len(returns) >= num_episodes * max(1, int(obs.shape[0])):
                break
        if not returns:
            return {"episode_return": float("nan"), "episode_length": float("nan")}
        return {
            "episode_return": float(sum(returns) / len(returns)),
            "episode_return_max": float(max(returns)),
            "episode_length": float(sum(lengths) / max(1, len(lengths))),
            "episodes": float(len(returns)),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "config": self.config.to_dict(),
            "history": list(self.history),
            "total_samples": self.total_samples,
            "total_env_steps": self.total_env_steps,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name in ("actor", "critic", "actor_target", "critic_target"):
            net = getattr(self, name)
            if state.get(name) is not None:
                try:
                    net.load_state_dict(state[name])
                except Exception:
                    pass
        for opt_name in ("optimizer_actor", "optimizer_critic"):
            if state.get(opt_name) is not None:
                try:
                    getattr(self, opt_name).load_state_dict(state[opt_name])
                except Exception:
                    pass
        self.history = list(state.get("history", self.history))
        self.total_samples = int(state.get("total_samples", self.total_samples))
        self.total_env_steps = int(state.get("total_env_steps", self.total_env_steps))
        self.update_count = int(state.get("update_count", self.update_count))

    def save(self, path: str) -> str:
        if not HAS_TORCH:  # pragma: no cover
            raise RuntimeError("PQLTrainer.save requires torch")
        try:  # pragma: no cover
            from ..utils.config import ensure_dir

            ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        except Exception:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, map_location: Any = None) -> "PQLTrainer":
        if not HAS_TORCH:  # pragma: no cover
            raise RuntimeError("PQLTrainer.load requires torch")
        state = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(state)
        return self


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_pql_config(
    config: Any = None,
    task: str = "regrasping",
    num_envs: Optional[int] = None,
    seed: Optional[int] = None,
    **overrides: Any,
) -> PQLConfig:
    """Build a :class:`PQLConfig` (inheriting task defaults where available)."""
    base: Any = config
    if base is None:
        try:  # pragma: no cover - optional project dependency
            from ..utils.config import build_config

            base = build_config(task, **{k: v for k, v in overrides.items()})
        except Exception:
            base = None
    cfg = PQLConfig.from_any(base, **overrides)
    cfg.task = str(task)
    cfg.method = "pql"
    if num_envs is not None:
        cfg.num_envs = int(num_envs)
    if seed is not None:
        cfg.seed = int(seed)
    if not cfg.obs_dim or not cfg.action_dim:
        try:  # pragma: no cover
            from ..envs.isaac_env import action_dim_for, obs_dim_for

            cfg.obs_dim = int(obs_dim_for(task))
            cfg.action_dim = int(action_dim_for(task))
        except Exception:
            pass
    cfg.__post_init__()
    return cfg


def make_pql_policy(config: PQLConfig, device: Any = None) -> Tuple[Any, Any]:
    """Return ``(actor, critic)`` for a PQL run."""
    dev = device or config.device
    actor = DeterministicActor(
        config.obs_dim,
        config.action_dim,
        mlp_units=config.actor_mlp_units,
        activation=config.activation,
        config=config,
    )
    critic = QNetwork(
        config.obs_dim,
        config.action_dim,
        mlp_units=config.critic_mlp_units,
        activation=config.activation,
        config=config,
    )
    try:
        actor.to(dev)
        critic.to(dev)
    except Exception:
        pass
    return actor, critic


def make_pql_env(config: PQLConfig, num_envs: Optional[int] = None, **overrides: Any) -> Any:
    """Build the vectorized environment used by the PQL baseline."""
    try:  # pragma: no cover - optional project dependency
        from ..envs import make_env

        kwargs = dict(overrides)
        if num_envs is not None:
            kwargs["num_envs"] = int(num_envs)
        return make_env(config.task, num_envs=num_envs or config.num_envs, **kwargs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not construct a PQL environment: {exc}") from exc


def train_pql(
    config: Any = None,
    env: Any = None,
    policy: Any = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[int] = None,
    verbose: bool = False,
    logger: Any = None,
    device: Any = None,
    num_envs: Optional[int] = None,
    seed: Optional[int] = None,
    return_result: bool = False,
    trainer: Optional[PQLTrainer] = None,
    **overrides: Any,
) -> Any:
    """Train the PQL baseline. Mirrors ``train_ppo_baseline``'s signature.

    Returns ``(trainer, history)`` by default, or a :class:`PQLResult` when
    ``return_result=True``.
    """
    if trainer is None:
        cfg = make_pql_config(config, num_envs=num_envs, seed=seed, **overrides)
        actor, critic = (None, None)
        if policy is not None:
            if isinstance(policy, (tuple, list)) and len(policy) == 2:
                actor, critic = policy
            else:
                actor = policy
        trainer = PQLTrainer(
            config=cfg,
            env=env,
            actor=actor,
            critic=critic,
            logger=logger,
            device=device,
            seed=seed,
        )
    history = trainer.learn(
        num_iterations=num_iterations, max_samples=max_samples, verbose=verbose
    )
    if return_result:
        return PQLResult(
            history=history,
            samples=trainer.total_samples,
            num_envs=trainer.num_envs,
            seed=trainer.seed,
            trainer=trainer,
        )
    return trainer, history


def run_pql_seeds(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    config: Any = None,
    num_envs: Optional[int] = None,
    max_samples: Optional[int] = None,
    num_iterations: Optional[int] = None,
    verbose: bool = False,
    trainer_factory: Optional[Callable[..., Any]] = None,
    **overrides: Any,
) -> List[PQLResult]:
    """Run 5 seeds (Section 5.2 protocol) and collect per-seed results."""
    results: List[PQLResult] = []
    for seed in seeds:
        cfg = make_pql_config(config, num_envs=num_envs, seed=int(seed), **overrides)
        if trainer_factory is not None:
            trainer = trainer_factory(config=cfg, seed=int(seed))
        else:
            trainer = PQLTrainer(config=cfg, seed=int(seed))
        history = trainer.learn(
            num_iterations=num_iterations, max_samples=max_samples, verbose=verbose
        )
        results.append(
            PQLResult(
                history=history,
                samples=getattr(trainer, "total_samples", 0),
                num_envs=getattr(trainer, "num_envs", 0),
                seed=int(seed),
                trainer=trainer,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation (paper's shaded-band rule, Section 5.2)
# ---------------------------------------------------------------------------
def paper_standard_error(curves: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Mean curve plus the paper's literal shaded-band width.

    Section 5.2 defines the solid line ``y(t) = (1/n) sum_i y_i(t)`` and the
    shaded band ``(2 / sqrt(n)) * sum_i (y(t) - y_i(t))^2``.
    """
    if not curves:
        return [], []
    length = min(len(c) for c in curves)
    if length == 0:
        return [], []
    n = len(curves)
    mean: List[float] = []
    band: List[float] = []
    for t in range(length):
        vals = [float(c[t]) for c in curves]
        m = sum(vals) / n
        variance_like = sum((m - v) ** 2 for v in vals)
        mean.append(m)
        band.append((2.0 / math.sqrt(n)) * variance_like)
    return mean, band


def aggregate_seed_histories(
    results: Sequence[Any],
    key: str = "episode_return",
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate several seeds' histories into mean/band curves via the paper rule."""
    curves: List[List[float]] = []
    xs_common: Optional[List[float]] = None
    for res in results:
        history = res.history if hasattr(res, "history") else res
        ys = [float(rec.get(key, float("nan"))) for rec in history]
        xs = [float(rec.get("samples", i)) for i, rec in enumerate(history)]
        if xs_common is None or len(xs) < len(xs_common):
            xs_common = xs
        curves.append(ys)
    if not curves:
        return {"samples": [], "mean": [], "band": [], "seed_curves": []}
    if num_bins is not None and xs_common is not None and len(xs_common) > num_bins:
        # Uniformly resample the common x-grid.
        step = len(xs_common) / float(num_bins)
        idx = [int(i * step) for i in range(num_bins)]
        xs_common = [xs_common[i] for i in idx]
        curves = [[c[i] for i in idx if i < len(c)] for c in curves]
    mean, band = paper_standard_error(curves)
    return {
        "samples": xs_common or [],
        "mean": mean,
        "band": band,
        "seed_curves": curves,
        "key": key,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="SAPG -- Parallel Q-Learning (PQL) baseline")
    parser.add_argument("--task", default="regrasping")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=None, help="run this many seeds")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = make_pql_config(
        None,
        task=args.task,
        num_envs=args.num_envs,
        seed=args.seed,
    )
    if args.seeds and args.seeds > 1:
        results = run_pql_seeds(
            seeds=tuple(range(args.seeds)),
            config=cfg,
            max_samples=int(args.max_samples) if args.max_samples else None,
            num_iterations=args.iterations,
            verbose=args.verbose,
        )
        agg = aggregate_seed_histories(results)
        for seed_res in results:
            print(f"seed={seed_res.seed} final={seed_res.final():.6g}")
        print(f"aggregate final mean={agg['mean'][-1] if agg['mean'] else float('nan'):.6g}")
        return 0

    trainer, history = train_pql(
        config=cfg,
        num_iterations=args.iterations,
        max_samples=int(args.max_samples) if args.max_samples else None,
        verbose=args.verbose,
        device=args.device,
    )
    print(f"PQL finished: samples={trainer.total_samples:.3e} updates={trainer.update_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
