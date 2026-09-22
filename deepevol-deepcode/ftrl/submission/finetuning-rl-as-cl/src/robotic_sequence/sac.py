"""Soft Actor-Critic (SAC) for the RoboticSequence benchmark.

This module implements the off-policy learner used for all Meta-World /
RoboticSequence experiments of *Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem* (Wolczyk et al., 2024).

Everything follows Appendix B.3 of the paper:

* Soft Actor-Critic (Haarnoja et al., 2018a) with the Continual World
  architecture (Wolczyk et al., 2021): 4-layer MLPs with 256 hidden neurons,
  Leaky-ReLU activations, layer normalization applied after the *first* layer.
* The entropy coefficient is tuned automatically (Haarnoja et al., 2018b).
* A separate output head is created for every stage in both the policy and the
  Q-networks; the stage ID selects the head that is used.
* Learning rate ``1e-3`` with the Adam optimizer (Kingma & Ba, 2014) and a
  batch size of 128 in all experiments.
* Terminal transitions (success *or* the time limit ``T = 200``) are not
  bootstrapped in the target Q-value.
* Knowledge retention is applied to the **actor only** (cf. Table 3):

    ===== =============== ================ ========
    Method actor reg. coef critic reg. coef memory
    ===== =============== ================ ========
    EWC   100             0                -
    BC    1               0                10000
    EM    -               -                10000
    ===== =============== ================ ========

The three retention mechanisms are integrated as *additive modifications* of
the actor objective:

* :class:`~src.retention.ewc.EWC` -- ``L_aux = sum_i F^i (theta_pre - theta)^2``
  (Appendix C.1), actor coefficient 100.
* :class:`~src.retention.behavioral_cloning.BehavioralCloning` --
  ``L_BC = E_{s ~ B}[KL^s(pi_theta || pi_*)]`` (Appendix C.2), actor
  coefficient 1, buffer of 10000 pre-training samples.
* :class:`~src.retention.episodic_memory.EpisodicMemory` -- no auxiliary loss;
  10k prior-stage tuples are inserted into the replay buffer and the first 10%
  of the (100k) buffer is protected from being overwritten (Appendix C.3).

The module is deliberately dependency-light: ``torch`` is imported defensively
so that the file can be imported (for documentation / static tooling) even in
environments without PyTorch.  Constructing or running an agent requires torch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - defensive import
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Normal = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - numpy is used for the replay storage
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


__all__ = [
    "SACConfig",
    "SACBatch",
    "MLPTrunk",
    "PerStageHeads",
    "SquashedNormal",
    "SACPolicy",
    "SACTwinQ",
    "ReplayBuffer",
    "SACAgent",
    "build_sac_agent",
    "LOG_STD_MIN",
    "LOG_STD_MAX",
]

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required to use src.robotic_sequence.sac but could not be imported."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    """Hyperparameters of the SAC learner (Appendix B.3, Table 3).

    Defaults match the paper's RoboticSequence configuration.
    """

    hidden_dim: int = 256
    num_hidden_layers: int = 4
    activation: str = "leaky_relu"
    leaky_relu_slope: float = 0.01
    layer_norm_after_first: bool = True
    per_stage_heads: bool = True

    lr: float = 1e-3
    optimizer: str = "adam"
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005

    automatic_entropy_tuning: bool = True
    target_entropy: Optional[float] = None
    initial_alpha: float = 0.2
    alpha_lr: float = 1e-3
    entropy_enabled: bool = True

    buffer_size: int = 100_000
    protected_fraction: float = 0.0  # >0 only for episodic memory

    grad_clip: Optional[float] = None
    policy_delay: int = 1
    updates_per_step: int = 1
    use_twin_q: bool = True

    # --- retention (Table 3) -------------------------------------------------
    retention_method: str = "none"  # none | ewc | bc | ks | em
    ewc_actor_coef: float = 100.0
    ewc_critic_coef: float = 0.0
    bc_actor_coef: float = 1.0
    bc_critic_coef: float = 0.0
    bc_memory_size: int = 10_000
    em_memory_size: int = 10_000
    em_fraction: float = 0.1

    device: str = "cpu"
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, cfg: Any, **overrides: Any) -> "SACConfig":
        """Builds a :class:`SACConfig` from a nested config/mapping.

        ``cfg`` may be a ``Config`` object (see ``src/common/config.py``), a
        plain mapping, or an object exposing attributes.  The ``sac`` sub-map is
        read first, then the ``retention`` sub-map supplies the method-specific
        coefficients.
        """

        def get(container: Any, key: str, default: Any = None) -> Any:
            if container is None:
                return default
            if isinstance(container, Mapping):
                return container.get(key, default)
            return getattr(container, key, default)

        sac = get(cfg, "sac", cfg)
        retention = get(cfg, "retention", None)

        values: Dict[str, Any] = {}
        for name in (
            "hidden_dim",
            "num_hidden_layers",
            "activation",
            "leaky_relu_slope",
            "layer_norm_after_first",
            "per_stage_heads",
            "lr",
            "optimizer",
            "batch_size",
            "gamma",
            "tau",
            "automatic_entropy_tuning",
            "target_entropy",
            "initial_alpha",
            "alpha_lr",
            "buffer_size",
            "grad_clip",
            "policy_delay",
            "updates_per_step",
            "use_twin_q",
            "device",
            "seed",
        ):
            value = get(sac, name, None)
            if value is not None:
                values[name] = value

        if retention is not None:
            method = get(retention, "method", None)
            if method:
                values["retention_method"] = str(method).lower()
            ewc = get(retention, "ewc", None)
            if ewc is not None:
                coef = get(ewc, "actor_coef", None)
                if coef is not None:
                    values["ewc_actor_coef"] = float(coef)
                values["ewc_critic_coef"] = float(get(ewc, "critic_coef", 0.0) or 0.0)
            bc = get(retention, "bc", None)
            if bc is not None:
                coef = get(bc, "actor_coef", None)
                if coef is not None:
                    values["bc_actor_coef"] = float(coef)
                values["bc_critic_coef"] = float(get(bc, "critic_coef", 0.0) or 0.0)
                memory = get(bc, "memory_size", None)
                if memory is not None:
                    values["bc_memory_size"] = int(memory)
            em = get(retention, "em", None)
            if em is not None:
                memory = get(em, "memory_size", None)
                if memory is not None:
                    values["em_memory_size"] = int(memory)
                fraction = get(em, "fraction", None)
                if fraction is not None:
                    values["em_fraction"] = float(fraction)

        # episodic memory reserves a protected slice of the replay buffer
        if values.get("retention_method") == "em":
            values["protected_fraction"] = float(values.get("em_fraction", 0.1))

        values.update(overrides)
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in values.items() if k in allowed})


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _make_activation(name: str, slope: float = 0.01) -> Any:
    name = (name or "leaky_relu").lower()
    if name in ("leaky_relu", "leakyrelu", "lrelu"):
        return lambda: nn.LeakyReLU(slope)
    if name in ("relu",):
        return lambda: nn.ReLU()
    if name in ("elu",):
        return lambda: nn.ELU()
    if name in ("tanh",):
        return lambda: nn.Tanh()
    if name in ("gelu",):
        return lambda: nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPTrunk(nn.Module):
    """4-layer MLP with 256 hidden units, Leaky-ReLU, LayerNorm after layer 1.

    ``num_hidden_layers`` counts the hidden layers (default 4, per Appendix B.3).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.01,
        layer_norm_after_first: bool = True,
    ) -> None:
        super().__init__()
        _require_torch()
        act_factory = _make_activation(activation, leaky_relu_slope)
        layers: List[Any] = []
        last = in_dim
        for i in range(max(1, int(num_hidden_layers))):
            layers.append(nn.Linear(last, hidden_dim))
            if i == 0 and layer_norm_after_first:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_factory())
            last = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: Any) -> Any:
        return self.net(x)


class PerStageHeads(nn.Module):
    """A container with one linear output head per stage.

    The stage ID (a one-hot vector or an integer index) selects the head; this is
    the mechanism described in Appendix B.3 ("we create a separate output head
    for each stage in the neural networks and then we use the stage ID
    information to choose the correct head").
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_stages: int = 1,
        init_scale: float = 1.0,
        per_stage: bool = True,
    ) -> None:
        super().__init__()
        _require_torch()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_stages = max(1, int(n_stages))
        self.per_stage = bool(per_stage) and self.n_stages > 1
        if self.per_stage:
            self.heads = nn.ModuleList(
                [nn.Linear(self.in_dim, self.out_dim) for _ in range(self.n_stages)]
            )
        else:
            self.heads = nn.ModuleList([nn.Linear(self.in_dim, self.out_dim)])
        with torch.no_grad():
            for head in self.heads:
                head.weight.mul_(init_scale)
                head.bias.mul_(init_scale)

    @staticmethod
    def stage_index(stage_id: Any, n_stages: int) -> Any:
        """Normalises ``stage_id`` into a ``(batch,)`` long tensor of indices."""

        if stage_id is None:
            return torch.zeros(1, dtype=torch.long)
        if torch.is_tensor(stage_id):
            idx = stage_id
        else:
            idx = torch.as_tensor(stage_id)
        if idx.dim() == 2:  # one-hot
            idx = idx.argmax(dim=-1)
        idx = idx.reshape(-1).long()
        if n_stages > 1 and int(idx.max()) >= n_stages:
            # tolerate transient off-by-one from the env's stage bookkeeping
            idx = idx.clamp(0, n_stages - 1)
        return idx

    def forward(self, features: Any, stage_id: Any = None) -> Any:
        idx = self.stage_index(stage_id, self.n_stages)
        if idx.numel() != features.shape[0]:
            if idx.numel() == 1:
                idx = idx.expand(features.shape[0])
            else:
                idx = idx[: features.shape[0]]
        if not self.per_stage:
            return self.heads[0](features)
        out = features.new_zeros((features.shape[0], self.out_dim))
        for stage in range(self.n_stages):
            mask = idx == stage
            if bool(mask.any()):
                out = out.index_copy(0, mask.nonzero(as_tuple=True)[0], self.heads[stage](features[mask]))
        return out


class SquashedNormal:
    """Tanh-squashed Gaussian policy distribution.

    Provides ``rsample``/``sample`` in the bounded action space together with the
    exact change-of-variables correction to the log-density:

    ``log pi(a|s) = log N(z; mu, sigma) - sum log(1 - tanh(z)^2 + eps)``
    """

    def __init__(self, mean: Any, log_std: Any, batch_dims: int = 1) -> None:
        _require_torch()
        self.mean = mean
        self.log_std = log_std
        self.std = log_std.exp()
        self._normal = Normal(mean, self.std)
        self.batch_dims = batch_dims
        self.mean_action = torch.tanh(mean)

    # -- convenience accessors used by the retention losses -----------------
    @property
    def dist(self) -> Any:
        return self

    @property
    def loc(self) -> Any:
        return self.mean

    @property
    def scale(self) -> Any:
        return self.std

    def sample(self, sample_shape: Any = None) -> Any:
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: Any = None) -> Any:
        z = self._normal.rsample(sample_shape) if sample_shape is not None else self._normal.rsample()
        return torch.tanh(z)

    def sample_z(self) -> Any:
        return self._normal.rsample()

    def log_prob(self, action: Any) -> Any:
        action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        z = torch.atanh(action)
        log_prob = self._normal.log_prob(z) - torch.log(1.0 - action.pow(2) + EPS)
        if self.batch_dims > 0:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def sample_with_log_prob(self, sample_shape: Any = None) -> Tuple[Any, Any]:
        z = self._normal.rsample(sample_shape) if sample_shape is not None else self._normal.rsample()
        action = torch.tanh(z)
        log_prob = self._normal.log_prob(z) - torch.log(1.0 - action.pow(2) + EPS)
        if self.batch_dims > 0:
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def entropy(self) -> Any:
        return self._normal.entropy().sum(dim=-1)

    def kl_divergence(self, other: "SquashedNormal") -> Any:
        """KL(self || other) estimated from the normal components (analytic)."""
        return torch.distributions.kl_divergence(self._normal, other._normal).sum(dim=-1)


class SACPolicy(nn.Module):
    """Gaussian policy ``pi_theta(a|s)`` with per-stage output heads.

    ``forward`` returns a :class:`SquashedNormal` (a torch-compatible
    distribution object exposing ``.dist``, ``.loc``/``.scale``, ``log_prob``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_stages: int = 1,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.01,
        layer_norm_after_first: bool = True,
        per_stage_heads: bool = True,
        stage_id_in_obs: bool = False,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        _require_torch()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_stages = max(1, int(n_stages))
        self.stage_id_in_obs = bool(stage_id_in_obs)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.trunk = MLPTrunk(
            self.obs_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            leaky_relu_slope=leaky_relu_slope,
            layer_norm_after_first=layer_norm_after_first,
        )
        self.mean_head = PerStageHeads(
            hidden_dim, self.action_dim, self.n_stages, per_stage=per_stage_heads
        )
        self.log_std_head = PerStageHeads(
            hidden_dim, self.action_dim, self.n_stages, per_stage=per_stage_heads
        )
        with torch.no_grad():
            self.log_std_head.heads[0].bias.fill_(-1.0) if not self.log_std_head.per_stage else None

    # -- helpers -------------------------------------------------------------
    def resolve_stage_id(self, obs: Any, stage_id: Any = None) -> Any:
        """Falls back to reading the stage one-hot from the tail of ``obs``."""

        if stage_id is not None:
            return stage_id
        if self.stage_id_in_obs and self.n_stages > 1 and obs.shape[-1] >= self.n_stages:
            return obs[..., -self.n_stages :].argmax(dim=-1)
        return None

    def features(self, obs: Any) -> Any:
        return self.trunk(obs)

    def distribution(self, obs: Any, stage_id: Any = None) -> SquashedNormal:
        stage_id = self.resolve_stage_id(obs, stage_id)
        feats = self.trunk(obs)
        mean = self.mean_head(feats, stage_id)
        log_std = self.log_std_head(feats, stage_id)
        if self.n_stages == 1 and not self.mean_head.per_stage:
            pass
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return SquashedNormal(mean, log_std)

    def forward(self, obs: Any, stage_id: Any = None) -> SquashedNormal:
        return self.distribution(obs, stage_id)

    def sample(self, obs: Any, stage_id: Any = None) -> Tuple[Any, Any]:
        dist = self.distribution(obs, stage_id)
        return dist.sample_with_log_prob()

    def mean_action(self, obs: Any, stage_id: Any = None) -> Any:
        dist = self.distribution(obs, stage_id)
        return dist.mean_action

    def act(self, obs: Any, stage_id: Any = None, deterministic: bool = False) -> Any:
        dist = self.distribution(obs, stage_id)
        if deterministic:
            return dist.mean_action
        action, _ = dist.sample_with_log_prob()
        return action

    def log_prob(self, obs: Any, action: Any, stage_id: Any = None) -> Any:
        return self.distribution(obs, stage_id).log_prob(action)

    def get_actions(self, obs: Any, stage_id: Any = None, deterministic: bool = False) -> Tuple[Any, Any]:
        dist = self.distribution(obs, stage_id)
        if deterministic:
            return dist.mean_action, dist.log_prob(dist.mean_action)
        return dist.sample_with_log_prob()


class SACTwinQ(nn.Module):
    """Twin Q-networks ``Q1``, ``Q2`` with per-stage heads.

    Each of the two critics has its own trunk (as in the reference SAC
    implementation) and one output head per stage.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_stages: int = 1,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.01,
        layer_norm_after_first: bool = True,
        per_stage_heads: bool = True,
        use_twin_q: bool = True,
    ) -> None:
        super().__init__()
        _require_torch()
        self.n_stages = max(1, int(n_stages))
        self.use_twin_q = bool(use_twin_q)

        def build() -> Any:
            trunk = MLPTrunk(
                int(obs_dim) + int(action_dim),
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
                leaky_relu_slope=leaky_relu_slope,
                layer_norm_after_first=layer_norm_after_first,
            )
            head = PerStageHeads(hidden_dim, 1, self.n_stages, per_stage=per_stage_heads)
            return nn.ModuleDict({"trunk": trunk, "head": head})

        self.q1 = build()
        self.q2 = build() if self.use_twin_q else None

    def _one(
        self, module: Mapping[str, Any], obs: Any, action: Any, stage_id: Any
    ) -> Any:
        x = torch.cat([obs, action], dim=-1)
        feats = module["trunk"](x)
        return module["head"](feats, stage_id).squeeze(-1)

    def forward(self, obs: Any, action: Any, stage_id: Any = None) -> Tuple[Any, Any]:
        q1 = self._one(self.q1, obs, action, stage_id)
        if self.q2 is None:
            return q1, q1
        q2 = self._one(self.q2, obs, action, stage_id)
        return q1, q2

    def q1_value(self, obs: Any, action: Any, stage_id: Any = None) -> Any:
        return self._one(self.q1, obs, action, stage_id)


# ---------------------------------------------------------------------------
# Replay buffer (with protected prior-task region for episodic memory)
# ---------------------------------------------------------------------------


@dataclass
class SACBatch:
    """A mini-batch sampled from the replay buffer."""

    obs: Any
    actions: Any
    rewards: Any
    next_obs: Any
    dones: Any
    stage_ids: Any
    next_stage_ids: Any
    weights: Any = None
    indices: Any = None

    def to(self, device: Any) -> "SACBatch":
        def move(value: Any) -> Any:
            if value is None:
                return None
            if _HAS_TORCH and torch.is_tensor(value):
                return value.to(device)
            return value

        return SACBatch(
            obs=move(self.obs),
            actions=move(self.actions),
            rewards=move(self.rewards),
            next_obs=move(self.next_obs),
            dones=move(self.dones),
            stage_ids=move(self.stage_ids),
            next_stage_ids=move(self.next_stage_ids),
            weights=move(self.weights),
            indices=self.indices,
        )

    def __len__(self) -> int:
        try:
            return int(self.obs.shape[0])
        except Exception:  # pragma: no cover
            return len(self.obs)


class ReplayBuffer:
    """Replay buffer with an optional protected (never overwritten) prefix.

    Following Appendix C.3, episodic memory reserves ``fraction`` of the buffer
    for trajectories gathered with the pre-trained policy ``pi_*``; those slots
    are protected for the whole fine-tuning run.  Every batch is sampled with a
    guaranteed number of prior-task transitions (``round(fraction * batch)``).
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        n_stages: int = 1,
        fraction: float = 0.0,
        device: str = "cpu",
        seed: Optional[int] = None,
        storage: str = "numpy",
    ) -> None:
        _require_torch()
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_stages = max(1, int(n_stages))
        self.fraction = float(fraction or 0.0)
        self.device = device
        self.storage = storage if _np is not None else "tensor"

        self.protected_capacity = int(round(self.fraction * self.capacity))
        self.n_prior = 0  # filled protected slots
        self.n_current = 0  # rolling FIFO slots (0 <= n_current <= capacity - protected_capacity)
        self.n_total = 0

        self._obs: Any = None
        self._next_obs: Any = None
        self._actions: Any = None
        self._rewards: Any = None
        self._dones: Any = None
        self._stage_ids: Any = None
        self._next_stage_ids: Any = None
        self._initialised = False

        if seed is None:
            self._rng = _np.random.default_rng() if _np is not None else None
        else:
            self._rng = _np.random.default_rng(seed) if _np is not None else None

    # -- allocation ----------------------------------------------------------
    def _allocate(self, obs: Any, action: Any, next_obs: Any) -> None:
        if self._initialised:
            return
        obs_shape = tuple(obs.shape)
        action_shape = tuple(action.shape)
        self.obs_dim = obs_shape[-1]
        self.action_dim = action_shape[-1]
        if self.storage == "numpy":
            self._obs = _np.zeros((self.capacity,) + obs_shape, dtype=_np.float32)
            self._next_obs = _np.zeros((self.capacity,) + tuple(next_obs.shape), dtype=_np.float32)
            self._actions = _np.zeros((self.capacity,) + action_shape, dtype=_np.float32)
            self._rewards = _np.zeros((self.capacity,), dtype=_np.float32)
            self._dones = _np.zeros((self.capacity,), dtype=_np.float32)
            self._stage_ids = _np.zeros((self.capacity,), dtype=_np.int64)
            self._next_stage_ids = _np.zeros((self.capacity,), dtype=_np.int64)
        else:  # pragma: no cover - tensor storage fallback
            self._obs = torch.zeros((self.capacity,) + obs_shape)
            self._next_obs = torch.zeros((self.capacity,) + tuple(next_obs.shape))
            self._actions = torch.zeros((self.capacity,) + action_shape)
            self._rewards = torch.zeros((self.capacity,))
            self._dones = torch.zeros((self.capacity,))
            self._stage_ids = torch.zeros((self.capacity,), dtype=torch.long)
            self._next_stage_ids = torch.zeros((self.capacity,), dtype=torch.long)
        self._initialised = True

    @staticmethod
    def _index_of(stage_id: Any) -> int:
        if stage_id is None:
            return 0
        if _np is not None and isinstance(stage_id, _np.ndarray):
            stage_id = stage_id.reshape(-1)
            if stage_id.size == 1:
                return int(stage_id[0])
            return int(stage_id.argmax())
        if _HAS_TORCH and torch.is_tensor(stage_id):
            flat = stage_id.reshape(-1)
            if flat.numel() == 1:
                return int(flat.item())
            return int(flat.argmax().item())
        if isinstance(stage_id, (list, tuple)):
            if len(stage_id) == 1:
                return int(stage_id[0])
            return int(max(range(len(stage_id)), key=lambda i: stage_id[i]))
        return int(stage_id)

    def _write(self, index: int, transition: Mapping[str, Any]) -> None:
        self._obs[index] = transition["obs"]
        self._next_obs[index] = transition["next_obs"]
        self._actions[index] = transition["action"]
        self._rewards[index] = float(transition["reward"])
        self._dones[index] = float(transition["done"])
        self._stage_ids[index] = self._index_of(transition.get("stage_id"))
        self._next_stage_ids[index] = self._index_of(
            transition.get("next_stage_id", transition.get("stage_id"))
        )

    # -- insertion -----------------------------------------------------------
    def add(self, transition: Mapping[str, Any], protected: bool = False) -> int:
        """Adds a single transition; ``protected=True`` writes into the prior region."""

        if not self._initialised and "obs" in transition:
            self._allocate(
                _np.asarray(transition["obs"], dtype=_np.float32) if self.storage == "numpy" else torch.as_tensor(transition["obs"]),
                _np.asarray(transition["action"], dtype=_np.float32) if self.storage == "numpy" else torch.as_tensor(transition["action"]),
                _np.asarray(transition["next_obs"], dtype=_np.float32) if self.storage == "numpy" else torch.as_tensor(transition["next_obs"]),
            )

        if protected:
            if self.n_prior >= self.protected_capacity:
                return -1  # prior region full: prior data is never overwritten
            index = self.n_prior
            self.n_prior += 1
        else:
            rolling = max(1, self.capacity - self.protected_capacity)
            index = self.protected_capacity + (self.n_current % rolling)
            self.n_current += 1
        self._write(index, transition)
        self.n_total = min(self.capacity, self.n_prior + self.n_current)
        return index

    def add_batch(self, transitions: Sequence[Mapping[str, Any]], protected: bool = False) -> int:
        count = 0
        for transition in transitions:
            if self.add(transition, protected=protected) >= 0:
                count += 1
        return count

    def add_raw(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
        stage_id: Any = 0,
        next_stage_id: Any = None,
        protected: bool = False,
    ) -> int:
        return self.add(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "stage_id": stage_id,
                "next_stage_id": stage_id if next_stage_id is None else next_stage_id,
            },
            protected=protected,
        )

    def set_prior_data(self, transitions: Sequence[Mapping[str, Any]]) -> int:
        """Fills the protected region with ``pi_*`` transitions."""

        return self.add_batch(transitions, protected=True)

    @property
    def protected_indices(self) -> List[int]:
        return list(range(self.n_prior))

    def is_protected(self, index: int) -> bool:
        return index < self.protected_capacity

    def __len__(self) -> int:
        return int(self.n_total)

    # -- sampling ------------------------------------------------------------
    def sample_indices(self, batch_size: int, generator: Any = None) -> List[int]:
        """Samples indices, guaranteeing prior-task data in every batch."""

        rng = generator if generator is not None else self._rng
        batch_size = int(batch_size)
        if self.n_total == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        ranges = [
            (0, self.n_prior),
            (self.protected_capacity, self.protected_capacity + min(self.n_current, max(1, self.capacity - self.protected_capacity))),
        ]
        available = [(lo, hi) for lo, hi in ranges if hi > lo]
        if not available:
            return list(self._choice(self.n_total, batch_size, replace=self.n_total < batch_size, rng=rng))

        n_prior_target = 0
        if self.fraction > 0 and self.n_prior > 0:
            n_prior_target = max(1, int(round(self.fraction * batch_size)))
            n_prior_target = min(n_prior_target, batch_size)

        indices: List[int] = []
        if n_prior_target:
            indices.extend(self._choice(self.n_prior, n_prior_target, replace=self.n_prior < n_prior_target, rng=rng))
        remaining = batch_size - len(indices)
        if remaining > 0:
            if self.n_current > 0:
                rolling = self.protected_capacity
                hi = self.protected_capacity + min(self.n_current, max(1, self.capacity - self.protected_capacity))
                pool = list(range(rolling, hi))
                indices.extend(self._choice_from(pool, remaining, replace=len(pool) < remaining, rng=rng))
            else:
                indices.extend(
                    self._choice(self.n_prior, remaining, replace=self.n_prior < remaining, rng=rng)
                )
        return [int(i) for i in indices]

    @staticmethod
    def _choice(n: int, size: int, replace: bool, rng: Any) -> List[int]:
        if _np is not None:
            rng = rng if rng is not None else _np.random.default_rng()
            return list(rng.choice(int(n), size=int(size), replace=bool(replace)).tolist())
        import random as _random  # pragma: no cover

        rng = rng or _random
        return [rng.randrange(int(n)) for _ in range(int(size))]  # pragma: no cover

    @staticmethod
    def _choice_from(pool: Sequence[int], size: int, replace: bool, rng: Any) -> List[int]:
        if _np is not None:
            rng = rng if rng is not None else _np.random.default_rng()
            arr = _np.asarray(pool)
            return list(arr[rng.choice(len(arr), size=int(size), replace=bool(replace))].tolist())
        import random as _random  # pragma: no cover

        rng = rng or _random
        return [rng.choice(pool) for _ in range(int(size))]  # pragma: no cover

    def _gather(self, indices: Sequence[int], array: Any) -> Any:
        if _HAS_TORCH and torch.is_tensor(array):
            idx = torch.as_tensor(indices, dtype=torch.long, device=array.device)
            return array[idx]
        return _np.asarray(array)[_np.asarray(indices)]

    def sample(self, batch_size: int, generator: Any = None, device: Any = None) -> SACBatch:
        indices = self.sample_indices(batch_size, generator=generator)
        device = device if device is not None else self.device
        batch = SACBatch(
            obs=self._to_tensor(self._gather(indices, self._obs), device),
            actions=self._to_tensor(self._gather(indices, self._actions), device),
            rewards=self._to_tensor(self._gather(indices, self._rewards), device),
            next_obs=self._to_tensor(self._gather(indices, self._next_obs), device),
            dones=self._to_tensor(self._gather(indices, self._dones), device),
            stage_ids=self._to_tensor(self._gather(indices, self._stage_ids), device),
            next_stage_ids=self._to_tensor(self._gather(indices, self._next_stage_ids), device),
            indices=list(indices),
        )
        return batch

    @staticmethod
    def _to_tensor(value: Any, device: Any) -> Any:
        if _HAS_TORCH and not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if _HAS_TORCH and torch.is_tensor(value):
            value = value.to(device)
            if value.dtype in (torch.float64,):
                value = value.float()
            if value.dtype not in (torch.float32, torch.float16, torch.float64) and value.dtype != torch.long:
                value = value.float()
        return value

    # -- persistence ---------------------------------------------------------
    def clear(self, keep_prior: bool = False) -> None:
        self.n_current = 0
        self.n_total = self.n_prior if keep_prior else 0
        if not keep_prior:
            self.n_prior = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_prior": self.n_prior,
            "n_current": self.n_current,
            "n_total": self.n_total,
            "capacity": self.capacity,
            "fraction": self.fraction,
            "obs": self._obs,
            "next_obs": self._next_obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "dones": self._dones,
            "stage_ids": self._stage_ids,
            "next_stage_ids": self._next_stage_ids,
            "initialised": self._initialised,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.n_prior = int(state.get("n_prior", 0))
        self.n_current = int(state.get("n_current", 0))
        self.n_total = int(state.get("n_total", 0))
        self._initialised = bool(state.get("initialised", False))
        for key in (
            "obs",
            "next_obs",
            "actions",
            "rewards",
            "dones",
            "stage_ids",
            "next_stage_ids",
        ):
            if key in state:
                setattr(self, "_" + key, state[key])

    def describe(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": len(self),
            "protected_capacity": self.protected_capacity,
            "n_prior": self.n_prior,
            "n_current": self.n_current,
            "fraction": self.fraction,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SACAgent:
    """Soft Actor-Critic agent with actor-only knowledge retention.

    Parameters
    ----------
    policy, q_network:
        Networks built with :class:`SACPolicy` / :class:`SACTwinQ`.
    config:
        :class:`SACConfig` with the hyperparameters of Appendix B.3.
    retention:
        Optional sequence of retention objects (EWC/BC/KS/EM).  Each item must be
        a callable ``loss(actor=policy)`` or expose ``loss()``/``penalty()``.
        Retention is *always* applied to the actor only; the critic coefficient is
        zero (Table 3).
    """

    def __init__(
        self,
        policy: Any,
        q_network: Any,
        config: Optional[SACConfig] = None,
        retention: Optional[Sequence[Any]] = None,
        device: Any = None,
        seed: Optional[int] = None,
    ) -> None:
        _require_torch()
        self.cfg = config or SACConfig()
        self.device = torch.device(device or self.cfg.device or "cpu")
        self.policy = policy.to(self.device)
        self.q_network = q_network.to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        for param in self.target_q_network.parameters():
            param.requires_grad_(False)

        betas = (0.9, 0.999)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.lr, betas=betas
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.cfg.lr, betas=betas
        )

        # --- automatic entropy tuning (Haarnoja et al., 2018b) --------------
        self.automatic_entropy_tuning = bool(self.cfg.automatic_entropy_tuning)
        action_dim = int(getattr(self.policy, "action_dim", 1))
        if self.cfg.target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(self.cfg.target_entropy)
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr, betas=betas)
            self.alpha = float(self.cfg.initial_alpha)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = float(self.cfg.initial_alpha)

        self.retention: List[Any] = list(retention or [])
        self.update_step = 0
        self.gradient_steps = 0

        self._rng = None
        if seed is not None and _np is not None:
            self._rng = _np.random.default_rng(seed)

    # -- helpers -------------------------------------------------------------
    @property
    def entropy_enabled(self) -> bool:
        return bool(self.cfg.entropy_enabled)

    def set_retention(self, retention: Sequence[Any]) -> None:
        """Attaches the retention mechanisms applied to the actor objective."""

        self.retention = list(retention or [])

    def add_retention(self, loss: Any) -> None:
        self.retention.append(loss)

    def _retention_loss(self) -> Any:
        """Sum of the actor-only auxiliary retention penalties."""

        total = None
        for loss in self.retention:
            value = None
            if callable(loss) and not hasattr(loss, "loss") and not hasattr(loss, "penalty"):
                value = loss(actor=self.policy)
            elif hasattr(loss, "loss"):
                try:
                    value = loss.loss(actor=self.policy)
                except TypeError:
                    value = loss.loss()
            elif hasattr(loss, "penalty"):
                try:
                    value = loss.penalty(actor=self.policy)
                except TypeError:
                    value = loss.penalty()
            elif callable(loss):  # pragma: no cover
                value = loss(self.policy)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            total = value if total is None else total + value
        if total is None:
            total = self._zero_loss()
        return total

    def _zero_loss(self) -> Any:
        params = list(self.policy.parameters())
        if not params:
            return torch.zeros((), device=self.device, requires_grad=True)
        return sum(p.sum() * 0.0 for p in params)

    def step_retention(self, n: int = 1) -> None:
        """Advances the retention schedules (used e.g. by kickstarting decay)."""

        for loss in self.retention:
            if hasattr(loss, "step"):
                try:
                    loss.step(n)
                except TypeError:  # pragma: no cover
                    pass

    # -- acting --------------------------------------------------------------
    def select_action(
        self,
        obs: Any,
        stage_id: Any = None,
        deterministic: bool = False,
        return_log_prob: bool = False,
    ) -> Any:
        def to_tensor(value: Any) -> Any:
            if value is None:
                return None
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
            tensor = tensor.to(self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        obs_t = to_tensor(obs)
        stage_ids = to_tensor(stage_id) if stage_id is not None else None
        with torch.no_grad():
            if return_log_prob:
                action, log_prob = self.policy.get_actions(
                    obs_t, stage_ids, deterministic=deterministic
                )
                return (
                    self._to_numpy(action)[0],
                    self._to_numpy(log_prob)[0],
                )
            action = self.policy.act(obs_t, stage_ids, deterministic=deterministic)
        return self._to_numpy(action)[0]

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if _HAS_TORCH and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return value

    # -- learning ------------------------------------------------------------
    def update(
        self,
        batch: SACBatch,
        retention: Optional[Sequence[Any]] = None,
        entropy_enabled: Optional[bool] = None,
    ) -> Dict[str, float]:
        """One SAC gradient step; returns a dict of scalar losses."""

        batch = batch.to(self.device)
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones
        stage_ids = batch.stage_ids
        next_stage_ids = batch.next_stage_ids

        entropy_on = self.entropy_enabled if entropy_enabled is None else bool(entropy_enabled)
        alpha = self.alpha if entropy_on else 0.0

        # --- critic update --------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_obs, next_stage_ids)
            q1_target, q2_target = self.target_q_network(next_obs, next_action, next_stage_ids)
            q_target = torch.min(q1_target, q2_target)
            if entropy_on:
                q_target = q_target - self.alpha * next_log_prob
            # terminal transitions (success or time limit) are NOT bootstrapped
            backup = rewards + (1.0 - dones) * self.cfg.gamma * q_target

        q1, q2 = self.q_network(obs, actions, stage_ids)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)
        critic_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.grad_clip)
        self.q_optimizer.step()

        # --- actor update ---------------------------------------------------
        metrics: Dict[str, float] = {
            "critic_loss": float(critic_loss.detach().item()),
            "q1_loss": float(q1_loss.detach().item()),
            "q2_loss": float(q2_loss.detach().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q_target_mean": float(backup.detach().mean().item()),
            "alpha": float(self.alpha),
        }

        retention_loss_value = 0.0
        retention_losses = list(self.retention if retention is None else retention)
        do_policy_update = (self.update_step % max(1, int(self.cfg.policy_delay))) == 0

        if do_policy_update:
            new_action, log_prob = self.policy.sample(obs, stage_ids)
            q1_pi, q2_pi = self.q_network(obs, new_action, stage_ids)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * log_prob - q_pi).mean()

            # actor-only knowledge retention (critic coefficient is always 0)
            aux_loss = None
            for loss in retention_losses:
                value = self._call_retention(loss)
                if value is None:
                    continue
                aux_loss = value if aux_loss is None else aux_loss + value
            if aux_loss is not None:
                retention_loss_value = float(aux_loss.detach().item())
                actor_loss = actor_loss + aux_loss

            self.policy_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
            self.policy_optimizer.step()

            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach().item()),
                    "policy_loss": float((alpha * log_prob - q_pi).detach().mean().item()),
                    "retention_loss": retention_loss_value,
                    "log_prob": float(log_prob.detach().mean().item()),
                    "entropy": float(-log_prob.detach().mean().item()),
                }
            )

            # --- entropy coefficient update ---------------------------------
            if self.automatic_entropy_tuning and entropy_on and self.log_alpha is not None:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().item())
                metrics["alpha_loss"] = float(alpha_loss.detach().item())
                metrics["alpha"] = float(self.alpha)

            self.step_retention(1)

        # --- target networks -------------------------------------------------
        self._soft_update()

        self.update_step += 1
        self.gradient_steps += 1
        return metrics

    def _call_retention(self, loss: Any) -> Any:
        value = None
        if hasattr(loss, "loss") and callable(getattr(loss, "loss")):
            try:
                value = loss.loss(actor=self.policy)
            except TypeError:
                value = loss.loss()
        elif hasattr(loss, "penalty") and callable(getattr(loss, "penalty")):
            try:
                value = loss.penalty(actor=self.policy)
            except TypeError:
                value = loss.penalty()
        elif callable(loss):
            try:
                value = loss(actor=self.policy)
            except TypeError:
                value = loss(self.policy)
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return value

    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = float(self.cfg.tau)
        for target, source in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
        for target, source in zip(
            self.target_q_network.buffers(), self.q_network.buffers()
        ):
            target.data.copy_(source.data)

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state = {
            "policy": self.policy.state_dict(),
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha": self.alpha,
            "update_step": self.update_step,
            "gradient_steps": self.gradient_steps,
        }
        if self.log_alpha is not None:
            state["log_alpha"] = self.log_alpha.detach().cpu()
            if self.alpha_optimizer is not None:
                state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any], load_optimizers: bool = True) -> None:
        self.policy.load_state_dict(state["policy"])
        self.q_network.load_state_dict(state["q_network"])
        if "target_q_network" in state:
            self.target_q_network.load_state_dict(state["target_q_network"])
        if load_optimizers:
            try:
                self.policy_optimizer.load_state_dict(state["policy_optimizer"])
                self.q_optimizer.load_state_dict(state["q_optimizer"])
            except Exception:  # pragma: no cover - optimizer state may be missing
                pass
        self.alpha = float(state.get("alpha", self.alpha))
        if self.log_alpha is not None and "log_alpha" in state:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.update_step = int(state.get("update_step", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))

    def load_pretrained(self, path_or_state: Any) -> None:
        """Loads the pre-trained model ``pi_*`` (actor + critics)."""

        if isinstance(path_or_state, str):
            state = torch.load(path_or_state, map_location=self.device)
        else:
            state = path_or_state
        if isinstance(state, Mapping) and "policy" in state and "q_network" not in state:
            state = {"policy": state}
        self.load_state_dict(state, load_optimizers=False)

    def save(self, path: str) -> str:
        torch.save(self.state_dict(), path)
        return path

    def describe(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "lr": self.cfg.lr,
            "batch_size": self.cfg.batch_size,
            "gamma": self.cfg.gamma,
            "tau": self.cfg.tau,
            "hidden_dim": self.cfg.hidden_dim,
            "num_hidden_layers": self.cfg.num_hidden_layers,
            "per_stage_heads": self.cfg.per_stage_heads,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "target_entropy": self.target_entropy,
            "alpha": self.alpha,
            "retention": [type(r).__name__ for r in self.retention],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_sac_agent(
    cfg: Any,
    obs_dim: int,
    action_dim: int,
    n_stages: int = 1,
    device: Optional[Any] = None,
    retention: Optional[Sequence[Any]] = None,
    seed: Optional[int] = None,
    stage_id_in_obs: bool = False,
) -> SACAgent:
    """Builds a :class:`SACAgent` from a config object (or ``SACConfig``).

    Mirrors Appendix B.3: 4 hidden layers of 256 units, Leaky-ReLU, layer norm
    after the first layer, one head per stage, automatic entropy tuning, Adam
    with learning rate ``1e-3`` and batch size 128.
    """

    config = cfg if isinstance(cfg, SACConfig) else SACConfig.from_config(cfg)
    if device is not None:
        config.device = str(device)
    if seed is not None:
        config.seed = seed

    policy = SACPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_stages=n_stages,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        leaky_relu_slope=config.leaky_relu_slope,
        layer_norm_after_first=config.layer_norm_after_first,
        per_stage_heads=config.per_stage_heads,
        stage_id_in_obs=stage_id_in_obs,
    )
    q_network = SACTwinQ(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_stages=n_stages,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        leaky_relu_slope=config.leaky_relu_slope,
        layer_norm_after_first=config.layer_norm_after_first,
        per_stage_heads=config.per_stage_heads,
        use_twin_q=config.use_twin_q,
    )
    return SACAgent(
        policy=policy,
        q_network=q_network,
        config=config,
        retention=retention,
        device=device or config.device,
        seed=seed if seed is not None else config.seed,
    )
