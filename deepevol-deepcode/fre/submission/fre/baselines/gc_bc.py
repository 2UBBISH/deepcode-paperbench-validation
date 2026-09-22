"""Goal-Conditioned Behavioral Cloning (GC-BC) baseline for the FRE benchmark.

Reference: FRE paper, Section 5.2 and the benchmark addendum section
"Additional Details on GC-BC":

* Network architecture
    - MLP with three hidden layers of size 512
    - ReLU activations between each hidden layer
    - Layer normalization applied *before* each activation
    - Output layer predicts a Gaussian over actions:
        * mean action: linear output
        * log standard deviation: clamped with a lower bound of -5.0
* Loss function: maximum likelihood estimation (MLE), i.e.
      L_pi = -E_{(s, g, a) ~ D} log pi(a | s, g)
* Training: hindsight relabeling with *geometric* goal sampling only, i.e. goals
  are future states sampled from a geometric distribution along the trajectory.
  No random goals and no goals equal to the current state.
* Evaluation: the agent conditions on the ground-truth goal of the evaluation
  task.

GC-BC is implemented within the same codebase as FRE / GC-IQL and shares the
same network structure conventions (512-unit hidden layers), the same dataset
pipeline (including AntMaze XY 32-bin discretization) and the same evaluation
harness, so its reported numbers are directly comparable to FRE.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is a hard runtime dependency but we keep the import guarded so
    import torch  # that pure-data utilities remain importable.
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch should always be installed
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Constants (paper / addendum defaults)
# ---------------------------------------------------------------------------

DEFAULT_GCBC_HIDDEN_DIMS: Tuple[int, ...] = (512, 512, 512)
DEFAULT_GCBC_ACTIVATION = "relu"
DEFAULT_GCBC_LOG_STD_MIN = -5.0
DEFAULT_GCBC_LEARNING_RATE = 1e-4
DEFAULT_GCBC_BATCH_SIZE = 512
DEFAULT_GCBC_TRAIN_STEPS = 1_000_000
DEFAULT_GCBC_GEOMETRIC_P = 0.5
DEFAULT_GCBC_MIN_OFFSET = 1
DEFAULT_GCBC_DISCOUNT = 0.88
DEFAULT_GCBC_LOG_STD_MAX: Optional[float] = 2.0
DEFAULT_GCBC_INIT_GAIN = 0.01  # small init on the final head for stable log-std

# Goal-conditioned reward convention shared with GC-IQL: 0 at goal, -1 otherwise.
DEFAULT_REWARD_UNACHIEVED = -1.0
DEFAULT_REWARD_REACHED = 0.0

__all__ = [
    "GaussianBCPolicy",
    "GCBCAgent",
    "GCBCConfig",
    "GCBCHistory",
    "build_gc_bc",
    "sample_geometric_goal_indices",
    "sample_geometric_goals",
    "sample_gcbc_batch",
    "train_gc_bc",
    "bc_action_fn",
    "gcbc_action_fn",
    "evaluate_gc_bc_agent",
    "evaluate_gc_bc",
    "compose_inputs",
    "main",
    "DEFAULT_GCBC_HIDDEN_DIMS",
    "DEFAULT_GCBC_LOG_STD_MIN",
    "DEFAULT_GCBC_LEARNING_RATE",
    "DEFAULT_GCBC_BATCH_SIZE",
    "DEFAULT_GCBC_TRAIN_STEPS",
    "DEFAULT_GCBC_GEOMETRIC_P",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("GC-BC requires PyTorch (>= 2.0) to be installed.")


def get_activation(name: str) -> nn.Module:
    """Resolve an activation name to a module (defaults to ReLU, per addendum)."""
    name = (name or "relu").lower()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
        "linear": nn.Identity,
        "none": nn.Identity,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'")
    return mapping[name]()


def compose_inputs(states: Any, goals: Any) -> Any:
    """Concatenate ``(state, goal)`` along the last dimension.

    Torch tensors (including ones of different leading shapes) and numpy arrays
    are both supported; broadcasting is delegated to the caller (as in GC-IQL).
    """
    if _HAS_TORCH and isinstance(states, torch.Tensor):
        if not isinstance(goals, torch.Tensor):
            goals = torch.as_tensor(
                np.asarray(goals), dtype=states.dtype, device=states.device
            )
        goals = goals.to(dtype=states.dtype, device=states.device)
        return torch.cat([states, goals], dim=-1)
    states_np = np.asarray(states, dtype=np.float32)
    goals_np = np.asarray(goals, dtype=np.float32)
    return np.concatenate([states_np, goals_np], axis=-1)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class GaussianBCPolicy(nn.Module):
    """Goal-conditioned Gaussian policy ``pi(a | s, g)`` trained by MLE.

    Architecture (addendum "Additional Details on GC-BC"):

    * ``len(hidden_dims)`` hidden layers (default three of size 512)
    * LayerNorm applied *before* each activation
    * ReLU activations between hidden layers
    * Final linear layer producing ``2 * action_dim`` outputs: action mean and
      log standard deviation, with ``log_std`` clamped to ``log_std_min``.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = DEFAULT_GCBC_HIDDEN_DIMS,
        activation: str = DEFAULT_GCBC_ACTIVATION,
        use_layer_norm: bool = True,
        action_low: float = -1.0,
        action_high: float = 1.0,
        log_std_min: float = DEFAULT_GCBC_LOG_STD_MIN,
        log_std_max: Optional[float] = DEFAULT_GCBC_LOG_STD_MAX,
        tanh_squash: bool = False,
        init_gain: float = DEFAULT_GCBC_INIT_GAIN,
    ) -> None:
        super().__init__()
        _require_torch()
        hidden_dims = tuple(int(h) for h in (hidden_dims or ()))
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.use_layer_norm = bool(use_layer_norm)
        self.log_std_min = float(log_std_min)
        self.log_std_max = log_std_max
        self.tanh_squash = bool(tanh_squash)

        layers: List[nn.Module] = []
        prev = self.input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if self.use_layer_norm:
                # LayerNorm before the activation (addendum).
                layers.append(nn.LayerNorm(hidden))
            layers.append(get_activation(activation))
            prev = hidden
        head = nn.Linear(prev, 2 * self.action_dim)
        layers.append(head)
        self.net = nn.Sequential(*layers)

        # Action bounds used when sampling / clamping at evaluation time.
        if isinstance(action_low, (int, float)):
            low = np.full(self.action_dim, float(action_low), dtype=np.float32)
        else:
            low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        if isinstance(action_high, (int, float)):
            high = np.full(self.action_dim, float(action_high), dtype=np.float32)
        else:
            high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.register_buffer("action_low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high, dtype=torch.float32))
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self._init_parameters(init_gain)

    # -- initialization ----------------------------------------------------
    def _init_parameters(self, init_gain: float = DEFAULT_GCBC_INIT_GAIN) -> None:
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            if i == len(linears) - 1:
                nn.init.xavier_uniform_(layer.weight, gain=float(init_gain))
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for m in self.net:
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -- distribution ------------------------------------------------------
    def distribution(self, inputs: Any) -> Any:
        """Return the (clamped) Gaussian action distribution for composed inputs."""
        _require_torch()
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
        out = self.net(inputs)
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min)
        if self.log_std_max is not None:
            log_std = torch.clamp(log_std, max=float(self.log_std_max))
        if self.tanh_squash:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def forward(self, inputs: Any) -> Any:
        return self.distribution(inputs)

    def log_prob(self, inputs: Any, actions: Any, reduce: bool = True) -> Any:
        dist = self.distribution(inputs)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        log_prob = dist.log_prob(actions.to(device=dist.mean.device))
        if reduce:
            return log_prob.sum(dim=-1)
        return log_prob

    def mean_action(self, inputs: Any) -> torch.Tensor:
        dist = self.distribution(inputs)
        return dist.mean

    def sample(self, inputs: Any, deterministic: bool = False) -> torch.Tensor:
        dist = self.distribution(inputs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample() if self.tanh_squash else dist.sample()
        if not self.tanh_squash:
            low = self.action_low.to(action.device)
            high = self.action_high.to(action.device)
            action = torch.max(torch.min(action, high), low)
        return action

    # -- convenience for the FRE eval harness (agent-like interface) --------
    def act(self, inputs: Any, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            action = self.sample(inputs, deterministic=deterministic)
        return action.detach().cpu().numpy()

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, action_dim={self.action_dim}, "
            f"hidden_dims={self.hidden_dims}, layernorm={self.use_layer_norm}, "
            f"log_std_min={self.log_std_min}"
        )


# ---------------------------------------------------------------------------
# Agent wrapper (agent-like interface so the shared eval harness can be used)
# ---------------------------------------------------------------------------


class GCBCAgent(nn.Module):
    """Thin goal-conditioned wrapper exposing the same interface as GC-IQL's agent.

    This lets GC-BC reuse ``fre.baselines.gc_iql.evaluate_gc_agent`` (rollout-based
    evaluation conditioned on the ground-truth task goal) unchanged.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = DEFAULT_GCBC_HIDDEN_DIMS,
        activation: str = DEFAULT_GCBC_ACTIVATION,
        use_layer_norm: bool = True,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        log_std_min: float = DEFAULT_GCBC_LOG_STD_MIN,
        log_std_max: Optional[float] = DEFAULT_GCBC_LOG_STD_MAX,
        learning_rate: float = DEFAULT_GCBC_LEARNING_RATE,
        tanh_squash: bool = False,
        device: Optional[Any] = None,
        name: str = "gc-bc",
    ) -> None:
        super().__init__()
        _require_torch()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim) if goal_dim is not None else int(state_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = activation
        self.use_layer_norm = bool(use_layer_norm)
        self.learning_rate = float(learning_rate)
        self.name = name

        self.policy = GaussianBCPolicy(
            input_dim=self.state_dim + self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            action_low=action_low,
            action_high=action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            tanh_squash=tanh_squash,
        )
        if device is not None:
            self.to(device)

    # -- batch composition -------------------------------------------------
    def compose(self, states: Any, goals: Any) -> Any:
        """Concatenate state and goal into the policy input."""
        return compose_inputs(states, goals)

    # -- training ----------------------------------------------------------
    def bc_loss(self, states: Any, actions: Any, goals: Any) -> torch.Tensor:
        """MLE loss: ``-E log pi(a | s, g)`` (addendum, Eq. 1)."""
        inputs = self.compose(states, goals)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        log_prob = self.policy.log_prob(inputs, actions, reduce=True)
        return -log_prob.mean()

    loss_fn = bc_loss

    def update(self, batch: Dict[str, Any], optimizer: Optional[Any] = None, grad_clip: Optional[float] = 10.0) -> Dict[str, float]:
        """One MLE gradient step. ``batch`` must contain obs/actions/goals."""
        _require_torch()
        device = next(self.parameters()).device
        states = _to_tensor(batch.get("observations", batch.get("obs", batch.get("states"))), device)
        actions = _to_tensor(batch.get("actions"), device)
        goals = _to_tensor(batch.get("goals", batch.get("goal")), device)

        loss = self.bc_loss(states, actions, goals)
        if optimizer is None:
            optimizer = self.optimizer()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip))
        optimizer.step()
        return {"bc_loss": float(loss.detach().cpu()), "log_prob": float(-loss.detach().cpu())}

    train_step = update

    def update_from_batch(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Alias accepting either ``(batch)`` or ``(states, actions, goals)``."""
        optimizer = kwargs.pop("optimizer", None)
        if args and isinstance(args[0], dict):
            return self.update(args[0], optimizer=optimizer, **kwargs)
        batch = {"observations": None, "actions": None, "goals": None}
        if len(args) >= 1:
            batch["observations"] = args[0]
        if len(args) >= 2:
            batch["actions"] = args[1]
        if len(args) >= 3:
            batch["goals"] = args[2]
        batch.update({k: v for k, v in kwargs.items() if k in ("observations", "actions", "goals")})
        return self.update(batch, optimizer=optimizer)

    def optimizer(self, learning_rate: Optional[float] = None) -> torch.optim.Optimizer:
        lr = float(learning_rate if learning_rate is not None else self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=lr)

    # -- acting ------------------------------------------------------------
    def select_action(
        self,
        states: Any,
        goals: Any = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return an action for a single state (or batch), conditioned on ``goals``."""
        _require_torch()
        device = next(self.parameters()).device
        if goals is None:
            raise ValueError("GC-BC requires a goal to select an action.")
        states_t = _to_tensor(states, device)
        goals_t = _to_tensor(goals, device)
        if states_t.dim() == 1:
            states_t = states_t.unsqueeze(0)
        if goals_t.dim() == 1:
            goals_t = goals_t.unsqueeze(0)
        if goals_t.shape[0] != states_t.shape[0]:
            if goals_t.shape[0] == 1:
                goals_t = goals_t.expand(states_t.shape[0], -1)
            else:
                raise ValueError("Cannot broadcast goals to states in select_action.")
        inputs = self.compose(states_t, goals_t)
        with torch.no_grad():
            actions = self.policy.sample(inputs, deterministic=deterministic)
        return actions.detach().cpu().numpy()

    def act(self, states: Any, goals: Any = None, deterministic: bool = True) -> np.ndarray:
        return self.select_action(states, goals=goals, deterministic=deterministic)

    def select_actions(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.select_action(*args, **kwargs)

    # -- bookkeeping -------------------------------------------------------
    def hparams(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "use_layer_norm": self.use_layer_norm,
            "log_std_min": self.policy.log_std_min,
            "log_std_max": self.policy.log_std_max,
            "learning_rate": self.learning_rate,
            "name": self.name,
        }

    def extra_repr(self) -> str:
        return f"GC-BC(state_dim={self.state_dim}, goal_dim={self.goal_dim}, action_dim={self.action_dim})"


def _to_tensor(x: Any, device: Any = None) -> Any:
    _require_torch()
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Geometric hindsight goal sampling
# ---------------------------------------------------------------------------


def _trajectory_end_lookup(dataset: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function mapping transition indices to their trajectory end."""
    traj_starts = getattr(dataset, "traj_starts", None)
    traj_ends = getattr(dataset, "traj_ends", None)
    num = len(dataset) if dataset is not None else 0

    if traj_starts is not None and traj_ends is not None and len(np.asarray(traj_starts)) > 0:
        traj_starts = np.asarray(traj_starts, dtype=np.int64)
        traj_ends = np.asarray(traj_ends, dtype=np.int64)

        def lookup(indices: np.ndarray) -> np.ndarray:
            ordinals = np.searchsorted(traj_starts, indices, side="right") - 1
            ordinals = np.clip(ordinals, 0, len(traj_ends) - 1)
            return traj_ends[ordinals]

        return lookup

    # Fallback: treat the whole dataset as a single trajectory.
    def lookup_single(indices: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(indices, dtype=np.int64), max(num - 1, 0))

    return lookup_single


def sample_geometric_goal_indices(
    indices: np.ndarray,
    rng: np.random.Generator,
    traj_end_lookup: Callable[[np.ndarray], np.ndarray],
    geometric_p: float = DEFAULT_GCBC_GEOMETRIC_P,
    min_offset: int = DEFAULT_GCBC_MIN_OFFSET,
) -> np.ndarray:
    """Sample future goal indices with a geometric distribution over offsets.

    Only *geometric* future-state sampling is used by GC-BC (no random goals and
    no current-state goals).  The offset ``k >= 0`` is drawn from
    ``Geometric(p)`` (number of failures before the first success), so with
    ``p = 0.5`` offsets are small with high probability and occasionally large;
    the goal index is clipped to the end of the same trajectory.
    """
    indices = np.asarray(indices, dtype=np.int64)
    ends = np.asarray(traj_end_lookup(indices), dtype=np.int64)
    # rng.geometric returns values >= 1; subtract 1 to allow k = 0.
    failures = rng.geometric(geometric_p, size=indices.shape[0]) - 1
    max_offset = np.maximum(ends - indices, 0)
    offset = np.minimum(np.maximum(failures + int(min_offset), int(min_offset)), max_offset)
    goals = np.minimum(indices + offset, ends)
    # Guard against selecting the current state (only possible when the
    # trajectory end equals the current index).
    goals = np.where(goals == indices, ends, goals)
    return goals.astype(np.int64)


def sample_geometric_goals(
    dataset: Any,
    indices: np.ndarray,
    rng: np.random.Generator,
    geometric_p: float = DEFAULT_GCBC_GEOMETRIC_P,
    min_offset: int = DEFAULT_GCBC_MIN_OFFSET,
) -> np.ndarray:
    """Return future goal *states* for the given transition indices."""
    lookup = _trajectory_end_lookup(dataset)
    goal_indices = sample_geometric_goal_indices(
        np.asarray(indices),
        rng,
        lookup,
        geometric_p=geometric_p,
        min_offset=min_offset,
    )
    observations = np.asarray(dataset.observations)
    return observations[goal_indices].astype(np.float32)


def sample_gcbc_batch(
    dataset: Any,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional["GCBCConfig"] = None,
    geometric_p: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Build a BC training batch with geometric-only hindsight relabeling."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(dataset)
    indices = rng.integers(0, n, size=int(batch_size))
    observations = np.asarray(dataset.observations)[indices].astype(np.float32)
    actions = np.asarray(dataset.actions)[indices].astype(np.float32)
    p = float(geometric_p if geometric_p is not None else (config.geometric_p if config else DEFAULT_GCBC_GEOMETRIC_P))
    goals = sample_geometric_goals(dataset, indices, rng, geometric_p=p)
    return {
        "observations": observations,
        "actions": actions,
        "goals": goals,
        "indices": indices.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GCBCConfig:
    """Configuration for GC-BC training / evaluation."""

    domain: str = "antmaze"
    env_name: Optional[str] = None
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    goal_dim: Optional[int] = None
    hidden_dims: Tuple[int, ...] = DEFAULT_GCBC_HIDDEN_DIMS
    activation: str = DEFAULT_GCBC_ACTIVATION
    use_layer_norm: bool = True
    log_std_min: float = DEFAULT_GCBC_LOG_STD_MIN
    log_std_max: Optional[float] = DEFAULT_GCBC_LOG_STD_MAX
    learning_rate: float = DEFAULT_GCBC_LEARNING_RATE
    batch_size: int = DEFAULT_GCBC_BATCH_SIZE
    train_steps: int = DEFAULT_GCBC_TRAIN_STEPS
    geometric_p: float = DEFAULT_GCBC_GEOMETRIC_P
    min_offset: int = DEFAULT_GCBC_MIN_OFFSET
    discount: float = DEFAULT_GCBC_DISCOUNT
    tanh_squash: bool = False
    discretize_antmaze: bool = True
    num_bins: int = 32
    action_low: Any = -1.0
    action_high: Any = 1.0
    seed: int = 0
    device: Optional[Any] = None
    log_every: int = 5000
    name: str = "gc-bc"
    extra: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **overrides: Any) -> "GCBCConfig":
        return _dc_replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["hidden_dims"] = list(self.hidden_dims)
        out["device"] = str(self.device) if self.device is not None else None
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class GCBCHistory:
    """Lightweight training-history container."""

    steps: List[int] = field(default_factory=list)
    bc_loss: List[float] = field(default_factory=list)
    log_prob: List[float] = field(default_factory=list)

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        self.steps.append(int(step))
        self.bc_loss.append(float(metrics.get("bc_loss", float("nan"))))
        self.log_prob.append(float(metrics.get("log_prob", float("nan"))))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": list(self.steps),
            "bc_loss": list(self.bc_loss),
            "log_prob": list(self.log_prob),
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


def build_gc_bc(
    state_dim: int,
    action_dim: int,
    goal_dim: Optional[int] = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> GCBCAgent:
    """Tolerant factory for a GC-BC agent."""
    allowed = {
        "hidden_dims",
        "activation",
        "use_layer_norm",
        "action_low",
        "action_high",
        "log_std_min",
        "log_std_max",
        "learning_rate",
        "tanh_squash",
        "name",
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    hidden = filtered.get("hidden_dims")
    if hidden is not None:
        filtered["hidden_dims"] = tuple(int(h) for h in hidden)
    return GCBCAgent(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        goal_dim=goal_dim,
        device=device,
        **filtered,
    )


def train_gc_bc(
    dataset: Any = None,
    config: Optional[GCBCConfig] = None,
    steps: Optional[int] = None,
    device: Optional[Any] = None,
    agent: Optional[GCBCAgent] = None,
    progress: bool = True,
    log_every: int = 5000,
    log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
    **config_overrides: Any,
) -> Dict[str, Any]:
    """Train GC-BC with MLE on geometrically relabeled hindsight goals.

    Returns a dict with ``agent``, ``config``, ``steps``, ``history``,
    ``final_metrics`` and ``wall_time``.
    """
    _require_torch()
    if config is None:
        config = GCBCConfig()
    if config_overrides:
        config = config.replace(**config_overrides)
    if steps is not None:
        config = config.replace(train_steps=int(steps))
    if device is None:
        device = config.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if dataset is None:
        from fre.data import load_dataset  # local import: heavy / optional deps

        dataset = load_dataset(config.domain)

    state_dim = int(config.state_dim or np.asarray(dataset.observations).shape[-1])
    action_dim = int(config.action_dim or np.asarray(dataset.actions).shape[-1])
    config = config.replace(state_dim=state_dim, action_dim=action_dim)

    if agent is None:
        agent = build_gc_bc(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=config.goal_dim,
            device=device,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            action_low=config.action_low,
            action_high=config.action_high,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            learning_rate=config.learning_rate,
            tanh_squash=config.tanh_squash,
            name=config.name,
        )
    agent.to(device)
    agent.train()

    rng = np.random.default_rng(config.seed)
    optimizer = agent.optimizer(config.learning_rate)
    history = GCBCHistory()

    iterator: Iterable[int]
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(range(config.train_steps), desc="gc-bc")
        except Exception:  # pragma: no cover
            iterator = range(config.train_steps)
    else:
        iterator = range(config.train_steps)

    start = time.time()
    metrics: Dict[str, float] = {}
    log_every = int(log_every or config.log_every or 5000)
    for step in iterator:
        batch = sample_gcbc_batch(dataset, config.batch_size, rng=rng, config=config)
        metrics = agent.update(batch, optimizer=optimizer)
        if log_every and (step + 1) % log_every == 0:
            history.log(step + 1, metrics)
            if log_fn is not None:
                log_fn(step + 1, metrics)
    wall = time.time() - start

    steps_done = int(config.train_steps)
    if not history.steps or history.steps[-1] != steps_done:
        history.log(steps_done, metrics)
        if log_fn is not None:
            log_fn(steps_done, metrics)

    return {
        "agent": agent,
        "config": config,
        "steps": steps_done,
        "history": history,
        "final_metrics": metrics,
        "wall_time": wall,
    }


# ---------------------------------------------------------------------------
# Evaluation (ground-truth goal conditioning)
# ---------------------------------------------------------------------------


def bc_action_fn(
    agent: Any,
    goal: Any,
    deterministic: bool = True,
    state_fn: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Any], np.ndarray]:
    """Build an ``action_fn`` for ``fre.evaluation.evaluate.rollout_episode``."""
    goal_arr = np.asarray(goal, dtype=np.float32)

    def action_fn(state: Any) -> np.ndarray:
        obs = state_fn(state) if state_fn is not None else state
        action = agent.select_action(obs, goals=goal_arr, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 1:
            action = action[0]
        return action

    return action_fn


gcbc_action_fn = bc_action_fn


def evaluate_gc_bc_agent(
    agent: Any,
    domain: str = "antmaze",
    task_set: str = "all",
    dataset: Any = None,
    num_episodes: int = 20,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    deterministic: bool = True,
    max_episode_steps: Optional[int] = None,
    device: Optional[Any] = None,
    verbose: bool = False,
    discretize_antmaze: bool = True,
    goal_override: Optional[Any] = None,
    **task_kwargs: Any,
) -> Dict[str, Any]:
    """Zero-shot evaluation of GC-BC with the ground-truth task goal.

    Delegates to the shared GC-IQL evaluation driver (same rollout, metrics and
    normalization pipeline as FRE/GC-IQL) but reports under the ``gc_bc`` key.
    """
    from fre.baselines.gc_iql import evaluate_gc_agent  # local import: avoids cycles

    results = evaluate_gc_agent(
        agent,
        domain=domain,
        task_set=task_set,
        dataset=dataset,
        num_episodes=num_episodes,
        seeds=seeds,
        deterministic=deterministic,
        max_episode_steps=max_episode_steps,
        device=device,
        verbose=verbose,
        discretize_antmaze=discretize_antmaze,
        goal_override=goal_override,
        **task_kwargs,
    )
    if isinstance(results, dict):
        results.setdefault("method", "gc_bc")
        summary = results.get("summary")
        if isinstance(summary, dict):
            summary.setdefault("method", "gc_bc")
    return results


evaluate_gc_bc = evaluate_gc_bc_agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate the GC-BC baseline.")
    parser.add_argument("--domain", default="antmaze", help="antmaze | exorl:walker | exorl:cheetah | kitchen")
    parser.add_argument("--dataset", default=None, help="Optional dataset path override.")
    parser.add_argument("--steps", type=int, default=DEFAULT_GCBC_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_GCBC_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_GCBC_LEARNING_RATE)
    parser.add_argument("--geometric-p", type=float, default=DEFAULT_GCBC_GEOMETRIC_P)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_GCBC_HIDDEN_DIMS))
    parser.add_argument("--log-std-min", type=float, default=DEFAULT_GCBC_LOG_STD_MIN)
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--save", default=None, help="Checkpoint output path.")
    parser.add_argument("--history", default=None, help="Training history JSON path.")
    parser.add_argument("--eval", action="store_true", help="Run zero-shot evaluation after training.")
    parser.add_argument("--eval-task-set", default="all")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--eval-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    from fre.data import load_dataset  # local import: heavy / optional deps

    dataset = load_dataset(args.domain, dataset_path=args.dataset)
    state_dim = int(np.asarray(dataset.observations).shape[-1])
    action_dim = int(np.asarray(dataset.actions).shape[-1])

    config = GCBCConfig(
        domain=args.domain,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        use_layer_norm=not args.no_layer_norm,
        log_std_min=args.log_std_min,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_steps=args.steps,
        geometric_p=args.geometric_p,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )

    result = train_gc_bc(
        dataset=dataset,
        config=config,
        progress=not args.quiet,
    )
    agent = result["agent"]

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save(
            {
                "state_dict": agent.state_dict(),
                "hparams": agent.hparams(),
                "config": config.to_dict(),
            },
            args.save,
        )
        print(f"Saved GC-BC checkpoint to {args.save}")
    if args.history:
        result["history"].save(args.history)
        print(f"Saved GC-BC history to {args.history}")

    if args.eval:
        seeds = tuple(range(int(args.eval_seeds)))
        summary = evaluate_gc_bc_agent(
            agent,
            domain=args.domain,
            task_set=args.eval_task_set,
            dataset=dataset,
            num_episodes=int(args.eval_episodes),
            seeds=seeds,
            device=config.device,
            verbose=not args.quiet,
        )
        if args.eval_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.eval_out)), exist_ok=True)
            with open(args.eval_out, "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
            print(f"Saved GC-BC evaluation to {args.eval_out}")
        else:
            print(json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
