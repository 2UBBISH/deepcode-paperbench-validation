"""GC-BC baseline (Goal-Conditioned Behavioral Cloning) for FRE.

Implements the baseline described in the paper's addendum, section
"Additional Details on GC-BC":

Network architecture
    * a multi-layer perceptron (MLP) with three hidden layers of size 512
    * ReLU activations are applied between each hidden layer
    * Layer normalization is applied *before* each activation
    * the output layer predicts a Gaussian distribution over actions, i.e. two
      outputs: a linear **mean action** and a **log std** clamped with a lower
      bound of ``-5.0``

Loss function (maximum likelihood estimation, MLE)::

    L_pi = -E_{(s, g, a) ~ D} [ log pi(a | s, g) ]                  (1)

Training
    * hindsight relabeling where the goal is sampled from the dataset; for
      GC-BC *only geometric sampling* is used to sample goals from future
      states in the trajectory (no random goals, no current-state goals).

Evaluation
    * the goal-conditioned agent is given the ground-truth goal that the
      specific evaluation task contains, to condition on.

The agent is a :class:`fre.eval.baselines.BaselineAgent`: it exposes
``condition(task, context=...)`` (returns the ground-truth goal supplied by the
evaluation task) and ``act(observation, conditioning, deterministic=True)``.

Paper-silent details chosen here (documented, not from the paper):
    * the geometric distribution parameter used for future-goal sampling
      (``DEFAULT_GEOMETRIC_P = 0.2``, matching ``fre.reward_priors``);
    * plain (non-squashed) Gaussian likelihood, exactly matching Equation (1);
      an optional tanh-squashed variant is provided but off by default;
    * Adam at learning rate 1e-4 with batch size 512 and a step budget equal to
      the policy-training budget of the corresponding domain.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# tolerant imports (package-relative first, then path based fallback)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised depending on how the module is imported
    from fre.eval.baselines import (
        FRE_CONTEXT_SAMPLES,
        NUM_EVAL_EPISODES,
        BaselineAgent,
        as_2d,
        episode_length_for,
        normalize_score,
        task_reward,
        task_succeeded,
    )
except Exception:  # pragma: no cover
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    for _candidate in (_ROOT, os.path.dirname(_ROOT)):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    try:
        from fre.eval.baselines import (  # type: ignore
            FRE_CONTEXT_SAMPLES,
            NUM_EVAL_EPISODES,
            BaselineAgent,
            as_2d,
            episode_length_for,
            normalize_score,
            task_reward,
            task_succeeded,
        )
    except Exception:  # pragma: no cover - fully standalone fallback

        class BaselineAgent:  # type: ignore
            name = "baseline"
            num_skills = 1

            def condition(self, task, context=None, rng=None):
                return None

            def act(self, observation, conditioning, deterministic=True):
                raise NotImplementedError

        FRE_CONTEXT_SAMPLES = 32
        NUM_EVAL_EPISODES = 20

        def as_2d(observations):  # type: ignore
            arr = np.asarray(observations, dtype=np.float64)
            return arr[None] if arr.ndim == 1 else arr

        def episode_length_for(task, default=1000):  # type: ignore
            return int(getattr(task, "eval_episode_length", default) or default)

        def normalize_score(task, episode_return, succeeded=False, episode_length=None, clip=False):  # type: ignore
            return float(episode_return)

        def task_reward(task, observations):  # type: ignore
            return np.asarray(task.reward(observations), dtype=np.float64).reshape(-1)

        def task_succeeded(task, observations, threshold=None):  # type: ignore
            r = task_reward(task, observations)
            return r > 0.5 * float(np.nanmax(np.abs(r)) + 1e-12)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is required in practice
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


__all__ = [
    "GCBCConfig",
    "GCBCNetwork",
    "GCBCAgent",
    "make_gc_bc_agent",
    "GC_BC_DEFAULTS",
    "sample_geometric_goal_steps",
    "task_goal_state",
    "gaussian_log_prob",
    "DEFAULT_HIDDEN_DIMS",
    "DEFAULT_LOG_STD_MIN",
    "DEFAULT_LOG_STD_MAX",
    "DEFAULT_GEOMETRIC_P",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ACTIVATION",
]

# ---------------------------------------------------------------------------
# constants (paper-specified where noted, otherwise documented defaults)
# ---------------------------------------------------------------------------
DEFAULT_HIDDEN_DIMS: Tuple[int, ...] = (512, 512, 512)  # paper: 3 hidden layers of 512
DEFAULT_ACTIVATION = "relu"  # paper: ReLU between each hidden layer
DEFAULT_LAYER_NORM = True  # paper: LayerNorm before each activation
DEFAULT_LOG_STD_MIN = -5.0  # paper: log std lower bound -5.0
DEFAULT_LOG_STD_MAX = 2.0  # paper-silent upper bound (numerical guard)
DEFAULT_GEOMETRIC_P = 0.2  # paper-silent geometric sampling parameter
DEFAULT_LEARNING_RATE = 1e-4  # Table 3 learning rate
DEFAULT_BATCH_SIZE = 512  # Table 3 batch size
DEFAULT_TARGET_STEPS = 850_000  # Table 3 policy-training budget (AntMaze)
DEFAULT_TANH_SQUASH = False  # paper states a plain Gaussian MLE objective
DEFAULT_BEST_OF_SKILLS = False  # GC-BC has a single conditioning (the goal)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class GCBCConfig:
    """Hyper-parameters of the GC-BC baseline.

    Values marked "(paper)" come from the addendum's GC-BC description; the
    remaining ones are documented defaults (paper is silent).
    """

    hidden_dims: Tuple[int, ...] = DEFAULT_HIDDEN_DIMS  # (paper)
    activation: str = DEFAULT_ACTIVATION  # (paper)
    layer_norm: bool = DEFAULT_LAYER_NORM  # (paper)
    log_std_min: float = DEFAULT_LOG_STD_MIN  # (paper)
    log_std_max: float = DEFAULT_LOG_STD_MAX
    tanh_squash: bool = DEFAULT_TANH_SQUASH
    geometric_p: float = DEFAULT_GEOMETRIC_P
    learning_rate: float = DEFAULT_LEARNING_RATE  # (paper, Table 3)
    batch_size: int = DEFAULT_BATCH_SIZE  # (paper, Table 3)
    num_steps: int = DEFAULT_TARGET_STEPS
    weight_decay: float = 0.0
    max_grad_norm: Optional[float] = 10.0
    normalize_observations: bool = False
    state_mean: Optional[Sequence[float]] = None
    state_std: Optional[Sequence[float]] = None
    log_interval: int = 1000
    device: Optional[str] = None
    seed: int = 0
    name: str = "GC-BC"

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["hidden_dims"] = list(self.hidden_dims)
        return out

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None, **overrides) -> "GCBCConfig":
        values = dict(values or {})
        values.update(overrides)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in values.items() if k in known}
        if "hidden_dims" in kwargs and kwargs["hidden_dims"] is not None:
            kwargs["hidden_dims"] = tuple(int(h) for h in kwargs["hidden_dims"])
        return cls(**kwargs)


#: default configuration exported for ``fre.eval`` (name expected by the package).
GC_BC_DEFAULTS: Dict[str, Any] = GCBCConfig().to_dict()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _activation_module(name: str):
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required for GC-BC")
    name = (name or "relu").lower()
    table = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "mish": nn.Mish,
    }
    if name not in table:
        raise ValueError(f"unknown activation '{name}'")
    return table[name]()


def gaussian_log_prob(
    mean,
    log_std,
    action,
    tanh_squash: bool = False,
    log_std_min: float = DEFAULT_LOG_STD_MIN,
    log_std_max: float = DEFAULT_LOG_STD_MAX,
):
    """log pi(a | s, g) for a diagonal Gaussian (optionally tanh-squashed).

    Implements the quantity inside Equation (1).  With ``tanh_squash=False``
    this is a plain diagonal Gaussian log-density, exactly as written in the
    addendum.  The squashed variant adds the standard change-of-variables term
    and is therefore only used when explicitly requested.
    """
    log_std = log_std.clamp(min=log_std_min, max=log_std_max)
    if tanh_squash:
        # invert the squashing to evaluate the density at the pre-tanh action
        eps = 1e-6
        a = action.clamp(-1.0 + eps, 1.0 - eps)
        pre = 0.5 * torch.log((1.0 + a) / (1.0 - a))
        head = F.logsigmoid(torch.log1p(torch.exp(pre)) * 2.0)  # log(1 - tanh(x)^2)
        log_prob = -0.5 * (((pre - mean) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        return (log_prob + head).sum(dim=-1)
    log_prob = -0.5 * (((action - mean) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
    return log_prob.sum(dim=-1)


def sample_geometric_goal_steps(
    step_index: np.ndarray,
    trajectory_lengths: np.ndarray,
    p: float = DEFAULT_GEOMETRIC_P,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample future goal steps with a geometric look-ahead distribution.

    ``offset ~ Geometric(p)`` (>= 1) and the goal step is
    ``min(step + offset, trajectory_length)`` where ``trajectory_length`` is the
    number of transitions of the trajectory (hence the final state index).  The
    addendum states that GC-BC uses *only* geometric sampling of future states
    (no random goals and no current-state goals), which this function realizes.
    """
    rng = rng or np.random.default_rng()
    step_index = np.asarray(step_index, dtype=np.int64).reshape(-1)
    trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64).reshape(-1)
    offsets = rng.geometric(p=float(p), size=step_index.shape[0]).astype(np.int64)
    goals = np.minimum(step_index + np.maximum(offsets, 1), trajectory_lengths)
    # never allow a goal that is strictly in the past
    goals = np.maximum(goals, step_index)
    return goals


def _extract_goal_vector(goal: Any, obs_dim: Optional[int] = None) -> Optional[np.ndarray]:
    """Coerce a task goal specification into a flat float vector."""
    if goal is None:
        return None
    if isinstance(goal, dict):
        for key in ("state", "goal_state", "position", "goal"):
            if key in goal:
                return _extract_goal_vector(goal[key], obs_dim)
        return None
    arr = np.asarray(goal, dtype=np.float64).reshape(-1)
    if obs_dim is not None and arr.size > obs_dim:
        arr = arr[:obs_dim]
    return arr


def task_goal_state(
    task: Any,
    obs_dim: Optional[int] = None,
    reference_state: Optional[np.ndarray] = None,
    position_dims: Sequence[int] = (0, 1),
    discretized: bool = True,
) -> np.ndarray:
    """Build the ground-truth goal the GC-BC agent conditions on.

    * If the task carries a full observation-space goal (``task.goal`` with
      ``len(goal) >= obs_dim``, e.g. ExORL/Walker goals) it is returned as is.
    * If the task goal is a low-dimensional target position (AntMaze goals are
      (X, Y) locations) the position dimensions of ``reference_state`` (or
      zeros) are overwritten, matching the shared 32-bin preprocessing used by
      FRE / GC-IQL / GC-BC / OPAL.
    * ``task.metadata['goal_state']`` is honoured as a fallback.
    """
    obs_dim = int(obs_dim) if obs_dim is not None else None
    goal = _extract_goal_vector(getattr(task, "goal", None), obs_dim)

    if goal is None:
        meta = getattr(task, "metadata", None) or {}
        if isinstance(meta, dict):
            goal = _extract_goal_vector(meta.get("goal_state"), obs_dim)
    if goal is None:
        raise ValueError(
            f"task '{getattr(task, 'name', task)}' does not expose a ground-truth goal "
            "(GC-BC evaluation requires the true goal of the evaluation task)."
        )

    if obs_dim is None or goal.size >= obs_dim:
        if obs_dim is not None and goal.size > obs_dim:
            goal = goal[:obs_dim]
        return goal.astype(np.float64)

    # low-dimensional target (e.g. AntMaze XY): place it into the state vector
    state = np.zeros(obs_dim, dtype=np.float64)
    if reference_state is not None:
        ref = np.asarray(reference_state, dtype=np.float64).reshape(-1)
        state[: min(ref.size, obs_dim)] = ref[: min(ref.size, obs_dim)]
    dims = [int(d) for d in position_dims if int(d) < obs_dim]
    if len(dims) != goal.size:
        dims = list(range(min(goal.size, obs_dim)))
    for value, dim in zip(goal.tolist(), dims):
        state[dim] = float(value)
    return state


def _gather_states(replay_buffer: Any, traj_index: np.ndarray, step_index: np.ndarray) -> np.ndarray:
    """Fetch ``replay_buffer.state_at(traj, step)`` for a batch of indices."""
    traj_index = np.asarray(traj_index, dtype=np.int64).reshape(-1)
    step_index = np.asarray(step_index, dtype=np.int64).reshape(-1)
    state_at = getattr(replay_buffer, "state_at", None)
    if state_at is None:  # pragma: no cover - minimal buffer stand-ins
        raise AttributeError("replay buffer must implement state_at(traj_index, step_index)")
    out = [np.asarray(state_at(int(t), int(s)), dtype=np.float64).reshape(-1) for t, s in zip(traj_index, step_index)]
    return np.stack(out, axis=0)


def _trajectory_lengths(replay_buffer: Any, traj_index: np.ndarray) -> np.ndarray:
    lengths = getattr(replay_buffer, "_traj_offsets", None)
    num_traj = int(getattr(replay_buffer, "num_trajectories", 0) or 0)
    fn = getattr(replay_buffer, "trajectory_length", None)
    traj_index = np.asarray(traj_index, dtype=np.int64).reshape(-1)
    if fn is not None:
        return np.asarray([int(fn(int(t))) for t in traj_index], dtype=np.int64)
    if lengths is not None and num_traj:  # pragma: no cover
        return np.asarray([int(lengths[t]) for t in traj_index], dtype=np.int64)
    return np.full(traj_index.shape[0], np.iinfo(np.int64).max // 2, dtype=np.int64)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class GCBCNetwork(nn.Module):
        """Goal-conditioned Gaussian policy: ``(s, g) -> N(mu(s,g), sigma(s,g))``.

        Follows the addendum exactly: three hidden layers of size 512, ReLU
        between layers, LayerNorm applied before each activation, and a
        two-headed output (linear mean, log std clamped below at -5.0).
        """

        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            goal_dim: Optional[int] = None,
            hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
            activation: str = DEFAULT_ACTIVATION,
            layer_norm: bool = DEFAULT_LAYER_NORM,
            log_std_min: float = DEFAULT_LOG_STD_MIN,
            log_std_max: float = DEFAULT_LOG_STD_MAX,
            tanh_squash: bool = DEFAULT_TANH_SQUASH,
            normalize_observations: bool = False,
            state_mean: Optional[Sequence[float]] = None,
            state_std: Optional[Sequence[float]] = None,
            name: str = "gc_bc_network",
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.act_dim = int(act_dim)
            self.goal_dim = int(goal_dim) if goal_dim is not None else int(obs_dim)
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)
            self.tanh_squash = bool(tanh_squash)
            self.name = name
            self.normalize_observations = bool(normalize_observations)

            layers: List[nn.Module] = []
            in_dim = self.obs_dim + self.goal_dim
            for hidden in hidden_dims:
                layers.append(nn.Linear(in_dim, int(hidden)))
                if layer_norm:
                    layers.append(nn.LayerNorm(int(hidden)))  # before each activation
                layers.append(_activation_module(activation))
                in_dim = int(hidden)
            self.trunk = nn.Sequential(*layers)
            self.mean_head = nn.Linear(in_dim, self.act_dim)
            self.log_std_head = nn.Linear(in_dim, self.act_dim)

            self.register_buffer(
                "state_mean",
                torch.as_tensor(state_mean, dtype=torch.float32) if state_mean is not None else torch.zeros(self.obs_dim),
                persistent=False,
            )
            self.register_buffer(
                "state_std",
                torch.as_tensor(state_std, dtype=torch.float32) if state_std is not None else torch.ones(self.obs_dim),
                persistent=False,
            )

        # -- input handling -------------------------------------------------
        def _prepare(self, observation, goal):
            obs = self._as_tensor(observation)
            gl = self._as_tensor(goal)
            if obs.ndim == 1:
                obs = obs[None]
            if gl.ndim == 1:
                gl = gl[None]
            if gl.shape[0] == 1 and obs.shape[0] > 1:
                gl = gl.expand(obs.shape[0], -1)
            if self.normalize_observations:
                obs = (obs - self.state_mean) / self.state_std.clamp(min=1e-6)
            return obs, gl

        def _as_tensor(self, value):
            if isinstance(value, torch.Tensor):
                return value.to(dtype=torch.float32)
            return torch.as_tensor(np.asarray(value, dtype=np.float32))

        # -- API ------------------------------------------------------------
        def forward(self, observation, goal, deterministic: bool = False, action=None):
            if action is not None:
                a = action if isinstance(action, torch.Tensor) else torch.as_tensor(np.asarray(action, dtype=np.float32))
                if a.ndim == 1:
                    a = a[None]
                a = a.to(dtype=torch.float32)
            else:
                a = None
            obs, gl = self._prepare(observation, goal)
            features = self.trunk(torch.cat([obs, gl], dim=-1))
            mean = self.mean_head(features)
            log_std = self.log_std_head(features).clamp(min=self.log_std_min, max=self.log_std_max)
            if deterministic:
                act = torch.tanh(mean) if self.tanh_squash else mean
                log_prob = None
            else:
                std = log_std.exp()
                eps = torch.randn_like(mean)
                raw = mean + std * eps
                act = torch.tanh(raw) if self.tanh_squash else raw
                if a is not None:
                    log_prob = gaussian_log_prob(mean, log_std, a, self.tanh_squash, self.log_std_min, self.log_std_max)
            if a is not None and log_prob is None:
                log_prob = gaussian_log_prob(mean, log_std, a, self.tanh_squash, self.log_std_min, self.log_std_max)
            return mean, log_std, act, log_prob

        # -- losses ---------------------------------------------------------
        def mle_loss(self, observation, goal, action, reduction: str = "mean"):
            """Equation (1): ``-E log pi(a | s, g)`` (maximum likelihood)."""
            obs, gl = self._prepare(observation, goal)
            a = action if isinstance(action, torch.Tensor) else torch.as_tensor(np.asarray(action, dtype=np.float32))
            if a.ndim == 1:
                a = a[None]
            a = a.to(dtype=torch.float32)
            features = self.trunk(torch.cat([obs, gl], dim=-1))
            mean = self.mean_head(features)
            log_std = self.log_std_head(features).clamp(min=self.log_std_min, max=self.log_std_max)
            log_prob = gaussian_log_prob(mean, log_std, a, self.tanh_squash, self.log_std_min, self.log_std_max)
            loss = -log_prob
            if reduction == "mean":
                return loss.mean(), {"log_prob": float(log_prob.mean().detach().cpu()), "mse": float(((mean - a) ** 2).mean().detach().cpu())}
            if reduction == "sum":
                return loss.sum(), {"log_prob": float(log_prob.sum().detach().cpu())}
            return loss, {"log_prob": float(log_prob.mean().detach().cpu())}

        @torch.no_grad()
        def act_numpy(self, observation, goal, deterministic: bool = True) -> np.ndarray:
            _, _, act, _ = self.forward(observation, goal, deterministic=deterministic)
            return act.detach().cpu().numpy()

        def describe(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "obs_dim": self.obs_dim,
                "goal_dim": self.goal_dim,
                "act_dim": self.act_dim,
                "log_std_min": self.log_std_min,
                "log_std_max": self.log_std_max,
                "tanh_squash": self.tanh_squash,
                "num_parameters": int(sum(p.numel() for p in self.parameters())),
            }

else:  # pragma: no cover - torch missing

    class GCBCNetwork:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required to instantiate GCBCNetwork")


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------
class GCBCAgent(BaselineAgent):
    """Goal-conditioned behavioral-cloning baseline trained with MLE (Eq. 1).

    Parameters
    ----------
    replay_buffer:
        Offline (unlabeled) trajectory buffer.  Must provide
        ``sample_transitions(batch_size, rng=..., with_encoder_inputs=False)``
        returning a batch with ``observations``/``actions``/``traj_index``/
        ``step_index``, plus ``state_at`` and ``trajectory_length``.
    obs_dim / act_dim:
        Shapes; inferred from the replay buffer when omitted.
    config / **config_overrides:
        Any :class:`GCBCConfig` field.
    """

    name = "GC-BC"
    num_skills = 1
    is_goal_conditioned = True
    uses_latent = False

    def __init__(
        self,
        replay_buffer: Any = None,
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        goal_dim: Optional[int] = None,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        state_mean: Optional[Sequence[float]] = None,
        state_std: Optional[Sequence[float]] = None,
        **config_overrides,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for the GC-BC baseline")
        if isinstance(config, GCBCConfig):
            cfg = config
            if config_overrides:
                cfg = GCBCConfig.from_dict(cfg.to_dict(), **config_overrides)
        else:
            cfg = GCBCConfig.from_dict(config if isinstance(config, dict) else None, **config_overrides)

        self.config = cfg
        self.replay_buffer = replay_buffer
        self.obs_dim = int(obs_dim if obs_dim is not None else getattr(replay_buffer, "obs_dim", 0))
        self.act_dim = int(act_dim if act_dim is not None else getattr(replay_buffer, "act_dim", 0))
        self.goal_dim = int(goal_dim) if goal_dim is not None else self.obs_dim
        if self.obs_dim <= 0 or self.act_dim <= 0:
            raise ValueError("GC-BC requires obs_dim and act_dim (or a replay buffer providing them)")
        self.seed = int(seed if seed is not None else cfg.seed)
        self.rng = np.random.default_rng(self.seed)

        self.device = torch.device(device or cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if state_mean is None:
            state_mean = cfg.state_mean
        if state_std is None:
            state_std = cfg.state_std
        self.network = GCBCNetwork(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            goal_dim=self.goal_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            layer_norm=cfg.layer_norm,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            tanh_squash=cfg.tanh_squash,
            normalize_observations=cfg.normalize_observations,
            state_mean=state_mean,
            state_std=state_std,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.train_step_count = 0
        self.history: List[Dict[str, float]] = []
        self._goal_cache: Dict[str, np.ndarray] = {}

    # -- data ---------------------------------------------------------------
    def _sample_goal_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a goal-state batch from a *future* state of each trajectory (geometric)."""
        batch = self._sample_transitions(batch_size)
        obs = np.asarray(batch["observations"], dtype=np.float64)
        traj_index = np.asarray(batch["traj_index"], dtype=np.int64)
        step_index = np.asarray(batch["step_index"], dtype=np.int64)
        lengths = _trajectory_lengths(self.replay_buffer, traj_index)
        goal_steps = sample_geometric_goal_steps(step_index, lengths, p=self.config.geometric_p, rng=self.rng)
        goals = _gather_states(self.replay_buffer, traj_index, goal_steps)
        return obs, goals

    def _sample_transitions(self, batch_size: int) -> Dict[str, np.ndarray]:
        buf = self.replay_buffer
        if buf is None:
            raise ValueError("GC-BC requires a replay buffer to train")
        fn = getattr(buf, "sample_transitions", None)
        if fn is not None:
            try:
                batch = fn(batch_size, rng=self.rng, with_encoder_inputs=False)
            except TypeError:
                batch = fn(batch_size, rng=self.rng)
            if hasattr(batch, "observations"):
                return {
                    "observations": np.asarray(batch.observations, dtype=np.float64),
                    "actions": np.asarray(batch.actions, dtype=np.float64),
                    "traj_index": np.asarray(getattr(batch, "traj_index", np.zeros(batch_size, dtype=np.int64))),
                    "step_index": np.asarray(getattr(batch, "step_index", np.zeros(batch_size, dtype=np.int64))),
                }
            if isinstance(batch, dict):
                return {
                    "observations": np.asarray(batch["observations"], dtype=np.float64),
                    "actions": np.asarray(batch["actions"], dtype=np.float64),
                    "traj_index": np.asarray(batch.get("traj_index", np.zeros(batch_size, dtype=np.int64))),
                    "step_index": np.asarray(batch.get("step_index", np.zeros(batch_size, dtype=np.int64))),
                }
        raise AttributeError("replay buffer must implement sample_transitions(batch_size, ...)")

    # -- training -----------------------------------------------------------
    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """One MLE update of Equation (1) on a relabeled ``(s, g, a)`` batch."""
        batch_size = int(batch_size or self.config.batch_size)
        observations, goals = self._sample_goal_batch(batch_size)
        # the actions must come from the *same* sampled transitions
        batch = self._sample_transitions(batch_size)
        actions = np.asarray(batch["actions"], dtype=np.float64)

        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        goal_t = torch.as_tensor(goals, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        loss, stats = self.network.mle_loss(obs_t, goal_t, act_t)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.max_grad_norm))
        self.optimizer.step()
        self.train_step_count += 1

        metrics = {
            "step": float(self.train_step_count),
            "loss": float(loss.detach().cpu()),
            "log_prob": float(stats.get("log_prob", 0.0)),
            "mse": float(stats.get("mse", 0.0)),
        }
        self.history.append(metrics)
        return metrics

    def train(self, num_steps: Optional[int] = None, callback=None, log_interval: Optional[int] = None) -> List[Dict[str, float]]:
        """Run ``num_steps`` MLE updates; returns the loss history."""
        num_steps = int(num_steps or self.config.num_steps)
        log_interval = int(log_interval or self.config.log_interval)
        self.network.train()
        out: List[Dict[str, float]] = []
        for i in range(num_steps):
            metrics = self.train_step()
            out.append(metrics)
            if callback is not None:
                callback(self.train_step_count, metrics)
            if log_interval and (self.train_step_count % log_interval == 0):
                print(
                    f"[{self.name}] step {self.train_step_count} loss={metrics['loss']:.4f} "
                    f"log_prob={metrics['log_prob']:.4f} mse={metrics['mse']:.4f}",
                    flush=True,
                )
        return out

    #: alias matching typical training-script naming
    fit = train

    # -- evaluation interface ----------------------------------------------
    def goal_for_task(self, task: Any, reference_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Ground-truth goal the agent conditions on (addendum: GC-BC evaluation)."""
        key = getattr(task, "name", repr(task))
        if key in self._goal_cache and reference_state is None:
            return self._goal_cache[key]
        goal = task_goal_state(task, obs_dim=self.goal_dim, reference_state=reference_state)
        if goal.size != self.goal_dim:
            padded = np.zeros(self.goal_dim, dtype=np.float64)
            padded[: min(goal.size, self.goal_dim)] = goal[: min(goal.size, self.goal_dim)]
            goal = padded
        if reference_state is None:
            self._goal_cache[key] = goal
        return goal

    def condition(self, task: Any, context: Any = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:  # type: ignore[override]
        """Return the ground-truth goal of ``task`` (no learned encoder involved)."""
        reference = None
        if context is not None:
            arr = np.asarray(context, dtype=np.float64)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            if arr.shape[0] >= 1:
                reference = arr[0]
        elif self.replay_buffer is not None:
            try:
                reference = np.asarray(self.replay_buffer.sample_states(1, rng=self.rng), dtype=np.float64).reshape(-1)
            except Exception:  # pragma: no cover
                reference = None
        return self.goal_for_task(task, reference_state=reference)

    def condition_many(self, task: Any, num_skills: int = 1, context: Any = None, rng=None) -> np.ndarray:  # type: ignore[override]
        """GC-BC has a single deterministic conditioning (the task goal)."""
        goal = self.condition(task, context=context, rng=rng)
        return np.repeat(goal[None], max(int(num_skills), 1), axis=0)

    def act(self, observation, conditioning, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        """Return an action given the observation and the goal conditioning."""
        obs = as_2d(observation)
        goal = np.asarray(conditioning, dtype=np.float64).reshape(-1)
        actions = self.network.act_numpy(obs, goal, deterministic=deterministic)
        return actions[0] if np.asarray(observation).ndim == 1 else actions

    def act_batch(self, observations, conditioning, deterministic: bool = True) -> np.ndarray:
        return self.network.act_numpy(as_2d(observations), np.asarray(conditioning, dtype=np.float64).reshape(-1), deterministic=deterministic)

    # -- persistence --------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "goal_dim": self.goal_dim,
            "train_step_count": self.train_step_count,
            "seed": self.seed,
        }

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        self.network.load_state_dict(state["network"])
        if load_optimizer and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:  # pragma: no cover
                pass
        self.train_step_count = int(state.get("train_step_count", 0))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, load_optimizer=load_optimizer)
        return state

    def describe(self) -> Dict[str, Any]:  # type: ignore[override]
        info = self.network.describe()
        info.update(
            {
                "name": self.name,
                "method": self.name,
                "goal_conditioned": True,
                "uses_latent": False,
                "num_skills": self.num_skills,
                "train_step_count": self.train_step_count,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "geometric_p": self.config.geometric_p,
                "context_samples": FRE_CONTEXT_SAMPLES,
                "config": self.config.to_dict(),
            }
        )
        return info


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_gc_bc_agent(
    replay_buffer: Any = None,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
    goal_dim: Optional[int] = None,
    config: Optional[Any] = None,
    device: Optional[str] = None,
    seed: int = 0,
    **config_overrides,
) -> GCBCAgent:
    """Factory mirroring ``make_gc_iql_agent`` / ``make_opal_agent``."""
    return GCBCAgent(
        replay_buffer=replay_buffer,
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        config=config,
        device=device,
        seed=seed,
        **config_overrides,
    )


# ---------------------------------------------------------------------------
# self-check (synthetic data, no simulator required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    class _FakeBatch:
        def __init__(self, observations, actions, traj_index, step_index):
            self.observations = observations
            self.actions = actions
            self.traj_index = traj_index
            self.step_index = step_index

    class _FakeBuffer:
        def __init__(self, obs_dim=8, act_dim=3, num_traj=4, length=20, rng=None):
            self.obs_dim, self.act_dim = obs_dim, act_dim
            self.num_trajectories = num_traj
            self._length = length
            self.rng = rng or np.random.default_rng(0)
            self._states = self.rng.normal(size=(num_traj, length + 1, obs_dim))

        def trajectory_length(self, traj_index):
            return self._length

        def state_at(self, traj_index, step_index):
            return self._states[int(traj_index), int(np.clip(step_index, 0, self._length))]

        def sample_transitions(self, batch_size, rng=None, with_encoder_inputs=False):
            rng = rng or self.rng
            traj = rng.integers(0, self.num_trajectories, size=batch_size)
            step = rng.integers(0, self._length, size=batch_size)
            obs = np.stack([self._states[t, s] for t, s in zip(traj, step)])
            return _FakeBatch(obs, rng.uniform(-1, 1, size=(batch_size, self.act_dim)), traj, step)

    buf = _FakeBuffer()
    agent = make_gc_bc_agent(buf, device="cpu")
    hist = agent.train(num_steps=200, log_interval=100)
    print("first loss", hist[0]["loss"], "last loss", hist[-1]["loss"])
    assert hist[-1]["loss"] < hist[0]["loss"], "MLE loss should decrease"
    print(agent.describe())
