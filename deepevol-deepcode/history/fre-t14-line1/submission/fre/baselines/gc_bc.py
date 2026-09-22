"""Goal-Conditioned Behavioral Cloning (GC-BC) baseline for FRE.

Implements the GC-BC baseline described in the FRE paper (Sec. 5.2) and its
Addendum ("Additional Details on GC-BC"):

  - Network Architecture
      * MLP with three hidden layers of size 512
      * ReLU activations between each hidden layer
      * Layer normalization applied *before* each activation
      * Output layer predicts a Gaussian over actions (mean + log-std), with the
        log-std clamped from below at -5.0

  - Loss Function (maximum likelihood estimation, MLE)
        L_pi = -E_{(s, g, a) ~ D} log pi(a | s, g)                       (Eq. GC-BC)

  - Training: hindsight relabeling with *geometric-only* goal sampling (goals
    drawn from future states within the same trajectory; no random goals and no
    current-state goals).

  - Evaluation: the goal-conditioned agent is given the ground-truth goal of the
    evaluation task to condition on.

The goal is concatenated to the observation: the policy consumes ``cat(s, g)``.
This is implemented by reusing the shared :class:`fre.rl.networks.GaussianPolicy`
with ``latent_dim = goal_dim`` (the goal plays the role of ``z``), exactly as
:mod:`fre.baselines.gc_iql` does for IQL.  FRE, GC-IQL and GC-BC are all
implemented "within the same codebase and with the same network structure", so
reusing :mod:`fre.rl.networks` keeps them directly comparable.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fre.rl.networks import GaussianPolicy, MLP, make_activation

__all__ = [
    # constants
    "GEOMETRIC_P",
    "LOG_STD_MIN",
    "GCBC_TABLE1_REFERENCE",
    # helpers
    "geometric_goal_indices",
    "goal_rewards_and_dones",
    "bc_log_prob_loss",
    # trajectory bookkeeping
    "FlatTrajectoryIndex",
    "GeometricGoalSampler",
    "GoalBatch",
    # agent
    "GCBC",
    "make_gc_bc",
    "train_gc_bc",
    "make_gc_bc_policy_fn",
]

# ---------------------------------------------------------------------------
# Constants (paper + addendum)
# ---------------------------------------------------------------------------

#: Geometric (future-state) goal sampling probability used by GC-BC.  The
#: addendum states that GC-BC uses *only* geometric sampling, i.e. p=1.0 of the
#: goals come from future states (no random / current-state goals).  The value
#: below is the geometric distance-decay exponent-style probability used to pick
#: *which* future state (matching the GC-IQL ``p_geometric_goal=0.5`` used in
#: the HER relabeling), while the goal *type* is always "geometric".
GEOMETRIC_P: float = 0.5

#: Lower clamp on the predicted log standard deviation (addendum: -5.0).
LOG_STD_MIN: float = -5.0

#: Upper clamp.  The paper/addendum only specify a lower bound (-5.0); the upper
#: bound is unspecified and kept permissive so it does not alter behaviour.
LOG_STD_MAX: float = 2.0

#: Reference Table 1 numbers for the GC-BC baseline (paper Table 1).
GCBC_TABLE1_REFERENCE: Dict[str, float] = {
    "kitchen": 35.0,
    "kitchen_std": 9.0,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def geometric_goal_indices(
    indices: np.ndarray,
    traj_end: np.ndarray,
    rng: np.random.Generator,
    geometric_p: float = GEOMETRIC_P,
) -> np.ndarray:
    """Sample a *future* state index inside the same trajectory.

    Implements the "geometric" hindsight strategy: with probability
    ``geometric_p`` choose a uniformly random future timestep, otherwise choose
    the final state of the trajectory (a common convention that always yields a
    valid future goal).  All returned indices are strictly ``> index`` so the
    goal is never the current state (addendum: "no random goals, or goals which
    are the current state").

    Args:
        indices: (N,) flat dataset indices of the current transitions.
        traj_end: (T,) exclusive end index of each trajectory (prefix sums).
        rng: numpy random generator.
        geometric_p: probability of a uniformly random future state.

    Returns:
        (N,) int64 array of goal indices.
    """
    indices = np.asarray(indices, dtype=np.int64)
    traj_end = np.asarray(traj_end, dtype=np.int64)
    if indices.size == 0:
        return indices.astype(np.int64)

    # Locate the trajectory each index belongs to.
    traj_id = np.searchsorted(traj_end, indices, side="right")
    traj_id = np.clip(traj_id, 0, len(traj_end) - 1)
    ends = traj_end[traj_id]

    # Valid future range is (index, end)  -> exclusive end `ends`, inclusive
    # lower bound `index + 1`.
    lo = indices + 1
    # Guard degenerate trajectories (index is already the last state).
    degenerate = lo >= ends
    lo = np.where(degenerate, np.maximum(ends - 1, 0), lo)
    hi = np.maximum(ends, lo + 1)  # exclusive

    uniform_future = rng.integers(lo, hi)
    final_state = np.maximum(ends - 1, 0)

    use_uniform = rng.random(indices.shape[0]) < geometric_p
    goals = np.where(use_uniform, uniform_future, final_state)
    goals = np.clip(goals, 0, len(traj_end) and (traj_end[-1] - 1) or 0)
    return goals.astype(np.int64)


def goal_rewards_and_dones(
    states: np.ndarray,
    goals: np.ndarray,
    tolerance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Goal-conditioned reward/done preprocessor.

    ``reward = 0`` when the goal has been reached, ``-1`` otherwise; the episode
    is done when the goal is reached (addendum / GC-IQL convention).  GC-BC only
    consumes the goals (BC ignores rewards), but this helper is shared so the
    goal conventions stay identical across the goal-conditioned baselines.

    Returns:
        (rewards, dones) as float32 / bool arrays.
    """
    states = np.asarray(states, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    if goals.ndim == 1:
        goals = goals[None, :]
    distance = np.linalg.norm(states - goals, axis=-1)
    reached = distance <= tolerance + 1e-12
    rewards = np.where(reached, 0.0, -1.0).astype(np.float32)
    return rewards, reached.astype(bool)


def bc_log_prob_loss(
    log_prob: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MLE objective ``L_pi = -E log pi(a | s, g)`` (addendum).

    Args:
        log_prob: per-sample ``log pi(a_t | s_t, g)`` values.
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``.

    Returns:
        Scalar negative log-likelihood.
    """
    loss = -log_prob
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction '{reduction}'")


# ---------------------------------------------------------------------------
# Trajectory bookkeeping (geometric future-goal sampling)
# ---------------------------------------------------------------------------

class FlatTrajectoryIndex:
    """Flat dataset + trajectory boundaries for future-state goal sampling.

    The offline D4RL datasets are stored as concatenated episodes.  This class
    recovers episode boundaries (from ``terminals``/``timeouts`` or an explicit
    ``trajectory_ids`` array) and allows sampling
    ``idx -> goal_idx`` pairs where ``goal_idx`` is a *future* state of the same
    episode.
    """

    def __init__(
        self,
        states: np.ndarray,
        traj_end: np.ndarray,
        observations: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
    ) -> None:
        self.states = np.asarray(states, dtype=np.float32)
        self.traj_end = np.asarray(traj_end, dtype=np.int64)
        self.observations = (
            None if observations is None else np.asarray(observations, dtype=np.float32)
        )
        self.actions = None if actions is None else np.asarray(actions, dtype=np.float32)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_dataset(cls, dataset: Any) -> "FlatTrajectoryIndex":
        """Build the index from a ``ReplayBuffer``-like dataset."""
        states = _dataset_states(dataset)
        obs = getattr(dataset, "observations", None)
        obs = None if obs is None else np.asarray(obs, dtype=np.float32)
        actions = getattr(dataset, "actions", None)
        actions = None if actions is None else np.asarray(actions, dtype=np.float32)
        traj_end = _infer_trajectory_end(dataset, num_states=len(states))
        return cls(states, traj_end, observations=obs, actions=actions)

    @classmethod
    def from_transitions(
        cls,
        states: np.ndarray,
        terminals: Optional[np.ndarray] = None,
        timeouts: Optional[np.ndarray] = None,
        trajectory_ids: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
    ) -> "FlatTrajectoryIndex":
        states = np.asarray(states, dtype=np.float32)
        if trajectory_ids is not None:
            traj_end = _trajectory_end_from_ids(np.asarray(trajectory_ids))
        else:
            traj_end = _trajectory_end_from_masks(
                len(states), terminals=terminals, timeouts=timeouts
            )
        return cls(states, traj_end, observations=states, actions=actions)

    # -- sampling -----------------------------------------------------------
    @property
    def num_transitions(self) -> int:
        return int(len(self.states))

    @property
    def obs_dim(self) -> int:
        return int(self.states.shape[-1]) if self.states.ndim > 1 else 1

    def sample_indices(self, num: int, rng: np.random.Generator) -> np.ndarray:
        """Uniformly sample flat transition indices."""
        return rng.integers(0, self.num_transitions, size=int(num), dtype=np.int64)

    def future_indices(
        self,
        indices: np.ndarray,
        geometric_p: float = GEOMETRIC_P,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample future goal indices for the given transition indices."""
        if rng is None:
            rng = np.random.default_rng(0)
        return geometric_goal_indices(indices, self.traj_end, rng, geometric_p)

    def transitions(self, indices: np.ndarray) -> np.ndarray:
        """Return states for the given flat indices."""
        return self.states[np.asarray(indices, dtype=np.int64)]

    def actions_for(self, indices: np.ndarray) -> Optional[np.ndarray]:
        if self.actions is None:
            return None
        return self.actions[np.asarray(indices, dtype=np.int64)]


@dataclass
class GoalBatch:
    """A batch of hindsight-relabeled goals for GC-BC."""

    states: np.ndarray
    goals: np.ndarray
    actions: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    dones: Optional[np.ndarray] = None
    types: Optional[np.ndarray] = None
    dataset_indices: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def batch_size(self) -> int:
        return len(self)

    def type_counts(self) -> Dict[str, int]:
        if self.types is None:
            return {}
        values, counts = np.unique(self.types, return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "states": self.states,
            "goals": self.goals,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "types": self.types,
            "dataset_indices": self.dataset_indices,
        }


class GeometricGoalSampler:
    """Hindsight goal sampler using *geometric-only* goal selection.

    Unlike :class:`fre.baselines.gc_iql.GoalSampler` (which mixes current,
    geometric and random goals at 0.2/0.5/0.3), GC-BC only ever uses goals drawn
    from future states of the same trajectory (addendum).
    """

    def __init__(
        self,
        dataset: Any = None,
        traj_index: Optional[FlatTrajectoryIndex] = None,
        geometric_p: float = GEOMETRIC_P,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.traj_index = traj_index
        if self.traj_index is None and dataset is not None:
            self.traj_index = FlatTrajectoryIndex.from_dataset(dataset)
        self.geometric_p = float(geometric_p)
        self.rng = np.random.default_rng(seed)

    def attach_dataset(self, dataset: Any) -> None:
        self.dataset = dataset
        self.traj_index = FlatTrajectoryIndex.from_dataset(dataset)

    def sample(
        self,
        states: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> GoalBatch:
        """Sample a relabeled goal batch (states, goals, actions)."""
        if self.traj_index is None:
            raise RuntimeError("GeometricGoalSampler has no dataset attached.")

        if indices is None:
            size = int(batch_size or 512)
            indices = self.traj_index.sample_indices(size, self.rng)
        indices = np.asarray(indices, dtype=np.int64)

        states = self.traj_index.transitions(indices)
        goal_indices = self.traj_index.future_indices(
            indices, geometric_p=self.geometric_p, rng=self.rng
        )
        goals = self.traj_index.transitions(goal_indices)
        actions = self.traj_index.actions_for(indices)

        rewards, dones = goal_rewards_and_dones(states, goals)
        types = np.array(["geometric"] * len(indices), dtype=object)
        return GoalBatch(
            states=states,
            goals=goals,
            actions=actions,
            rewards=rewards,
            dones=dones,
            types=types,
            dataset_indices=indices,
        )


# ---------------------------------------------------------------------------
# GC-BC agent
# ---------------------------------------------------------------------------

class GCBC:
    """Goal-Conditioned Behavioral Cloning agent.

    The policy is the shared :class:`GaussianPolicy` from
    :mod:`fre.rl.networks` instantiated with ``latent_dim = goal_dim`` so that
    the goal is concatenated to the observation (``pi(a | cat(s, g))``), matching
    the FRE/GC-IQL codebase convention.  Only the policy is trained (MLE); no
    value function or critic is required.

    Args mirror :class:`fre.baselines.gc_iql.GCIQL` so the two baselines share
    the same interface (``select_action``/``update``/``train_steps``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        layernorm: bool = True,
        learning_rate: float = 1e-4,
        grad_clip_norm: float = 10.0,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        tanh_squash: bool = True,
        geometric_p: float = GEOMETRIC_P,
        max_log_std_clamp: Optional[float] = None,
        device: str = "cpu",
        discretize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: int = 0,
        **policy_kwargs: Any,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim if goal_dim is not None else obs_dim)
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.activation = activation
        self.layernorm = bool(layernorm)
        self.learning_rate = float(learning_rate)
        self.grad_clip_norm = float(grad_clip_norm)
        self.log_std_min = float(log_std_min)
        # The addendum only specifies a *lower* clamp (-5.0); `max_log_std_clamp`
        # lets callers apply the GC-BC specific lower bound while leaving the
        # upper bound as the shared default.
        self.log_std_max = float(
            log_std_max if max_log_std_clamp is None else max_log_std_clamp
        )
        self.tanh_squash = bool(tanh_squash)
        self.geometric_p = float(geometric_p)
        self.device = torch.device(device)
        self.discretize_fn = discretize_fn
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.policy = GaussianPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.goal_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            tanh_squash=self.tanh_squash,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            layernorm=self.layernorm,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

        self.traj_index: Optional[FlatTrajectoryIndex] = None
        self.goal_sampler: Optional[GeometricGoalSampler] = None
        self._train_steps = 0

    # -- dataset / preprocessing -------------------------------------------
    def attach_dataset(self, dataset: Any) -> None:
        """Attach an offline dataset for geometric future-goal sampling."""
        self.traj_index = FlatTrajectoryIndex.from_dataset(dataset)
        self.goal_sampler = GeometricGoalSampler(
            dataset=dataset,
            traj_index=self.traj_index,
            geometric_p=self.geometric_p,
            seed=self.seed,
        )

    def prepare(self, states: np.ndarray) -> np.ndarray:
        """Apply the shared observation/goal preprocessing (e.g. AntMaze XY 32-bin)."""
        states = np.asarray(states, dtype=np.float32)
        if self.discretize_fn is not None:
            states = self.discretize_fn(states)
        states = np.asarray(states, dtype=np.float32)
        if states.shape[-1] > self.obs_dim:
            states = states[..., : self.obs_dim]
        elif states.shape[-1] < self.obs_dim:
            pad = np.zeros(states.shape[:-1] + (self.obs_dim - states.shape[-1],),
                           dtype=states.dtype)
            states = np.concatenate([states, pad], axis=-1)
        return states

    def _prepare_goals(self, goals: np.ndarray) -> np.ndarray:
        goals = self.prepare(goals)
        if goals.shape[-1] > self.goal_dim:
            goals = goals[..., : self.goal_dim]
        elif goals.shape[-1] < self.goal_dim:
            pad = np.zeros(goals.shape[:-1] + (self.goal_dim - goals.shape[-1],),
                           dtype=goals.dtype)
            goals = np.concatenate([goals, pad], axis=-1)
        return goals

    # -- sampling -----------------------------------------------------------
    def sample_goals(
        self,
        states: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> GoalBatch:
        """Geometric-only hindsight goal batch."""
        if self.goal_sampler is None:
            raise RuntimeError(
                "GCBC.sample_goals requires a dataset; call attach_dataset(...) first."
            )
        batch = self.goal_sampler.sample(
            states=states, indices=indices, batch_size=batch_size
        )
        return batch

    # -- update -------------------------------------------------------------
    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    def update(
        self,
        batch: Any = None,
        dataset: Any = None,
        batch_size: int = 512,
        geometric_p: Optional[float] = None,
    ) -> Dict[str, float]:
        """One MLE gradient step on ``L_pi = -E log pi(a | s, g)``.

        Either pass a pre-sampled ``GoalBatch`` via ``batch`` or a ``dataset``
        from which geometric future goals will be sampled.
        """
        if batch is None:
            if dataset is not None:
                if self.goal_sampler is None or self.goal_sampler.dataset is not dataset:
                    self.attach_dataset(dataset)
            elif self.goal_sampler is None:
                raise ValueError("GCBC.update requires either `batch` or `dataset`.")
            if geometric_p is not None:
                self.goal_sampler.geometric_p = float(geometric_p)
            batch = self.sample_goals(batch_size=batch_size)

        if isinstance(batch, dict):
            states = batch["states"]
            goals = batch["goals"]
            actions = batch.get("actions")
        else:
            states = batch.states
            goals = batch.goals
            actions = batch.actions

        if actions is None:
            raise ValueError(
                "GCBC.update requires dataset actions (BC is an off-policy "
                "supervised method)."
            )

        states_t = self._to_tensor(self.prepare(states))
        goals_t = self._to_tensor(self._prepare_goals(goals))
        actions_t = self._to_tensor(actions)[..., : self.action_dim]

        _, log_prob = self.policy(states_t, goals_t, with_log_prob=True)
        loss = bc_log_prob_loss(log_prob, reduction="mean")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self._train_steps += 1

        with torch.no_grad():
            mean = log_prob.mean()
            per_sample_nll = -log_prob
        return {
            "loss": float(loss.detach().cpu()),
            "nll": float(per_sample_nll.mean().detach().cpu()),
            "log_prob": float(mean.detach().cpu()),
            "steps": self._train_steps,
        }

    def train_steps(
        self,
        dataset: Any,
        steps: int,
        batch_size: int = 512,
        log_interval: int = 1000,
        logger: Any = None,
        prefix: str = "gc_bc",
        geometric_p: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Run ``steps`` MLE updates, logging periodically.

        Returns a list of ``{"step", "loss", ...}`` metric dicts.
        """
        if self.goal_sampler is None or self.goal_sampler.dataset is not dataset:
            self.attach_dataset(dataset)
        if geometric_p is not None:
            self.geometric_p = float(geometric_p)
            self.goal_sampler.geometric_p = float(geometric_p)

        history: List[Dict[str, float]] = []
        for step in range(1, int(steps) + 1):
            metrics = self.update(dataset=dataset, batch_size=batch_size)
            metrics["step"] = step
            history.append(metrics)
            if log_interval and (step % int(log_interval) == 0 or step == 1):
                if logger is not None:
                    payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
                    try:
                        logger.log(payload, step=step)
                    except TypeError:
                        logger.log(payload)
                else:
                    print(
                        f"[{prefix}] step {step}/{steps} "
                        f"loss={metrics['loss']:.4f} logp={metrics['log_prob']:.4f}"
                    )
        return history

    # -- acting -------------------------------------------------------------
    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        deterministic: bool = True,
        clip: bool = True,
    ) -> np.ndarray:
        """Return an action conditioned on the ground-truth evaluation goal."""
        obs_b = np.asarray(obs, dtype=np.float32)
        single = obs_b.ndim == 1
        if single:
            obs_b = obs_b[None, :]
        goal_b = np.asarray(goal, dtype=np.float32)
        if goal_b.ndim == 1:
            goal_b = goal_b[None, :]

        obs_t = self._to_tensor(self.prepare(obs_b))
        goal_t = self._to_tensor(self._prepare_goals(goal_b))
        if goal_t.shape[0] == 1 and obs_t.shape[0] > 1:
            goal_t = goal_t.expand(obs_t.shape[0], -1)

        action = self.policy.act(obs_t, goal_t, deterministic=deterministic)
        action_np = action.detach().cpu().numpy()
        if clip:
            action_np = np.clip(action_np, -1.0, 1.0)
        return action_np[0] if single else action_np

    # -- misc ---------------------------------------------------------------
    def to(self, device: Any) -> "GCBC":
        self.device = torch.device(device)
        self.policy.to(self.device)
        return self

    def train(self) -> "GCBC":
        self.policy.train()
        return self

    def eval(self) -> "GCBC":
        self.policy.eval()
        return self

    def parameters(self):
        return self.policy.parameters()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self._train_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        payload = state_dict.get("policy", state_dict)
        self.policy.load_state_dict(payload)
        if "optimizer" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            except (ValueError, KeyError):
                pass
        self._train_steps = int(state_dict.get("steps", 0))

    def save(self, path: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, map_location: Optional[str] = None) -> "GCBC":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload)
        return self

    # -- factory ------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: Any,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        discretize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **overrides: Any,
    ) -> "GCBC":
        """Build a GC-BC agent from a ``fre.config.default.Config``-like object."""
        kwargs: Dict[str, Any] = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            hidden_layers=tuple(
                getattr(config, "rl_hidden_layers", (512, 512, 512))
            ),
            activation=getattr(config, "rl_activation", "relu"),
            layernorm=bool(getattr(config, "rl_layernorm", True)),
            learning_rate=float(getattr(config, "learning_rate", 1e-4)),
            grad_clip_norm=float(getattr(config, "grad_clip_norm", 10.0)),
            log_std_min=float(getattr(config, "log_std_min", LOG_STD_MIN)),
            log_std_max=float(getattr(config, "log_std_max", LOG_STD_MAX)),
            tanh_squash=bool(getattr(config, "rl_tanh_squash", True)),
            geometric_p=float(getattr(config, "geometric_p", GEOMETRIC_P)),
            device=getattr(config, "device", "cpu"),
            discretize_fn=discretize_fn,
            seed=int(getattr(config, "seed", 0)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Convenience drivers
# ---------------------------------------------------------------------------

def make_gc_bc(
    config: Any,
    obs_dim: int,
    action_dim: int,
    goal_dim: Optional[int] = None,
    discretize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **overrides: Any,
) -> GCBC:
    """Factory mirroring :func:`fre.baselines.gc_iql.make_gc_iql`."""
    return GCBC.from_config(
        config,
        obs_dim,
        action_dim,
        goal_dim=goal_dim,
        discretize_fn=discretize_fn,
        **overrides,
    )


def train_gc_bc(
    config: Any,
    dataset: Any,
    obs_dim: int,
    action_dim: int,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    log_interval: int = 1000,
    logger: Any = None,
    discretize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **overrides: Any,
) -> Tuple[GCBC, List[Dict[str, float]]]:
    """Train a GC-BC agent on an offline dataset.

    Defaults follow the FRE paper/comparison protocol: the same network
    structure and learning rate as FRE/GC-IQL, trained for the same number of
    steps as the model-free baselines.
    """
    agent = make_gc_bc(
        config=config,
        obs_dim=obs_dim,
        action_dim=action_dim,
        discretize_fn=discretize_fn,
        **overrides,
    )
    steps = int(steps if steps is not None else getattr(config, "policy_train_steps", 250_000))
    batch_size = int(batch_size if batch_size is not None else getattr(config, "batch_size", 512))
    agent.attach_dataset(dataset)
    history = agent.train_steps(
        dataset=dataset,
        steps=steps,
        batch_size=batch_size,
        log_interval=log_interval,
        logger=logger,
        prefix="gc_bc",
    )
    return agent, history


def make_gc_bc_policy_fn(
    agent: GCBC,
    goal: np.ndarray,
    deterministic: bool = True,
    clip: bool = True,
    discretize: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a trained GC-BC agent into ``act_fn(obs) -> action``.

    The evaluation task supplies the ground-truth goal (addendum: "the
    goal-conditioned agent is given the ground-truth goal that the specific
    evaluation task contains, to condition on").
    """
    goal_arr = np.asarray(goal, dtype=np.float32)
    if discretize and getattr(agent, "discretize_fn", None) is not None:
        goal_arr = agent.discretize_fn(goal_arr)

    def act_fn(obs: np.ndarray) -> np.ndarray:
        return agent.select_action(obs, goal_arr, deterministic=deterministic, clip=clip)

    return act_fn


# ---------------------------------------------------------------------------
# Internal dataset helpers (shared with GC-IQL conventions)
# ---------------------------------------------------------------------------

def _dataset_states(dataset: Any) -> np.ndarray:
    """Obtain a flat ``(N, state_dim)`` array from a ReplayBuffer-like object."""
    for attr in ("states", "observations", "obs"):
        value = getattr(dataset, attr, None)
        if value is not None:
            arr = np.asarray(value)
            # `states` may already be the physics-augmented view.
            return arr
    if hasattr(dataset, "sample_states"):
        return np.asarray(dataset.sample_states(len(dataset)) if hasattr(dataset, "__len__") else dataset.sample_states(10_000))
    raise TypeError("Dataset does not expose states/observations for GC-BC.")


def _infer_trajectory_end(dataset: Any, num_states: int) -> np.ndarray:
    """Recover exclusive trajectory-end indices from a ReplayBuffer-like object."""
    trajectory_ids = getattr(dataset, "trajectory_ids", None)
    if trajectory_ids is None:
        trajectory_ids = getattr(dataset, "episode_ids", None)
    if trajectory_ids is not None:
        return _trajectory_end_from_ids(np.asarray(trajectory_ids))

    terminals = getattr(dataset, "terminals", None)
    timeouts = getattr(dataset, "timeouts", None)
    if terminals is not None or timeouts is not None:
        return _trajectory_end_from_masks(
            num_states, terminals=terminals, timeouts=timeouts
        )

    # Fall back to a single trajectory (rare; keeps sampling well-defined).
    return np.array([num_states], dtype=np.int64)


def _trajectory_end_from_ids(ids: np.ndarray) -> np.ndarray:
    """Exclusive end index per trajectory from an episode-id array."""
    ids = np.asarray(ids).reshape(-1)
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    # `ids` are assumed to be non-decreasing (D4RL convention).
    change = np.nonzero(np.diff(ids) != 0)[0] + 1
    ends = np.concatenate([change, [len(ids)]]).astype(np.int64)
    return ends


def _trajectory_end_from_masks(
    num_states: int,
    terminals: Optional[np.ndarray] = None,
    timeouts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exclusive end index per trajectory from done/timeout masks."""
    done = np.zeros(int(num_states), dtype=bool)
    if terminals is not None:
        done |= np.asarray(terminals).reshape(-1).astype(bool)
    if timeouts is not None:
        done |= np.asarray(timeouts).reshape(-1).astype(bool)
    if done.size:
        done[-1] = True
        ends = (np.nonzero(done)[0] + 1).astype(np.int64)
    else:
        ends = np.array([int(num_states)], dtype=np.int64)
    if ends.size == 0 or ends[-1] != num_states:
        ends = np.concatenate([ends, [int(num_states)]]).astype(np.int64)
    return ends
