"""ExORL (Cheetah / Walker) zero-shot evaluation tasks for FRE.

Paper references
----------------
Appendix C.2 (ExORL):
    "We utilize cheetah-run, cheetah-walk, cheetah-run-backwards,
    cheetah-walk-backwards and walker-run, walker-walk as evaluation tasks.
    Agents are evaluated for 1000 timesteps. For goal-reaching tasks, we select
    five consistent goal states from the offline dataset."
    Physics values (``horizontal_velocity``, ``torso_upright``,
    ``torso_height`` for Walker; ``speed`` for Cheetah) are appended to the
    encoder input.  "Goals in ExORL are computed when the Euclidean distance
    between the current state and the goal state is less than 0.1.  Each state
    dimension is normalized according to the standard deviation along that
    dimension within the offline dataset.  Augmented information is not
    utilized when calculating goal distance."

Addendum -- ExORL evaluation tasks:
    * ``exorl-cheetah-velocity``: average over 4 tasks (``cheetah-run`` with
      threshold 10, ``cheetah-run-backwards``, ``cheetah-walk`` with threshold
      1, ``cheetah-walk-backwards``).  "The reward is 1 if the velocity is 10
      and linearly decays to 0 for values below 10.  If the agent's horizontal
      velocity is in the opposite direction of the target velocity, the reward
      is 0."
    * ``exorl-walker-velocity``: 4 tasks with thresholds 0.1, 1, 4 and 8, same
      reward shape.
    * ``exorl-cheetah-goals`` / ``exorl-walker-goals``: 5 fixed goal states
      sampled from the offline dataset, reward -1 per step unless within
      Euclidean distance 0.1 of the goal, in which case reward 0.
    * "The online evaluation is performed with a *maximum* length of 1000
      steps *per trajectory*."
    * "Finally, the information about the physics used for training that is
      mentioned in Appendix C.2 is also used during evaluation."

Design notes (paper silent -- sensible defaults marked below)
-------------------------------------------------------------
* The engine of the velocity reward is implemented exactly as the text above
  reads: ``r = clip(v_target / threshold, 0, 1)`` so that ``r = 1`` at the
  threshold, decays linearly to 0 for smaller velocities, and is 0 when
  ``v_target < 0`` (moving opposite to the target direction).
* Backwards tasks are implemented by negating the signed velocity before
  applying the shared rule (target direction is the negative axis).
* Goal distance normalizes each dimension by the offline dataset standard
  deviation and ignores the appended physics features, exactly as C.2 says.
* The paper does not state whether velocities are clipped above the threshold;
  ``clip=True`` (reward capped at 1) is used because the text says "the reward
  is 1 if the velocity is <threshold>".
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Imports of the shared task containers (with a direct-execution fallback and
# a minimal degraded fallback so this module can always be imported).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from fre.envs import EvalTask, TaskSuite, build_suite
except Exception:  # pragma: no cover - direct execution / partial installs
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_HERE))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    try:
        from fre.envs import EvalTask, TaskSuite, build_suite
    except Exception:

        @dataclass
        class EvalTask:  # type: ignore[no-redef]
            """Minimal stand-in used only if ``fre.envs`` cannot be imported."""

            name: str
            reward_fn: Callable[[np.ndarray], np.ndarray]
            domain: str = "generic"
            task_group: str = "generic"
            is_goal_task: bool = False
            goal: Any = None
            threshold: Optional[float] = None
            eval_episode_length: int = 1000
            reward_min: Optional[float] = None
            reward_max: Optional[float] = None
            metadata: Dict[str, Any] = field(default_factory=dict)

            def reward(self, observations):
                return np.asarray(self.reward_fn(observations), dtype=np.float64)

            def __call__(self, observations):
                return self.reward(observations)

            def success(self, observations, threshold=None):
                r = self.reward(observations)
                t = threshold if threshold is not None else self.reward_max
                if t is None:
                    t = float(np.max(r)) if r.size else 0.0
                return r >= (t - 1e-8)

            def describe(self):
                return {
                    "name": self.name,
                    "domain": self.domain,
                    "task_group": self.task_group,
                    "is_goal_task": self.is_goal_task,
                }

        @dataclass
        class TaskSuite:  # type: ignore[no-redef]
            name: str
            tasks: List[Any]
            eval_episode_length: int = 1000
            aggregate: Any = None
            domain: str = "generic"
            metadata: Dict[str, Any] = field(default_factory=dict)

            def __len__(self):
                return len(self.tasks)

            def __iter__(self):
                return iter(self.tasks)

            def __getitem__(self, index):
                return self.tasks[index]

            @property
            def names(self):
                return [t.name for t in self.tasks]

            def get(self, name):
                for t in self.tasks:
                    if t.name == name:
                        return t
                raise KeyError(name)

            def describe(self):
                return {"name": self.name, "tasks": self.names}

        def build_suite(name, tasks, eval_episode_length=None, aggregate=None, domain=None, metadata=None):  # type: ignore[no-redef]
            tasks = list(tasks)
            if eval_episode_length is None and tasks:
                eval_episode_length = tasks[0].eval_episode_length
            if domain is None and tasks:
                domain = tasks[0].domain
            return TaskSuite(
                name=name,
                tasks=tasks,
                eval_episode_length=eval_episode_length if eval_episode_length else 1000,
                aggregate=aggregate,
                domain=domain if domain else "generic",
                metadata=dict(metadata or {}),
            )

# ---------------------------------------------------------------------------
# Constants from the ExORL data module (fall back to the paper values).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from fre.data.exorl_dataset import (
        CHEETAH_PHYSICS_FEATURES,
        EXORL_EVAL_EPISODE_LENGTH,
        EXORL_GOAL_DISTANCE_THRESHOLD,
        NUM_EXORL_GOALS,
        PHYSICS_FEATURES,
        WALKER_PHYSICS_FEATURES,
    )
except Exception:  # pragma: no cover - direct execution / partial installs
    CHEETAH_PHYSICS_FEATURES = ("speed",)
    WALKER_PHYSICS_FEATURES = ("horizontal_velocity", "torso_upright", "torso_height")
    PHYSICS_FEATURES = {"cheetah": CHEETAH_PHYSICS_FEATURES, "walker": WALKER_PHYSICS_FEATURES}
    EXORL_EVAL_EPISODE_LENGTH = 1000
    EXORL_GOAL_DISTANCE_THRESHOLD = 0.1
    NUM_EXORL_GOALS = 5

__all__ = [
    # task classes
    "ExORLTask",
    "ExORLVelocityTask",
    "ExORLGoalTask",
    # velocity factories
    "make_cheetah_velocity_tasks",
    "make_walker_velocity_tasks",
    "make_exorl_velocity_tasks",
    # goal factories
    "make_exorl_goal_tasks",
    # suites
    "make_exorl_task_suite",
    "exorl_suite_for_group",
    "exorl_task_groups",
    # helpers
    "exorl_velocity_reward",
    "normalized_goal_distance",
    "physics_velocity",
    "resolve_velocity_dim",
    "velocity_dim_for_domain",
    # constants
    "EXORL_DOMAIN",
    "EXORL_DOMAINS",
    "EXORL_EVAL_EPISODE_LENGTH",
    "EXORL_GOAL_DISTANCE_THRESHOLD",
    "EXORL_NUM_GOALS",
    "EXORL_VELOCITY_REWARD_MAX",
    "CHEETAH_VELOCITY_THRESHOLDS",
    "WALKER_VELOCITY_THRESHOLDS",
    "CHEETAH_TASK_NAMES",
    "WALKER_TASK_NAMES",
    "EXORL_TASK_GROUPS",
    "EXORL_GOAL_TASK_GROUP_TEMPLATE",
    "EXORL_VELOCITY_TASK_GROUP_TEMPLATE",
    "PHYSICS_FEATURES",
    "CHEETAH_PHYSICS_FEATURES",
    "WALKER_PHYSICS_FEATURES",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXORL_DOMAIN = "exorl"
EXORL_DOMAINS: Tuple[str, ...] = ("cheetah", "walker")

#: Maximum length of an evaluation trajectory (Addendum: "maximum length of
#: 1000 steps per trajectory").
EXORL_EVAL_EPISODE_LENGTH = int(EXORL_EVAL_EPISODE_LENGTH)

#: Goal-reaching threshold (Appendix C.2: Euclidean distance less than 0.1).
EXORL_GOAL_DISTANCE_THRESHOLD = float(EXORL_GOAL_DISTANCE_THRESHOLD)

#: Five fixed goal states per domain (Appendix C.2 / Addendum).
EXORL_NUM_GOALS = int(NUM_EXORL_GOALS)

#: Velocity rewards are bounded in [0, 1].
EXORL_VELOCITY_REWARD_MAX = 1.0
EXORL_VELOCITY_REWARD_MIN = 0.0

#: Addendum: cheetah velocity tasks reward velocities of at least 10 and 1.
CHEETAH_VELOCITY_THRESHOLDS: Tuple[float, ...] = (10.0, 1.0)

#: Addendum: "The 4 tasks use values of 0.1, 1, 4, and 8 respectively."
WALKER_VELOCITY_THRESHOLDS: Tuple[float, ...] = (0.1, 1.0, 4.0, 8.0)

#: Task names (matching the addendum's names where it gives them).
CHEETAH_RUN = "cheetah-run"
CHEETAH_RUN_BACKWARDS = "cheetah-run-backwards"
CHEETAH_WALK = "cheetah-walk"
CHEETAH_WALK_BACKWARDS = "cheetah-walk-backwards"
CHEETAH_TASK_NAMES: Tuple[str, ...] = (
    CHEETAH_RUN,
    CHEETAH_RUN_BACKWARDS,
    CHEETAH_WALK,
    CHEETAH_WALK_BACKWARDS,
)
WALKER_TASK_NAMES: Tuple[str, ...] = (
    "walker-velocity-0.1",
    "walker-velocity-1",
    "walker-velocity-4",
    "walker-velocity-8",
)

EXORL_VELOCITY_TASK_GROUP_TEMPLATE = "exorl-{domain}-velocity"
EXORL_GOAL_TASK_GROUP_TEMPLATE = "exorl-{domain}-goals"

#: Suite/group names accepted by :func:`make_exorl_task_suite`.
EXORL_TASK_GROUPS: Tuple[str, ...] = (
    "exorl-all",
    "exorl-velocity",
    "exorl-goals",
    "exorl-cheetah",
    "exorl-walker",
    "exorl-cheetah-velocity",
    "exorl-cheetah-goals",
    "exorl-walker-velocity",
    "exorl-walker-goals",
)


# ---------------------------------------------------------------------------
# Reward helpers (pure functions of the observations / physics values)
# ---------------------------------------------------------------------------
def exorl_velocity_reward(
    velocity: np.ndarray,
    threshold: float,
    backwards: bool = False,
    clip: bool = True,
) -> np.ndarray:
    """Per-state velocity reward of the ExORL velocity tasks.

    Implements the addendum rule verbatim:

    * ``reward = 1`` when the (signed) target-direction velocity is at least
      ``threshold``;
    * linearly decays to ``0`` for values below ``threshold``;
    * ``reward = 0`` when the velocity is in the opposite direction of the
      target velocity.

    Parameters
    ----------
    velocity:
        Horizontal velocity of the agent (1-D array or scalar).
    threshold:
        Target velocity magnitude, e.g. 10 (cheetah-run) or 0.1 (walker).
    backwards:
        If ``True`` the target velocity points along the negative axis, i.e.
        the reward uses ``-velocity``.
    clip:
        Cap the reward at 1 (correct: "the reward is 1 if the velocity is
        <threshold>").  The reward is always floored at 0.

    Returns
    -------
    numpy.ndarray
        Reward with the same shape as ``velocity``.
    """
    v = np.asarray(velocity, dtype=np.float64)
    if backwards:
        v = -v
    threshold = float(max(abs(threshold), 1e-8))
    reward = v / threshold  # linear decay from 1 at the threshold to 0 at 0
    if clip:
        reward = np.clip(reward, 0.0, 1.0)
    else:
        reward = np.maximum(reward, 0.0)
    return reward


def normalized_goal_distance(
    observations: np.ndarray,
    goal: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    dims: Optional[Sequence[int]] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Euclidean distance between states and a goal on std-normalized dims.

    Appendix C.2: "Each state dimension is normalized according to the
    standard deviation along that dimension within the offline dataset.
    Augmented information is not utilized when calculating goal distance."
    """
    obs = _as_obs_matrix(observations)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    if dims is not None:
        dims = np.asarray(dims, dtype=int)
        obs = obs[:, dims]
        goal = goal[dims]
    scale = np.ones(obs.shape[-1], dtype=np.float64)
    if std is not None and std.size:
        scale = np.asarray(std, dtype=np.float64).reshape(-1)
        if dims is not None:
            scale = scale[dims]
        scale = np.where(np.abs(scale) < eps, 1.0, scale)
    shift = np.zeros(obs.shape[-1], dtype=np.float64)
    if mean is not None and mean.size:
        shift = np.asarray(mean, dtype=np.float64).reshape(-1)
        if dims is not None:
            shift = shift[dims]
    diff = (obs - shift) / scale - (goal - shift) / scale
    return np.sqrt(np.sum(diff * diff, axis=-1))


def resolve_velocity_dim(
    domain: str,
    agent_obs_dim: Optional[int] = None,
    num_physics: Optional[int] = None,
    velocity_dim: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Return the index of the velocity feature inside an observation vector.

    Appendix C.2 appends the physics quantities *after* the environment state,
    so with augmented observations the velocity lives at index
    ``agent_obs_dim`` (the first appended physics feature: ``speed`` for
    cheetah, ``horizontal_velocity`` for walker).

    If ``velocity_dim`` is given explicitly it wins.  If ``agent_obs_dim`` is
    unknown as well, dimension 0 is used as a documented fallback (this is the
    correct choice when the caller passes the physics vector alone).
    """
    if velocity_dim is not None:
        return int(velocity_dim), {"velocity_dim_source": "explicit"}
    if agent_obs_dim is not None:
        return int(agent_obs_dim), {"velocity_dim_source": "agent_obs_dim"}
    if num_physics is None:
        num_physics = len(PHYSICS_FEATURES.get(domain, ()))
    # Fallback (paper silent): with no observation space information assume the
    # caller supplied the augmented physics block itself, whose first entry is
    # the velocity feature.
    return 0, {
        "velocity_dim_source": "physics_block_fallback",
        "num_physics": int(num_physics),
    }


def physics_velocity(
    observations: np.ndarray,
    velocity_dim: int,
    dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Extract the signed velocity used by the velocity reward functions."""
    obs = _as_obs_matrix(observations)
    if dims is not None and obs.shape[-1] > max(dims) and len(dims) == obs.shape[-1]:
        # Observations are already the physics vector.
        return obs[:, 0]
    idx = int(velocity_dim)
    if idx >= obs.shape[-1]:
        idx = obs.shape[-1] - 1
    return obs[:, idx]


def velocity_dim_for_domain(domain: str, agent_obs_dim: Optional[int] = None) -> int:
    """Convenience wrapper around :func:`resolve_velocity_dim`."""
    return resolve_velocity_dim(domain, agent_obs_dim=agent_obs_dim)[0]


def _as_obs_matrix(observations: Any) -> np.ndarray:
    """Coerce observations to a 2-D float array ``(N, obs_dim)``."""
    if hasattr(observations, "detach"):  # torch tensor without importing torch
        observations = observations.detach().cpu().numpy()
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim == 0:
        obs = obs.reshape(1, 1)
    elif obs.ndim == 1:
        obs = obs.reshape(1, -1)
    return obs


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------
class ExORLTask(EvalTask):
    """Base class for ExORL evaluation tasks.

    Subclasses ``fre.envs.EvalTask`` so the zero-shot harness can treat every
    task uniformly (``task.reward(obs)`` -> rewards of shape ``(N,)``).
    """

    def __init__(
        self,
        name: str,
        reward_fn: Callable[[Any], np.ndarray],
        domain: str,
        task_group: str,
        is_goal_task: bool = False,
        goal: Any = None,
        threshold: Optional[float] = None,
        eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        meta = dict(metadata or {})
        meta.setdefault("domain", domain)
        meta.setdefault("task_group", task_group)
        meta.setdefault(
            "score_bounds",
            (0.0 if reward_min is None else float(reward_min),
             1.0 if reward_max is None else float(reward_max)),
        )
        if extra:
            meta.setdefault("extra", extra)
        super().__init__(
            name=name,
            reward_fn=reward_fn,
            domain=domain,
            task_group=task_group,
            is_goal_task=is_goal_task,
            goal=goal,
            threshold=threshold,
            eval_episode_length=int(eval_episode_length),
            reward_min=reward_min,
            reward_max=reward_max,
            metadata=meta,
        )

    # -- shared interface ---------------------------------------------------
    def rewards(self, observations: Any) -> np.ndarray:
        """Vectorized reward for a batch of observations (shape ``(N,)``)."""
        return np.asarray(self.reward_fn(observations), dtype=np.float64).reshape(-1)

    def scores_bounds(self) -> Tuple[float, float]:
        """Return the ``(min, max)`` reward used for 0-100 normalization."""
        lo = 0.0 if self.reward_min is None else float(self.reward_min)
        hi = 1.0 if self.reward_max is None else float(self.reward_max)
        return lo, hi

    def copy_with_name(self, name: str) -> "ExORLTask":
        """Copy this task under a different name (used by suite aggregation)."""
        import copy as _copy

        clone = _copy.copy(self)
        clone.name = name
        return clone

    def describe(self) -> Dict[str, Any]:
        info = {
            "name": self.name,
            "domain": self.domain,
            "task_group": self.task_group,
            "is_goal_task": bool(self.is_goal_task),
            "eval_episode_length": int(self.eval_episode_length),
            "reward_bounds": self.scores_bounds(),
        }
        if self.goal is not None:
            info["goal"] = np.asarray(self.goal, dtype=np.float64).tolist()
        if self.threshold is not None:
            info["threshold"] = float(self.threshold)
        for key in ("velocity_threshold", "backwards", "velocity_dim",
                    "num_physics", "goal_distance_threshold"):
            if key in self.metadata:
                info[key] = self.metadata[key]
        return info


class ExORLVelocityTask(ExORLTask):
    """Velocity-tracking task: ``r = clip(v_target / threshold, 0, 1)``.

    The velocity is read from the physics block appended to the observation
    vector (Appendix C.2 / Addendum: the physics information used for training
    "is also used during evaluation").
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        domain: str,
        backwards: bool = False,
        velocity_dim: Optional[int] = None,
        agent_obs_dim: Optional[int] = None,
        num_physics: Optional[int] = None,
        physics_dims: Optional[Sequence[int]] = None,
        clip: bool = True,
        eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
        task_group: Optional[str] = None,
    ) -> None:
        if num_physics is None:
            num_physics = len(PHYSICS_FEATURES.get(domain, ()))
        if physics_dims is None:
            physics_dims = list(PHYSICS_FEATURES.get(domain, ()))
            physics_dims = list(range(len(physics_dims)))
        self.threshold = float(threshold)
        self.backwards = bool(backwards)
        self.domain = domain
        self.clip_velocity_reward = bool(clip)
        self.physics_dims = tuple(int(d) for d in physics_dims)
        self.num_physics = int(num_physics)
        self.velocity_dim, dim_info = resolve_velocity_dim(
            domain,
            agent_obs_dim=agent_obs_dim,
            num_physics=num_physics,
            velocity_dim=velocity_dim,
        )
        group = task_group or EXORL_VELOCITY_TASK_GROUP_TEMPLATE.format(domain=domain)
        metadata = {
            "velocity_threshold": float(threshold),
            "backwards": bool(backwards),
            "velocity_dim": int(self.velocity_dim),
            "num_physics": int(self.num_physics),
            "score_bounds": (EXORL_VELOCITY_REWARD_MIN, EXORL_VELOCITY_REWARD_MAX),
            "physics_features": list(PHYSICS_FEATURES.get(domain, ())),
        }
        metadata.update(dim_info)
        super().__init__(
            name=name,
            reward_fn=self._eval_reward,
            domain=domain,
            task_group=group,
            is_goal_task=False,
            goal=None,
            threshold=float(threshold),
            eval_episode_length=eval_episode_length,
            reward_min=EXORL_VELOCITY_REWARD_MIN,
            reward_max=EXORL_VELOCITY_REWARD_MAX,
            metadata=metadata,
        )

    # -- reward -------------------------------------------------------------
    def velocities(self, observations: Any) -> np.ndarray:
        """Signed horizontal velocity for each observation."""
        return physics_velocity(observations, self.velocity_dim)

    def reward(self, observations: Any) -> np.ndarray:
        return self._eval_reward(observations)

    def _eval_reward(self, observations: Any) -> np.ndarray:
        v = self.velocities(observations)
        return exorl_velocity_reward(
            v,
            threshold=self.threshold,
            backwards=self.backwards,
            clip=self.clip_velocity_reward,
        )

    def __call__(self, observations: Any) -> np.ndarray:
        return self._eval_reward(observations)

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["velocity_threshold"] = float(self.threshold)
        info["backwards"] = bool(self.backwards)
        return info


class ExORLGoalTask(ExORLTask):
    """Fixed-goal reaching task with the paper's -1 / 0 sparse reward.

    Appendix C.2: "Goals in ExORL are computed when the Euclidean distance
    between the current state and the goal state is less than 0.1.  Each state
    dimension is normalized according to the standard deviation along that
    dimension within the offline dataset.  Augmented information is not
    utilized when calculating goal distance."
    """

    def __init__(
        self,
        name: str,
        goal: np.ndarray,
        domain: str,
        threshold: float = EXORL_GOAL_DISTANCE_THRESHOLD,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        agent_obs_dim: Optional[int] = None,
        dims: Optional[Sequence[int]] = None,
        reward_unreached: float = -1.0,
        reward_reached: float = 0.0,
        terminate_on_success: bool = False,
        eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
        task_group: Optional[str] = None,
    ) -> None:
        self.domain = domain
        self.goal_state = np.asarray(goal, dtype=np.float64).reshape(-1)
        self.threshold_distance = float(threshold)
        self.state_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float64).reshape(-1)
        self.state_std = None if state_std is None else np.asarray(state_std, dtype=np.float64).reshape(-1)
        self.agent_obs_dim = None if agent_obs_dim is None else int(agent_obs_dim)
        self.dims = None if dims is None else np.asarray(dims, dtype=int)
        self.reward_unreached = float(reward_unreached)
        self.reward_reached = float(reward_reached)
        self.terminate_on_success = bool(terminate_on_success)
        group = task_group or EXORL_GOAL_TASK_GROUP_TEMPLATE.format(domain=domain)
        super().__init__(
            name=name,
            reward_fn=self._eval_reward,
            domain=domain,
            task_group=group,
            is_goal_task=True,
            goal=self.goal_state,
            threshold=self.threshold_distance,
            eval_episode_length=eval_episode_length,
            reward_min=self.reward_unreached,
            reward_max=self.reward_reached,
            metadata={
                "goal_distance_threshold": float(self.threshold_distance),
                "score_bounds": (self.reward_unreached, self.reward_reached),
                "num_dims": int(self.goal_state.size),
            },
        )

    # -- reward -------------------------------------------------------------
    def _goal_view(self, observations: np.ndarray) -> np.ndarray:
        """Drop the appended physics features (not used for goal distance)."""
        if self.agent_obs_dim is not None and observations.shape[-1] > self.agent_obs_dim:
            return observations[..., : self.agent_obs_dim]
        if self.dims is not None and observations.shape[-1] > int(np.max(self.dims)):
            return observations[..., self.dims]
        return observations

    def distance(self, observations: Any) -> np.ndarray:
        """Std-normalized Euclidean distance to the goal state."""
        obs = _as_obs_matrix(observations)
        obs = self._goal_view(obs)
        goal = self.goal_state
        if self.dims is not None and goal.size > int(np.max(self.dims)):
            goal = goal[self.dims]
        if goal.size != obs.shape[-1]:
            goal = goal[: obs.shape[-1]]
        std = self.state_std
        mean = self.state_mean
        if std is not None and std.size > goal.size:
            std = std[: goal.size]
        if mean is not None and mean.size > goal.size:
            mean = mean[: goal.size]
        return normalized_goal_distance(obs, goal, mean=mean, std=std)

    def rewards(self, observations: Any) -> np.ndarray:
        return self._eval_reward(observations)

    def reward(self, observations: Any) -> np.ndarray:
        return self._eval_reward(observations)

    def _eval_reward(self, observations: Any) -> np.ndarray:
        d = self.distance(observations)
        reached = d < self.threshold_distance
        return np.where(reached, self.reward_reached, self.reward_unreached).astype(np.float64)

    def done(self, observations: Any) -> np.ndarray:
        """Done mask (``True`` once the goal has been reached)."""
        if not self.terminate_on_success:
            return np.zeros_like(self.distance(observations), dtype=bool)
        return self.distance(observations) < self.threshold_distance

    def __call__(self, observations: Any) -> np.ndarray:
        return self._eval_reward(observations)

    def success(self, observations: Any, threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold_distance if threshold is None else float(threshold)
        return self.distance(observations) < t

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["goal_distance_threshold"] = float(self.threshold_distance)
        info["dims"] = int(self.goal_state.size)
        return info


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_cheetah_velocity_tasks(
    thresholds: Sequence[float] = CHEETAH_VELOCITY_THRESHOLDS,
    velocity_dim: Optional[int] = None,
    agent_obs_dim: Optional[int] = None,
    eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> List[ExORLVelocityTask]:
    """The four ``exorl-cheetah-velocity`` tasks.

    ``cheetah-run`` (threshold 10), ``cheetah-run-backwards`` (10, backwards),
    ``cheetah-walk`` (1) and ``cheetah-walk-backwards`` (1, backwards).
    """
    thresholds = list(thresholds)
    run_threshold = float(thresholds[0]) if thresholds else 10.0
    walk_threshold = float(thresholds[1]) if len(thresholds) > 1 else 1.0
    specs = (
        (CHEETAH_RUN, run_threshold, False),
        (CHEETAH_RUN_BACKWARDS, run_threshold, True),
        (CHEETAH_WALK, walk_threshold, False),
        (CHEETAH_WALK_BACKWARDS, walk_threshold, True),
    )
    tasks: List[ExORLVelocityTask] = []
    for name, threshold, backwards in specs:
        tasks.append(
            ExORLVelocityTask(
                name=name,
                threshold=threshold,
                domain="cheetah",
                backwards=backwards,
                velocity_dim=velocity_dim,
                agent_obs_dim=agent_obs_dim,
                eval_episode_length=eval_episode_length,
                **kwargs,
            )
        )
    return tasks


def make_walker_velocity_tasks(
    thresholds: Sequence[float] = WALKER_VELOCITY_THRESHOLDS,
    velocity_dim: Optional[int] = None,
    agent_obs_dim: Optional[int] = None,
    eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> List[ExORLVelocityTask]:
    """The four ``exorl-walker-velocity`` tasks with thresholds 0.1/1/4/8."""
    thresholds = list(thresholds) or list(WALKER_VELOCITY_THRESHOLDS)
    tasks: List[ExORLVelocityTask] = []
    for threshold in thresholds:
        label = ("%g" % float(threshold))
        tasks.append(
            ExORLVelocityTask(
                name=f"walker-velocity-{label}",
                threshold=float(threshold),
                domain="walker",
                backwards=False,
                velocity_dim=velocity_dim,
                agent_obs_dim=agent_obs_dim,
                eval_episode_length=eval_episode_length,
                **kwargs,
            )
        )
    return tasks


def make_exorl_velocity_tasks(
    domain: str = "cheetah",
    **kwargs: Any,
) -> List[ExORLVelocityTask]:
    """Velocity tasks of a single domain (``"cheetah"`` or ``"walker"``)."""
    domain = str(domain).lower()
    if domain == "cheetah":
        return make_cheetah_velocity_tasks(**kwargs)
    if domain == "walker":
        return make_walker_velocity_tasks(**kwargs)
    raise ValueError(f"unknown ExORL domain {domain!r}; expected 'cheetah' or 'walker'")


def make_exorl_goal_tasks(
    domain: str = "cheetah",
    goal_states: Optional[Any] = None,
    num_goals: int = EXORL_NUM_GOALS,
    threshold: float = EXORL_GOAL_DISTANCE_THRESHOLD,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    agent_obs_dim: Optional[int] = None,
    eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> List[ExORLGoalTask]:
    """Goal-reaching tasks over ``num_goals`` fixed states of the dataset.

    Appendix C.2: "For goal-reaching tasks, we select five consistent goal
    states from the offline dataset."  The states must be supplied by the
    caller (for example ``ExORLDataset.select_goal_states(5)``); they are not
    re-sampled here so that the same goals are used for every evaluation
    episode and every agent.

    If ``goal_states`` is ``None`` the task set is empty (the harness is then
    expected to pass dataset goals).
    """
    domain = str(domain).lower()
    if goal_states is None:
        return []
    goals = _as_obs_matrix(goal_states)
    if goals.shape[0] > num_goals:
        goals = goals[:num_goals]
    tasks: List[ExORLGoalTask] = []
    for i in range(goals.shape[0]):
        tasks.append(
            ExORLGoalTask(
                name=f"{domain}-goal-{i}",
                goal=goals[i],
                domain=domain,
                threshold=threshold,
                state_mean=state_mean,
                state_std=state_std,
                agent_obs_dim=agent_obs_dim,
                eval_episode_length=eval_episode_length,
                **kwargs,
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
_GROUP_ALIASES: Dict[str, str] = {
    "all": "exorl-all",
    "exorl": "exorl-all",
    "velocity": "exorl-velocity",
    "goals": "exorl-goals",
    "goal": "exorl-goals",
    "cheetah": "exorl-cheetah",
    "walker": "exorl-walker",
    "exorl-cheetah-velocity": "exorl-cheetah-velocity",
    "exorl-cheetah-goals": "exorl-cheetah-goals",
    "exorl-walker-velocity": "exorl-walker-velocity",
    "exorl-walker-goals": "exorl-walker-goals",
    "exorl-all": "exorl-all",
    "exorl-velocity": "exorl-velocity",
    "exorl-goals": "exorl-goals",
}

#: Which task groups build up each aggregate group (for suite aggregation).
EXORL_AGGREGATE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "exorl-velocity": ("exorl-cheetah-velocity", "exorl-walker-velocity"),
    "exorl-goals": ("exorl-cheetah-goals", "exorl-walker-goals"),
    "exorl-all": (
        "exorl-cheetah-velocity",
        "exorl-cheetah-goals",
        "exorl-walker-velocity",
        "exorl-walker-goals",
    ),
    "exorl-cheetah": ("exorl-cheetah-velocity", "exorl-cheetah-goals"),
    "exorl-walker": ("exorl-walker-velocity", "exorl-walker-goals"),
}


def exorl_task_groups() -> Tuple[str, ...]:
    """Names of all accepted ExORL suite groups."""
    return EXORL_TASK_GROUPS


def _resolve_group(group: str) -> str:
    key = str(group).strip().lower()
    if key in _GROUP_ALIASES:
        return _GROUP_ALIASES[key]
    raise ValueError(
        f"unknown ExORL task group {group!r}; expected one of {sorted(EXORL_TASK_GROUPS)}"
    )


def make_exorl_task_suite(
    group: str = "exorl-all",
    domain: Optional[str] = None,
    goal_states: Optional[Dict[str, Any]] = None,
    velocity_dim: Optional[int] = None,
    agent_obs_dim: Optional[int] = None,
    agent_obs_dims: Optional[Dict[str, int]] = None,
    state_mean: Optional[Dict[str, np.ndarray]] = None,
    state_std: Optional[Dict[str, np.ndarray]] = None,
    eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> TaskSuite:
    """Build an ExORL :class:`~fre.envs.TaskSuite`.

    Parameters
    ----------
    group:
        One of ``exorl-all``, ``exorl-velocity``, ``exorl-goals``,
        ``exorl-cheetah``, ``exorl-walker``, ``exorl-cheetah-velocity``,
        ``exorl-cheetah-goals``, ``exorl-walker-velocity``,
        ``exorl-walker-goals`` (common aliases such as ``velocity`` are
        accepted).
    domain:
        Optional domain filter (``"cheetah"``/``"walker"``); narrows whatever
        ``group`` selects.
    goal_states:
        Mapping ``domain -> (num_goals, obs_dim)`` array of fixed goal states
        selected from the offline dataset (Appendix C.2).  Goal tasks are only
        created for domains present in this mapping.
    agent_obs_dim / agent_obs_dims:
        Dimension of the underlying environment observation (physics features
        are appended after it).  Used to locate the velocity feature and to
        strip the physics block for goal distance.
    state_mean / state_std:
        Per-dimension offline-dataset statistics used to normalize the goal
        distance (Appendix C.2).
    """
    resolved = _resolve_group(group)
    goal_states = goal_states or {}
    agent_obs_dims = dict(agent_obs_dims or {})
    state_mean = state_mean or {}
    state_std = state_std or {}

    def _domains_for(target: str) -> Tuple[str, ...]:
        if target.startswith("exorl-cheetah"):
            return ("cheetah",)
        if target.startswith("exorl-walker"):
            return ("walker",)
        if domain is not None:
            return (str(domain).lower(),)
        return EXORL_DOMAINS

    def _aim_dim(d: str) -> Optional[int]:
        if d in agent_obs_dims:
            return int(agent_obs_dims[d])
        return agent_obs_dim

    tasks: List[ExORLTask] = []
    targets = EXORL_AGGREGATE_GROUPS.get(resolved, (resolved,))
    for target in targets:
        for d in _domains_for(target):
            dim = _aim_dim(d)
            if target.endswith("velocity"):
                tasks.extend(
                    make_exorl_velocity_tasks(
                        domain=d,
                        velocity_dim=velocity_dim,
                        agent_obs_dim=dim,
                        eval_episode_length=eval_episode_length,
                        **kwargs,
                    )
                )
            elif target.endswith("goals"):
                tasks.extend(
                    make_exorl_goal_tasks(
                        domain=d,
                        goal_states=goal_states.get(d),
                        state_mean=state_mean.get(d),
                        state_std=state_std.get(d),
                        agent_obs_dim=dim,
                        eval_episode_length=eval_episode_length,
                        **kwargs,
                    )
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"unrecognized ExORL task group {target!r}")
    suite = build_suite(
        name=resolved,
        tasks=tasks,
        eval_episode_length=eval_episode_length,
        aggregate=resolved,
        domain=EXORL_DOMAIN,
        metadata={
            "group": resolved,
            "sub_groups": list(targets),
            "maximum_episode_length": int(eval_episode_length),
            "goal_distance_threshold": EXORL_GOAL_DISTANCE_THRESHOLD,
            "velocity_thresholds": {
                "cheetah": list(CHEETAH_VELOCITY_THRESHOLDS),
                "walker": list(WALKER_VELOCITY_THRESHOLDS),
            },
        },
    )
    return suite


def exorl_suite_for_group(group: str = "exorl-all", **kwargs: Any) -> TaskSuite:
    """Alias of :func:`make_exorl_task_suite` (suite-for-group convention)."""
    return make_exorl_task_suite(group=group, **kwargs)
