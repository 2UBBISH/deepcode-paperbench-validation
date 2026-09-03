"""Kitchen environment utilities for the FRE reproduction.

This module provides a dependency-tolerant wrapper around D4RL Kitchen
environments together with helpers for computing the seven standard Kitchen
subtask reward functions used throughout the paper.  All heavy D4RL imports
are deferred until a live environment is actually requested so the rest of
the repository can be imported without a MuJoCo/D4RL installation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KITCHEN_ENV: str = "kitchen-complete-v0"
DEFAULT_KITCHEN_TASK: str = "kitchen-complete-v0"

# Seven standard D4RL Kitchen subtask names.  These are the downstream tasks
# used in the paper's Kitchen domain.
KITCHEN_TASK_NAMES: Tuple[str, ...] = (
    "bottom_burner",
    "light_switch",
    "slide_cabinet",
    "hinge_cabinet",
    "microwave",
    "kettle",
    "top_burner",
)

# Indices into a 60-dimensional D4RL Kitchen observation.  The first 30
# dimensions are joint/object positions; the remaining 30 are velocities.
# These indices follow the standard D4RL ``OBS_ELEMENT_INDICES`` mapping and
# are used to score the seven subtasks from raw observations.
KITCHEN_SUBTASK_OBS_INDICES: Dict[str, Sequence[int]] = {
    "bottom_burner": (11, 12),
    "light_switch": (13, 14),
    "slide_cabinet": (15,),
    "hinge_cabinet": (16, 17),
    "microwave": (18,),
    "kettle": (19, 20, 21),
    "top_burner": (22, 23),
}

# Target positions for each subtask in the same coordinate frame.
KITCHEN_SUBTASK_GOALS: Dict[str, np.ndarray] = {
    "bottom_burner": np.array([-0.88, -0.01], dtype=np.float32),
    "light_switch": np.array([-0.69, -0.05], dtype=np.float32),
    "slide_cabinet": np.array([0.37], dtype=np.float32),
    "hinge_cabinet": np.array([0.0, 1.45], dtype=np.float32),
    "microwave": np.array([-0.75], dtype=np.float32),
    "kettle": np.array([-0.23, 0.75, 1.62], dtype=np.float32),
    "top_burner": np.array([-0.88, -0.01], dtype=np.float32),
}

# Aliases that evaluation code may use.
KITCHEN_SUBTASKS: Tuple[str, ...] = KITCHEN_TASK_NAMES
KITCHEN_TASKS: Tuple[str, ...] = KITCHEN_TASK_NAMES


def _as_numpy(x: Any, name: str = "value") -> np.ndarray:
    """Convert an array-like object or torch tensor to a float32 NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _normalize_obs(obs: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return obs
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = np.where(std < 1e-6, 1.0, std)
    return (obs - mean) / std_safe


def _unnormalize_obs(obs: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return obs
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return obs * std + mean


class KitchenEnv:
    """Thin wrapper around a D4RL Kitchen environment.

    Parameters
    ----------
    env_name:
        D4RL Kitchen task name, e.g. ``kitchen-complete-v0``.
    state_mean, state_std:
        Optional observation statistics used when ``normalize_obs`` is True.
    normalize_obs:
        Whether to normalize observations returned by ``reset``/``step``.
    max_episode_steps:
        Episode truncation horizon.  D4RL Kitchen episodes are typically 280
        steps; this wrapper only uses the value for bookkeeping and truncation
        semantics.
    """

    def __init__(
        self,
        env_name: str = DEFAULT_KITCHEN_ENV,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        normalize_obs: bool = False,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.env_name = env_name
        self.state_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float32)
        self.state_std = None if state_std is None else np.asarray(state_std, dtype=np.float32)
        self.normalize_obs = bool(normalize_obs)
        self.max_episode_steps = max_episode_steps

        try:
            import gym  # type: ignore
            import d4rl  # noqa: F401  (registers kitchen environments)
        except Exception as exc:  # pragma: no cover - depends on optional stack
            raise ImportError(
                "Gym and D4RL are required to create a live Kitchen environment. "
                f"Original error: {exc}"
            )

        self._gym = gym
        self.env = gym.make(env_name, **kwargs)
        if seed is not None:
            self.seed(seed)

        self._raw_obs: Optional[np.ndarray] = None
        self._episode_step: int = 0
        self._trajectory_obs: list = []
        self._trajectory_rewards: list = []

    # ------------------------------------------------------------------
    # Gym API compatibility
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> Any:
        if seed is not None:
            np.random.seed(seed)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def reset(self) -> np.ndarray:
        reset_result = self.env.reset()
        # Gym 0.26+ returns (obs, info); older versions return obs only.
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        obs = _as_numpy(obs, "observation")
        self._raw_obs = obs
        self._episode_step = 0
        self._trajectory_obs = [obs]
        self._trajectory_rewards = []
        return self.normalize_state(obs) if self.normalize_obs else obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = _as_numpy(action, "action")
        step_result = self.env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
            info = dict(info or {})
            info["TimeLimit.truncated"] = bool(truncated)
        else:
            obs, reward, done, info = step_result
            info = dict(info or {})

        obs = _as_numpy(obs, "observation")
        reward = float(reward)
        self._raw_obs = obs
        self._episode_step += 1
        if self.max_episode_steps is not None and self._episode_step >= self.max_episode_steps:
            done = True
        self._trajectory_obs.append(obs)
        self._trajectory_rewards.append(reward)
        return (self.normalize_state(obs) if self.normalize_obs else obs), reward, bool(done), info

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            logger.debug("Kitchen environment close failed", exc_info=True)

    # ------------------------------------------------------------------
    # State normalization
    # ------------------------------------------------------------------
    def normalize_state(self, state: Any) -> np.ndarray:
        return _normalize_obs(_as_numpy(state, "state"), self.state_mean, self.state_std)

    def unnormalize_state(self, state: Any) -> np.ndarray:
        return _unnormalize_obs(_as_numpy(state, "state"), self.state_mean, self.state_std)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def trajectory_states(self) -> np.ndarray:
        if not self._trajectory_obs:
            return np.zeros((0, self.observation_space.shape[0]), dtype=np.float32)
        return np.stack(self._trajectory_obs, axis=0)

    @property
    def trajectory_rewards(self) -> np.ndarray:
        return np.asarray(self._trajectory_rewards, dtype=np.float32)

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space


def make_kitchen_env(
    env_name: str = DEFAULT_KITCHEN_ENV,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    normalize_obs: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Create a live D4RL Kitchen environment.

    Returns a :class:`KitchenEnv` wrapper when normalization statistics are
    supplied; otherwise returns the raw Gym environment for maximum
    compatibility with external roll-out code.
    """
    if state_mean is not None or state_std is not None or normalize_obs:
        return KitchenEnv(
            env_name=env_name,
            state_mean=state_mean,
            state_std=state_std,
            normalize_obs=normalize_obs,
            max_episode_steps=max_episode_steps,
            seed=seed,
            **kwargs,
        )

    try:
        import gym  # type: ignore
        import d4rl  # noqa: F401
    except Exception as exc:
        raise ImportError(f"Gym and D4RL are required for Kitchen env creation. Original error: {exc}")
    env = gym.make(env_name, **kwargs)
    if seed is not None:
        env.seed(seed)
    return env


# ----------------------------------------------------------------------
# Kitchen reward/geometry helpers
# ----------------------------------------------------------------------
def _resolve_subtask_indices(task_name: str) -> np.ndarray:
    key = task_name.lower().replace("-", "_").replace(" ", "_")
    if key in KITCHEN_SUBTASK_OBS_INDICES:
        indices = KITCHEN_SUBTASK_OBS_INDICES[key]
    else:
        # Tolerate common D4RL spellings with spaces and dashes.
        for known, idx in KITCHEN_SUBTASK_OBS_INDICES.items():
            if known.replace("_", " ") in task_name.lower() or known.replace("_", "-") in task_name.lower():
                indices = idx
                break
        else:
            raise KeyError(
                f"Unknown Kitchen subtask '{task_name}'. Valid tasks: {KITCHEN_TASK_NAMES}"
            )
    return np.asarray(indices, dtype=np.int64)


def _resolve_subtask_goal(task_name: str) -> np.ndarray:
    key = task_name.lower().replace("-", "_").replace(" ", "_")
    if key in KITCHEN_SUBTASK_GOALS:
        return KITCHEN_SUBTASK_GOALS[key]
    for known, goal in KITCHEN_SUBTASK_GOALS.items():
        if known.replace("_", " ") in task_name.lower() or known.replace("_", "-") in task_name.lower():
            return goal
    raise KeyError(f"Unknown Kitchen subtask '{task_name}'. Valid tasks: {KITCHEN_TASK_NAMES}")


def kitchen_subtask_distance(states: Any, task_name: str) -> np.ndarray:
    """Return Euclidean distance between states and a subtask target.

    ``states`` may have shape ``(obs_dim,)`` or ``(batch, obs_dim)``.  The
    returned array has shape ``(batch,)`` (or scalar-compatible ``(1,)`` for a
    single state).
    """
    states = _as_numpy(states, "states")
    if states.ndim == 1:
        states = states[None, :]
    indices = _resolve_subtask_indices(task_name)
    goal = _resolve_subtask_goal(task_name)
    if int(np.max(indices)) >= states.shape[1]:
        raise ValueError(
            f"Observation dimension {states.shape[1]} is too small for Kitchen subtask indices {indices}"
        )
    diff = states[:, indices] - goal[None, :]
    return np.linalg.norm(diff, axis=1)


def kitchen_subtask_achieved(states: Any, task_name: str, tol: float = 0.3) -> np.ndarray:
    """Return a boolean array indicating subtask completion."""
    return kitchen_subtask_distance(states, task_name) <= tol


def kitchen_subtask_reward(
    states: Any,
    task_name: str,
    tol: float = 0.3,
    dense: bool = False,
    reward_scale: float = 1.0,
) -> np.ndarray:
    """Compute a scalar reward for one of the seven Kitchen subtasks.

    By default the reward is sparse: 0.0 when the subtask target is reached
    and -1.0 otherwise.  If ``dense`` is True the negative distance to the
    target is returned instead.
    """
    dist = kitchen_subtask_distance(states, task_name)
    if dense:
        return (-dist * reward_scale).astype(np.float32)
    achieved = dist <= tol
    return np.where(achieved, 0.0, -1.0).astype(np.float32)


def kitchen_goal_distance(states: Any, goals: Any, use_velocity: bool = False) -> np.ndarray:
    """Euclidean distance between states and goal states.

    When ``use_velocity`` is False (the default) only the first half of the
    observation (positions) is used, which is the standard goal metric for
    D4RL Kitchen.
    """
    states = _as_numpy(states, "states")
    goals = _as_numpy(goals, "goals")
    if states.ndim == 1:
        states = states[None, :]
    if goals.ndim == 1:
        goals = goals[None, :]
    obs_dim = states.shape[1]
    if not use_velocity and obs_dim % 2 == 0:
        states = states[:, : obs_dim // 2]
        goals = goals[:, : obs_dim // 2]
    diff = states - goals
    return np.linalg.norm(diff, axis=1)


def kitchen_sparse_reward(
    states: Any,
    goals: Any,
    threshold: float = 0.3,
    use_velocity: bool = False,
) -> np.ndarray:
    """Sparse Kitchen goal-reaching reward: 0.0 within threshold, else -1.0."""
    dist = kitchen_goal_distance(states, goals, use_velocity=use_velocity)
    return np.where(dist <= threshold, 0.0, -1.0).astype(np.float32)


def sample_kitchen_goal(states: Any, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample a goal observation from a collection of dataset states."""
    states = _as_numpy(states, "states")
    if states.ndim == 1:
        states = states[None, :]
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, states.shape[0])
    return states[idx].astype(np.float32)


__all__ = [
    "DEFAULT_KITCHEN_ENV",
    "DEFAULT_KITCHEN_TASK",
    "KITCHEN_TASK_NAMES",
    "KITCHEN_SUBTASKS",
    "KITCHEN_TASKS",
    "KITCHEN_SUBTASK_OBS_INDICES",
    "KITCHEN_SUBTASK_GOALS",
    "KitchenEnv",
    "make_kitchen_env",
    "kitchen_subtask_distance",
    "kitchen_subtask_achieved",
    "kitchen_subtask_reward",
    "kitchen_goal_distance",
    "kitchen_sparse_reward",
    "sample_kitchen_goal",
]
