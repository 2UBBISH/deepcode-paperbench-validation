"""AntMaze environment helpers.

This module provides lightweight wrappers and utility functions for the
D4RL AntMaze domain used by FRE.  The wrappers are intentionally defensive:
``gym`` and ``d4rl`` are imported lazily so this module can be imported even
when the optional MuJoCo/D4RL stack is not installed (for example, when only
the reward-sampler or VAE code is exercised).

The utilities are organised around the AntMaze observation convention used by
D4RL: the first two observation coordinates are the ant's x/y position on the
maze floor.  Goal-reaching rewards and visualisation helpers therefore operate
on those first two dimensions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

DEFAULT_ANTMAZE_ENV = "antmaze-large-diverse-v2"
ANTMAZE_XY_DIM = 2
DEFAULT_GOAL_THRESHOLD = 1.0


def _try_import_gym() -> Any:
    """Return the ``gym`` module or ``None`` when it is unavailable."""
    try:
        import gym  # type: ignore

        return gym
    except Exception:  # pragma: no cover - depends on optional dependency
        return None


def _try_import_d4rl() -> Any:
    """Return the ``d4rl`` module or ``None`` when it is unavailable."""
    try:
        import d4rl  # type: ignore

        return d4rl
    except Exception:  # pragma: no cover - depends on optional dependency
        return None


def make_antmaze_env(
    env_name: str = DEFAULT_ANTMAZE_ENV,
    **kwargs: Any,
) -> Any:
    """Create a live D4RL AntMaze environment.

    Parameters
    ----------
    env_name:
        D4RL AntMaze task id, e.g. ``antmaze-large-diverse-v2``.
    kwargs:
        Extra keyword arguments forwarded to ``gym.make``.

    Returns
    -------
    A Gym environment.  D4RL registers AntMaze tasks lazily, so this function
    imports ``d4rl`` before calling ``gym.make``.
    """
    gym = _try_import_gym()
    if gym is None:
        raise ImportError(
            "Gym is required for AntMaze environment creation. "
            "Install the `gym` and `d4rl` packages to use this function."
        )
    d4rl = _try_import_d4rl()
    if d4rl is None:
        raise ImportError(
            "D4RL is required for AntMaze environment creation. "
            "Install `d4rl` to use this function."
        )
    env = gym.make(env_name, **kwargs)
    return env


def get_antmaze_xy(
    states: Any,
    xy_dim: int = ANTMAZE_XY_DIM,
) -> np.ndarray:
    """Extract the planar x/y coordinates from AntMaze observations.

    D4RL AntMaze observations place the ant's x/y position in the first two
    coordinates.  The function accepts a single observation, a batch of
    observations, or a PyTorch tensor (which is detached and converted to
    NumPy).
    """
    arr = states
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:xy_dim].copy()
    return arr[..., :xy_dim]


def antmaze_goal_distance(
    states: Any,
    goals: Any,
    xy_only: bool = True,
    xy_dim: int = ANTMAZE_XY_DIM,
) -> np.ndarray:
    """Euclidean distance(s) between AntMaze states and goals.

    By default only the first ``xy_dim`` observation coordinates are used,
    which corresponds to x/y positions in the maze.
    """
    s = get_antmaze_xy(states, xy_dim=xy_dim) if xy_only else np.asarray(states, dtype=np.float32)
    g = get_antmaze_xy(goals, xy_dim=xy_dim) if xy_only else np.asarray(goals, dtype=np.float32)

    # Support a single goal broadcast against a batch of states.
    if s.ndim == 2 and g.ndim == 1:
        g = g[None, :]
    elif s.ndim == 1 and g.ndim == 2:
        s = s[None, :]
    diff = s - g
    return np.linalg.norm(diff, axis=-1)


def antmaze_sparse_reward(
    states: Any,
    goals: Any,
    threshold: float = DEFAULT_GOAL_THRESHOLD,
    xy_only: bool = True,
    xy_dim: int = ANTMAZE_XY_DIM,
) -> np.ndarray:
    """Return the sparse goal-reaching reward used by FRE for AntMaze.

    ``0.0`` when the state is within ``threshold`` of the goal, otherwise
    ``-1.0``.
    """
    dist = antmaze_goal_distance(states, goals, xy_only=xy_only, xy_dim=xy_dim)
    return np.where(dist < threshold, 0.0, -1.0).astype(np.float32)


def sample_antmaze_goal(
    states: Any,
    rng: Optional[np.random.Generator] = None,
    xy_only: bool = True,
    xy_dim: int = ANTMAZE_XY_DIM,
) -> np.ndarray:
    """Sample an AntMaze goal from a collection of dataset states.

    By default the sampled goal is represented only by the first ``xy_dim``
    coordinates.  If ``xy_only`` is false, the full state vector is returned.
    This is useful for goal-conditioned baselines and for evaluation tasks
    whose reward is defined as reaching a specific dataset state.
    """
    arr = states
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]

    if rng is None:
        rng = np.random.default_rng()
    idx = int(rng.integers(0, arr.shape[0]))
    full_state = arr[idx]
    if xy_only:
        return full_state[:xy_dim].copy()
    return full_state.copy()


def get_antmaze_bounds(
    env: Any = None,
    env_name: str = DEFAULT_ANTMAZE_ENV,
    fallback: Tuple[float, float, float, float] = (0.0, 24.0, 0.0, 24.0),
) -> Tuple[float, float, float, float]:
    """Return plausible maze drawing bounds as ``(xmin, xmax, ymin, ymax)``.

    D4RL AntMaze environments often store ``maze_size`` and the underlying
    maze layout.  When those attributes are not available a conservative
    fallback for the large maze is returned.
    """
    if env is not None:
        maze_size = getattr(env, "maze_size", None)
        if maze_size is not None:
            try:
                size = float(maze_size)
                return (0.0, size, 0.0, size)
            except Exception:
                pass

        # Some D4RL versions expose the generated maze layout.
        maze = getattr(env, "maze", None)
        if maze is not None:
            map_attr = getattr(maze, "map", None)
            if map_attr is not None:
                try:
                    h, w = map_attr.shape[:2]
                    return (0.0, float(w), 0.0, float(h))
                except Exception:
                    pass

    if "ultra" in env_name.lower():
        return (0.0, 56.0, 0.0, 56.0)
    if "large" in env_name.lower():
        return (0.0, 24.0, 0.0, 24.0)
    if "medium" in env_name.lower():
        return (0.0, 24.0, 0.0, 24.0)
    return fallback


class AntMazeEnv:
    """Thin, dependency-tolerant wrapper around a D4RL AntMaze environment.

    The wrapper is intentionally not a subclass of ``gym.Wrapper`` so that the
    module can be imported without Gym installed.  It supports optional
    observation normalisation (using dataset statistics), tracks episode
    trajectories, and normalises the multi-step interface differences between
    Gym 0.23 (``(obs, reward, done, info)``) and Gym 0.26+
    (``(obs, reward, terminated, truncated, info)``).

    Parameters
    ----------
    env_name:
        D4RL AntMaze environment id.
    state_mean, state_std:
        Optional dataset statistics used for observation normalisation.
    normalize_obs:
        Whether to normalise incoming observations.  FRE policies are trained
        on normalised states, so this is typically ``True`` when state
        statistics are supplied.
    max_episode_steps:
        Optional step limit; when provided, truncation is injected once the
        limit is reached.
    seed:
        Optional random seed.
    """

    def __init__(
        self,
        env_name: str = DEFAULT_ANTMAZE_ENV,
        state_mean: Optional[Sequence[float]] = None,
        state_std: Optional[Sequence[float]] = None,
        normalize_obs: bool = False,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._gym = _try_import_gym()
        if self._gym is None:
            raise ImportError("Gym is required to construct AntMazeEnv.")

        self.env_name = env_name
        self.env = make_antmaze_env(env_name, **kwargs)
        self.normalize_obs = normalize_obs

        self.state_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float32)
        self.state_std = None if state_std is None else np.asarray(state_std, dtype=np.float32)
        if self.normalize_obs and (self.state_mean is None or self.state_std is None):
            raise ValueError(
                "Observation normalisation requires both `state_mean` and `state_std`."
            )
        if self.state_std is not None:
            self.state_std = np.maximum(self.state_std, 1e-6)

        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self.trajectory: Dict[str, list] = {"states": [], "actions": [], "rewards": []}
        if seed is not None:
            self.seed(seed)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalize_obs(self, obs: Any) -> Any:
        if not self.normalize_obs or self.state_mean is None or self.state_std is None:
            return obs
        arr = np.asarray(obs, dtype=np.float32)
        return (arr - self.state_mean) / self.state_std

    def normalize_state(self, state: Any) -> np.ndarray:
        """Normalise a state (or batch) using the configured statistics."""
        arr = np.asarray(state, dtype=np.float32)
        if self.state_mean is None or self.state_std is None:
            return arr.copy()
        return (arr - self.state_mean) / self.state_std

    def unnormalize_state(self, state: Any) -> np.ndarray:
        """Un-normalise a state (or batch) using the configured statistics."""
        arr = np.asarray(state, dtype=np.float32)
        if self.state_mean is None or self.state_std is None:
            return arr.copy()
        return arr * self.state_std + self.state_mean

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def seed(self, seed: int = 0) -> None:
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if hasattr(self._gym, "utils") and hasattr(self._gym.utils, "seeding"):
            try:
                self._gym.utils.seeding.np_random(seed)
            except Exception:
                pass

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._step_count = 0
        self.trajectory = {"states": [], "actions": [], "rewards": []}
        result = self.env.reset(*args, **kwargs)
        if (
            isinstance(result, tuple)
            and len(result) >= 2
            and isinstance(result[1], dict)
        ):
            # Gym >= 0.26 returns (obs, info).
            obs = result[0]
        else:
            obs = result
        obs = np.asarray(obs, dtype=np.float32)
        self.trajectory["states"].append(obs.copy())
        return self._normalize_obs(obs)

    def step(self, action: Any) -> Any:
        action = np.asarray(action, dtype=np.float32)
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        self._step_count += 1
        if self.max_episode_steps is not None and self._step_count >= self.max_episode_steps:
            truncated = True

        obs = np.asarray(obs, dtype=np.float32)
        self.trajectory["states"].append(obs.copy())
        self.trajectory["actions"].append(action.copy())
        self.trajectory["rewards"].append(float(reward))

        info = dict(info or {})
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["timeout"] = bool(truncated) and not bool(terminated)

        # Return the Gym 0.26 five-tuple, which is the most informative and is
        # easy to unpack in evaluation loops.  The raw four-tuple is also
        # available through ``self.env.step`` when needed.
        return self._normalize_obs(obs), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # AntMaze-specific helpers
    # ------------------------------------------------------------------
    def get_xy(self, obs: Any) -> np.ndarray:
        """Extract x/y position from a (possibly normalised) observation.

        This method is primarily useful for visualisation.  If the wrapper is
        configured with dataset statistics, the observation is first
        un-normalised before extracting x/y coordinates.
        """
        arr = np.asarray(obs, dtype=np.float32)
        if self.normalize_obs and self.state_mean is not None and self.state_std is not None:
            arr = arr * self.state_std + self.state_mean
        return get_antmaze_xy(arr)

    def goal_distance(self, obs: Any, goal: Any) -> float:
        return float(antmaze_goal_distance(self.get_xy(obs), goal))

    def bounds(self) -> Tuple[float, float, float, float]:
        return get_antmaze_bounds(self.unwrapped, env_name=self.env_name)

    def trajectory_states(self) -> np.ndarray:
        if not self.trajectory["states"]:
            return np.empty((0,), dtype=np.float32)
        return np.asarray(self.trajectory["states"], dtype=np.float32)

    def trajectory_xys(self) -> np.ndarray:
        states = self.trajectory_states()
        if states.size == 0:
            return states.reshape(0, 2)
        if self.normalize_obs and self.state_mean is not None and self.state_std is not None:
            states = states * self.state_std + self.state_mean
        return get_antmaze_xy(states)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped D4RL environment."""
        return getattr(self.env, name)


__all__ = [
    "AntMazeEnv",
    "DEFAULT_ANTMAZE_ENV",
    "DEFAULT_GOAL_THRESHOLD",
    "antmaze_goal_distance",
    "antmaze_sparse_reward",
    "get_antmaze_bounds",
    "get_antmaze_xy",
    "make_antmaze_env",
    "sample_antmaze_goal",
]
