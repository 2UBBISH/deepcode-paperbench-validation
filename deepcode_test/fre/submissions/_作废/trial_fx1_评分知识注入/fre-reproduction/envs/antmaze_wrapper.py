"""AntMaze-large-diverse-v2 environment wrapper and evaluation task rewards.

This module keeps the environment-specific details (observation layout,
goal coordinates, task reward functions, and score normalization) out of
the training/evaluation scripts.  It is intentionally written against a
plain ``gym`` interface plus a ``d4rl`` dataset dependency only for
dataset loading; policy evaluation is performed directly with
``gym.make``.

The FRE paper uses six AntMaze evaluation tasks::

    ant-goal-reaching
    ant-directional
    ant-random-simplex
    ant-path-loop
    ant-path-edges
    ant-path-center

Reward functions are vectorized over a leading state dimension and can be
called with either NumPy arrays or PyTorch tensors.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch  # optional; only used for tensor->numpy conversion
except Exception:  # pragma: no cover - torch is a core dependency anyway
    torch = None


ArrayLike = Union[np.ndarray, "torch.Tensor", Sequence[float]]


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
ANTMAZE_TASKS = [
    "ant-goal-reaching",
    "ant-directional",
    "ant-random-simplex",
    "ant-path-loop",
    "ant-path-edges",
    "ant-path-center",
]

# Default goal used by D4RL antmaze-large-diverse-v2.  If the environment
# exposes its own target coordinates we prefer those, but this is a safe
# fallback for offline reward generation.
DEFAULT_ANTMAZE_GOAL = (18.0, 12.0)
DEFAULT_ANTMAZE_START = (0.0, 0.0)


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert an array-like object to a NumPy float array."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    return arr


def _xy(states: ArrayLike) -> np.ndarray:
    """Extract the planar ``(x, y)`` position from AntMaze observations.

    D4RL AntMaze observations are ``qpos (15) + qvel (14)``; the first two
    qpos entries are the root x/y coordinates.
    """
    s = _to_numpy(states)
    return s[..., :2]


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance from ``points`` (N, 2) to segment ``a-b``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    t = np.clip(np.dot(points - a, ab) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    diff = points - proj
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _distance_to_polyline(points: np.ndarray, waypoints: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Distance from ``points`` (N, 2) to the nearest point on a polyline."""
    waypoints = np.asarray(waypoints, dtype=np.float32)
    dists = np.full(points.shape[0], np.inf, dtype=np.float32)
    for i in range(len(waypoints) - 1):
        d = _distance_to_segment(points, waypoints[i], waypoints[i + 1])
        dists = np.minimum(dists, d)
    return dists


class AntMazeReward:
    """A callable reward function with task metadata.

    Parameters
    ----------
    name: Task name (one of :data:`ANTMAZE_TASKS`).
    reward_fn: Vectorized reward function ``states -> rewards``.
    success_fn: Optional ``states -> bool`` success indicator used to
        compute the normalized 0-100 evaluation score.
    """

    def __init__(
        self,
        name: str,
        reward_fn: Callable[[ArrayLike], np.ndarray],
        success_fn: Optional[Callable[[ArrayLike], np.ndarray]] = None,
        goal: Optional[Tuple[float, float]] = None,
        scale: float = 1.0,
    ) -> None:
        self.name = name
        self.reward_fn = reward_fn
        self.success_fn = success_fn
        self.goal = goal
        self.scale = scale

    def __call__(self, states: ArrayLike) -> np.ndarray:
        return self.reward_fn(states)


def _sparse_goal_reward(states: ArrayLike, goal: Tuple[float, float], epsilon: float = 1.0) -> np.ndarray:
    """Return 0 within ``epsilon`` of ``goal`` and -1 elsewhere."""
    xy = _xy(states)
    goal_arr = np.asarray(goal, dtype=np.float32)
    dist = np.linalg.norm(xy - goal_arr, axis=-1)
    return np.where(dist <= epsilon, 0.0, -1.0).astype(np.float32)


def _success_within(states: ArrayLike, goal: Tuple[float, float], epsilon: float = 1.0) -> np.ndarray:
    xy = _xy(states)
    goal_arr = np.asarray(goal, dtype=np.float32)
    dist = np.linalg.norm(xy - goal_arr, axis=-1)
    return dist <= epsilon


def _direction_reward(states: ArrayLike, start: Tuple[float, float], goal: Tuple[float, float]) -> np.ndarray:
    """Reward based on progress along the vector from ``start`` to ``goal``.

    The raw projection is divided by a characteristic maze scale so rewards
    stay in a reasonable range before clipping.
    """
    xy = _xy(states)
    start_arr = np.asarray(start, dtype=np.float32)
    goal_arr = np.asarray(goal, dtype=np.float32)
    direction = goal_arr - start_arr
    denom = float(np.linalg.norm(direction)) + 1e-6
    direction = direction / denom
    projection = np.dot(xy - start_arr, direction)
    return np.clip(projection / 18.0, -1.0, 1.0).astype(np.float32)


def _path_reward(states: ArrayLike, waypoints: Sequence[Tuple[float, float]], scale: float = 5.0) -> np.ndarray:
    """Smooth path-following reward: ``exp(-distance / scale)`` in ``[0, 1]``."""
    xy = _xy(states)
    dist = _distance_to_polyline(xy, waypoints)
    return np.exp(-dist / scale).astype(np.float32)


def _random_simplex_reward(states: ArrayLike, seed: int = 0) -> np.ndarray:
    """A fixed random linear reward over position/angle features.

    We use a Dirichlet weight vector over a few interpretable AntMaze
    features, which gives the "random simplex" task in the paper.
    """
    s = _to_numpy(states)
    x = s[..., 0]
    y = s[..., 1]
    # qpos root z is often index 2; include some joint features to make the
    # task more diverse than a pure xy linear function.
    features = [
        np.tanh(x / 10.0),
        np.tanh(y / 10.0),
        np.sin(s[..., 2] if s.shape[-1] > 2 else x),
        np.cos(x / 5.0) * np.cos(y / 5.0),
    ]
    rng = np.random.RandomState(seed)
    w = rng.dirichlet(np.ones(len(features))).astype(np.float32)
    reward = np.zeros_like(x, dtype=np.float32)
    for wi, feat in zip(w, features):
        reward = reward + wi * feat
    return np.clip(2.0 * reward - 1.0, -1.0, 1.0).astype(np.float32)


# Polyline paths used for path-following tasks.  Coordinates are expressed
# in the AntMaze-large xy frame and were chosen to resemble the qualitative
# paths in the paper (center corridor, maze edges, and a closed loop).
_PATH_CENTER = [(0.0, 0.0), (6.0, 4.0), (12.0, 8.0), (18.0, 12.0)]
_PATH_EDGES = [(0.0, 0.0), (0.0, 12.0), (18.0, 12.0), (18.0, 0.0), (0.0, 0.0)]
_PATH_LOOP = [(4.0, 4.0), (16.0, 4.0), (16.0, 16.0), (4.0, 16.0), (4.0, 4.0)]


def make_antmaze_task_reward(task_name: str, goal: Optional[Tuple[float, float]] = None) -> AntMazeReward:
    """Create a vectorized reward function for an AntMaze evaluation task.

    Parameters
    ----------
    task_name: One of the strings in :data:`ANTMAZE_TASKS`.
    goal: Optional override of the default goal position.

    Returns
    -------
    An :class:`AntMazeReward` callable that accepts batched states and
    returns a reward vector.  The object also carries a ``success_fn`` for
    computing normalized 0-100 evaluation scores.
    """
    name = task_name.lower()
    if name not in ANTMAZE_TASKS:
        raise ValueError(f"Unknown AntMaze task {task_name!r}; expected one of {ANTMAZE_TASKS}")

    if name == "ant-goal-reaching":
        g = tuple(goal or DEFAULT_ANTMAZE_GOAL)
        return AntMazeReward(
            name=name,
            reward_fn=lambda s: _sparse_goal_reward(s, g),
            success_fn=lambda s: _success_within(s, g),
            goal=g,
            scale=1.0,
        )

    if name == "ant-directional":
        start = DEFAULT_ANTMAZE_START
        g = tuple(goal or DEFAULT_ANTMAZE_GOAL)
        return AntMazeReward(
            name=name,
            reward_fn=lambda s: _direction_reward(s, start, g),
            success_fn=None,
            goal=g,
            scale=1.0,
        )

    if name == "ant-random-simplex":
        return AntMazeReward(
            name=name,
            reward_fn=_random_simplex_reward,
            success_fn=None,
            goal=None,
            scale=1.0,
        )

    if name == "ant-path-loop":
        return AntMazeReward(
            name=name,
            reward_fn=lambda s: _path_reward(s, _PATH_LOOP),
            success_fn=lambda s: _distance_to_polyline(_xy(s), _PATH_LOOP) < 0.75,
            goal=None,
            scale=5.0,
        )

    if name == "ant-path-edges":
        return AntMazeReward(
            name=name,
            reward_fn=lambda s: _path_reward(s, _PATH_EDGES),
            success_fn=lambda s: _distance_to_polyline(_xy(s), _PATH_EDGES) < 0.75,
            goal=None,
            scale=5.0,
        )

    if name == "ant-path-center":
        return AntMazeReward(
            name=name,
            reward_fn=lambda s: _path_reward(s, _PATH_CENTER),
            success_fn=lambda s: _distance_to_polyline(_xy(s), _PATH_CENTER) < 0.75,
            goal=None,
            scale=5.0,
        )

    # Should never reach here; kept for type checkers.
    raise ValueError(f"Unhandled AntMaze task {task_name!r}")


class AntMazeWrapper:
    """Thin wrapper around a D4RL AntMaze gym environment.

    The wrapper exposes the observation/action shapes and a ``get_xy``
    helper, and it can evaluate a policy under one of the paper's task
    reward functions.
    """

    def __init__(self, env_name: str = "antmaze-large-diverse-v2", max_episode_steps: Optional[int] = None):
        import gym  # local import so dataset-only usage does not require gym

        self.env_name = env_name
        self._gym = gym
        self.env = gym.make(env_name)
        if max_episode_steps is not None:
            try:
                self.env._max_episode_steps = max_episode_steps
            except Exception:
                pass
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_dim = int(self.observation_space.shape[0])
        self.action_dim = int(self.action_space.shape[0])

        # Prefer the environment's own goal coordinates if available.
        self.goal = self._extract_goal()
        if self.goal is None:
            self.goal = DEFAULT_ANTMAZE_GOAL

    def _extract_goal(self) -> Optional[Tuple[float, float]]:
        env = self.env
        for attr in ("target_goal", "target_pos", "goal", "target_xy"):
            try:
                value = getattr(env, attr, None)
                if value is None:
                    value = getattr(getattr(env, "unwrapped", None), attr, None)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1])
            except Exception:
                continue
        try:
            if hasattr(env, "get_target_xy"):
                arr = np.asarray(env.get_target_xy(), dtype=np.float32).reshape(-1)
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1])
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Environment interaction
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        return np.asarray(self.env.reset(), dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state, reward, done, info = self.env.step(action)
        return (
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            bool(done),
            info,
        )

    def get_state(self) -> np.ndarray:
        """Return the current observation."""
        try:
            obs = self.env._get_obs()
        except Exception:
            obs = self.env.state_vector()
        return np.asarray(obs, dtype=np.float32)

    def get_xy(self) -> np.ndarray:
        """Return the current root x/y position."""
        try:
            return np.asarray(self.env.get_xy(), dtype=np.float32)
        except Exception:
            obs = self.get_state()
            return _xy(obs)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_policy(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        task_name: str = "ant-goal-reaching",
        num_episodes: int = 20,
        seed: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        render: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a policy under an AntMaze task reward.

        Parameters
        ----------
        policy_fn: Maps a current observation to an action.
        task_name: Task identifier from :data:`ANTMAZE_TASKS`.
        num_episodes: Number of rollout episodes.
        seed: Optional environment seed.
        max_episode_steps: Rollout truncation length.

        Returns
        -------
        Dictionary with ``mean_return``, ``std_return``, ``normalized_score``,
        and (for success-based tasks) ``success_rate``.
        """
        task = make_antmaze_task_reward(task_name)
        max_steps = max_episode_steps or getattr(self.env, "_max_episode_steps", 1000)

        if seed is not None:
            try:
                self.env.seed(int(seed))
            except Exception:
                pass

        episode_returns: list = []
        successes: list = []

        for _ in range(num_episodes):
            obs = self.reset()
            done = False
            truncated = False
            ep_return = 0.0
            step_count = 0
            ep_success = False

            while not (done or truncated) and step_count < max_steps:
                action = np.asarray(policy_fn(obs), dtype=np.float32)
                next_obs, _, done, info = self.step(action)
                r = float(task(obs[None, :])[0])
                ep_return += r

                if task.success_fn is not None:
                    try:
                        ep_success = bool(task.success_fn(next_obs[None, :])[0]) or ep_success
                    except Exception:
                        pass

                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass

                obs = next_obs
                step_count += 1

                # Gym versions sometimes signal truncation via TimeLimit info.
                if info is not None and bool(info.get("TimeLimit.truncated", False)):
                    truncated = True

            episode_returns.append(ep_return)
            successes.append(float(ep_success))

        mean_return = float(np.mean(episode_returns))
        std_return = float(np.std(episode_returns))
        success_rate = float(np.mean(successes)) if successes else 0.0

        if task.success_fn is not None:
            normalized_score = success_rate * 100.0
        else:
            # For dense directional/simplex rewards, map average clipped
            # reward to 0-100 using a scale derived from the reward range.
            normalized_score = float(np.clip((mean_return + 1.0) / 2.0, 0.0, 1.0) * 100.0)

        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "success_rate": success_rate,
            "normalized_score": normalized_score,
            "task": task_name,
            "episodes": num_episodes,
        }


def evaluate_antmaze_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    task_name: str = "ant-goal-reaching",
    num_episodes: int = 20,
    seed: Optional[int] = None,
    env_name: str = "antmaze-large-diverse-v2",
) -> Dict[str, float]:
    """Convenience function: create an :class:`AntMazeWrapper` and evaluate.

    This is the intended entry point for ``scripts/eval_zero_shot.py`` and
    ``scripts/eval_baselines.py``.
    """
    wrapper = AntMazeWrapper(env_name=env_name)
    try:
        return wrapper.evaluate_policy(
            policy_fn=policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
        )
    finally:
        wrapper.close()


def sample_task_reward_states(
    task_name: str,
    state_pool: ArrayLike,
    num_examples: int = 32,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample state-reward pairs for zero-shot task encoding.

    This mirrors the FRE evaluation protocol: exactly ``num_examples``
    states are drawn from the offline state pool and labeled by the task
    reward function.
    """
    rng = np.random.RandomState(seed)
    states = _to_numpy(state_pool)
    indices = rng.randint(0, len(states), size=num_examples)
    sampled_states = states[indices]
    task = make_antmaze_task_reward(task_name)
    rewards = task(sampled_states)
    return sampled_states, rewards


__all__ = [
    "ANTMAZE_TASKS",
    "AntMazeWrapper",
    "AntMazeReward",
    "make_antmaze_task_reward",
    "evaluate_antmaze_policy",
    "sample_task_reward_states",
]
