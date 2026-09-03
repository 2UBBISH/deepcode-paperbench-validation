"""D4RL Kitchen wrapper and seven-subtask reward utilities.

This module mirrors ``envs/antmaze_wrapper.py`` for the Franka Kitchen domain.
D4RL Kitchen observations are flattened as::

    [robot_qpos(9), robot_qvel(9),
     object_qpos(3), object_qvel(3),  ... 7 objects ...]

The object ordering follows the D4RL ``KitchenBase.TASK_ELEMENTS`` ordering:
``["bottom burner", "top burner", "light switch", "slide cabinet",
"hinge cabinet", "microwave", "kettle"]``.

The implementation is defensive: it lazily imports ``gym``/``d4rl`` so the
module can still be imported in CPU-only/test environments without MuJoCo.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch  # noqa: F401  (kept for policy-action conversion helpers)
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


ROBOT_QPOS_DIM = 9
ROBOT_QVEL_DIM = 9
OBJECT_QPOS_DIM = 3
OBJECT_QVEL_DIM = 3
OBJECT_START = ROBOT_QPOS_DIM + ROBOT_QVEL_DIM
OBJECT_STRIDE = OBJECT_QPOS_DIM + OBJECT_QVEL_DIM

KITCHEN_TASKS: Tuple[str, ...] = (
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
)

# Fallback goal object qpos values used when a D4RL environment cannot be
# constructed to extract ``_get_task_goal``. These are only a safe default;
# ``KitchenWrapper`` overwrites them from the environment when possible.
DEFAULT_KITCHEN_GOALS: Dict[str, np.ndarray] = {
    "bottom burner": np.array([-0.88, -0.01, -0.01], dtype=np.float64),
    "top burner": np.array([-0.88, -0.01, -0.01], dtype=np.float64),
    "light switch": np.array([0.05, -0.05, 0.0], dtype=np.float64),
    "slide cabinet": np.array([0.10, 0.0, 0.0], dtype=np.float64),
    "hinge cabinet": np.array([0.0, 0.0, 0.05], dtype=np.float64),
    "microwave": np.array([0.0, 0.0, 0.05], dtype=np.float64),
    "kettle": np.array([0.0, 0.0, 0.05], dtype=np.float64),
}


def _to_numpy(x: Union[np.ndarray, "torch.Tensor", Sequence, float]) -> np.ndarray:
    """Convert a torch tensor or sequence to a numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _policy_action(policy_fn: Callable, obs: np.ndarray) -> np.ndarray:
    """Call a policy and return a numpy action of shape ``(action_dim,)``.

    Supports policies that return numpy arrays, torch tensors, or
    ``(action, extra)`` tuples (common for some baseline policies).
    """
    obs = np.asarray(obs, dtype=np.float32)
    out = policy_fn(obs)
    if isinstance(out, tuple):
        out = out[0]
    action = _to_numpy(out)
    return np.squeeze(action).astype(np.float32)


class KitchenReward:
    """Sparse subtask-completion reward for one Kitchen task.

    The reward is ``0`` when the corresponding object qpos is within
    ``tolerance`` (Euclidean distance) of ``goal_qpos``, and ``-1`` otherwise.
    """

    def __init__(
        self,
        name: str,
        goal_qpos: np.ndarray,
        tolerance: float = 0.05,
        reward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        success_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.name = name
        self.goal_qpos = np.asarray(goal_qpos, dtype=np.float64)
        self.tolerance = float(tolerance)
        self._reward_fn = reward_fn
        self._success_fn = success_fn

    def _object_qpos(self, states: np.ndarray) -> np.ndarray:
        idx = KITCHEN_TASKS.index(self.name)
        start = OBJECT_START + idx * OBJECT_STRIDE
        return states[..., start : start + OBJECT_QPOS_DIM]

    def __call__(self, states: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        states_np = _to_numpy(states)
        if self._reward_fn is not None:
            return self._reward_fn(states_np)
        qpos = self._object_qpos(states_np)
        dist = np.linalg.norm(qpos - self.goal_qpos, axis=-1)
        reward = np.where(dist <= self.tolerance, 0.0, -1.0)
        return reward.astype(np.float32)

    def success(self, states: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        states_np = _to_numpy(states)
        if self._success_fn is not None:
            return self._success_fn(states_np)
        return (self.__call__(states_np) >= 0.0).astype(np.float32)


class KitchenWrapper:
    """Thin D4RL Kitchen environment wrapper.

    If a MuJoCo/D4RL environment is available, the wrapper lazily builds it
    and extracts task-goal object positions from the environment internals.
    The wrapper remains useful without an environment for state-only reward
    computation and task sampling.
    """

    def __init__(
        self,
        env_name: str = "kitchen-complete-v0",
        max_episode_steps: Optional[int] = None,
        goal_tolerance: float = 0.05,
    ) -> None:
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self._env = None
        self.goals: Dict[str, np.ndarray] = dict(DEFAULT_KITCHEN_GOALS)

        # Best-effort environment construction and goal extraction. This must
        # not raise on machines without MuJoCo/d4rl.
        try:
            import gym  # type: ignore

            env = gym.make(self.env_name)
            if self.max_episode_steps is not None:
                try:
                    env._max_episode_steps = int(self.max_episode_steps)
                except Exception:
                    pass
            self._env = env
            self._extract_goals_from_env(env)
        except Exception:
            self._env = None

    @property
    def env(self):
        return self._env

    def _extract_goals_from_env(self, env) -> None:
        """Populate goal qpos from D4RL Kitchen internals when possible."""
        if env is None:
            return
        for task in KITCHEN_TASKS:
            try:
                # D4RL Kitchen exposes ``_get_task_goal`` on KitchenBase.
                goal = env._get_task_goal(task)  # type: ignore[attr-defined]
                goal = np.asarray(goal, dtype=np.float64).reshape(-1)
                # Goal vectors may be quaternion joints (7D); object qpos is
                # the first three entries in the flattened observation.
                if goal.size >= OBJECT_QPOS_DIM:
                    self.goals[task] = goal[:OBJECT_QPOS_DIM].copy()
                else:
                    self.goals[task] = DEFAULT_KITCHEN_GOALS[task].copy()
            except Exception:
                # Try an alternative attribute name.
                try:
                    goal = env.get_task_goal(task)  # type: ignore[attr-defined]
                    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
                    if goal.size >= OBJECT_QPOS_DIM:
                        self.goals[task] = goal[:OBJECT_QPOS_DIM].copy()
                except Exception:
                    pass

    @property
    def observation_space(self):
        if self._env is not None:
            return self._env.observation_space
        import gym.spaces  # type: ignore

        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32)

    @property
    def action_space(self):
        if self._env is not None:
            return self._env.action_space
        import gym.spaces  # type: ignore

        return gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    @property
    def state_dim(self) -> int:
        space = self.observation_space
        shape = getattr(space, "shape", None)
        if shape is not None and len(shape) == 1:
            return int(shape[0])
        return 60

    @property
    def action_dim(self) -> int:
        space = self.action_space
        shape = getattr(space, "shape", None)
        if shape is not None and len(shape) == 1:
            return int(shape[0])
        return 9

    def reset(self) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("Kitchen environment is not available in this installation.")
        return np.asarray(self._env.reset(), dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._env is None:
            raise RuntimeError("Kitchen environment is not available in this installation.")
        obs, reward, done, info = self._env.step(action)
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        """Return the current flat observation (state)."""
        if self._env is None:
            raise RuntimeError("Kitchen environment is not available in this installation.")
        # D4RL Kitchen environments expose ``_get_obs``; fall back to sim.qpos.
        if hasattr(self._env, "_get_obs"):
            return np.asarray(self._env._get_obs(), dtype=np.float32)
        obs = np.asarray(self._env.sim.data.qpos[:ROBOT_QPOS_DIM], dtype=np.float32)
        return obs

    def evaluate_policy(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        task_name: str = "microwave",
        num_episodes: int = 20,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate a policy on one Kitchen subtask.

        Returns:
            Dict with ``task``, ``success_rate``, ``normalized_score``,
            ``mean_return``, and ``num_episodes``.
        """
        reward_fn = make_kitchen_task_reward(
            task_name,
            wrapper=self,
            tolerance=self.goal_tolerance,
        )

        returns = []
        successes = []
        for ep in range(int(num_episodes)):
            if seed is not None and self._env is not None:
                try:
                    self._env.seed(int(seed) + ep)
                except Exception:
                    pass
            obs = self.reset()
            done = False
            episode_return = 0.0
            episode_success = False
            steps = 0
            max_steps = self.max_episode_steps
            while not done:
                action = _policy_action(policy_fn, obs)
                obs, _, done, info = self.step(action)
                r = float(reward_fn(obs))
                episode_return += r
                episode_success = episode_success or bool(r >= 0.0)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            returns.append(episode_return)
            successes.append(1.0 if episode_success else 0.0)

        success_rate = float(np.mean(successes)) if successes else 0.0
        normalized_score = success_rate * 100.0
        return {
            "task": task_name,
            "success_rate": success_rate,
            "normalized_score": normalized_score,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "num_episodes": int(num_episodes),
        }

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass


def make_kitchen_task_reward(
    task_name: str,
    wrapper: Optional[KitchenWrapper] = None,
    goal_qpos: Optional[np.ndarray] = None,
    tolerance: float = 0.05,
) -> KitchenReward:
    """Create a state-only reward function for a Kitchen subtask.

    Args:
        task_name: One of ``KITCHEN_TASKS``.
        wrapper: Optional wrapper from which to source the goal object qpos.
        goal_qpos: Explicit goal object qpos. Overrides wrapper defaults.
        tolerance: Euclidean distance threshold for subtask completion.
    """
    if task_name not in KITCHEN_TASKS:
        raise ValueError(f"Unknown Kitchen task: {task_name}. Expected one of {KITCHEN_TASKS}")

    if goal_qpos is None and wrapper is not None:
        goal_qpos = wrapper.goals.get(task_name, DEFAULT_KITCHEN_GOALS[task_name])
    if goal_qpos is None:
        goal_qpos = DEFAULT_KITCHEN_GOALS[task_name]

    goal_qpos = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)[:OBJECT_QPOS_DIM]
    return KitchenReward(task_name, goal_qpos, tolerance=tolerance)


def evaluate_kitchen_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    task_name: str = "microwave",
    num_episodes: int = 20,
    seed: Optional[int] = None,
    env_name: str = "kitchen-complete-v0",
    max_episode_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Create a wrapper and evaluate ``policy_fn`` on one Kitchen task."""
    wrapper = KitchenWrapper(
        env_name=env_name,
        max_episode_steps=max_episode_steps,
    )
    try:
        return wrapper.evaluate_policy(
            policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
        )
    finally:
        wrapper.close()


def sample_task_reward_states(
    task_name: str,
    state_pool: Union[np.ndarray, "torch.Tensor", Sequence],
    num_examples: int = 32,
    seed: Optional[int] = None,
    goal_qpos: Optional[np.ndarray] = None,
    tolerance: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample exactly ``num_examples`` state-reward pairs for zero-shot encoding.

    This is the Kitchen analogue of the 32-example evaluation protocol.
    """
    pool = _to_numpy(state_pool)
    pool = pool.reshape(-1, pool.shape[-1])
    rng = np.random.RandomState(seed)
    idx = rng.choice(pool.shape[0], size=int(num_examples), replace=False)
    states = pool[idx].astype(np.float32)
    reward_fn = make_kitchen_task_reward(
        task_name,
        goal_qpos=goal_qpos,
        tolerance=tolerance,
    )
    rewards = reward_fn(states)
    return states, rewards


__all__ = [
    "KITCHEN_TASKS",
    "DEFAULT_KITCHEN_GOALS",
    "KitchenReward",
    "KitchenWrapper",
    "make_kitchen_task_reward",
    "evaluate_kitchen_policy",
    "sample_task_reward_states",
]
