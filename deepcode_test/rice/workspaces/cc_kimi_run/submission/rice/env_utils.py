"""Environment utilities and wrappers for RICE."""
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class SparseRewardWrapper(gym.Wrapper):
    """Convert a dense MuJoCo reward into a sparse reward.

    Following Mazoure et al. (2019), the agent only receives reward when the
    x-position exceeds a threshold.
    """

    def __init__(
        self,
        env: gym.Env,
        threshold: float = 0.6,
    ) -> None:
        super().__init__(env)
        self.threshold = threshold
        self._initial_x: Optional[float] = None

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._initial_x = self._get_x(obs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = self._get_x(obs)
        sparse_reward = max(0.0, x - self._initial_x - self.threshold) if self._initial_x is not None else 0.0
        # Override the reward reported in info for reference.
        info["dense_reward"] = reward
        return obs, sparse_reward, terminated, truncated, info

    def _get_x(self, obs: np.ndarray) -> float:
        # In MuJoCo locomotion tasks, the x position is the first observation coordinate.
        return float(obs[0])


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Normalize observations by running mean and standard deviation."""

    def __init__(self, env: gym.Env, eps: float = 1e-8) -> None:
        super().__init__(env)
        self.eps = eps
        self.running_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.running_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = eps

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.running_mean) / np.sqrt(self.running_var + self.eps)

    def update_stats(self, obs: np.ndarray) -> None:
        """Update running statistics with a batch of observations."""
        obs = np.asarray(obs)
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = obs.shape[0]
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        self.running_mean += delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.running_var = m2 / total_count
        self.count = total_count


def make_env(
    env_id: str,
    seed: Optional[int] = None,
    sparse: bool = False,
    sparse_threshold: Optional[float] = None,
    normalize_obs: bool = False,
) -> gym.Env:
    """Create a wrapped gymnasium environment."""
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    if sparse:
        threshold = sparse_threshold or 0.6
        if "HalfCheetah" in env_id:
            threshold = sparse_threshold or 5.0
        env = SparseRewardWrapper(env, threshold=threshold)
    if normalize_obs:
        env = NormalizeObservationWrapper(env)
    return env


class StateResetWrapper(gym.Wrapper):
    """Wrapper that allows resetting the environment to an arbitrary state.

    This is used by RICE to restart episodes from identified critical states.
    The implementation stores the most recent internal simulator state so that
    it can be restored on demand. For environments that do not expose the full
    simulator state (e.g. real-world simulators), a custom reset function should
    be provided.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._stored_state: Optional[Any] = None

    def store_current_state(self) -> Any:
        """Store the current environment state for later restoration."""
        # Gymnasium MuJoCo environments expose env.unwrapped.state().
        if hasattr(self.env.unwrapped, "state"):
            self._stored_state = self.env.unwrapped.state().copy()
        elif hasattr(self.env.unwrapped, "get_state"):
            self._stored_state = self.env.unwrapped.get_state()
        else:
            raise NotImplementedError(
                "Environment does not expose a state() or get_state() method."
            )
        return self._stored_state

    def reset_to_state(self, state: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the provided state (or last stored state)."""
        if state is None:
            state = self._stored_state
        if state is None:
            raise ValueError("No state available to reset to.")
        obs, info = self.env.reset()
        # Restore simulator state.
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "set_state"):
            # MuJoCo environments expect set_state(qpos, qvel).
            state = np.asarray(state)
            ndim = state.ndim
            if ndim == 1:
                qpos_size = unwrapped.model.nq
                qvel_size = unwrapped.model.nv
                qpos = state[:qpos_size]
                qvel = state[qpos_size:qpos_size + qvel_size]
                unwrapped.set_state(qpos, qvel)
            else:
                unwrapped.set_state(state)
        elif hasattr(unwrapped, "state"):
            unwrapped.state()[:] = state
        else:
            raise NotImplementedError(
                "Environment does not expose a set_state() method."
            )
        obs = unwrapped._get_obs() if hasattr(unwrapped, "_get_obs") else obs
        return obs, info

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._stored_state = None
        return self.env.reset(**kwargs)


def sample_random_action(env: gym.Env) -> np.ndarray:
    """Sample a random action from the environment's action space."""
    return env.action_space.sample()
