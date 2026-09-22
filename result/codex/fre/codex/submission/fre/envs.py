"""Environment factories and rollout wrappers for the three evaluation domains.

These wrappers expose a uniform ``reset()`` / ``step(action)`` interface plus
the task-specific reset behaviour described in the paper:

  * AntMaze: the ant starts in the *centre* of the maze (Appendix C.1), and
    goal-reaching tasks additionally set the D4RL target so that the standard
    environment reward agrees with the task reward;
  * ExORL: the DeepMind Control Suite ``walker`` / ``cheetah`` environments at
    a fixed episode length of 1000 steps;
  * Kitchen: ``kitchen-complete-v0`` with its sparse subtask rewards.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


class GymEnvWrapper:
    """Thin adapter so that ``step`` always returns ``(obs, reward, done, info)``."""

    def __init__(self, env, task=None) -> None:
        self.env = env
        self.task = task

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:  # gymnasium API
            obs, _reward, terminated, truncated, info = out
            return np.asarray(obs, dtype=np.float32).reshape(-1), 0.0, bool(terminated or truncated), info
        obs, reward, done, info = out
        return np.asarray(obs, dtype=np.float32).reshape(-1), reward, bool(done), info


class AntMazeEnvWrapper(GymEnvWrapper):
    """AntMaze rollout environment with the paper's centre reset."""

    def __init__(self, env, task=None, max_episode_steps: int = 2000) -> None:
        super().__init__(env, task)
        self.max_episode_steps = max_episode_steps
        self._step = 0

    def reset(self, seed: Optional[int] = None):
        from fre.tasks.antmaze import center_reset_xy

        obs = super().reset(seed)
        # Move the ant to the centre of the maze.
        try:
            base = self.env.unwrapped
            loco = getattr(base, "wrapped_env", None)
            if loco is not None and hasattr(loco, "set_xy"):
                loco.set_xy(center_reset_xy())
                obs = np.asarray(loco._get_obs(), dtype=np.float32).reshape(-1)
        except Exception:  # pragma: no cover - depends on the installed d4rl
            pass
        self._step = 0
        return obs

    def step(self, action):
        out = super().step(action)
        self._step += 1
        done = bool(out[2]) or self._step >= self.max_episode_steps
        return out[0], out[1], done, out[3]


class AntMazeGoalEnvWrapper(AntMazeEnvWrapper):
    """Additionally exposes the ant's (x, y) position and velocity for logging."""

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.env.unwrapped.wrapped_env.get_xy(), dtype=np.float32)


def make_antmaze_env_factory(env_name: str = "antmaze-large-diverse-v2", max_episode_steps: int = 2000):
    """Return ``env_factory(task) -> AntMazeEnvWrapper``."""
    import gym
    import d4rl  # noqa: F401

    def factory(task):
        env = gym.make(env_name)
        wrapper = AntMazeEnvWrapper(env, task, max_episode_steps=max_episode_steps)
        if task is not None and getattr(task, "goal", None) is not None and task.group == "goal-reaching":
            # Match the D4RL target to the task goal so that the environment's
            # own termination behaviour agrees with the evaluation reward.
            try:
                env.unwrapped.set_target(np.asarray(task.goal, dtype=np.float64))
            except Exception:
                pass
        return wrapper

    return factory


def make_exorl_env_factory(domain: str, max_episode_steps: int = 1000):
    """Return ``env_factory(task) -> GymEnvWrapper`` for DMC walker / cheetah."""
    from dm_control import suite

    task_name = {"walker": "walk", "cheetah": "run"}[domain]

    def factory(task):
        env = suite.load(domain_name=domain, task_name=task_name)
        env = _DMControlEpisodeWrapper(env, max_episode_steps)
        return GymEnvWrapper(env, task)

    return factory


class _DMControlEpisodeWrapper:
    """Wrap a dm_control environment behind the gym-style interface.

    dm_control observations are dictionaries; we flatten the ``observations``
    entry (the standard observation vector) in the same order used by the
    ExORL datasets.
    """

    def __init__(self, env, max_episode_steps: int = 1000) -> None:
        self.env = env
        self.max_episode_steps = max_episode_steps
        self._time_step = None
        self._steps = 0
        self._action_spec = env.action_spec()

    def _obs(self) -> np.ndarray:
        return np.asarray(self._time_step.observation["observations"], dtype=np.float32).reshape(-1)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.env.task._random = np.random.RandomState(seed)
        self._time_step = self.env.reset()
        self._steps = 0
        return self._obs()

    def step(self, action):
        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)
        self._time_step = self.env.step(action)
        self._steps += 1
        done = bool(self._time_step.last()) or self._steps >= self.max_episode_steps
        return self._obs(), 0.0, done, {}


def make_kitchen_env_factory(env_name: str = "kitchen-complete-v0", max_episode_steps: int = 280):
    """Return ``env_factory(task) -> GymEnvWrapper`` for D4RL Kitchen."""
    import gym
    import d4rl  # noqa: F401

    def factory(task):
        env = gym.make(env_name)
        return _KitchenEnvWrapper(env, task, max_episode_steps)

    return factory


class _KitchenEnvWrapper(GymEnvWrapper):
    """Kitchen wrapper that reports the sparse reward of a single subtask."""

    def __init__(self, env, task=None, max_episode_steps: int = 280) -> None:
        super().__init__(env, task)
        self.max_episode_steps = max_episode_steps
        self._steps = 0

    def reset(self, seed: Optional[int] = None):
        self._steps = 0
        obs = super().reset(seed)
        if self.task is not None:
            try:
                base = self.env.unwrapped
                base.set_goal(self.task.task_index)
            except Exception:
                pass
        return obs

    def step(self, action):
        out = super().step(action)
        self._steps += 1
        done = bool(out[2]) or self._steps >= self.max_episode_steps
        return out[0], out[1], done, out[3]
