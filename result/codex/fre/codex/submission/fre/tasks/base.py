"""Common interfaces for downstream evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class EvalTask:
    """A downstream reward function plus its episode configuration.

    ``reward(obs, action, next_obs)`` is vectorised: it accepts arrays of shape
    ``(N, obs_dim)`` / ``(N, act_dim)`` and returns ``(N,)`` rewards.
    """

    name: str = "task"
    group: str = "misc"
    max_episode_steps: int = 1000
    # Analytic output range of the reward, used to discretise rewards the same
    # way they were discretised during unsupervised pre-training.
    reward_range: Tuple[float, float] = (-1.0, 1.0)
    # Whether this task supplies a goal state the agent should be conditioned on
    # (goal-reaching tasks pass the ground-truth goal to GC-IQL / GC-BC).
    goal: Optional[np.ndarray] = None

    def reward(
        self, obs: np.ndarray, action: Optional[np.ndarray] = None,
        next_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare_env(self, env) -> None:
        """Optional hook to configure the environment before a rollout."""
        return None

    def reset_env(self, env, seed: Optional[int] = None):
        """Reset the environment (with the task-specific start state if any)."""
        return env.reset()

    # -- return normalisation --------------------------------------------------
    def normalize(self, raw_return: np.ndarray) -> np.ndarray:
        """Map a raw episode return into the [0, 100] range used in Table 1.

        The paper reports all results "normalized between 0 and 100".  The
        normalisation bounds are the analytic worst / best returns of the task
        over a full episode, which for each of our tasks equals
        ``max_episode_steps * reward_range[0]`` (worst) and
        ``max_episode_steps * reward_range[1]`` (best) -- except for
        goal-reaching rewards, where the best case is reaching the goal in a
        single step, i.e. accumulated reward ``-1`` (the reward is ``-1`` for
        every timestep that the goal has *not* been reached, ``0`` otherwise).
        """
        lo, hi = self.normalization_bounds()
        if hi <= lo:
            return np.zeros_like(raw_return, dtype=np.float64)
        out = 100.0 * (np.asarray(raw_return, dtype=np.float64) - lo) / (hi - lo)
        return np.clip(out, 0.0, 100.0)

    def normalization_bounds(self) -> Tuple[float, float]:
        lo = self.max_episode_steps * self.reward_range[0]
        hi = self.max_episode_steps * self.reward_range[1]
        return lo, hi


class GoalReachingEvalTask(EvalTask):
    """Sparse goal-reaching: reward ``-1`` until the goal is reached, ``0`` after."""

    group = "goal-reaching"
    reward_range = (-1.0, 0.0)

    def __init__(
        self,
        name: str,
        goal: np.ndarray,
        position_fn: Callable[[np.ndarray], np.ndarray],
        threshold: float = 2.0,
        max_episode_steps: int = 2000,
    ) -> None:
        self.name = name
        self.goal = np.asarray(goal, dtype=np.float32)
        self.position_fn = position_fn
        self.threshold = float(threshold)
        self.max_episode_steps = int(max_episode_steps)

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        pos = self.position_fn(np.asarray(obs))
        dist = np.linalg.norm(pos - self.goal, axis=-1)
        return np.where(dist < self.threshold, 0.0, -1.0).astype(np.float32)

    def normalization_bounds(self) -> Tuple[float, float]:
        # Worst case: the goal is never reached, so every step yields -1.
        # Best case: the goal is reached immediately, so a single -1 is incurred.
        return -float(self.max_episode_steps), -1.0


@dataclass
class TaskSuite:
    """A named collection of evaluation tasks (one column of Table 1)."""

    name: str
    tasks: List[EvalTask]
    max_episode_steps: Optional[int] = None
    description: str = ""

    @property
    def task_names(self) -> List[str]:
        return [t.name for t in self.tasks]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)


def evaluate_policy_on_task(
    env,
    task: EvalTask,
    latent: np.ndarray,
    act_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    num_episodes: int = 20,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Roll out a latent-conditioned policy on a single downstream task.

    Args:
        env: environment exposing ``reset()`` / ``step(action)`` where ``step``
            returns ``(obs, reward, done, info)``.
        task: the downstream reward function.
        latent: the ``z`` vector produced by the FRE encoder for this task.
        act_fn: callable ``(obs_batch, z_batch) -> action_batch``.
        num_episodes: number of evaluation episodes (20 in the paper).

    Returns:
        A dictionary with the mean raw return and the normalised (0-100) score.
    """
    max_steps = max_episode_steps or task.max_episode_steps
    returns = []
    for ep in range(num_episodes):
        obs = task.reset_env(env, None if seed is None else seed + ep)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        total = 0.0
        for _ in range(max_steps):
            action = act_fn(obs.reshape(1, -1), latent.reshape(1, -1))
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            step_out = env.step(action)
            next_obs = np.asarray(step_out[0], dtype=np.float32).reshape(-1)
            total += float(task.reward(next_obs.reshape(1, -1))[0])
            obs = next_obs
            if len(step_out) > 2 and bool(np.asarray(step_out[2]).reshape(-1)[0]):
                break
        returns.append(total)
    returns = np.asarray(returns, dtype=np.float64)
    return {
        "task": task.name,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "normalized": float(task.normalize(returns).mean()),
        "normalized_std": float(task.normalize(returns).std()),
    }
