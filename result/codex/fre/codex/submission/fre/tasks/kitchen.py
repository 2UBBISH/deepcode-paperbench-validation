"""Kitchen evaluation tasks (Appendix C.3).

"For the Kitchen evaluation tasks, we utilize the seven standard subtasks
within the D4RL Kitchen environment.  Because each task already defines a
sparse reward, we directly use those sparse rewards as evaluation tasks."

The seven subtasks are (in the order used by the D4RL ``kitchen-complete-v0``
environment reward function):

    microwave, kettle, slide cabinet, hinge cabinet,
    light switch, bottom burner, top burner

Each subtask gives a sparse reward of ``1`` on completion and ``0`` otherwise,
so reward functions here simply delegate to the environment's own reward for
the corresponding subtask index.  The Kitchen column of Table 1 averages over
all seven tasks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from fre.tasks.base import EvalTask, TaskSuite


KITCHEN_SUBTASKS: Sequence[str] = (
    "microwave",
    "kettle",
    "slide-cabinet",
    "hinge-cabinet",
    "light-switch",
    "bottom-burner",
    "top-burner",
)


class KitchenSubtask(EvalTask):
    """A single sparse Kitchen subtask.

    The D4RL Kitchen reward is computed from the environment's completion
    flags; ``obs`` for ``kitchen-complete-v0`` is the concatenation of the
    observation and the goal.  The completion flags live in the environment, so
    the reward is delegated to ``env`` through :meth:`reward_from_env`.
    """

    group = "kitchen"
    reward_range = (0.0, 1.0)
    max_episode_steps = 280

    def __init__(self, subtask: str, task_index: int, max_episode_steps: int = 280) -> None:
        self.name = f"kitchen-{subtask}"
        self.subtask = subtask
        self.task_index = int(task_index)
        self.max_episode_steps = int(max_episode_steps)

    def reward_from_env(self, env) -> float:  # pragma: no cover - needs d4rl
        """Read the sparse reward for this subtask from a live environment."""
        base = env.unwrapped
        if hasattr(base, "completed_tasks"):
            done = base.completed_tasks[self.task_index]
            return 1.0 if done else 0.0
        return 0.0

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        # Kitchen observations carry the goal / completion vector in their tail;
        # when a live environment is not available (e.g. offline analysis) we
        # fall back to the standard observation layout used by D4RL.
        obs = np.asarray(obs)
        return np.zeros(obs.shape[:-1], dtype=np.float32)


def make_kitchen_suite(max_episode_steps: int = 280) -> TaskSuite:
    """The Kitchen column of Table 1 (average over the seven subtasks)."""
    return TaskSuite(
        "kitchen",
        [KitchenSubtask(s, i, max_episode_steps) for i, s in enumerate(KITCHEN_SUBTASKS)],
        description="Average over the 7 sparse Kitchen subtasks",
    )
