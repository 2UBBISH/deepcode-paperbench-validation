"""Evaluation task definitions for the FRE zero-shot benchmarks.

Package initializer for :mod:`fre.envs`, which holds the *evaluation* reward
functions for the three domains studied in the paper:

* ``antmaze_tasks`` -- AntMaze goal-reaching (5 fixed goals), directional (4 unit
  directions), random-simplex (OpenSimplex seeds 1..5) and the corridor path tasks
  (Appendix "Ant Maze evaluation tasks").
* ``exorl_tasks``  -- ExORL cheetah/walker velocity tasks and 5 fixed goal states
  (Appendix "ExORL evaluation tasks", Appendix C.2).
* ``kitchen_tasks`` -- the seven standard sparse D4RL Kitchen subtasks, used
  directly as evaluation tasks (Appendix C.3).

This module deliberately contains no numerical model code: it only provides the
shared, framework-agnostic task containers (:class:`EvalTask`,
:class:`TaskSuite`) plus lazy re-exports of the domain factories, so that the
evaluation harness (``fre.eval.zero_shot_eval``) can iterate over heterogeneous
task suites through a single duck-typed interface.

Evaluation protocol (Source: 5.2): "All methods are evaluated using a mean over
twenty evaluation episodes, and each agent is trained using five random seeds,
with the standard deviation across seeds shown."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # shared containers
    "EvalTask",
    "TaskSuite",
    "build_suite",
    # protocol constants (Source: 5.2)
    "NUM_EVAL_EPISODES",
    "NUM_TRAINING_SEEDS",
    "FRE_CONTEXT_SAMPLES",
    "FB_SF_CONTEXT_SAMPLES",
    "NORMALIZED_RETURN_MIN",
    "NORMALIZED_RETURN_MAX",
    # antmaze
    "AntMazeTask",
    "make_antmaze_goal_tasks",
    "make_antmaze_directional_tasks",
    "make_antmaze_simplex_tasks",
    "make_antmaze_path_tasks",
    "make_antmaze_task_suite",
    "make_antmaze_goal_reaching_suite",
    "make_antmaze_directional_suite",
    "make_antmaze_simplex_suite",
    "make_antmaze_path_suite",
    "antmaze_suite_for_group",
    "ANTMAZE_EVAL_EPISODE_LENGTH",
    "ANTMAZE_GOAL_DISTANCE_THRESHOLD",
    "ANTMAZE_GOAL_LOCATIONS",
    "ANTMAZE_DIRECTIONS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_TASK_GROUPS",
    # exorl
    "ExORLTask",
    "make_cheetah_velocity_tasks",
    "make_walker_velocity_tasks",
    "make_exorl_goal_tasks",
    "make_exorl_task_suite",
    "exorl_suite_for_group",
    "EXORL_EVAL_EPISODE_LENGTH",
    "EXORL_GOAL_DISTANCE_THRESHOLD",
    "CHEETAH_VELOCITY_THRESHOLDS",
    "WALKER_VELOCITY_THRESHOLDS",
    "EXORL_NUM_GOALS",
    "EXORL_TASK_GROUPS",
    # kitchen
    "KitchenTask",
    "make_kitchen_tasks",
    "make_kitchen_task_suite",
    "kitchen_suite",
    "KITCHEN_TASK_NAMES",
    "KITCHEN_EVAL_EPISODE_LENGTH",
]

# --------------------------------------------------------------------------------------
# Evaluation protocol constants (Source: 5.2 and Table 1 caption)
# --------------------------------------------------------------------------------------
NUM_EVAL_EPISODES: int = 20
"""Mean over twenty evaluation episodes per task (Source: 5.2)."""

NUM_TRAINING_SEEDS: int = 5
"""Each agent is trained with five random seeds (Source: 5.2)."""

FRE_CONTEXT_SAMPLES: int = 32
"""FRE encodes 32 (state, reward) pairs at evaluation (Source: Table 1 caption)."""

FB_SF_CONTEXT_SAMPLES: int = 5120
"""FB/SF receive 5120 reward samples at evaluation (Source: Table 1 caption / 5.2)."""

NORMALIZED_RETURN_MIN: float = 0.0
NORMALIZED_RETURN_MAX: float = 100.0
"""Returns are reported normalized between 0 and 100 (Source: Table 1 caption)."""

# Episode length default for Kitchen when the addendum is silent.
KITCHEN_EVAL_EPISODE_LENGTH: int = 1000


# --------------------------------------------------------------------------------------
# Shared containers
# --------------------------------------------------------------------------------------
@dataclass
class EvalTask:
    """A single zero-shot evaluation task (a reward function plus metadata).

    The task is intentionally framework agnostic: ``reward_fn`` maps an array of
    observations ``(N, obs_dim)`` (or a single observation ``(obs_dim,)``) to an
    array of scalar rewards ``(N,)``.  Domain-specific subclasses simply fix the
    default ``domain``/``eval_episode_length``.

    Attributes
    ----------
    name:
        Task name, e.g. ``"goal-top"`` or ``"cheetah-run"``.
    domain:
        ``"antmaze"``, ``"exorl"`` or ``"kitchen"``.
    reward_fn:
        Callable ``observations -> rewards`` implementing the evaluation reward.
    task_group:
        Group the task belongs to for aggregation (e.g. ``"goal-reaching"``,
        ``"directional"``, ``"simplex"``, ``"path"``, ``"velocity"``, ``"goals"``).
    is_goal_task:
        True for the goal-reaching style tasks (reward ``-1`` until success,
        ``0`` (or ``1`` for Kitchen) on success) so the harness can report
        success rates alongside returns.
    goal:
        Optional goal location/state for goal-reaching tasks (used for logging and
        for sampling the context set).
    threshold:
        Optional success threshold (distance) for goal-reaching tasks.
    eval_episode_length:
        Maximum number of environment steps per evaluation episode.
    reward_min / reward_max:
        Optional known reward range, used to discretize rewards into the encoder's
        32 bins (Source: Appendix "Additional Details on the FRE architecture").
    metadata:
        Free-form extra information (directions, simplex seeds, velocity targets...).
    """

    name: str
    reward_fn: Callable[[Any], np.ndarray]
    domain: str = "generic"
    task_group: str = "generic"
    is_goal_task: bool = False
    goal: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    eval_episode_length: int = 1000
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def reward(self, observations: Any) -> np.ndarray:
        """Return the scalar reward for each observation (shape ``(N,)``)."""
        values = self.reward_fn(observations)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        return values

    def __call__(self, observations: Any) -> np.ndarray:
        return self.reward(observations)

    def success(self, observations: Any, threshold: Optional[float] = None) -> np.ndarray:
        """Boolean success indicator (goal tasks only).

        Goal-reaching rewards in AntMaze are ``-1`` until the goal is reached and
        ``0`` afterwards (Source: Appendix C.1 / "Ant Maze evaluation tasks"),
        while Kitchen subtasks give ``1`` on success and ``0`` otherwise
        (Source: Appendix C.3).  We therefore detect success from the reward
        value relative to the task's ``reward_max``.
        """
        values = self.reward(observations)
        if self.reward_max is not None:
            return values >= self.reward_max - 1e-9
        return values > -1.0 + 1e-9

    def describe(self) -> Dict[str, Any]:
        goal = None
        if self.goal is not None:
            goal = np.asarray(self.goal, dtype=np.float64).tolist()
        return {
            "name": self.name,
            "domain": self.domain,
            "task_group": self.task_group,
            "is_goal_task": bool(self.is_goal_task),
            "goal": goal,
            "threshold": self.threshold,
            "eval_episode_length": int(self.eval_episode_length),
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "metadata": dict(self.metadata),
        }


@dataclass
class TaskSuite:
    """An ordered collection of :class:`EvalTask` objects sharing a protocol.

    ``aggregate`` names the column the suite maps onto in the paper's result
    tables (e.g. ``"ant-goal-reaching"``, ``"exorl-cheetah-velocity"``),
    ``paper_name`` the row name used in the aggregated "all" tables.
    """

    name: str
    tasks: List[EvalTask]
    eval_episode_length: int = 1000
    aggregate: Optional[str] = None
    domain: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[EvalTask]:
        return iter(self.tasks)

    def __getitem__(self, item: int) -> EvalTask:
        return self.tasks[item]

    @property
    def names(self) -> List[str]:
        return [t.name for t in self.tasks]

    def get(self, name: str) -> EvalTask:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"task {name!r} not in suite {self.name!r} (have {self.names})")

    def rewards(self, observations: Any) -> np.ndarray:
        """Reward matrix of shape ``(num_tasks, N)`` for shared observations."""
        return np.stack([task.reward(observations) for task in self.tasks], axis=0)

    def subset(self, names: Sequence[str], name: Optional[str] = None) -> "TaskSuite":
        return TaskSuite(
            name=name or self.name,
            tasks=[self.get(n) for n in names],
            eval_episode_length=self.eval_episode_length,
            aggregate=self.aggregate,
            domain=self.domain,
            metadata=dict(self.metadata),
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "aggregate": self.aggregate,
            "eval_episode_length": int(self.eval_episode_length),
            "num_tasks": len(self.tasks),
            "tasks": self.describe_tasks(),
            "metadata": dict(self.metadata),
        }

    def describe_tasks(self) -> List[Dict[str, Any]]:
        return [t.describe() for t in self.tasks]


def build_suite(
    name: str,
    tasks: Iterable[EvalTask],
    eval_episode_length: Optional[int] = None,
    aggregate: Optional[str] = None,
    domain: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TaskSuite:
    """Assemble a :class:`TaskSuite` from an iterable of tasks.

    ``eval_episode_length``/``domain`` default to the first task's values, which
    keeps the domain modules terse (every task in a domain shares its step limit
    and domain tag).
    """
    tasks = list(tasks)
    if eval_episode_length is None:
        eval_episode_length = tasks[0].eval_episode_length if tasks else 1000
    if domain is None:
        domain = tasks[0].domain if tasks else "generic"
    return TaskSuite(
        name=name,
        tasks=tasks,
        eval_episode_length=int(eval_episode_length),
        aggregate=aggregate,
        domain=domain,
        metadata=dict(metadata or {}),
    )


# --------------------------------------------------------------------------------------
# Lazy re-exports of the domain task modules (PEP 562)
# --------------------------------------------------------------------------------------
_LAZY_ATTRS: Dict[str, str] = {
    "AntMazeTask": "antmaze_tasks",
    "make_antmaze_goal_tasks": "antmaze_tasks",
    "make_antmaze_directional_tasks": "antmaze_tasks",
    "make_antmaze_simplex_tasks": "antmaze_tasks",
    "make_antmaze_path_tasks": "antmaze_tasks",
    "make_antmaze_task_suite": "antmaze_tasks",
    "make_antmaze_goal_reaching_suite": "antmaze_tasks",
    "make_antmaze_directional_suite": "antmaze_tasks",
    "make_antmaze_simplex_suite": "antmaze_tasks",
    "make_antmaze_path_suite": "antmaze_tasks",
    "antmaze_suite_for_group": "antmaze_tasks",
    "ANTMAZE_EVAL_EPISODE_LENGTH": "antmaze_tasks",
    "ANTMAZE_GOAL_DISTANCE_THRESHOLD": "antmaze_tasks",
    "ANTMAZE_GOAL_LOCATIONS": "antmaze_tasks",
    "ANTMAZE_DIRECTIONS": "antmaze_tasks",
    "ANTMAZE_SIMPLEX_SEEDS": "antmaze_tasks",
    "ANTMAZE_TASK_GROUPS": "antmaze_tasks",
    "ExORLTask": "exorl_tasks",
    "make_cheetah_velocity_tasks": "exorl_tasks",
    "make_walker_velocity_tasks": "exorl_tasks",
    "make_exorl_goal_tasks": "exorl_tasks",
    "make_exorl_task_suite": "exorl_tasks",
    "exorl_suite_for_group": "exorl_tasks",
    "EXORL_EVAL_EPISODE_LENGTH": "exorl_tasks",
    "EXORL_GOAL_DISTANCE_THRESHOLD": "exorl_tasks",
    "CHEETAH_VELOCITY_THRESHOLDS": "exorl_tasks",
    "WALKER_VELOCITY_THRESHOLDS": "exorl_tasks",
    "EXORL_NUM_GOALS": "exorl_tasks",
    "EXORL_TASK_GROUPS": "exorl_tasks",
    "KitchenTask": "kitchen_tasks",
    "make_kitchen_tasks": "kitchen_tasks",
    "make_kitchen_task_suite": "kitchen_tasks",
    "kitchen_suite": "kitchen_tasks",
    "KITCHEN_TASK_NAMES": "kitchen_tasks",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy loader
    """Resolve a public name from its defining submodule (PEP 562)."""
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:  # pragma: no cover - trivial
    return sorted(set(list(globals().keys()) + __all__))
