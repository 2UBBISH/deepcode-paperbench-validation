"""Kitchen zero-shot evaluation tasks for FRE.

Paper references
----------------
Section 5.2 (evaluation protocol), verbatim:
    "All methods are evaluated using a mean over twenty evaluation episodes, and each agent
     is trained using five random seeds, with the standard deviation across seeds shown.
     FRE, GC-IQL, and GC-BC are implemented within the same codebase and with the same
     network structure."
Section 5.2 (context budget), verbatim:
    "...we give these methods 5120 reward samples during evaluation time (in comparison to
     only 32 for FRE)."
Appendix C.3 (Kitchen), verbatim:
    "we utilize the seven standard subtasks within the D4RL Kitchen environment. Because
     each task already defines a sparse reward, we directly use those sparse rewards as
     evaluation tasks."

Implementation notes
--------------------
*   Kitchen differs from AntMaze/ExORL in that the reward functions are *already defined*
    by the environment (binary sparse subtask indicators computed from the 30-d object
    state), so no random reward field is constructed here.  We simply wrap the sparse
    subtask rewards from :mod:`fre.data.kitchen_dataset` in the framework-agnostic
    :class:`fre.envs.EvalTask` container so the zero-shot harness can treat Kitchen exactly
    like the other domains (encode 32 ``(s, eta(s))`` pairs -> roll out ``pi(a | s, z)``).
*   Score normalisation follows Table 1 of the paper ("normalized between 0 and 100"): each
    sparse Kitchen subtask already lives in ``[0, 1]`` (sum of achieved subgoals in some
    D4RL variants), so the task declares ``reward_min=0`` / ``reward_max=1`` and the
    harness maps returns to ``[0, 100]``.
*   Episode length is *not* specified in the paper for Kitchen ("Source: not specified in
    the paper"); we default to 1000 steps (the D4RL Kitchen default), consistent with
    :mod:`fre.data.kitchen_dataset` (``KITCHEN_EVAL_EPISODE_LENGTH``).
*   The paper does not say whether Kitchen observations are normalized.  We expose the same
    optional per-dimension std normalization hook as the other domains
    (``normalize=True``) and keep it off by default, since the sparse subtask rewards are
    computed on the *raw* observation layout of the D4RL Kitchen environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Imports from the rest of the codebase, with graceful fallbacks so that this module can
# also be executed / imported standalone (mirrors antmaze_tasks.py / exorl_tasks.py).
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - package import path
    from fre.envs import EvalTask, TaskSuite, build_suite
except Exception:  # pragma: no cover - direct execution / partial build
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from fre.envs import EvalTask, TaskSuite, build_suite  # type: ignore
    except Exception:  # minimal self-contained fallbacks

        @dataclass
        class EvalTask:  # type: ignore[no-redef]
            """Minimal stand-in for fre.envs.EvalTask."""

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

            def reward(self, observations: np.ndarray) -> np.ndarray:
                return np.asarray(self.reward_fn(observations), dtype=np.float64).reshape(-1)

            def __call__(self, observations: np.ndarray) -> np.ndarray:
                return self.reward(observations)

        @dataclass
        class TaskSuite:  # type: ignore[no-redef]
            """Minimal stand-in for fre.envs.TaskSuite."""

            name: str
            tasks: List[EvalTask]
            eval_episode_length: int = 1000
            aggregate: Optional[str] = None
            domain: str = "generic"
            metadata: Dict[str, Any] = field(default_factory=dict)

            def __len__(self) -> int:
                return len(self.tasks)

            def __iter__(self):
                return iter(self.tasks)

            def __getitem__(self, index):
                return self.tasks[index]

            @property
            def names(self) -> List[str]:
                return [t.name for t in self.tasks]

        def build_suite(name, tasks, eval_episode_length=None, aggregate=None,
                        domain=None, metadata=None):  # type: ignore[no-redef]
            tasks = list(tasks)
            return TaskSuite(
                name=name,
                tasks=tasks,
                eval_episode_length=int(eval_episode_length or (tasks[0].eval_episode_length if tasks else 1000)),
                aggregate=aggregate,
                domain=domain or (tasks[0].domain if tasks else "generic"),
                metadata=dict(metadata or {}),
            )

# --------------------------------------------------------------------------------------
# Kitchen dataset constants / sparse reward helpers (single source of truth).
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - normal package import
    from fre.data.kitchen_dataset import (  # type: ignore
        KITCHEN_DATASET_NAME,
        KITCHEN_EVAL_EPISODE_LENGTH,
        KITCHEN_ELEMENT_GOALS,
        KITCHEN_ELEMENT_INDICES,
        KITCHEN_OBS_DIM,
        KITCHEN_TASK_ELEMENTS,
        KITCHEN_TASK_THRESHOLD,
        KITCHEN_TASKS,
        compute_all_task_rewards,
        compute_task_reward,
        element_reached,
        make_task_reward_function,
        resolve_task_name,
    )
except Exception:  # pragma: no cover - fallback: local definitions copied from D4RL
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.dirname(os.path.dirname(_here))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from fre.data.kitchen_dataset import (  # type: ignore
            KITCHEN_DATASET_NAME,
            KITCHEN_EVAL_EPISODE_LENGTH,
            KITCHEN_ELEMENT_GOALS,
            KITCHEN_ELEMENT_INDICES,
            KITCHEN_OBS_DIM,
            KITCHEN_TASK_ELEMENTS,
            KITCHEN_TASK_THRESHOLD,
            KITCHEN_TASKS,
            compute_all_task_rewards,
            compute_task_reward,
            element_reached,
            make_task_reward_function,
            resolve_task_name,
        )
    except Exception:
        # Documented fallback (values taken from the D4RL Kitchen environment).
        KITCHEN_TASKS: Tuple[str, ...] = (
            "microwave",
            "kettle",
            "light switch",
            "slide cabinet",
            "bottom burner",
            "top burner",
            "hinge cabinet",
        )
        KITCHEN_EVAL_EPISODE_LENGTH = 1000
        KITCHEN_OBS_DIM = 60
        KITCHEN_TASK_THRESHOLD = 0.3
        KITCHEN_DATASET_NAME = "kitchen-complete-v0"
        KITCHEN_ELEMENT_INDICES = {
            "microwave": (18, 19),
            "kettle": (6, 7),
            "light switch": (3, 4),
            "slide cabinet": (10,),
            "bottom burner": (14, 15),
            "top burner": (16, 17),
            "hinge cabinet": (11, 12),
        }
        KITCHEN_ELEMENT_GOALS = {
            "microwave": (0.0, -0.75),
            "kettle": (1.6, 0.0),
            "light switch": (0.0, 0.0),
            "slide cabinet": (0.37,),
            "bottom burner": (0.0, -0.75),
            "top burner": (0.0, -0.75),
            "hinge cabinet": (0.0, 0.0),
        }
        KITCHEN_TASK_ELEMENTS = {
            "microwave": ("microwave",),
            "kettle": ("kettle",),
            "light switch": ("light switch",),
            "slide cabinet": ("slide cabinet",),
            "bottom burner": ("bottom burner",),
            "top burner": ("top burner",),
            "hinge cabinet": ("hinge cabinet",),
        }

        def resolve_task_name(task: str) -> str:  # type: ignore[no-redef]
            key = str(task).strip().lower().replace("_", " ").replace("-", " ")
            for canonical in KITCHEN_TASKS:
                if key == canonical or key == canonical.replace(" ", ""):
                    return canonical
            raise ValueError(f"Unknown Kitchen task {task!r}; expected one of {KITCHEN_TASKS}")

        def element_reached(observations, element, threshold=KITCHEN_TASK_THRESHOLD):  # type: ignore[no-redef]
            obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
            idx = KITCHEN_ELEMENT_INDICES[element]
            goal = np.asarray(KITCHEN_ELEMENT_GOALS[element], dtype=np.float64)
            dist = np.linalg.norm(obs[:, list(idx)] - goal[None, :], axis=-1)
            return dist <= float(threshold)

        def compute_task_reward(observations, task, threshold=None, reward_success=1.0,
                                reward_failure=0.0, elements=None):  # type: ignore[no-redef]
            obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
            canonical = resolve_task_name(task)
            thresh = KITCHEN_TASK_THRESHOLD if threshold is None else float(threshold)
            elems = elements if elements is not None else KITCHEN_TASK_ELEMENTS[canonical]
            done = np.ones(obs.shape[0], dtype=bool)
            for elem in elems:
                done &= element_reached(obs, elem, threshold=thresh)
            return np.where(done, float(reward_success), float(reward_failure))

        def compute_all_task_rewards(observations, tasks=KITCHEN_TASKS, threshold=None):  # type: ignore[no-redef]
            return {t: compute_task_reward(observations, t, threshold=threshold) for t in tasks}

        def make_task_reward_function(task: str, threshold=None):  # type: ignore[no-redef]
            canonical = resolve_task_name(task)

            def _reward(observations):
                return compute_task_reward(observations, canonical, threshold=threshold)

            return _reward


__all__ = [
    "KitchenTask",
    "KitchenSubtask",
    "make_kitchen_tasks",
    "make_kitchen_task_suite",
    "kitchen_suite",
    "kitchen_task_groups",
    "kitchen_task_names",
    "kitchen_element_reached",
    "KITCHEN_DOMAIN",
    "KITCHEN_TASK_NAMES",
    "KITCHEN_TASK_GROUP",
    "KITCHEN_SUBTASK_REWARD_MIN",
    "KITCHEN_SUBTASK_REWARD_MAX",
    "KITCHEN_EVAL_EPISODE_LENGTH",
    "EVAL_NUM_EPISODES",
    "EVAL_NUM_SEEDS",
    "FRE_CONTEXT_SAMPLES",
    "FB_SF_CONTEXT_SAMPLES",
    "NORMALIZED_RETURN_MIN",
    "NORMALIZED_RETURN_MAX",
]

# --------------------------------------------------------------------------------------
# Protocol constants (paper Section 5.2 / Table 1 caption).
# --------------------------------------------------------------------------------------
KITCHEN_DOMAIN = "kitchen"
KITCHEN_TASK_NAMES: Tuple[str, ...] = tuple(KITCHEN_TASKS)
KITCHEN_TASK_GROUP = "kitchen"
KITCHEN_SUBTASK_REWARD_MIN = 0.0
KITCHEN_SUBTASK_REWARD_MAX = 1.0

#: Evaluation protocol of Section 5.2 ("a mean over twenty evaluation episodes",
#: "each agent is trained using five random seeds") and Table 1 caption for the
#: discounted-to-``[0, 100]`` normalisation and the context budgets.
EVAL_NUM_EPISODES = 20
EVAL_NUM_SEEDS = 5
FRE_CONTEXT_SAMPLES = 32
FB_SF_CONTEXT_SAMPLES = 5120
NORMALIZED_RETURN_MIN = 0.0
NORMALIZED_RETURN_MAX = 100.0


# --------------------------------------------------------------------------------------
# Task container
# --------------------------------------------------------------------------------------
@dataclass
class KitchenTask(EvalTask):
    """A single Kitchen zero-shot evaluation task.

    Kitchen reward functions are given by the environment (sparse subtask indicators),
    so the task simply forwards to :func:`fre.data.kitchen_dataset.compute_task_reward`.
    """

    task_name: str = ""
    reward_threshold: Optional[float] = None
    elements: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        # ``EvalTask`` is a dataclass; make sure our defaults stay consistent.
        if not self.task_name:
            self.task_name = str(self.name)
        if self.reward_min is None:
            self.reward_min = KITCHEN_SUBTASK_REWARD_MIN
        if self.reward_max is None:
            self.reward_max = KITCHEN_SUBTASK_REWARD_MAX
        if not self.domain:
            self.domain = KITCHEN_DOMAIN
        if not self.task_group:
            self.task_group = KITCHEN_TASK_GROUP
        if "score_bounds" not in (self.metadata or {}):
            self.metadata = dict(self.metadata or {})
            self.metadata["score_bounds"] = (float(self.reward_min), float(self.reward_max))

    # -- reward -------------------------------------------------------------------
    def subtask_reward(self, observations: Any) -> np.ndarray:
        """Sparse subtask indicator: 1.0 once the subgoal is achieved, else 0.0."""
        return compute_task_reward(
            observations,
            self.task_name,
            threshold=self.reward_threshold,
            reward_success=float(self.reward_max),
            reward_failure=float(self.reward_min),
            elements=self.elements,
        )

    def rewards(self, observations: Any) -> np.ndarray:
        """Batch (``(N, obs_dim)``) or single (``(obs_dim,)``) reward evaluation."""
        return np.asarray(self.subtask_reward(observations), dtype=np.float64).reshape(-1)

    def reward(self, observations: Any) -> np.ndarray:  # type: ignore[override]
        return self.rewards(observations)

    def __call__(self, observations: Any) -> np.ndarray:
        return self.rewards(observations)

    def success(self, observations: Any, threshold: Optional[float] = None) -> np.ndarray:
        """Boolean array marking states where the subtask subgoals are all achieved."""
        thresh = self.reward_threshold if threshold is None else threshold
        return compute_task_reward(
            observations,
            self.task_name,
            threshold=thresh,
            reward_success=1.0,
            reward_failure=0.0,
            elements=self.elements,
        ).astype(bool)

    # -- bounds / metadata ---------------------------------------------------------
    def scores_bounds(self) -> Tuple[float, float]:
        """Score bounds used by the harness to normalize returns to ``[0, 100]``."""
        return (float(self.reward_min or 0.0), float(self.reward_max or 1.0))

    def copy_with_name(self, name: str) -> "KitchenTask":
        return KitchenTask(
            name=name,
            reward_fn=self.reward_fn,
            domain=self.domain,
            task_group=self.task_group,
            is_goal_task=self.is_goal_task,
            goal=self.goal,
            threshold=self.threshold,
            eval_episode_length=self.eval_episode_length,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            metadata=dict(self.metadata or {}),
            task_name=self.task_name,
            reward_threshold=self.reward_threshold,
            elements=self.elements,
        )

    def describe(self) -> Dict[str, Any]:
        info = {
            "name": self.name,
            "domain": self.domain,
            "task_group": self.task_group,
            "task_name": self.task_name,
            "is_goal_task": self.is_goal_task,
            "eval_episode_length": int(self.eval_episode_length),
            "score_bounds": self.scores_bounds(),
            "threshold": self.reward_threshold,
            "elements": tuple(self.elements) if self.elements is not None else None,
        }
        info.update(dict(self.metadata or {}))
        return info


#: Alias kept for readability at call sites ("one Kitchen subtask").
KitchenSubtask = KitchenTask


def kitchen_element_reached(observations: Any, element: str,
                            threshold: Optional[float] = None) -> np.ndarray:
    """Boolean mask for one Kitchen element being at its target pose."""
    thresh = KITCHEN_TASK_THRESHOLD if threshold is None else float(threshold)
    return np.asarray(element_reached(observations, element, threshold=thresh), dtype=bool)


# --------------------------------------------------------------------------------------
# Task factories
# --------------------------------------------------------------------------------------
def make_kitchen_tasks(
    tasks: Sequence[str] = KITCHEN_TASK_NAMES,
    threshold: Optional[float] = None,
    eval_episode_length: int = KITCHEN_EVAL_EPISODE_LENGTH,
    task_group: str = KITCHEN_TASK_GROUP,
    normalize: bool = False,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    obs_dim: Optional[int] = None,
    elements: Optional[Dict[str, Tuple[str, ...]]] = None,
    **kwargs: Any,
) -> List[KitchenTask]:
    """Build the seven (by default) Kitchen sparse-subtask evaluation tasks.

    Parameters mirror the other domains so the harness can build every suite the same way.
    ``normalize``/``state_mean``/``state_std`` are accepted for interface symmetry with
    AntMaze/ExORL; Kitchen subtask rewards are evaluated on raw observations, so
    normalization is only recorded in the task metadata (the paper does not specify
    Kitchen observation normalization).
    """
    out: List[KitchenTask] = []
    for raw_name in tasks:
        canonical = resolve_task_name(raw_name)
        elems = None
        if elements is not None and canonical in elements:
            elems = tuple(elements[canonical])
        else:
            elems = tuple(KITCHEN_TASK_ELEMENTS.get(canonical, (canonical,)))
        reward_fn = make_task_reward_function(canonical, threshold=threshold)
        metadata: Dict[str, Any] = {
            "score_bounds": (KITCHEN_SUBTASK_REWARD_MIN, KITCHEN_SUBTASK_REWARD_MAX),
            "normalize": bool(normalize),
            "obs_dim": int(obs_dim) if obs_dim is not None else None,
            "threshold": KITCHEN_TASK_THRESHOLD if threshold is None else float(threshold),
        }
        out.append(
            KitchenTask(
                name=f"kitchen-{canonical.replace(' ', '-')}",
                reward_fn=reward_fn,
                domain=KITCHEN_DOMAIN,
                task_group=task_group,
                is_goal_task=True,  # sparse subgoal-reaching indicator
                goal=None,
                threshold=KITCHEN_TASK_THRESHOLD if threshold is None else float(threshold),
                eval_episode_length=int(eval_episode_length),
                reward_min=KITCHEN_SUBTASK_REWARD_MIN,
                reward_max=KITCHEN_SUBTASK_REWARD_MAX,
                metadata=metadata,
                task_name=canonical,
                reward_threshold=None if threshold is None else float(threshold),
                elements=elems,
            )
        )
    return out


def make_kitchen_task_suite(
    group: str = "kitchen",
    tasks: Sequence[str] = KITCHEN_TASK_NAMES,
    threshold: Optional[float] = None,
    eval_episode_length: int = KITCHEN_EVAL_EPISODE_LENGTH,
    name: Optional[str] = None,
    aggregate: Optional[str] = None,
    **kwargs: Any,
) -> TaskSuite:
    """Assemble a :class:`TaskSuite` over the Kitchen sparse subtasks."""
    key = str(group).strip().lower() if group else "kitchen"
    if key in ("kitchen-all", "all", "kitchen"):
        chosen = list(tasks)
    elif key.startswith("kitchen-"):
        canonical = resolve_task_name(key[len("kitchen-"):])
        chosen = [canonical]
    elif key in {t.replace(" ", "-") for t in KITCHEN_TASK_NAMES}:
        canonical = resolve_task_name(key)
        chosen = [canonical]
    else:
        # Unknown group: fall back to the full subtask list instead of failing hard.
        chosen = list(tasks)

    built = make_kitchen_tasks(
        tasks=chosen,
        threshold=threshold,
        eval_episode_length=eval_episode_length,
        **kwargs,
    )
    suite_name = name or ("kitchen" if len(built) > 1 else built[0].name)
    return build_suite(
        suite_name,
        built,
        eval_episode_length=eval_episode_length,
        aggregate=aggregate or "kitchen",
        domain=KITCHEN_DOMAIN,
        metadata={"group": key, "num_subtasks": len(built)},
    )


def kitchen_suite(**kwargs: Any) -> TaskSuite:
    """Convenience alias: the default ``kitchen`` suite (7 sparse subtasks)."""
    return make_kitchen_task_suite(**kwargs)


def kitchen_task_groups() -> Tuple[str, ...]:
    """Names of the available Kitchen suite groups."""
    return ("kitchen", "kitchen-all") + tuple(
        f"kitchen-{t.replace(' ', '-')}" for t in KITCHEN_TASK_NAMES
    )


def kitchen_task_names() -> Tuple[str, ...]:
    """Canonical Kitchen subtask names (in D4RL order)."""
    return KITCHEN_TASK_NAMES
