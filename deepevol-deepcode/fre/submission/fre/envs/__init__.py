"""Environment and evaluation-task package for Functional Reward Encodings (FRE).

This package aggregates:

* :mod:`fre.envs.reward_wrappers` -- gym/dm_control wrappers that replace the
  native environment reward with an arbitrary state reward function ``eta`` for
  zero-shot FRE evaluation, plus ExORL physics-feature observation augmentation.
* :mod:`fre.envs.antmaze_tasks` -- the AntMaze evaluation task suite (5
  goal-reaching tasks, 4 directional tasks, 5 opensimplex tasks, 3 path tasks).
* :mod:`fre.envs.exorl_tasks` -- the ExORL (walker/cheetah) evaluation task
  suite (velocity + fixed goal-state reaching).
* :mod:`fre.envs.kitchen_tasks` -- the 7 standard sparse Kitchen subtasks.

Heavy, optional dependencies (``gym``, ``d4rl``, ``dm_control``, ``torch``) are
imported lazily inside the individual modules so that ``import fre.envs`` stays
cheap and always works, even in a bare environment.
"""

from __future__ import annotations

from typing import Any, Tuple

from .reward_wrappers import (  # noqa: F401
    DEFAULT_EXORL_GOAL_THRESHOLD,
    DEFAULT_GOAL_THRESHOLD,
    DEFAULT_MAX_EPISODE_STEPS,
    PhysicsObservationWrapper,
    RewardFunctionWrapper,
    RewardWrapper,
    SuccessWrapper,
    TimeLimitWrapper,
    as_reward_fn,
    augment_observation_with_physics,
    call_env_reset,
    call_env_step,
    compute_exorl_physics_features,
    constant_reward_fn,
    evaluate_reward_function,
    exorl_reward_state_fn,
    inject_reward_fn,
    make_goal_success_fn,
    make_success_done_fn,
    normalize_reward_fn,
    remove_reward_fn,
    wrap_env,
    wrap_reward,
)

from .antmaze_tasks import (  # noqa: F401
    ANTMAZE_ENV_NAME,
    ANTMAZE_GOAL_DISTANCE,
    ANTMAZE_GOAL_TASKS,
    ANTMAZE_GRID_CENTER,
    ANTMAZE_GRID_EXTENT,
    ANTMAZE_MAX_EPISODE_STEPS,
    ANTMAZE_PATH_TASKS,
    ANTMAZE_SIMPLEX_SEEDS,
    ANTMAZE_TASK_SETS,
    ANTMAZE_XY_INDICES,
    ANTMAZE_XY_VELOCITY_INDICES,
    DEFAULT_CORRIDOR_WIDTH,
    DEFAULT_DIRECTIONS,
    AntMazeDirectionalReward,
    AntMazeGoalReward,
    AntMazePathReward,
    AntMazeSimplexReward,
    antmaze_encoder_states,
    ant_velocity,
    ant_xy,
    build_task_set as build_antmaze_task_set,
    build_tasks as build_antmaze_tasks,
    encoding_samples_for_task as antmaze_encoding_samples_for_task,
    encoding_samples_from_env as antmaze_encoding_samples_from_env,
    get_task as get_antmaze_task,
    get_sim_state,
    list_task_sets as list_antmaze_task_sets,
    make_antmaze_env,
    make_antmaze_task_env,
    make_directional_task,
    make_goal_task as make_antmaze_goal_task,
    make_path_task,
    make_simplex_task,
    maze_center_xy,
    path_distance,
    path_progress,
    reset_to_center,
    set_sim_state,
    task_names as antmaze_task_names,
    xy_grid_extent,
)

from .exorl_tasks import (  # noqa: F401
    CHEETAH_RUN_THRESHOLD,
    CHEETAH_WALK_THRESHOLD,
    EXORL_AUGMENT_DIM,
    EXORL_BASE_STATE_DIM,
    EXORL_DOMAINS,
    EXORL_ENV_TASKS,
    EXORL_GOAL_DISTANCE,
    EXORL_MAX_EPISODE_STEPS,
    EXORL_NUM_GOALS,
    EXORL_TASK_SETS,
    EXORL_VELOCITY_OFFSET,
    WALKER_VELOCITY_THRESHOLDS,
    DMControlEnv,
    ExORLGoalReward,
    ExORLVelocityReward,
    build_task_set as build_exorl_task_set,
    build_tasks as build_exorl_tasks,
    encoding_samples_for_task as exorl_encoding_samples_for_task,
    encoding_samples_from_env as exorl_encoding_samples_from_env,
    exorl_domain,
    exorl_encoder_states,
    exorl_goal_dims,
    exorl_physics_features,
    exorl_score_velocity,
    exorl_velocity,
    exorl_velocity_index,
    get_task as get_exorl_task,
    list_task_sets as list_exorl_task_sets,
    make_exorl_env,
    make_exorl_task_env,
    make_goal_task as make_exorl_goal_task,
    make_velocity_task,
    reset_exorl_env,
    select_goal_states,
    task_names as exorl_task_names,
)

from .kitchen_tasks import (  # noqa: F401
    KITCHEN_BONUS_THRESH,
    KITCHEN_ELEMENT_GOALS,
    KITCHEN_ENV_NAME,
    KITCHEN_MAX_EPISODE_STEPS,
    KITCHEN_OBS_DIM,
    KITCHEN_SUBTASKS,
    KITCHEN_TASK_SETS,
    KitchenAllSubtasksReward,
    KitchenSubtaskReward,
    build_task_set as build_kitchen_task_set,
    build_tasks as build_kitchen_tasks,
    encoding_samples_for_task as kitchen_encoding_samples_for_task,
    encoding_samples_from_env as kitchen_encoding_samples_from_env,
    get_task as get_kitchen_task,
    kitchen_completed_elements,
    kitchen_completion_fraction,
    kitchen_element_achieved,
    kitchen_encoder_states,
    kitchen_subtask_success,
    list_task_sets as list_kitchen_task_sets,
    make_all_subtasks_task,
    make_kitchen_env,
    make_kitchen_task_env,
    make_subtask as make_kitchen_subtask,
    reset_kitchen_env,
    task_names as kitchen_task_names,
)

# The ``TaskSpec`` dataclasses defined in the three task modules share the same
# public surface we rely on during evaluation (``reward``, ``is_done``,
# ``success_fn_for_wrapper``), so we expose them under domain-qualified aliases.
from .antmaze_tasks import TaskSpec as AntMazeTaskSpec  # noqa: F401
from .exorl_tasks import TaskSpec as ExORLTaskSpec  # noqa: F401
from .kitchen_tasks import TaskSpec as KitchenTaskSpec  # noqa: F401


# ---------------------------------------------------------------------------
# Domain dispatch
# ---------------------------------------------------------------------------
#: Aggregate task-set names supported by each domain, used by the evaluation
#: harness to enumerate the tasks of Table 1 of the paper.
DOMAIN_TASK_SETS = {
    "antmaze": ANTMAZE_TASK_SETS,
    "exorl:walker": EXORL_TASK_SETS,
    "exorl:cheetah": EXORL_TASK_SETS,
    "exorl": EXORL_TASK_SETS,
    "kitchen": KITCHEN_TASK_SETS,
}


def domain_from_task_name(name: str) -> str:
    """Infer the domain of a task from its (prefixed) name.

    >>> domain_from_task_name("ant-path-loop")
    'antmaze'
    >>> domain_from_task_name("exorl-walker-velocity")
    'exorl'
    >>> domain_from_task_name("kitchen-microwave")
    'kitchen'
    """
    lowered = str(name).lower()
    if lowered.startswith("ant") or "antmaze" in lowered:
        return "antmaze"
    if "exorl" in lowered or "cheetah" in lowered or "walker" in lowered:
        return "exorl"
    if "kitchen" in lowered:
        return "kitchen"
    raise ValueError(f"Could not infer domain from task name: {name!r}")


def build_tasks(
    domain: str,
    task_set: str = "all",
    **kwargs: Any,
) -> list:
    """Build the task list for ``domain`` / ``task_set``.

    A thin dispatch over the per-domain ``build_task_set`` implementations so
    the evaluation harness (``fre/evaluation/evaluate.py``) and the CLI entry
    point (``fre/main.py``) can enumerate tasks uniformly.
    """
    domain = str(domain).lower()
    if domain.startswith("antmaze") or domain == "ant":
        return build_antmaze_task_set(task_set=task_set, **kwargs)
    if domain.startswith("exorl"):
        domain_kwargs = dict(kwargs)
        # ``exorl:walker`` style strings carry the sub-domain.
        if ":" in domain:
            domain_kwargs.setdefault("domain", domain.split(":", 1)[1])
        return build_exorl_task_set(task_set=task_set, **domain_kwargs)
    if domain.startswith("kitchen"):
        return build_kitchen_task_set(task_set=task_set, **kwargs)
    raise ValueError(f"Unknown domain {domain!r}; expected antmaze/exorl/kitchen")


__all__ = [
    # reward wrappers
    "RewardWrapper",
    "RewardFunctionWrapper",
    "SuccessWrapper",
    "TimeLimitWrapper",
    "PhysicsObservationWrapper",
    "wrap_env",
    "wrap_reward",
    "as_reward_fn",
    "evaluate_reward_function",
    "normalize_reward_fn",
    "constant_reward_fn",
    "make_goal_success_fn",
    "make_success_done_fn",
    "call_env_reset",
    "call_env_step",
    "compute_exorl_physics_features",
    "augment_observation_with_physics",
    "exorl_reward_state_fn",
    "inject_reward_fn",
    "remove_reward_fn",
    "DEFAULT_GOAL_THRESHOLD",
    "DEFAULT_EXORL_GOAL_THRESHOLD",
    "DEFAULT_MAX_EPISODE_STEPS",
    # antmaze tasks
    "ANTMAZE_ENV_NAME",
    "ANTMAZE_GOAL_DISTANCE",
    "ANTMAZE_GOAL_TASKS",
    "ANTMAZE_GRID_CENTER",
    "ANTMAZE_GRID_EXTENT",
    "ANTMAZE_MAX_EPISODE_STEPS",
    "ANTMAZE_PATH_TASKS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_TASK_SETS",
    "ANTMAZE_XY_INDICES",
    "ANTMAZE_XY_VELOCITY_INDICES",
    "DEFAULT_CORRIDOR_WIDTH",
    "DEFAULT_DIRECTIONS",
    "AntMazeGoalReward",
    "AntMazeDirectionalReward",
    "AntMazeSimplexReward",
    "AntMazePathReward",
    "AntMazeTaskSpec",
    "make_goal_task",
    "make_antmaze_goal_task",
    "make_directional_task",
    "make_simplex_task",
    "make_path_task",
    "get_antmaze_task",
    "build_antmaze_task_set",
    "build_antmaze_tasks",
    "antmaze_task_names",
    "list_antmaze_task_sets",
    "make_antmaze_env",
    "make_antmaze_task_env",
    "reset_to_center",
    "get_sim_state",
    "set_sim_state",
    "maze_center_xy",
    "xy_grid_extent",
    "ant_xy",
    "ant_velocity",
    "path_distance",
    "path_progress",
    "antmaze_encoder_states",
    "antmaze_encoding_samples_for_task",
    "antmaze_encoding_samples_from_env",
    # exorl tasks
    "EXORL_DOMAINS",
    "EXORL_MAX_EPISODE_STEPS",
    "EXORL_GOAL_DISTANCE",
    "EXORL_AUGMENT_DIM",
    "EXORL_BASE_STATE_DIM",
    "EXORL_VELOCITY_OFFSET",
    "EXORL_ENV_TASKS",
    "EXORL_NUM_GOALS",
    "EXORL_TASK_SETS",
    "WALKER_VELOCITY_THRESHOLDS",
    "CHEETAH_RUN_THRESHOLD",
    "CHEETAH_WALK_THRESHOLD",
    "ExORLVelocityReward",
    "ExORLGoalReward",
    "ExORLTaskSpec",
    "DMControlEnv",
    "exorl_domain",
    "exorl_velocity_index",
    "exorl_velocity",
    "exorl_physics_features",
    "exorl_score_velocity",
    "exorl_goal_dims",
    "select_goal_states",
    "make_velocity_task",
    "make_exorl_goal_task",
    "get_exorl_task",
    "build_exorl_task_set",
    "build_exorl_tasks",
    "exorl_task_names",
    "list_exorl_task_sets",
    "make_exorl_env",
    "make_exorl_task_env",
    "reset_exorl_env",
    "exorl_encoder_states",
    "exorl_encoding_samples_for_task",
    "exorl_encoding_samples_from_env",
    # kitchen tasks
    "KITCHEN_ENV_NAME",
    "KITCHEN_MAX_EPISODE_STEPS",
    "KITCHEN_OBS_DIM",
    "KITCHEN_BONUS_THRESH",
    "KITCHEN_SUBTASKS",
    "KITCHEN_ELEMENT_GOALS",
    "KITCHEN_TASK_SETS",
    "KitchenSubtaskReward",
    "KitchenAllSubtasksReward",
    "KitchenTaskSpec",
    "kitchen_element_achieved",
    "kitchen_subtask_success",
    "kitchen_completed_elements",
    "kitchen_completion_fraction",
    "make_kitchen_subtask",
    "make_all_subtasks_task",
    "get_kitchen_task",
    "build_kitchen_task_set",
    "build_kitchen_tasks",
    "kitchen_task_names",
    "list_kitchen_task_sets",
    "make_kitchen_env",
    "make_kitchen_task_env",
    "reset_kitchen_env",
    "kitchen_encoder_states",
    "kitchen_encoding_samples_for_task",
    "kitchen_encoding_samples_from_env",
    # dispatch
    "DOMAIN_TASK_SETS",
    "domain_from_task_name",
    "build_tasks",
]
