"""Environment utilities for FRE experiments.

This package exposes dependency-tolerant wrappers and helpers for the three
evaluation domains used in the paper:

* AntMaze (D4RL)
* Kitchen (D4RL)
* DeepMind Control Suite (ExORL Walker/Cheetah)

All heavy imports (``gym``, ``d4rl``, ``dm_control``) are performed lazily inside
the respective submodules so that importing ``fre.envs`` never fails solely
because an optional MuJoCo/D4RL/DMC stack is unavailable.
"""

from __future__ import annotations

from fre.envs.antmaze import (
    ANTMAZE_XY_DIM,
    DEFAULT_ANTMAZE_ENV,
    DEFAULT_GOAL_THRESHOLD,
    AntMazeEnv,
    antmaze_goal_distance,
    antmaze_sparse_reward,
    get_antmaze_bounds,
    get_antmaze_xy,
    make_antmaze_env,
    sample_antmaze_goal,
)
from fre.envs.kitchen import (
    DEFAULT_KITCHEN_ENV,
    DEFAULT_KITCHEN_TASK,
    KITCHEN_SUBTASK_GOALS,
    KITCHEN_SUBTASK_OBS_INDICES,
    KITCHEN_TASK_NAMES,
    KITCHEN_TASKS,
    KitchenEnv,
    kitchen_goal_distance,
    kitchen_sparse_reward,
    kitchen_subtask_achieved,
    kitchen_subtask_distance,
    kitchen_subtask_reward,
    make_kitchen_env,
    sample_kitchen_goal,
)
from fre.envs.dmc import (
    DMCEnv,
    make_dmc_env,
    parse_dmc_env_name,
)

__all__ = [
    # AntMaze
    "AntMazeEnv",
    "make_antmaze_env",
    "get_antmaze_xy",
    "antmaze_goal_distance",
    "antmaze_sparse_reward",
    "sample_antmaze_goal",
    "get_antmaze_bounds",
    "DEFAULT_ANTMAZE_ENV",
    "ANTMAZE_XY_DIM",
    "DEFAULT_GOAL_THRESHOLD",
    # Kitchen
    "KitchenEnv",
    "make_kitchen_env",
    "kitchen_subtask_distance",
    "kitchen_subtask_achieved",
    "kitchen_subtask_reward",
    "kitchen_goal_distance",
    "kitchen_sparse_reward",
    "sample_kitchen_goal",
    "DEFAULT_KITCHEN_ENV",
    "DEFAULT_KITCHEN_TASK",
    "KITCHEN_TASK_NAMES",
    "KITCHEN_TASKS",
    "KITCHEN_SUBTASK_OBS_INDICES",
    "KITCHEN_SUBTASK_GOALS",
    # DMC / ExORL
    "DMCEnv",
    "make_dmc_env",
    "parse_dmc_env_name",
]
