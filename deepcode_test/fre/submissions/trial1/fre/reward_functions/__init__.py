"""
Package initializer for fre.reward_functions.

Re-exports all reward function classes and utilities for convenient imports.
"""

from fre.reward_functions.base import RewardFunction
from fre.reward_functions.singleton import SingletonRewardFunction
from fre.reward_functions.linear import LinearRewardFunction
from fre.reward_functions.mlp import MLPRewardFunction
from fre.reward_functions.mixture import MixtureRewardFunction
from fre.reward_functions.eval_rewards import (
    AntMazeGoalReward,
    AntMazeDirectionalReward,
    AntMazeRandomSimplexReward,
    AntMazePathReward,
    ExORLGoalReward,
    ExORLVelocityReward,
    KitchenSubtaskReward,
    KitchenAllSubtasksReward,
    create_eval_reward_function,
    get_eval_tasks,
    ANTMAZE_EVAL_TASKS,
    EXORL_WALKER_EVAL_TASKS,
    EXORL_CHEETAH_EVAL_TASKS,
    KITCHEN_EVAL_TASKS,
    test_eval_rewards,
)

__all__ = [
    # Base
    "RewardFunction",
    # Prior reward functions
    "SingletonRewardFunction",
    "LinearRewardFunction",
    "MLPRewardFunction",
    "MixtureRewardFunction",
    # Evaluation reward functions
    "AntMazeGoalReward",
    "AntMazeDirectionalReward",
    "AntMazeRandomSimplexReward",
    "AntMazePathReward",
    "ExORLGoalReward",
    "ExORLVelocityReward",
    "KitchenSubtaskReward",
    "KitchenAllSubtasksReward",
    # Utility functions
    "create_eval_reward_function",
    "get_eval_tasks",
    # Task lists
    "ANTMAZE_EVAL_TASKS",
    "EXORL_WALKER_EVAL_TASKS",
    "EXORL_CHEETAH_EVAL_TASKS",
    "KITCHEN_EVAL_TASKS",
    # Test
    "test_eval_rewards",
]