"""
Forward-Backward (FB) and Successor Features (SF) baseline integrations.

Per the addendum, these baselines are trained and evaluated using the
facebookresearch/controllable_agent codebase.

This module provides wrapper classes that interface with that codebase
and defines the evaluation tasks used in the paper.

Key points (from addendum):
- All SF/FB ExORL experiments use the RND dataset
- ICM features are used for SF
- No changes to the controllable_agent codebase needed for training
- Custom reward functions were introduced into pre-existing environments for evaluation
- FB and SF require 5120 reward samples during evaluation (vs 32 for FRE)
"""

import numpy as np
import torch
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MethodType(Enum):
    FB = "fb"
    SF = "sf"


@dataclass
class FBConfig:
    """Configuration for FB baseline training/evaluation."""
    # Dataset
    replay_buffer_path: str = ""
    # Training
    num_train_steps: int = 1_000_000
    batch_size: int = 1024
    # Architecture
    hidden_dim: int = 1024
    representation_dim: int = 50
    # SF-specific
    use_icm_features: bool = False
    # Evaluation
    num_encoding_samples: int = 5120  # FB/SF use 5120 vs FRE's 32
    num_eval_episodes: int = 20
    # Environment
    env_name: str = "antmaze"


class FBWrapper:
    """
    Wrapper around facebookresearch/controllable_agent for FB.

    Provides the interface for training FB policies and evaluating them
    on the paper's custom evaluation tasks.

    The controllable_agent codebase handles:
    - Building the replay buffer from downloaded RND dataset
    - Training the FB agent (forward-backward representation learning)
    - Logging evaluation numbers during training

    This wrapper defines the evaluation task interface: given a trained
    FB agent, evaluate it on paper-specified tasks with custom reward functions.
    """

    def __init__(self, config: FBConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        # The actual FB model would be loaded from controllable_agent checkpoint
        self._agent = None  # placeholder for loaded model

    def train(
        self,
        dataset_path: str,
        env_name: str,
        log_dir: str,
    ):
        """
        Train FB agent using controllable_agent.

        Steps (per addendum):
        1. Download offline RND dataset
        2. Construct replay buffer using code from repo README
        3. Run training command
        4. Evaluation numbers are logged during training
        """
        # This is a documentation-level implementation — the actual
        # training is performed by the controllable_agent codebase.
        # The paper authors didn't change anything in that codebase.
        raise NotImplementedError(
            "FB training uses facebookresearch/controllable_agent. "
            "Import and call that codebase's training pipeline."
        )

    def evaluate(
        self,
        task_reward_fn: Callable[[torch.Tensor], torch.Tensor],
        num_encoding_samples: int = 5120,
        num_episodes: int = 20,
        max_steps: int = 500,
        env_step_fn: Callable = None,
        initial_state_fn: Callable = None,
    ) -> Dict[str, float]:
        """
        Evaluate FB agent on a downstream task.

        FB uses linear regression on 5120 reward samples to adapt to
        the downstream task at test time (vs FRE's learned encoder with 32 samples).
        """
        # FB test-time adaptation:
        # 1. Sample 5120 (state, reward) pairs from the task reward function
        # 2. Solve w* = argmin_w E[(z(s)^T w - r(s))^2] via linear regression
        #    where z(s) is the FB forward representation
        # 3. π(a|s) = π_z(a|s; w*) (policy conditioned on w*)
        raise NotImplementedError(
            "FB evaluation uses facebookresearch/controllable_agent. "
            "Import and call that codebase's evaluation functions."
        )


class SFWrapper:
    """
    Wrapper around facebookresearch/controllable_agent for SF.

    Key differences from FB:
    - Uses ICM features (Pathak et al., 2017) as state features
    - These are pre-trained features that approximate a universal
      family of reward functions as linear combinations
    - At test time, linear regression finds the best feature combination
      for the downstream reward function
    """

    def __init__(self, config: FBConfig, device: str = "cpu"):
        self.config = config
        self.config.use_icm_features = True
        self.device = torch.device(device)
        self._agent = None

    def train(self, dataset_path: str, env_name: str, log_dir: str):
        """
        Train SF agent using controllable_agent with ICM features.

        Per addendum: ICM features are used (reported as strongest method
        on ExORL Walker and Cheetah tasks).
        """
        raise NotImplementedError(
            "SF training uses facebookresearch/controllable_agent with ICM features. "
            "Import and call that codebase's training pipeline."
        )

    def evaluate(
        self,
        task_reward_fn: Callable,
        num_encoding_samples: int = 5120,
        num_episodes: int = 20,
        max_steps: int = 500,
        env_step_fn: Callable = None,
        initial_state_fn: Callable = None,
    ) -> Dict[str, float]:
        """
        Evaluate SF agent on a downstream task.

        SF uses linear regression in the learned feature space,
        same as FB but with ICM features instead of learned representations.
        """
        raise NotImplementedError(
            "SF evaluation uses facebookresearch/controllable_agent. "
            "Import and call that codebase's evaluation functions."
        )


# ---------- Custom Evaluation Tasks for FB/SF ----------

def make_antmaze_custom_reward_fb_sf(
    task_name: str,
    env,
) -> Callable:
    """
    Create a custom reward function for an AntMaze evaluation task,
    compatible with the controllable_agent codebase.

    Per addendum: "the authors introduced a custom reward function into
    the pre-existing environments (e.g., antmaze, walker, cheetah, kitchen)
    that replaced the default reward with their custom rewards."
    """
    # These custom reward functions modify the environment's reward
    # computation to match the paper's evaluation tasks.
    raise NotImplementedError(
        "Custom reward functions for FB/SF evaluation are defined in the "
        "controllable_agent codebase's environment wrappers."
    )


# ---------- Task Definitions for Zero-Shot RL Paper ----------

ANTMAZE_EVAL_TASKS = [
    "ant-goal-reaching",   # 5 fixed goal locations
    "ant-directional",     # 4 directional velocity tasks
    "ant-random-simplex",  # 5 seeded noise-based tasks
    "ant-path-loop",       # navigating in a loop
    "ant-path-edges",      # navigating along edges
    "ant-path-center",     # navigating along central corridor
]

EXORL_WALKER_EVAL_TASKS = [
    "exorl-walker-goals",    # 5 goal-reaching tasks
    "exorl-walker-velocity", # 4 velocity tasks (0.1, 1, 4, 8)
]

EXORL_CHEETAH_EVAL_TASKS = [
    "exorl-cheetah-goals",    # 5 goal-reaching tasks
    "exorl-cheetah-velocity", # run, run-backwards, walk, walk-backwards
]

KITCHEN_EVAL_TASKS = [
    "kitchen-complete",  # all 7 subtasks
]