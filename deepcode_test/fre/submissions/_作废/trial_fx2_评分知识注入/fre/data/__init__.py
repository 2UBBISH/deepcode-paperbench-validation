"""Data loading and reward-function sampling utilities for FRE.

This package exposes a unified :class:`OfflineDataset` wrapper together with
D4RL/ExORL loaders and the prior reward-function mixture used by FRE.
"""

from fre.data.dataset import (
    Dataset,
    Episode,
    OfflineDataset,
    TransitionBatch,
)
from fre.data.d4rl_loader import (
    load_antmaze_dataset,
    load_d4rl_dataset,
    load_d4rl_dataset_and_env,
    load_d4rl_env,
    load_kitchen_dataset,
)
from fre.data.exorl_loader import (
    load_cheetah_dataset,
    load_exorl_dataset,
    load_exorl_dataset_and_env,
    load_walker_dataset,
)
from fre.data.reward_sampler import (
    LinearRewardFunction,
    MLPRewardFunction,
    RewardFunction,
    SingletonRewardFunction,
    sample_reward,
    sample_rewards_batch,
)

__all__ = [
    # dataset primitives
    "OfflineDataset",
    "Dataset",
    "TransitionBatch",
    "Episode",
    # D4RL loaders
    "load_d4rl_dataset",
    "load_d4rl_dataset_and_env",
    "load_d4rl_env",
    "load_antmaze_dataset",
    "load_kitchen_dataset",
    # ExORL loaders
    "load_exorl_dataset",
    "load_exorl_dataset_and_env",
    "load_walker_dataset",
    "load_cheetah_dataset",
    # reward prior
    "RewardFunction",
    "SingletonRewardFunction",
    "LinearRewardFunction",
    "MLPRewardFunction",
    "sample_reward",
    "sample_rewards_batch",
]
