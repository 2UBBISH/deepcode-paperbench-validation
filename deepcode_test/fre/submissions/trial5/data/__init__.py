"""
Data module for FRE (Functional Reward Encodings).

Provides offline dataset loading (D4RL, ExORL) and replay buffer
for on-the-fly reward computation during training.
"""

from .dataset import (
    OfflineDataset,
    load_dataset,
    load_d4rl_dataset,
    load_exorl_dataset,
    create_dataset_from_arrays,
)

from .replay_buffer import (
    ReplayBuffer,
    create_replay_buffer,
)

__all__ = [
    "OfflineDataset",
    "load_dataset",
    "load_d4rl_dataset",
    "load_exorl_dataset",
    "create_dataset_from_arrays",
    "ReplayBuffer",
    "create_replay_buffer",
]