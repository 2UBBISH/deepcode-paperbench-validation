"""Offline data pipeline for FRE.

This package provides dependency-light offline dataset loading, trajectory
indexing and domain specific preprocessing for the three FRE domains:

* AntMaze (D4RL ``antmaze-large-diverse-v2``) with 32-bin X/Y discretization.
* ExORL (RND walker / cheetah) with physics augmentation (encoder stream only).
* Kitchen (D4RL ``kitchen-mixed-v0``).

The heavy RL/data dependencies (``d4rl``, ``gym``, ``dm_control``, ``h5py``)
are imported lazily inside the loaders so that importing :mod:`fre.data` stays
cheap and works in environments where those packages are not installed.
"""

from __future__ import annotations

from .dataset import (
    ANTMAZE_DATASET,
    EXORL_DATASETS,
    KITCHEN_DATASET,
    DATASET_REGISTRY,
    OfflineDataset,
    TransitionBatch,
    build_trajectory_index,
    future_indices,
    load_antmaze_dataset,
    load_dataset,
    load_d4rl_dataset,
    load_exorl_dataset,
    load_kitchen_dataset,
    load_npz_dataset,
    synthetic_dataset,
)
from .preprocessing import (
    ANTMAZE_XY_INDICES,
    EXORL_AUGMENT_DIM,
    EXORL_DM_CONTROL_DOMAINS,
    EXORL_PHYSICS_FIELDS,
    NUM_XY_BINS,
    antmaze_xy_bins,
    augment_exorl_physics,
    compute_state_normalization,
    count_xy_bins,
    denormalize_states,
    discretize_antmaze_xy,
    euclidean_goal_distance,
    exorl_augment_dim,
    exorl_domain,
    exorl_goal_state_dims,
    exorl_physics_augmentation,
    goal_reached_mask,
    normalize_states,
    preprocess_antmaze,
    preprocess_exorl,
    scale_by_std,
)

__all__ = [
    # dataset
    "OfflineDataset",
    "TransitionBatch",
    "DATASET_REGISTRY",
    "ANTMAZE_DATASET",
    "KITCHEN_DATASET",
    "EXORL_DATASETS",
    "load_dataset",
    "load_d4rl_dataset",
    "load_antmaze_dataset",
    "load_kitchen_dataset",
    "load_exorl_dataset",
    "load_npz_dataset",
    "synthetic_dataset",
    "build_trajectory_index",
    "future_indices",
    # preprocessing
    "NUM_XY_BINS",
    "ANTMAZE_XY_INDICES",
    "EXORL_PHYSICS_FIELDS",
    "EXORL_AUGMENT_DIM",
    "EXORL_DM_CONTROL_DOMAINS",
    "antmaze_xy_bins",
    "discretize_antmaze_xy",
    "count_xy_bins",
    "exorl_domain",
    "exorl_augment_dim",
    "exorl_goal_state_dims",
    "exorl_physics_augmentation",
    "augment_exorl_physics",
    "compute_state_normalization",
    "normalize_states",
    "denormalize_states",
    "scale_by_std",
    "euclidean_goal_distance",
    "goal_reached_mask",
    "preprocess_antmaze",
    "preprocess_exorl",
]
