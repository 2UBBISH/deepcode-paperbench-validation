"""Environment utilities and task-specific wrappers for RICE."""

from rice.envs.mujoco_wrappers import (
    RunningObsNormalizer,
    SparseRewardWrapper,
    SparseHopperWrapper,
    SparseWalker2dWrapper,
    SparseHalfCheetahWrapper,
    make_mujoco_env,
    make_sparse_mujoco_env,
    is_mujoco_env_id,
    should_normalize_obs,
)

__all__ = [
    "RunningObsNormalizer",
    "SparseRewardWrapper",
    "SparseHopperWrapper",
    "SparseWalker2dWrapper",
    "SparseHalfCheetahWrapper",
    "make_mujoco_env",
    "make_sparse_mujoco_env",
    "is_mujoco_env_id",
    "should_normalize_obs",
]
