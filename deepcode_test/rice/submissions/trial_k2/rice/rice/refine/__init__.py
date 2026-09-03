"""RICE refinement module.

This module implements the RICE policy-refinement stage: mixed initial-state
resets from a critical-state buffer, an RND exploration bonus, and PPO training
of the refined policy.
"""

from rice.refine.critical_state_buffer import (
    CriticalState,
    CriticalStateBuffer,
    build_critical_buffer_from_trajectories,
)
from rice.refine.mixed_reset_env import (
    MixedResetEnv,
    default_restore_state,
    default_fallback_reset,
    make_mixed_reset_env,
)
from rice.refine.rnd_bonus import (
    RNDBonus,
    RNDNetwork,
    RNDRewardWrapper,
    RunningMeanStd,
    build_rnd_networks,
    make_rnd_bonus,
)
from rice.refine.refine_trainer import (
    RefineTrainer,
    refine_policy,
)

__all__ = [
    "CriticalState",
    "CriticalStateBuffer",
    "build_critical_buffer_from_trajectories",
    "MixedResetEnv",
    "default_restore_state",
    "default_fallback_reset",
    "make_mixed_reset_env",
    "RNDBonus",
    "RNDNetwork",
    "RNDRewardWrapper",
    "RunningMeanStd",
    "build_rnd_networks",
    "make_rnd_bonus",
    "RefineTrainer",
    "refine_policy",
]
