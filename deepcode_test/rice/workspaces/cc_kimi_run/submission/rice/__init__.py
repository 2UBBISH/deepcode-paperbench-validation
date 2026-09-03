"""RICE: Refining Reinforcement Learning with Explanation.

This package implements the core algorithms and baselines from the RICE paper.
"""
from rice.mask_network import MaskNetwork, MaskNetworkTrainer
from rice.refining import RICERefiningEnv, refine_rice
from rice.rnd import RNDBonus
from rice.explanations import (
    ExplanationMethod,
    MaskExplanation,
    RandomExplanation,
    StateMaskExplanation,
)
from rice.fidelity import compute_fidelity_score, sample_trajectory
from rice.baselines import ppo_finetune, statemask_r_finetune, jsrl_finetune
from rice.env_utils import make_env, StateResetWrapper, NormalizeObservationWrapper

__all__ = [
    "MaskNetwork",
    "MaskNetworkTrainer",
    "RICERefiningEnv",
    "refine_rice",
    "RNDBonus",
    "ExplanationMethod",
    "MaskExplanation",
    "RandomExplanation",
    "StateMaskExplanation",
    "compute_fidelity_score",
    "sample_trajectory",
    "ppo_finetune",
    "statemask_r_finetune",
    "jsrl_finetune",
    "make_env",
    "StateResetWrapper",
    "NormalizeObservationWrapper",
]
