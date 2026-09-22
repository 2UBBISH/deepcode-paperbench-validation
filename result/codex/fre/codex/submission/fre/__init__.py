"""Functional Reward Encoding (FRE).

Reference: Frans, Park, Abbeel, Levine. "Unsupervised Zero-Shot Reinforcement
Learning via Functional Reward Encodings", ICML 2024.

This package contains a from-scratch re-implementation of FRE:

    fre.reward_functions : prior distributions over random unsupervised rewards
    fre.encoder          : permutation-invariant transformer VAE encoder
    fre.decoder          : MLP reward decoder
    fre.fre              : the full FRE module (encoder + decoder + IB objective)
    fre.iql              : IQL agent whose Q/V/policy are conditioned on z
    fre.training         : strided encoder -> policy training loop
    fre.tasks            : downstream evaluation reward functions (AntMaze/ExORL/Kitchen)
"""

from fre.fre import FRE
from fre.encoder import TransformerEncoder
from fre.decoder import RewardDecoder
from fre.reward_functions import (
    RewardPrior,
    GoalReachingPrior,
    LinearPrior,
    MLPPrior,
    MixturePrior,
    PRIOR_REGISTRY,
    build_prior,
    discretize_reward,
)

__all__ = [
    "FRE",
    "TransformerEncoder",
    "RewardDecoder",
    "RewardPrior",
    "GoalReachingPrior",
    "LinearPrior",
    "MLPPrior",
    "MixturePrior",
    "PRIOR_REGISTRY",
    "build_prior",
    "discretize_reward",
]
