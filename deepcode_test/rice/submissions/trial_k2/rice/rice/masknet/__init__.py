"""RICE mask-network (explanation) module.

This module exposes the MaskNet architecture, the masked training environment,
the PPO trainer for the mask network, and the intrinsic blinding reward used to
encourage sparse, high-fidelity critical-state explanations.
"""

from rice.masknet.intrinsic_reward import MaskIntrinsicReward, mask_reward
from rice.masknet.mask_network import (
    MaskNetwork,
    build_mask_network,
    match_target_mask_network,
)
from rice.masknet.mask_trainer import (
    MaskActorCritic,
    MaskTorchPolicy,
    MaskTrainer,
    train_mask_network,
)
from rice.masknet.masked_env import MaskedEnv

__all__ = [
    "MaskNetwork",
    "build_mask_network",
    "match_target_mask_network",
    "MaskedEnv",
    "MaskIntrinsicReward",
    "mask_reward",
    "MaskActorCritic",
    "MaskTorchPolicy",
    "MaskTrainer",
    "train_mask_network",
]
