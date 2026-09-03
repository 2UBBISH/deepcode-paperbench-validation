"""
RICE: Refining via Critical State Explanation

A method to refine reinforcement learning agents by training a mask network
to identify critical states (explanation), then using those states to construct
a mixed initial state distribution for further training with an exploration
bonus (RND), yielding improved policy performance.
"""

__version__ = "0.1.0"
__author__ = "RICE Implementation"

from rice.mask_net import MaskNetwork, train_mask_network
from rice.rnd import RNDModule
from rice.refine import RICERefine
from rice.utils import (
    collect_trajectories,
    compute_gae,
    compute_returns,
    save_state_dict,
    load_state_dict,
)
from rice.env_wrappers import StateSaveWrapper

__all__ = [
    "MaskNetwork",
    "train_mask_network",
    "RNDModule",
    "RICERefine",
    "collect_trajectories",
    "compute_gae",
    "compute_returns",
    "save_state_dict",
    "load_state_dict",
    "StateSaveWrapper",
]