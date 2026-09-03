"""
FRE Models Package

Contains the core neural network architectures:
- FREEncoder: Permutation-invariant transformer VAE encoder
- RewardDecoder: Feedforward reward decoder
- IQL networks: Q, V, and policy networks conditioned on latent z
"""

from fre.models.encoder import FREEncoder, RewardDiscretizer, test_permutation_invariance
from fre.models.decoder import RewardDecoder
from fre.models.iql import (
    QNetwork,
    ValueNetwork,
    GaussianPolicy,
    IQLNetworks,
    test_iql_networks,
)

__all__ = [
    "FREEncoder",
    "RewardDiscretizer",
    "RewardDecoder",
    "QNetwork",
    "ValueNetwork",
    "GaussianPolicy",
    "IQLNetworks",
    "test_permutation_invariance",
    "test_iql_networks",
]