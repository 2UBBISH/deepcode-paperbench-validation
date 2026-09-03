"""
models package for Functional Reward Encodings (FRE).

Provides the core neural network architectures:
- RewardEmbedding: Discretizes scalar rewards into learned embedding tokens.
- FREEncoder: Transformer-based VAE encoder for functional reward representation.
- FREDecoder: Feedforward decoder that predicts rewards from states and latent codes.
- IQLAgent: Implicit Q-Learning agent with Q, V, and policy networks conditioned on latent z.
- QNetwork, ValueNetwork, GaussianPolicy: Individual IQL sub-networks.
"""

from .reward_embedding import RewardEmbedding
from .fre_encoder import FREEncoder
from .fre_decoder import FREDecoder
from .iql_agent import IQLAgent, QNetwork, ValueNetwork, GaussianPolicy

__all__ = [
    "RewardEmbedding",
    "FREEncoder",
    "FREDecoder",
    "IQLAgent",
    "QNetwork",
    "ValueNetwork",
    "GaussianPolicy",
]