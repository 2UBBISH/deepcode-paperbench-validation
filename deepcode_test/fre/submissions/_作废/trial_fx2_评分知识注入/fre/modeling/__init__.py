"""Modeling components for Functional Reward Encodings (FRE).

This subpackage exposes the reward embedding, permutation-invariant transformer
encoder, reward decoder, and the full FRE variational autoencoder.
"""

from fre.modeling.reward_embedding import RewardEmbedding
from fre.modeling.transformer_encoder import TransformerEncoder
from fre.modeling.decoder import RewardDecoder
from fre.modeling.fre_vae import FREVAE, FREOutput

__all__ = [
    "RewardEmbedding",
    "TransformerEncoder",
    "RewardDecoder",
    "FREVAE",
    "FREOutput",
]
