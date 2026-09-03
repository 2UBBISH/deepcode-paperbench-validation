"""Permutation-invariant transformer VAE encoder for Functional Reward Encodings.

The encoder receives a set of K tokens produced by
:class:`fre.reward_embedding.RewardEmbedding`. Each token represents one
``(state, reward)`` pair. The set is encoded by a standard Transformer encoder
with *no* positional encoding and *no* causal mask, making the representation
invariant to permutations of the input set. The averaged final-layer token
representation is projected to a Gaussian posterior over the latent reward
code ``z``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class FREEncoder(nn.Module):
    """Transformer VAE encoder mapping ``(B, K, d_model)`` tokens to ``z``.

    Parameters
    ----------
    d_model:
        Transformer token dimension. Must match the token dimension produced
        by :class:`fre.reward_embedding.RewardEmbedding`.
    nhead:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers.
    latent_dim:
        Dimensionality of the latent reward code ``z``.
    dim_feedforward:
        Feed-forward hidden dimension inside each Transformer layer.
    dropout:
        Dropout probability used by the Transformer layers.
    activation:
        Activation function for Transformer feed-forward blocks.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        latent_dim: int = 128,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=None,
        )

        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        self.d_model = d_model
        self.latent_dim = latent_dim

    def forward(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a set of tokens and return posterior parameters and a sample.

        Parameters
        ----------
        tokens:
            Tensor of shape ``(B, K, d_model)``. The leading dimensions are
            arbitrary; ``B`` may be a single batch dimension or a flattened
            batch of reward functions.
        mask:
            Optional boolean mask of shape ``(B, K)`` where ``True`` denotes a
            valid token. If provided, it is converted to a Transformer source
            key-padding mask.

        Returns
        -------
        mu:
            Posterior mean, shape ``(*B, latent_dim)``.
        logvar:
            Posterior log variance, shape ``(*B, latent_dim)``.
        z:
            Reparameterized latent sample, shape ``(*B, latent_dim)``.
        """
        # src_key_padding_mask expects shape (N, S); for batched inputs use
        # the same mask if it has shape (B, K).
        if mask is not None and mask.ndim == tokens.ndim - 1:
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer(
            tokens, src_key_padding_mask=src_key_padding_mask
        )

        # Average over the sequence (token) dimension. This makes the
        # representation permutation invariant.
        if mask is not None:
            valid_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(
                encoded.dtype
            )
            h = (encoded * mask.unsqueeze(-1)).sum(dim=-2) / valid_counts
        else:
            h = encoded.mean(dim=-2)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z`` using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl_divergence(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute unit-Gaussian KL divergence per latent code.

        Returns a tensor of shape ``(*B,)``. The mean over this tensor is the
        scalar KL term used in the FRE VAE objective.
        """
        return 0.5 * (
            -1.0 - logvar + mu.pow(2) + logvar.exp()
        ).sum(dim=-1)
