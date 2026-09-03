"""
FRE Encoder: Permutation-invariant transformer VAE encoder.

Encodes a set of K (state, reward) pairs into a latent vector z ~ N(μ, σ²I).
The encoder is permutation-invariant: no positional encoding, no causal mask,
and average pooling over output tokens ensures order invariance.

Architecture:
  - State s_i → Linear projection → d_model
  - Reward r_i → Discretized into bins → Embedding lookup → d_model
  - Token = Concat(s_emb, r_emb) → Linear → d_model
  - TransformerEncoder (N layers, M heads, no pos encoding, no causal mask)
  - Average pooling over tokens
  - Two linear heads: μ, log σ (both dim d_z)
"""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardEncoder(nn.Module):
    """
    Permutation-invariant transformer encoder for reward functions.

    Given K (state, reward) pairs sampled from an unknown reward function η,
    encodes them into a Gaussian latent representation z.

    Args:
        state_dim: Dimensionality of state space.
        d_model: Hidden dimension of transformer (default: 256).
        nhead: Number of attention heads (default: 4).
        num_layers: Number of transformer encoder layers (default: 3).
        d_z: Dimensionality of latent vector z (default: 64).
        num_reward_bins: Number of discretization bins for rewards (default: 50).
        reward_min: Minimum expected reward value (default: -1.0).
        reward_max: Maximum expected reward value (default: 1.0).
        dropout: Dropout rate in transformer layers (default: 0.1).
        ff_dim: Feedforward dimension in transformer; if None, uses 4 * d_model.
        use_sum_tokens: If True, sum state and reward embeddings instead of
            concatenating and projecting (default: False, uses concat+proj).
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        d_z: int = 64,
        num_reward_bins: int = 50,
        reward_min: float = -1.0,
        reward_max: float = 1.0,
        dropout: float = 0.1,
        ff_dim: Optional[int] = None,
        use_sum_tokens: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.d_z = d_z
        self.num_reward_bins = num_reward_bins
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.use_sum_tokens = use_sum_tokens

        # State projection: state_dim → d_model
        self.state_proj = nn.Linear(state_dim, d_model)

        # Reward discretization: scalar reward → bin index → embedding
        self.reward_embed = nn.Embedding(num_reward_bins, d_model)

        # Token combination
        if use_sum_tokens:
            # Summation: token = s_emb + r_emb (both d_model)
            self.token_proj = None
        else:
            # Concatenation + projection: [s_emb; r_emb] (2*d_model) → d_model
            self.token_proj = nn.Linear(2 * d_model, d_model)

        # Transformer encoder (no positional encoding, no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim if ff_dim is not None else 4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output heads: d_model → d_z for μ and log σ
        self.mu_head = nn.Linear(d_model, d_z)
        self.logvar_head = nn.Linear(d_model, d_z)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Xavier uniform and embedding with normal."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if "state_proj" in name or "token_proj" in name or "mu_head" in name or "logvar_head" in name:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Embedding init
        nn.init.normal_(self.reward_embed.weight, mean=0.0, std=1.0 / np.sqrt(self.d_model))

    def _discretize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Map continuous reward values to discrete bin indices.

        Args:
            rewards: Tensor of shape (...,) with values in [reward_min, reward_max].

        Returns:
            LongTensor of bin indices, same shape as rewards, in [0, num_reward_bins-1].
        """
        # Clamp to valid range
        rewards = torch.clamp(rewards, self.reward_min, self.reward_max)
        # Normalize to [0, 1]
        normalized = (rewards - self.reward_min) / (self.reward_max - self.reward_min + 1e-8)
        # Map to bin index
        bin_indices = (normalized * (self.num_reward_bins - 1)).long()
        # Clamp to ensure valid indices
        bin_indices = torch.clamp(bin_indices, 0, self.num_reward_bins - 1)
        return bin_indices

    def forward(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a set of (state, reward) pairs into latent distribution parameters.

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K).

        Returns:
            mu: Tensor of shape (batch_size, d_z) — mean of latent Gaussian.
            logvar: Tensor of shape (batch_size, d_z) — log-variance of latent Gaussian.
        """
        batch_size, K, _ = states.shape

        # 1. Project states
        s_emb = self.state_proj(states)  # (batch, K, d_model)

        # 2. Discretize and embed rewards
        reward_bins = self._discretize_rewards(rewards)  # (batch, K)
        r_emb = self.reward_embed(reward_bins)  # (batch, K, d_model)

        # 3. Form tokens
        if self.use_sum_tokens:
            tokens = s_emb + r_emb  # (batch, K, d_model)
        else:
            tokens = torch.cat([s_emb, r_emb], dim=-1)  # (batch, K, 2*d_model)
            tokens = self.token_proj(tokens)  # (batch, K, d_model)

        # 4. Transformer encoder (no src_key_padding_mask → no causal mask)
        out = self.transformer(tokens)  # (batch, K, d_model)

        # 5. Average pooling over K tokens (permutation-invariant aggregation)
        pooled = out.mean(dim=1)  # (batch, d_model)

        # 6. Output heads
        mu = self.mu_head(pooled)  # (batch, d_z)
        logvar = self.logvar_head(pooled)  # (batch, d_z)

        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample z from N(mu, σ²I) using the reparameterization trick.

        Args:
            mu: Mean tensor of shape (batch_size, d_z).
            logvar: Log-variance tensor of shape (batch_size, d_z).

        Returns:
            z: Sampled latent vector of shape (batch_size, d_z).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_and_sample(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method: encode and sample z in one call.

        Args:
            states: (batch_size, K, state_dim)
            rewards: (batch_size, K)

        Returns:
            mu: (batch_size, d_z)
            logvar: (batch_size, d_z)
            z: (batch_size, d_z)
        """
        mu, logvar = self.forward(states, rewards)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def encode_deterministic(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode to deterministic z (using μ only, no sampling).
        Useful for evaluation.

        Args:
            states: (batch_size, K, state_dim)
            rewards: (batch_size, K)

        Returns:
            z: (batch_size, d_z) — the mean μ.
        """
        mu, _ = self.forward(states, rewards)
        return mu


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence KL(N(μ, σ²I) || N(0, I)).

    Args:
        mu: Mean tensor of shape (batch_size, d_z).
        logvar: Log-variance tensor of shape (batch_size, d_z).

    Returns:
        KL divergence per sample, shape (batch_size,), averaged over latent dims.
    """
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl


def vae_loss(
    recon_loss: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute the full VAE loss: reconstruction + β * KL divergence.

    Args:
        recon_loss: Scalar reconstruction loss (e.g., MSE).
        mu: Mean tensor (batch_size, d_z).
        logvar: Log-variance tensor (batch_size, d_z).
        beta: Weight for KL term (default: 1.0).

    Returns:
        Total VAE loss (scalar).
    """
    kl = kl_divergence(mu, logvar).mean()
    return recon_loss + beta * kl