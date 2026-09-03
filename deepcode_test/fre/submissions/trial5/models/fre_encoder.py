"""
FRE Encoder: Transformer-based Variational Auto-Encoder for functional reward encoding.

Encodes a set of (state, reward) pairs into a latent Gaussian distribution
pθ(z | {(s_i, η(s_i))}) using a permutation-invariant transformer architecture.

Architecture:
- Input: K states + K reward embeddings → concatenated tokens
- Permutation-invariant Transformer (no positional encoding, no causal mask)
- Mean pooling over output tokens → single vector h
- Two linear heads: μ and log σ for Gaussian latent distribution
- Reparameterization trick for sampling z
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reward_embedding import RewardEmbedding


class FREEncoder(nn.Module):
    """
    Transformer-based encoder that maps a set of (state, reward) pairs to a
    latent Gaussian distribution over z.

    The encoder is permutation-invariant: no positional encodings and no
    causal masking are used, treating the input as an unordered set.

    Args:
        state_dim: Dimensionality of state vectors.
        embed_dim: Dimensionality of reward embeddings (and transformer hidden dim).
        latent_dim: Dimensionality of latent variable z.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate for transformer layers.
        num_bins: Number of reward discretization bins (for RewardEmbedding).
        reward_min: Optional fixed minimum reward value for binning.
        reward_max: Optional fixed maximum reward value for binning.
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_bins: int = 64,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Reward embedding module: discretizes scalar rewards into learned embeddings
        self.reward_embedding = RewardEmbedding(
            num_bins=num_bins,
            embed_dim=embed_dim,
            reward_min=reward_min,
            reward_max=reward_max,
        )

        # Input projection: maps concatenated [state, reward_embedding] to embed_dim
        # state_dim + embed_dim → embed_dim
        self.input_projection = nn.Linear(state_dim + embed_dim, embed_dim)

        # Permutation-invariant Transformer encoder (no positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Use batch_first for convenience
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer normalization after transformer
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Projection heads for latent distribution parameters
        self.mu_head = nn.Linear(embed_dim, latent_dim)
        self.logvar_head = nn.Linear(embed_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in [self.input_projection, self.mu_head, self.logvar_head]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a set of (state, reward) pairs into latent distribution parameters
        and sample z via reparameterization.

        Args:
            states: Tensor of shape (batch_size, K, state_dim) — K encoding states.
            rewards: Tensor of shape (batch_size, K) — scalar rewards for each state.

        Returns:
            z: Sampled latent vector, shape (batch_size, latent_dim).
            mu: Mean of latent Gaussian, shape (batch_size, latent_dim).
            logvar: Log-variance of latent Gaussian, shape (batch_size, latent_dim).
        """
        batch_size, K, _ = states.shape

        # Embed rewards: (batch_size, K) → (batch_size, K, embed_dim)
        # Reshape rewards to 2D for embedding, then reshape back
        rewards_flat = rewards.reshape(-1)  # (batch_size * K,)
        reward_embeds_flat, _ = self.reward_embedding(rewards_flat)
        reward_embeds = reward_embeds_flat.reshape(batch_size, K, self.embed_dim)

        # Concatenate states and reward embeddings: (batch_size, K, state_dim + embed_dim)
        tokens = torch.cat([states, reward_embeds], dim=-1)

        # Project to embed_dim: (batch_size, K, embed_dim)
        tokens = self.input_projection(tokens)

        # Pass through permutation-invariant transformer
        # No positional encoding, no causal mask → treats tokens as a set
        tokens = self.transformer(tokens)

        # Layer normalization
        tokens = self.layer_norm(tokens)

        # Mean pooling over K tokens → (batch_size, embed_dim)
        h = tokens.mean(dim=1)

        # Compute latent distribution parameters
        mu = self.mu_head(h)          # (batch_size, latent_dim)
        logvar = self.logvar_head(h)  # (batch_size, latent_dim)

        # Sample z via reparameterization trick
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + epsilon * exp(0.5 * logvar).

        Args:
            mu: Mean tensor, shape (batch_size, latent_dim).
            logvar: Log-variance tensor, shape (batch_size, latent_dim).

        Returns:
            z: Sampled latent vector, shape (batch_size, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During evaluation, just return the mean
            return mu

    def encode_deterministic(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode without sampling (returns mu only). Useful for evaluation.

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K).

        Returns:
            z: Deterministic latent vector (mu), shape (batch_size, latent_dim).
        """
        _, mu, _ = self.forward(states, rewards)
        return mu

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between the encoded Gaussian N(mu, sigma^2)
        and the standard normal prior N(0, I).

        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Args:
            mu: Mean tensor, shape (batch_size, latent_dim).
            logvar: Log-variance tensor, shape (batch_size, latent_dim).

        Returns:
            kl: KL divergence per sample, shape (batch_size,). Mean over latent dim.
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl

    def get_reward_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the current reward range used by the reward embedding."""
        return self.reward_embedding.reward_min, self.reward_embedding.reward_max

    def set_reward_range(self, reward_min: float, reward_max: float):
        """Set a fixed reward range for the reward embedding."""
        self.reward_embedding.set_reward_range(reward_min, reward_max)