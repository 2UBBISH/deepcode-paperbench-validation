"""
Permutation-Invariant Transformer Encoder for Functional Reward Encodings (FRE).

Encodes a set of (state, reward) pairs into a latent Gaussian distribution
p_θ(z | {(s_i, η(s_i))}) using a transformer architecture without positional
encodings or causal masking, treating inputs as an unordered set.

Architecture:
    1. Reward discretization: scalar reward bucketed into B bins, each with a
       learned embedding vector.
    2. State concatenated with reward embedding, projected to d_model.
    3. Stack of L transformer encoder layers with H attention heads.
    4. Average pooling over output tokens -> summary vector h.
    5. Two linear heads: μ and log_σ for Gaussian latent distribution.
    6. Reparameterization trick for sampling z.
"""

from typing import Tuple, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardDiscretizer(nn.Module):
    """
    Discretizes scalar rewards into bins and maps each bin to a learned embedding.

    The reward range is divided into B equal-width bins. Each bin corresponds
    to a learned embedding vector of dimension d_emb. Rewards outside the range
    are clamped to the nearest bin.
    """

    def __init__(
        self,
        num_bins: int = 100,
        embedding_dim: int = 64,
        reward_min: float = -10.0,
        reward_max: float = 10.0,
    ):
        """
        Args:
            num_bins: Number of discretization bins (B).
            embedding_dim: Dimension of each bin's embedding vector (d_emb).
            reward_min: Minimum expected reward value.
            reward_max: Maximum expected reward value.
        """
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.reward_min = reward_min
        self.reward_max = reward_max

        # Learned embedding table: (num_bins, embedding_dim)
        self.embeddings = nn.Parameter(
            torch.randn(num_bins, embedding_dim) * 0.02
        )

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar rewards to embedding vectors.

        Args:
            rewards: Tensor of shape (batch_size, num_states) or (batch_size,)
                     containing scalar reward values.

        Returns:
            Embedding tensor of shape (..., embedding_dim).
        """
        # Clamp rewards to [reward_min, reward_max]
        rewards_clamped = torch.clamp(rewards, self.reward_min, self.reward_max)

        # Compute bin indices: 0 to num_bins-1
        bin_width = (self.reward_max - self.reward_min) / self.num_bins
        indices = ((rewards_clamped - self.reward_min) / bin_width).long()
        # Clamp indices to valid range (edge case for reward == reward_max)
        indices = torch.clamp(indices, 0, self.num_bins - 1)

        # Lookup embeddings
        # embeddings shape: (num_bins, embedding_dim)
        # indices shape: (...)
        # output shape: (..., embedding_dim)
        return self.embeddings[indices]


class FREEncoder(nn.Module):
    """
    Permutation-invariant transformer encoder for FRE.

    Takes a set of K (state, reward) pairs and outputs the parameters
    (μ, log_σ) of a Gaussian latent distribution over z, along with
    a sampled latent vector z.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 1024,
        d_emb: int = 64,
        num_bins: int = 100,
        reward_min: float = -10.0,
        reward_max: float = 10.0,
        dropout: float = 0.0,
        max_num_states: int = 32,
    ):
        """
        Args:
            state_dim: Dimension of state vectors (d_s).
            latent_dim: Dimension of latent variable z (d_z).
            d_model: Transformer hidden dimension.
            num_layers: Number of transformer encoder layers (L).
            num_heads: Number of attention heads (H).
            d_ff: Feedforward hidden dimension in transformer.
            d_emb: Dimension of reward embedding vectors.
            num_bins: Number of reward discretization bins (B).
            reward_min: Minimum expected reward value.
            reward_max: Maximum expected reward value.
            dropout: Dropout rate for transformer layers.
            max_num_states: Maximum number of encoding states (K), used for
                            optional padding mask.
        """
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_emb = d_emb
        self.num_bins = num_bins
        self.max_num_states = max_num_states

        # Reward discretizer and embedding
        self.reward_discretizer = RewardDiscretizer(
            num_bins=num_bins,
            embedding_dim=d_emb,
            reward_min=reward_min,
            reward_max=reward_max,
        )

        # Input projection: [state; reward_embedding] -> d_model
        self.input_projection = nn.Linear(state_dim + d_emb, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Use (batch, seq, feature) format
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer normalization for the pooled output
        self.layer_norm = nn.LayerNorm(d_model)

        # Output heads for Gaussian parameters
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        # Small constant for numerical stability in log variance
        self.logvar_min = -10.0
        self.logvar_max = 10.0

    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_distribution: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode state-reward pairs into latent distribution and sample z.

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K) — scalar rewards.
            mask: Optional boolean mask of shape (batch_size, K), True for
                  valid (non-padding) positions. If None, all positions are
                  treated as valid.
            return_distribution: If True, also return μ and log_σ.

        Returns:
            z: Sampled latent vector of shape (batch_size, latent_dim).
            mu: Mean of latent Gaussian, shape (batch_size, latent_dim).
            logvar: Log variance of latent Gaussian, shape (batch_size, latent_dim).
        """
        batch_size, K, _ = states.shape

        # 1. Get reward embeddings
        # rewards shape: (batch_size, K)
        reward_emb = self.reward_discretizer(rewards)  # (batch_size, K, d_emb)

        # 2. Concatenate state and reward embedding
        # states: (batch_size, K, state_dim), reward_emb: (batch_size, K, d_emb)
        combined = torch.cat([states, reward_emb], dim=-1)  # (batch_size, K, state_dim + d_emb)

        # 3. Project to d_model
        tokens = self.input_projection(combined)  # (batch_size, K, d_model)

        # 4. Transformer encoding (no positional encoding, no causal mask)
        # src_key_padding_mask: (batch_size, K) with True for positions to ignore
        if mask is not None:
            # Transformer expects True for positions to mask (ignore)
            src_key_padding_mask = ~mask  # invert: True = ignore
        else:
            src_key_padding_mask = None

        # TransformerEncoder with batch_first=True expects (batch, seq, feature)
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )  # (batch_size, K, d_model)

        # 5. Average pooling over the K tokens (mask-aware)
        if mask is not None:
            # Masked average: sum over valid positions, divide by count
            mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, K, 1)
            encoded_masked = encoded * mask_expanded
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch_size, 1)
            pooled = encoded_masked.sum(dim=1) / valid_counts  # (batch_size, d_model)
        else:
            pooled = encoded.mean(dim=1)  # (batch_size, d_model)

        # 6. Layer normalization
        pooled = self.layer_norm(pooled)

        # 7. Gaussian parameter heads
        mu = self.mu_head(pooled)  # (batch_size, latent_dim)
        logvar = self.logvar_head(pooled)  # (batch_size, latent_dim)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        # 8. Reparameterization trick: sample z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def encode_to_params(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state-reward pairs to distribution parameters only (no sampling).

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K).
            mask: Optional boolean mask of shape (batch_size, K).

        Returns:
            mu: Mean of latent Gaussian, shape (batch_size, latent_dim).
            logvar: Log variance of latent Gaussian, shape (batch_size, latent_dim).
        """
        batch_size, K, _ = states.shape

        reward_emb = self.reward_discretizer(rewards)
        combined = torch.cat([states, reward_emb], dim=-1)
        tokens = self.input_projection(combined)

        if mask is not None:
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            encoded_masked = encoded * mask_expanded
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = encoded_masked.sum(dim=1) / valid_counts
        else:
            pooled = encoded.mean(dim=1)

        pooled = self.layer_norm(pooled)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        return mu, logvar

    def sample_z(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample latent z from Gaussian parameters using reparameterization.

        Args:
            mu: Mean tensor of shape (batch_size, latent_dim).
            logvar: Log variance tensor of shape (batch_size, latent_dim).

        Returns:
            z: Sampled latent vector of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between N(mu, sigma^2) and N(0, I).

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        Args:
            mu: Mean tensor of shape (batch_size, latent_dim).
            logvar: Log variance tensor of shape (batch_size, latent_dim).

        Returns:
            KL divergence per sample, shape (batch_size,).
        """
        kl = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(),
            dim=-1,
        )
        return kl


def test_permutation_invariance():
    """
    Quick test to verify that the encoder is permutation-invariant:
    the output distribution should be identical for any permutation of
    the input state-reward pairs.
    """
    torch.manual_seed(42)

    state_dim = 10
    latent_dim = 64
    K = 32
    batch_size = 4

    encoder = FREEncoder(
        state_dim=state_dim,
        latent_dim=latent_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
    )

    # Create random input
    states = torch.randn(batch_size, K, state_dim)
    rewards = torch.randn(batch_size, K) * 2.0

    # Forward pass with original order
    encoder.eval()
    with torch.no_grad():
        z1, mu1, logvar1 = encoder(states, rewards)

    # Permute the K dimension
    perm = torch.randperm(K)
    states_perm = states[:, perm, :]
    rewards_perm = rewards[:, perm]

    with torch.no_grad():
        z2, mu2, logvar2 = encoder(states_perm, rewards_perm)

    # Check that mu and logvar are identical (within numerical precision)
    mu_diff = (mu1 - mu2).abs().max().item()
    logvar_diff = (logvar1 - logvar2).abs().max().item()

    print(f"Max mu difference after permutation: {mu_diff:.8f}")
    print(f"Max logvar difference after permutation: {logvar_diff:.8f}")

    assert mu_diff < 1e-5, f"Encoder is NOT permutation invariant! mu diff = {mu_diff}"
    assert logvar_diff < 1e-5, f"Encoder is NOT permutation invariant! logvar diff = {logvar_diff}"
    print("✓ Encoder is permutation-invariant!")


if __name__ == "__main__":
    test_permutation_invariance()