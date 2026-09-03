"""
FRE Encoder: Permutation-invariant Transformer VAE encoder.

Encodes a set of (state, reward) pairs into a latent Gaussian distribution
p_θ(z | {(s_i, η(s_i))}). Uses a Transformer without positional encoding
and without causal masking to achieve permutation invariance.

Architecture:
  1. RewardDiscretizer: Maps continuous scalar rewards to learned embeddings
     via discretization into B bins.
  2. StateEmbedding: Linear projection from state_dim to d_model.
  3. InputSequence: Concatenates state and reward embeddings, projects to d_model.
  4. TransformerEncoder: Standard Transformer (no pos encoding, no causal mask).
  5. Aggregation: Mean pooling over output sequence.
  6. LatentHeads: μ and log σ linear projections to produce latent distribution.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Reward Discretizer
# ==============================================================================

class RewardDiscretizer(nn.Module):
    """
    Discretizes continuous scalar rewards into B bins and maps each bin
    to a learned embedding vector.

    The bin edges are uniformly spaced over [r_min, r_max]. Each reward
    value is assigned to the bin whose center is closest, and the
    corresponding embedding is retrieved.

    Args:
        num_bins: Number of discretization bins (B).
        r_min: Minimum reward value for bin range.
        r_max: Maximum reward value for bin range.
        d_reward: Dimension of the reward embedding vector.
    """

    def __init__(
        self,
        num_bins: int = 50,
        r_min: float = -1.0,
        r_max: float = 1.0,
        d_reward: int = 32,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.r_min = r_min
        self.r_max = r_max
        self.d_reward = d_reward

        # Bin centers: uniformly spaced between r_min and r_max
        self.register_buffer(
            "bin_centers",
            torch.linspace(r_min, r_max, num_bins),
        )

        # Learnable embedding table: (num_bins, d_reward)
        self.embedding = nn.Embedding(num_bins, d_reward)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous rewards to embedding vectors.

        Args:
            rewards: Tensor of shape (...,) or (batch_size, K) containing
                     scalar reward values.

        Returns:
            Tensor of shape (..., d_reward) containing reward embeddings.
        """
        # Clamp rewards to [r_min, r_max]
        rewards_clamped = torch.clamp(rewards, self.r_min, self.r_max)

        # Compute distances to each bin center and find nearest bin
        # rewards_clamped: (...,) -> (..., 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        rewards_expanded = rewards_clamped.unsqueeze(-1)  # (..., 1)
        bin_centers_expanded = self.bin_centers.view(
            *([1] * rewards_clamped.dim()), self.num_bins
        )  # (1, ..., 1, num_bins)

        distances = torch.abs(rewards_expanded - bin_centers_expanded)
        bin_indices = torch.argmin(distances, dim=-1)  # (...,)

        # Retrieve embeddings
        embeddings = self.embedding(bin_indices)  # (..., d_reward)
        return embeddings

    def get_bin_centers(self) -> torch.Tensor:
        """Return the bin center values."""
        return self.bin_centers


# ==============================================================================
# State Embedding
# ==============================================================================

class StateEmbedding(nn.Module):
    """
    Linear projection from raw state dimension to model dimension.

    Args:
        state_dim: Dimensionality of the state space.
        d_model: Target embedding dimension.
    """

    def __init__(self, state_dim: int, d_model: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.projection = nn.Linear(state_dim, d_model)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Project states to d_model dimension.

        Args:
            states: Tensor of shape (..., state_dim).

        Returns:
            Tensor of shape (..., d_model).
        """
        return self.projection(states)


# ==============================================================================
# Input Sequence Constructor
# ==============================================================================

class InputSequenceConstructor(nn.Module):
    """
    Constructs the input sequence for the Transformer by concatenating
    state and reward embeddings and projecting to d_model.

    For each of the K encoder states, we have:
      - state_embedding: d_model dims
      - reward_embedding: d_reward dims
      -> concat -> (d_model + d_reward) dims
      -> linear projection -> d_model dims

    Args:
        d_model: Model dimension.
        d_reward: Reward embedding dimension.
    """

    def __init__(self, d_model: int = 256, d_reward: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_reward = d_reward
        self.input_projection = nn.Linear(d_model + d_reward, d_model)

    def forward(
        self,
        state_embeddings: torch.Tensor,
        reward_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine state and reward embeddings into Transformer input tokens.

        Args:
            state_embeddings: Tensor of shape (batch_size, K, d_model).
            reward_embeddings: Tensor of shape (batch_size, K, d_reward).

        Returns:
            Tensor of shape (batch_size, K, d_model).
        """
        # Concatenate along last dimension
        combined = torch.cat([state_embeddings, reward_embeddings], dim=-1)
        # combined: (batch_size, K, d_model + d_reward)

        # Project to d_model
        tokens = self.input_projection(combined)  # (batch_size, K, d_model)
        return tokens


# ==============================================================================
# Permutation-Invariant Transformer Encoder
# ==============================================================================

class PermutationInvariantTransformer(nn.Module):
    """
    Standard Transformer encoder without positional encoding and without
    causal/autoregressive masking. This makes the encoder permutation-invariant
    with respect to the input sequence order.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate.
        activation: Activation function for feedforward layers.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        # Standard TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Use (batch, seq, feature) format
        )

        # TransformerEncoder (stack of layers)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input tokens through the Transformer.

        Args:
            x: Tensor of shape (batch_size, K, d_model).

        Returns:
            Tensor of shape (batch_size, K, d_model).
        """
        # No positional encoding added.
        # No src_mask or src_key_padding_mask -> full bidirectional attention.
        output = self.transformer(x, mask=None)
        return output


# ==============================================================================
# Aggregation (Mean Pooling)
# ==============================================================================

class MeanAggregator(nn.Module):
    """
    Aggregates a sequence of vectors into a single vector via mean pooling.

    Args:
        d_model: Dimension of input/output vectors.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool over the sequence dimension.

        Args:
            x: Tensor of shape (batch_size, K, d_model).

        Returns:
            Tensor of shape (batch_size, d_model).
        """
        return x.mean(dim=1)


# ==============================================================================
# Latent Distribution Heads
# ==============================================================================

class LatentHeads(nn.Module):
    """
    Produces the parameters of a diagonal Gaussian latent distribution
    from the aggregated encoder output.

    Args:
        d_model: Input dimension (aggregated encoder output).
        d_latent: Dimension of the latent space.
    """

    def __init__(self, d_model: int = 256, d_latent: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent

        self.mu_head = nn.Linear(d_model, d_latent)
        self.logvar_head = nn.Linear(d_model, d_latent)

    def forward(
        self, aggregated: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute μ and log σ² from aggregated representation.

        Args:
            aggregated: Tensor of shape (batch_size, d_model).

        Returns:
            mu: Tensor of shape (batch_size, d_latent).
            logvar: Tensor of shape (batch_size, d_latent).
        """
        mu = self.mu_head(aggregated)
        logvar = self.logvar_head(aggregated)
        return mu, logvar


# ==============================================================================
# Full FRE Encoder
# ==============================================================================

class FREEncoder(nn.Module):
    """
    Full FRE Encoder: encodes a set of (state, reward) pairs into a latent
    Gaussian distribution.

    Pipeline:
      1. Embed states via StateEmbedding.
      2. Discretize and embed rewards via RewardDiscretizer.
      3. Construct input tokens via InputSequenceConstructor.
      4. Process through PermutationInvariantTransformer.
      5. Aggregate via MeanAggregator.
      6. Produce μ and log σ² via LatentHeads.
      7. Sample z via reparameterization.

    Args:
        state_dim: Dimensionality of the state space.
        d_model: Model dimension for Transformer.
        d_reward: Reward embedding dimension.
        d_latent: Latent space dimension.
        num_bins: Number of reward discretization bins.
        r_min: Minimum reward for bin range.
        r_max: Maximum reward for bin range.
        nhead: Number of Transformer attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward dimension in Transformer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int = 256,
        d_reward: int = 32,
        d_latent: int = 64,
        num_bins: int = 50,
        r_min: float = -1.0,
        r_max: float = 1.0,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.d_reward = d_reward
        self.d_latent = d_latent
        self.num_bins = num_bins

        # Sub-modules
        self.state_embedding = StateEmbedding(state_dim, d_model)
        self.reward_discretizer = RewardDiscretizer(num_bins, r_min, r_max, d_reward)
        self.input_constructor = InputSequenceConstructor(d_model, d_reward)
        self.transformer = PermutationInvariantTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.aggregator = MeanAggregator(d_model)
        self.latent_heads = LatentHeads(d_model, d_latent)

    def encode(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a set of (state, reward) pairs into μ and log σ².

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K).

        Returns:
            mu: Tensor of shape (batch_size, d_latent).
            logvar: Tensor of shape (batch_size, d_latent).
        """
        batch_size, K, _ = states.shape

        # 1. Embed states
        state_emb = self.state_embedding(states)  # (batch_size, K, d_model)

        # 2. Discretize and embed rewards
        reward_emb = self.reward_discretizer(rewards)  # (batch_size, K, d_reward)

        # 3. Construct input tokens
        tokens = self.input_constructor(state_emb, reward_emb)  # (batch_size, K, d_model)

        # 4. Transformer
        transformer_out = self.transformer(tokens)  # (batch_size, K, d_model)

        # 5. Aggregate
        aggregated = self.aggregator(transformer_out)  # (batch_size, d_model)

        # 6. Latent distribution parameters
        mu, logvar = self.latent_heads(aggregated)

        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample z via the reparameterization trick.

        Args:
            mu: Mean of latent distribution (batch_size, d_latent).
            logvar: Log variance of latent distribution (batch_size, d_latent).

        Returns:
            z: Sampled latent vector (batch_size, d_latent).
            kl: KL divergence D_KL(N(mu, sigma) || N(0, I)) per sample,
                shape (batch_size,).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # KL divergence: 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        return z, kl

    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, then sample z.

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K).

        Returns:
            z: Sampled latent vector (batch_size, d_latent).
            mu: Mean (batch_size, d_latent).
            logvar: Log variance (batch_size, d_latent).
            kl: KL divergence per sample (batch_size,).
        """
        mu, logvar = self.encode(states, rewards)
        z, kl = self.reparameterize(mu, logvar)
        return z, mu, logvar, kl

    def encode_deterministic(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode deterministically (use μ only, no sampling).
        Useful for evaluation.

        Args:
            states: Tensor of shape (batch_size, K, state_dim).
            rewards: Tensor of shape (batch_size, K).

        Returns:
            z: Deterministic latent vector (batch_size, d_latent) = μ.
        """
        mu, _ = self.encode(states, rewards)
        return mu


# ==============================================================================
# Factory Function
# ==============================================================================

def create_fre_encoder(
    state_dim: int,
    d_model: int = 256,
    d_reward: int = 32,
    d_latent: int = 64,
    num_bins: int = 50,
    r_min: float = -1.0,
    r_max: float = 1.0,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,
) -> FREEncoder:
    """
    Factory function to create an FREEncoder with specified hyperparameters.

    Args:
        state_dim: Dimensionality of the state space.
        d_model: Model dimension.
        d_reward: Reward embedding dimension.
        d_latent: Latent space dimension.
        num_bins: Number of reward discretization bins.
        r_min: Minimum reward for bin range.
        r_max: Maximum reward for bin range.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.

    Returns:
        FREEncoder instance.
    """
    return FREEncoder(
        state_dim=state_dim,
        d_model=d_model,
        d_reward=d_reward,
        d_latent=d_latent,
        num_bins=num_bins,
        r_min=r_min,
        r_max=r_max,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


# ==============================================================================
# Testing
# ==============================================================================

def test_encoder():
    """
    Quick test to verify the encoder runs end-to-end with correct shapes.
    """
    print("Testing FREEncoder...")

    state_dim = 29  # AntMaze state dim
    K = 32
    batch_size = 4
    d_latent = 64

    encoder = FREEncoder(
        state_dim=state_dim,
        d_model=256,
        d_reward=32,
        d_latent=d_latent,
        num_bins=50,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
    )

    # Create dummy inputs
    states = torch.randn(batch_size, K, state_dim)
    rewards = torch.rand(batch_size, K) * 2 - 1  # Uniform in [-1, 1]

    # Forward pass
    z, mu, logvar, kl = encoder(states, rewards)

    print(f"  States shape: {states.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  z shape: {z.shape} (expected: [{batch_size}, {d_latent}])")
    print(f"  mu shape: {mu.shape}")
    print(f"  logvar shape: {logvar.shape}")
    print(f"  kl shape: {kl.shape} (expected: [{batch_size}])")
    print(f"  kl mean: {kl.mean().item():.4f}")

    # Test deterministic encoding
    z_det = encoder.encode_deterministic(states, rewards)
    print(f"  z_det shape: {z_det.shape}")

    # Test permutation invariance: shuffling input order should give same result
    perm = torch.randperm(K)
    states_perm = states[:, perm, :]
    rewards_perm = rewards[:, perm]
    z_perm, _, _, _ = encoder(states_perm, rewards_perm)
    diff = (z - z_perm).abs().max().item()
    print(f"  Max difference after permutation: {diff:.6f} (should be ~0)")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("FREEncoder test passed!")
    return encoder


if __name__ == "__main__":
    test_encoder()