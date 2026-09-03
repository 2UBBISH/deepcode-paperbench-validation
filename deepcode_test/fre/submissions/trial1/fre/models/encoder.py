"""
FRE Encoder: Permutation-invariant Transformer VAE that encodes reward functions
into a latent space from K state-reward pairs.

Architecture:
  1. Reward discretization: Map scalar reward to bin index, learn embedding table
  2. Token construction: Concatenate state projection and reward embedding
  3. Transformer encoder: No positional encodings, no causal masking
  4. Aggregation: Average K output vectors to get a single vector h
  5. VAE latent projection: mu = Linear(h), logvar = Linear(h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from fre.config import config


class RewardDiscretizer(nn.Module):
    """
    Discretizes scalar rewards into bins and learns an embedding for each bin.
    
    Given a scalar reward r_k, maps it to:
        bin_idx = clamp(round((r_k + R_max) / (2 * R_max) * num_bins), 0, num_bins-1)
    Then looks up embedding: e_k = E_reward[bin_idx]
    """
    
    def __init__(
        self,
        num_bins: int = 64,
        r_max: float = 10.0,
        d_embed: int = 128,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.r_max = r_max
        self.d_embed = d_embed
        
        # Embedding table: (num_bins, d_embed)
        self.embedding = nn.Embedding(num_bins, d_embed)
        
    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: (batch_size,) or (batch_size, K) tensor of scalar rewards
        
        Returns:
            embeddings: (..., d_embed) tensor of reward embeddings
        """
        # Clip rewards to [-r_max, r_max]
        rewards_clipped = torch.clamp(rewards, -self.r_max, self.r_max)
        
        # Map to [0, num_bins-1]
        # bin_idx = round((r + r_max) / (2 * r_max) * (num_bins - 1))
        normalized = (rewards_clipped + self.r_max) / (2.0 * self.r_max)
        bin_indices = torch.round(normalized * (self.num_bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Lookup embeddings
        embeddings = self.embedding(bin_indices)
        return embeddings


class FREEncoder(nn.Module):
    """
    Permutation-invariant Transformer VAE encoder.
    
    Input: K state-reward pairs {(s_k, r_k)}_{k=1..K}
    Output: mu, logvar of latent distribution over z
    
    The encoder is permutation-invariant: shuffling the order of input pairs
    yields the same output (no positional encodings, mean aggregation).
    """
    
    def __init__(
        self,
        state_dim: int,
        d_embed: int = 128,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        d_latent: int = 64,
        num_reward_bins: int = 64,
        r_max: float = 10.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            state_dim: Dimensionality of state space
            d_embed: Embedding dimension for reward and state projection
            d_model: Transformer hidden dimension (token dim = 2 * d_embed)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            d_latent: Dimension of latent variable z
            num_reward_bins: Number of bins for reward discretization
            r_max: Maximum absolute reward value for clipping
            dropout: Dropout rate for transformer
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.d_embed = d_embed
        self.d_model = d_model
        self.d_latent = d_latent
        
        # State projection: maps state to d_embed dimension
        self.state_proj = nn.Linear(state_dim, d_embed)
        
        # Reward discretizer and embedding
        self.reward_discretizer = RewardDiscretizer(
            num_bins=num_reward_bins,
            r_max=r_max,
            d_embed=d_embed,
        )
        
        # Token dimension: concatenation of state embedding and reward embedding
        token_dim = 2 * d_embed
        
        # Project token to d_model if different
        self.token_proj = nn.Linear(token_dim, d_model) if token_dim != d_model else nn.Identity()
        
        # Transformer encoder (no positional encoding, permutation invariant)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Input shape: (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # VAE latent projection heads
        self.mu_head = nn.Linear(d_model, d_latent)
        self.logvar_head = nn.Linear(d_model, d_latent)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in [self.state_proj, self.mu_head, self.logvar_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if isinstance(self.token_proj, nn.Linear):
            nn.init.xavier_uniform_(self.token_proj.weight)
            if self.token_proj.bias is not None:
                nn.init.zeros_(self.token_proj.bias)
    
    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode K state-reward pairs into latent distribution parameters.
        
        Args:
            states: (batch_size, K, state_dim) tensor of states
            rewards: (batch_size, K) tensor of scalar rewards
            deterministic: If True, return mu as z (no sampling)
        
        Returns:
            z: (batch_size, d_latent) sampled latent vector
            mu: (batch_size, d_latent) mean of latent distribution
            logvar: (batch_size, d_latent) log variance of latent distribution
        """
        batch_size, K, _ = states.shape
        
        # 1. Project states to d_embed
        # states: (batch_size, K, state_dim) -> (batch_size, K, d_embed)
        state_embeds = self.state_proj(states)
        
        # 2. Discretize rewards and get embeddings
        # rewards: (batch_size, K) -> (batch_size, K, d_embed)
        reward_embeds = self.reward_discretizer(rewards)
        
        # 3. Construct tokens: concatenate state and reward embeddings
        # tokens: (batch_size, K, 2*d_embed)
        tokens = torch.cat([state_embeds, reward_embeds], dim=-1)
        
        # 4. Project tokens to d_model
        tokens = self.token_proj(tokens)
        
        # 5. Pass through transformer encoder
        # No positional encoding, no src_key_padding_mask (treat as set)
        # transformer_output: (batch_size, K, d_model)
        transformer_output = self.transformer(tokens)
        
        # 6. Aggregate: mean over K dimension (permutation invariant)
        # h: (batch_size, d_model)
        h = transformer_output.mean(dim=1)
        
        # 7. VAE latent projection
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        # 8. Sample z using reparameterization trick
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        
        return z, mu, logvar
    
    def encode_deterministic(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode with deterministic z = mu (used during evaluation).
        
        Args:
            states: (batch_size, K, state_dim)
            rewards: (batch_size, K)
        
        Returns:
            z: (batch_size, d_latent)
        """
        z, _, _ = self.forward(states, rewards, deterministic=True)
        return z
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between N(mu, sigma^2) and N(0, I).
        
        KL = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        
        Args:
            mu: (batch_size, d_latent)
            logvar: (batch_size, d_latent)
        
        Returns:
            kl: (batch_size,) KL divergence per sample
        """
        kl = 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        return kl


def test_permutation_invariance():
    """Quick test to verify permutation invariance of the encoder."""
    import numpy as np
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    state_dim = 10
    K = 32
    batch_size = 4
    
    encoder = FREEncoder(
        state_dim=state_dim,
        d_embed=128,
        d_model=256,
        num_layers=2,
        num_heads=4,
        d_latent=64,
    )
    
    # Create random states and rewards
    states = torch.randn(batch_size, K, state_dim)
    rewards = torch.randn(batch_size, K)
    
    # Forward pass with original order
    z1, mu1, logvar1 = encoder(states, rewards, deterministic=True)
    
    # Shuffle the K dimension
    perm = torch.randperm(K)
    states_shuffled = states[:, perm, :]
    rewards_shuffled = rewards[:, perm]
    
    # Forward pass with shuffled order
    z2, mu2, logvar2 = encoder(states_shuffled, rewards_shuffled, deterministic=True)
    
    # Check invariance
    mu_diff = (mu1 - mu2).abs().max().item()
    print(f"Max mu difference after permutation: {mu_diff:.8f}")
    
    if mu_diff < 1e-5:
        print("✓ Encoder is permutation invariant!")
    else:
        print("✗ Encoder is NOT permutation invariant (diff > 1e-5)")
    
    return mu_diff < 1e-5


if __name__ == "__main__":
    test_permutation_invariance()