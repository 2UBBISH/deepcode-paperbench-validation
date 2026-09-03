"""
FRE Encoder: Permutation-invariant transformer for encoding reward functions.
"""

import torch
import torch.nn as nn
import numpy as np


class FREEncoder(nn.Module):
    """
    Functional Reward Encoder using a permutation-invariant transformer.

    Encodes a set of (state, reward) pairs into a latent representation z.
    """

    def __init__(self,
                 state_dim,
                 latent_dim=128,
                 state_embed_dim=64,
                 reward_embed_dim=64,
                 num_reward_bins=32,
                 num_heads=4,
                 num_layers=4,
                 mlp_hidden_dim=256,
                 beta=0.01):
        """
        Args:
            state_dim: Dimension of state space
            latent_dim: Dimension of latent embedding z
            state_embed_dim: Dimension of state embeddings
            reward_embed_dim: Dimension of reward embeddings
            num_reward_bins: Number of bins for reward discretization
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_hidden_dim: Hidden dimension of transformer MLP
            beta: KL divergence weight in variational objective
        """
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.state_embed_dim = state_embed_dim
        self.reward_embed_dim = reward_embed_dim
        self.num_reward_bins = num_reward_bins
        self.beta = beta

        # State embedding: linear projection
        self.state_embedding = nn.Linear(state_dim, state_embed_dim)

        # Reward embedding: learned lookup table
        self.reward_embedding = nn.Embedding(num_reward_bins, reward_embed_dim)

        # Total embedding dimension
        embed_dim = state_embed_dim + reward_embed_dim

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden_dim,
            dropout=0.0,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projections for mean and log_std
        self.fc_mean = nn.Linear(embed_dim, latent_dim)
        self.fc_logstd = nn.Linear(embed_dim, latent_dim)

    def discretize_reward(self, rewards):
        """
        Discretize rewards into bins.
        Rescale to [0, 1], multiply by num_bins, and floor.

        Args:
            rewards: (batch_size, K) tensor of rewards

        Returns:
            (batch_size, K) tensor of discretized reward bins (integers)
        """
        # Rescale to [0, 1]
        # Assume rewards are in [-1, 1] range based on paper's reward functions
        rewards_normalized = (rewards + 1.0) / 2.0
        rewards_normalized = torch.clamp(rewards_normalized, 0.0, 1.0)

        # Discretize
        bins = (rewards_normalized * self.num_reward_bins).long()
        bins = torch.clamp(bins, 0, self.num_reward_bins - 1)

        return bins

    def forward(self, states, rewards):
        """
        Encode (state, reward) pairs into latent z.

        Args:
            states: (batch_size, K, state_dim) tensor of states
            rewards: (batch_size, K) tensor of rewards

        Returns:
            mean: (batch_size, latent_dim) mean of latent distribution
            logstd: (batch_size, latent_dim) log std of latent distribution
        """
        batch_size, K, _ = states.shape

        # Embed states
        state_embeds = self.state_embedding(states)  # (batch_size, K, state_embed_dim)

        # Discretize and embed rewards
        reward_bins = self.discretize_reward(rewards)  # (batch_size, K)
        reward_embeds = self.reward_embedding(reward_bins)  # (batch_size, K, reward_embed_dim)

        # Concatenate state and reward embeddings
        embeddings = torch.cat([state_embeds, reward_embeds], dim=-1)  # (batch_size, K, embed_dim)

        # Pass through transformer (permutation-invariant)
        # No positional encoding, no causal mask
        transformer_output = self.transformer(embeddings)  # (batch_size, K, embed_dim)

        # Average over the set dimension to get permutation-invariant representation
        pooled = transformer_output.mean(dim=1)  # (batch_size, embed_dim)

        # Project to mean and log_std
        mean = self.fc_mean(pooled)
        logstd = self.fc_logstd(pooled)

        return mean, logstd

    def sample_z(self, mean, logstd):
        """
        Sample z from the latent distribution using reparameterization trick.

        Args:
            mean: (batch_size, latent_dim)
            logstd: (batch_size, latent_dim)

        Returns:
            z: (batch_size, latent_dim) sampled latent vector
        """
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z

    def compute_kl_loss(self, mean, logstd):
        """
        Compute KL divergence from unit Gaussian prior.

        KL(q(z|x) || p(z)) where p(z) = N(0, I)

        Args:
            mean: (batch_size, latent_dim)
            logstd: (batch_size, latent_dim)

        Returns:
            kl: scalar KL divergence
        """
        var = torch.exp(2 * logstd)
        kl = -0.5 * torch.sum(1 + 2 * logstd - mean.pow(2) - var)
        kl = kl / mean.shape[0]  # Average over batch
        return kl

    def encode(self, states, rewards):
        """
        Convenience method to encode and sample z.

        Args:
            states: (batch_size, K, state_dim)
            rewards: (batch_size, K)

        Returns:
            z: (batch_size, latent_dim) sampled latent vector
            mean: (batch_size, latent_dim)
            logstd: (batch_size, latent_dim)
        """
        mean, logstd = self.forward(states, rewards)
        z = self.sample_z(mean, logstd)
        return z, mean, logstd
