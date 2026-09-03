"""
Functional Reward Encoding (FRE) — Core Model

Implements the variational auto-encoder architecture that encodes arbitrary
reward functions into a compressed latent representation z.

Reference: "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"
by Frans, Park, Abbeel, Levine (ICML 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class RewardEmbedding(nn.Module):
    """
    Discretizes a scalar reward into bins and maps to a learned continuous embedding.

    The reward is rescaled to [0, 1], multiplied by num_bins, and floored.
    The resulting integer is used as an index into an embedding table.
    """

    def __init__(self, num_bins: int = 32, embedding_dim: int = 64):
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_bins, embedding_dim)

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reward: scalar rewards, shape (...,)
        Returns:
            embeddings, shape (..., embedding_dim)
        """
        # Clamp reward to [0, 1]
        reward_clamped = reward.clamp(0.0, 1.0)
        # Discretize: scale to [0, num_bins-1]
        indices = (reward_clamped * (self.num_bins - 1)).long()
        indices = indices.clamp(0, self.num_bins - 1)
        return self.embedding(indices)


class StateEmbedding(nn.Module):
    """Linear projection of raw environment state into a learned embedding."""

    def __init__(self, state_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(state_dim, embedding_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: raw state, shape (..., state_dim)
        Returns:
            embeddings, shape (..., embedding_dim)
        """
        return self.projection(state)


class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block without positional encodings or causal masking.
    Used for permutation-invariant encoding of (state, reward) sets.
    """

    def __init__(self, embed_dim: int = 128, mlp_dim: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        # MLP with residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class FREEncoder(nn.Module):
    """
    Permutation-invariant transformer encoder that maps a set of (state, reward)
    pairs to a latent Gaussian distribution over z.

    Architecture:
    - State is projected via a learned linear layer (64-dim).
    - Reward is discretized into 32 bins and embedded (64-dim).
    - Concatenated state+reward embeddings (128-dim) are fed into a transformer.
    - No positional encodings or causal masking — inputs are treated as an unordered set.
    - The average of the final layer representations parametrizes μ and log σ.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_reward_bins: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        embed_dim = state_embed_dim + reward_embed_dim  # 128

        self.state_embed = StateEmbedding(state_dim, state_embed_dim)
        self.reward_embed = RewardEmbedding(num_reward_bins, reward_embed_dim)
        self.latent_dim = latent_dim

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, mlp_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

        # Output projections for μ and log σ of Gaussian z
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states:  (batch, K, state_dim) — K encoder states
            rewards: (batch, K) — corresponding scalar rewards
        Returns:
            mu:     (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        B, K, _ = states.shape

        # Embed states and rewards
        state_emb = self.state_embed(states)       # (B, K, 64)
        reward_emb = self.reward_embed(rewards)     # (B, K, 64)

        # Concatenate
        x = torch.cat([state_emb, reward_emb], dim=-1)  # (B, K, 128)

        # Pass through transformer blocks (no positional encoding, no causal mask)
        for block in self.encoder_blocks:
            x = block(x)

        x = self.ln_final(x)

        # Mean-pool over the K set elements
        x_pooled = x.mean(dim=1)  # (B, 128)

        mu = self.fc_mu(x_pooled)          # (B, latent_dim)
        logvar = self.fc_logvar(x_pooled)  # (B, latent_dim)

        return mu, logvar

    def encode(self, states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Deterministic encoding: sample z ~ N(μ, σ²) using reparameterization.
        """
        mu, logvar = self.forward(states, rewards)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class FREDecoder(nn.Module):
    """
    Feedforward decoder that predicts η(s) given a state s and latent encoding z.

    The raw state and z-vector are concatenated directly (no state embedding).
    Network: [512, 512, 512] MLP (per Appendix Table 3).
    """

    def __init__(
        self, state_dim: int, latent_dim: int = 128, hidden_dims: list = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim + latent_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # scalar reward output
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch, K', state_dim) — decoder states
            z:      (batch, latent_dim)
        Returns:
            predicted rewards: (batch, K', 1)
        """
        B, Kp, _ = states.shape
        # Expand z to match each decoder state
        z_expanded = z.unsqueeze(1).expand(-1, Kp, -1)  # (B, K', latent_dim)
        # Concatenate raw state with z
        x = torch.cat([states, z_expanded], dim=-1)       # (B, K', state_dim+latent_dim)
        return self.net(x).squeeze(-1)                     # (B, K')


class FREModel(nn.Module):
    """
    Full FRE encoder-decoder model.

    Jointly optimizes the variational information bottleneck objective:
      L = MSE(η_decoder, η_true) + β * KL(N(μ,σ²) || N(0,I))

    where the KL term penalizes deviation from the unit Gaussian prior u(z).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_reward_bins: int = 32,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        decoder_hidden_dims: list = None,
        beta: float = 0.01,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [512, 512, 512]

        self.encoder = FREEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            state_embed_dim=state_embed_dim,
            reward_embed_dim=reward_embed_dim,
            num_reward_bins=num_reward_bins,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
        )
        self.decoder = FREDecoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
        )
        self.beta = beta
        self.latent_dim = latent_dim

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_states:  (B, K, state_dim)
            encoder_rewards: (B, K)
            decoder_states:  (B, K', state_dim)
            decoder_rewards: (B, K')
        Returns:
            total_loss: scalar
            mse_loss:   scalar — reconstruction MSE on decoder states
            kl_loss:    scalar — KL divergence from unit Gaussian prior
        """
        mu, logvar = self.encoder.forward(encoder_states, encoder_rewards)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode
        pred_rewards = self.decoder.forward(decoder_states, z)  # (B, K')

        # MSE reconstruction loss
        mse_loss = F.mse_loss(pred_rewards, decoder_rewards)

        # KL divergence against unit Gaussian N(0, I)
        # D_KL(N(μ,σ²) || N(0,I)) = ½ Σ(μ² + σ² - 1 - log σ²)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # Total: Equation (6) variational lower bound
        total_loss = mse_loss + self.beta * kl_loss

        return total_loss, mse_loss, kl_loss

    def encode(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Produce latent encoding z for a reward function given (s, η(s)) samples.
        """
        return self.encoder.encode(states, rewards)