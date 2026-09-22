"""FRE encoder (Sec 4.1, App A, Addendum).

Maps K (state, reward) pairs to a Gaussian p_theta(z | context) over a 128-dim
latent task embedding z.

Token construction per (state, reward) pair:
  * state -> learned linear projection to 64-d (no position/observation embedding)
  * scalar reward -> discretized: rescale to [0, 1], * 32, floor -> one of 32 bins,
    then looked up in a learned embedding table (32 entries, each 64-d)
  * concat -> 128-d token

Transformer:
  * NO positional encoding, NO causal masking (context is an unordered set)
  * attention / residual dim = 128
  * MLP block expands 128 -> 256 -> 128
  * 4 attention heads, 4 transformer blocks
  * final mean-pool over tokens -> two linear projections -> (mu, log_sigma)

Sampling: z = mu + sigma * eps, eps ~ N(0, I)  (reparameterization).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reward discretization constants (addendum: rescale -> *32 -> floor into 32 bins).
NUM_REWARD_BINS = 32
REWARD_RESCALE = 32.0

# Token dimensionality: 64 (state) + 64 (reward) = 128 (corrected addendum value).
STATE_EMB_DIM = 64
REWARD_EMB_DIM = 64
TOKEN_DIM = STATE_EMB_DIM + REWARD_EMB_DIM  # 128


def discretize_reward(reward: torch.Tensor) -> torch.Tensor:
    """Rescale reward to [0, 1], multiply by 32, floor, and clamp into [0, 31].

    Assumes rewards live in [-1, 1] (the FRE prior is defined on that range).
    """
    rescaled = (reward + 1.0) / 2.0  # -> [0, 1]
    scaled = torch.floor(rescaled * REWARD_RESCALE)
    return scaled.clamp(0, NUM_REWARD_BINS - 1).long()


class MultiheadSelfAttention(nn.Module):
    """Standard multi-head self-attention (bidirectional, no mask)."""

    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, n, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # no causal masking
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (b, heads, n, head_dim)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + 128 -> 256 -> 128 MLP."""

    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FREEncoder(nn.Module):
    """Permutation-invariant transformer encoder producing q_theta(z | context).

    Args:
        state_dim: raw observation dimensionality.
        latent_dim: dimensionality of z (128).
        num_blocks: number of transformer blocks (4).
        num_heads: attention heads (4).
        mlp_dim: transformer MLP hidden dim (256; App "Encoder Layers [256,256,256,256]").
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.state_proj = nn.Linear(state_dim, STATE_EMB_DIM)
        self.reward_embedding = nn.Embedding(NUM_REWARD_BINS, REWARD_EMB_DIM)
        nn.init.normal_(self.reward_embedding.weight, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(TOKEN_DIM, num_heads, mlp_dim, dropout)
             for _ in range(num_blocks)]
        )
        self.norm_out = nn.LayerNorm(TOKEN_DIM)

        self.fc_mu = nn.Linear(TOKEN_DIM, latent_dim)
        self.fc_log_sigma = nn.Linear(TOKEN_DIM, latent_dim)

    def encode_tokens(self, states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Build the unordered token set.

        Args:
            states: (..., K, state_dim)
            rewards: (..., K) rewards in [-1, 1]
        Returns:
            tokens: (..., K, 128)
        """
        state_emb = self.state_proj(states)
        reward_bins = discretize_reward(rewards)
        reward_emb = self.reward_embedding(reward_bins)
        return torch.cat([state_emb, reward_emb], dim=-1)

    def forward(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_sigma) of the posterior q_theta(z | context).

        Args:
            states: (B, K, state_dim)
            rewards: (B, K)
        """
        tokens = self.encode_tokens(states, rewards)  # (B, K, 128)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm_out(tokens)
        pooled = tokens.mean(dim=1)  # mean-pool over the unordered set
        mu = self.fc_mu(pooled)
        log_sigma = self.fc_log_sigma(pooled).clamp(-10.0, 5.0)
        return mu, log_sigma

    def sample(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample plus (mu, log_sigma).

        z = mu + sigma * eps, eps ~ N(0, I).
        """
        mu, log_sigma = self.forward(states, rewards)
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z, mu, log_sigma

    @torch.no_grad()
    def encode(
        self, states: torch.Tensor, rewards: torch.Tensor, use_mean: bool = False
    ) -> torch.Tensor:
        """Encode a context into z (used at zero-shot evaluation time).

        Note: the paper samples z stochastically even at eval (32 reward samples),
        but `use_mean` allows using the posterior mean for deterministic behaviour.
        """
        mu, log_sigma = self.forward(states, rewards)
        if use_mean:
            return mu
        sigma = torch.exp(0.5 * log_sigma)
        return mu + sigma * torch.randn_like(sigma)

    @torch.no_grad()
    def encode_set(self, states: torch.Tensor, rewards: torch.Tensor,
                   num_samples: int = 8, use_mean: bool = True) -> torch.Tensor:
        """Encode one context repeatedly, returning (num_samples, latent_dim)."""
        mus, log_sigmas = self.forward(states, rewards)
        if use_mean:
            return mus
        sigma = torch.exp(0.5 * log_sigmas)
        eps = torch.randn(num_samples, *sigma.shape, device=sigma.device)
        return mus.unsqueeze(0) + sigma.unsqueeze(0) * eps
