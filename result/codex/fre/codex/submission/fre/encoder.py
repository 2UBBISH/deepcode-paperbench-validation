"""Permutation-invariant transformer encoder for reward functions (Section 4.1).

The encoder maps a *set* of ``K`` (state, reward) pairs, produced by evaluating
an arbitrary reward function ``eta`` on ``K`` dataset states, to the parameters
of a Gaussian posterior ``p_theta(z | {(s_k, eta(s_k))})``.

Implementation details (Section 4.1 "Practical Implementation", Appendix A and
the addendum):

  * each state is projected into a 64-dimensional embedding with a learned
    linear transformation;
  * the scalar reward is discretised into 32 bins and mapped to a learned
    64-dimensional embedding token;
  * the reward embedding is concatenated to the end of the state embedding,
    giving a 128-dimensional input vector per (state, reward) pair;
  * the sequence is passed through a transformer with 4 attention heads and
    128-dimensional residual/attention activations, whose MLP block expands to
    256 then projects back to 128;
  * no positional encodings and no causal masking are used, so the encoder is
    permutation invariant over the set of pairs;
  * the average of the final layer representations is fed to two linear
    projections parameterising the mean and standard deviation of ``z``.

Note on the appendix config: the row labelled "Encoder Layers [256, 256, 256,
256]" lists the MLP dimensions of the transformer (the addendum clarifies that
the residual stream is 128-dimensional and the MLP expands to 256).  Likewise
the appendix row "Reward Embedding Dim 128" is a typo: the state embedding and
reward embedding are each 64-dimensional and concatenate to 128.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetTransformerBlock(nn.Module):
    """Pre-LN transformer block over a set (no positional encoding, no mask)."""

    def __init__(self, d_model: int, n_heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention over the whole set.  ``attn_mask`` is None and
        # ``key_padding_mask`` is None: every element attends to every other
        # element, which makes the block permutation equivariant.
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Permutation-invariant transformer VAE encoder for reward functions."""

    def __init__(
        self,
        state_dim: int,
        z_dim: int = 128,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_dim: int = 256,
        num_reward_bins: int = 32,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if state_embed_dim + reward_embed_dim != d_model:
            raise ValueError(
                "state_embed_dim + reward_embed_dim must equal d_model "
                f"(got {state_embed_dim} + {reward_embed_dim} != {d_model})"
            )
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.num_reward_bins = num_reward_bins
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.state_proj = nn.Linear(state_dim, state_embed_dim)
        self.reward_embed = nn.Embedding(num_reward_bins, reward_embed_dim)
        self.blocks = nn.ModuleList(
            [SetTransformerBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_mean = nn.Linear(d_model, z_dim)
        self.fc_log_std = nn.Linear(d_model, z_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.0)
        # Start with a small posterior variance (large negative log std).
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.zeros_(self.fc_log_std.weight)
        nn.init.constant_(self.fc_log_std.bias, -2.0)

    def forward(
        self, states: torch.Tensor, reward_bins: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of reward-annotated state sets.

        Args:
            states: ``(B, K, state_dim)`` encoder states.
            reward_bins: ``(B, K)`` discretised rewards (int64 in
                ``[0, num_reward_bins)``).

        Returns:
            ``(mean, log_std)``, each of shape ``(B, z_dim)``.
        """
        if states.dim() != 3:
            raise ValueError(f"states must be (B, K, state_dim), got {tuple(states.shape)}")
        if reward_bins.shape != states.shape[:2]:
            raise ValueError(
                f"reward_bins must be (B, K)={tuple(states.shape[:2])}, got {tuple(reward_bins.shape)}"
            )
        s = self.state_proj(states)  # (B, K, state_embed_dim)
        r = self.reward_embed(reward_bins)  # (B, K, reward_embed_dim)
        # "for each state, the reward embedding is concatenated to end of the
        # state embedding before the set ... is passed through the encoder"
        x = torch.cat([s, r], dim=-1)  # (B, K, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        pooled = x.mean(dim=1)  # permutation-invariant pooling over the set
        mean = self.fc_mean(pooled)
        log_std = self.fc_log_std(pooled).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def encode(
        self,
        states: torch.Tensor,
        reward_bins: torch.Tensor,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return a latent ``z`` for each reward function in the batch."""
        mean, log_std = self.forward(states, reward_bins)
        if not sample:
            return mean
        std = log_std.exp()
        eps = torch.randn(mean.shape, device=mean.device, generator=generator)
        return mean + std * eps


def gaussian_kl(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """KL( N(mean, std^2) || N(0, I) ), summed over latent dimensions then batch-meaned."""
    var = torch.exp(2.0 * log_std)
    kl_per_dim = 0.5 * (mean.pow(2) + var - 1.0 - 2.0 * log_std)
    return kl_per_dim.sum(dim=-1).mean()
