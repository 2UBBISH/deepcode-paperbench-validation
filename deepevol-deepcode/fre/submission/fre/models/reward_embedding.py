"""Reward discretization + embedding table.

Implements the scalar reward tokenization used by the FRE encoder.

Specification (paper Addendum "Additional Details on the FRE architecture" + Table 3):
  - the scalar reward is discretized into 32 bins by rescaling the reward to
    [0, 1] and then multiplying by 32 and flooring to the nearest integer.
  - the discretized reward is mapped to a continuous (64-d) vector representation
    using a learned embedding table.
  - the state is projected into a 64-d embedding with a learned linear transform
    and the reward embedding is concatenated to the end of the state embedding,
    giving a 128-d token.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RewardEmbedding(nn.Module):
    """Discretize scalar rewards into bins and embed them.

    Args:
        num_bins: number of reward bins (Table 3: 32).
        reward_embed_dim: dimension of the learned reward embedding (64).
        normalize: if True, per-reward-function min/max normalize to [0, 1]
            before binning (paper-faithful default; the paper states rewards are
            "rescaled to [0, 1]").
        eps: small constant used to avoid division by zero in normalization.
    """

    def __init__(
        self,
        num_bins: int = 32,
        reward_embed_dim: int = 64,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.reward_embed_dim = reward_embed_dim
        self.normalize = normalize
        self.eps = eps

        # Embedding table over the 32 discrete bins.
        self.embedding = nn.Embedding(num_bins, reward_embed_dim)

    def normalize_rewards(
        self,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rescale rewards to [0, 1].

        Args:
            rewards: (..., K) tensor of scalar rewards.
            mask: optional (..., K) boolean/float mask selecting samples used to
                compute min/max (e.g. valid encoder samples). If None all samples
                are used.

        Returns:
            Rewards rescaled to [0, 1].
        """
        if not self.normalize:
            return rewards.clamp(0.0, 1.0)

        if mask is None:
            rmin = rewards.min(dim=-1, keepdim=True).values
            rmax = rewards.max(dim=-1, keepdim=True).values
        else:
            m = mask.to(rewards.dtype).bool()
            big = torch.finfo(rewards.dtype).max
            neg = torch.finfo(rewards.dtype).min
            rmin = torch.where(m, rewards, torch.full_like(rewards, big)).min(
                dim=-1, keepdim=True
            ).values
            rmax = torch.where(m, rewards, torch.full_like(rewards, neg)).max(
                dim=-1, keepdim=True
            ).values

        denom = (rmax - rmin).clamp_min(self.eps)
        return (rewards - rmin) / denom

    def discretize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Map rewards in [0, 1] to integer bin indices in [0, num_bins - 1]."""
        scaled = rewards * self.num_bins
        idx = torch.floor(scaled)
        # Clamp into the valid embedding range (reward==1.0 would give num_bins).
        idx = idx.clamp(0, self.num_bins - 1)
        return idx.long()

    def forward(
        self,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """Embed scalar rewards.

        Args:
            rewards: (..., K) scalar rewards.
            mask: optional mask for normalization.
            already_normalized: if True, assumes rewards already in [0, 1].

        Returns:
            (..., K, reward_embed_dim) embedding tensor.
        """
        if not already_normalized:
            rewards = self.normalize_rewards(rewards, mask=mask)
        idx = self.discretize(rewards)
        return self.embedding(idx)
