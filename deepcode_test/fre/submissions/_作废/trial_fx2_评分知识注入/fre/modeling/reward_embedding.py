"""Learned reward embedding.

FRE maps each scalar reward `r_k = eta(s_k)` to a learned vector embedding.
By default rewards are discretized uniformly into `num_bins` bins over a
bounded interval and the bin index is passed through an ``nn.Embedding``
table.  A continuous linear projection fallback is also provided for
environments or reward families where fixed discretization is unstable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RewardEmbedding(nn.Module):
    """Discretize scalar rewards and embed them into a continuous vector space.

    Parameters
    ----------
    num_bins:
        Number of uniform reward bins ``M``. Default 128.
    embedding_dim:
        Output embedding dimension. Default 128.
    reward_min, reward_max:
        Bounds used for reward normalization. Rewards outside the range are
        clamped into ``[0, M-1]`` bin ids.
    use_linear:
        If ``True``, bypass discretization and use a learned linear projection
        from the scalar reward to ``embedding_dim``. This is the fallback
        mentioned in the reproduction plan.
    """

    def __init__(
        self,
        num_bins: int = 128,
        embedding_dim: int = 128,
        reward_min: float = -1.0,
        reward_max: float = 1.0,
        use_linear: bool = False,
    ) -> None:
        super().__init__()
        self.num_bins = max(int(num_bins), 1)
        self.embedding_dim = int(embedding_dim)
        self.use_linear = bool(use_linear)

        # Buffers are not trainable and move with ``.to(device)``.
        self.register_buffer("reward_min", torch.tensor(float(reward_min)))
        self.register_buffer("reward_max", torch.tensor(float(reward_max)))

        if self.use_linear:
            self.linear = nn.Linear(1, self.embedding_dim)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(self.num_bins, self.embedding_dim)
            self.linear = None

        # Keep a small epsilon to avoid division by zero when min == max.
        self.register_buffer("_eps", torch.tensor(1e-8, dtype=torch.float32))

    def discretize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Map scalar rewards to long bin ids.

        ``bin_id = clamp(round((r - r_min) / (r_max - r_min) * (M - 1)), 0, M - 1)``.
        """
        rewards = rewards.to(dtype=torch.float32)
        denom = (self.reward_max - self.reward_min).abs() + self._eps
        normalized = (rewards - self.reward_min) / denom
        normalized = normalized.clamp(0.0, 1.0)
        bin_ids = torch.round(normalized * (self.num_bins - 1)).long()
        return bin_ids.clamp(0, self.num_bins - 1)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """Return reward embeddings with shape ``(..., embedding_dim)``."""
        rewards = rewards.to(dtype=torch.float32)
        if self.use_linear:
            assert self.linear is not None
            return self.linear(rewards.unsqueeze(-1))

        assert self.embedding is not None
        bin_ids = self.discretize(rewards)
        return self.embedding(bin_ids)

    def extra_repr(self) -> str:
        return (
            f"num_bins={self.num_bins}, embedding_dim={self.embedding_dim}, "
            f"reward_min={self.reward_min.item():.3f}, reward_max={self.reward_max.item():.3f}, "
            f"use_linear={self.use_linear}"
        )


__all__ = ["RewardEmbedding"]
