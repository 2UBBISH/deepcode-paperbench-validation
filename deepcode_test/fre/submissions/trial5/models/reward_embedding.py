"""
Reward Embedding Module for Functional Reward Encodings (FRE).

Discretizes scalar rewards into a learned embedding token space for
transformer input. Each reward value is binned and mapped to a learned
embedding vector, enabling the transformer encoder to process
(state, reward) pairs as tokens.

Reference: "Functional Reward Encodings (FRE) for Zero-Shot Offline RL"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RewardEmbedding(nn.Module):
    """
    Discretizes scalar rewards into B bins and maps each bin to a learned
    embedding vector of dimension d_embed.

    The embedding table is learned end-to-end as part of the FRE encoder
    training. Binning can be either fixed-range (if reward_min and reward_max
    are known) or adaptive (computed per-batch from min/max).

    Args:
        num_bins: Number of discretization bins (B). Default: 64.
        embed_dim: Dimension of each reward embedding vector (d_embed).
                   Should match the state feature dimension for concatenation.
        reward_min: Minimum expected reward value for fixed-range binning.
                    If None, adaptive binning is used.
        reward_max: Maximum expected reward value for fixed-range binning.
                    If None, adaptive binning is used.
    """

    def __init__(
        self,
        num_bins: int = 64,
        embed_dim: int = 256,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim
        self.reward_min = reward_min
        self.reward_max = reward_max

        # Learnable embedding table: (num_bins, embed_dim)
        self.embedding = nn.Embedding(num_bins, embed_dim)

        # Initialize embeddings with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def _compute_bin_indices(
        self, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Map scalar rewards to bin indices in [0, num_bins-1].

        Args:
            rewards: Tensor of shape (batch_size,) or (batch_size, 1)
                     containing scalar reward values.

        Returns:
            bin_indices: LongTensor of shape (batch_size,) with bin indices.
            r_min: The minimum reward value used for binning.
            r_max: The maximum reward value used for binning.
        """
        rewards = rewards.view(-1).float()  # Ensure flat float tensor

        if self.reward_min is not None and self.reward_max is not None:
            r_min = self.reward_min
            r_max = self.reward_max
        else:
            # Adaptive binning: use batch min/max
            r_min = rewards.min().item()
            r_max = rewards.max().item()

        # Handle edge case where all rewards are identical
        if r_max - r_min < 1e-8:
            # All rewards are the same; map all to middle bin
            bin_indices = torch.full_like(
                rewards, self.num_bins // 2, dtype=torch.long
            )
            return bin_indices, r_min, r_max

        # Normalize rewards to [0, 1] and discretize
        normalized = (rewards - r_min) / (r_max - r_min)
        # Clamp to [0, 1] to handle values outside the range
        normalized = torch.clamp(normalized, 0.0, 1.0)

        # Map to bin indices [0, num_bins-1]
        # Use floor: bin i covers [i/B, (i+1)/B)
        bin_indices = (normalized * self.num_bins).long()
        # Edge case: if normalized == 1.0, map to last bin
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        return bin_indices, r_min, r_max

    def forward(
        self, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert scalar rewards to embedding vectors.

        Args:
            rewards: Tensor of shape (batch_size,) or (batch_size, 1)
                     containing scalar reward values.

        Returns:
            embeddings: Tensor of shape (batch_size, embed_dim) containing
                        the learned embedding for each reward.
            bin_indices: LongTensor of shape (batch_size,) with the bin
                         index for each reward (useful for debugging).
        """
        bin_indices, _, _ = self._compute_bin_indices(rewards)
        embeddings = self.embedding(bin_indices)
        return embeddings, bin_indices

    def get_bin_centers(self) -> torch.Tensor:
        """
        Return the reward value at the center of each bin (for fixed-range mode).

        Returns:
            Tensor of shape (num_bins,) with center reward values.
        """
        if self.reward_min is None or self.reward_max is None:
            raise ValueError(
                "Bin centers are only defined when reward_min and reward_max "
                "are specified."
            )
        bin_width = (self.reward_max - self.reward_min) / self.num_bins
        centers = torch.linspace(
            self.reward_min + bin_width / 2,
            self.reward_max - bin_width / 2,
            self.num_bins,
        )
        return centers

    def set_reward_range(self, reward_min: float, reward_max: float):
        """
        Set fixed reward range for binning (disables adaptive mode).

        Args:
            reward_min: Minimum expected reward value.
            reward_max: Maximum expected reward value.
        """
        self.reward_min = reward_min
        self.reward_max = reward_max

    def extra_repr(self) -> str:
        return (
            f"num_bins={self.num_bins}, embed_dim={self.embed_dim}, "
            f"reward_min={self.reward_min}, reward_max={self.reward_max}"
        )