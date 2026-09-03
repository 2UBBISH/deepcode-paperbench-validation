"""Learned token embeddings for scalar reward values.

The FRE encoder consumes a set of ``(state, reward)`` tokens.  Scalar rewards
are first discretized into ``num_bins`` equally spaced bins covering the
clipped reward range ``[-1, 1]``.  Each bin index is mapped to a learned
embedding and concatenated with a linear projection of the state before being
mapped to the transformer token dimension.

Paper specification
--------------------
* ``B = num_bins = 64`` reward bins over ``[-1, 1]``.
* Reward embedding dimension ``embedding_dim = 64``.
* State token dimension ``state_proj_dim = 192``.
* Final token dimension ``token_dim = 256``.
* No positional encoding and no causal mask (handled by the encoder module).
"""

from __future__ import annotations

import torch
from torch import nn


def reward_to_bins(rewards: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Map scalar rewards in ``[-1, 1]`` to bin indices ``[0, num_bins - 1]``.

    The mapping is ``floor((r + 1) * num_bins / 2)`` followed by clamping, as
    specified in the reproduction plan.  ``rewards`` may have any leading shape;
    the returned tensor has the same shape and dtype ``torch.long``.
    """
    rewards = rewards.float()
    scaled = (rewards + 1.0) * (num_bins / 2.0)
    bin_idx = torch.floor(scaled).long()
    return bin_idx.clamp(min=0, max=num_bins - 1)


class RewardEmbedding(nn.Module):
    """Project ``(state, scalar_reward)`` pairs to transformer tokens.

    Parameters
    ----------
    state_dim:
        Dimensionality of the raw state vectors.
    num_bins:
        Number of reward discretization bins (default 64).
    embedding_dim:
        Learned embedding dimension for each reward bin (default 64).
    state_proj_dim:
        Linear projection dimension for states (default 192).
    token_dim:
        Output token dimension (default 256, the transformer ``d_model``).
    """

    def __init__(
        self,
        state_dim: int,
        num_bins: int = 64,
        embedding_dim: int = 64,
        state_proj_dim: int = 192,
        token_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_bins = int(num_bins)
        self.embedding_dim = int(embedding_dim)
        self.state_proj_dim = int(state_proj_dim)
        self.token_dim = int(token_dim)

        self.reward_embedding = nn.Embedding(self.num_bins, self.embedding_dim)
        self.state_proj = nn.Linear(self.state_dim, self.state_proj_dim)
        self.token_proj = nn.Linear(
            self.state_proj_dim + self.embedding_dim, self.token_dim
        )

    def forward(self, states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Return tokens of shape ``(*batch_dims, token_dim)``.

        ``states`` must have shape ``(..., state_dim)`` and ``rewards`` shape
        ``(...,)``.  Leading batch dimensions must match.
        """
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected last state dimension {self.state_dim}, got {states.shape[-1]}"
            )

        bins = reward_to_bins(rewards, self.num_bins)
        reward_tokens = self.reward_embedding(bins)  # (..., embedding_dim)
        state_tokens = self.state_proj(states)       # (..., state_proj_dim)

        combined = torch.cat([state_tokens, reward_tokens], dim=-1)
        return self.token_proj(combined)             # (..., token_dim)

    def reward_bins(self, rewards: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper exposing bin indices for a reward tensor."""
        return reward_to_bins(rewards, self.num_bins)
