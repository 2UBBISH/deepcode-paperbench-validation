"""Reward discretization + learned embedding table for FRE.

Paper specification (verbatim):

  Sec. 4.1 "Practical Implementation":
    "K encoder states are sampled uniformly from the offline dataset, then labeled
     with a scalar reward according to the given reward function eta. The resulting
     reward is discretized according to magnitude into a learned embedding token
     space. The reward embeddings and states are then concatenated as input to the
     transformer."

  Addendum, "Additional Details on the FRE architecture":
    - "the scalar reward is discretized into 32 bins by rescaling the reward to
       [0, 1] and then multiplying by 32 and flooring to the nearest integer"
    - "The discretized reward is mapped to a continuous vector representation using
       a learned embedding table."
    - "the state embedding is 64-dimensional and the reward embedding is
       64-dimensional, and, concatenated together give a 128-dimensional embedding
       vector."

So: idx = clamp(floor(rescale(r, [0, 1]) * 32), 0, 31), emb = W[idx], W: (32, 64).

Notes / defaults chosen where the paper is silent:
  * The paper does not state the reward range used for the [0, 1] rescaling. The
    prior reward functions (goal-reaching: {-1, 0}; random linear / MLP: bounded,
    roughly [-1, 1]) are defined with a natural range of [-1, 1], so the default
    `reward_low=-1.0, reward_high=1.0` is used. Both bounds are configurable
    (Config.reward_low / Config.reward_high) so a per-domain normalization can be
    supplied if a dataset's rewards exceed this range.
  * Values above `reward_high` land in the top bin (bin num_bins-1) rather than in a
    (num_bins+1)-th bin, matching the "32 bins" statement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def rescale_reward(
    reward: torch.Tensor,
    reward_low: float = -1.0,
    reward_high: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Linearly rescale rewards into the [0, 1] interval.

    ``rescaled = (reward - low) / (high - low)``.
    The result is *not* clamped here (clamping happens inside :func:`discretize_reward`);
    this keeps the function a pure affine transform.
    """
    if reward_high <= reward_low:
        raise ValueError(
            f"reward_high ({reward_high}) must be strictly greater than "
            f"reward_low ({reward_low})"
        )
    denom = max(float(reward_high) - float(reward_low), eps)
    return (reward - float(reward_low)) / denom


def discretize_reward(
    reward: torch.Tensor,
    num_bins: int = 32,
    reward_low: float = -1.0,
    reward_high: float = 1.0,
    clamp: bool = True,
) -> torch.Tensor:
    """Paper-exact reward discretization: rescale to [0, 1], x num_bins, floor.

    Args:
        reward: arbitrary-shaped tensor of scalar rewards.
        num_bins: number of bins (32 in the paper).
        reward_low: value mapped to 0 after rescaling.
        reward_high: value mapped to 1 after rescaling.
        clamp: if True (default), clamp bin indices into ``[0, num_bins - 1]`` so
            out-of-range rewards saturate in the extreme bins.

    Returns:
        ``torch.long`` tensor with the same shape as ``reward``.
    """
    scaled = rescale_reward(reward, reward_low, reward_high)  # in [0, 1] if in range
    idx = torch.floor(scaled * float(num_bins)).to(torch.long)
    if clamp:
        idx = idx.clamp_(0, num_bins - 1)
    return idx


def reward_to_one_hot(
    reward: torch.Tensor,
    num_bins: int = 32,
    reward_low: float = -1.0,
    reward_high: float = 1.0,
) -> torch.Tensor:
    """Convenience helper returning a one-hot encoding of the discretized reward.

    Not used by FRE (which uses a *learned* embedding table) but useful for
    diagnostics / tests and for ablations that want a fixed encoding.
    """
    idx = discretize_reward(reward, num_bins, reward_low, reward_high)
    return torch.nn.functional.one_hot(idx, num_classes=num_bins).float()


class RewardEmbedding(nn.Module):
    """Maps a scalar reward to a continuous embedding vector via a learned table.

    Pipeline (exactly as in the addendum):
        1. rescale the scalar reward to [0, 1]
        2. multiply by ``num_bins`` (32) and floor -> discrete bin index
        3. look the bin index up in a learned ``nn.Embedding`` table of dim
           ``embedding_dim`` (64)

    The module keeps the reward range as buffers so that the discretization is
    checkpointed together with the weights, and remains device/dtype-consistent
    with the input tensor.
    """

    def __init__(
        self,
        num_bins: int = 32,
        embedding_dim: int = 64,
        reward_low: float = -1.0,
        reward_high: float = 1.0,
        clamp: bool = True,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.embedding_dim = int(embedding_dim)
        self.clamp = bool(clamp)

        self.table = nn.Embedding(self.num_bins, self.embedding_dim)
        if init_std is None:
            # Default nn.Embedding init is N(0, 1); a small std keeps the reward and
            # state embeddings on a comparable scale at initialization.
            init_std = 1.0
        nn.init.normal_(self.table.weight, mean=0.0, std=float(init_std))

        self.register_buffer("reward_low", torch.tensor(float(reward_low)))
        self.register_buffer("reward_high", torch.tensor(float(reward_high)))

    # ------------------------------------------------------------------ #
    # discretization
    # ------------------------------------------------------------------ #
    def discretize(self, reward: torch.Tensor) -> torch.Tensor:
        """Return integer bin indices (long tensor) for the given rewards."""
        return discretize_reward(
            reward,
            num_bins=self.num_bins,
            reward_low=float(self.reward_low),
            reward_high=float(self.reward_high),
            clamp=self.clamp,
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """Embed rewards.

        Accepts shapes ``(...)`` (any shape) and returns ``(..., embedding_dim)``.
        A trailing singleton dimension is tolerated so that callers may pass
        ``(batch, K, 1)`` tensors.
        """
        if reward.dim() >= 1 and reward.shape[-1] == 1:
            reward = reward.squeeze(-1)
        idx = self.discretize(reward)
        return self.table(idx)

    # ------------------------------------------------------------------ #
    # factories / utilities
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config) -> "RewardEmbedding":
        """Build from a :class:`fre.config.default.Config`-like object."""
        return cls(
            num_bins=getattr(config, "num_reward_bins", 32),
            embedding_dim=getattr(config, "reward_embedding_dim", 64),
            reward_low=getattr(config, "reward_low", -1.0),
            reward_high=getattr(config, "reward_high", 1.0),
            clamp=getattr(config, "reward_clamp", True),
        )

    def reward_range(self) -> Tuple[float, float]:
        """Return ``(reward_low, reward_high)`` as plain floats."""
        return float(self.reward_low), float(self.reward_high)

    def set_reward_range(self, reward_low: float, reward_high: float) -> None:
        """Update the rescaling range in place (buffers stay on the right device)."""
        self.reward_low.fill_(float(reward_low))
        self.reward_high.fill_(float(reward_high))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"num_bins={self.num_bins}, embedding_dim={self.embedding_dim}, "
            f"reward_low={float(self.reward_low):.3f}, "
            f"reward_high={float(self.reward_high):.3f}, clamp={self.clamp}"
        )


__all__ = [
    "RewardEmbedding",
    "discretize_reward",
    "rescale_reward",
    "reward_to_one_hot",
]
