"""Reward discretization + embedding module for FRE.

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings"

Each (state, reward) context pair is turned into a single 128-d token:

    token = [ state_projection(s) (64-d) ; reward_embedding(bin(eta(s))) (64-d) ]

The scalar reward is rescaled from ``[-1, 1]`` to ``[0, 1]``, multiplied by 32,
floored (i.e. discretized into 32 uniformly-spaced bins) and looked up in a
learned embedding table of shape ``(32, 64)``.

This module centralizes:

* :func:`discretize_reward`   -- reward -> integer bin index in ``[0, 31]``
* :class:`StateProjection`    -- learned linear map ``state -> 64-d``
* :class:`RewardEmbedding`    -- 32 x 64 embedding table
* :class:`RewardEncoderToken` -- concat of the two, producing 128-d tokens

All functions operate on arbitrary leading batch dimensions so the same code
path can be used for the encoder context (K=32 pairs) and for evaluation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (corrected addendum values)
# ---------------------------------------------------------------------------

#: Number of uniformly-spaced reward bins.
NUM_REWARD_BINS: int = 32

#: Multiplicative factor used by the discretization: ``floor(rescale * r)``.
REWARD_RESCALE: float = 32.0

#: Dimensionality of the learned state projection.
STATE_EMB_DIM: int = 64

#: Dimensionality of the learned reward-bin embedding.
REWARD_EMB_DIM: int = 64

#: Resulting per-pair token dimensionality (64 + 64 = 128).
TOKEN_DIM: int = STATE_EMB_DIM + REWARD_EMB_DIM

#: Assumed range of reward functions produced by the prior ``p(eta)``.
REWARD_MIN: float = -1.0
REWARD_MAX: float = 1.0

__all__ = [
    "NUM_REWARD_BINS",
    "REWARD_RESCALE",
    "STATE_EMB_DIM",
    "REWARD_EMB_DIM",
    "TOKEN_DIM",
    "REWARD_MIN",
    "REWARD_MAX",
    "discretize_reward",
    "reward_to_one_hot",
    "StateProjection",
    "RewardEmbedding",
    "RewardEncoderToken",
]


# ---------------------------------------------------------------------------
# Reward discretization
# ---------------------------------------------------------------------------


def discretize_reward(reward: torch.Tensor) -> torch.Tensor:
    """Discretize a scalar reward into one of ``NUM_REWARD_BINS`` bins.

    The reward is assumed to live in ``[-1, 1]``.  It is rescaled to ``[0, 1]``,
    multiplied by 32 and floored, then clamped to ``[0, 31]`` so that values
    outside the assumed range (or floating-point edge cases at ``r == 1``) map
    to a valid bin.

    Args:
        reward: Tensor of arbitrary shape holding rewards in ``[-1, 1]``.

    Returns:
        ``LongTensor`` with the same shape as ``reward`` holding bin indices in
        ``[0, NUM_REWARD_BINS - 1]``.
    """
    if not torch.is_tensor(reward):
        reward = torch.as_tensor(reward)
    reward = reward.to(torch.float32)

    # Rescale [-1, 1] -> [0, 1].
    rescaled = (reward - REWARD_MIN) / (REWARD_MAX - REWARD_MIN)
    # Discretize into 32 bins.
    scaled = rescaled * float(NUM_REWARD_BINS)
    bins = torch.floor(scaled)
    bins = torch.clamp(bins, min=0.0, max=float(NUM_REWARD_BINS - 1))
    return bins.to(torch.long)


def reward_to_one_hot(reward: torch.Tensor) -> torch.Tensor:
    """Return a one-hot ``(..., NUM_REWARD_BINS)`` encoding of a scalar reward."""
    bins = discretize_reward(reward)
    return F.one_hot(bins, num_classes=NUM_REWARD_BINS).to(torch.float32)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class StateProjection(nn.Module):
    """Learned linear projection of a raw state to a 64-d embedding."""

    def __init__(
        self,
        state_dim: int,
        emb_dim: int = STATE_EMB_DIM,
        bias: bool = True,
        init_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.emb_dim = int(emb_dim)
        self.linear = nn.Linear(self.state_dim, self.emb_dim, bias=bias)
        if init_std is not None and init_std > 0:
            bound = init_std / max(1.0, (self.state_dim ** 0.5))
            nn.init.uniform_(self.linear.weight, -bound, bound)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Map ``(..., state_dim)`` -> ``(..., emb_dim)``."""
        return self.linear(states)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"state_dim={self.state_dim}, emb_dim={self.emb_dim}"


class RewardEmbedding(nn.Module):
    """Learned embedding of a discretized scalar reward.

    The forward pass accepts continuous rewards, discretizes them into 32 bins
    and returns the corresponding ``emb_dim``-dimensional embedding vectors.
    """

    def __init__(
        self,
        num_bins: int = NUM_REWARD_BINS,
        emb_dim: int = REWARD_EMB_DIM,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.emb_dim = int(emb_dim)
        self.embedding = nn.Embedding(self.num_bins, self.emb_dim)
        if init_std is not None and init_std > 0:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """Map ``(...,)`` rewards -> ``(..., emb_dim)`` embeddings."""
        bins = discretize_reward(reward)
        return self.embedding(bins)

    def forward_bins(self, bins: torch.Tensor) -> torch.Tensor:
        """Map pre-computed ``(...,)`` integer bins -> ``(..., emb_dim)``."""
        return self.embedding(bins.to(torch.long))

    def table(self) -> torch.Tensor:
        """Return the raw embedding table ``(num_bins, emb_dim)``."""
        return self.embedding.weight

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"num_bins={self.num_bins}, emb_dim={self.emb_dim}"


class RewardEncoderToken(nn.Module):
    """Build 128-d tokens from ``(state, reward)`` pairs.

    Token = ``[ state_projection(state) ; reward_embedding(discretize(reward)) ]``

    Each half is 64-d (see the corrected addendum), yielding 128-d tokens that
    match the transformer's hidden dimension.
    """

    def __init__(
        self,
        state_dim: int,
        state_emb_dim: int = STATE_EMB_DIM,
        reward_emb_dim: int = REWARD_EMB_DIM,
        num_bins: int = NUM_REWARD_BINS,
        reward_init_std: float = 0.02,
        state_init_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.state_projection = StateProjection(
            state_dim, state_emb_dim, init_std=state_init_std
        )
        self.reward_embedding = RewardEmbedding(
            num_bins, reward_emb_dim, init_std=reward_init_std
        )
        self.token_dim = int(state_emb_dim) + int(reward_emb_dim)

    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        return_parts: bool = False,
    ):
        """Encode ``(..., state_dim)`` states and ``(...,)`` rewards into tokens.

        Args:
            states: ``(..., state_dim)`` float tensor.
            rewards: ``(...,)`` float tensor broadcastable to ``states.shape[:-1]``.
            return_parts: when ``True`` also return the ``(state_emb, reward_emb)``
                pair of individual halves.

        Returns:
            ``(..., token_dim)`` tensor, or a tuple when ``return_parts=True``.
        """
        state_emb = self.state_projection(states)

        rewards = rewards.to(state_emb.dtype)
        if rewards.shape != state_emb.shape[:-1]:
            rewards = rewards.reshape(state_emb.shape[:-1])

        reward_emb = self.reward_embedding(rewards)

        token = torch.cat([state_emb, reward_emb], dim=-1)
        if return_parts:
            return token, (state_emb, reward_emb)
        return token

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        """Project states only (without reward embedding)."""
        return self.state_projection(states)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"state_dim={self.state_dim}, "
            f"state_emb_dim={self.state_projection.emb_dim}, "
            f"reward_emb_dim={self.reward_embedding.emb_dim}, "
            f"token_dim={self.token_dim}"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_encoder_tokenizer(
    state_dim: int,
    state_emb_dim: int = STATE_EMB_DIM,
    reward_emb_dim: int = REWARD_EMB_DIM,
) -> Tuple[RewardEncoderToken, int]:
    """Factory returning a :class:`RewardEncoderToken` and its token dimension."""
    tok = RewardEncoderToken(state_dim, state_emb_dim, reward_emb_dim)
    return tok, tok.token_dim


def check_reward_range(reward: torch.Tensor, tol: float = 1e-3) -> bool:
    """Return ``True`` if all rewards lie within ``[-1, 1]`` (up to ``tol``)."""
    return bool(
        (reward.min().item() >= REWARD_MIN - tol)
        and (reward.max().item() <= REWARD_MAX + tol)
    )
