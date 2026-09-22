"""FRE encoder: permutation-invariant transformer variational encoder.

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

The encoder maps a set of ``K`` reward-labelled states (a *context set*) to a
Gaussian posterior over a latent task embedding ``z`` (Section 4.1):

    Encoder: p_theta(z | s_1^e, eta(s_1^e), s_2^e, eta(s_2^e), ..., s_K^e, eta(s_K^e))

Practical implementation (Section 4.1 / 5 / Addendum):

* ``K`` states are sampled uniformly from the offline dataset and labelled with a
  scalar reward by the sampled reward function ``eta``;
* the scalar reward is discretized into 32 bins by rescaling to ``[0, 1]``,
  multiplying by 32 and flooring to the nearest integer (we additionally clip the
  index into ``[0, 31]`` so it indexes a 32-row embedding table -- the paper does
  not specify the clipping, see ``fre.utils.reward_discretize``);
* the discrete reward is mapped to a continuous vector with a learned embedding
  table and the state is projected to an embedding with a learned linear
  transformation; for each state the reward embedding is concatenated to the end
  of the state embedding;
* the state embedding is 64-d, the reward embedding is 64-d, together 128-d; the
  latent ``z`` is 128-d;
* the transformer has **no positional encodings and no causal masking** -- the
  inputs are treated as an unordered set;
* the average of the final layer representations is passed to two linear
  projections parametrizing the mean and the standard deviation of the Gaussian
  ``p_theta(z | .)``.

Hyper-parameters (Table 3): 32 reward embeddings, 4 attention heads, ``beta=0.01``,
encoder MLP layers ``[256, 256, 256, 256]`` (i.e. the residual/attention width is
128 and each MLP block expands to 256 then back to 128, so there are 4 blocks).
The number of transformer blocks is not stated explicitly in the paper; the
addendum's four MLP dimensions imply 4 blocks (documented default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fre.utils.reward_discretize import (
        MAX_REWARD_INDEX,
        NUM_REWARD_EMBEDDINGS,
        discretize_reward,
        rescale_reward,
    )
except Exception:  # pragma: no cover - direct module execution fallback
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from fre.utils.reward_discretize import (  # type: ignore
        MAX_REWARD_INDEX,
        NUM_REWARD_EMBEDDINGS,
        discretize_reward,
        rescale_reward,
    )


__all__ = [
    "FREEncoder",
    "EncoderOutput",
    "TransformerBlock",
    "make_fre_encoder",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_TOKEN_DIM",
    "DEFAULT_STATE_EMBED_DIM",
    "DEFAULT_REWARD_EMBED_DIM",
    "DEFAULT_NUM_BLOCKS",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_MLP_DIM",
    "DEFAULT_NUM_ENCODER_STATES",
    "EncoderInputs",
]


# --------------------------------------------------------------------------------------
# Defaults taken from the paper (Table 3 / Section 4.1 / Addendum)
# --------------------------------------------------------------------------------------
DEFAULT_STATE_EMBED_DIM: int = 64       # addendum: state embedding is 64-d
DEFAULT_REWARD_EMBED_DIM: int = 64      # addendum: reward embedding is 64-d
DEFAULT_TOKEN_DIM: int = 128            # 64 + 64 concatenated
DEFAULT_LATENT_DIM: int = 128           # addendum: z is 128-d
DEFAULT_NUM_BLOCKS: int = 4             # addendum: [256] x 4 MLP dims -> 4 blocks
DEFAULT_NUM_HEADS: int = 4              # Table 3: "Encoder Attention Heads = 4"
DEFAULT_MLP_DIM: int = 256              # Table 3: "Encoder Layers = [256,256,256,256]"
DEFAULT_NUM_ENCODER_STATES: int = 32    # Table 3: "Reward Pairs to Encode = 32"
DEFAULT_NUM_REWARD_EMBEDDINGS: int = NUM_REWARD_EMBEDDINGS  # Table 3: 32
DEFAULT_REWARD_MIN: float = -1.0
DEFAULT_REWARD_MAX: float = 1.0
DEFAULT_BETA: float = 0.01              # Table 3: beta KL weight
DEFAULT_LOG_STD_MIN: float = -5.0       # numerical stability clamp (not in the paper)
DEFAULT_LOG_STD_MAX: float = 2.0        # numerical stability clamp (not in the paper)


@dataclass
class EncoderInputs:
    """Container for the token inputs of the encoder.

    Attributes:
        states: ``(B, K, state_dim)`` reward-labelled states from the context set.
        rewards: ``(B, K)`` scalar rewards ``eta(s)`` for those states.
    """

    states: torch.Tensor
    rewards: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_states(self) -> int:
        return int(self.states.shape[1])


@dataclass
class EncoderOutput:
    """Output of :class:`FREEncoder`.

    Attributes:
        z: ``(B, latent_dim)`` sampled latent (posterior sample when ``sample=True``,
            otherwise the posterior mean).
        mu: ``(B, latent_dim)`` posterior mean.
        log_std: ``(B, latent_dim)`` posterior log standard deviation (clamped).
        std: ``(B, latent_dim)`` posterior standard deviation.
        token_embeddings: optional ``(B, K, token_dim)`` final-layer representations
            before mean pooling (useful for diagnostics / OPAL-style re-use).
    """

    z: torch.Tensor
    mu: torch.Tensor
    log_std: torch.Tensor
    std: torch.Tensor
    token_embeddings: Optional[torch.Tensor] = field(default=None)

    @property
    def variance(self) -> torch.Tensor:
        return self.std ** 2

    def kl_to_unit_gaussian(self) -> torch.Tensor:
        """``D_KL(N(mu, sigma) || N(0, I))`` per batch element (Section 4.1, eq. 6).

        The uninformative prior ``u(z)`` is defined as the unit Gaussian, so
            KL = 0.5 * sum_i (mu_i^2 + sigma_i^2 - 1 - 2 log sigma_i).
        """
        kl = 0.5 * (self.mu.pow(2) + self.std.pow(2) - 1.0 - 2.0 * self.log_std)
        return kl.sum(dim=-1)

    def mean_kl_to_unit_gaussian(self) -> torch.Tensor:
        """Mean over the batch of :meth:`kl_to_unit_gaussian` (scalar tensor)."""
        return self.kl_to_unit_gaussian().mean()


def reparameterize(mu: torch.Tensor, std: torch.Tensor, sample: bool = True) -> torch.Tensor:
    """Reparameterization trick ``z = mu + sigma * eps``, ``eps ~ N(0, I)``."""
    if not sample:
        return mu
    eps = torch.randn_like(std)
    return mu + eps * std


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with no positional encoding and no causal mask.

    The addendum states that the "Encoder Layers" list ``[256, 256, 256, 256]``
    refers to the *MLP dimensions*: the residual/attention activations are all
    128-dimensional and the MLP block expands to 256 then back to 128.
    """

    def __init__(
        self,
        token_dim: int = DEFAULT_TOKEN_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_dim: int = DEFAULT_MLP_DIM,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)

        self.norm1 = nn.LayerNorm(token_dim)
        # batch_first=True: inputs are (B, K, token_dim) -- positional encodings are
        # intentionally omitted, causal masking is intentionally disabled.
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(mlp_dim, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: ``(B, K, token_dim)``. Returns ``(B, K, token_dim)``."""
        # self-attention over the unordered set (no causal mask -> no attn_mask)
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False, attn_mask=None)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class FREEncoder(nn.Module):
    """Permutation-invariant transformer variational encoder ``p_theta(z | context)``.

    Inputs are ``(B, K, state_dim)`` states and ``(B, K)`` rewards. The reward is
    discretized into 32 bins and looked up in a learned embedding table; the state
    is projected to a 64-d embedding with a learned linear layer; the two are
    concatenated into a 128-d token. The tokens are processed by a stack of
    transformer blocks without positional encodings or causal masking, mean-pooled
    over the set, and mapped to the mean / log-std of a Gaussian over ``z``.

    Example:
        >>> enc = FREEncoder(state_dim=29)
        >>> out = enc(torch.randn(4, 32, 29), torch.rand(4, 32) * 2 - 1)
        >>> out.z.shape, out.mu.shape
        (torch.Size([4, 128]), torch.Size([4, 128]))
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        state_embed_dim: int = DEFAULT_STATE_EMBED_DIM,
        reward_embed_dim: int = DEFAULT_REWARD_EMBED_DIM,
        num_reward_embeddings: int = DEFAULT_NUM_REWARD_EMBEDDINGS,
        num_layers: int = DEFAULT_NUM_BLOCKS,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_dim: int = DEFAULT_MLP_DIM,
        dropout: float = 0.0,
        activation: str = "relu",
        reward_min: float = DEFAULT_REWARD_MIN,
        reward_max: float = DEFAULT_REWARD_MAX,
        use_state_norm: bool = False,
        state_mean: Optional[torch.Tensor] = None,
        state_std: Optional[torch.Tensor] = None,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.state_embed_dim = int(state_embed_dim)
        self.reward_embed_dim = int(reward_embed_dim)
        self.num_reward_embeddings = int(num_reward_embeddings)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.token_dim = int(state_embed_dim) + int(reward_embed_dim)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Learned linear projection of the state (addendum: "the environment state
        # projected into an embedding using a learned linear transformation").
        self.state_projection = nn.Linear(self.state_dim, self.state_embed_dim)
        # Learned embedding table for the 32 discrete reward bins.
        self.reward_embedding = nn.Embedding(self.num_reward_embeddings, self.reward_embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    token_dim=self.token_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Two linear projections parametrizing the mean and the standard deviation
        # of the Gaussian posterior (Section 4.1, "Practical Implementation").
        self.mu_head = nn.Linear(self.token_dim, self.latent_dim)
        self.log_std_head = nn.Linear(self.token_dim, self.latent_dim)

        # Optional observation normalization (disabled by default; dataset
        # statistics normalization is handled upstream where needed).
        self.use_state_norm = bool(use_state_norm)
        if self.use_state_norm:
            if state_mean is None or state_std is None:
                raise ValueError("state_mean and state_std are required when use_state_norm=True")
            self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
            self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        else:
            self.register_buffer("state_mean", torch.zeros(self.state_dim))
            self.register_buffer("state_std", torch.ones(self.state_dim))

        self._init_parameters()

    # ------------------------------------------------------------------ helpers
    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.state_projection.weight)
        nn.init.zeros_(self.state_projection.bias)
        # The reward embedding table is learned; small init keeps the concatenated
        # token scale comparable between the state and reward halves.
        nn.init.normal_(self.reward_embedding.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.mlp[-1].weight)
            nn.init.zeros_(block.mlp[-1].bias)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def discretize_rewards(
        self,
        rewards: torch.Tensor,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> torch.Tensor:
        """Reward -> embedding index in ``[0, num_reward_embeddings - 1]``.

        Follows the addendum: rescale the reward to ``[0, 1]``, multiply by 32,
        floor to the nearest integer. The resulting index is clipped into the valid
        embedding range (``floor(1.0 * 32) == 32`` would otherwise index outside a
        32-row table; the paper does not specify this clipping).
        """
        lo = self.reward_min if reward_min is None else float(reward_min)
        hi = self.reward_max if reward_max is None else float(reward_max)
        idx = discretize_reward(
            rewards,
            reward_min=lo,
            reward_max=hi,
            num_embeddings=self.num_reward_embeddings,
            clip=True,
        )
        idx = torch.clamp(idx, min=0, max=self.num_reward_embeddings - 1)
        return idx.long()

    def make_tokens(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> torch.Tensor:
        """Build the ``(B, K, 128)`` token set from reward-labelled states.

        Per-token: ``concat(state_embedding_64, reward_embedding_64)``.
        """
        if states.dim() != 3:
            raise ValueError(f"states must be (B, K, state_dim), got shape {tuple(states.shape)}")
        if rewards.dim() == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.dim() != 2:
            raise ValueError(f"rewards must be (B, K), got shape {tuple(rewards.shape)}")
        if states.shape[:2] != rewards.shape[:2]:
            raise ValueError(
                f"states {tuple(states.shape[:2])} and rewards {tuple(rewards.shape[:2])} "
                "must agree on (batch, num_states)"
            )

        states = states.to(dtype=self.state_projection.weight.dtype)
        rewards = rewards.to(dtype=self.state_projection.weight.dtype)

        if self.use_state_norm:
            states = (states - self.state_mean) / (self.state_std + 1e-6)

        reward_idx = self.discretize_rewards(
            rewards, reward_min=reward_min, reward_max=reward_max
        )
        reward_tokens = self.reward_embedding(reward_idx)  # (B, K, reward_embed_dim)
        state_tokens = self.state_projection(states)       # (B, K, state_embed_dim)
        tokens = torch.cat([state_tokens, reward_tokens], dim=-1)  # (B, K, token_dim)
        return tokens

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        sample: bool = True,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        return_tokens: bool = False,
    ) -> EncoderOutput:
        """Encode reward-labelled states into the Gaussian posterior over ``z``.

        Args:
            states: ``(B, K, state_dim)`` context states.
            rewards: ``(B, K)`` rewards ``eta(s)`` for the context states.
            sample: if ``True`` draw ``z ~ p_theta(z | .)`` (training); if ``False``
                return the posterior mean (deterministic evaluation default).
            reward_min / reward_max: reward range used for the 32-bin
                discretization; defaults to the encoder's configured range.
            return_tokens: also return the pooled final representations.

        Returns:
            :class:`EncoderOutput` with ``z``, ``mu``, ``log_std``, ``std``.
        """
        tokens = self.make_tokens(states, rewards, reward_min=reward_min, reward_max=reward_max)

        for block in self.blocks:
            tokens = block(tokens)

        pooled = tokens.mean(dim=1)  # mean over the unordered set of K tokens

        mu = self.mu_head(pooled)
        log_std = torch.clamp(self.log_std_head(pooled), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        z = reparameterize(mu, std, sample=sample)

        return EncoderOutput(
            z=z,
            mu=mu,
            log_std=log_std,
            std=std,
            token_embeddings=tokens if return_tokens else None,
        )

    def encode(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        sample: bool = False,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> torch.Tensor:
        """Convenience wrapper returning only the latent ``z``."""
        return self.forward(
            states, rewards, sample=sample, reward_min=reward_min, reward_max=reward_max
        ).z

    # ------------------------------------------------------- prior-fitting utils
    def set_reward_range(self, reward_min: float, reward_max: float) -> None:
        """Bind the discretization range to a sampled reward function's range.

        Sections B / Addendum do not specify per-``eta`` re-binding; the trainer may
        either keep a fixed ``[-1, 1]`` range (all three prior classes produce
        rewards within ``[-1, 1]``) or re-bind per sampled ``eta``.
        """
        if not math.isfinite(reward_min) or not math.isfinite(reward_max) or reward_max <= reward_min:
            return
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)

    def fit_reward_range_from_function(self, reward_function, states: Optional[torch.Tensor] = None) -> None:
        """Set the reward range from a prior reward function's ``reward_bounds``."""
        lo = getattr(reward_function, "reward_min", None)
        hi = getattr(reward_function, "reward_max", None)
        bounds = getattr(reward_function, "reward_bounds", None)
        if bounds is not None and (lo is None or hi is None):
            try:
                lo, hi = bounds(states)
            except Exception:
                pass
        if lo is not None and hi is not None:
            self.set_reward_range(float(lo), float(hi))

    def describe(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "state_embed_dim": self.state_embed_dim,
            "reward_embed_dim": self.reward_embed_dim,
            "token_dim": self.token_dim,
            "num_reward_embeddings": self.num_reward_embeddings,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "reward_range": (self.reward_min, self.reward_max),
            "positional_encoding": False,
            "causal_masking": False,
            "pooling": "mean",
        }


def make_fre_encoder(
    state_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    **kwargs,
) -> FREEncoder:
    """Factory matching the paper's hyper-parameters (K=32, 4 heads, 4 blocks)."""
    return FREEncoder(state_dim=state_dim, latent_dim=latent_dim, **kwargs)
