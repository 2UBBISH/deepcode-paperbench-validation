"""Permutation-invariant transformer encoder for Functional Reward Encodings (FRE).

Implements the encoder :math:`p_\\theta(z \\mid s^e_1, \\eta(s^e_1), \\ldots, s^e_K, \\eta(s^e_K))`
described in Section 4.1 ("Practical Implementation") of the FRE paper, together with the
architecture clarifications given in the addendum:

* ``K`` encoder states are sampled uniformly from the offline dataset and labeled with a scalar
  reward ``eta(s)``.
* The scalar reward is discretized into 32 bins (rescale to ``[0, 1]``, multiply by 32, floor)
  and mapped through a learned embedding table -> 64-d.
* The state is projected with a learned *linear* map -> 64-d.
* Reward embedding and state embedding are concatenated -> 128-d token.
* No positional encodings and no causal masking are used: the inputs are treated as an unordered
  set.  We therefore use full self-attention over the tokens.
* The residual / attention stream is 128-d; the MLP block inside each transformer layer expands
  to 256 then back to 128 (the appendix's ``Encoder Layers [256, 256, 256, 256]`` refers to these
  MLP dimensions, i.e. four transformer blocks).
* The average over the final-layer token representations is fed to two linear projections that
  parametrize the mean and standard deviation of the Gaussian posterior over the 128-d ``z``.

The mean pooling over tokens makes the encoder permutation invariant: shuffling the ``K`` input
pairs leaves ``z`` unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fre.models.reward_embedding import RewardEmbedding


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block without positional information.

    Attention and residual stream are ``token_dim`` wide; the MLP expands to ``mlp_dim`` and back.
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_dim),
            _get_activation(activation),
            nn.Linear(mlp_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, K, token_dim)`` token representations (already concatenated state+reward).
            key_padding_mask: ``(B, K)`` bool tensor, ``True`` for tokens that should be ignored.

        Returns:
            ``(B, K, token_dim)`` updated token representations.
        """
        if self.norm_first:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.mlp(self.norm2(x)))
        else:
            attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.mlp(x)))
        # Keep masked tokens at a finite value (they are excluded from the pooling below).
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


def _get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'.")


class FREEncoder(nn.Module):
    """Permutation-invariant transformer encoder producing the 128-d latent ``z``.

    Args:
        state_dim: dimensionality of the environment state ``s``.
        latent_dim: dimensionality of ``z`` (128 in the paper).
        state_embed_dim: learned linear projection output width (64).
        reward_embed_dim: reward embedding width (64); ``state_embed_dim + reward_embed_dim``
            gives the 128-d transformer token width.
        num_layers: number of transformer blocks (4, from the appendix MLP dimensions).
        num_heads: attention heads (4).
        mlp_dim: hidden width of the transformer MLP block (256).
        num_reward_bins: number of reward discretization bins (32).
        dropout: dropout probability (0 for the paper configuration).
        activation: activation used inside the MLP blocks.
        norm_first: pre-norm (True, default) vs post-norm transformer blocks.
        log_std_min / log_std_max: clamping range for the posterior log standard deviation.
        normalize_rewards: whether to per-reward-function min/max rescale rewards before binning.
        use_state_bias: add a learned bias to the projected state embedding (harmless; off by
            default so that the token is exactly ``[W s ; emb(reward)]``).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        num_reward_bins: int = 32,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        normalize_rewards: bool = True,
        use_state_bias: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.state_embed_dim = int(state_embed_dim)
        self.reward_embed_dim = int(reward_embed_dim)
        self.token_dim = self.state_embed_dim + self.reward_embed_dim
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.normalize_rewards = bool(normalize_rewards)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Learned linear projection of the raw environment state.
        self.state_projection = nn.Linear(self.state_dim, self.state_embed_dim)
        self.state_bias = (
            nn.Parameter(torch.zeros(self.state_embed_dim)) if use_state_bias else None
        )

        # Discretized reward -> learned embedding table.
        self.reward_embedding = RewardEmbedding(
            num_bins=num_reward_bins,
            reward_embed_dim=self.reward_embed_dim,
            normalize=self.normalize_rewards,
        )

        # Unordered set of tokens: full self-attention, no positional encodings, no causal mask.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    token_dim=self.token_dim,
                    num_heads=self.num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.token_dim)

        # Two linear heads parametrize the Gaussian posterior p_theta(z | L^e).
        self.mean_head = nn.Linear(self.token_dim, self.latent_dim)
        self.log_std_head = nn.Linear(self.token_dim, self.latent_dim)

        self._init_parameters()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_parameters(self) -> None:
        nn.init.orthogonal_(self.state_projection.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.state_projection.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    # ------------------------------------------------------------------ #
    # Token construction
    # ------------------------------------------------------------------ #
    def build_tokens(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        reward_mask: Optional[torch.Tensor] = None,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """Concatenate the projected state embedding and the reward embedding.

        Args:
            states: ``(B, K, state_dim)`` encoder states (or ``(..., state_dim)``).
            rewards: ``(B, K)`` (or ``(..., K)``) scalar rewards ``eta(s)``.
            reward_mask: optional ``(B, K)`` mask restricting the reward min/max rescaling.
            already_normalized: set when the caller has already rescaled rewards to ``[0, 1]``.

        Returns:
            ``(B, K, token_dim)`` tokens with ``token_dim = state_embed_dim + reward_embed_dim``.
        """
        if states.dim() != 3:
            raise ValueError(f"Expected states of shape (B, K, state_dim), got {tuple(states.shape)}")
        if rewards.dim() != rewards.dim():
            raise ValueError("unreachable")  # pragma: no cover
        if rewards.dim() != states.dim() - 1:
            raise ValueError(
                f"Expected rewards of shape (B, K), got {tuple(rewards.shape)} for states "
                f"{tuple(states.shape)}"
            )

        state_emb = self.state_projection(states)
        if self.state_bias is not None:
            state_emb = state_emb + self.state_bias

        reward_emb = self.reward_embedding(
            rewards, mask=reward_mask, already_normalized=already_normalized
        )
        if reward_emb.shape[-1] != self.reward_embed_dim:
            raise ValueError(
                f"Reward embedding width {reward_emb.shape[-1]} does not match configured "
                f"{self.reward_embed_dim}."
            )
        return torch.cat([state_emb, reward_emb], dim=-1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample: bool = True,
        already_normalized: bool = False,
        deterministic: Optional[bool] = None,
        return_tokens: bool = False,
    ) -> dict:
        """Encode a set of reward-labeled states into the Gaussian posterior over ``z``.

        Args:
            states: ``(B, K, state_dim)`` encoder states.
            rewards: ``(B, K)`` rewards ``eta(s)``.
            mask: optional ``(B, K)`` boolean/float mask; ``True`` marks *valid* tokens.  Invalid
                tokens are excluded from attention and from the mean pooling.
            sample: if ``True`` reparameterize ``z = mu + sigma * eps``; otherwise return the mean.
            already_normalized: pass through to :class:`RewardEmbedding`.
            deterministic: (deprecated alias) ``True`` behaves like ``sample=False``.
            return_tokens: also return the final-layer token representations.

        Returns:
            dict with keys ``z`` ``(B, latent_dim)``, ``mean``, ``std``, ``log_std`` and optionally
            ``tokens`` ``(B, K, token_dim)``.
        """
        if deterministic is not None:
            sample = not bool(deterministic)

        key_padding_mask = self._prepare_mask(mask, states.shape[0], states.shape[1], states.device)
        tokens = self.build_tokens(
            states, rewards, reward_mask=mask, already_normalized=already_normalized
        )
        if key_padding_mask is not None:
            tokens = tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        h = tokens
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        h = self.final_norm(h)

        pooled = self._masked_mean(h, mask if key_padding_mask is not None else None)
        mean = self.mean_head(pooled)
        log_std = self.log_std_head(pooled).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if sample:
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        out = {"z": z, "mean": mean, "std": std, "log_std": log_std}
        if return_tokens:
            out["tokens"] = h
        return out

    def encode(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample: bool = False,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """Convenience wrapper returning only ``z`` (posterior mean by default)."""
        return self.forward(
            states,
            rewards,
            mask=mask,
            sample=sample,
            already_normalized=already_normalized,
        )["z"]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare_mask(
        mask: Optional[torch.Tensor], batch_size: int, num_tokens: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Normalize an arbitrary mask into a ``(B, K)`` boolean tensor (True = ignore)."""
        if mask is None:
            return None
        m = mask.to(device)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(batch_size, -1)
        if m.shape != (batch_size, num_tokens):
            raise ValueError(
                f"Mask shape {tuple(m.shape)} incompatible with {(batch_size, num_tokens)}."
            )
        # Convention inside this module: True == valid.  ``key_padding_mask`` needs True == ignore.
        return m < 0.5

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Average over the token dimension, optionally ignoring masked-out tokens."""
        if valid_mask is None:
            return tokens.mean(dim=1)
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1e-6)
        return (tokens * weights).sum(dim=1) / denom


def kl_divergence_to_unit_gaussian(
    mean: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """``D_KL( N(mean, std^2) || N(0, I) )`` summed over the latent dimensions (per example)."""
    return 0.5 * (mean.pow(2) + torch.exp(2.0 * log_std) - 1.0 - 2.0 * log_std).sum(dim=-1)


__all__ = ["FREEncoder", "TransformerBlock", "kl_divergence_to_unit_gaussian"]
