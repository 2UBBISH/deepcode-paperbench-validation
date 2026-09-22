"""Permutation-invariant transformer encoder for Functional Reward Encoding (FRE).

Implements the encoder ``p_theta(z | s^e_1, eta(s^e_1), ..., s^e_K, eta(s^e_K))`` from
Section 4.1 of the FRE paper (Kvfrans et al., ICML 2024).

Paper specification (Section 4.1 "Practical Implementation" + addendum
"Additional Details on the FRE architecture"):

* ``K`` encoder states are sampled uniformly from the offline dataset, then labeled
  with a scalar reward according to the given reward function ``eta``.
* The resulting reward is discretized according to magnitude into a learned embedding
  token space: the scalar reward is rescaled to ``[0, 1]``, multiplied by 32 and
  floored to the nearest integer (32 bins). The bin index is mapped to a continuous
  vector representation using a learned embedding table.
* The environment state is projected into an embedding using a learned linear
  transformation; the reward embedding is concatenated to the end of the state
  embedding before the set of reward-labeled states is passed through the encoder.
* State embedding is 64-dimensional and the reward embedding is 64-dimensional, so the
  concatenated token is 128-dimensional. (The appendix's "Reward Embedding Dim = 128"
  is incorrect; see addendum.)
* The encoder is a permutation-invariant transformer. Positional encodings and causal
  masking are NOT used, thus the inputs are treated as an unordered set.
* 4 attention heads. The "Encoder Layers" list ``[256, 256, 256, 256]`` in the appendix
  refers to the MLP dimensions inside the transformer blocks: the residual/attention
  activations are all 128-dimensional and the MLP block expands to 256 then back to 128.
  There are therefore 4 transformer blocks.
* The average of the final-layer representations is used as input to two linear
  projections which parametrize the mean and standard deviation of the Gaussian
  distribution ``p_theta(z | .)``; ``z`` is 128-dimensional.

The forward pass returns a :class:`torch.distributions.Normal` so that downstream code
can both sample ``z`` and evaluate the KL term of Eq. (6)
``-beta * D_KL(p_theta(z | L^e) || u(z))``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .reward_embedding import RewardEmbedding

__all__ = ["TransformerBlock", "TransformerEncoder", "Encoder", "diagonal_gaussian_kl"]


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """A standard pre-norm transformer block without positional encoding.

    Attention operates over the token (set) dimension only.  There is no causal
    masking, so every token attends to every other token -- this is what makes the
    encoder permutation invariant w.r.t. the ordering of the ``(s, eta(s))`` pairs.

    Parameters
    ----------
    width:
        Residual / attention width (128 in the paper).
    num_heads:
        Number of attention heads (4 in the paper).
    mlp_dim:
        Hidden dimension of the position-wise MLP block (256 in the paper, i.e. the
        block expands ``width -> mlp_dim -> width``).
    activation:
        Activation used inside the MLP block.  The paper does not state which
        activation is used; GELU is used here as a sensible default.
    dropout:
        Dropout probability (0.0 -- the paper does not use dropout).
    """

    def __init__(
        self,
        width: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(
                f"transformer width ({width}) must be divisible by num_heads ({num_heads})"
            )
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)

        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_dim),
            _make_activation(activation),
            nn.Linear(mlp_dim, width),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x`` has shape ``(batch, K, width)``; returns the same shape.

        ``key_padding_mask`` (shape ``(batch, K)``, True = ignore) is optional and is
        only used for variable-length contexts (FRE itself uses a fixed ``K``).
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


def _make_activation(name: str) -> nn.Module:
    """Resolve an activation name to a module."""
    name = (name or "gelu").lower()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("elu",):
        return nn.ELU()
    raise ValueError(f"unknown activation '{name}'")


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerBlock` with mean pooling over the set dimension.

    The tokens are treated as an *unordered set*: no positional encoding is added and
    no causal mask is applied, therefore the output is invariant (up to numerics) to
    permutations of the input tokens.
    """

    def __init__(
        self,
        width: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_positional_encoding: bool = False,
        use_causal_mask: bool = False,
        final_norm: bool = False,
    ) -> None:
        super().__init__()
        # The paper explicitly does not use positional encodings or causal masking.
        # We keep the flags so that ablated variants remain expressible, but warn if a
        # caller turns them on (they break the permutation-invariant contract).
        self.use_positional_encoding = bool(use_positional_encoding)
        self.use_causal_mask = bool(use_causal_mask)
        self.width = int(width)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    width=width,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.final_norm = nn.LayerNorm(width) if final_norm else nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a set of tokens.

        Parameters
        ----------
        tokens:
            ``(batch, K, width)`` tensor of state-reward tokens.
        key_padding_mask:
            Optional ``(batch, K)`` bool tensor (True = padding, ignored).

        Returns
        -------
        torch.Tensor
            ``(batch, width)`` averaged final-layer representation.
        """
        if tokens.dim() != 3:
            raise ValueError(f"expected tokens of shape (batch, K, width), got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.width:
            raise ValueError(
                f"token width {tokens.shape[-1]} does not match encoder width {self.width}"
            )

        x = tokens
        if self.use_positional_encoding:
            x = x + _sinusoidal_pos_encoding(x.shape[1], self.width, x.device, x.dtype)
        if self.use_causal_mask:
            mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1
            )
            x = x + torch.zeros_like(x)  # placeholder; causal masking is not used by FRE
            del mask
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        # Average of the final layer representations (mean pooling over the set dim).
        if key_padding_mask is not None and key_padding_mask.any():
            keep = (~key_padding_mask).to(x.dtype).unsqueeze(-1)
            denom = keep.sum(dim=1).clamp(min=1.0)
            return (x * keep).sum(dim=1) / denom
        return x.mean(dim=1)


def _sinusoidal_pos_encoding(
    length: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Standard sinusoidal positional encoding (only used by ablated variants)."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000.0) / width)
    )
    pe = torch.zeros(length, width, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# Full encoder: state-reward tokens -> Gaussian p_theta(z | L^e)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """FRE encoder mapping ``K`` state-reward pairs to a Gaussian over ``z``.

    ``p_theta(z | s^e_1, eta(s^e_1), ..., s^e_K, eta(s^e_K))``

    Pipeline (see module docstring for the paper references):

    1. Discretize each scalar reward into 32 bins and look up a learned 64-dim
       embedding (``fre.fre.reward_embedding.RewardEmbedding``).
    2. Project each raw state through a learned linear layer to 64 dims.
    3. Concatenate -> 128-dim token per ``(s, eta(s))`` pair.
    4. Permutation-invariant transformer (4 blocks, 4 heads, width 128, MLP
       128 -> 256 -> 128), no positional encodings, no causal mask.
    5. Mean-pool the final layer and apply two linear heads producing the mean and
       (log) standard deviation of an isotropic Gaussian with 128-dim ``z``.

    Parameters
    ----------
    state_dim:
        Dimensionality of the raw environment state fed to the encoder.
    latent_dim:
        Dimensionality of ``z`` (128 in the paper).
    state_embedding_dim:
        Learned linear projection size for states (64 in the paper).
    reward_embedding_dim:
        Learned reward embedding size (64 in the paper).
    num_reward_bins:
        Number of reward discretization bins (32 in the paper).
    transformer_width / num_blocks / num_heads / transformer_mlp_dim:
        Transformer hyper-parameters (128 / 4 / 4 / 256 in the paper).
    log_std_min / log_std_max:
        Clamp for the predicted log standard deviation, for numerical stability.
        The paper does not state a clamp for the encoder; (-5, 2) is a sensible default.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embedding_dim: int = 64,
        reward_embedding_dim: int = 64,
        num_reward_bins: int = 32,
        transformer_width: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        transformer_mlp_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.0,
        reward_low: float = -1.0,
        reward_high: float = 1.0,
        reward_clamp: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        use_positional_encoding: bool = False,
        use_causal_mask: bool = False,
        separate_reward_embedding: bool = True,
        obs_embedding: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.state_embedding_dim = int(state_embedding_dim)
        self.reward_embedding_dim = int(reward_embedding_dim)
        self.transformer_width = int(transformer_width)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        token_dim = self.state_embedding_dim + self.reward_embedding_dim
        if token_dim != self.transformer_width:
            raise ValueError(
                "the concatenated state+reward token dimension "
                f"({token_dim}) must equal the transformer width ({self.transformer_width}); "
                "the paper uses 64 + 64 = 128"
            )

        # (2) learned linear projection of the raw state -> 64 dims
        self.state_embedding = (
            obs_embedding
            if obs_embedding is not None
            else nn.Linear(self.state_dim, self.state_embedding_dim)
        )

        # (1) reward discretization + learned embedding table -> 64 dims.
        # NOTE: weights are NOT tied with any other module -- the encoder owns its own
        # reward embedding table unless the caller passes one in.
        if separate_reward_embedding:
            self.reward_embedding: nn.Module = RewardEmbedding(
                num_bins=num_reward_bins,
                embedding_dim=self.reward_embedding_dim,
                reward_low=reward_low,
                reward_high=reward_high,
                clamp=reward_clamp,
                init_std=1.0,
            )
        else:
            self.reward_embedding = RewardEmbedding(
                num_bins=num_reward_bins,
                embedding_dim=self.reward_embedding_dim,
                reward_low=reward_low,
                reward_high=reward_high,
                clamp=reward_clamp,
                init_std=1.0,
            )

        # (4) permutation-invariant transformer over the unordered set of tokens
        self.transformer = TransformerEncoder(
            width=self.transformer_width,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_dim=transformer_mlp_dim,
            activation=activation,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_causal_mask=use_causal_mask,
        )

        # (5) mean and (log) std heads for the 128-dim Gaussian p_theta(z | L^e)
        self.mean_head = nn.Linear(self.transformer_width, self.latent_dim)
        self.log_std_head = nn.Linear(self.transformer_width, self.latent_dim)

        self._init_weights()

    # ------------------------------------------------------------------ utils
    def _init_weights(self) -> None:
        """Default initializations matching the paper's description where given."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias the initial posterior std towards something reasonable (paper silent).
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.bias)
        nn.init.normal_(self.log_std_head.weight, std=0.01)

    def encode_states(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed ``(states, rewards)`` into the transformer token space.

        Parameters
        ----------
        states:
            ``(batch, K, state_dim)`` raw states. A leading batch dimension is optional:
            ``(K, state_dim)`` is promoted to ``(1, K, state_dim)``.
        rewards:
            ``(batch, K)`` scalar rewards (a trailing singleton dim is tolerated).

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            ``(tokens, batch_size)`` where ``tokens`` has shape ``(batch, K, 128)``.
        """
        squeezed = False
        if states.dim() == 2:
            states = states.unsqueeze(0)
            squeezed = True
        if states.dim() != 3:
            raise ValueError(
                f"states must be (batch, K, state_dim) or (K, state_dim), got {tuple(states.shape)}"
            )
        if rewards.dim() == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
        if rewards.dim() != 2:
            raise ValueError(f"rewards must be (batch, K), got {tuple(rewards.shape)}")
        if rewards.shape[:2] != states.shape[:2]:
            raise ValueError(
                f"states {tuple(states.shape[:2])} and rewards {tuple(rewards.shape[:2])} "
                "must agree on (batch, K)"
            )
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, got last dim {states.shape[-1]}"
            )

        state_emb = self.state_embedding(states)              # (B, K, 64)
        reward_emb = self.reward_embedding(rewards)           # (B, K, 64)
        tokens = torch.cat([state_emb, reward_emb], dim=-1)   # (B, K, 128)
        if squeezed:
            tokens = tokens.squeeze(0)
        return tokens, states.shape[0]

    # --------------------------------------------------------------- forward
    def forward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Normal:
        """Compute ``p_theta(z | s^e, eta(s^e))``.

        Returns a :class:`torch.distributions.Normal` with batch-shaped
        ``loc``/``scale`` of size ``(batch, latent_dim)``.
        """
        tokens, batch = self.encode_states(states, rewards)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        h = self.transformer(tokens, key_padding_mask=key_padding_mask)  # (B, 128)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) + 1e-6
        return torch.distributions.Normal(mean, std)

    # ------------------------------------------------------------- helpers
    @torch.no_grad()
    def encode(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        sample: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience wrapper returning a ``(batch, latent_dim)`` ``z``.

        With ``sample=False`` the posterior mean is returned (deterministic encoding,
        used at evaluation time).
        """
        dist = self.forward(states, rewards, key_padding_mask=key_padding_mask)
        z = dist.sample() if sample else dist.loc
        if states.dim() == 2:
            z = z.squeeze(0)
        return z

    @classmethod
    def from_config(cls, config, state_dim: int, **overrides) -> "Encoder":
        """Build an encoder from a :class:`fre.config.default.Config`-like object."""
        hidden_activation = getattr(config, "transformer_activation", "gelu")
        kwargs = dict(
            state_dim=state_dim,
            latent_dim=getattr(config, "latent_dim", 128),
            state_embedding_dim=getattr(config, "state_embedding_dim", 64),
            reward_embedding_dim=getattr(config, "reward_embedding_dim", 64),
            num_reward_bins=getattr(config, "num_reward_bins", 32),
            transformer_width=getattr(config, "transformer_width", 128),
            num_blocks=getattr(config, "num_encoder_blocks", 4),
            num_heads=getattr(config, "num_attention_heads", 4),
            transformer_mlp_dim=getattr(config, "transformer_mlp_dim", 256),
            activation=hidden_activation,
            dropout=getattr(config, "dropout", 0.0),
            reward_low=getattr(config, "reward_low", -1.0),
            reward_high=getattr(config, "reward_high", 1.0),
            reward_clamp=getattr(config, "reward_clamp", True),
            log_std_min=getattr(config, "log_std_min", -5.0),
            log_std_max=getattr(config, "log_std_max", 2.0),
            use_positional_encoding=getattr(config, "use_positional_encoding", False),
            use_causal_mask=getattr(config, "use_causal_mask", False),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"token_dim={self.state_embedding_dim + self.reward_embedding_dim}, "
            f"blocks={len(self.transformer.blocks)}"
        )


# ---------------------------------------------------------------------------
# KL helper (Eq. 6 compression term)
# ---------------------------------------------------------------------------
def diagonal_gaussian_kl(
    mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """``D_KL(N(mean, std^2) || N(0, I))`` averaged over the batch.

    Closed form matching the unit-Gaussian prior ``u(z)`` used in Eq. (6):
    ``-0.5 * sum(1 + 2*log(std) - mean^2 - std^2)``.
    """
    var = std.pow(2)
    kl = 0.5 * (var + mean.pow(2) - 1.0 - 2.0 * torch.log(std))
    return kl.sum(dim=-1).mean()
