"""Transformer score network of the Simformer (Fig. 2, Sec. 3.1-3.3).

The network consumes a sequence of tokens (one token per variable of the joint
``x_hat = (theta, x)``) and returns a scalar score per token, i.e. an estimate of
``s_phi(x_hat_t, t)_i ~ grad_{x_hat_i} log p_t(x_hat_t)``.

Implementation choices that follow Appendix A2.1:

* token dimension 50,
* diffusion time embedded with a 128-dimensional random Gaussian Fourier
  embedding; the (linearly projected) time embedding is added to the output of
  *each* feed-forward block (see addendum / paper Appendix),
* 6 layers, 4 heads, attention size 10, widening factor 3 (hidden dim 150),
* the interaction between variables is controlled with an attention mask
  ``M_E`` (Sec. 3.2), i.e. edges of the adjacency matrix denote which variables a
  token may attend to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .tokenizer import FourierFeatures, IdentifierEmbedding, Tokenizer, \
    TokenizerConfig


@dataclass
class TransformerConfig:
    n_variables: int
    d_model: int = 50
    n_heads: int = 4
    attention_size: int = 10      # per-head dimension of queries and keys
    n_layers: int = 6
    widening_factor: float = 3.0  # feed-forward hidden dim = widening * d_model
    time_embed_dim: int = 128
    dropout: float = 0.0
    layer_norm: bool = True


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention with an explicit (bool) attention mask."""

    def __init__(self, d_model: int, n_heads: int, attention_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.attention_size = attention_size
        inner = n_heads * attention_size
        self.q_proj = nn.Linear(d_model, inner)
        self.k_proj = nn.Linear(d_model, inner)
        self.v_proj = nn.Linear(d_model, inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.scale = attention_size ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        batch, n, _ = x.shape
        h, d = self.n_heads, self.attention_size
        q = self.q_proj(x).view(batch, n, h, d).transpose(1, 2)
        k = self.k_proj(x).view(batch, n, h, d).transpose(1, 2)
        v = self.v_proj(x).view(batch, n, h, d).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            logits = logits + _mask_to_bias(mask, logits)
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)                     # (B, H, n, d)
        out = out.transpose(1, 2).reshape(batch, n, h * d)
        out = self.out_proj(out)
        if return_attention:
            return out, attn
        return out


def _mask_to_bias(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Convert an attention mask (1 = allowed) into an additive attention bias."""
    mask = mask.to(logits.dtype)
    if mask.dim() == 2:            # (n, n)          -> (1, 1, n, n)
        mask = mask[None, None]
    elif mask.dim() == 3:          # (batch, n, n)   -> (batch, 1, n, n)
        mask = mask[:, None]
    if mask.shape[0] == 1 and logits.shape[0] > 1:
        mask = mask.expand(logits.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and logits.shape[1] > 1:
        mask = mask.expand(-1, logits.shape[1], -1, -1)
    return (1.0 - mask) * torch.finfo(logits.dtype).min


class TransformerLayer(nn.Module):
    """Pre-norm transformer block with the time-conditional feed-forward block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        d_model = config.d_model
        hidden = int(round(config.widening_factor * d_model))
        self.attention = MultiHeadSelfAttention(
            d_model, config.n_heads, config.attention_size)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, hidden)
        self.ff_out = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        # "a linear projection [of the time embedding] is added to the output of
        # each feed-forward block in the transformer"
        self.time_proj = nn.Linear(config.time_embed_dim, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        attn_out = self.attention(self.norm_attn(x), mask,
                                  return_attention=return_attention)
        if return_attention:
            attn_out, attn = attn_out
        x = x + self.dropout(attn_out)
        h = self.ff_out(self.act(self.ff_in(self.norm_ff(x))))
        h = h + self.time_proj(time_embedding)[:, None, :]
        x = x + self.dropout(h)
        if return_attention:
            return x, attn
        return x


class TransformerScoreNet(nn.Module):
    """Transformer that predicts a per-variable score of the joint distribution."""

    def __init__(self, config: TransformerConfig, tokenizer: Optional[Tokenizer] = None,
                 identifiers: Optional[IdentifierEmbedding] = None,
                 tokenizer_config: Optional[TokenizerConfig] = None):
        super().__init__()
        self.config = config
        if tokenizer is None:
            if tokenizer_config is None:
                tokenizer_config = TokenizerConfig(
                    n_variables=config.n_variables, d_model=config.d_model)
            tokenizer = Tokenizer(tokenizer_config)
        self.tokenizer = tokenizer
        self.identifiers = identifiers
        self.time_embedding = FourierFeatures(config.time_embed_dim, in_dim=1,
                                              scale=1.0, seed=1)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)
        self.score_head = nn.Linear(config.d_model, 1)

    # ---------------------------------------------------------------- forward
    def forward(self, values: torch.Tensor, condition_state: torch.Tensor,
                t: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                index: Optional[torch.Tensor] = None,
                variable_kind: Optional[torch.Tensor] = None,
                use_fourier: Optional[torch.Tensor] = None,
                metadata: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        """Return the score estimate with shape ``(batch, n_variables)``.

        ``values``          : ``(batch, n_variables)`` values ``x_hat_t`` (clean
                              for conditioned variables, noisy for latent ones).
        ``condition_state`` : ``(batch, n_variables)`` condition mask ``M_C``.
        ``t``               : ``(batch,)`` diffusion times in ``[t_min, 1]``.
        ``attention_mask``  : ``(batch, n, n)`` or ``(n, n)`` mask ``M_E`` where
                              ``mask[b, i, j] = 1`` means token ``i`` attends to
                              token ``j``.
        """
        if t.dim() == 0:
            t = t.expand(values.shape[0])
        time_embedding = self.time_embedding(t)
        id_embedding = None
        if self.identifiers is not None:
            if variable_kind is None:
                raise ValueError(
                    "`variable_kind` is required when using identifier "
                    "embeddings.")
            id_embedding = self.identifiers(variable_kind, index, use_fourier)
        h = self.tokenizer(values, condition_state, identifiers=id_embedding,
                           metadata=metadata)
        for layer in self.layers:
            h = layer(h, time_embedding, attention_mask)
        h = self.norm_out(h)
        if return_hidden:
            return h
        return self.score_head(h).squeeze(-1)
