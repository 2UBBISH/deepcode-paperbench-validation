"""Permutation-invariant transformer encoder for FRE.

This module consumes per-context-token state embeddings and reward embeddings,
concatenates them, projects the concatenation down to ``d_model``, and runs a
standard multi-head self-attention transformer encoder. Positional encodings and
causal masking are intentionally omitted so the representation is equivariant to
permutations of the encoder context, and the final per-token features are averaged
into a single permutation-invariant vector.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["TransformerEncoder"]


class TransformerEncoder(nn.Module):
    """Transformer encoder over state/reward context tokens.

    Args:
        d_model: Model/hidden dimensionality.
        d_ff: Feedforward hidden dimensionality inside each transformer layer.
        n_heads: Number of self-attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout probability used in attention and feedforward sublayers.
        activation: Feedforward activation name ("relu" or "gelu").
        input_dim: Dimensionality of the concatenated input features. If ``None``,
            it defaults to ``2 * d_model``, which is the expected case when the
            state embedding and reward embedding both have dimension ``d_model``.
        layer_norm_eps: Epsilon for layer normalization.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        input_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_dim = input_dim if input_dim is not None else 2 * d_model

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.input_proj = nn.Linear(self.input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        # A final layer norm is intentionally omitted; averaging raw final-layer
        # features matches the paper's simple average-pooling recipe.
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=None,
        )

    def forward(
        self,
        state_embeddings: torch.Tensor,
        reward_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        """Encode a set of state/reward context tokens.

        Args:
            state_embeddings: Tensor of shape ``(batch, num_tokens, d_state)``.
            reward_embeddings: Tensor of shape ``(batch, num_tokens, d_reward)``.
                Its last dimension must combine with ``state_embeddings`` to have
                size ``input_dim``.
            mask: Optional boolean attention mask. ``True``/non-zero values mask a
                token. Defaults to no masking (full bidirectional attention).
            return_tokens: If ``True``, returns the final per-token transformer
                features of shape ``(batch, num_tokens, d_model)`` instead of the
                average-pooled representation.

        Returns:
            Tensor of shape ``(batch, d_model)``, or ``(batch, num_tokens, d_model)``
            when ``return_tokens=True``.
        """
        if state_embeddings.shape[:-1] != reward_embeddings.shape[:-1]:
            raise ValueError(
                "state_embeddings and reward_embeddings must share batch and "
                f"token shapes, got {state_embeddings.shape[:-1]} and "
                f"{reward_embeddings.shape[:-1]}"
            )

        x = torch.cat([state_embeddings, reward_embeddings], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x, mask=mask, is_causal=False)

        if return_tokens:
            return x
        # Permutation-invariant aggregation.
        return x.mean(dim=1)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, dropout={self.dropout}, "
            f"input_dim={self.input_dim}"
        )
