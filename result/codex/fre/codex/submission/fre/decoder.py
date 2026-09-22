"""Reward decoder q_theta(eta(s) | s, z) (Section 4.1).

The decoder independently predicts the reward at each *decoding* state given
the shared latent encoding ``z``.  Appendix A specifies a 3-hidden-layer MLP
with 512 units per layer ("Decoder Network Layers [512, 512, 512]").

The addendum notes explicitly that "there is no embedding step for the
observation state passed to the decoder -- the raw state and the z-vector are
concatenated directly".
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Utility builder for feed-forward MLPs used throughout the codebase."""
    act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class RewardDecoder(nn.Module):
    """Predicts ``eta(s)`` for a batch of decoding states, conditioned on ``z``."""

    def __init__(
        self,
        state_dim: int,
        z_dim: int = 128,
        hidden_dims: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.net = mlp(state_dim + z_dim, hidden_dims, 1, activation=activation)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Args:
            states: ``(B, ..., state_dim)`` decoding states.
            z: ``(B, z_dim)`` latent encodings.

        Returns:
            ``(B, ...)`` predicted rewards (a trailing singleton dim is
            squeezed away so a scalar reward per state is returned).
        """
        if z.dim() != 2:
            raise ValueError(f"z must be (B, z_dim), got {tuple(z.shape)}")
        lead = states.shape[:-1]
        z_exp = z.reshape(z.shape[0], *([1] * (len(lead) - 1)), z.shape[-1]).expand(*lead, -1)
        inp = torch.cat([states, z_exp], dim=-1)
        out = self.net(inp)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out
