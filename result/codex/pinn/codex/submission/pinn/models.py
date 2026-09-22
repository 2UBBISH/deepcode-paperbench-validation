"""The fully connected tanh network used for ``u(x; w)`` (Section 2.2)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP with ``len(hidden_widths)`` hidden layers and tanh activations.

    Parameters
    ----------
    in_dim:
        dimension of the input (2 for all problems: ``(x, t)``).
    hidden_widths:
        width of every hidden layer; the paper uses three hidden layers with
        width in ``{50, 100, 200, 400}``.
    out_dim:
        dimension of the output (1 for all problems).
    init:
        initialization of the weights.  The paper uses the Xavier normal
        initialization (Glorot & Bengio, 2010) with zero biases.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_widths: Sequence[int] = (200, 200, 200),
        out_dim: int = 1,
        init: str = "xavier_normal",
    ):
        super().__init__()
        widths = [in_dim] + list(hidden_widths) + [out_dim]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()
        self.init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if self.init == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif self.init == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            else:
                raise ValueError(f"unknown init {self.init!r}")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    @property
    def width(self) -> int:
        return self.layers[0].out_features

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    width: int = 200,
    n_layers: int = 3,
    in_dim: int = 2,
    out_dim: int = 1,
    init: str = "xavier_normal",
    seed: int | None = None,
) -> MLP:
    """Construct an ``n_layers``-deep MLP with a single hidden width."""
    if seed is not None:
        torch.manual_seed(int(seed))
    model = MLP(in_dim, tuple([width] * n_layers), out_dim, init=init)
    return model
