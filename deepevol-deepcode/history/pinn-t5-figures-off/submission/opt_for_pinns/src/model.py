"""MLP model for PINNs.

Architecture (Section 2.2 / Appendix A of the paper):
    - Input: (x, t) -> 2 dims
    - 3 hidden layers with tanh activation
    - Output: scalar u(x, t)
    - Widths considered: {50, 100, 200, 400}
    - Initialization: Xavier normal for weights, zero biases (Glorot & Bengio 2010)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected MLP with tanh activations.

    Parameters
    ----------
    in_dim : int
        Input dimension (2 for (x, t)).
    out_dim : int
        Output dimension (1 for scalar u).
    width : int
        Number of units per hidden layer.
    depth : int
        Number of hidden layers (default 3 per the paper).
    activation : str
        Activation function name; only "tanh" is used in the paper.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        width: int = 100,
        depth: int = 3,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.activation_name = activation

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sin":
            self.activation = torch.sin
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        dims: List[int] = [in_dim] + [width] * depth + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier normal weights, zero biases (Glorot & Bengio 2010)."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x, t : torch.Tensor
            Spatial / temporal coordinates, each of shape (N, 1) or (N,).

        Returns
        -------
        torch.Tensor
            u(x, t) of shape (N, 1).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = torch.cat([x, t], dim=-1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    width: int = 100,
    depth: int = 3,
    in_dim: int = 2,
    out_dim: int = 1,
    activation: str = "tanh",
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> MLP:
    """Convenience factory for building an MLP with optional seeding."""
    if seed is not None:
        torch.manual_seed(seed)
    model = MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        width=width,
        depth=depth,
        activation=activation,
    )
    if device is not None:
        model = model.to(device)
    return model


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """Return a flat vector of all parameters (concatenated)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_parameters(model: nn.Module, flat: torch.Tensor) -> None:
    """Set model parameters from a flat vector (in-place)."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n
