"""Multilayer perceptron used as the PINN ansatz u(x; w).

Specification (paper Section 2.2 + Appendix A):
  * fully-connected MLP  R^d -> R  with tanh activations and **three hidden layers**
  * widths in {50, 100, 200, 400}
  * Xavier-normal weight initialization (Glorot & Bengio, 2010) and all biases = 0
  * must be twice differentiable wrt the *inputs* (for residual operators D and B)
    and twice differentiable wrt the *parameters* (for Hessian-vector products)

The input dimension d is the spatial-time dimension, d = 2 for all three problems
(one spatial coordinate x, one time coordinate t; the wave equation is written in
first-order-in-time form with two output channels handled by the problem wrapper).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def xavier_normal_init_(module: nn.Linear, gain: float = 1.0) -> None:
    """Xavier normal initialization for weights, zero for biases."""
    nn.init.xavier_normal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """u(x; w) : R^in_dim -> R^out_dim, tanh, 3 hidden layers, Xavier-normal + zero bias.

    Parameters
    ----------
    in_dim : int
        spatial-time dimension d (2 for every PDE considered here).
    out_dim : int
        number of output channels (1 in the paper).
    width : int
        number of units in each hidden layer (50 / 100 / 200 / 400).
    depth : int
        number of hidden layers (3 in the paper).
    activation : str
        only "tanh" is used by the paper.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        width: int = 50,
        depth: int = 3,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.activation_name = activation

        if activation.lower() != "tanh":
            raise ValueError("The paper only uses tanh activations for the PINN MLP.")

        layers: List[nn.Module] = []
        prev = self.in_dim
        for _ in range(self.depth):
            layers.append(nn.Linear(prev, self.width))
            layers.append(nn.Tanh())
            prev = self.width
        layers.append(nn.Linear(prev, self.out_dim))

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    # ------------------------------------------------------------------ #
    def reset_parameters(self) -> None:
        """Xavier-normal weights, zero biases (paper Section 2.2)."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                xavier_normal_init_(module)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network. `x` has shape (..., in_dim)."""
        return self.net(x)

    # ------------------------------------------------------------------ #
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"width={self.width}, depth={self.depth}, act={self.activation_name}"
        )


def make_pinn(
    in_dim: int = 2,
    out_dim: int = 1,
    width: int = 50,
    depth: int = 3,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> MLP:
    """Convenience constructor with optional seeding."""
    if seed is not None:
        torch.manual_seed(seed)
    model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
    return model.to(device)
