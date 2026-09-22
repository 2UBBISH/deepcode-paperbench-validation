"""MLP model for PINNs.

Section 2.2 of "Challenges in Training PINNs: A Loss Landscape Perspective":
    - MLP with tanh activations
    - 3 hidden layers
    - width in {50, 100, 200, 400}
    - Xavier-normal weight initialization
    - zero biases
    - input = (x, t) coordinates; output = scalar u
"""

import math

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
        Number of hidden layers (default 3).
    activation : callable
        Activation function (default tanh).
    """

    def __init__(self, in_dim=2, out_dim=1, width=100, depth=3,
                 activation=torch.tanh):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.activation = activation

        layers = []
        prev = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, width))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier-normal weights, zero biases."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t=None):
        """Forward pass.

        Accepts either a single tensor of shape (N, 2) or two tensors x, t.
        """
        if t is not None:
            inp = torch.cat([x, t], dim=-1)
        else:
            inp = x
        h = inp
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        out = self.layers[-1](h)
        return out


def build_model(width=100, depth=3, in_dim=2, out_dim=1, seed=None,
                dtype=torch.float32, device="cpu"):
    """Convenience factory for an MLP."""
    if seed is not None:
        torch.manual_seed(seed)
    model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
    model = model.to(dtype=dtype, device=device)
    return model
