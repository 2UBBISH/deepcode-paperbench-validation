"""Reward decoder for the FRE variational autoencoder.

This module implements the conditional reward decoder
:math:`q_\\theta(\\eta(s) \\mid s, z)` described in the paper. Given a
state :math:`s` and a latent reward vector :math:`z`, the decoder predicts
the scalar reward :math:`\\eta(s)` using a small MLP:

* input: ``[state; z]``
* two hidden layers of size 256 with ReLU activations
* scalar output (squeezed to shape ``(...,)``)

The decoder is used both during FRE pretraining (reconstruction loss) and
for downstream reward prediction if needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["RewardDecoder"]


class RewardDecoder(nn.Module):
    """Feed-forward reward decoder :math:`q(\\eta(s) \\mid s, z)`.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state observations.
    z_dim : int
        Dimensionality of the latent reward vector.
    hidden_dim : int, default 256
        Width of each hidden layer.
    num_hidden : int, default 2
        Number of hidden layers (each followed by ReLU).
    activation : str, default "relu"
        Hidden activation. Only ``"relu"`` is used by default.
    """

    def __init__(
        self,
        state_dim: int,
        z_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_hidden < 1:
            raise ValueError("RewardDecoder requires at least one hidden layer.")

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU}.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = state_dim + z_dim
        for i in range(num_hidden):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict scalar rewards for state/latent pairs.

        Parameters
        ----------
        states : torch.Tensor
            Shape ``(..., state_dim)``.
        z : torch.Tensor
            Shape ``(..., z_dim)``. Must broadcast with ``states`` across
            leading dimensions.

        Returns
        -------
        torch.Tensor
            Scalar reward predictions of shape ``(...,)``.
        """
        inputs = torch.cat([states, z], dim=-1)
        out = self.net(inputs).squeeze(-1)
        return out

    def extra_repr(self) -> str:
        return (
            f"state_dim+{self.net[0].in_features - 0} -> "
            f"hidden={self.net[0].out_features} -> 1"
        )
