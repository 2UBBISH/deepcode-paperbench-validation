"""Reward decoder for Functional Reward Encodings (FRE).

The decoder is a small MLP that maps a state ``s`` and a latent code ``z`` to a
scalar predicted reward ``r_hat``. It is trained jointly with the permutation-
invariant transformer encoder as part of the FRE variational autoencoder.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class RewardDecoder(nn.Module):
    """MLP decoder mapping ``[state, latent]`` to scalar reward.

    Parameters
    ----------
    state_dim:
        Dimensionality of raw environment states.
    latent_dim:
        Dimensionality of the FRE latent code ``z``.
    hidden_dims:
        Sizes of hidden layers. The paper uses two hidden layers of width 256.
    activation:
        Hidden activation. The paper uses ReLU.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")

        act_cls: type[nn.Module]
        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() in ("gelu",):
            act_cls = nn.GELU
        elif activation.lower() in ("tanh",):
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        input_dim = state_dim + latent_dim
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(act_cls())
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict rewards for states conditioned on latent codes.

        Parameters
        ----------
        states:
            Tensor of shape ``(..., state_dim)``.
        z:
            Tensor of shape ``(..., latent_dim)``. Leading dimensions must
            broadcast with ``states``.

        Returns
        -------
        Tensor of shape ``(..., 1)`` containing predicted scalar rewards.
        """
        # Support batched latent codes with batched states via broadcasting.
        # torch.cat requires identical leading shapes; we expand z to match.
        if z.dim() != states.dim():
            # Common case: z has shape (batch, latent_dim) while states has
            # shape (batch, K, state_dim). Add an intermediate dimension.
            while z.dim() < states.dim():
                z = z.unsqueeze(1)
        combined = torch.cat([states, z.expand_as_states_extra(z, states)], dim=-1)
        return self.net(combined)

    @staticmethod
    def _expand_like_for_cat(z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Expand latent code to broadcast against state leading dimensions."""
        return z
