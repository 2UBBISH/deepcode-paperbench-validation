"""Shared conditioned networks for SAPG.

This module implements the shared actor network ``B_theta`` and critic network
``C_psi`` described in the SAPG paper (Section 4.4 / Addendum).

Key idea
--------
Instead of maintaining a *separate* network per follower/leader (which would be
prohibitively expensive when scaling to tens of thousands of environments), SAPG
uses a single shared network that is *conditioned* on a per-worker parameter
vector ``phi_j``.  Each follower ``j`` therefore has its own effective policy
``pi_j(a | s) = B_theta(s, phi_j)`` while all workers share the bulk of the
computation.

The conditioning is implemented as a learned embedding ``phi_j`` that is
concatenated to the observation (and, for recurrent policies, to the LSTM input).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _make_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "elu",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Construct a plain MLP with the requested activation."""
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation.lower()]
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        out_act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[output_activation.lower()]
        layers.append(out_act())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Shared conditioned actor / critic
# ---------------------------------------------------------------------------
class SharedConditionedNetwork(nn.Module):
    """A shared network conditioned on a per-worker embedding ``phi_j``.

    The network is shared across all workers; the only worker-specific
    parameters are the rows of the ``phi`` embedding table.  This is the
    "shared B_theta / C_psi conditioned on phi_j" design from the paper.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the raw observation.
    phi_dim:
        Dimensionality of the per-worker conditioning vector ``phi_j``.
    hidden_dims:
        Hidden layer sizes of the trunk MLP.
    out_dim:
        Output dimensionality (action mean dim for actor, 1 for critic).
    num_workers:
        Number of workers (followers + optional leader) that get their own
        ``phi_j`` embedding row.
    activation:
        Activation function used throughout the trunk.
    """

    def __init__(
        self,
        obs_dim: int,
        phi_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        num_workers: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.phi_dim = phi_dim
        self.out_dim = out_dim
        self.num_workers = num_workers

        # Per-worker conditioning embeddings phi_j.  Initialised small so that
        # all workers start from (approximately) the same policy.
        self.phi = nn.Parameter(torch.zeros(num_workers, phi_dim))
        nn.init.normal_(self.phi, mean=0.0, std=0.01)

        self.trunk = _make_mlp(
            in_dim=obs_dim + phi_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor, worker_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        obs:
            Tensor of shape ``(..., obs_dim)``.
        worker_ids:
            Long tensor of shape ``(...)`` giving the worker index for each
            observation.  Broadcast against ``obs``'s leading dims.
        """
        phi = self.phi[worker_ids]  # (..., phi_dim)
        x = torch.cat([obs, phi], dim=-1)
        return self.trunk(x)


# ---------------------------------------------------------------------------
# Recurrent (LSTM) conditioned network -- used for AllegroKuka
# ---------------------------------------------------------------------------
class SharedConditionedLSTMNetwork(nn.Module):
    """Shared LSTM network conditioned on ``phi_j`` (AllegroKuka mean net).

    The observation is first passed through an MLP (768 -> 512 -> 256) with ELU
    activations, then fed into a single-layer LSTM with 768 hidden units, as
    specified in the paper.
    """

    def __init__(
        self,
        obs_dim: int,
        phi_dim: int,
        num_workers: int,
        pre_hidden_dims: Sequence[int] = (512, 256),
        lstm_hidden: int = 768,
        out_dim: int = 23,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.phi_dim = phi_dim
        self.lstm_hidden = lstm_hidden
        self.out_dim = out_dim

        self.phi = nn.Parameter(torch.zeros(num_workers, phi_dim))
        nn.init.normal_(self.phi, mean=0.0, std=0.01)

        self.pre = _make_mlp(
            in_dim=obs_dim + phi_dim,
            hidden_dims=pre_hidden_dims,
            out_dim=lstm_hidden,
            activation=activation,
        )
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=1)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(
        self,
        obs: torch.Tensor,
        worker_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over a sequence.

        Parameters
        ----------
        obs:
            Tensor of shape ``(T, B, obs_dim)`` (sequence-first, as required by
            ``nn.LSTM``).
        worker_ids:
            Long tensor of shape ``(B,)`` giving the worker index per batch
            element.
        hidden:
            Optional ``(h_0, c_0)`` tuple of shape ``(1, B, lstm_hidden)``.

        Returns
        -------
        out:
            Tensor of shape ``(T, B, out_dim)``.
        hidden:
            Updated ``(h_T, c_T)``.
        """
        T, B, _ = obs.shape
        phi = self.phi[worker_ids].unsqueeze(0).expand(T, B, self.phi_dim)
        x = torch.cat([obs, phi], dim=-1)
        x = self.pre(x)
        out, hidden = self.lstm(x, hidden)
        out = self.head(out)
        return out, hidden


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_actor(
    obs_dim: int,
    action_dim: int,
    num_workers: int,
    hidden_dims: Sequence[int],
    phi_dim: int = 8,
    recurrent: bool = False,
    lstm_hidden: int = 768,
    activation: str = "elu",
) -> nn.Module:
    """Build the shared actor network ``B_theta`` for a given task."""
    if recurrent:
        return SharedConditionedLSTMNetwork(
            obs_dim=obs_dim,
            phi_dim=phi_dim,
            num_workers=num_workers,
            pre_hidden_dims=hidden_dims,
            lstm_hidden=lstm_hidden,
            out_dim=action_dim,
            activation=activation,
        )
    return SharedConditionedNetwork(
        obs_dim=obs_dim,
        phi_dim=phi_dim,
        hidden_dims=hidden_dims,
        out_dim=action_dim,
        num_workers=num_workers,
        activation=activation,
    )


def build_critic(
    obs_dim: int,
    num_workers: int,
    hidden_dims: Sequence[int],
    phi_dim: int = 8,
    recurrent: bool = False,
    lstm_hidden: int = 768,
    activation: str = "elu",
) -> nn.Module:
    """Build the shared critic network ``C_psi`` for a given task."""
    if recurrent:
        return SharedConditionedLSTMNetwork(
            obs_dim=obs_dim,
            phi_dim=phi_dim,
            num_workers=num_workers,
            pre_hidden_dims=hidden_dims,
            lstm_hidden=lstm_hidden,
            out_dim=1,
            activation=activation,
        )
    return SharedConditionedNetwork(
        obs_dim=obs_dim,
        phi_dim=phi_dim,
        hidden_dims=hidden_dims,
        out_dim=1,
        num_workers=num_workers,
        activation=activation,
    )
