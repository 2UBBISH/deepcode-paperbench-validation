"""Gaussian policy for SAPG.

Implements the stochastic policy pi_phi(a | s) used by both followers and the
leader.  The mean is produced by a shared conditioned network ``B_theta``
(see :mod:`sapg.networks`) and the standard deviation is a *learnable vector
that is independent of the observation* (as specified in the paper for the
AllegroKuka task).  An entropy-exploration variant is also provided where each
environment block owns its own learnable sigma vector.

Key design points from the paper / addendum:
  * Actor network ``B_theta`` is shared across all workers and conditioned on a
    per-worker embedding ``phi_j``.
  * Gaussian sigma is a fixed learnable vector independent of the observation.
  * ELU activations are used throughout the mean network.
  * For AllegroKuka the mean net is an LSTM (1 layer, 768 hidden) preceded by
    an MLP 768x512x256; for ShadowHand an MLP 512x512x256x128; for AllegroHand
    an MLP 512x256x128.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import build_actor


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    """Gaussian policy with a shared conditioned mean network.

    Parameters
    ----------
    obs_dim:
        Dimension of the observation vector.
    action_dim:
        Dimension of the action vector.
    num_workers:
        Number of workers (followers + optional leader) sharing the network.
    hidden_dims:
        Hidden layer sizes for the mean network trunk.
    phi_dim:
        Dimension of the per-worker embedding ``phi_j``.
    recurrent:
        If True, use an LSTM mean network (AllegroKuka).
    lstm_hidden:
        Hidden size of the LSTM (only used when ``recurrent`` is True).
    activation:
        Activation function name (default ``"elu"``).
    per_worker_sigma:
        If True, each worker owns its own learnable sigma vector
        (entropy-exploration variant).  Otherwise a single shared sigma vector
        is used.
    init_log_std:
        Initial value of the (shared) log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_workers: int = 1,
        hidden_dims: Sequence[int] = (512, 256, 128),
        phi_dim: int = 8,
        recurrent: bool = False,
        lstm_hidden: int = 768,
        activation: str = "elu",
        per_worker_sigma: bool = False,
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_workers = num_workers
        self.recurrent = recurrent
        self.per_worker_sigma = per_worker_sigma

        # Shared conditioned mean network B_theta(obs, phi_j).
        self.mean_net = build_actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_workers=num_workers,
            hidden_dims=hidden_dims,
            phi_dim=phi_dim,
            recurrent=recurrent,
            lstm_hidden=lstm_hidden,
            activation=activation,
        )

        # Learnable sigma, independent of the observation.
        if per_worker_sigma:
            self.log_std = nn.Parameter(
                torch.full((num_workers, action_dim), init_log_std)
            )
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _log_std_for(self, worker_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the log-std tensor broadcastable to ``(..., action_dim)``."""
        if self.per_worker_sigma:
            if worker_ids is None:
                # Fall back to worker 0 when no ids are supplied.
                return self.log_std[0]
            return self.log_std[worker_ids]
        return self.log_std

    def _clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Compute the mean of the action distribution.

        Returns ``(mean, hidden)`` where ``hidden`` is ``None`` for MLP nets.
        """
        if self.recurrent:
            mean, hidden = self.mean_net(obs, worker_ids, hidden)
            return mean, hidden
        mean = self.mean_net(obs, worker_ids)
        return mean, None

    def distribution(
        self,
        obs: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Normal, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Build the ``Normal`` action distribution for the given observations."""
        mean, hidden = self.forward(obs, worker_ids, hidden)
        log_std = self._log_std_for(worker_ids)
        log_std = self._clamp_log_std(log_std)
        std = torch.exp(log_std)
        # Broadcast std to the shape of mean.
        std = std.expand_as(mean)
        return Normal(mean, std), hidden

    # ------------------------------------------------------------------
    # action sampling / evaluation
    # ------------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        """Sample an action.

        Returns ``(action, log_prob, hidden)``.
        """
        dist, hidden = self.distribution(obs, worker_ids, hidden)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Evaluate ``log pi(a|s)`` and the entropy for given actions.

        Returns ``(log_prob, entropy, hidden)``.
        """
        dist, hidden = self.distribution(obs, worker_ids, hidden)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, hidden

    def get_actions_log_prob(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Convenience wrapper returning only the summed log probability."""
        log_prob, _, _ = self.evaluate_actions(obs, actions, worker_ids, hidden)
        return log_prob


def build_policy(
    obs_dim: int,
    action_dim: int,
    num_workers: int = 1,
    hidden_dims: Sequence[int] = (512, 256, 128),
    phi_dim: int = 8,
    recurrent: bool = False,
    lstm_hidden: int = 768,
    activation: str = "elu",
    per_worker_sigma: bool = False,
) -> GaussianPolicy:
    """Factory mirroring the task-specific network configurations.

    Task presets (hidden dims / recurrent):
      * AllegroKuka: recurrent=True, lstm_hidden=768, pre-MLP (512, 256)
      * ShadowHand:  hidden_dims=(512, 512, 256, 128)
      * AllegroHand: hidden_dims=(512, 256, 128)
    """
    return GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_workers=num_workers,
        hidden_dims=hidden_dims,
        phi_dim=phi_dim,
        recurrent=recurrent,
        lstm_hidden=lstm_hidden,
        activation=activation,
        per_worker_sigma=per_worker_sigma,
    )
