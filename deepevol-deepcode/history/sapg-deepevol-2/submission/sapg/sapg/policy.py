"""Gaussian policy for SAPG.

The policy wraps a shared actor network ``B_theta`` (see :mod:`sapg.networks`)
that outputs the mean of a diagonal Gaussian.  The standard deviation is a
*learnable, input-independent* vector (as described in the SAPG paper), unless
the per-block entropy-exploration variant is enabled, in which case each block
of environments owns its own learnable sigma vector.

Key features
------------
* Diagonal Gaussian sampling / log-prob / entropy.
* Fixed learnable ``sigma`` (input independent) by default.
* Optional per-block (per-worker) learnable ``sigma`` for entropy-based
  exploration (``use_entropy_exploration``).
* Action-bound handling: actions are squashed through ``tanh`` when
  ``use_tanh`` is enabled, with the corresponding change-of-variables applied
  to the log-probability.  A ``bounds_loss_coef`` (default ``1e-4``) penalises
  actions that violate the environment bounds.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ActorNetwork, build_actor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _cfg_getter(cfg):
    """Return a ``get(key, default)`` callable for dicts or attribute objects."""
    if cfg is None:
        return lambda key, default=None: default
    if isinstance(cfg, dict):
        return lambda key, default=None: cfg.get(key, default)

    def _get(key, default=None):
        return getattr(cfg, key, default)

    return _get


# ---------------------------------------------------------------------------
# Gaussian policy
# ---------------------------------------------------------------------------
class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy conditioned on per-worker embeddings.

    Parameters
    ----------
    actor : ActorNetwork
        Shared actor network producing the Gaussian mean.
    action_dim : int
        Dimensionality of the action space.
    num_workers : int
        Number of workers (followers + leader).  Used when
        ``use_entropy_exploration`` is enabled so that each worker owns its own
        learnable sigma vector.
    init_log_std : float
        Initial value of the (log) standard deviation.
    use_entropy_exploration : bool
        If ``True`` each worker has its own learnable sigma vector.
    use_tanh : bool
        If ``True`` squash the mean through ``tanh`` and apply the
        change-of-variables correction to the log-probability.
    bounds_loss_coef : float
        Coefficient for the action-bounds penalty.
    action_low, action_high : float or Tensor
        Action bounds used for the bounds penalty and tanh squashing.
    """

    def __init__(
        self,
        actor: ActorNetwork,
        action_dim: int,
        num_workers: int = 1,
        init_log_std: float = -1.0,
        use_entropy_exploration: bool = False,
        use_tanh: bool = False,
        bounds_loss_coef: float = 1e-4,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.action_dim = int(action_dim)
        self.num_workers = int(num_workers)
        self.use_entropy_exploration = bool(use_entropy_exploration)
        self.use_tanh = bool(use_tanh)
        self.bounds_loss_coef = float(bounds_loss_coef)

        if self.use_entropy_exploration:
            # One learnable sigma vector per worker/block.
            self.log_std = nn.Parameter(
                torch.full((self.num_workers, self.action_dim), float(init_log_std))
            )
        else:
            # Single shared learnable sigma vector (input independent).
            self.log_std = nn.Parameter(
                torch.full((self.action_dim,), float(init_log_std))
            )

        if isinstance(action_low, torch.Tensor):
            self.register_buffer("action_low", action_low.float())
        else:
            self.register_buffer(
                "action_low", torch.full((self.action_dim,), float(action_low))
            )
        if isinstance(action_high, torch.Tensor):
            self.register_buffer("action_high", action_high.float())
        else:
            self.register_buffer(
                "action_high", torch.full((self.action_dim,), float(action_high))
            )

    # ------------------------------------------------------------------
    # sigma helpers
    # ------------------------------------------------------------------
    def _get_log_std(self, worker_ids: Optional[torch.Tensor], batch_size: int):
        """Return the log-std broadcast to ``(batch_size, action_dim)``."""
        if self.use_entropy_exploration:
            if worker_ids is None:
                worker_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=self.log_std.device
                )
            log_std = self.log_std[worker_ids]  # (B, action_dim)
        else:
            log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)
        return log_std

    def _std(self, worker_ids: Optional[torch.Tensor], batch_size: int):
        log_std = self._get_log_std(worker_ids, batch_size)
        return torch.exp(log_std)

    # ------------------------------------------------------------------
    # Forward / sampling
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden_state=None,
        masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the Gaussian mean and std for ``obs``.

        Returns
        -------
        mean : Tensor ``(B, action_dim)``
        std : Tensor ``(B, action_dim)``
        new_hidden_state : optional recurrent hidden state
        """
        mean, new_hidden = self.actor(obs, worker_ids, hidden_state, masks)
        batch_size = mean.shape[0]
        std = self._std(worker_ids, batch_size)
        return mean, std, new_hidden

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map raw actions to the environment bounds via tanh."""
        mid = (self.action_high + self.action_low) / 2.0
        half = (self.action_high - self.action_low) / 2.0
        return mid + half * torch.tanh(raw_action)

    def _squash_log_prob_correction(self, raw_action: torch.Tensor) -> torch.Tensor:
        """log|det d(action)/d(raw_action)| for the tanh squashing."""
        # tanh derivative in log-space: log(1 - tanh(x)^2)
        return torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)

    def sample(
        self,
        obs: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden_state=None,
        masks=None,
        deterministic: bool = False,
    ):
        """Sample an action from the policy.

        Returns
        -------
        action : Tensor ``(B, action_dim)``
        log_prob : Tensor ``(B,)``
        mean : Tensor ``(B, action_dim)``
        new_hidden_state : optional recurrent hidden state
        """
        mean, std, new_hidden = self.forward(obs, worker_ids, hidden_state, masks)
        if deterministic:
            raw_action = mean
        else:
            raw_action = mean + std * torch.randn_like(std)

        log_prob = self._log_prob_from_raw(raw_action, mean, std)

        if self.use_tanh:
            action = self._squash(raw_action)
        else:
            action = raw_action

        return action, log_prob, mean, new_hidden

    def _log_prob_from_raw(
        self, raw_action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        var = std.pow(2)
        log_prob = -0.5 * (
            ((raw_action - mean).pow(2) / var)
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1)
        if self.use_tanh:
            log_prob = log_prob - self._squash_log_prob_correction(raw_action).sum(
                dim=-1
            )
        return log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden_state=None,
        masks=None,
    ):
        """Recompute log-prob, entropy and mean for stored ``actions``.

        When ``use_tanh`` is enabled the stored ``actions`` are assumed to be
        the *squashed* actions; the inverse tanh is applied to recover the raw
        actions before computing the Gaussian log-probability.

        Returns
        -------
        log_prob : Tensor ``(B,)``
        entropy : Tensor ``(B,)``
        mean : Tensor ``(B, action_dim)``
        new_hidden_state : optional recurrent hidden state
        """
        mean, std, new_hidden = self.forward(obs, worker_ids, hidden_state, masks)

        if self.use_tanh:
            mid = (self.action_high + self.action_low) / 2.0
            half = (self.action_high - self.action_low) / 2.0
            normalized = (actions - mid) / (half + 1e-8)
            normalized = torch.clamp(normalized, -1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = torch.atanh(normalized)
        else:
            raw_action = actions

        log_prob = self._log_prob_from_raw(raw_action, mean, std)
        entropy = self.entropy(std)
        return log_prob, entropy, mean, new_hidden

    # ------------------------------------------------------------------
    # Entropy / bounds
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(std: torch.Tensor) -> torch.Tensor:
        """Differential entropy of a diagonal Gaussian (summed over dims)."""
        return (0.5 * math.log(2.0 * math.pi * math.e) + torch.log(std)).sum(dim=-1)

    def bounds_loss(self, mean: torch.Tensor) -> torch.Tensor:
        """Penalty for actions outside the environment bounds.

        Mirrors the IsaacGymEnvs ``bounds_loss``: soft penalty on the amount by
        which the mean exceeds the bounds.
        """
        if self.bounds_loss_coef <= 0.0:
            return torch.zeros((), device=mean.device)
        soft_bound = 1.0
        mu_loss_high = torch.clamp_min(mean - soft_bound, 0.0).pow(2)
        mu_loss_low = torch.clamp_max(mean + soft_bound, 0.0).pow(2)
        return (mu_loss_high + mu_loss_low).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, device=None):
        return self.actor.init_hidden(batch_size, device)

    @property
    def is_recurrent(self) -> bool:
        return getattr(self.actor, "use_lstm", False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_policy(cfg, obs_dim: int, action_dim: int, num_workers: int = 1) -> GaussianPolicy:
    """Build a :class:`GaussianPolicy` from a config dict/object."""
    get = _cfg_getter(cfg)
    actor = build_actor(cfg, obs_dim, action_dim, num_workers)

    action_low = get("action_low", -1.0)
    action_high = get("action_high", 1.0)
    if isinstance(action_low, (list, tuple)):
        action_low = torch.tensor(action_low, dtype=torch.float32)
    if isinstance(action_high, (list, tuple)):
        action_high = torch.tensor(action_high, dtype=torch.float32)

    return GaussianPolicy(
        actor=actor,
        action_dim=action_dim,
        num_workers=num_workers,
        init_log_std=get("init_log_std", -1.0),
        use_entropy_exploration=get("use_entropy_exploration", False),
        use_tanh=get("use_tanh", False),
        bounds_loss_coef=get("bounds_loss_coef", 1e-4),
        action_low=action_low,
        action_high=action_high,
    )
