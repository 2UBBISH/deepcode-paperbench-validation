"""Entropy regularization for SAPG followers (Section 4.5).

The paper adds an entropy bonus to each *follower* policy's objective to
encourage exploration and diversity across the M policies:

    L(pi_i) = L_on(pi_i) + sigma * (i - 1) * H(pi_i(a | s))          (Sec. 4.5)

where

    * ``i`` is the 1-based policy index (``i = 1`` is the leader),
    * ``H(pi_i(a | s))`` is the differential entropy of the Gaussian policy,
    * ``sigma`` is a scalar entropy coefficient.

Key design points from the paper / reproduction plan:

    * The **leader** (``i = 1``) receives **no** entropy bonus, because the
      factor ``(i - 1)`` is zero.  This is intentional: the leader is the
      "greedy" aggregator policy and should not be pushed toward randomness.
    * Followers receive progressively larger entropy bonuses as their index
      grows (``i - 1``), which spreads the M policies apart in behaviour
      space and makes the off-policy aggregation meaningful.
    * ``sigma`` is swept over ``{0, 0.003, 0.005}`` in the paper's ablations
      (Figure 6).  ``sigma = 0`` is best for most tasks, while ``sigma = 0.005``
      is best for the Reorientation task.
    * When entropy exploration is enabled, **each block has its own learnable
      sigma vector** (i.e. a per-policy, per-action-dimension coefficient).
      This module supports both a fixed scalar ``sigma`` and a learnable
      per-policy parameter vector.

This module is deliberately self-contained: it only depends on ``torch`` and
exposes a small, well-tested API that ``algorithm.py`` and ``ppo.py`` consume.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = [
    "EntropyCoef",
    "FollowerEntropyRegularizer",
    "entropy_bonus",
    "follower_entropy_bonus",
    "make_entropy_coef",
]


# ---------------------------------------------------------------------------
# Entropy coefficient containers
# ---------------------------------------------------------------------------
class EntropyCoef(nn.Module):
    """Learnable per-policy entropy coefficient vector.

    The paper states that when entropy exploration is used, *each block has its
    own learnable sigma vector*.  This module stores one ``sigma`` vector per
    policy (shape ``(num_policies, act_dim)``) and exposes a ``forward`` that
    returns the coefficient for a given policy index.

    The leader's coefficient (index 0) is kept at zero and is **not** a
    parameter, so it never receives gradients and never contributes to the
    loss.  Followers' coefficients are initialized to ``init_sigma`` and are
    optimized jointly with the rest of the network.

    Args:
        num_policies: Total number of policies ``M`` (leader + followers).
        act_dim: Action dimensionality (one coefficient per action dim).
        init_sigma: Initial value for follower coefficients.
        learnable: If ``False``, coefficients are frozen buffers (fixed sigma).
        min_sigma: Lower clamp applied to the coefficient (keeps it >= 0).
    """

    def __init__(
        self,
        num_policies: int,
        act_dim: int,
        init_sigma: float = 0.0,
        learnable: bool = True,
        min_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        if num_policies < 1:
            raise ValueError(f"num_policies must be >= 1, got {num_policies}")
        if act_dim < 1:
            raise ValueError(f"act_dim must be >= 1, got {act_dim}")

        self.num_policies = int(num_policies)
        self.act_dim = int(act_dim)
        self.min_sigma = float(min_sigma)

        # Followers only: indices 1..M-1.  The leader (index 0) has no coef.
        num_followers = max(self.num_policies - 1, 0)
        init = torch.full((num_followers, self.act_dim), float(init_sigma))
        if learnable and num_followers > 0:
            self.sigma = nn.Parameter(init)
        else:
            self.register_buffer("sigma", init)

    def forward(self, policy_index: int) -> torch.Tensor:
        """Return the entropy coefficient vector for ``policy_index``.

        Args:
            policy_index: 0-based policy index.  ``0`` is the leader.

        Returns:
            Tensor of shape ``(act_dim,)``.  For the leader this is a zero
            vector (no entropy bonus).
        """
        if policy_index <= 0:
            # Leader: no entropy bonus.  Return a zero vector on the right
            # device/dtype without creating a graph node.
            return torch.zeros(self.act_dim, device=self._device, dtype=self._dtype)
        idx = min(policy_index - 1, self.sigma.shape[0] - 1)
        coef = self.sigma[idx]
        if self.min_sigma > 0.0:
            coef = coef.clamp_min(self.min_sigma)
        return coef

    @property
    def _device(self) -> torch.device:
        return self.sigma.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.sigma.dtype

    def parameters_for_optimizer(self):
        """Yield only the learnable follower coefficients (if any)."""
        if isinstance(self.sigma, nn.Parameter):
            yield self.sigma


# ---------------------------------------------------------------------------
# Functional entropy bonuses
# ---------------------------------------------------------------------------
def entropy_bonus(
    dist_entropy: torch.Tensor,
    policy_index: int,
    sigma: Union[float, torch.Tensor] = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the follower entropy term ``sigma * (i - 1) * H(pi)``.

    This is the low-level functional form used by :class:`FollowerEntropyRegularizer`.
    It returns a **loss to be maximized** (i.e. the caller subtracts it from the
    total loss, or adds its negative).

    Args:
        dist_entropy: Per-sample entropy, shape ``(B,)`` or ``(B, act_dim)``.
        policy_index: 0-based policy index (``0`` = leader).
        sigma: Scalar or per-action-dim tensor entropy coefficient.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        Scalar tensor (or per-sample tensor if ``reduction == "none"``).
    """
    if policy_index <= 0:
        # Leader has no entropy bonus.
        if reduction == "none":
            return torch.zeros_like(dist_entropy)
        return dist_entropy.new_zeros(())

    scale = float(policy_index)  # (i - 1) with 1-based i == policy_index (0-based)

    if isinstance(sigma, torch.Tensor):
        # Per-action-dim coefficient: broadcast over the last dim if needed.
        if sigma.dim() == 1 and dist_entropy.dim() == 2 and sigma.shape[0] == dist_entropy.shape[-1]:
            weighted = dist_entropy * sigma.unsqueeze(0)
            entropy_term = weighted.sum(dim=-1)
        else:
            entropy_term = dist_entropy * sigma
    else:
        entropy_term = dist_entropy * float(sigma)

    entropy_term = entropy_term * scale

    if reduction == "mean":
        return entropy_term.mean()
    if reduction == "sum":
        return entropy_term.sum()
    if reduction == "none":
        return entropy_term
    raise ValueError(f"Unknown reduction: {reduction!r}")


def follower_entropy_bonus(
    dist_entropy: torch.Tensor,
    policy_index: int,
    sigma: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """Convenience wrapper returning the mean follower entropy bonus."""
    return entropy_bonus(dist_entropy, policy_index, sigma, reduction="mean")


# ---------------------------------------------------------------------------
# High-level regularizer
# ---------------------------------------------------------------------------
class FollowerEntropyRegularizer(nn.Module):
    """Manages entropy coefficients and computes per-policy entropy bonuses.

    This class bundles the entropy-coefficient bookkeeping (fixed scalar vs.
    learnable per-policy vector) with the actual bonus computation, so that
    ``algorithm.py`` / ``ppo.py`` only need to call :meth:`bonus`.

    Args:
        num_policies: Number of policies ``M``.
        act_dim: Action dimensionality.
        sigma: Entropy coefficient.  Either a scalar (shared by all followers)
            or a sequence of length ``M`` giving a per-policy scalar.  The
            leader's entry is ignored.
        learnable: If ``True`` and ``sigma`` is a scalar, a learnable
            per-policy, per-action-dim coefficient vector is created (paper's
            "each block has its own learnable sigma vector").
        min_sigma: Lower clamp for learnable coefficients.
    """

    def __init__(
        self,
        num_policies: int,
        act_dim: int,
        sigma: Union[float, Sequence[float]] = 0.0,
        learnable: bool = False,
        min_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_policies = int(num_policies)
        self.act_dim = int(act_dim)

        self._per_policy_scalar: Optional[torch.Tensor] = None
        self.coef: Optional[EntropyCoef] = None

        if isinstance(sigma, (list, tuple)):
            if len(sigma) != self.num_policies:
                raise ValueError(
                    f"sigma sequence length {len(sigma)} != num_policies {self.num_policies}"
                )
            self._per_policy_scalar = torch.tensor([float(s) for s in sigma])
        else:
            scalar = float(sigma)
            if learnable and scalar > 0.0:
                self.coef = EntropyCoef(
                    num_policies=self.num_policies,
                    act_dim=self.act_dim,
                    init_sigma=scalar,
                    learnable=True,
                    min_sigma=min_sigma,
                )
            else:
                self._per_policy_scalar = torch.full((self.num_policies,), scalar)

    # -- coefficient access -------------------------------------------------
    def coefficient(self, policy_index: int) -> Union[float, torch.Tensor]:
        """Return the entropy coefficient for ``policy_index`` (0-based)."""
        if policy_index <= 0:
            return 0.0
        if self.coef is not None:
            return self.coef(policy_index)
        assert self._per_policy_scalar is not None
        idx = min(policy_index, self._per_policy_scalar.shape[0] - 1)
        return float(self._per_policy_scalar[idx].item())

    # -- bonus computation --------------------------------------------------
    def bonus(
        self,
        dist_entropy: torch.Tensor,
        policy_index: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the follower entropy bonus for ``policy_index``."""
        return entropy_bonus(
            dist_entropy,
            policy_index,
            sigma=self.coefficient(policy_index),
            reduction=reduction,
        )

    def parameters_for_optimizer(self):
        """Yield learnable entropy parameters (empty if fixed sigma)."""
        if self.coef is not None:
            yield from self.coef.parameters_for_optimizer()

    def extra_repr(self) -> str:
        mode = "learnable" if self.coef is not None else "fixed"
        return f"num_policies={self.num_policies}, act_dim={self.act_dim}, mode={mode}"


def make_entropy_coef(
    num_policies: int,
    act_dim: int,
    sigma: Union[float, Sequence[float]] = 0.0,
    learnable: bool = False,
    min_sigma: float = 0.0,
) -> FollowerEntropyRegularizer:
    """Factory mirroring the paper's entropy-coefficient configuration."""
    return FollowerEntropyRegularizer(
        num_policies=num_policies,
        act_dim=act_dim,
        sigma=sigma,
        learnable=learnable,
        min_sigma=min_sigma,
    )
