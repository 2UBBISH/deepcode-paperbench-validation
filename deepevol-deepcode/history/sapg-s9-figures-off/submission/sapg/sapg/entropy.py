"""Entropy regularization for SAPG (Sec 4.5).

The paper adds an entropy bonus to *follower* policies to encourage exploration
diversity across the M blocks:

    L(pi_i) = L_on(pi_i) + sigma * (i - 1) * H(pi_i(a|s))          (followers)
    L(pi_1) = L_on(pi_1) + lambda * L_off(pi_1; X)                (leader, NO entropy)

where ``i`` is the 1-based policy index (leader = 1, so the leader's entropy
coefficient is exactly zero), ``H`` is the policy entropy, and ``sigma`` is a
scalar coefficient.

The paper sweeps ``sigma in {0, 0.003, 0.005}`` and reports the best value per
task (0.005 for Reorientation, 0 for the rest).  When entropy exploration is
enabled, *each env block has its own learnable sigma vector* -- i.e. the
coefficient is a per-block learnable parameter rather than a fixed scalar.

This module provides:

* :class:`EntropyScheduler` -- computes the effective entropy coefficient for a
  given policy index under a chosen schedule (fixed / linear / learnable).
* :class:`LearnableEntropyCoef` -- an ``nn.Module`` holding one learnable sigma
  per block (per policy), with a positivity transform (softplus) so the
  coefficient stays non-negative.
* :func:`follower_entropy_coef` -- the ``sigma * (i - 1)`` scaling rule.
* :func:`entropy_bonus` -- the (negative) entropy term to add to a loss.
* :func:`build_entropy_coefs` -- factory producing the per-policy coefficient
  tensor used by the training loop.

All coefficients are returned as *positive* values; the caller (``losses.py`` /
``algorithm.py``) subtracts ``coef * entropy`` from the loss because the
objectives in the paper are written as maximization problems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SAPGConfig


__all__ = [
    "EntropyScheduler",
    "LearnableEntropyCoef",
    "follower_entropy_coef",
    "entropy_bonus",
    "build_entropy_coefs",
    "EntropyCoefOutput",
]


# ---------------------------------------------------------------------------
# Core scaling rule
# ---------------------------------------------------------------------------
def follower_entropy_coef(
    policy_idx: int,
    sigma: float,
    leader_has_entropy: bool = False,
) -> float:
    """Return the entropy coefficient ``sigma * (i - 1)`` for a policy.

    Parameters
    ----------
    policy_idx:
        **0-based** policy index.  Policy 0 is the leader, policies 1..M-1 are
        followers.  The paper uses 1-based indexing where the leader is
        ``i = 1``; the conversion is ``i = policy_idx + 1``.
    sigma:
        Base entropy coefficient (one of ``{0, 0.003, 0.005}`` in the paper).
    leader_has_entropy:
        If ``False`` (paper default) the leader receives no entropy term, i.e.
        its coefficient is forced to 0 regardless of ``sigma``.

    Returns
    -------
    float
        The effective entropy coefficient for the policy.
    """
    if policy_idx < 0:
        raise ValueError(f"policy_idx must be non-negative, got {policy_idx}")
    if policy_idx == 0 and not leader_has_entropy:
        return 0.0
    # 1-based follower index i = policy_idx + 1  =>  sigma * (i - 1) = sigma * policy_idx
    return float(sigma) * float(policy_idx)


def entropy_bonus(entropy: torch.Tensor, coef: Union[float, torch.Tensor]) -> torch.Tensor:
    """Return the (negative) entropy term to add to a *minimization* loss.

    The paper's objective is a maximization problem, so the entropy bonus
    ``+ coef * H`` becomes ``- coef * H`` when the loss is minimized.  This
    helper returns ``-coef * H`` (a scalar tensor) so callers can simply add it
    to their loss.

    Parameters
    ----------
    entropy:
        Per-sample or already-reduced entropy tensor.
    coef:
        Scalar or broadcastable tensor coefficient (already includes the
        ``sigma * (i - 1)`` scaling).
    """
    if isinstance(coef, torch.Tensor):
        coef = coef.to(entropy.device, entropy.dtype)
    return -coef * entropy


# ---------------------------------------------------------------------------
# Learnable per-block entropy coefficient
# ---------------------------------------------------------------------------
class LearnableEntropyCoef(nn.Module):
    """Per-block learnable entropy coefficient (Sec 4.5).

    "When entropy exploration is used, each env block has its own learnable
    sigma vector."  We parameterize ``sigma_j = softplus(raw_j)`` so the
    coefficient is always non-negative, and initialize ``raw_j`` such that
    ``softplus(raw_j) == sigma_init``.

    The leader's coefficient is pinned to zero (the paper gives the leader no
    entropy term), which is enforced by :meth:`effective_coefs`.
    """

    def __init__(
        self,
        num_blocks: int,
        sigma_init: float = 0.0,
        leader_has_entropy: bool = False,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.leader_has_entropy = bool(leader_has_entropy)
        self.learnable = bool(learnable)

        # Inverse softplus so that softplus(raw) == sigma_init.
        raw_init = _inverse_softplus(max(float(sigma_init), 1e-8))
        raw = torch.full((self.num_blocks,), float(raw_init), dtype=torch.float32)
        if self.learnable:
            self.raw = nn.Parameter(raw)
        else:
            self.register_buffer("raw", raw)

    # -- helpers -----------------------------------------------------------
    def sigma(self) -> torch.Tensor:
        """Return the non-negative per-block base coefficients ``sigma_j``."""
        return F.softplus(self.raw)

    def effective_coefs(self) -> torch.Tensor:
        """Return ``sigma_j * (i - 1)`` per block (leader pinned to 0)."""
        sigma = self.sigma()
        idx = torch.arange(self.num_blocks, device=sigma.device, dtype=sigma.dtype)
        coefs = sigma * idx
        if not self.leader_has_entropy and self.num_blocks > 0:
            coefs = coefs.clone()
            coefs[0] = 0.0
        return coefs

    def coef_for(self, policy_idx: int) -> torch.Tensor:
        """Return the scalar effective coefficient for one policy."""
        return self.effective_coefs()[policy_idx]

    def forward(self, policy_idx: int) -> torch.Tensor:
        return self.coef_for(policy_idx)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"num_blocks={self.num_blocks}, learnable={self.learnable}, "
            f"leader_has_entropy={self.leader_has_entropy}"
        )


def _inverse_softplus(y: float) -> float:
    """Numerically stable inverse of ``softplus`` for ``y > 0``."""
    y = max(float(y), 1e-8)
    # softplus(x) = log(1 + exp(x))  =>  x = log(exp(y) - 1)
    return math.log(math.expm1(y))


# ---------------------------------------------------------------------------
# Scheduler / container
# ---------------------------------------------------------------------------
@dataclass
class EntropyCoefOutput:
    """Container describing the entropy coefficients for all policies.

    Attributes
    ----------
    coefs:
        Tensor of shape ``[num_blocks]`` with the effective coefficient for
        each policy (leader first).  Differentiable when the underlying
        coefficient is learnable.
    sigma:
        Tensor of shape ``[num_blocks]`` with the base ``sigma_j`` values.
    learnable:
        Whether the coefficients are being optimized.
    """

    coefs: torch.Tensor
    sigma: torch.Tensor
    learnable: bool = False


class EntropyScheduler:
    """Computes per-policy entropy coefficients for SAPG.

    Supports three modes:

    * ``"fixed"`` -- a single scalar ``sigma`` scaled by ``(i - 1)``.
    * ``"learnable"`` -- one learnable ``sigma_j`` per block (Sec 4.5).
    * ``"linear"`` -- ``sigma`` annealed linearly from ``sigma`` to
      ``sigma_final`` over ``total_iterations`` (not used in the paper's main
      results but useful for ablations).

    The leader always receives coefficient 0 unless ``leader_has_entropy`` is
    explicitly set.
    """

    def __init__(
        self,
        num_blocks: int,
        sigma: float = 0.0,
        mode: str = "fixed",
        leader_has_entropy: bool = False,
        sigma_final: Optional[float] = None,
        total_iterations: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.sigma = float(sigma)
        self.mode = str(mode).lower()
        self.leader_has_entropy = bool(leader_has_entropy)
        self.sigma_final = float(sigma_final) if sigma_final is not None else float(sigma)
        self.total_iterations = max(int(total_iterations), 1)
        self.device = device

        self._module: Optional[LearnableEntropyCoef] = None
        if self.mode == "learnable":
            self._module = LearnableEntropyCoef(
                num_blocks=self.num_blocks,
                sigma_init=self.sigma,
                leader_has_entropy=self.leader_has_entropy,
                learnable=True,
            )
            if device is not None:
                self._module = self._module.to(device)

    # -- public API --------------------------------------------------------
    @property
    def module(self) -> Optional[LearnableEntropyCoef]:
        """The learnable coefficient module (``None`` for non-learnable modes)."""
        return self._module

    def parameters(self):
        """Return learnable parameters (empty iterator for fixed modes)."""
        if self._module is not None:
            return self._module.parameters()
        return iter(())

    def coef_for(self, policy_idx: int, iteration: int = 0) -> torch.Tensor:
        """Return the effective entropy coefficient for one policy."""
        if self.mode == "learnable" and self._module is not None:
            return self._module.coef_for(policy_idx)

        sigma = self._current_sigma(iteration)
        coef = follower_entropy_coef(
            policy_idx, sigma, leader_has_entropy=self.leader_has_entropy
        )
        return torch.tensor(coef, dtype=torch.float32, device=self.device)

    def coefficients(self, iteration: int = 0) -> EntropyCoefOutput:
        """Return coefficients for all policies as an :class:`EntropyCoefOutput`."""
        if self.mode == "learnable" and self._module is not None:
            return EntropyCoefOutput(
                coefs=self._module.effective_coefs(),
                sigma=self._module.sigma(),
                learnable=True,
            )

        sigma = self._current_sigma(iteration)
        idx = torch.arange(self.num_blocks, dtype=torch.float32, device=self.device)
        coefs = sigma * idx
        if not self.leader_has_entropy and self.num_blocks > 0:
            coefs = coefs.clone()
            coefs[0] = 0.0
        sigma_vec = torch.full(
            (self.num_blocks,), float(sigma), dtype=torch.float32, device=self.device
        )
        return EntropyCoefOutput(coefs=coefs, sigma=sigma_vec, learnable=False)

    def state_dict(self):
        if self._module is not None:
            return self._module.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._module is not None:
            self._module.load_state_dict(state_dict)

    # -- internals ---------------------------------------------------------
    def _current_sigma(self, iteration: int) -> float:
        if self.mode == "linear":
            frac = min(max(float(iteration) / float(self.total_iterations), 0.0), 1.0)
            return self.sigma + frac * (self.sigma_final - self.sigma)
        return self.sigma


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_entropy_coefs(
    config: SAPGConfig,
    device: Optional[torch.device] = None,
    mode: Optional[str] = None,
) -> EntropyScheduler:
    """Build an :class:`EntropyScheduler` from a SAPG config.

    The mode is ``"learnable"`` when ``config.learnable_entropy_coef`` is set
    and ``config.entropy_coef > 0``; otherwise ``"fixed"``.  When
    ``config.entropy_coef == 0`` the scheduler still returns a valid object but
    every coefficient is zero (the paper's default for most tasks).
    """
    if mode is None:
        if getattr(config, "learnable_entropy_coef", False) and config.entropy_coef > 0:
            mode = "learnable"
        else:
            mode = "fixed"

    return EntropyScheduler(
        num_blocks=config.num_blocks,
        sigma=config.entropy_coef,
        mode=mode,
        leader_has_entropy=False,
        total_iterations=getattr(config, "num_iterations", 1),
        device=device,
    )
