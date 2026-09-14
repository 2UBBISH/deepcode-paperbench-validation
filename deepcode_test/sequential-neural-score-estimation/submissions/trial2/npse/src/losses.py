"""Denoising score-matching losses for NPSE, NLSE, prior estimation, and SNPSE.

This module implements the objectives described in the paper:

* ``npse_dsm_loss`` : non-sequential Neural Posterior Score Estimation (NPSE)
  denoising posterior score matching objective
  ``J_NPSE_DSM(psi)`` (paper Section 3 / Appendix A.1).

* ``prior_dsm_loss`` : denoising score matching for an implicit prior score
  network ``s_prior(theta_t, t)`` (paper Appendix / Component 8).

* ``nlse_dsm_loss`` : Neural Likelihood Score Estimation (NLSE) likelihood
  score-matching objective
  ``J_lik_DSM`` (paper Section 4 / Component 9).

* ``weighted_npse_dsm_loss`` : SNPSE-B importance-weighted posterior score
  objective (paper Appendix C / Component 7).

All objectives are instances of

    J = 1/2 * E[ lambda(t) * || s_psi(theta_t, x, t) - target ||^2 ]

where ``target = grad_theta log p_{t|0}(theta_t | theta_0)`` is the analytic
transition score and ``lambda(t) = g(t)^2`` is the standard score-matching
weighting (the default used throughout the codebase).

At the optimum, the NPSE score equals the posterior score
``grad_theta log p_t(theta_t | x)`` (Appendix A.1); hence minimizing the
plain MSE is sufficient.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch

from .networks import PriorScoreNetwork, ScoreNetwork
from .sde import SDE

__all__ = [
    "sample_times",
    "get_weighting",
    "npse_dsm_loss",
    "npse_dsm_squared_error",
    "prior_dsm_loss",
    "nlse_dsm_loss",
    "weighted_npse_dsm_loss",
]


Weighting = Union[str, Callable[[SDE, torch.Tensor], torch.Tensor]]


def sample_times(
    batch_size: int,
    T: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sample training times ``t ~ U(0, T)``.

    Parameters
    ----------
    batch_size:
        Number of time samples.
    T:
        Terminal diffusion time (usually ``sde.T``).
    device:
        Target device. Defaults to CPU.
    dtype:
        Target dtype. Defaults to ``torch.float32``.
    eps:
        Small floor used to keep times strictly positive. This avoids the
        (measure-zero) ``t = 0`` edge case and corresponding numerical issues
        without changing the objective in practice.

    Returns
    -------
    torch.Tensor of shape ``(batch_size,)``.
    """
    t = torch.rand(batch_size, device=device, dtype=dtype) * T
    t = t.clamp_min(min(eps, T * 0.5))
    return t


def get_weighting(
    sde: SDE,
    t: torch.Tensor,
    weighting: Weighting = "g2",
) -> torch.Tensor:
    """Return the per-sample loss weighting ``lambda(t)``.

    Parameters
    ----------
    sde:
        The forward SDE (provides ``g(t)``).
    t:
        Time tensor of shape ``(batch,)`` (or a scalar tensor).
    weighting:
        Either ``"g2"`` (default; ``lambda(t) = g(t)**2``), ``"none"`` /
        ``None`` (unit weights), or a callable ``(sde, t) -> weights``.

    Returns
    -------
    torch.Tensor broadcastable against a ``(batch,)`` loss vector.
    """
    if weighting is None or weighting == "none":
        return torch.ones_like(t, dtype=t.dtype, device=t.device)
    if callable(weighting):
        return weighting(sde, t)
    if isinstance(weighting, str) and weighting.lower() == "g2":
        g = sde.g(t)
        return g * g
    raise ValueError(
        f"Unknown weighting {weighting!r}. Expected 'g2', 'none', or a callable."
    )


def _squared_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample squared L2 error ``||pred - target||^2``."""
    diff = pred - target
    return torch.sum(diff * diff, dim=-1)


def npse_dsm_squared_error(
    sde: SDE,
    score_net: ScoreNetwork,
    theta0: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Compute unweighted per-sample NPSE denoising score-matching error.

    Returns the vector ``||s_psi(theta_t, x, t) - grad log p_{t|0}||^2`` for
    each sample. Useful for validation monitoring and diagnostics.
    """
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    pred = score_net(theta_t, x, t)
    return _squared_error(pred, target)


def npse_dsm_loss(
    sde: SDE,
    score_net: ScoreNetwork,
    theta0: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    weighting: Weighting = "g2",
) -> torch.Tensor:
    """NPSE denoising posterior score-matching loss (paper Section 3).

    Implements

        J_NPSE_DSM(psi) =
            1/2 * E[ lambda(t) * ||s_psi(theta_t, x, t)
                             - grad_theta log p_{t|0}(theta_t | theta_0)||^2 ]

    with ``theta0 ~ p(theta)``, ``x ~ p(x | theta0)``, ``t ~ U(0, T)`` and
    ``theta_t ~ p_{t|0}(theta_t | theta0)``.

    Parameters
    ----------
    sde:
        Forward SDE providing the transition kernel and analytic target score.
    score_net:
        Posterior score network ``s_psi(theta_t, x, t)``.
    theta0:
        Prior samples, shape ``(batch, d)``.
    x:
        Simulated observations, shape ``(batch, p)``.
    t:
        Diffusion times, shape ``(batch,)``.
    weighting:
        Loss weighting (default ``g(t)^2``).

    Returns
    -------
    Scalar loss tensor.
    """
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    pred = score_net(theta_t, x, t)
    weights = get_weighting(sde, t, weighting)
    squared = _squared_error(pred, target)
    return 0.5 * (weights * squared).mean()


def prior_dsm_loss(
    sde: SDE,
    prior_score_net: PriorScoreNetwork,
    theta0: torch.Tensor,
    t: torch.Tensor,
    weighting: Weighting = "g2",
) -> torch.Tensor:
    """Prior denoising score-matching loss for an implicit prior.

    Trains ``s_prior(theta_t, t)`` to approximate ``grad_theta log p_t(theta_t)``
    by regressing against the analytic transition score
    ``grad_theta log p_{t|0}(theta_t | theta0)``.
    """
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    pred = prior_score_net(theta_t, t)
    weights = get_weighting(sde, t, weighting)
    squared = _squared_error(pred, target)
    return 0.5 * (weights * squared).mean()


def nlse_dsm_loss(
    sde: SDE,
    likelihood_score_net: ScoreNetwork,
    prior_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    theta0: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    weighting: Weighting = "g2",
) -> torch.Tensor:
    """NLSE likelihood score-matching loss (paper Section 4).

    Using the decomposition

        grad_theta log p_t(theta_t | x)
            = grad_theta log p_t(x | theta_t) + grad_theta log p_t(theta_t),

    the likelihood score network ``s_psi_lik`` is trained with target

        grad_theta log p_{t|0}(theta_t | theta0)
            - grad_theta log p_t(theta_t).

    The final posterior score is recovered as
    ``s_psi_post = s_psi_lik + grad_theta log p_t(theta_t)`` (see ``src/nlse.py``).
    """
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    prior_score = prior_score_fn(theta_t, t)
    pred = likelihood_score_net(theta_t, x, t) + prior_score
    weights = get_weighting(sde, t, weighting)
    squared = _squared_error(pred, target)
    return 0.5 * (weights * squared).mean()


def weighted_npse_dsm_loss(
    sde: SDE,
    score_net: ScoreNetwork,
    theta0: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
    weighting: Weighting = "g2",
) -> torch.Tensor:
    """Importance-weighted posterior score-matching loss (SNPSE-B).

    Each sample is reweighted by ``w_i = p(theta0_i) / p_tilde(theta0_i)`` so
    that the proposal-prior expectation is corrected to the target prior
    expectation (Appendix C / Component 7).
    """
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    pred = score_net(theta_t, x, t)
    time_weights = get_weighting(sde, t, weighting)
    squared = _squared_error(pred, target)
    return 0.5 * (weights * time_weights * squared).mean()
