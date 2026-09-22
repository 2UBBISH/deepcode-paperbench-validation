"""Denoising score-matching losses for SNPSE / NPSE / NLSE.

This module implements Monte-Carlo estimates of every training objective used in
the paper *Sequential Neural Posterior Score Estimation*:

======================  ==========================================
Objective               Paper equation
======================  ==========================================
NPSE  (eq. 7 / 6)       ``npse_loss`` / ``dsm_loss``
TSNPSE (eq. 11)         ``tsnpse_loss``
SNPSE-A (eq. 79)        ``snpse_a_loss``
SNPSE-B (eq. 15 / 99)   ``snpse_b_loss``
SNPSE-C (eq. 102)       ``snpse_c_loss``
NLSE likelihood (57)    ``nlse_loss``
NLSE prior score (64)   ``prior_score_loss``
======================  ==========================================

All objectives share the same abstract shape

    J(psi) = 1/2 int_0^T lambda_t E_{...}[ || pred - target ||^2 ] dt

and are estimated with a single sample per datapoint:

    t          ~ U(0, T)
    theta_t    ~ p_{t|0}(theta_t | theta_0)      (forward process, eq. 2)
    target     = grad_{theta_t} log p_{t|0}(theta_t | theta_0)   (closed form)

The ``lambda_t`` weighting is taken from the SDE (see :mod:`snpse.sdes`);
following Song et al. (2021) the default is ``lambda_t = g(t)^2``, which is the
maximum-likelihood / likelihood-weighting used throughout the paper.

Conventions
-----------
* ``theta0`` has shape ``(B, d)``, ``x`` has shape ``(B, p)``, ``t`` has shape
  ``(B,)`` and lives in ``[0, T]``.
* A *score function* is any callable with signature
  ``score_fn(theta_t, x, t) -> (B, d)`` returning the score of the perturbed
  posterior (i.e. the output of the score network, possibly re-parameterised).
* A *log-density function* is any callable with signature
  ``log_prob_fn(theta_t, t) -> (B,)`` that is differentiable w.r.t. ``theta_t``.
  Its gradient is obtained with ``torch.autograd.grad`` (see
  :func:`grad_log_density`).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .sdes import SDE

__all__ = [
    "ScoreFn",
    "grad_log_density",
    "time_perturbation",
    "dsm_loss",
    "npse_loss",
    "tsnpse_loss",
    "snpse_a_loss",
    "snpse_b_loss",
    "snpse_c_loss",
    "nlse_loss",
    "prior_score_loss",
    "importance_weight",
    "dsm_loss_from_batch",
]

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
LogProbFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _bcast(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast ``value`` so that it has the same number of dims as ``ref``."""
    if value.dim() == ref.dim():
        return value
    out = value
    while out.dim() < ref.dim():
        out = out.unsqueeze(-1)
    return out


def _as_batch_t(t, batch_size: int, device, dtype, generator=None) -> torch.Tensor:
    """Return ``t`` as a ``(B,)`` tensor living on ``device``."""
    if t is None:
        return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    t = torch.as_tensor(t, device=device, dtype=dtype)
    if t.dim() == 0:
        t = t.expand(batch_size)
    return t


def _reduce(per_sample: torch.Tensor, reduce: str) -> torch.Tensor:
    if reduce == "none":
        return per_sample
    if reduce == "sum":
        return per_sample.sum()
    if reduce == "mean":
        return per_sample.mean()
    raise ValueError(f"unknown reduction '{reduce}'")


def grad_log_density(
    log_prob_fn: LogProbFn,
    theta_t: torch.Tensor,
    t: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """``grad_{theta_t} log_prob_fn(theta_t, t)`` via automatic differentiation.

    The paper estimates the perturbed *prior* score (Appendix B.2.1) and the
    proposal prior score (Appendix C.4.3) through ``torch.autograd`` exactly as
    done here.  The input ``theta_t`` must require grad; it is enabled on the
    fly so that the caller does not have to worry about it.
    """
    with torch.enable_grad():
        theta = theta_t if theta_t.requires_grad else theta_t.detach().requires_grad_(True)
        logp = log_prob_fn(theta, t)
        if logp.dim() > 1:
            logp = logp.sum(dim=tuple(range(1, logp.dim())))
        (grad,) = torch.autograd.grad(
            logp.sum(), theta, create_graph=create_graph, retain_graph=create_graph
        )
    return grad


def importance_weight(
    theta0: torch.Tensor,
    prior_log_prob_fn: LogProbFn,
    proposal_log_prob_fn: LogProbFn,
    t: Optional[torch.Tensor] = None,
    normalise: bool = False,
    clip: Optional[float] = None,
) -> torch.Tensor:
    """Importance weight ``p(theta_0) / ptilde^r(theta_0)`` (paper eq. 13/100/15).

    Here ``prior_log_prob_fn(theta0, t)`` and ``proposal_log_prob_fn(theta0, t)``
    are log-density callables evaluated at ``theta_0`` (the ``t`` argument is
    ``None`` for plain densities, but the signature is kept uniform).
    """
    with torch.no_grad():
        log_p = prior_log_prob_fn(theta0, t)
        log_q = proposal_log_prob_fn(theta0, t)
        if log_p.dim() > 1:
            log_p = log_p.sum(dim=tuple(range(1, log_p.dim())))
        if log_q.dim() > 1:
            log_q = log_q.sum(dim=tuple(range(1, log_q.dim())))
        w = torch.exp(log_p - log_q)
        if clip is not None:
            w = w.clamp(max=float(clip))
        if normalise:
            w = w / w.sum().clamp_min(1e-12) * w.numel()
    return w


def time_perturbation(
    sde: SDE,
    theta0: torch.Tensor,
    t=None,
    generator=None,
):
    """Draw ``t ~ U(0, T)`` / ``theta_t ~ p_{t|0}(.|theta_0)`` and the target.

    Returns ``(t, theta_t, score_target)``.
    """
    t = _as_batch_t(
        t, theta0.shape[0], theta0.device, theta0.dtype, generator=generator
    )
    theta_t = sde.sample_perturbed(theta0, t, generator=generator)
    target = sde.score_target(theta0, theta_t, t)
    return t, theta_t, target


# --------------------------------------------------------------------------- #
# core DSM loss (covers eq. 7, 11 and 79)
# --------------------------------------------------------------------------- #
def dsm_loss(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    generator=None,
    lambda_t: Optional[torch.Tensor] = None,
    use_sde_weighting: bool = True,
    reduce: str = "mean",
    create_graph: bool = True,
    theta_t: Optional[torch.Tensor] = None,
    score_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Monte-Carlo estimate of the (weighted) denoising score-matching loss.

    .. math::
        \mathcal{J}(\psi) = \frac{1}{2}\int_0^T \lambda_t
        \mathbb{E}_{p_{t|0}(\theta_t|\theta_0)p(x|\theta_0)q(\theta_0)}
        \left[w(\theta_0)\left\| s_\psi(\theta_t,x,t)
        - \nabla_{\theta_t}\log p_{t|0}(\theta_t|\theta_0)\right\|^2\right] dt

    * ``weight=None`` reduces exactly to eq. (7) (NPSE, with ``q = p``) and to
      eq. (11) (TSNPSE, with ``q = \tilde p^r``) -- the only difference between
      the two is the distribution the samples ``theta0`` are drawn from.
    * ``weight=p(theta_0)/\tilde p^r(theta_0)`` gives the SNPSE-B objective
      (eq. 15 / 99).
    """
    t, theta_t, score_target = _maybe_perturb(
        sde, theta0, t, generator, theta_t, score_target
    )

    pred = score_fn(theta_t, x, t)
    diff = pred - score_target
    per_sample = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=-1)

    if use_sde_weighting or lambda_t is not None:
        lam = sde.weighting(t) if lambda_t is None else lambda_t
        per_sample = _bcast(lam, per_sample) * per_sample

    per_sample = 0.5 * per_sample

    if weight is not None:
        per_sample = _bcast(weight.view(-1), per_sample) * per_sample

    return _reduce(per_sample, reduce)


def _maybe_perturb(sde, theta0, t, generator, theta_t, score_target):
    """Reuse provided perturbed samples when available (cheaper, same objective)."""
    if theta_t is None or score_target is None:
        t, theta_t, score_target = time_perturbation(
            sde, theta0, t=t, generator=generator
        )
    elif t is None:
        t = _as_batch_t(None, theta0.shape[0], theta0.device, theta0.dtype)
    return t, theta_t, score_target


# --------------------------------------------------------------------------- #
# NPSE / TSNPSE
# --------------------------------------------------------------------------- #
def npse_loss(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """NPSE objective: Monte-Carlo estimate of eq. (7).

    Samples ``theta0 ~ p(theta)``.  Used in round 1 (and in the amortised /
    non-sequential setting).
    """
    return dsm_loss(score_fn, sde, theta0, x, **kwargs)


def tsnpse_loss(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """TSNPSE objective: Monte-Carlo estimate of eq. (11).

    Functionally identical to :func:`npse_loss`; the samples ``theta0`` must be
    drawn from the *truncated proposal* ``\\tilde p^r`` (see
    :mod:`snpse.hpr` and :mod:`snpse.tsnpse`).  By Proposition 3.1 no importance
    correction is required, which is precisely what this function encodes.
    """
    return dsm_loss(score_fn, sde, theta0, x, **kwargs)


def snpse_a_loss(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """SNPSE-A objective: Monte-Carlo estimate of eq. (79).

    Identical to :func:`dsm_loss` over samples from the proposal prior
    ``\\tilde p^r`` (no importance weight inside the loss).  The correction is
    applied *post hoc* by SIR, see :mod:`snpse.snpse_variants`.
    """
    return dsm_loss(score_fn, sde, theta0, x, **kwargs)


def snpse_b_loss(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""SNPSE-B objective: Monte-Carlo estimate of eq. (15) (eq. 99 in App. C.3).

    .. math::
        \frac{1}{2}\int_0^T \lambda_t \mathbb{E}_{p_{t|0} p(x|\theta_0)\tilde p^r(\theta_0)}
        \left[\frac{p(\theta_0)}{\tilde p^r(\theta_0)}
        \left\|s_\psi - \nabla_{\theta_t}\log p_{t|0}\right\|^2\right]dt

    ``weight`` is the per-sample importance ratio
    ``p(theta0) / \tilde p^r(theta0)`` (see :func:`importance_weight`).
    """
    return dsm_loss(score_fn, sde, theta0, x, weight=weight, **kwargs)


# --------------------------------------------------------------------------- #
# SNPSE-C (eq. 102): score-space correction
# --------------------------------------------------------------------------- #
def snpse_c_loss(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    proposal_prior_log_prob_fn: LogProbFn,
    prior_log_prob_fn: LogProbFn,
    t: Optional[torch.Tensor] = None,
    generator=None,
    reduce: str = "mean",
    **kwargs,
) -> torch.Tensor:
    r"""SNPSE-C objective: Monte-Carlo estimate of eq. (102) (App. C.4.1).

    The network is trained to approximate the score of the *proposal* posterior
    ``\tilde p^r_t(\theta_t|x)`` through the re-parameterisation

    .. math::
        \tilde s^r_\psi(\theta_t, x, t) = s_\psi(\theta_t,x,t)
            + \nabla_\theta \log \tilde p^r_t(\theta_t)
            - \nabla_\theta \log p_t(\theta_t)

    and the loss is the usual DSM loss applied to ``\tilde s^r_\psi``.
    ``proposal_prior_log_prob_fn``/``prior_log_prob_fn`` are log-density
    callables of ``(theta_t, t)`` for the perturbed proposal prior and the
    perturbed prior respectively; their gradients are taken with autograd.
    """
    t, theta_t, score_target = time_perturbation(
        sde, theta0, t=t, generator=generator
    )

    s = score_fn(theta_t, x, t)
    proposal_prior_score = grad_log_density(
        proposal_prior_log_prob_fn, theta_t, t, create_graph=True
    )
    prior_score = grad_log_density(
        prior_log_prob_fn, theta_t, t, create_graph=True
    )
    tilde_s = s + proposal_prior_score - prior_score

    per_sample = (tilde_s - score_target).reshape(tilde_s.shape[0], -1).pow(2).sum(-1)
    per_sample = 0.5 * _bcast(sde.weighting(t), per_sample) * per_sample
    return _reduce(per_sample, reduce)


# --------------------------------------------------------------------------- #
# NLSE (Appendix B.1)
# --------------------------------------------------------------------------- #
def nlse_loss(
    score_lik_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    prior_score_fn: Optional[ScoreFn] = None,
    t: Optional[torch.Tensor] = None,
    generator=None,
    lambda_t: Optional[torch.Tensor] = None,
    use_sde_weighting: bool = True,
    reduce: str = "mean",
) -> torch.Tensor:
    r"""NLSE likelihood-score objective: Monte-Carlo estimate of eq. (57).

    .. math::
        \mathcal{J}_{\text{lik}}^{\text{DSM}}(\psi_{\text{lik}}) =
        \frac{1}{2}\int_0^T \lambda_t
        \mathbb{E}_{p_{t|0}(\theta_t|\theta_0)p(\theta_0,x)}
        \left[\left\| s_{\psi_{\text{lik}}}(\theta_t,x,t)
        + \nabla_\theta\log p_t(\theta_t)
        - \nabla_{\theta_t}\log p_{t|0}(\theta_t|\theta_0)\right\|^2\right]dt

    ``prior_score_fn(theta_t, t)`` returns the perturbed **prior** score
    ``\nabla_\theta \log p_t(\theta_t)``; when it is ``None`` the term is taken
    to be zero, which is the SDE-specific convention ``f = 0, g = \tau_t``
    combined with an (approximately) flat prior (Appendix B.2.1).
    """
    t, theta_t, score_target = time_perturbation(
        sde, theta0, t=t, generator=generator
    )
    pred = score_lik_fn(theta_t, x, t)

    if prior_score_fn is None:
        prior_score = torch.zeros_like(theta_t)
    else:
        prior_score = prior_score_fn(theta_t, t)
        if isinstance(prior_score, tuple):
            prior_score = prior_score[0]

    diff = pred + prior_score - score_target
    per_sample = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=-1)

    if use_sde_weighting or lambda_t is not None:
        lam = sde.weighting(t) if lambda_t is None else lambda_t
        per_sample = _bcast(lam, per_sample) * per_sample

    per_sample = 0.5 * per_sample
    return _reduce(per_sample, reduce)


def prior_score_loss(
    score_pri_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    generator=None,
    lambda_t: Optional[torch.Tensor] = None,
    use_sde_weighting: bool = True,
    reduce: str = "mean",
) -> torch.Tensor:
    """Prior-score matching objective: Monte-Carlo estimate of eq. (64) (Algorithm 2).

    Trains ``s_{psi_pri}(theta_t, t) ~= grad log p_t(theta_t)`` on prior samples.
    """
    t, theta_t, score_target = time_perturbation(
        sde, theta0, t=t, generator=generator
    )
    pred = score_pri_fn(theta_t, t)
    if isinstance(pred, tuple):
        pred = pred[0]
    diff = pred - score_target
    per_sample = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=-1)

    if use_sde_weighting or lambda_t is not None:
        lam = sde.weighting(t) if lambda_t is None else lambda_t
        per_sample = _bcast(lam, per_sample) * per_sample

    per_sample = 0.5 * per_sample
    return _reduce(per_sample, reduce)


# --------------------------------------------------------------------------- #
# convenience: loss from a pre-drawn batch (used by the sequential drivers)
# --------------------------------------------------------------------------- #
def dsm_loss_from_batch(
    score_fn: ScoreFn,
    sde: SDE,
    theta0: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    generator=None,
    **kwargs,
) -> torch.Tensor:
    """Alias of :func:`dsm_loss` (kept for readability at the call site)."""
    return dsm_loss(
        score_fn, sde, theta0, x, weight=weight, t=t, generator=generator, **kwargs
    )
