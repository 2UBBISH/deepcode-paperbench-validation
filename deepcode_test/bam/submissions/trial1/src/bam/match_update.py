"""Closed-form Batch-and-Match (MATCH) mean and covariance updates.

This module implements the optimality conditions for the BaM proximal
objective

    L_BaM(q) = D_hat_{q_t}(q; p) + (2 / lambda) KL(q_t ; q),

where q = N(mu, Sigma) is a full-covariance Gaussian variational family and
q_t = N(mu_t, Sigma_t) is the previous iterate.  Using the empirical score
divergence evaluated on a batch drawn from q_t, the minimizer of L_BaM has a
closed form:

    Sigma_{t+1} solves   Sigma U Sigma + Sigma = V,
    mu_{t+1} = lambda/(1+lambda) (zbar + Sigma_{t+1} gbar)
               + 1/(1+lambda) mu_t,

with

    U = lambda Gamma + lambda/(1+lambda) gbar gbar^T,
    V = Sigma_t + lambda C
        + lambda/(1+lambda) (mu_t - zbar)(mu_t - zbar)^T.

The quadratic matrix equation is solved with the full-rank/low-rank solvers
provided by :mod:`src.bam.quadratic_solver`.
"""

from __future__ import annotations

import jax.numpy as jnp

from .batch_stats import BatchStats
from .divergences import kl_gaussian, score_divergence_batch_form
from .quadratic_solver import solve_quadratic, solve_quadratic_full

__all__ = [
    "match_uv",
    "match_covariance",
    "match_mean",
    "match_update",
    "bam_objective",
    "match_gsm_update",
]


def _validate_inputs(
    mu_t: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    zbar: jnp.ndarray,
    gbar: jnp.ndarray,
    C: jnp.ndarray,
    Gamma: jnp.ndarray,
    lam: float,
) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Validate shapes and symmetrize square matrices."""
    if lam < 0.0:
        raise ValueError(f"lambda must be non-negative, got {lam}")

    mu_t = jnp.atleast_1d(mu_t)
    zbar = jnp.atleast_1d(zbar)
    gbar = jnp.atleast_1d(gbar)
    d = mu_t.shape[0]

    for name, arr in (
        ("Sigma_t", Sigma_t),
        ("C", C),
        ("Gamma", Gamma),
    ):
        if arr.ndim != 2 or arr.shape[0] != d or arr.shape[1] != d:
            raise ValueError(
                f"{name} must have shape ({d}, {d}), got {arr.shape}"
            )

    for name, arr in (("zbar", zbar), ("gbar", gbar)):
        if arr.shape != (d,):
            raise ValueError(f"{name} must have shape ({d},), got {arr.shape}")

    Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
    C = 0.5 * (C + C.T)
    Gamma = 0.5 * (Gamma + Gamma.T)
    return d, mu_t, Sigma_t, zbar, gbar, C, Gamma


def match_uv(
    mu_t: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    zbar: jnp.ndarray,
    gbar: jnp.ndarray,
    C: jnp.ndarray,
    Gamma: jnp.ndarray,
    lam: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the symmetric positive-semidefinite matrices U and V.

    These define the covariance update ``Sigma U Sigma + Sigma = V``.

    Parameters
    ----------
    mu_t : (D,) array
        Current variational mean.
    Sigma_t : (D, D) array
        Current variational covariance.
    zbar : (D,) array
        Batch mean of samples from q_t.
    gbar : (D,) array
        Batch mean of target scores ``grad log p(z)``.
    C : (D, D) array
        Batch covariance of samples.
    Gamma : (D, D) array
        Batch covariance of target scores.
    lam : float
        Proximal/learning-rate parameter ``lambda_t >= 0``.

    Returns
    -------
    U : (D, D) array
        ``lam * Gamma + lam/(1+lam) * gbar gbar^T``.
    V : (D, D) array
        ``Sigma_t + lam * C + lam/(1+lam) * (mu_t - zbar)(mu_t - zbar)^T``.
    """
    _, mu_t, Sigma_t, zbar, gbar, C, Gamma = _validate_inputs(
        mu_t, Sigma_t, zbar, gbar, C, Gamma, lam
    )

    lam = jnp.asarray(lam, dtype=Sigma_t.dtype)
    tilt = lam / (1.0 + lam)

    U = lam * Gamma + tilt * jnp.outer(gbar, gbar)
    V = Sigma_t + lam * C + tilt * jnp.outer(mu_t - zbar, mu_t - zbar)

    U = 0.5 * (U + U.T)
    V = 0.5 * (V + V.T)
    return U, V


def match_covariance(
    Sigma_t: jnp.ndarray,
    C: jnp.ndarray,
    Gamma: jnp.ndarray,
    zbar: jnp.ndarray,
    gbar: jnp.ndarray,
    mu_t: jnp.ndarray,
    lam: float,
    rank_tol: float = 1e-10,
) -> jnp.ndarray:
    """Compute the BaM covariance update in closed form.

    Parameters
    ----------
    Sigma_t : (D, D) array
        Current variational covariance.
    C : (D, D) array
        Batch covariance of samples.
    Gamma : (D, D) array
        Batch covariance of target scores.
    zbar : (D,) array
        Batch mean of samples.
    gbar : (D,) array
        Batch mean of target scores.
    mu_t : (D,) array
        Current variational mean.
    lam : float
        Proximal/learning-rate parameter.
    rank_tol : float, optional
        Relative eigenvalue tolerance used when selecting the low-rank solver.

    Returns
    -------
    Sigma_new : (D, D) array
        Solution of ``Sigma_new U Sigma_new + Sigma_new = V``.
    """
    U, V = match_uv(mu_t, Sigma_t, zbar, gbar, C, Gamma, lam)
    return solve_quadratic(U, V, rank_tol=rank_tol)


def match_mean(
    mu_t: jnp.ndarray,
    zbar: jnp.ndarray,
    gbar: jnp.ndarray,
    Sigma_new: jnp.ndarray,
    lam: float,
) -> jnp.ndarray:
    """Compute the BaM mean update in closed form.

    ``mu_new = lam/(1+lam) * (zbar + Sigma_new @ gbar) + 1/(1+lam) * mu_t``.

    Parameters
    ----------
    mu_t : (D,) array
        Current variational mean.
    zbar : (D,) array
        Batch mean of samples.
    gbar : (D,) array
        Batch mean of target scores.
    Sigma_new : (D, D) array
        Updated covariance from :func:`match_covariance`.
    lam : float
        Proximal/learning-rate parameter.

    Returns
    -------
    mu_new : (D,) array
        Updated variational mean.
    """
    mu_t = jnp.atleast_1d(mu_t)
    zbar = jnp.atleast_1d(zbar)
    gbar = jnp.atleast_1d(gbar)

    if lam < 0.0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    if mu_t.shape != zbar.shape or mu_t.shape != gbar.shape:
        raise ValueError(
            "mu_t, zbar, and gbar must have identical shapes; got "
            f"{mu_t.shape}, {zbar.shape}, {gbar.shape}"
        )

    lam = jnp.asarray(lam, dtype=mu_t.dtype)
    c = lam / (1.0 + lam)
    return c * (zbar + Sigma_new @ gbar) + (1.0 - c) * mu_t


def match_update(
    mu_t: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    stats: BatchStats,
    lam: float,
    rank_tol: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform one full BaM (Batch-and-Match) update.

    Parameters
    ----------
    mu_t : (D,) array
        Current variational mean.
    Sigma_t : (D, D) array
        Current variational covariance.
    stats : BatchStats
        Batch statistics ``(zbar, gbar, C, Gamma)`` computed from samples
        and target scores.
    lam : float
        Proximal/learning-rate parameter.
    rank_tol : float, optional
        Tolerance passed to the quadratic solver.

    Returns
    -------
    mu_new : (D,) array
    Sigma_new : (D, D) array
        Updated variational parameters.
    """
    Sigma_new = match_covariance(
        Sigma_t,
        stats.C,
        stats.Gamma,
        stats.zbar,
        stats.gbar,
        mu_t,
        lam,
        rank_tol=rank_tol,
    )
    mu_new = match_mean(mu_t, stats.zbar, stats.gbar, Sigma_new, lam)
    return mu_new, Sigma_new


def bam_objective(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    mu_t: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    zbar: jnp.ndarray,
    gbar: jnp.ndarray,
    C: jnp.ndarray,
    Gamma: jnp.ndarray,
    lam: float,
    include_const: bool = False,
    z: jnp.ndarray | None = None,
    g: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Evaluate the BaM objective for candidate ``q = N(mu, Sigma)``.

    The objective is

        L_BaM(q) = D_hat_{q_t}(q; p) + (2 / lam) KL(q_t ; q),

    where the score-divergence term is evaluated with the batch statistics of
    ``q_t``.

    Parameters
    ----------
    mu, Sigma : candidate variational parameters.
    mu_t, Sigma_t : current variational parameters defining ``q_t``.
    zbar, gbar, C, Gamma : batch statistics from ``q_t``.
    lam : proximal/learning-rate parameter.
    include_const : bool, optional
        If ``True``, include the data-dependent constant of the empirical
        divergence by passing raw ``z`` and ``g``.
    z, g : optional raw batch samples and scores used only when
        ``include_const`` is ``True``.

    Returns
    -------
    objective : scalar array
    """
    if lam <= 0.0:
        raise ValueError(f"lambda must be positive, got {lam}")

    div_kwargs = {}
    if include_const:
        if z is None or g is None:
            raise ValueError(
                "include_const=True requires raw batch arrays z and g"
            )
        div_kwargs = {"z": z, "g": g}

    div = score_divergence_batch_form(
        mu, Sigma, zbar, gbar, C, Gamma, **div_kwargs
    )
    kl = kl_gaussian(mu_t, Sigma_t, mu, Sigma)
    return div + (2.0 / lam) * kl


def match_gsm_update(
    mu_t: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    z: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GSM update recovered as the ``B=1, lambda -> infinity`` BaM limit.

    For a single sample ``z`` and its score ``g = grad log p(z)``,

        U = g g^T,
        V = Sigma_t + (mu_t - z)(mu_t - z)^T,
        mu_{t+1} = Sigma_{t+1} g + z.

    Parameters
    ----------
    mu_t : (D,) array
    Sigma_t : (D, D) array
    z : (D,) array
        Single latent sample.
    g : (D,) array
        Target score evaluated at ``z``.

    Returns
    -------
    mu_new : (D,) array
    Sigma_new : (D, D) array
    """
    mu_t = jnp.atleast_1d(mu_t)
    z = jnp.atleast_1d(z)
    g = jnp.atleast_1d(g)
    d = mu_t.shape[0]

    if z.shape != (d,) or g.shape != (d,):
        raise ValueError(
            f"z and g must have shape ({d},), got {z.shape} and {g.shape}"
        )

    U = jnp.outer(g, g)
    V = Sigma_t + jnp.outer(mu_t - z, mu_t - z)
    Sigma_new = solve_quadratic_full(U, V)
    mu_new = Sigma_new @ g + z
    return mu_new, Sigma_new
