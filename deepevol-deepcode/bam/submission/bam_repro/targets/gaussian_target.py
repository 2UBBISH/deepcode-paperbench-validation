"""Gaussian synthetic target distributions (Section 5.1, Appendix E.3).

The paper constructs Gaussian targets of increasing dimension ``D = 4, 16, 64,
256``.  Each target distribution is generated randomly: the covariance is
constructed by generating a ``D x D`` matrix ``A`` and computing
``Sigma_* = A A^T`` (Appendix E.3).

For a Gaussian target ``p = N(mu_*, Sigma_*)`` everything of interest is
available in closed form:

* the score ``s(z) = grad_z log p(z) = Sigma_*^{-1} (mu_* - z)``,
* the value of the log density,
* the forward KL divergence ``KL(p ; q)`` and the reverse KL divergence
  ``KL(q ; p)`` between the target ``p`` and a *full-covariance* Gaussian
  variational distribution ``q = N(mu, Sigma)``.

The KL between two multivariate Gaussians is

    KL(N(m1, S1) ; N(m2, S2))
        = 1/2 [ tr(S2^{-1} S1) + (m2 - m1)^T S2^{-1} (m2 - m1) - D
                + log det(S2) - log det(S1) ] .

Note that in the paper's notation ``KL(p ; q)`` is the *forward* (mass
covering) direction with ``p`` the target, i.e. ``KL(N(mu_*, Sigma_*) ||
N(mu, Sigma))`` and ``KL(q ; p)`` is the reverse direction
``KL(N(mu, Sigma) || N(mu_*, Sigma_*))``.

The score-based divergence ``D(q ; p)`` from Proposition A.7 is also provided
for completeness,

    D(q ; p) = tr[(I - Sigma Sigma_*^{-1})^2]
               + (nu - mu)^T Sigma_*^{-1} Sigma Sigma_*^{-1} (nu - mu) ,

(with ``mu``/``Sigma`` the parameters of ``p`` and ``nu``/``Psi`` those of
``q`` in the paper's notation).

The module works with either NumPy or JAX arrays (namespace inferred from the
inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from ..bam.matrix_equations import (
    _HAS_JAX,
    array_namespace,
    ensure_spd,
    inverse_spd,
    symmetrize,
)

if _HAS_JAX:  # pragma: no cover - optional dependency
    import jax  # noqa: F401
    import jax.numpy as jnp

__all__ = [
    "GaussianTarget",
    "random_gaussian_target",
    "random_gaussian_A",
    "gaussian_forward_kl",
    "gaussian_reverse_kl",
    "gaussian_kl",
    "gaussian_score_divergence",
    "log_det_spd",
]


# ---------------------------------------------------------------------------
# small linear algebra helpers
# ---------------------------------------------------------------------------
def log_det_spd(A, jitter: float = 0.0):
    """Log-determinant of a symmetric positive-definite matrix via Cholesky."""
    xp = array_namespace(A)
    D = A.shape[-1]
    A = symmetrize(A)
    if jitter > 0.0 or True:  # always nudge slightly for numerical robustness
        eps = max(float(jitter), 1e-12)
        A = A + eps * xp.eye(D, dtype=A.dtype)
    L = xp.linalg.cholesky(A)
    return 2.0 * xp.sum(xp.log(xp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)


def _quad_form(x, M):
    """Return ``x^T M x`` for vectors (..., D) and matrices (..., D, D)."""
    xp = array_namespace(x, M)
    return xp.sum(x * (M @ x), axis=-1)


# ---------------------------------------------------------------------------
# the target
# ---------------------------------------------------------------------------
@dataclass
class GaussianTarget:
    """Multivariate Gaussian target ``p = N(mu_*, Sigma_*)``.

    Parameters
    ----------
    mean:
        ``mu_*`` of shape ``(D,)``.
    cov:
        ``Sigma_*`` of shape ``(D, D)``; symmetrised and projected to the
        positive-definite cone on construction.
    A:
        Optional factor with ``A A^T = Sigma_*`` (kept for reproducibility /
        debugging; the paper generates the covariance this way).
    name:
        Optional label used in plots/tables.
    """

    mean: Any
    cov: Any
    A: Optional[Any] = None
    name: str = "gaussian"
    _precision: Optional[Any] = field(default=None, repr=False, compare=False)

    # -- basic properties -------------------------------------------------
    @property
    def dim(self) -> int:
        return int(np.shape(self.mean)[-1])

    @property
    def xp(self):
        return array_namespace(self.mean, self.cov)

    @property
    def precision(self):
        """``Sigma_*^{-1}`` (cached)."""
        if self._precision is None:
            object.__setattr__(self, "_precision", inverse_spd(self.cov))
        return self._precision

    # -- target interface -------------------------------------------------
    def score(self, z):
        """Score ``grad_z log p(z) = Sigma_*^{-1}(mu_* - z)``."""
        xp = array_namespace(z)
        P = self.precision
        return (self.mean - z) @ P.T

    # alias
    def grad_log_prob(self, z):
        return self.score(z)

    def log_prob(self, z):
        """Log density of ``p`` at ``z`` (up to full normalising constant)."""
        xp = array_namespace(z)
        D = self.dim
        d = z - self.mean
        quad = _quad_form(d, self.precision)
        log_norm = 0.5 * (D * xp.log(2.0 * xp.pi) + log_det_spd(self.cov))
        return -0.5 * quad - log_norm

    def sample(self, n: int, rng=None, key=None, dtype=None):
        """Draw ``n`` samples from the target (used for MC metrics/reference)."""
        from ..bam.vi_base import standard_normal

        xp = self.xp
        D = self.dim
        eps = standard_normal((n, D), xp=xp, rng=rng, key=key, dtype=dtype)
        L = xp.linalg.cholesky(ensure_spd(self.cov))
        return self.mean[None, :] + eps @ L.T

    # -- divergences to a variational Gaussian q --------------------------
    def forward_kl(self, mu, Sigma, jitter: float = 0.0):
        """``KL(p ; q) = KL(N(mu_*, Sigma_*) || N(mu, Sigma))`` in closed form."""
        return gaussian_forward_kl(self.mean, self.cov, mu, Sigma, jitter=jitter)

    def reverse_kl(self, mu, Sigma, jitter: float = 0.0):
        """``KL(q ; p) = KL(N(mu, Sigma) || N(mu_*, Sigma_*))`` in closed form."""
        return gaussian_reverse_kl(self.mean, self.cov, mu, Sigma, jitter=jitter)

    # generic dispatch, ``direction`` in {"forward", "reverse"}
    def kl(self, mu, Sigma, direction: str = "forward", jitter: float = 0.0):
        direction = str(direction).lower()
        if direction in ("forward", "fwd", "f", "p;q", "pq"):
            return self.forward_kl(mu, Sigma, jitter=jitter)
        if direction in ("reverse", "rev", "r", "q;p", "qp"):
            return self.reverse_kl(mu, Sigma, jitter=jitter)
        raise ValueError(
            f"unknown KL direction {direction!r}; expected 'forward' or 'reverse'"
        )

    # extra divergence from Proposition A.7
    def score_divergence(self, mu, Sigma):
        """``D(q ; p)`` score-based divergence, Proposition A.7."""
        return gaussian_score_divergence(mu, Sigma, self.mean, self.cov)

    # sampling-based estimate of the *forward* KL using the target's own
    # log density (useful for validating the closed form)
    def mc_forward_kl(self, mu, Sigma, n: int = 20000, rng=None):
        z = self.sample(n, rng=rng)
        return float(np.mean(self.log_prob(z) - _gaussian_log_prob_generic(z, mu, Sigma)))

    def mc_reverse_kl(self, mu, Sigma, n: int = 20000, rng=None):
        from ..bam.vi_base import GaussianVariational

        q = GaussianVariational(mu, Sigma)
        z = np.asarray(q.sample(n, rng=rng))
        return float(np.mean(q.log_prob(z) - self.log_prob(z)))

    # convenience -----------------------------------------------------
    def numpy(self):
        return GaussianTarget(
            mean=np.asarray(self.mean),
            cov=np.asarray(self.cov),
            A=None if self.A is None else np.asarray(self.A),
            name=self.name,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "dim": self.dim,
            "mean": np.asarray(self.mean),
            "cov": np.asarray(self.cov),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GaussianTarget(name={self.name!r}, dim={self.dim})"


def _gaussian_log_prob_generic(z, mu, Sigma):
    xp = array_namespace(z)
    P = inverse_spd(Sigma)
    d = z - mu
    D = int(np.shape(mu)[-1])
    return -0.5 * _quad_form(d, P) - 0.5 * (
        D * xp.log(2.0 * xp.pi) + log_det_spd(Sigma)
    )


# ---------------------------------------------------------------------------
# closed-form KL divergences between two multivariate Gaussians
# ---------------------------------------------------------------------------
def gaussian_kl(mu1, Sigma1, mu2, Sigma2, jitter: float = 1e-12):
    """``KL(N(mu1, Sigma1) || N(mu2, Sigma2))``.

    Returns ``1/2 [ tr(S2^{-1} S1) + (m2 - m1)^T S2^{-1} (m2 - m1) - D
    + logdet(S2) - logdet(S1) ]``.
    """
    xp = array_namespace(mu1, Sigma1, mu2, Sigma2)
    D = int(np.shape(mu1)[-1])
    Sigma1 = ensure_spd(symmetrize(Sigma1))
    Sigma2 = ensure_spd(symmetrize(Sigma2))
    P2 = inverse_spd(Sigma2, jitter=jitter)
    diff = mu2 - mu1
    tr_term = xp.sum(P2 * Sigma1.T)  # tr(P2 Sigma1)
    quad = _quad_form(diff, P2)
    ld1 = log_det_spd(Sigma1)
    ld2 = log_det_spd(Sigma2)
    return 0.5 * (tr_term + quad - D + ld2 - ld1)


def gaussian_reverse_kl(mu_star, Sigma_star, mu, Sigma, jitter: float = 1e-12):
    """``KL(q ; p) = KL(N(mu, Sigma) || N(mu_*, Sigma_*))``."""
    return gaussian_kl(mu, Sigma, mu_star, Sigma_star, jitter=jitter)


def gaussian_forward_kl(mu_star, Sigma_star, mu, Sigma, jitter: float = 1e-12):
    """``KL(p ; q) = KL(N(mu_*, Sigma_*) || N(mu, Sigma))``."""
    return gaussian_kl(mu_star, Sigma_star, mu, Sigma, jitter=jitter)


def gaussian_score_divergence(mu, Sigma, mu_star, Sigma_star):
    """Score-based divergence of Proposition A.7.

    ``D(q ; p) = tr[(I - Psi Sigma^{-1})^2]
                 + (nu - mu)^T Sigma^{-1} Psi Sigma^{-1} (nu - mu)``

    where here ``p = N(mu_star, Sigma_star)`` and ``q = N(mu, Sigma)``.
    """
    xp = array_namespace(mu, Sigma, mu_star, Sigma_star)
    P_star = inverse_spd(ensure_spd(Sigma_star))
    Sigma = ensure_spd(symmetrize(Sigma))
    M = xp.eye(int(np.shape(mu)[-1]), dtype=Sigma.dtype) - Sigma @ P_star
    tr_term = xp.sum(M * M.T)
    d = mu - mu_star
    quad = d @ P_star @ Sigma @ P_star @ d
    return tr_term + quad


# ---------------------------------------------------------------------------
# random target construction (Appendix E.3)
# ---------------------------------------------------------------------------
def random_gaussian_A(
    dim: int,
    rng=None,
    key=None,
    xp=np,
    dtype=None,
    scale: float = 1.0,
    eps: float = 1.0,
    cond: Optional[float] = None,
):
    """Generate the ``D x D`` matrix ``A`` with ``Sigma_* = A A^T``.

    The paper only states that a random ``D x D`` matrix ``A`` is drawn and the
    covariance formed as ``A A^T``.  The unspecified details are set to a
    documented default:

    * entries are drawn i.i.d. from ``N(0, eps^2)`` scaled by ``scale``;
    * when ``cond`` is given, ``A``'s singular values are rescaled so that the
      resulting covariance has condition number ``cond`` (prioritising a
      well-conditioned, size-independent target).

    Returns
    -------
    A : array of shape ``(D, D)``
    """
    if rng is None and key is None:
        rng = np.random.default_rng(0)
    if key is not None and _HAS_JAX:
        A = jax.random.normal(key, (dim, dim))
        if dtype is not None:
            A = A.astype(dtype)
        xp = jnp
    else:
        if rng is None:
            rng = np.random.default_rng()
        A = xp.asarray(rng.standard_normal((dim, dim)))
        if dtype is not None:
            A = A.astype(dtype)
    A = scale * eps * A
    if cond is not None:
        A = _rescale_singular_values(A, float(cond))
    return A


def _rescale_singular_values(A, cond: float):
    xp = array_namespace(A)
    U, s, Vt = xp.linalg.svd(A)
    lo = float(s[-1])
    if lo <= 0.0:
        lo = 1e-12
    hi = lo * cond
    s_new = lo * (hi / lo) ** xp.linspace(0.0, 1.0, s.shape[0])
    return (U * s_new) @ Vt


def random_gaussian_target(
    dim: int,
    rng=None,
    key=None,
    mean_scale: float = 0.1,
    A: Optional[Any] = None,
    scale: float = 1.0,
    cond: Optional[float] = None,
    name: Optional[str] = None,
    xp=np,
    dtype=None,
) -> GaussianTarget:
    """Construct a random Gaussian target of dimension ``dim``.

    * covariance ``Sigma_* = A A^T`` with random ``D x D`` matrix ``A``
      (Appendix E.3);
    * mean ``mu_* ~ Uniform[0, 0.1]^D``, matching the initialisation
      distribution used for all algorithms in Appendix E.3.

    (The paper does not specify the target mean for the Gaussian experiments,
    so the documented default above is used; ``mean_scale`` can be set to
    ``0.0`` for a target centred at the origin.)
    """
    if rng is None and key is None:
        rng = np.random.default_rng(0)
    if A is None:
        A = random_gaussian_A(
            dim, rng=rng, key=key, xp=xp, dtype=dtype, scale=scale, cond=cond
        )
    cov = ensure_spd(symmetrize(A @ A.T))
    if key is not None and _HAS_JAX:
        mu = mean_scale * jax.random.uniform(key, (dim,))
    else:
        if rng is None:
            rng = np.random.default_rng()
        mu = xp.asarray(mean_scale * rng.uniform(0.0, 1.0, size=dim), dtype=cov.dtype)
    return GaussianTarget(
        mean=mu,
        cov=cov,
        A=A,
        name=name if name is not None else f"gaussian-D{dim}",
    )


# ---------------------------------------------------------------------------
# one-step recovery check for Corollary D.5 (used in tests)
# ---------------------------------------------------------------------------
def one_step_recovery(
    target: GaussianTarget,
    batch_size: int = 4096,
    lam: Optional[float] = None,
    rng=None,
):
    """Run a single large-batch, large-``lambda`` BaM step against ``target``.

    With a large batch and a large ``lambda_0`` the BaM match step recovers the
    exact Gaussian target in one step (Corollary D.5 / the ``B -> inf``,
    ``lambda -> inf`` limit).  Returns the updated ``(mu, Sigma)``.
    """
    from ..bam.bam import bam_match_step

    if rng is None:
        rng = np.random.default_rng(0)
    D = target.dim
    mu0 = np.zeros(D)
    Sigma0 = np.eye(D)
    z = target.sample(batch_size, rng=rng)
    g = np.asarray(target.score(z))
    lam_val = float(lam) if lam is not None else float(batch_size * D)
    mu1, Sigma1 = bam_match_step(mu0, Sigma0, z=z, g=g, lam=lam_val)
    return np.asarray(mu1), np.asarray(Sigma1)
