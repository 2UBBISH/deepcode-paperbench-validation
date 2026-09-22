"""Shared Gaussian variational state, reparameterized sampling, and Gaussian-KL helpers.

Paper reference (SS2.2): the variational family used throughout the paper is the family
of full-covariance Gaussians

    Q = { N(mu, Sigma) : mu in R^D, Sigma in S^D_{++} },                (1)

i.e. every variational density is parameterised by a mean vector ``mu`` and a
symmetric positive-definite covariance matrix ``Sigma``.  Because the whole paper
(eq. (2) score-based divergence) is built on this family, this module provides the
common representation that ``bam.py`` (Algorithm 1) and every baseline in
``baselines/`` share:

* reparameterized sampling z = mu + L eps, eps ~ N(0, I), L L^T = Sigma,
* closed-form Gaussian log-density / score of the variational density,
* the entropy and the Gaussian KL divergence
  KL(N(mu1,S1) || N(mu2,S2)) used for the forward KL(p; q) and reverse KL(q; p)
  metrics of SS5.1,
* small numerical helpers (empirical mean/covariance of a batch, SPD re-projection)
  that keep Sigma symmetric positive definite across iterations.

The code is written against a namespace-inferred array module so that the very same
state works with ``numpy`` arrays and with ``jax.numpy`` arrays (the paper implements
everything in JAX; JAX is optional here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .matrix_equations import (
    _HAS_JAX,
    array_namespace,
    ensure_spd,
    inverse_spd,
    matrix_sqrt,
    symmetrize,
)

if _HAS_JAX:  # pragma: no cover - depends on the environment
    import jax
    import jax.numpy as jnp
else:  # pragma: no cover
    jax = None
    jnp = None

__all__ = [
    "GaussianVariational",
    "init_gaussian_state",
    "standard_normal",
    "reparameterize",
    "batch_mean",
    "batch_covariance",
    "gaussian_kl",
    "gaussian_entropy",
    "gaussian_log_density",
    "gaussian_score",
    "project_spd",
]


# --------------------------------------------------------------------------------------
# Random number helpers (numpy rng or explicit jax PRNGKey)
# --------------------------------------------------------------------------------------
def _is_jax_namespace(xp: Any) -> bool:
    return _HAS_JAX and jnp is not None and xp is jnp


def standard_normal(
    shape: Tuple[int, ...],
    xp: Any = np,
    rng: Optional[Any] = None,
    key: Optional[Any] = None,
    dtype: Any = None,
) -> Any:
    """Draw standard normal samples with either a numpy ``rng`` or a jax ``key``."""
    if key is not None:
        if not _is_jax_namespace(xp):
            raise ValueError("A jax PRNG ``key`` was given but the arrays are not jax arrays.")
        return jax.random.normal(key, shape, dtype=dtype or jnp.float32)
    if isinstance(rng, np.random.Generator):
        eps = rng.standard_normal(shape)
    elif rng is None:
        eps = np.random.default_rng().standard_normal(shape)
    else:  # a seed or a RandomState
        eps = np.random.RandomState(rng).standard_normal(shape)
    eps = np.asarray(eps, dtype=np.float64)
    if not _is_jax_namespace(xp):
        return eps.astype(dtype) if dtype is not None else eps
    return xp.asarray(eps, dtype=dtype or xp.float32)


def reparameterize(mu: Any, L: Any, eps: Any) -> Any:
    """Reparameterization map z = mu + L eps (used for all gradient estimators)."""
    xp = array_namespace(mu, L, eps)
    mu = xp.reshape(mu, (1, -1))
    return mu + eps @ xp.swapaxes(L, -1, -2)


def _triangular_solve(L: Any, B: Any) -> Any:
    """Solve L X = B for lower-triangular ``L`` (falls back to a general solve)."""
    xp = array_namespace(L, B)
    try:
        return xp.linalg.solve_triangular(L, B, lower=True)
    except Exception:  # pragma: no cover - jax/numpy without solve_triangular
        return xp.linalg.solve(L, B)


def _log_det_from_chol(L: Any) -> Any:
    xp = array_namespace(L)
    diag = xp.diagonal(L, axis1=-2, axis2=-1)
    return 2.0 * xp.sum(xp.log(diag), axis=-1)


# --------------------------------------------------------------------------------------
# Batch statistics (used by the batch step of Algorithm 1 and by every baseline)
# --------------------------------------------------------------------------------------
def batch_mean(z: Any) -> Any:
    """Empirical mean of a batch stored as an (B, D) array."""
    xp = array_namespace(z)
    return xp.mean(z, axis=0)


def batch_covariance(z: Any, center: bool = True) -> Any:
    """Empirical covariance (B, D) -> (D, D); ``center=False`` returns E[z z^T]."""
    xp = array_namespace(z)
    if center:
        z = z - xp.mean(z, axis=0, keepdims=True)
    B = xp.shape(z)[0]
    return symmetrize((xp.swapaxes(z, -1, -2) @ z) / B)


# --------------------------------------------------------------------------------------
# Closed-form Gaussian quantities of a variational / target density
# --------------------------------------------------------------------------------------
def gaussian_entropy(Sigma: Any) -> Any:
    """Entropy of N(mu, Sigma) (the mean is irrelevant)."""
    xp = array_namespace(Sigma)
    D = xp.shape(Sigma)[0]
    L = xp.linalg.cholesky(ensure_spd(Sigma))
    return 0.5 * (D * (1.0 + float(np.log(2.0 * np.pi))) + _log_det_from_chol(L))


def gaussian_log_density(z: Any, mu: Any, Sigma: Any, jitter: float = 0.0) -> Any:
    """log N(z; mu, Sigma) for ``z`` of shape (B, D) (or (D,))."""
    xp = array_namespace(z, mu, Sigma)
    D = xp.shape(mu)[0]
    single = (xp.ndim(z) == 1)
    if single:
        z = xp.reshape(z, (1, -1))
    diff = z - xp.reshape(mu, (1, -1))
    L = xp.linalg.cholesky(ensure_spd(Sigma, jitter=jitter))
    y = _triangular_solve(L, xp.swapaxes(diff, -1, -2))
    quad = xp.sum(y ** 2, axis=0)
    logdet = _log_det_from_chol(L)
    out = -0.5 * (D * float(np.log(2.0 * np.pi)) + logdet + quad)
    return out[0] if single else out


def gaussian_score(z: Any, mu: Any, Sigma: Any, jitter: float = 0.0) -> Any:
    """Score of the variational density: grad_z log N(z; mu, Sigma) = -Sigma^{-1}(z-mu)."""
    xp = array_namespace(z, mu, Sigma)
    single = (xp.ndim(z) == 1)
    if single:
        z = xp.reshape(z, (1, -1))
    Sigma_inv = inverse_spd(Sigma, jitter=jitter)
    diff = xp.swapaxes(z - xp.reshape(mu, (1, -1)), -1, -2)
    out = -xp.swapaxes(Sigma_inv @ diff, -1, -2)
    return out[0] if single else out


def gaussian_kl(mu1: Any, Sigma1: Any, mu2: Any, Sigma2: Any, jitter: float = 0.0) -> Any:
    """KL( N(mu1, Sigma1) || N(mu2, Sigma2) ).

    Used for both metrics of SS5.1:
      * forward KL(p; q)  = gaussian_kl(mu_p, Sigma_p, mu_q, Sigma_q),
      * reverse KL(q; p)  = gaussian_kl(mu_q, Sigma_q, mu_p, Sigma_p).

    KL = 0.5 [ tr(Sigma2^{-1} Sigma1) + (mu2-mu1)^T Sigma2^{-1} (mu2-mu1)
               - D + log det Sigma2 - log det Sigma1 ].
    """
    xp = array_namespace(mu1, Sigma1, mu2, Sigma2)
    D = xp.shape(mu1)[0]
    Sigma2_inv = inverse_spd(ensure_spd(Sigma2, jitter=jitter))
    diff = xp.reshape(mu2 - mu1, (-1, 1))
    quad = (xp.swapaxes(diff, -1, -2) @ (Sigma2_inv @ diff))[0, 0]
    trace = xp.trace(Sigma2_inv @ Sigma1)
    L1 = xp.linalg.cholesky(ensure_spd(Sigma1, jitter=jitter))
    L2 = xp.linalg.cholesky(ensure_spd(Sigma2, jitter=jitter))
    logdet1 = _log_det_from_chol(L1)
    logdet2 = _log_det_from_chol(L2)
    return 0.5 * (trace + quad - D + logdet2 - logdet1)


def project_spd(Sigma: Any, jitter: float = 1e-10, min_eig: float = 1e-8) -> Any:
    """Symmetrize and, if needed, shift a covariance matrix back into S^D_{++}."""
    return ensure_spd(Sigma, jitter=jitter, min_eig=min_eig)


# --------------------------------------------------------------------------------------
# Variational state
# --------------------------------------------------------------------------------------
@dataclass
class GaussianVariational:
    """Full-covariance Gaussian variational distribution N(mu, Sigma), see eq. (1).

    Attributes
    ----------
    mu : (D,) array
    Sigma : (D, D) array, symmetric positive definite.

    The state is intentionally dumb (two arrays) so that the algorithms can treat it as
    a pytree / immutable value: updates return a fresh state via :meth:`replace`.
    """

    mu: Any
    Sigma: Any

    # -- basic accessors ---------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(np.shape(self.mu)[0])

    @property
    def xp(self) -> Any:
        return array_namespace(self.mu, self.Sigma)

    def mean(self) -> Any:
        """Variational mean."""
        return self.mu

    def covariance(self) -> Any:
        """Variational covariance."""
        return self.Sigma

    def cholesky(self, jitter: float = 1e-8) -> Any:
        """Lower Cholesky factor L with L L^T = Sigma (reparameterization matrix)."""
        xp = self.xp
        return xp.linalg.cholesky(ensure_spd(self.Sigma, jitter=jitter))

    def covariance_sqrt(self) -> Any:
        """Symmetric square root Sigma^{1/2}."""
        return matrix_sqrt(self.Sigma)

    def covariance_inverse(self, jitter: float = 1e-10) -> Any:
        """Sigma^{-1} (needed for the variational score)."""
        return inverse_spd(self.Sigma, jitter=jitter)

    def precision(self, jitter: float = 1e-10) -> Any:
        """Alias for :meth:`covariance_inverse`."""
        return self.covariance_inverse(jitter=jitter)

    # -- sampling ---------------------------------------------------------------
    def sample(
        self,
        n: int,
        rng: Optional[Any] = None,
        key: Optional[Any] = None,
        eps: Optional[Any] = None,
        jitter: float = 1e-8,
    ) -> Tuple[Any, Any]:
        """Reparameterized batch of ``n`` samples from N(mu, Sigma).

        Returns
        -------
        z : (n, D) samples, ``z = mu + L eps``
        eps : the standard normal draws, reused for pathwise gradients.
        """
        xp = self.xp
        if eps is None:
            eps = standard_normal((int(n), self.dim), xp=xp, rng=rng, key=key)
        L = self.cholesky(jitter=jitter)
        z = reparameterize(self.mu, L, eps)
        return z, eps

    def sample_single(
        self, rng: Optional[Any] = None, key: Optional[Any] = None, eps: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """A single draw (used by GSM, which updates from one sample at a time)."""
        z, eps = self.sample(1, rng=rng, key=key, eps=None if eps is None else eps[None, :])
        return z[0], eps[0]

    # -- density / score --------------------------------------------------------
    def log_prob(self, z: Any, jitter: float = 1e-8) -> Any:
        """log q(z) for the Gaussian variational density."""
        return gaussian_log_density(z, self.mu, self.Sigma, jitter=jitter)

    def score(self, z: Any, jitter: float = 1e-8) -> Any:
        """grad_z log q(z) = -Sigma^{-1}(z - mu)."""
        return gaussian_score(z, self.mu, self.Sigma, jitter=jitter)

    def entropy(self) -> Any:
        """Differential entropy of q."""
        return gaussian_entropy(self.Sigma)

    def kl_from(self, mu_p: Any, Sigma_p: Any) -> Any:
        """KL(q || p) when p = N(mu_p, Sigma_p): the reverse KL metric of SS5.1."""
        return gaussian_kl(self.mu, self.Sigma, mu_p, Sigma_p)

    def kl_to(self, mu_p: Any, Sigma_p: Any) -> Any:
        """KL(p || q) when p = N(mu_p, Sigma_p): the forward KL metric of SS5.1."""
        return gaussian_kl(mu_p, Sigma_p, self.mu, self.Sigma)

    # -- updates ----------------------------------------------------------------
    def replace(self, mu: Any = None, Sigma: Any = None, jitter: float = 0.0) -> "GaussianVariational":
        """Return a new state with (optionally) a new mean and/or covariance.

        Any new covariance is symmetrized and (if needed) shifted into the PD cone, which
        is the contract every covariance update in ``bam.py``/``baselines/`` must respect.
        """
        xp = self.xp
        new_mu = self.mu if mu is None else xp.reshape(mu, self.mu.shape)
        new_Sigma = self.Sigma if Sigma is None else project_spd(Sigma, jitter=max(jitter, 1e-12))
        return GaussianVariational(mu=new_mu, Sigma=new_Sigma)

    def numpy(self) -> "GaussianVariational":
        """Copy of the state as numpy arrays (for metrics / plotting / bookkeeping)."""
        return GaussianVariational(mu=np.asarray(self.mu), Sigma=np.asarray(self.Sigma))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GaussianVariational(D={self.dim})"


# --------------------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------------------
def init_gaussian_state(
    dim: int,
    mean: Optional[Any] = None,
    cov: Optional[Any] = None,
    mu_scale: float = 0.0,
    rng: Optional[Any] = None,
    xp: Any = np,
    dtype: Any = None,
) -> GaussianVariational:
    """Build the initial variational state used by the experiments.

    Defaults follow the reproduction protocol: mu_0 ~ Uniform[0, mu_scale]^D and
    Sigma_0 = I (mu_scale = 0.1 for the Gaussian-target experiments, 0 elsewhere),
    matching "init mu_0 ~ Uniform[0,0.1], Sigma_0 = I".
    """
    if dtype is None:
        dtype = xp.float32 if _is_jax_namespace(xp) else np.float64
    if mean is None:
        if rng is None:
            rng = np.random.default_rng()
        if isinstance(rng, np.random.Generator):
            u = rng.uniform(0.0, 1.0, size=(dim,))
        else:
            u = np.random.RandomState(rng).uniform(0.0, 1.0, size=(dim,))
        mean = float(mu_scale) * u
    if cov is None:
        cov = np.eye(dim)
    mean = xp.asarray(mean, dtype=dtype)
    cov = xp.asarray(cov, dtype=dtype)
    if xp.shape(cov) == ():
        cov = mean * 0.0 + cov * np.eye(dim)
    return GaussianVariational(mu=mean, Sigma=symmetrize(cov))
