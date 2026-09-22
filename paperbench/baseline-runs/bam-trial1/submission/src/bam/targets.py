"""Target distributions for Batch-and-Match variational inference experiments.

This module implements the two families of synthetic target distributions used
in Section 5.1 of the paper:

* Gaussian targets ``p = N(mu, Sigma)`` with a closed-form score function.
* Sinh-arcsinh normal targets obtained by applying the univariate
  sinh-arcsinh transformation to a multivariate Gaussian.  These provide
  controllable skewness and tail-weight and are used to evaluate behaviour on
  non-Gaussian targets.

All log-density and score functions are JAX-traceable, so their scores can be
computed with automatic differentiation and are compatible with the batch
sampling machinery in :mod:`src.bam.algorithm` and :mod:`src.bam.baselines`.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

__all__ = [
    "GaussianTarget",
    "SinhArcsinhTarget",
    "random_gaussian_target",
    "make_initial_params",
    "sample_mvn",
    "symmetrize",
]


def symmetrize(a: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
    """Symmetrize a square matrix and optionally add diagonal jitter.

    Args:
        a: Square matrix.
        jitter: Non-negative diagonal jitter added to improve conditioning.

    Returns:
        Symmetric matrix ``(a + a.T) / 2 + jitter * I``.
    """
    a = jnp.asarray(a)
    out = (a + a.T) / 2.0
    if jitter:
        out = out + jitter * jnp.eye(a.shape[0], dtype=out.dtype)
    return out


def sample_mvn(
    key: jax.Array,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    num_samples: int,
    jitter: float = 1e-8,
) -> jnp.ndarray:
    """Draw samples from a multivariate normal distribution.

    Args:
        key: JAX PRNG key.
        mu: Mean vector of shape ``(D,)``.
        Sigma: Covariance matrix of shape ``(D, D)``.
        num_samples: Number of samples ``B``.
        jitter: Diagonal jitter added to ``Sigma`` before Cholesky.

    Returns:
        Array of shape ``(num_samples, D)``.
    """
    mu = jnp.asarray(mu)
    Sigma = symmetrize(jnp.asarray(Sigma), jitter=jitter)
    d = mu.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError(
            f"Sigma must have shape {(d, d)}, got {tuple(Sigma.shape)}"
        )
    chol = jnp.linalg.cholesky(Sigma)
    eps = jax.random.normal(key, (num_samples, d), dtype=mu.dtype)
    return mu[None, :] + eps @ chol.T


def _gaussian_log_prob_single(
    z: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray
) -> jnp.ndarray:
    """Log density of a multivariate Gaussian at a single point."""
    diff = z - mu
    sol = jnp.linalg.solve(Sigma, diff)
    quad = jnp.dot(diff, sol)
    _, logdet = jnp.linalg.slogdet(Sigma)
    return -0.5 * (mu.shape[0] * jnp.log(2.0 * jnp.pi) + logdet + quad)


def _gaussian_score_single(
    z: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray
) -> jnp.ndarray:
    """Score ``grad_z log N(z; mu, Sigma)`` at a single point."""
    return -jnp.linalg.solve(Sigma, z - mu)


def _sinh_arcsinh_y(z: jnp.ndarray, skew: float, tailweight: float) -> jnp.ndarray:
    """Inverse sinh-arcsinh transform: y = sinh(tail * asinh(z) - skew)."""
    return jnp.sinh(tailweight * jnp.arcsinh(z) - skew)


def _sinh_arcsinh_log_jacobian(
    z: jnp.ndarray, skew: float, tailweight: float
) -> jnp.ndarray:
    """Log absolute Jacobian determinant of the elementwise transform.

    For ``y(z) = sinh(tau * asinh(z) - s)`` we have
    ``dy/dz = tau * cosh(tau * asinh(z) - s) / sqrt(1 + z^2)``.
    """
    return jnp.sum(
        jnp.log(tailweight)
        + jnp.log(jnp.cosh(tailweight * jnp.arcsinh(z) - skew))
        - 0.5 * jnp.log1p(z**2)
    )


def _sinh_arcsinh_log_prob_single(
    z: jnp.ndarray,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    skew: float,
    tailweight: float,
) -> jnp.ndarray:
    """Log density of a sinh-arcsinh normal target at a single point."""
    y = _sinh_arcsinh_y(z, skew, tailweight)
    diff = y - mu
    sol = jnp.linalg.solve(Sigma, diff)
    quad = jnp.dot(diff, sol)
    _, logdet = jnp.linalg.slogdet(Sigma)
    log_py = -0.5 * (mu.shape[0] * jnp.log(2.0 * jnp.pi) + logdet + quad)
    return log_py + _sinh_arcsinh_log_jacobian(z, skew, tailweight)


class GaussianTarget:
    """A multivariate Gaussian target distribution ``p = N(mu, Sigma)``.

    Attributes:
        mu: Mean vector of shape ``(D,)``.
        Sigma: Covariance matrix of shape ``(D, D)``.
        D: Dimensionality.
    """

    def __init__(self, mu: jnp.ndarray, Sigma: jnp.ndarray) -> None:
        self.mu = jnp.asarray(mu)
        self.Sigma = symmetrize(jnp.asarray(Sigma), jitter=1e-12)

        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D vector")
        if self.Sigma.ndim != 2 or self.Sigma.shape[0] != self.Sigma.shape[1]:
            raise ValueError("Sigma must be a square matrix")
        if self.Sigma.shape[0] != self.mu.shape[0]:
            raise ValueError("mu and Sigma dimensions must agree")

        self._log_prob_single = lambda z: _gaussian_log_prob_single(
            z, self.mu, self.Sigma
        )
        self._score_single = lambda z: _gaussian_score_single(
            z, self.mu, self.Sigma
        )
        self._log_prob_batch = jax.jit(jax.vmap(self._log_prob_single))
        self._score_batch = jax.jit(jax.vmap(self._score_single))

    @property
    def D(self) -> int:
        return int(self.mu.shape[0])

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the log density.

        Accepts either a single vector of shape ``(D,)`` or a batch of shape
        ``(B, D)`` and returns a scalar or a vector of shape ``(B,)``.
        """
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self._log_prob_single(z)
        return self._log_prob_batch(z)

    def score(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate ``grad_z log p(z)`` for a vector or batch of vectors."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self._score_single(z)
        return self._score_batch(z)

    def sample(self, key: jax.Array, num_samples: int) -> jnp.ndarray:
        """Draw samples from the target."""
        return sample_mvn(key, self.mu, self.Sigma, num_samples)

    def score_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a vectorized score callable for batch inputs ``(B, D)``."""
        return self._score_batch

    def log_prob_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a vectorized log-density callable for batch inputs."""
        return self._log_prob_batch


class SinhArcsinhTarget:
    """A sinh-arcsinh normal target distribution.

    The base random variable ``y ~ N(mu, Sigma)`` is transformed elementwise by

    .. math::
        z_i = \\sinh\\left(\\frac{\\operatorname{asinh}(y_i) + s}{\\tau}\\right),

    where ``s`` controls skewness and ``tau`` controls tail weight.  The
    inverse mapping is ``y_i = sinh(tau * asinh(z_i) - s)``.
    """

    def __init__(
        self,
        mu: jnp.ndarray,
        Sigma: jnp.ndarray,
        skew: float,
        tailweight: float,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.Sigma = symmetrize(jnp.asarray(Sigma), jitter=1e-12)
        self.skew = float(skew)
        self.tailweight = float(tailweight)

        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D vector")
        if self.Sigma.ndim != 2 or self.Sigma.shape[0] != self.Sigma.shape[1]:
            raise ValueError("Sigma must be a square matrix")
        if self.Sigma.shape[0] != self.mu.shape[0]:
            raise ValueError("mu and Sigma dimensions must agree")
        if self.tailweight <= 0:
            raise ValueError("tailweight must be positive")

        self._log_prob_single = lambda z: _sinh_arcsinh_log_prob_single(
            z, self.mu, self.Sigma, self.skew, self.tailweight
        )
        # Score is obtained with JAX automatic differentiation through the
        # transformed log-density, matching the paper's specification.
        self._score_single = jax.grad(self._log_prob_single)
        self._log_prob_batch = jax.jit(jax.vmap(self._log_prob_single))
        self._score_batch = jax.jit(jax.vmap(self._score_single))

    @property
    def D(self) -> int:
        return int(self.mu.shape[0])

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformed log density for a vector or batch."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self._log_prob_single(z)
        return self._log_prob_batch(z)

    def score(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate ``grad_z log p(z)`` via autodiff for a vector or batch."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self._score_single(z)
        return self._score_batch(z)

    def sample(self, key: jax.Array, num_samples: int) -> jnp.ndarray:
        """Draw samples from the sinh-arcsinh normal target."""
        y = sample_mvn(key, self.mu, self.Sigma, num_samples)
        return jnp.sinh((jnp.arcsinh(y) + self.skew) / self.tailweight)

    def score_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a vectorized score callable for batch inputs ``(B, D)``."""
        return self._score_batch

    def log_prob_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a vectorized log-density callable for batch inputs."""
        return self._log_prob_batch


def random_gaussian_target(
    key: jax.Array,
    D: int,
    *,
    mu: Optional[jnp.ndarray] = None,
    Sigma: Optional[jnp.ndarray] = None,
    scale: float = 1.0,
    jitter: float = 1e-10,
) -> GaussianTarget:
    """Construct a random Gaussian target.

    By default the covariance is ``Sigma = A A^T / D`` where ``A`` is a
    ``D x D`` matrix with standard normal entries.  This matches the paper's
    ``Sigma_* = A A^T`` construction while keeping the marginal variances of
    order one for numerical stability.  The optional ``scale`` factor scales
    the random matrix ``A``.

    Args:
        key: JAX PRNG key.
        D: Dimensionality.
        mu: Optional fixed mean.  If ``None`` a small random mean is drawn.
        Sigma: Optional fixed covariance.  If ``None`` a random PSD matrix is
            generated from ``A A^T``.
        scale: Scale applied to the random factor ``A``.
        jitter: Diagonal jitter added to ensure positive definiteness.

    Returns:
        A :class:`GaussianTarget`.
    """
    key_mu, key_A = jax.random.split(key)
    if mu is None:
        mu = jax.random.normal(key_mu, (D,)) * 0.1
    if Sigma is None:
        A = jax.random.normal(key_A, (D, D)) * (scale / jnp.sqrt(D))
        Sigma = A @ A.T
    Sigma = symmetrize(jnp.asarray(Sigma), jitter=jitter)
    return GaussianTarget(jnp.asarray(mu), Sigma)


def make_initial_params(
    key: jax.Array,
    D: int,
    *,
    mu_low: float = 0.0,
    mu_high: float = 0.1,
    sigma0: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create the paper's Gaussian variational initialization.

    The mean is drawn elementwise uniformly from ``[mu_low, mu_high]`` and the
    covariance is initialized to ``sigma0 * I``.

    Returns:
        Tuple ``(mu0, Sigma0)``.
    """
    mu0 = jax.random.uniform(
        key, (D,), minval=mu_low, maxval=mu_high, dtype=jnp.float32
    )
    Sigma0 = sigma0 * jnp.eye(D, dtype=mu0.dtype)
    return mu0, Sigma0
