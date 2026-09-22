"""Target distributions p used in the experiments of the paper.

Every target in the paper is used as a *black box*: the algorithms only need

* a score function ``s(z) = grad_z log p(z)`` (for BaM, GSM, ADVI, Score and
  Fisher), and
* optionally the log density ``log p(z)`` (for the ELBO of ADVI).

Targets that are used for *evaluation* additionally provide

* exact sampling (so that forward/reverse KL divergences can be estimated by
  Monte Carlo), and
* the (normalized) log density ``log p(z)``.

All scores are obtained by automatic differentiation of the log density with
JAX, mirroring the "black-box" setting of the paper.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .linalg import symmetrize

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _as_batch(Z) -> jnp.ndarray:
    Z = jnp.asarray(Z, dtype=jnp.float64)
    if Z.ndim == 1:
        return Z[None, :]
    return Z


def gaussian_logpdf(Z: jnp.ndarray, mu: jnp.ndarray, chol: jnp.ndarray) -> jnp.ndarray:
    """``log N(z; mu, Sigma)`` for ``Sigma = chol chol^T`` (chol lower triangular)."""
    D = mu.shape[-1]
    diff = Z - mu
    sol = jnp.linalg.solve(chol, diff.T).T
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5 * (D * jnp.log(2.0 * jnp.pi) + logdet + jnp.sum(sol**2, axis=-1))


def sample_gaussian(key, mu: np.ndarray, chol: np.ndarray, n: int) -> np.ndarray:
    eps = np.asarray(jax.random.normal(key, (n, mu.shape[0]), dtype=jnp.float64))
    return mu[None, :] + eps @ chol.T


def gaussian_cholesky(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky factor of ``Sigma``, with an eigendecomposition fallback.

    Variational covariances produced by the experiments are positive definite,
    but on ill-conditioned targets they can be PD only up to round-off; in that
    case we fall back to an eigen-based factor so that sampling never fails.
    """
    Sigma = symmetrize(np.asarray(Sigma, dtype=np.float64))
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(Sigma)
        floor = max(1e-12, 1e-12 * max(float(w.max()), 0.0))
        return (Q * np.sqrt(np.maximum(w, floor))) 


def _logcosh(x):
    """Numerically stable ``log cosh(x)``."""
    return jnp.abs(x) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(x))) - jnp.log(2.0)


class Target:
    """Base class for the (black-box) target distributions."""

    name: str = "target"
    dim: int = 0
    #: whether ``log_density`` is the *normalized* log density of p
    is_normalized: bool = True
    #: whether exact samples from p are available
    is_samplable: bool = False

    # -- black-box interface -------------------------------------------------
    def log_density(self, Z) -> np.ndarray:
        raise NotImplementedError

    def score(self, Z) -> np.ndarray:
        raise NotImplementedError

    # pure-JAX batched versions (used to build differentiable objectives such
    # as the ELBO, and for the score-based / Fisher losses of the baselines)
    def log_density_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def score_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    # -- optional ------------------------------------------------------------
    def sample(self, key, n: int) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not provide exact samples")


class JaxTarget(Target):
    """Target defined through a (possibly unnormalized) JAX log density."""

    def __init__(self, dim: int, log_density_fn: Callable[[jnp.ndarray], jnp.ndarray], name: str = "jax_target",
                 is_normalized: bool = True, feasible_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None):
        self.dim = int(dim)
        self.name = name
        self.is_normalized = is_normalized
        self._log_density = log_density_fn
        self._feasible = feasible_fn
        self._grad = jax.grad(log_density_fn)
        self._score_single = jax.jit(self._grad)
        self._score_batch = jax.jit(jax.vmap(self._grad))
        self._logd_batch = jax.jit(jax.vmap(log_density_fn))
        if feasible_fn is not None:
            self._feasible_single = jax.jit(feasible_fn)
            self._feasible_batch = jax.jit(jax.vmap(feasible_fn))

    def log_density(self, Z) -> np.ndarray:
        Z = _as_batch(Z)
        return np.asarray(self._logd_batch(Z))

    def log_density_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._logd_batch(_as_batch(Z))

    def score_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._score_batch(_as_batch(Z))

    def score(self, Z) -> np.ndarray:
        Z = _as_batch(Z)
        if self._feasible is None:
            return np.asarray(self._score_batch(Z))
        # the log density is -inf outside the support; the score is defined as
        # the gradient of the smooth part, zeroed out where the target has no
        # support (this never happens for the posteriors considered here).
        ok = np.asarray(self._feasible_batch(Z))[:, None]
        return np.where(ok, np.asarray(self._score_batch(Z)), 0.0)


# ----------------------------------------------------------------------------
# target 1: multivariate Gaussian
# ----------------------------------------------------------------------------


class GaussianTarget(Target):
    """``p = N(mu_*, Sigma_*)`` -- Section 5.1 "Gaussian targets"."""

    is_normalized = True
    is_samplable = True

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, name: str = "gaussian"):
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        self.Sigma = symmetrize(np.asarray(Sigma, dtype=np.float64))
        self.chol = np.linalg.cholesky(self.Sigma)
        self.prec = np.linalg.inv(self.Sigma)
        self.mu_jnp = jnp.asarray(self.mu)
        self.chol_jnp = jnp.asarray(self.chol)
        self.prec_jnp = jnp.asarray(self.prec)
        self.dim = self.mu.shape[0]
        self.name = name

    def sample(self, key, n: int) -> np.ndarray:
        return sample_gaussian(key, self.mu, self.chol, n)

    def log_density(self, Z) -> np.ndarray:
        Z = np.atleast_2d(Z)
        D = self.dim
        diff = Z - self.mu
        quad = np.einsum("bi,ij,bj->b", diff, self.prec, diff)
        logdet = 2.0 * np.sum(np.log(np.diag(self.chol)))
        return -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)

    def log_density_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        Z = jnp.atleast_2d(Z)
        D = self.dim
        diff = Z - self.mu_jnp
        quad = jnp.einsum("bi,ij,bj->b", diff, self.prec_jnp, diff)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(self.chol_jnp)))
        return -0.5 * (D * jnp.log(2.0 * jnp.pi) + logdet + quad)

    def score_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        Z = jnp.atleast_2d(Z)
        return (self.mu_jnp[None, :] - Z) @ self.prec_jnp.T

    def score(self, Z) -> np.ndarray:
        Z = np.atleast_2d(Z)
        return (self.mu[None, :] - Z) @ self.prec.T


def random_gaussian_target(dim: int, seed: int = 0, scale: float = 1.0 / np.sqrt(2.0)) -> GaussianTarget:
    """Random Gaussian target with covariance ``Sigma_* = A A^T`` (Appendix E.3)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) * scale
    Sigma = symmetrize(A @ A.T) + 1e-6 * np.eye(dim)
    mu = np.zeros(dim)
    return GaussianTarget(mu, Sigma, name=f"gaussian-D{dim}-seed{seed}")


# ----------------------------------------------------------------------------
# target 2: sinh-arcsinh normal (non-Gaussian targets)
# ----------------------------------------------------------------------------


class SinhArcsinhTarget(Target):
    """Sinh-arcsinh normal distribution (Jones & Pewsey, 2009; 2019).

    ``z = sinh( (asinh(y) + s) / tau )`` for ``y ~ N(mu, Sigma)``, where ``s``
    controls the skew and ``tau`` the heaviness of the tails.  The Gaussian is
    recovered with ``s = 0`` and ``tau = 1``.

    The density follows from the change of variables

        p(z) = N(y(z); mu, Sigma) prod_d tau_d cosh(tau_d asinh(z_d) - s_d) / sqrt(1 + z_d^2),
        y(z) = sinh(tau asinh(z) - s),

    which is normalized on R^D.
    """

    is_normalized = True
    is_samplable = True

    def __init__(self, base_mu: np.ndarray, base_Sigma: np.ndarray, skew: float, tail: float,
                 name: Optional[str] = None):
        self.base_mu = np.asarray(base_mu, dtype=np.float64).ravel()
        self.base_Sigma = symmetrize(np.asarray(base_Sigma, dtype=np.float64))
        self.chol = np.linalg.cholesky(self.base_Sigma)
        self.base_prec = np.linalg.inv(self.base_Sigma)
        D = self.base_mu.shape[0]
        self.skew = np.full(D, float(skew))
        self.tail = np.full(D, float(tail))
        self.dim = D
        self.name = name or f"shash-s{skew}-tau{tail}-D{D}"

        s = jnp.asarray(self.skew)
        tau = jnp.asarray(self.tail)
        mu = jnp.asarray(self.base_mu)

        def log_density(z):
            u = tau * jnp.arcsinh(z) - s
            y = jnp.sinh(u)
            diff = y - mu
            log_base = self._log_base_const - 0.5 * (diff @ self.base_prec @ diff)
            log_jac = jnp.sum(jnp.log(tau) + _logcosh(u) - 0.5 * jnp.log1p(z**2))
            return log_base + log_jac

        log_base_const = -0.5 * (D * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(self.chol))))
        self._log_base_const = float(log_base_const)
        self._log_density_jax = jax.jit(log_density)
        self._log_density_batch = jax.jit(jax.vmap(log_density))
        self._score_batch = jax.jit(jax.vmap(jax.grad(log_density)))

    def sample(self, key, n: int) -> np.ndarray:
        Y = sample_gaussian(key, self.base_mu, self.chol, n)
        return np.sinh((np.arcsinh(Y) + self.skew[None, :]) / self.tail[None, :])

    def log_density(self, Z) -> np.ndarray:
        return np.asarray(self._log_density_batch(_as_batch(Z)))

    def log_density_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._log_density_batch(_as_batch(Z))

    def score_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._score_batch(_as_batch(Z))

    def score(self, Z) -> np.ndarray:
        return np.asarray(self._score_batch(_as_batch(Z)))


# ----------------------------------------------------------------------------
# target 3: posterior inference in Bayesian models (Section 5.2)
# ----------------------------------------------------------------------------


class PosteriorTarget(Target):
    """Unnormalized posterior of a Bayesian model, plus reference summaries.

    ``log_joint`` is the (unnormalized) log posterior on the *unconstrained*
    parameter space; ``feasible`` encodes the support constraints of the model
    (e.g. ``sigma > 0``).  Reference posterior means and standard deviations,
    obtained from Hamiltonian Monte Carlo samples (posteriordb), are used to
    compute the relative error metrics of Section 5.2.

    Stan (and hence BridgeStan, which the paper uses for the gradients of the
    posterior log densities) samples the *unconstrained* parameter space: a
    parameter declared ``real<lower=0> sigma`` becomes ``sigma = exp(u)`` and
    the log density picks up the log-Jacobian ``u`` of the transform.  Both
    parameterizations are supported here:

    * ``parameterization='unconstrained'`` (default) -- the target is a density
      on ``R^D`` with the log-Jacobian adjustments; this is the numerically
      robust choice and the one used for the Section 5.2 experiments.
    * ``parameterization='constrained'`` -- the density of the model parameters
      themselves (zero outside the support), with the score zeroed out on the
      infeasible set.

    ``positive_dims`` lists the coordinates that are mapped through ``exp``, so
    that variational summaries can be mapped back to the constrained space
    before they are compared with the HMC reference summaries.

    In the constrained parameterization the density is only defined on the
    support of the model (e.g. ``sigma > 0``), while the variational family is a
    Gaussian on all of ``R^D``.  The algorithms therefore evaluate a *smooth
    extension* of the density outside the support: the same closed-form
    expression, with ``log |x|`` in place of ``log x`` where a positive
    parameter appears, so that the log density and its gradient stay finite
    everywhere.  This keeps the score-based updates well defined for samples
    that cross the boundary (the density and its score are unchanged on the
    support).  Set ``mask_infeasible=True`` for the strict version (``-inf``
    outside the support, zero score).
    """

    is_normalized = False
    is_samplable = False

    def __init__(self, dim: int, log_joint: Callable[[jnp.ndarray], jnp.ndarray], name: str,
                 feasible: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 ref_mean: Optional[np.ndarray] = None, ref_sd: Optional[np.ndarray] = None,
                 param_names: Optional[list] = None, positive_dims: Sequence[int] = (),
                 parameterization: str = "unconstrained",
                 constrained_log_joint: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 mask_infeasible: bool = False):
        self.dim = int(dim)
        self.name = name
        self.param_names = param_names
        self.positive_dims = tuple(int(i) for i in positive_dims)
        if parameterization not in ("unconstrained", "constrained"):
            raise ValueError(parameterization)
        self.parameterization = parameterization
        self._constrained_log_joint = constrained_log_joint if constrained_log_joint is not None else log_joint
        if parameterization == "unconstrained":
            # log p(u) = log p(T(u)) + sum_{d in positive_dims} u_d,
            # where T maps the positive coordinates through exp
            pos = jnp.asarray(np.asarray(self.positive_dims, dtype=int))

            def log_joint_unc(u):
                w = u if len(self.positive_dims) == 0 else u.at[pos].set(jnp.exp(u[pos]))
                lp = log_joint(w)
                if len(self.positive_dims):
                    lp = lp + jnp.sum(u[pos])
                return lp

            self.log_joint = log_joint_unc
            self.feasible = None
        else:
            self.log_joint = log_joint
            self.feasible = feasible if mask_infeasible else None
        self.ref_mean = None if ref_mean is None else np.asarray(ref_mean, dtype=np.float64)
        self.ref_sd = None if ref_sd is None else np.asarray(ref_sd, dtype=np.float64)
        self._score_batch = jax.jit(jax.vmap(jax.grad(self.log_joint)))
        self._logd_batch = jax.jit(jax.vmap(self.log_joint))
        if self.feasible is not None:
            self._feas_batch = jax.jit(jax.vmap(self.feasible))

    # -- mapping between the target space and the constrained (model) space ----
    def to_constrained(self, Z: np.ndarray) -> np.ndarray:
        """Map points of the target space to the model's (constrained) parameters."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        if not self.positive_dims or self.parameterization == "constrained":
            return Z
        out = Z.copy()
        out[:, list(self.positive_dims)] = np.exp(out[:, list(self.positive_dims)])
        return out

    def variational_summaries(self, mu: np.ndarray, Sigma: np.ndarray):
        """Mean and SD of the *model parameters* under ``q = N(mu, Sigma)``.

        For coordinates mapped through ``exp`` the summaries follow from the
        lognormal moments of the corresponding marginal Gaussian.
        """
        mu = np.asarray(mu, dtype=np.float64).ravel()
        var = np.diag(np.asarray(Sigma, dtype=np.float64))
        mean = mu.copy()
        sd = np.sqrt(var)
        if self.parameterization == "unconstrained":
            for d in self.positive_dims:
                mean[d] = np.exp(mu[d] + 0.5 * var[d])
                sd[d] = np.sqrt((np.exp(var[d]) - 1.0) * np.exp(2.0 * mu[d] + var[d]))
        return mean, sd

    def log_density(self, Z) -> np.ndarray:
        Z = _as_batch(Z)
        vals = np.asarray(self._logd_batch(Z))
        if self.feasible is not None:
            vals = np.where(np.asarray(self._feas_batch(Z)), vals, -np.inf)
        return vals

    def log_density_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._logd_batch(_as_batch(Z))

    def score_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._score_batch(_as_batch(Z))

    def score(self, Z) -> np.ndarray:
        Z = _as_batch(Z)
        grads = np.asarray(self._score_batch(Z))
        if self.feasible is not None:
            grads = np.where(np.asarray(self._feas_batch(Z))[:, None], grads, 0.0)
        return grads
