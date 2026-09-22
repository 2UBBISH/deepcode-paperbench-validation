"""Sinh-arcsinh normal target distribution (Section 5.1 / Appendix E.4).

The sinh-arcsinh normal distribution (Jones & Pewsey, 2009; 2019) transforms a
Gaussian random variable ``y ~ N(mu, Sigma)`` via the hyperbolic sine function,

.. math::

    z = \\sinh\\left(\\frac{1}{\\tau}\\left(\\sinh^{-1}(y) + s\\right)\\right)

where :math:`s \\in \\mathbb{R}` controls the skew and :math:`\\tau > 0` controls
the heaviness of the tails.  The Gaussian distribution is recovered when
``s = 0`` and ``tau = 1``.

The paper uses this family to build non-Gaussian targets of dimension ``D = 10``
with

* ``tau = 1`` and ``s in {0.2, 1.0, 1.8}`` (varying skew), and
* ``s = 0`` and ``tau in {0.1, 0.9, 1.7}`` (varying tail weight).

This module implements

* the forward transform ``T(y)`` and its inverse ``T^{-1}(z)``,
* the log-density of the resulting target via the change-of-variables formula,
* the score ``grad_z log p(z)`` using the inverse-function Jacobian,
* sampling from the target (transform a Gaussian sample),
* a Monte-Carlo forward/reverse KL interface compatible with the metrics layer,
* a convenience constructor :func:`sinh_arcsinh_target` mirroring the paper's
  experimental settings.

Mathematical details
--------------------

Write the elementwise map as :math:`z = f(y)` with

.. math::

    f_i(y) = \\sinh\\left(a\\,(\\sinh^{-1}(y_i) + s_i)\\right), \\qquad a = 1/\\tau .

Its inverse is

.. math::

    f_i^{-1}(z) = \\sinh\\left(\\tau\\,\\sinh^{-1}(z_i) - s_i\\right) .

Because :math:`f` is applied elementwise, the Jacobian is diagonal with entries

.. math::

    f_i'(y) = a \\cdot \\frac{\\cosh\\left(a(\\sinh^{-1} y_i + s_i)\\right)}
                              {\\sqrt{1 + y_i^2}} .

The target density is therefore

.. math::

    p(z) = \\mathcal{N}\\big(f^{-1}(z); \\mu, \\Sigma\\big)\\;
           \\left|\\det \\nabla f^{-1}(z)\\right| ,

and the score follows from the chain rule:

.. math::

    \\nabla_z \\log p(z) =
        \\big[\\nabla f^{-1}(z)\\big]^{T}\\,
        \\nabla_{y} \\log \\mathcal{N}(y; \\mu, \\Sigma)
        \\Big|_{y = f^{-1}(z)}
        + \\nabla_z \\log \\left|\\det \\nabla f^{-1}(z)\\right| .

Since the map is elementwise and :math:`(f^{-1})'(z_i) = 1 / f'(y_i)`, the
Jacobian term simplifies to :math:`\\log|f'(y_i)|` per coordinate; the score of
that term is easiest evaluated in the ``y``-parameterisation and transformed.

For a fixed sample the score is computed by differentiating through everything,
but numerically it is more robust to use the closed form derived below, which is
exact because the Jacobian is diagonal.  Both paths are provided:
:meth:`SinhArcsinhTarget.score` uses the analytic expression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover - numpy-only environment
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

from ..bam.matrix_equations import (
    array_namespace,
    ensure_spd,
    inverse_spd,
    symmetrize,
)
from ..bam.vi_base import standard_normal

__all__ = [
    "SinhArcsinhTarget",
    "sinh_arcsinh_target",
    "sinh_arcsinh_forward",
    "sinh_arcsinh_inverse",
    "sinh_arcsinh_log_det_jacobian",
    "sinh_arcsinh_jacobian",
    "sinh_arcsinh_score",
    "PAPER_SKEW_VALUES",
    "PAPER_TAIL_VALUES",
    "PAPER_DIM",
]

# --------------------------------------------------------------------------- #
# Paper settings
# --------------------------------------------------------------------------- #

#: Dimensionality used throughout the sinh-arcsinh experiments (Appendix E.4).
PAPER_DIM: int = 10

#: Skew values for the ``tau = 1`` sweep (Section 5.1).
PAPER_SKEW_VALUES: Tuple[float, ...] = (0.2, 1.0, 1.8)

#: Tail weights for the ``s = 0`` sweep (Section 5.1).
PAPER_TAIL_VALUES: Tuple[float, ...] = (0.1, 0.9, 1.7)

# Numerical guard for the inverse-sinh / cosh terms.
_EPS = 1e-12
_MAX_ASINH_ARG = 1e150


def _asarray_namespace(x: Any):
    """Return the array namespace for ``x`` (numpy or ``jax.numpy``)."""
    return array_namespace(x)


def _isfinite_all(z: Any) -> Any:
    xp = _asarray_namespace(z)
    return bool(np.all(np.isfinite(np.asarray(z))))


# --------------------------------------------------------------------------- #
# Elementwise sinh-arcsinh map and its inverse
# --------------------------------------------------------------------------- #


def sinh_arcsinh_forward(y: Any, s: Any = 0.0, tau: float = 1.0) -> Any:
    """Forward transform ``z = sinh((sinh^{-1}(y) + s) / tau)``.

    Parameters
    ----------
    y:
        Points in the latent Gaussian space, shape ``(..., D)`` or ``(D,)``.
    s:
        Skew parameter; scalar or array broadcastable against ``y``.
    tau:
        Tail weight; positive scalar (or broadcastable array).

    Returns
    -------
    Array of the same shape as ``y``.
    """
    if tau is None:
        tau = 1.0
    return _sinh((_arcsinh(y) + s) / tau)


def sinh_arcsinh_inverse(z: Any, s: Any = 0.0, tau: float = 1.0) -> Any:
    """Inverse transform ``y = sinh(tau * sinh^{-1}(z) - s)``.

    This is the exact inverse of :func:`sinh_arcsinh_forward`.  The dense
    (row-wise) variant :func:`sinh_arcsinh_inverse_dense` handles the general
    case where ``mu``/``Sigma`` mix coordinates; for the elementwise map here,
    this pointwise inverse is all that is required to evaluate the density.
    """
    if tau is None:
        tau = 1.0
    return _sinh(tau * _arcsinh(z) - s)


def _sinh(x: Any) -> Any:
    xp = _asarray_namespace(x)
    if xp is np:
        return np.sinh(x)
    return xp.sinh(x)


def _cosh(x: Any) -> Any:
    xp = _asarray_namespace(x)
    if xp is np:
        return np.cosh(x)
    return xp.cosh(x)


def _arcsinh(x: Any) -> Any:
    xp = _asarray_namespace(x)
    if xp is np:
        return np.arcsinh(x)
    return xp.arcsinh(x)


def _arctanh(x: Any) -> Any:
    xp = _asarray_namespace(x)
    if xp is np:
        return np.arctanh(x)
    return xp.arctanh(x)


# --------------------------------------------------------------------------- #
# Jacobian of the transform
# --------------------------------------------------------------------------- #


def sinh_arcsinh_jacobian(y: Any, s: Any = 0.0, tau: float = 1.0) -> Any:
    """Diagonal of the forward-map Jacobian evaluated at ``y``.

    ``f_i'(y_i) = (1/tau) * cosh((sinh^{-1}(y_i) + s_i)/tau) / sqrt(1 + y_i^2)``.

    Returns an array of the same shape as ``y`` giving the *diagonal* entries
    (the map is elementwise, so the full Jacobian is diagonal).
    """
    if tau is None:
        tau = 1.0
    arg = (_arcsinh(y) + s) / tau
    denom = _sqrt(1.0 + y * y)
    return _cosh(arg) / (tau * denom)


def sinh_arcsinh_inverse_jacobian(z: Any, s: Any = 0.0, tau: float = 1.0) -> Any:
    """Diagonal of the inverse-map Jacobian evaluated at ``z``.

    Since the forward map is elementwise and monotone,
    ``(f^{-1})'(z_i) = 1 / f'(f^{-1}(z_i))``.
    """
    y = sinh_arcsinh_inverse(z, s=s, tau=tau)
    return 1.0 / sinh_arcsinh_jacobian(y, s=s, tau=tau)


def _sqrt(x: Any) -> Any:
    xp = _asarray_namespace(x)
    if xp is np:
        return np.sqrt(x)
    return xp.sqrt(x)


def sinh_arcsinh_log_det_jacobian(
    z: Any,
    s: Any = 0.0,
    tau: float = 1.0,
) -> Any:
    """``log |det J_{f^{-1}}(z)|`` summed over the coordinates.

    For an elementwise monotone map the log absolute determinant of the inverse
    Jacobian is the sum over coordinates of ``log (f_i^{-1})'(z_i)``.  When
    ``z`` has shape ``(..., D)`` the sum is over the last axis.
    """
    z = np.asarray(z)
    log_abs = _log_abs(sinh_arcsinh_inverse_jacobian(z, s=s, tau=tau))
    return log_abs.sum(axis=-1)


def _log_abs(x: Any) -> Any:
    xp = _asarray_namespace(x)
    if xp is np:
        return np.log(np.abs(x) + _EPS)
    return xp.log(xp.abs(x) + _EPS)


def sinh_arcsinh_log_det_jacobian_forward(
    y: Any,
    s: Any = 0.0,
    tau: float = 1.0,
) -> Any:
    """``log |det J_f(y)|`` summed over coordinates (forward direction)."""
    y = np.asarray(y)
    log_abs = _log_abs(sinh_arcsinh_jacobian(y, s=s, tau=tau))
    return log_abs.sum(axis=-1)


# --------------------------------------------------------------------------- #
# Score in closed form
# --------------------------------------------------------------------------- #


def sinh_arcsinh_score(
    z: Any,
    mu: Any,
    Sigma: Any,
    s: Any = 0.0,
    tau: float = 1.0,
    jitter: float = 1e-12,
) -> Any:
    r"""Exact score ``\nabla_z \log p(z)`` of the sinh-arcsinh target.

    Derivation (elementwise map ``z_i = f_i(y_i)``):

    * ``y = f^{-1}(z)``,
    * ``\log p(z) = \log N(y; \mu, \Sigma) + \sum_i \log (f_i^{-1})'(z_i)``.

    Differentiating with respect to ``z`` and using
    ``d y_i / d z_i = (f_i^{-1})'(z_i) =: inv_i'`` gives

    .. math::

        \partial_{z_i} \log p(z)
          = inv_i' \cdot [\Sigma^{-1}(\mu - y)]_i
            + \partial_{z_i} \log inv_i'(z_i) .

    With :math:`u_i = \tau \,\sinh^{-1}(z_i) - s_i` (so :math:`y_i = \sinh u_i`)
    and :math:`g_i = \cosh(a(\sinh^{-1}(y_i) + s_i))` where :math:`a = 1/\tau`,
    one has

    .. math::

        \log (f_i^{-1})'(z_i) = \log\left(\tau \cosh u_i \sqrt{1 + z_i^2}\right)
                                + \log a - \log g_i

    and since :math:`a(\sinh^{-1}(y_i) + s_i) = \sinh^{-1}(z_i)` when
    :math:`y = f^{-1}(z)`, we get :math:`g_i = \cosh(\sinh^{-1} z_i)
    = \sqrt{1 + z_i^2}`, hence :math:`inv_i' = \tau \cosh u_i \sqrt{1+z_i^2}
    / ( (1/\tau) \sqrt{1+z_i^2}) = \tau^2 \cosh u_i`.  In practice we evaluate
    the derivative of the log-Jacobian numerically-stably as

    .. math::

        \partial_{z_i}\log inv_i'
          = \frac{\tau}{\sqrt{1+z_i^2}}\tanh u_i
            - \frac{z_i}{1 + z_i^2} .
    """
    xp = array_namespace(np.asarray(z), np.asarray(mu), np.asarray(Sigma))
    z = np.asarray(z)
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    s_arr = np.asarray(s, dtype=float)
    tau = float(tau)

    # Inverse transform: y = sinh(tau * asinh(z) - s)
    asinh_z = np.arcsinh(z)
    u = tau * asinh_z - s_arr
    y = np.sinh(u)  # shape (..., D)

    # gradient of the Gaussian part w.r.t. y: Sigma^{-1} (mu - y)
    prec = inverse_spd(ensure_spd(Sigma, jitter=jitter))
    diff = mu - y  # (..., D)
    grad_y = diff @ prec.T  # (..., D) if y is (..., D)

    # dy/dz (elementwise)
    dy_dz = tau * np.cosh(u) / np.sqrt(1.0 + z * z)

    # d/dz log(dy/dz): derivative of log(tau) + log(cosh u) - 0.5*log(1+z^2)
    dlog_jac = tau * np.tanh(u) / np.sqrt(1.0 + z * z) - z / (1.0 + z * z)

    return grad_y * dy_dz + dlog_jac


# --------------------------------------------------------------------------- #
# Target class
# --------------------------------------------------------------------------- #


@dataclass
class SinhArcsinhTarget:
    """Sinh-arcsinh normal target distribution.

    The *base* Gaussian is ``N(mu, Sigma)`` (called ``mu``/``Sigma`` in the
    paper's transform); the transform parameters ``s`` (skew) and ``tau`` (tail)
    are broadcastable against a ``D``-vector.

    Parameters
    ----------
    mean:
        Base-Gaussian mean, shape ``(D,)``.
    cov:
        Base-Gaussian covariance (SPD), shape ``(D, D)``.
    s:
        Skew parameter(s); scalar or shape ``(D,)``.
    tau:
        Tail weight(s); positive scalar or shape ``(D,)``.
    name:
        Optional label used by experiments/plots.
    """

    mean: Any
    cov: Any
    s: Union[float, Sequence[float], Any] = 0.0
    tau: Union[float, Sequence[float], Any] = 1.0
    name: str = "sinh-arcsinh"
    _precision: Optional[Any] = field(default=None, repr=False, compare=False)
    _chol: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        cov = np.asarray(self.cov, dtype=float)
        if mean.ndim == 0:
            mean = mean.reshape(1)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", symmetrize(ensure_spd(cov)))

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        """Dimensionality ``D``."""
        return int(np.asarray(self.mean).shape[0])

    @property
    def base_mean(self) -> Any:
        """Base-Gaussian mean ``mu``."""
        return self.mean

    @property
    def base_cov(self) -> Any:
        """Base-Gaussian covariance ``Sigma``."""
        return self.cov

    def _s_vector(self) -> np.ndarray:
        s = np.asarray(self.s, dtype=float)
        if s.ndim == 0:
            return np.full(self.dim, float(s))
        return np.broadcast_to(s, (self.dim,)).astype(float)

    def _tau_vector(self) -> np.ndarray:
        tau = np.asarray(self.tau, dtype=float)
        if tau.ndim == 0:
            return np.full(self.dim, float(tau))
        return np.broadcast_to(tau, (self.dim,)).astype(float)

    @property
    def precision(self) -> Any:
        """Precision ``Sigma^{-1}`` of the base Gaussian (cached)."""
        if self._precision is None:
            object.__setattr__(self, "_precision", inverse_spd(ensure_spd(self.cov)))
        return self._precision

    @property
    def cholesky(self) -> Any:
        """Lower Cholesky factor of the base covariance (cached)."""
        if self._chol is None:
            cov = ensure_spd(self.cov)
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(self.dim))
            object.__setattr__(self, "_chol", L)
        return self._chol

    @property
    def is_gaussian(self) -> bool:
        """True when the target reduces to a Gaussian (``s=0, tau=1``)."""
        return bool(np.allclose(self._s_vector(), 0.0)) and bool(
            np.allclose(self._tau_vector(), 1.0)
        )

    # ------------------------------------------------------------------ #
    # Transform helpers
    # ------------------------------------------------------------------ #
    def forward(self, y: Any) -> Any:
        """Apply the sinh-arcsinh transform ``y -> z``."""
        return sinh_arcsinh_forward(y, s=self._s_vector(), tau=self._tau_vector())

    def inverse(self, z: Any) -> Any:
        """Apply the inverse transform ``z -> y``."""
        return sinh_arcsinh_inverse(z, s=self._s_vector(), tau=self._tau_vector())

    def log_det_jacobian(self, z: Any) -> Any:
        """``log |det J_{f^{-1}}|(z)`` (summed over coordinates)."""
        return sinh_arcsinh_log_det_jacobian(
            z, s=self._s_vector(), tau=self._tau_vector()
        )

    # ------------------------------------------------------------------ #
    # Density and score
    # ------------------------------------------------------------------ #
    def log_prob(self, z: Any) -> Any:
        """Log density of the target at ``z`` (up to an additive constant).

        Uses the change-of-variables formula with the exact base-Gaussian
        normaliser, so the returned value is a *normalised* log density.
        """
        z = np.asarray(z)
        single = z.ndim == 1
        if single:
            z = z[None, :]

        y = self.inverse(z)
        diff = y - self.mean
        quad = np.einsum("nd,de,ne->n", diff, self.precision, diff)
        log_det_cov = 2.0 * np.sum(np.log(np.diag(self.cholesky)))
        log_base = -0.5 * (self.dim * math.log(2.0 * math.pi) + log_det_cov + quad)
        log_jac = self.log_det_jacobian(z)
        out = log_base + log_jac
        return out[0] if single else out

    def score(self, z: Any) -> Any:
        """Score ``grad_z log p(z)`` (exact, closed form).

        For a single point ``z`` of shape ``(D,)`` returns shape ``(D,)``; for a
        batch of shape ``(B, D)`` returns ``(B, D)``.
        """
        single = np.asarray(z).ndim == 1
        zz = np.asarray(z)[None, :] if single else np.asarray(z)
        out = sinh_arcsinh_score(
            zz,
            self.mean,
            self.cov,
            s=self._s_vector(),
            tau=self._tau_vector(),
        )
        return out[0] if single else out

    grad_log_prob = score

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
        key: Any = None,
        dtype: Any = None,
    ) -> Any:
        """Draw ``n`` samples from the sinh-arcsinh target.

        Samples are produced by transforming Gaussian draws
        ``y ~ N(mu, Sigma)`` with the elementwise sinh-arcsinh map.
        """
        eps = standard_normal((int(n), self.dim), rng=rng, key=key)
        eps = np.asarray(eps)
        y = self.mean[None, :] + eps @ self.cholesky.T
        z = self.forward(y)
        return z.astype(float if dtype is None else dtype)

    def sample_y(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> Any:
        """Draw ``n`` samples in the *latent* Gaussian space."""
        eps = standard_normal((int(n), self.dim), rng=rng)
        eps = np.asarray(eps)
        return self.mean[None, :] + eps @ self.cholesky.T

    def reparameterize(self, z: Any) -> Any:
        """Alias kept for interface compatibility with other targets."""
        return self.inverse(z)

    # ------------------------------------------------------------------ #
    # KL helpers (Monte Carlo; the target is non-Gaussian in general)
    # ------------------------------------------------------------------ #
    def mc_forward_kl(
        self,
        mu: Any,
        Sigma: Any,
        n: int = 4096,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Monte-Carlo estimate of the forward KL ``KL(p ; q)``.

        ``KL(p;q) = E_p[log p(z) - log q(z)]``.  Samples come from the target
        ``p`` and ``log q`` is the full-covariance Gaussian log density.
        """
        from ..bam.vi_base import gaussian_log_density

        def exact_log_q(z: Any) -> Any:
            return gaussian_log_density(np.asarray(z), np.asarray(mu), np.asarray(Sigma))

        return self._mc_kl(exact_log_q, n=n, rng=rng, sample_from="target")

    def mc_reverse_kl(
        self,
        mu: Any,
        Sigma: Any,
        n: int = 4096,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Monte-Carlo estimate of the reverse KL ``KL(q ; p)``."""
        from ..bam.vi_base import gaussian_log_density

        def exact_log_q(z: Any) -> Any:
            return gaussian_log_density(np.asarray(z), np.asarray(mu), np.asarray(Sigma))

        return self._mc_kl(exact_log_q, n=n, rng=rng, sample_from="q")

    def _mc_kl(
        self,
        log_q_fn,
        n: int,
        rng: Optional[np.random.Generator],
        sample_from: str,
    ) -> float:
        if rng is None:
            rng = np.random.default_rng(0)
        if sample_from == "target":
            z = self.sample(n, rng=rng)
        else:
            raise ValueError("_mc_kl requires sample_from='target' for forward KL")
        log_p = np.asarray(self.log_prob(z))
        log_q = np.asarray(log_q_fn(z))
        return float(np.mean(log_p - log_q))

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Plain-python description of the target."""
        return {
            "name": self.name,
            "dim": self.dim,
            "mean": np.asarray(self.mean).tolist(),
            "cov": np.asarray(self.cov).tolist(),
            "s": self._s_vector().tolist(),
            "tau": self._tau_vector().tolist(),
            "is_gaussian": self.is_gaussian,
        }

    def numpy(self) -> "SinhArcsinhTarget":
        """Return a NumPy-backed copy (identity here; JAX handled elsewhere)."""
        return SinhArcsinhTarget(
            mean=np.asarray(self.mean),
            cov=np.asarray(self.cov),
            s=np.asarray(self.s),
            tau=np.asarray(self.tau),
            name=self.name,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SinhArcsinhTarget(dim={self.dim}, s={self._s_vector().tolist()}, "
            f"tau={self._tau_vector().tolist()})"
        )


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #


def random_sinh_arcsinh_target(
    dim: int = PAPER_DIM,
    s: Union[float, Sequence[float]] = 0.0,
    tau: Union[float, Sequence[float]] = 1.0,
    rng: Optional[np.random.Generator] = None,
    key: Any = None,
    mean_scale: float = 0.1,
    mean: Any = None,
    cov: Any = None,
    scale: float = 1.0,
    name: Optional[str] = None,
) -> SinhArcsinhTarget:
    """Build a random sinh-arcsinh target (paper's experimental protocol).

    Follows Appendix E.4: a random initial *variational* mean is used by the
    algorithms, while the target itself is constructed with a base Gaussian
    ``N(mu, Sigma)``.  We default to ``Sigma = scale^2 I`` and
    ``mu ~ Uniform[0, mean_scale]^D`` unless supplied explicitly, mirroring the
    Gaussian-target construction in Section 5.1 / E.3.
    """
    if mean is None:
        if rng is None and key is None:
            rng = np.random.default_rng(0)
        base = standard_normal((1, dim), rng=rng, key=key)
        base = np.asarray(base)
        mean_arr = mean_scale * np.abs(base[0])
    else:
        mean_arr = np.asarray(mean, dtype=float)

    if cov is None:
        cov_arr = float(scale) ** 2 * np.eye(dim)
    else:
        cov_arr = np.asarray(cov, dtype=float)

    label = name if name is not None else f"sinh-arcsinh(s={s},tau={tau})"
    return SinhArcsinhTarget(mean=mean_arr, cov=cov_arr, s=s, tau=tau, name=label)


def sinh_arcsinh_target(
    dim: int = PAPER_DIM,
    s: Union[float, Sequence[float]] = 0.0,
    tau: Union[float, Sequence[float]] = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> SinhArcsinhTarget:
    """Convenience constructor (alias of :func:`random_sinh_arcsinh_target`)."""
    return random_sinh_arcsinh_target(dim=dim, s=s, tau=tau, rng=rng, **kwargs)


def paper_sweep_targets(
    rng: Optional[np.random.Generator] = None,
    dim: int = PAPER_DIM,
    **kwargs: Any,
) -> Dict[str, SinhArcsinhTarget]:
    """Return the six targets used in Section 5.1 / Figure 5.2 / E.4.

    The returned mapping has keys ``"skew{s}"`` for the ``tau = 1`` sweep and
    ``"tail{tau}"`` for the ``s = 0`` sweep.
    """
    targets: Dict[str, SinhArcsinhTarget] = {}
    for s in PAPER_SKEW_VALUES:
        targets[f"skew{s}"] = random_sinh_arcsinh_target(
            dim=dim, s=s, tau=1.0, rng=rng, **kwargs
        )
    for tau in PAPER_TAIL_VALUES:
        targets[f"tail{tau}"] = random_sinh_arcsinh_target(
            dim=dim, s=0.0, tau=tau, rng=rng, **kwargs
        )
    return targets
