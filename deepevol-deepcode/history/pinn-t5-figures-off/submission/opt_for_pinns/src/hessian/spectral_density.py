"""Hessian spectral density estimation via Stochastic Lanczos Quadrature (SLQ).

This module implements the PyHessian-style spectral density estimator used to
produce Figures 3 and 7 of the paper ("Challenges in Training PINNs: A Loss
Landscape Perspective").

The spectral density of a symmetric matrix ``H`` (here the PINN Hessian,
accessed only through matrix-vector products) is approximated by

    phi(t) = (1/m) * sum_{j=1}^{m} sum_{i=1}^{k} tau_{j,i}^2 * delta(t - lambda_{j,i})

where for each of ``m`` random Rademacher probe vectors ``v_j`` we run ``k``
steps of the Lanczos algorithm to obtain the tridiagonal matrix ``T_j`` with
eigenpairs ``(lambda_{j,i}, y_{j,i})`` and ``tau_{j,i} = e_1^T y_{j,i}`` is the
first component of the i-th eigenvector.  The resulting distribution is then
smoothed with a Gaussian kernel to obtain a continuous density.

References
----------
- Yao et al. (2020), "PyHessian: Neural Networks Through the Lens of the
  Hessian" -- the SLQ estimator used here follows that implementation.
- Golub & Meurant (2009), "Matrices, Moments and Quadrature with Applications".
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch


__all__ = [
    "lanczos_tridiag",
    "slq_spectral_density",
    "spectral_density",
    "SpectralDensityEstimator",
    "gaussian_smoothing",
]


# ---------------------------------------------------------------------------
# Lanczos tridiagonalization
# ---------------------------------------------------------------------------
def lanczos_tridiag(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iter: int = 100,
    v0: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``num_iter`` steps of the Lanczos algorithm.

    Parameters
    ----------
    matvec:
        Callable applying the symmetric operator ``H`` to a vector of shape
        ``(dim,)``.
    dim:
        Dimension ``p`` of the operator.
    num_iter:
        Number of Lanczos iterations ``k`` (truncated at ``dim``).
    v0:
        Optional starting vector (will be normalized).  If ``None`` a random
        Rademacher vector is used.
    dtype, device:
        Working dtype/device for the Lanczos recurrence.

    Returns
    -------
    alpha, beta:
        Diagonal ``alpha`` of shape ``(k,)`` and off-diagonal ``beta`` of shape
        ``(k-1,)`` of the symmetric tridiagonal matrix ``T``.
    """
    if device is None:
        device = torch.device("cpu")
    k = min(int(num_iter), int(dim))

    if v0 is None:
        v0 = torch.randint(0, 2, (dim,), dtype=dtype, device=device) * 2.0 - 1.0
    else:
        v0 = v0.to(dtype=dtype, device=device).reshape(-1)

    v_prev = torch.zeros(dim, dtype=dtype, device=device)
    v = v0 / (v0.norm() + 1e-30)

    alpha = torch.zeros(k, dtype=dtype, device=device)
    beta = torch.zeros(max(k - 1, 0), dtype=dtype, device=device)

    for j in range(k):
        w = matvec(v.to(dtype=torch.float32) if dtype == torch.float32 else v)
        w = w.to(dtype=dtype, device=device).reshape(-1)
        a = torch.dot(w, v)
        alpha[j] = a
        w = w - a * v - (beta[j - 1] * v_prev if j > 0 else 0.0)
        # Full re-orthogonalization for numerical stability.
        if j > 0:
            # Re-orthogonalize against all previous Lanczos vectors is expensive;
            # a single re-orthogonalization pass against v and v_prev suffices
            # for the moderate iteration counts used here.
            w = w - torch.dot(w, v) * v
        b = w.norm()
        if j < k - 1:
            beta[j] = b
            if b < 1e-14:
                # Invariant subspace found: truncate.
                alpha = alpha[: j + 1]
                beta = beta[:j]
                break
            v_prev = v
            v = w / b

    return alpha, beta


def _tridiag_eig(alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigen-decompose the symmetric tridiagonal matrix ``T``.

    Returns eigenvalues (ascending) and the first components ``tau`` of the
    corresponding eigenvectors.
    """
    k = alpha.shape[0]
    T = torch.zeros(k, k, dtype=alpha.dtype, device=alpha.device)
    T.diagonal().copy_(alpha)
    if k > 1:
        idx = torch.arange(k - 1, device=alpha.device)
        T[idx, idx + 1] = beta
        T[idx + 1, idx] = beta
    eigvals, eigvecs = torch.linalg.eigh(T)
    tau = eigvecs[0, :]  # first row of eigenvectors
    return eigvals, tau


# ---------------------------------------------------------------------------
# SLQ spectral density
# ---------------------------------------------------------------------------
def slq_spectral_density(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_matvecs: int = 100,
    num_repeats: int = 1,
    num_lanczos: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate the spectral density of ``H`` via SLQ.

    Parameters
    ----------
    matvec:
        Matrix-vector product closure for the symmetric operator.
    dim:
        Dimension ``p`` of the operator.
    num_matvecs:
        Total number of matrix-vector products to spend (PyHessian convention).
    num_repeats:
        Number of independent Lanczos runs ``m``.  ``num_lanczos`` steps are
        performed per run so that ``num_repeats * num_lanczos == num_matvecs``.
    num_lanczos:
        Lanczos steps per run.  If ``None`` it is derived from ``num_matvecs``
        and ``num_repeats``.
    dtype, device:
        Working dtype/device.
    generator:
        Optional torch generator for reproducible probe vectors.

    Returns
    -------
    eigenvalues, weights:
        Concatenated Ritz values and their associated quadrature weights
        (``tau^2 / m``), suitable for building a histogram / KDE.
    """
    if device is None:
        device = torch.device("cpu")
    if num_lanczos is None:
        num_lanczos = max(1, int(num_matvecs) // max(1, int(num_repeats)))
    num_lanczos = min(int(num_lanczos), int(dim))

    all_eigs: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []

    for _ in range(int(num_repeats)):
        if generator is not None:
            v0 = (
                torch.randint(
                    0, 2, (dim,), dtype=dtype, device=device, generator=generator
                )
                * 2.0
                - 1.0
            )
        else:
            v0 = torch.randint(0, 2, (dim,), dtype=dtype, device=device) * 2.0 - 1.0

        alpha, beta = lanczos_tridiag(
            matvec, dim, num_iter=num_lanczos, v0=v0, dtype=dtype, device=device
        )
        eigvals, tau = _tridiag_eig(alpha, beta)
        all_eigs.append(eigvals)
        all_weights.append(tau.pow(2) / float(num_repeats))

    eigenvalues = torch.cat(all_eigs)
    weights = torch.cat(all_weights)
    return eigenvalues, weights


def gaussian_smoothing(
    eigenvalues: torch.Tensor,
    weights: torch.Tensor,
    grid: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Smooth a discrete spectral measure onto ``grid`` with a Gaussian kernel."""
    if sigma <= 0:
        sigma = 1e-8
    diff = grid.reshape(-1, 1) - eigenvalues.reshape(1, -1)
    kernel = torch.exp(-0.5 * (diff / sigma) ** 2) / (sigma * (2.0 * torch.pi) ** 0.5)
    density = kernel @ weights.reshape(-1, 1)
    return density.reshape(-1)


def spectral_density(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_matvecs: int = 100,
    num_repeats: int = 1,
    num_lanczos: Optional[int] = None,
    num_bins: int = 200,
    sigma: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Compute a smoothed spectral density estimate.

    Returns a dict with keys ``eigenvalues``, ``weights``, ``grid`` and
    ``density``.  The ``grid`` spans ``[lambda_min, lambda_max]`` of the Ritz
    values with a small padding.
    """
    eigenvalues, weights = slq_spectral_density(
        matvec,
        dim,
        num_matvecs=num_matvecs,
        num_repeats=num_repeats,
        num_lanczos=num_lanczos,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    lam_min = float(eigenvalues.min())
    lam_max = float(eigenvalues.max())
    if lam_max <= lam_min:
        lam_max = lam_min + 1.0
    pad = 0.05 * (lam_max - lam_min)
    grid = torch.linspace(lam_min - pad, lam_max + pad, int(num_bins), dtype=dtype)
    if sigma is None:
        sigma = (lam_max - lam_min) / max(int(num_bins), 1)
    density = gaussian_smoothing(eigenvalues, weights, grid, float(sigma))
    return {
        "eigenvalues": eigenvalues,
        "weights": weights,
        "grid": grid,
        "density": density,
    }


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------
class SpectralDensityEstimator:
    """Convenience wrapper around :func:`spectral_density`.

    Parameters
    ----------
    num_matvecs:
        Total matrix-vector budget (PyHessian default ~100).
    num_repeats:
        Number of independent Lanczos runs.
    num_lanczos:
        Lanczos steps per run (derived from the budget if ``None``).
    num_bins:
        Number of grid points for the smoothed density.
    sigma:
        Gaussian smoothing bandwidth (auto if ``None``).
    dtype, device:
        Working dtype/device.
    """

    def __init__(
        self,
        num_matvecs: int = 100,
        num_repeats: int = 1,
        num_lanczos: Optional[int] = None,
        num_bins: int = 200,
        sigma: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_matvecs = int(num_matvecs)
        self.num_repeats = int(num_repeats)
        self.num_lanczos = num_lanczos
        self.num_bins = int(num_bins)
        self.sigma = sigma
        self.dtype = dtype
        self.device = device

    def estimate(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        return spectral_density(
            matvec,
            dim,
            num_matvecs=self.num_matvecs,
            num_repeats=self.num_repeats,
            num_lanczos=self.num_lanczos,
            num_bins=self.num_bins,
            sigma=self.sigma,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

    __call__ = estimate
