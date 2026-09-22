"""Hessian-vector products and spectral density estimation for PINN loss landscapes.

Implements:
  * Hessian-vector products (HVPs) via the Pearlmutter trick (double backprop).
  * Stochastic Lanczos Quadrature (SLQ) for estimating the spectral density of a
    symmetric operator accessed only through matvecs.
  * Condition number estimation (lambda_1 / lambda_n).

These utilities are used by the spectral-density experiments (Figures 3 & 7) and,
indirectly, by the NNCG optimizer (which relies on HVPs of the PINN loss Hessian).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch


__all__ = [
    "flatten_params",
    "unflatten_params",
    "make_hvp",
    "hessian_vector_product",
    "lanczos_tridiag",
    "slq_spectral_density",
    "estimate_extreme_eigenvalues",
    "condition_number",
]


# ---------------------------------------------------------------------------
# Parameter (un)flattening helpers
# ---------------------------------------------------------------------------
def flatten_params(params) -> torch.Tensor:
    """Concatenate a list of parameter tensors into a single flat vector."""
    return torch.cat([p.reshape(-1) for p in params])


def unflatten_params(params, vec: torch.Tensor) -> None:
    """Copy a flat vector back into the parameter tensors in place."""
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].reshape(p.shape))
        offset += n


# ---------------------------------------------------------------------------
# Hessian-vector products (Pearlmutter)
# ---------------------------------------------------------------------------
def make_hvp(loss_fn: Callable[[], torch.Tensor], params) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a Hessian-vector-product callable for ``loss_fn`` w.r.t. ``params``.

    Parameters
    ----------
    loss_fn : callable
        Zero-argument callable returning a scalar loss tensor. It must be built
        from the current parameter values (i.e. a fresh forward pass).
    params : iterable of torch.nn.Parameter
        Parameters the Hessian is taken with respect to.

    Returns
    -------
    hvp : callable
        ``hvp(v)`` returns ``H v`` where ``H`` is the Hessian of ``loss_fn``.
    """
    params = list(params)

    def hvp(v: torch.Tensor) -> torch.Tensor:
        return hessian_vector_product(loss_fn, params, v)

    return hvp


def hessian_vector_product(
    loss_fn: Callable[[], torch.Tensor],
    params,
    v: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute ``H v`` using the Pearlmutter trick (double backprop).

    The gradient is computed with ``create_graph=True`` so that a second
    backward pass through the gradient graph yields the Hessian-vector product.
    """
    params = list(params)

    # Fresh forward + first-order gradient with graph retained.
    loss = loss_fn()
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=create_graph, allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

    # Second backward pass: d/dparams (grads . v)
    flat_grads = torch.cat([g.reshape(-1) for g in grads])
    grad_dot_v = torch.dot(flat_grads, v)

    second = torch.autograd.grad(
        grad_dot_v, params, retain_graph=False, allow_unused=True
    )
    second = [g if g is not None else torch.zeros_like(p) for g, p in zip(second, params)]
    return torch.cat([g.reshape(-1) for g in second])


# ---------------------------------------------------------------------------
# Lanczos tridiagonalization
# ---------------------------------------------------------------------------
def lanczos_tridiag(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    m: int,
    v0: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``m`` steps of the Lanczos algorithm on a symmetric operator.

    Returns
    -------
    alpha : (m,) tensor of diagonal entries of the tridiagonal matrix T.
    beta : (m,) tensor of off-diagonal entries (beta[0] unused / zero).
    """
    if v0 is None:
        v0 = torch.randn(p, dtype=dtype, device=device)
    v0 = v0.to(dtype=dtype, device=device)
    v0 = v0 / torch.linalg.norm(v0)

    alpha = torch.zeros(m, dtype=dtype, device=device)
    beta = torch.zeros(m, dtype=dtype, device=device)

    v_prev = torch.zeros_like(v0)
    v_cur = v0
    b = 0.0

    for j in range(m):
        w = matvec(v_cur).to(dtype=dtype, device=device)
        a = torch.dot(w, v_cur)
        alpha[j] = a
        w = w - a * v_cur - b * v_prev
        # Full re-orthogonalization for numerical stability (small m).
        # (Optional; skipped for speed but kept simple here.)
        b_new = torch.linalg.norm(w)
        if j + 1 < m:
            beta[j + 1] = b_new
        if b_new < 1e-14:
            # Invariant subspace reached; truncate.
            alpha = alpha[: j + 1]
            beta = beta[: j + 1]
            break
        v_prev = v_cur
        v_cur = w / b_new
        b = b_new

    return alpha, beta


def _tridiag_eig(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of the symmetric tridiagonal matrix defined by alpha/beta."""
    m = alpha.shape[0]
    T = torch.diag(alpha)
    if m > 1:
        off = beta[1:m]
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    eigvals = torch.linalg.eigvalsh(T)
    return eigvals


# ---------------------------------------------------------------------------
# Stochastic Lanczos Quadrature (SLQ) spectral density
# ---------------------------------------------------------------------------
def slq_spectral_density(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    n_iters: int = 100,
    n_probes: int = 1,
    n_bins: int = 100,
    grid: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate the spectral density of a symmetric operator via SLQ.

    Parameters
    ----------
    matvec : callable
        Symmetric operator matvec ``v -> A v``.
    p : int
        Dimension of the operator.
    n_iters : int
        Number of Lanczos iterations per probe (paper uses ~100).
    n_probes : int
        Number of random probe vectors (Gauss quadrature nodes averaged).
    n_bins : int
        Number of histogram bins for the density.
    grid : tensor, optional
        Evaluation grid for the density. If None, derived from estimated
        extreme eigenvalues.

    Returns
    -------
    dict with keys:
        ``grid``  : (n_bins,) evaluation points
        ``density`` : (n_bins,) estimated spectral density
        ``eigvals`` : concatenated Ritz values across probes
        ``weights`` : corresponding quadrature weights
        ``lambda_max``, ``lambda_min`` : estimated extreme eigenvalues
    """
    if device is None:
        device = torch.device("cpu")

    all_eigvals: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []

    for _ in range(n_probes):
        if generator is not None:
            v0 = torch.randn(p, dtype=dtype, device=device, generator=generator)
        else:
            v0 = torch.randn(p, dtype=dtype, device=device)
        alpha, beta = lanczos_tridiag(matvec, p, n_iters, v0=v0, dtype=dtype, device=device)
        eigvals = _tridiag_eig(alpha, beta)
        # Quadrature weights: squared first component of eigenvectors of T.
        m = alpha.shape[0]
        T = torch.diag(alpha)
        if m > 1:
            off = beta[1:m]
            T = T + torch.diag(off, 1) + torch.diag(off, -1)
        _, evecs = torch.linalg.eigh(T)
        weights = (evecs[0, :] ** 2)
        all_eigvals.append(eigvals)
        all_weights.append(weights)

    eigvals = torch.cat(all_eigvals)
    weights = torch.cat(all_weights)
    weights = weights / weights.sum()

    lam_max = float(eigvals.max().item())
    lam_min = float(eigvals.min().item())

    if grid is None:
        lo = min(lam_min, 0.0)
        hi = max(lam_max, 1e-12)
        grid = torch.linspace(lo, hi, n_bins, dtype=dtype, device=device)

    # Gaussian kernel density estimate from weighted Ritz values.
    # Bandwidth via Silverman's rule of thumb.
    std = float(eigvals.std().item()) if eigvals.numel() > 1 else 1.0
    span = max(lam_max - lam_min, 1e-12)
    bw = max(1.06 * std * (eigvals.numel() ** (-1.0 / 5.0)), span / (n_bins * 4.0))
    bw = max(bw, 1e-12)

    diff = (grid[:, None] - eigvals[None, :]) / bw
    kernel = torch.exp(-0.5 * diff ** 2) / (bw * (2.0 * torch.pi) ** 0.5)
    density = kernel @ weights

    return {
        "grid": grid,
        "density": density,
        "eigvals": eigvals,
        "weights": weights,
        "lambda_max": torch.tensor(lam_max, dtype=dtype),
        "lambda_min": torch.tensor(lam_min, dtype=dtype),
    }


def estimate_extreme_eigenvalues(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    n_iters: int = 100,
    n_probes: int = 1,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tuple[float, float]:
    """Estimate the largest and smallest eigenvalues via Lanczos Ritz values."""
    res = slq_spectral_density(
        matvec, p, n_iters=n_iters, n_probes=n_probes, dtype=dtype, device=device
    )
    return float(res["lambda_max"].item()), float(res["lambda_min"].item())


def condition_number(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    n_iters: int = 100,
    n_probes: int = 1,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> float:
    """Estimate the condition number ``lambda_1 / lambda_n`` of a PSD operator."""
    lam_max, lam_min = estimate_extreme_eigenvalues(
        matvec, p, n_iters=n_iters, n_probes=n_probes, dtype=dtype, device=device
    )
    if lam_min <= 0:
        return float("inf")
    return lam_max / lam_min
