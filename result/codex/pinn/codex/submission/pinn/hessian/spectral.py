"""Stochastic Lanczos quadrature (SLQ) estimates of Hessian spectral densities.

SLQ approximates the spectral density of a symmetric matrix ``M`` with a
Gaussian quadrature rule built from a Krylov subspace (Golub & Meurant, 2009;
Lin et al., 2016).  PyHessian (Yao et al., 2020) uses exactly this technique to
visualise neural-network Hessians; the implementation below follows the same
recipe and only requires matrix-vector products.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def slq_lanczos(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    n_iter: int = 100,
    v0: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    dtype: torch.dtype = torch.float64,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Run one Lanczos run and return ``(nodes, weights, (alpha, beta))``.

    ``nodes``/``weights`` define the Gaussian quadrature rule of the Krylov
    subspace: the spectral density estimate is ``sum_i weights_i delta(x - nodes_i)``.
    """
    if v0 is None:
        v0 = torch.randn(dim, dtype=dtype)
    q = v0 / torch.linalg.norm(v0)
    alpha = torch.zeros(n_iter, dtype=dtype)
    beta = torch.zeros(n_iter, dtype=dtype)
    Q = torch.zeros((n_iter, dim), dtype=dtype)
    # full reorthogonalisation makes the tridiagonalisation numerically stable
    for j in range(n_iter):
        Q[j] = q
        r = matvec(q).to(dtype)
        alpha[j] = torch.dot(q, r)
        r = r - alpha[j] * q
        if j > 0:
            r = r - beta[j - 1] * Q[j - 1]
        # re-orthogonalise (twice) against the whole basis
        for _ in range(2):
            r = r - Q[: j + 1].T @ (Q[: j + 1] @ r)
        b = torch.linalg.norm(r)
        if b < tol:
            n_used = j + 1
            break
        beta[j] = b
        q = r / b
        n_used = j + 1
    else:
        n_used = n_iter

    T = torch.zeros((n_used, n_used), dtype=dtype)
    T[range(n_used), range(n_used)] = alpha[:n_used]
    for j in range(n_used - 1):
        T[j, j + 1] = beta[j]
        T[j + 1, j] = beta[j]
    evals, evecs = torch.linalg.eigh(T)
    weights = evecs[0, :] ** 2
    return evals.numpy(), weights.numpy(), (alpha[:n_used].numpy(), beta[: n_used - 1].numpy())


def spectral_density(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    n_runs: int = 1,
    n_iter: int = 100,
    num_bins: int = 800,
    sigma: Optional[float] = None,
    bandwidth_scale: float = 2.0,
    overhead: float = 0.01,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the spectral density of ``M`` and evaluate it on a grid.

    Returns ``(grids, density, eigenvalues)`` with one row of Ritz values per
    Lanczos run.  The quadrature measure ``sum_i w_i delta(x - theta_i)`` is
    smoothed with a Gaussian kernel whose bandwidth defaults to
    ``bandwidth_scale * (lambda_max - lambda_min) / n_iter``: small enough to
    separate the outlier eigenvalues from the bulk of the spectrum, large
    enough to avoid a comb of delta spikes.
    """
    gen = np.random.default_rng(seed)
    eigenvalues: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    for r in range(n_runs):
        v0 = torch.tensor(gen.standard_normal(dim), dtype=torch.float64)
        nodes, w, _ = slq_lanczos(matvec, dim, n_iter=n_iter, v0=v0)
        eigenvalues.append(nodes)
        weights.append(w)
        if verbose:
            print(f"  [slq] run {r + 1}/{n_runs}: ", end="")
            print(
                f"min={nodes.min():.3e} max={nodes.max():.3e} "
                f"top3={np.sort(nodes)[-3:]}"
            )
    ev = np.stack(eigenvalues)
    wt = np.stack(weights)

    lambda_max = float(np.mean(np.max(ev, axis=1))) + overhead
    lambda_min = float(np.mean(np.min(ev, axis=1))) - overhead
    # ---- kernel bandwidth -------------------------------------------- #
    nodes_flat = ev.reshape(-1)
    w_flat = wt.reshape(-1)
    if sigma is None:
        sigma = float(
            max(bandwidth_scale * (lambda_max - lambda_min) / max(nodes_flat.size, 1), 1e-12)
        )
    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    density = np.zeros(num_bins)
    for nodes, w in zip(ev, wt):
        density += (w[:, None] * np.exp(-((grids[None, :] - nodes[:, None]) ** 2) / (2 * sigma**2))).sum(
            axis=0
        )
    density = density / (np.sqrt(2 * np.pi) * sigma * max(len(ev), 1))
    return grids, density, ev
