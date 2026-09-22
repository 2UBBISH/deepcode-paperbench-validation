"""Spectral-density estimation for the PINN loss Hessian and its L-BFGS preconditioning.

This module implements the estimator used to produce Figures 3 and 7 of
"Challenges in Training PINNs: A Loss Landscape Perspective" (ICML 2024).

The paper (Sections 5.1 - 5.3, Appendix C.2) examines the eigenvalue distribution
of the PINN loss Hessian ``H_L(w)`` and of the L-BFGS-preconditioned Hessian

    Htilde_k^T H_L(w) Htilde_k,

whose (non-zero) eigenvalues coincide with those of ``H_k H_L(w)``
(Theorem 1.3.22 of Horn & Johnson).  Both matrices are analysed with **stochastic
Lanczos quadrature (SLQ)** (Golub & Meurant, 2009; Lin et al., 2016), the same
machinery used by PyHessian (Yao et al., 2020).  SLQ only requires matrix-vector
products, so the (possibly huge) Hessian is never materialised:

* ``H_L(w) v`` is obtained by a Pearlmutter double-backward pass through the loss
  (:mod:`src.spectral.hvp`).
* ``Htilde_k u`` / ``Htilde_k^T u`` are obtained from the unrolled L-BFGS factors
  (:mod:`src.spectral.lbfgs_unroll`), and ``Htilde_k^T H_L(w) Htilde_k`` is applied
  with Algorithm 3 (:mod:`src.spectral.preconditioned_mvp`).

Key deliverables reproduced here
-------------------------------
* ``spectral_density`` / ``slq_density`` : SLQ density estimate of an operator.
* ``lanczos_tridiag`` : Lanczos tridiagonalisation (the quadrature backbone).
* ``estimate_condition_number`` : ``|lambda_max| / |lambda_min|`` of an operator.
* ``SpectralDensityEstimator`` : high-level driver handling
  ``H_L`` / preconditioned ``H_L`` / per-component (residual, IC, BC) losses.

Everything runs in ``float64`` for numerical stability, matching the rest of the
second-order code in this repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

# --------------------------------------------------------------------------- #
# imports from sibling modules (tolerant of the two import roots in this repo)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import shim
    from .hvp import (
        DEFAULT_DTYPE,
        HessianOperator,
        LinearOperator,
        as_operator,
        cast_model_dtype,
        hvp as _hvp_fn,
        num_parameters,
        power_iteration,
        top_eigenvalues,
    )
except ImportError:  # pragma: no cover - import shim
    from src.spectral.hvp import (  # type: ignore
        DEFAULT_DTYPE,
        HessianOperator,
        LinearOperator,
        as_operator,
        cast_model_dtype,
        hvp as _hvp_fn,
        num_parameters,
        power_iteration,
        top_eigenvalues,
    )

try:  # pragma: no cover - import shim
    from .lbfgs_unroll import LBFGSFactors, unroll_from_history, unroll_lbfgs
except ImportError:  # pragma: no cover - import shim
    try:
        from src.spectral.lbfgs_unroll import (  # type: ignore
            LBFGSFactors,
            unroll_from_history,
            unroll_lbfgs,
        )
    except ImportError:  # pragma: no cover - allowed: unroll is optional at import time
        LBFGSFactors = None  # type: ignore
        unroll_from_history = None  # type: ignore
        unroll_lbfgs = None  # type: ignore

try:  # pragma: no cover - import shim
    from .preconditioned_mvp import (
        PreconditionedHessian,
        make_spectral_operator,
    )
except ImportError:  # pragma: no cover - import shim
    try:
        from src.spectral.preconditioned_mvp import (  # type: ignore
            PreconditionedHessian,
            make_spectral_operator,
        )
    except ImportError:  # pragma: no cover - allowed
        PreconditionedHessian = None  # type: ignore
        make_spectral_operator = None  # type: ignore

__all__ = [
    "DEFAULT_DTYPE",
    "MatVec",
    "DensityResult",
    "LanczosResult",
    "lanczos_tridiag",
    "slq_density",
    "spectral_density",
    "SpectralDensity",
    "estimate_condition_number",
    "top_eigenvalue",
    "top_k_eigenvalues",
    "COMPONENTS",
    "component_hvp",
    "component_operator",
    "SpectralDensityEstimator",
    "SLQBackend",
    "pyhessian_available",
]

MatVec = Callable[[torch.Tensor], torch.Tensor]

#: loss components analysed in Section 5.2 / Figures 3 (bottom) and 7.
COMPONENTS: Tuple[str, ...] = ("residual", "initial", "boundary")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    return DEFAULT_DTYPE if dtype is None else dtype


def _as_matvec(A, *, n: Optional[int] = None, dtype: Optional[torch.dtype] = None,
               device=None) -> MatVec:
    """Normalise a dense tensor / operator object / callable into a matvec callable."""
    if hasattr(A, "matvec") and callable(getattr(A, "matvec")):
        return lambda v: A.matvec(v)
    if callable(A):
        return A
    if isinstance(A, torch.Tensor):
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("dense operator must be square")
        return lambda v: A @ v
    if n is not None:
        raise TypeError("cannot interpret the given operator as a matvec function")
    raise TypeError("A must be a callable, a square Tensor, or expose .matvec()")


def _operator_dim(A, n: Optional[int] = None) -> int:
    if n is not None:
        return int(n)
    if isinstance(A, torch.Tensor):
        return int(A.shape[0])
    for attr in ("n", "size", "dim"):
        val = getattr(A, attr, None)
        if isinstance(val, int):
            return int(val)
    shape = getattr(A, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0])
    raise ValueError("operator dimension `n` must be provided")


def _symmetrize(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * (v + v.transpose(-1, -2))


def pyhessian_available() -> bool:
    """Return True when the optional ``pyhessian`` package can be imported."""
    try:  # pragma: no cover - depends on environment
        import pyhessian  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


SLQBackend = str  # "native" | "pyhessian"


# --------------------------------------------------------------------------- #
# Lanczos tridiagonalisation
# --------------------------------------------------------------------------- #
@dataclass
class LanczosResult:
    """Outcome of a Lanczos tridiagonalisation."""

    alpha: torch.Tensor          # (k,) diagonal of T
    beta: torch.Tensor           # (k-1,) off-diagonal of T
    n_iter: int
    breakdown: bool
    n_matvecs: int
    dtype: torch.dtype
    norm: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.alpha.numel())

    def tridiag(self) -> torch.Tensor:
        k = self.k
        T = torch.diag(self.alpha)
        if k > 1 and self.beta.numel() >= k - 1:
            off = self.beta[: k - 1]
            T = T + torch.diag(off, 1) + torch.diag(off, -1)
        return T

    def eigenvalues(self) -> torch.Tensor:
        """Ritz values (eigenvalue estimates) of the Lanczos tridiagonal matrix."""
        return torch.linalg.eigvalsh(self.tridiag())

    def as_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha.detach().cpu().tolist(),
            "beta": self.beta.detach().cpu().tolist(),
            "n_iter": self.n_iter,
            "breakdown": self.breakdown,
            "n_matvecs": self.n_matvecs,
        }


def lanczos_tridiag(
    A,
    *,
    n: Optional[int] = None,
    n_iter: int = 100,
    q0: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    device=None,
    generator: Optional[torch.Generator] = None,
    return_basis: bool = False,
):
    """Lanczos tridiagonalisation of a symmetric operator.

    Runs the standard three-term recurrence starting from ``q0`` (default: a random
    unit vector) so that ``Q^T A Q = T`` with ``T`` tridiagonal.  This is the
    quadrature backbone of stochastic Lanczos quadrature.

    Args:
        A: symmetric operator (callable matvec, dense Tensor, or object with
            ``.matvec``).
        n: dimension of the operator (inferred when possible).
        n_iter: maximum number of Lanczos iterations (PyHessian default: 100).
        q0: optional starting vector.
        tol: breakdown tolerance on ``beta``.
        dtype, device: numeric options.
        generator: torch RNG for the random start vector.
        return_basis: if True also return the Lanczos basis ``Q (n, k)``.

    Returns:
        :class:`LanczosResult` (and ``Q`` when ``return_basis`` is True).
    """
    dtype = _resolve_dtype(dtype)
    n_dim = _operator_dim(A, n)
    matvec = _as_matvec(A, n=n_dim, dtype=dtype, device=device)

    if q0 is None:
        q0 = torch.randn(n_dim, dtype=dtype, device=device, generator=generator)
    q0 = q0.to(dtype=dtype, device=device).reshape(-1)
    if q0.numel() != n_dim:
        raise ValueError(f"q0 has {q0.numel()} entries but operator dim is {n_dim}")

    norm0 = float(q0.norm())
    if norm0 == 0.0 or not math.isfinite(norm0):
        q0 = torch.zeros(n_dim, dtype=dtype, device=device)
        q0[0] = 1.0
        norm0 = 1.0
    q_prev = torch.zeros_like(q0)
    q_cur = q0 / norm0
    beta_prev = 0.0

    alphas: List[float] = []
    betas: List[float] = []
    basis: List[torch.Tensor] = []
    breakdown = False
    n_matvecs = 0

    for _ in range(int(n_iter)):
        if return_basis:
            basis.append(q_cur)
        w = matvec(q_cur)
        n_matvecs += 1
        w = w.reshape(-1).to(dtype=dtype)
        alpha = float(torch.dot(q_cur, w))
        w = w - alpha * q_cur - beta_prev * q_prev
        # full re-orthogonalisation keeps the basis orthonormal for clustered spectra
        if return_basis and len(basis) > 0:
            Qs = torch.stack(basis, dim=1)
            w = w - Qs @ (Qs.transpose(0, 1) @ w)
        beta = float(w.norm())

        alphas.append(alpha)
        if beta <= tol * max(1.0, abs(alpha)):
            breakdown = True
            break
        betas.append(beta)
        q_prev = q_cur
        q_cur = w / beta
        beta_prev = beta

    res = LanczosResult(
        alpha=torch.tensor(alphas, dtype=dtype, device=device),
        beta=torch.tensor(betas, dtype=dtype, device=device),
        n_iter=len(alphas),
        breakdown=breakdown,
        n_matvecs=n_matvecs,
        dtype=dtype,
        norm=norm0,
        extra={},
    )
    if return_basis:
        Q = torch.stack(basis, dim=1) if basis else torch.zeros(n_dim, 0, dtype=dtype, device=device)
        return res, Q
    return res


# --------------------------------------------------------------------------- #
# SLQ density estimation
# --------------------------------------------------------------------------- #
@dataclass
class DensityResult:
    """Estimated spectral density of an operator."""

    grid: torch.Tensor                # (n_grid,) evaluation points (eigenvalue axis)
    density: torch.Tensor             # (n_grid,) averaged density
    eigenvalues: torch.Tensor         # (n_iter,) concatenated Ritz values
    weights: torch.Tensor             # (n_iter,) quadrature weights of those Ritz values
    alpha: torch.Tensor               # (n_iter, n_vec) Lanczos alphas
    beta: torch.Tensor                # (n_iter-1, n_vec) Lanczos betas
    n_vec: int
    n_iter: int
    n_matvecs: int
    eigenvalues_top: Optional[torch.Tensor] = None
    condition_number: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "grid": self.grid.detach().cpu().tolist(),
            "density": self.density.detach().cpu().tolist(),
            "n_vec": self.n_vec,
            "n_iter": self.n_iter,
            "n_matvecs": self.n_matvecs,
            "condition_number": self.condition_number,
        }


def _gaussian_kernel(x: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian smoothing kernel: ``(n_grid, n_centers)``."""
    if sigma <= 0:
        out = torch.zeros(x.numel(), centers.numel(), dtype=x.dtype, device=x.device)
        idx = torch.argmin(torch.abs(x.reshape(-1, 1) - centers.reshape(1, -1)), dim=0)
        out[idx, torch.arange(centers.numel(), device=x.device)] = 1.0
        return out
    return torch.exp(-((x.reshape(-1, 1) - centers.reshape(1, -1)) ** 2) / (2.0 * sigma ** 2))


def _default_grid(lo: float, hi: float, n_grid: int, dtype, device) -> torch.Tensor:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = (lo if math.isfinite(lo) else 0.0), (hi if math.isfinite(hi) else 1.0) 
        if hi <= lo:
            hi = lo + 1.0
    return torch.linspace(lo, hi, int(n_grid), dtype=dtype, device=device)


def slq_density(
    A,
    *,
    n: Optional[int] = None,
    n_vec: int = 1,
    n_iter: int = 100,
    grid: Optional[torch.Tensor] = None,
    n_grid: int = 200,
    sigma: Optional[float] = None,
    eigenvalues: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device=None,
    generator: Optional[torch.Generator] = None,
    largest: Optional[float] = None,
    top_k: int = 0,
    return_raw: bool = False,
) -> DensityResult:
    """Stochastic Lanczos quadrature estimate of an operator's spectral density.

    For each of ``n_vec`` random probe vectors (Rademacher entries, as in
    PyHessian) a Lanczos tridiagonalisation is performed; the density is the
    Gaussian-smoothed (``sigma``) average of the Ritz-value Dirac masses weighted
    by the first components of the eigenvectors of ``T`` (the SLQ quadrature
    weights).

    Args:
        A: symmetric operator (matvec callable / dense Tensor / object with
            ``.matvec``).
        n: operator dimension.
        n_vec: number of probe vectors (PyHessian default 1 for a full run; use
            more for smoother pictures).
        n_iter: Lanczos iterations per probe (PyHessian default 100).
        grid: explicit evaluation grid; otherwise ``n_grid`` points spanning
            ``[min(ritz), max(ritz)]`` are used (unless ``eigenvalues`` given).
        sigma: Gaussian smoothing bandwidth; defaults to a fraction of the
            grid spacing.
        eigenvalues: explicit eigenvalue axis ``(n_grid,)``/``(2,)`` bounds.
        largest: known upper bound (e.g. from power iteration) to extend the grid.
        top_k: if > 0, also estimate the ``top_k`` eigenvalues (deflated power
            iteration) and store them in ``eigenvalues_top``.
        return_raw: keep the full Lanczos ``alpha``/``beta`` (always kept; kept for
            API symmetry).

    Returns:
        :class:`DensityResult`.
    """
    dtype = _resolve_dtype(dtype)
    n_dim = _operator_dim(A, n)
    if device is None:
        if isinstance(A, torch.Tensor):
            device = A.device
        else:
            device = getattr(A, "device", None)
    matvec = _as_matvec(A, n=n_dim, dtype=dtype, device=device)

    n_vec = max(1, int(n_vec))
    n_iter = max(2, int(n_iter))

    alphas: List[List[float]] = []
    betas: List[List[float]] = []
    ritz_all: List[torch.Tensor] = []
    weights_all: List[torch.Tensor] = []
    n_matvecs = 0

    for _ in range(n_vec):
        v0 = torch.randint(0, 2, (n_dim,), dtype=dtype, device=device, generator=generator) * 2.0 - 1.0
        res = lanczos_tridiag(matvec, n=n_dim, n_iter=n_iter, q0=v0, dtype=dtype, device=device)
        n_matvecs += res.n_matvecs
        T = res.tridiag().to(dtype=dtype)
        try:
            evals, evecs = torch.linalg.eigh(T)
        except Exception:  # pragma: no cover - degenerate T
            evals = torch.linalg.eigvalsh(T)
            evecs = torch.eye(T.shape[0], dtype=dtype, device=T.device)
        w = (evecs[0, :] ** 2).to(dtype=dtype)
        w = w / torch.clamp(w.sum(), min=1e-300)
        ritz_all.append(evals)
        weights_all.append(w)
        alphas.append([float(x) for x in res.alpha])
        betas.append([float(x) for x in res.beta])

    ritz = torch.cat(ritz_all)
    weights = torch.cat(weights_all)

    # ---- eigenvalue axis -------------------------------------------------- #
    lo = float(ritz.min())
    hi = float(ritz.max())
    if largest is not None and math.isfinite(float(largest)):
        hi = max(hi, float(largest))
    pad = 0.05 * max(1e-12, hi - lo)
    lo_g, hi_g = lo - pad, hi + pad
    if eigenvalues is not None:
        ev = eigenvalues.to(dtype=dtype, device=device).reshape(-1)
        if ev.numel() == 2:
            grid = _default_grid(float(ev[0]), float(ev[1]), n_grid, dtype, device)
        else:
            grid = ev
    elif grid is not None:
        grid = grid.to(dtype=dtype, device=device).reshape(-1)
    else:
        grid = _default_grid(lo_g, hi_g, n_grid, dtype, device)

    if sigma is None:
        dx = float(grid[1] - grid[0]) if grid.numel() > 1 else 1.0
        sigma = max(1e-12, 1.0 * abs(dx))

    K = _gaussian_kernel(grid, ritz, float(sigma))                 # (n_grid, n_ritz)
    density = K @ weights
    # normalise so that the density integrates to 1 over the grid
    if grid.numel() > 1:
        dx = float(grid[1] - grid[0])
        integral = float(density.sum()) * abs(dx)
        if integral > 0:
            density = density / integral

    top = None
    if top_k and top_k > 0:
        try:
            top = top_eigenvalues(matvec, k=int(top_k), n=n_dim, dtype=dtype, device=device)
        except Exception:  # pragma: no cover
            top = None

    cond = None
    if ritz.numel() > 1:
        r_min = float(ritz.min())
        r_max = float(ritz.max())
        if abs(r_min) > 0:
            cond = abs(r_max) / abs(r_min)

    alpha_t = torch.tensor(alphas, dtype=dtype, device=device).transpose(0, 1)
    beta_t = torch.tensor(betas, dtype=dtype, device=device).transpose(0, 1)

    return DensityResult(
        grid=grid,
        density=density,
        eigenvalues=ritz,
        weights=weights,
        alpha=alpha_t,
        beta=beta_t,
        n_vec=n_vec,
        n_iter=n_iter,
        n_matvecs=n_matvecs,
        eigenvalues_top=top,
        condition_number=cond,
        extra={"sigma": float(sigma), "n": int(n_dim)},
    )


def spectral_density(A, **kwargs) -> DensityResult:
    """Alias of :func:`slq_density` (paper term for the SLQ estimate)."""
    return slq_density(A, **kwargs)


SpectralDensity = DensityResult


# --------------------------------------------------------------------------- #
# conditioning diagnostics
# --------------------------------------------------------------------------- #
def top_eigenvalue(A, *, n: Optional[int] = None, n_iter: int = 100,
                   dtype: Optional[torch.dtype] = None, device=None) -> float:
    """Largest-magnitude eigenvalue of a symmetric operator (power iteration)."""
    dtype = _resolve_dtype(dtype)
    n_dim = _operator_dim(A, n)
    matvec = _as_matvec(A, n=n_dim, dtype=dtype, device=device)
    return float(power_iteration(matvec, n=n_dim, n_iter=n_iter, dtype=dtype, device=device))


def top_k_eigenvalues(A, k: int = 10, *, n: Optional[int] = None, n_iter: int = 200,
                      dtype: Optional[torch.dtype] = None, device=None) -> torch.Tensor:
    """Top-``k`` eigenvalues of a symmetric operator (deflated power iteration)."""
    dtype = _resolve_dtype(dtype)
    n_dim = _operator_dim(A, n)
    matvec = _as_matvec(A, n=n_dim, dtype=dtype, device=device)
    return top_eigenvalues(matvec, k=int(k), n=n_dim, n_iter=n_iter, dtype=dtype, device=device)


def estimate_condition_number(
    A,
    *,
    n: Optional[int] = None,
    n_iter: int = 100,
    n_vec: int = 1,
    dtype: Optional[torch.dtype] = None,
    device=None,
    generator: Optional[torch.Generator] = None,
    return_details: bool = False,
):
    """Estimate ``|lambda_max| / |lambda_min|`` of a symmetric operator via SLQ.

    The Lanczos Ritz values are used as eigenvalue estimates (extremal Ritz values
    converge rapidly to the extremal eigenvalues).  When ``return_details`` is
    True a dict with ``cond``, ``lambda_max``, ``lambda_min`` and the Ritz values is
    returned.
    """
    res = slq_density(
        A,
        n=n,
        n_vec=n_vec,
        n_iter=n_iter,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    ritz = res.eigenvalues
    lam_max = float(ritz.max()) if ritz.numel() else float("nan")
    lam_min = float(ritz.min()) if ritz.numel() else float("nan")
    cond = abs(lam_max) / abs(lam_min) if lam_min not in (0.0,) and math.isfinite(lam_min) else float("inf")
    if return_details:
        return {
            "cond": cond,
            "lambda_max": lam_max,
            "lambda_min": lam_min,
            "ritz": ritz.detach().cpu(),
            "n_matvecs": res.n_matvecs,
        }
    return cond


# --------------------------------------------------------------------------- #
# loss components (Section 5.2 / Figures 3 bottom, 7)
# --------------------------------------------------------------------------- #
def _component_index(problem, component: str) -> Optional[int]:
    """Map a component name to the index of its condition (or None for residual)."""
    if component == "residual":
        return None
    conds = list(problem.conditions())
    wanted_initial = component in ("initial", "ic", "initial_condition")
    wanted_boundary = component in ("boundary", "bc", "boundary_condition")
    for i, c in enumerate(conds):
        name = getattr(c, "name", "") or ""
        kind = getattr(c, "kind", "") or ""
        is_initial = ("ic" in kind.lower()) or ("init" in name.lower())
        is_boundary = ("periodic" in kind.lower()) or ("dirichlet" in kind.lower()) \
            or ("bc" in name.lower()) or ("bound" in name.lower())
        if wanted_initial and is_initial:
            return i
        if wanted_boundary and is_boundary:
            return i
    return None


def component_hvp(
    model,
    problem,
    sampler=None,
    component: str = "residual",
    *,
    dtype: Optional[torch.dtype] = None,
) -> MatVec:
    """Hessian matvec oracle for a single loss component (residual / IC / BC).

    Builds a loss closure restricted to ``component`` using
    :mod:`src.pinns.loss` and wraps it with :class:`~src.spectral.hvp.HessianOperator`.
    """
    dtype = _resolve_dtype(dtype)
    try:  # pragma: no cover - import shim
        from ..pinns.loss import PINNLoss, loss_breakdown
    except ImportError:  # pragma: no cover
        from src.pinns.loss import PINNLoss, loss_breakdown  # type: ignore

    loss_obj = PINNLoss(
        model,
        problem,
        sampler=sampler,
        components="all",
        dtype=dtype,
    )
    cond_index = _component_index(problem, component)
    target = {"residual"} if component == "residual" else None

    def loss_fn() -> torch.Tensor:
        bd = loss_obj.breakdown(include_components=target if target is None else ["residual"])
        if component == "residual":
            return bd.residual
        if cond_index is None:
            # fall back: initial terms by kind, otherwise boundary terms
            if component in ("initial", "ic", "initial_condition"):
                return bd.initial
            return bd.boundary
        conds = list(problem.conditions())
        vals = loss_obj.condition_values_all()[cond_index]
        n = max(1, vals.numel())
        return (vals ** 2).sum() / (2.0 * n)

    return HessianOperator(loss_fn, model, dtype=dtype)


def component_operator(model, problem, sampler=None, component: str = "residual",
                       *, dtype: Optional[torch.dtype] = None) -> HessianOperator:
    """Return the :class:`HessianOperator` of a single loss component."""
    return component_hvp(model, problem, sampler, component, dtype=dtype)


# --------------------------------------------------------------------------- #
# high level estimator
# --------------------------------------------------------------------------- #
class SpectralDensityEstimator:
    """SLQ spectral-density estimator for the PINN Hessian.

    The estimator mirrors the analysis of Section 5 / Appendix C.2:

    * ``loss_density``            -> ``H_L(w)``           (Fig. 3 top, solid)
    * ``preconditioned_density``  -> ``Htilde_k^T H_L(w) Htilde_k`` (Fig. 3 top, dashed)
    * ``component_density``       -> same two operators for residual / IC / BC
      (Fig. 3 bottom, Fig. 7)

    Args:
        model: the trained :class:`torch.nn.Module` PINN.
        loss_fn: zero-argument closure returning the (scalar, graph-carrying)
            total PINN loss.  When ``None`` it is built from ``model``/``problem``.
        problem: optional :class:`~src.pinns.problems.PDEProblem`.
        sampler: optional :class:`~src.pinns.sampling.PINNSampler`.
        lbfgs_history: recorded L-BFGS curvature history (``LBFGSHistory``) used
            to build the preconditioner.
        n_iter: Lanczos iterations per probe (PyHessian default 100).
        n_vec: probe vectors per estimate.
        n_grid: number of points on the density grid.
        dtype: numeric dtype (default ``float64``).
        device: torch device.
        seed: RNG seed for the probe vectors.
    """

    def __init__(
        self,
        model,
        loss_fn: Optional[Callable[[], torch.Tensor]] = None,
        problem=None,
        sampler=None,
        lbfgs_history=None,
        *,
        n_iter: int = 100,
        n_vec: int = 1,
        n_grid: int = 200,
        sigma: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        seed: Optional[int] = None,
        backend: SLQBackend = "native",
    ) -> None:
        self.dtype = _resolve_dtype(dtype)
        self.device = device
        self.model = model
        self.problem = problem
        self.sampler = sampler
        self.lbfgs_history = lbfgs_history
        self.n_iter = int(n_iter)
        self.n_vec = int(n_vec)
        self.n_grid = int(n_grid)
        self.sigma = sigma
        self.seed = seed
        self.backend = backend
        self._generator = None
        self._loss_fn = loss_fn
        self._opf = None
        if loss_fn is not None:
            self._opf = HessianOperator(loss_fn, model, dtype=self.dtype)

    # -- infrastructure ---------------------------------------------------- #
    @staticmethod
    def from_training_run(model, problem=None, sampler=None, loss_fn=None,
                          lbfgs_history=None, **kwargs) -> "SpectralDensityEstimator":
        """Convenience constructor mirroring the experiment-runner call sites."""
        if loss_fn is None and problem is not None:
            try:  # pragma: no cover - import shim
                from ..pinns.loss import make_loss_fn
            except ImportError:  # pragma: no cover
                from src.pinns.loss import make_loss_fn  # type: ignore

            loss_obj, closure = make_loss_fn(model, problem, sampler=sampler, dtype=kwargs.get("dtype"))
            loss_fn = closure
        return SpectralDensityEstimator(
            model, loss_fn, problem=problem, sampler=sampler,
            lbfgs_history=lbfgs_history, **kwargs
        )

    def _gen(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        if self._generator is None:
            g = torch.Generator(device=self.device or "cpu")
            g.manual_seed(int(self.seed))
            self._generator = g
        return self._generator

    @property
    def loss_operator(self) -> HessianOperator:
        if self._opf is None:
            raise ValueError(
                "no loss closure provided; pass loss_fn (or use .from_training_run)"
            )
        return self._opf

    @property
    def n_parameters(self) -> int:
        return num_parameters(self.model)

    # -- preconditioned operator ------------------------------------------- #
    def make_preconditioned_operator(
        self,
        factors=None,
        history=None,
        *,
        hvp: Optional[MatVec] = None,
        m: Optional[int] = None,
    ):
        """Build ``Htilde_k^T H_L(w) Htilde_k`` from unrolled L-BFGS factors."""
        if make_spectral_operator is None or PreconditionedHessian is None:
            raise ImportError("src.spectral.preconditioned_mvp is required for preconditioned densities")
        hist = history if history is not None else self.lbfgs_history
        hvp_fn = hvp if hvp is not None else (lambda v: self.loss_operator.matvec(v))
        if factors is None:
            if hist is None:
                raise ValueError("either `factors` or an L-BFGS history is required")
            if unroll_from_history is None:
                raise ImportError("src.spectral.lbfgs_unroll is required to unroll L-BFGS buffers")
            factors = unroll_from_history(hist, m=m, dtype=self.dtype)
        return make_spectral_operator(factors, hvp_fn, dtype=self.dtype)

    # -- density entry points ---------------------------------------------- #
    def density(self, operator, *, n: Optional[int] = None, **kwargs) -> DensityResult:
        """SLQ spectral density of an arbitrary symmetric operator."""
        params = dict(
            n=n,
            n_vec=self.n_vec,
            n_iter=self.n_iter,
            n_grid=self.n_grid,
            sigma=self.sigma,
            dtype=self.dtype,
            device=self.device,
            generator=self._gen(),
        )
        params.update(kwargs)
        return slq_density(operator, **params)

    def loss_density(self, **kwargs) -> DensityResult:
        """Spectral density of the raw Hessian ``H_L(w)`` (Fig. 3, solid lines)."""
        return self.density(self.loss_operator.matvec, n=self.n_parameters, **kwargs)

    def preconditioned_density(self, factors=None, history=None, *, m=None, **kwargs) -> DensityResult:
        """Spectral density of ``Htilde_k^T H_L(w) Htilde_k`` (Fig. 3, dashed lines)."""
        op = self.make_preconditioned_operator(factors, history, m=m)
        return self.density(op.matvec, n=op.size, **kwargs)

    def component_density(self, component: str, *, preconditioned: bool = False,
                          factors=None, history=None, m=None, **kwargs) -> DensityResult:
        """Spectral density of one loss component (residual / initial / boundary)."""
        if self.problem is None:
            raise ValueError("`problem` is required for per-component analysis")
        op = component_operator(
            self.model, self.problem, self.sampler, component, dtype=self.dtype
        )
        if not preconditioned:
            return self.density(op.matvec, n=self.n_parameters, **kwargs)
        return self.preconditioned_density(
            factors, history, m=m, hvp=op.matvec, **kwargs
        )

    def component_densities(self, *, preconditioned: bool = False,
                            components: Iterable[str] = COMPONENTS,
                            factors=None, history=None, m=None, **kwargs) -> Dict[str, DensityResult]:
        """Per-component densities for all components in ``components``."""
        out: Dict[str, DensityResult] = {}
        for comp in components:
            out[comp] = self.component_density(
                comp, preconditioned=preconditioned, factors=factors,
                history=history, m=m, **kwargs
            )
        return out

    # -- conditioning ------------------------------------------------------- #
    def condition_number(self, operator=None, *, n: Optional[int] = None,
                         return_details: bool = False, **kwargs):
        """Estimated condition number of ``operator`` (default: raw Hessian)."""
        if operator is None:
            operator = self.loss_operator.matvec
            n = self.n_parameters
        params = dict(
            n=n, n_iter=self.n_iter, n_vec=self.n_vec, dtype=self.dtype,
            device=self.device, generator=self._gen(), return_details=return_details,
        )
        params.update(kwargs)
        return estimate_condition_number(operator, **params)

    def conditioning_report(self, factors=None, history=None, m=None,
                            *, top_k: int = 5, **kwargs) -> Dict[str, object]:
        """Condition-number / top-eigenvalue report for raw and preconditioned Hessians.

        This is the quantitative counterpart of the Section 5.3 statement that
        L-BFGS preconditioning reduces the magnitude of the eigenvalues and the
        condition number by at least ``1e3``.
        """
        raw = self.condition_number(return_details=True, top_k=top_k, **kwargs)
        op = self.make_preconditioned_operator(factors, history, m=m)
        pre = self.condition_number(op.matvec, n=op.size, return_details=True, top_k=top_k, **kwargs)
        ratio_cond = (
            raw["cond"] / pre["cond"]
            if pre["cond"] not in (0.0, float("inf")) and math.isfinite(pre["cond"])
            else float("inf")
        )
        ratio_max = (
            raw["lambda_max"] / pre["lambda_max"]
            if pre["lambda_max"] not in (0.0,) and math.isfinite(pre["lambda_max"])
            else float("inf")
        )
        return {
            "raw": raw,
            "preconditioned": pre,
            "condition_number_reduction": ratio_cond,
            "top_eigenvalue_reduction": ratio_max,
        }

    def run(self, *, with_components: bool = True, with_preconditioned: bool = True,
            factors=None, history=None, m=None, components: Iterable[str] = COMPONENTS,
            **kwargs) -> Dict[str, object]:
        """Compute every density required by Figures 3 and 7 in one call.

        Returns a dict with keys ``loss``, ``preconditioned`` (when
        ``with_preconditioned``), ``components``/``components_preconditioned``
        (when ``with_components``) and ``conditioning``.
        """
        out: Dict[str, object] = {}
        out["loss"] = self.loss_density(**kwargs)
        if with_preconditioned:
            try:
                out["preconditioned"] = self.preconditioned_density(
                    factors, history, m=m, **kwargs
                )
            except Exception as exc:  # pragma: no cover - only when factors missing
                out["preconditioned"] = None
                out["preconditioned_error"] = repr(exc)
        if with_components:
            out["components"] = self.component_densities(
                components=components, factors=factors, history=history, m=m, **kwargs
            )
            if with_preconditioned:
                out["components_preconditioned"] = self.component_densities(
                    preconditioned=True, components=components,
                    factors=factors, history=history, m=m, **kwargs
                )
        return out


# --------------------------------------------------------------------------- #
# light self-test (run as a module: python -m src.spectral.spectral_density)
# --------------------------------------------------------------------------- #
def _self_test(seed: int = 0) -> None:  # pragma: no cover - exercised manually
    torch.manual_seed(seed)
    n = 100
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    true_eigs = torch.cat([
        torch.tensor([5e3, 1e3, 100.0]),  # outliers
        torch.linspace(1e-2, 1.0, n - 3),
    ])
    A = Q @ torch.diag(true_eigs) @ Q.T
    A = 0.5 * (A + A.T)

    res = slq_density(A, n=n, n_vec=3, n_iter=60, n_grid=200, top_k=3)
    ritz_max = float(res.eigenvalues.max())
    assert abs(ritz_max - 5e3) / 5e3 < 0.05, ritz_max
    assert res.density.min() >= 0.0
    assert res.eigenvalues_top is not None and float(res.eigenvalues_top[0]) > 1e3
    cond = estimate_condition_number(A, n=n, n_iter=60)
    assert cond > 1e3, cond
    print(f"[spectral_density] ok | ritz_max={ritz_max:.3e} cond={cond:.3e}")
    return True


if __name__ == "__main__":  # pragma: no cover
    _self_test()
