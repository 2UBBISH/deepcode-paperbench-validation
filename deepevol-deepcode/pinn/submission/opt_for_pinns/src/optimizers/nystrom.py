"""Randomized Nyström approximation (Algorithm 5) and NyströmPCG (Algorithm 6).

These two subroutines of NysNewton-CG (NNCG, Algorithm 4) are taken from the
"Challenges in Training PINNs: A Loss Landscape Perspective" paper (Appendix E.2),
which follows Frangella et al. (2023).

Algorithm 5 -- RandomizedNyströmApproximation(M, s)
---------------------------------------------------
Returns the top-``s`` approximate eigenvectors/values of the symmetric operator
``M`` (here the PINN Hessian ``H_L(w)``, accessed only through Hessian-vector
products).  Elements:

    S      = randn(n, s)                       # test matrix
    Y      = M S                               # via Hessian-vector products
    Q, _   = qr_econ(Y)
    nu     = sqrt(p) * eps * ||Y||_2
    Y_nu   = Y + nu Q
    B      = chol(Q^T Y_nu)                    # fail-safe if this fails:
                                               #   eig-decompose sym(Q^T Y_nu),
                                               #   shift nu so that nu >= |lam_min|
    B      = Uhat Sigma Vhat^T                 # thin SVD
    Lambda_hat = max(0, Sigma^2 - (nu + |lam|))

Algorithm 6 -- NyströmPCG(A, b, x_0, U, Lambda_hat, s, mu, epsilon, M)
----------------------------------------------------------------------
Preconditioned CG solving ``(A + mu I) x = b`` warm-started at ``x_0`` with the
Nyström preconditioner

    P      = 1/(lambda_hat_s + mu) U (Lambda_hat + mu I) U^T + (I - U U^T)
    P^{-1} = (lambda_hat_s + mu) U (Lambda_hat + mu I)^{-1} U^T + (I - U U^T)

with ``lambda_hat_s = Lambda_hat[-1]`` the smallest eigenvalue estimate.  The
matrix ``A`` is only ever accessed through a matrix-vector product (HVP).

The module works on *flat* parameter vectors (1-D tensors), matching the
interface of :mod:`opt_for_pinns.src.spectral.hvp` and
:mod:`opt_for_pinns.src.optimizers.armijo`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    # Algorithm 5
    "RandomizedNystromInfo",
    "randomized_nystrom_approximation",
    "RandomizedNystromApproximation",
    # Algorithm 6
    "NystromPreconditioner",
    "NystromPCGInfo",
    "nystrom_pcg",
    "NystromPCG",
    # helpers
    "as_matvec",
    "matvec_apply",
]

MatVec = Callable[[Tensor], Tensor]
MatOrMatVec = Union[Tensor, MatVec]

_DEFAULT_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def as_matvec(A: MatOrMatVec) -> MatVec:
    """Turn a matrix or a callable into a matrix-vector-product callable.

    A callable is returned unchanged (it must accept ``(n,)`` or ``(n, k)``
    tensors and return the same shape).  A square tensor ``A`` becomes the
    callable ``lambda v: A @ v``.
    """
    if callable(A):
        return A
    if not torch.is_tensor(A):
        raise TypeError(f"expected a Tensor or a callable matvec, got {type(A)!r}")
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {tuple(A.shape)}")

    def _mv(v: Tensor) -> Tensor:
        return A @ v

    return _mv


def matvec_apply(A: MatOrMatVec, V: Tensor) -> Tensor:
    """Apply ``A`` to one vector or to a block of columns (``(n,)`` or ``(n, k)``)."""
    if callable(A):
        out = A(V)
    else:
        out = A @ V
    if not torch.is_tensor(out):
        out = torch.as_tensor(out, dtype=V.dtype, device=V.device)
    return out


def _machine_eps(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return float(torch.finfo(dtype).eps)
    return 1e-16


def _norm2(Y: Tensor) -> Tensor:
    """Spectral norm ``||Y||_2`` (largest singular value)."""
    try:
        return torch.linalg.matrix_norm(Y, ord=2)
    except Exception:  # pragma: no cover - fallback for exotic dtypes
        return torch.linalg.svdvals(Y).max()


def _safe_cholesky(C: Tensor, jitter: float = 0.0, max_tries: int = 12) -> Optional[Tensor]:
    """Upper-triangular Cholesky factor of ``C`` or ``None`` if it fails.

    A small increasing jitter is added to the diagonal on retries.
    """
    if jitter:
        C = C + jitter * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    scale = max(float(C.diagonal().abs().max()), 1.0) if C.numel() else 1.0
    eps = _machine_eps(C.dtype)
    j = 0.0
    for _ in range(max_tries):
        try:
            try:
                return torch.linalg.cholesky(C + j * scale * torch.eye(
                    C.shape[0], dtype=C.dtype, device=C.device))
            except Exception:
                # some CUDA/old torch versions need a sync'ed error check
                pass
            L = torch.linalg.cholesky_ex(
                C + j * scale * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
            )
            if int(L.info) == 0:
                return L.L
        except Exception:
            pass
        j = max(10 * max(j, eps), eps * 10)
    return None


# ---------------------------------------------------------------------------
# Algorithm 5: RandomizedNyströmApproximation
# ---------------------------------------------------------------------------
@dataclass
class RandomizedNystromInfo:
    """Diagnostics produced by :func:`randomized_nystrom_approximation`."""

    n: int = 0
    s: int = 0
    nu: float = 0.0
    lam_min: float = 0.0
    used_failsafe: bool = False
    cholesky_ok: bool = True

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": int(self.n),
            "s": int(self.s),
            "nu": float(self.nu),
            "lam_min": float(self.lam_min),
            "used_failsafe": bool(self.used_failsafe),
            "cholesky_ok": bool(self.cholesky_ok),
        }


def randomized_nystrom_approximation(
    A: MatOrMatVec,
    s: int,
    *,
    n: Optional[int] = None,
    p: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    device: Optional[Union[str, torch.device]] = None,
    eps: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    return_info: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, RandomizedNystromInfo]]:
    """Algorithm 5: top-``s`` randomized Nyström approximation of ``A``.

    Parameters
    ----------
    A:
        Symmetric linear operator: either a square tensor or a callable matvec
        (e.g. a Hessian-vector-product closure).  ``A`` must accept ``(n,)`` and
        ``(n, p)`` inputs.
    s:
        Sketch size / number of approximate eigenpairs returned.
    n:
        Dimension of the operator.  Required when ``A`` is a *callable*; inferred
        from ``A`` otherwise.
    p:
        Number of columns of the test matrix (defaults to ``s``).
    rng / generator:
        Optional ``torch.Generator`` used for the Gaussian test matrix.

    Returns
    -------
    ``(U, Lambda_hat)`` with ``U`` of shape ``(n, s)`` (orthonormal columns,
    approximate eigenvectors) and ``Lambda_hat`` of shape ``(s,)``
    (non-negative, approximate eigenvalues, descending).  With
    ``return_info=True`` a third :class:`RandomizedNystromInfo` is returned.
    """
    if generator is not None and rng is None:
        rng = generator

    if not callable(A):
        if not torch.is_tensor(A):
            raise TypeError(f"expected a Tensor or a callable matvec, got {type(A)!r}")
        n = int(A.shape[0])
        dtype = A.dtype if A.is_floating_point() else dtype
        device = A.device if device is None else device
    if n is None:
        raise ValueError("`n` must be provided when `A` is a callable matvec")
    n = int(n)
    s = int(min(s, n))
    if p is None:
        p = s
    p = int(max(s, min(p, n)))

    if device is None:
        device = "cpu"
    device = torch.device(device)

    if eps is None:
        eps = _machine_eps(dtype)

    info = RandomizedNystromInfo(n=n, s=s)

    # 1-2. test matrix S and sketch Y = A S (via Hessian-vector products)
    S = torch.randn(n, p, dtype=dtype, device=device, generator=rng)
    Y = matvec_apply(A, S)
    Y = _to_dtype(Y, dtype, device)

    # 3. economy QR of the sketch
    Q, _R = torch.linalg.qr(Y, mode="reduced")  # (n, p)

    # 4-5. regularisation nu from the sketch norm
    nu = math.sqrt(p) * float(eps) * float(_norm2(Y).detach())
    if not math.isfinite(nu) or nu <= 0.0:
        nu = float(eps)
    info.nu = nu
    Y_nu = Y + nu * Q

    # 6. Cholesky of the core, with the fail-safe for indefinite A
    lam_min = 0.0
    core = Q.transpose(0, 1) @ Y_nu
    core = 0.5 * (core + core.transpose(0, 1))  # symmetrise numerically
    B = _safe_cholesky(core, jitter=0.0)
    if B is None:
        # --- fail-safe (portion in red in the paper) -----------------------
        info.used_failsafe = True
        info.cholesky_ok = False
        eigvals = torch.linalg.eigvalsh(core)
        lam_min = float(eigvals.min().detach())
        nu = max(nu, abs(lam_min) * (1.0 + float(eps)))
        info.nu = nu
        core2 = Q.transpose(0, 1) @ Y + nu * (
            Q.transpose(0, 1) @ Q
        )  # = Q^T Y + nu I (Q orthonormal)
        core2 = 0.5 * (core2 + core2.transpose(0, 1))
        B = _safe_cholesky(core2, jitter=max(nu, float(eps)))
        if B is None:  # last resort: strong jitter
            scale = max(float(core2.diagonal().abs().max()), 1.0)
            B = torch.linalg.cholesky(
                core2 + (1e-8 * scale + max(nu, eps)) * torch.eye(
                    core2.shape[0], dtype=dtype, device=device
                )
            )
    else:
        info.cholesky_ok = True

    info.lam_min = lam_min

    # 7. thin SVD of the Cholesky factor
    Uhat, Sigma, VhatT = torch.linalg.svd(B, full_matrices=False)
    # B = Uhat Sigma Vhat^T ; approximate eigenvectors of A are Q @ Vhat
    U = Q @ VhatT.transpose(0, 1)  # (n, s)

    # 8. corrected eigenvalues, clamped to be non-negative
    shift = nu + abs(lam_min)
    Lambda_hat = torch.clamp(Sigma.pow(2) - shift, min=0.0)
    order = torch.argsort(Lambda_hat, descending=True)
    Lambda_hat = Lambda_hat[order]
    U = U[:, order]

    # keep orthonormality of the returned basis (QR absorbs the ill-conditioning)
    if U.shape[1] > 0:
        try:
            U, _ = torch.linalg.qr(U, mode="reduced")
        except Exception:  # pragma: no cover
            pass

    if return_info:
        return U, Lambda_hat, info
    return U, Lambda_hat


def _to_dtype(X: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    if X.dtype != dtype or X.device != device:
        X = X.to(dtype=dtype, device=device)
    return X


class RandomizedNystromApproximation:
    """Callable wrapper of Algorithm 5 (paper naming).

    ``RandomizedNystromApproximation(A, s)(...)`` is equivalent to calling
    :func:`randomized_nystrom_approximation` with the bound operator/size.
    """

    def __init__(self, A: MatOrMatVec, s: int, **kwargs):
        self.A = A
        self.s = int(s)
        self.kwargs = dict(kwargs)

    def __call__(self, A: Optional[MatOrMatVec] = None, s: Optional[int] = None, **kwargs):
        opts = dict(self.kwargs)
        opts.update(kwargs)
        return randomized_nystrom_approximation(
            self.A if A is None else A, self.s if s is None else s, **opts
        )

    __repr__ = lambda self: f"RandomizedNystromApproximation(s={self.s})"  # type: ignore


# ---------------------------------------------------------------------------
# Nyström preconditioner
# ---------------------------------------------------------------------------
@dataclass
class NystromPreconditioner:
    """Nyström PCG preconditioner ``P`` (and its inverse) for ``A + mu I``.

    ``U`` has orthonormal columns, ``Lambda_hat`` are the non-negative
    approximate eigenvalues and ``mu`` the damping parameter.  Following the
    paper,

        P      = 1/(lambda_hat_s + mu) U (Lambda_hat + mu I) U^T + (I - U U^T)
        P^{-1} = (lambda_hat_s + mu) U (Lambda_hat + mu I)^{-1} U^T + (I - U U^T)

    with ``lambda_hat_s = Lambda_hat[-1]`` (the smallest eigenvalue estimate).
    """

    U: Tensor
    Lambda_hat: Tensor
    mu: float = 1e-2
    floor: float = 0.0
    _dtype: torch.dtype = field(default=_DEFAULT_DTYPE, repr=False)

    def __post_init__(self) -> None:
        if self.U.dim() != 2:
            raise ValueError("U must be a 2-D tensor (n, s)")
        self.mu = float(self.mu)
        self.Lambda_hat = self.Lambda_hat.to(dtype=self.U.dtype, device=self.U.device)
        if self.Lambda_hat.dim() != 1:
            self.Lambda_hat = self.Lambda_hat.reshape(-1)
        if self.Lambda_hat.numel() != self.U.shape[1]:
            raise ValueError(
                f"Lambda_hat has {self.Lambda_hat.numel()} entries but U has "
                f"{self.U.shape[1]} columns"
            )
        self._dtype = self.U.dtype

    # -- convenience -------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.U.shape[0])

    @property
    def s(self) -> int:
        return int(self.U.shape[1])

    @property
    def lambda_hat_s(self) -> float:
        """Smallest approximate eigenvalue ``lambda_hat_s`` used by ``P``."""
        if self.s == 0:
            return 0.0
        return float(self.Lambda_hat.min().detach())

    @property
    def lambdas(self) -> Tensor:
        """``Lambda_hat + mu I`` clipped away from zero (for stable division)."""
        lam = self.Lambda_hat + self.mu
        tiny = torch.finfo(self._dtype).tiny
        return torch.where(lam.abs() < tiny, torch.full_like(lam, tiny), lam)

    # -- operations --------------------------------------------------------
    def apply(self, v: Tensor) -> Tensor:
        """Apply ``P`` to ``v`` (``(n,)`` or ``(n, k)``)."""
        coeff = 1.0 / (self.lambda_hat_s + self.mu)
        Utv = self.U.transpose(0, 1) @ v
        return coeff * (self.U @ (self.lambdas * Utv)) + (v - self.U @ Utv)

    def apply_inv(self, v: Tensor) -> Tensor:
        """Apply ``P^{-1}`` to ``v`` (``(n,)`` or ``(n, k)``)."""
        coeff = self.lambda_hat_s + self.mu
        Utv = self.U.transpose(0, 1) @ v
        return coeff * (self.U @ (Utv / self.lambdas)) + (v - self.U @ Utv)

    def mat(self) -> Tensor:
        """Dense ``P`` (only for debugging / small ``n``)."""
        eye = torch.eye(self.n, dtype=self._dtype, device=self.U.device)
        return self.apply(eye)

    def mat_inv(self) -> Tensor:
        """Dense ``P^{-1}`` (only for debugging / small ``n``)."""
        eye = torch.eye(self.n, dtype=self._dtype, device=self.U.device)
        return self.apply_inv(eye)

    __call__ = apply_inv

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "s": self.s,
            "mu": self.mu,
            "lambda_max": float(self.Lambda_hat.max().detach()) if self.s else 0.0,
            "lambda_min": self.lambda_hat_s,
        }


# ---------------------------------------------------------------------------
# Algorithm 6: NyströmPCG
# ---------------------------------------------------------------------------
@dataclass
class NystromPCGInfo:
    """Convergence diagnostics of :func:`nystrom_pcg`."""

    iterations: int = 0
    residual_norm: float = float("nan")
    r0_norm: float = float("nan")
    rel_residual: float = float("nan")
    converged: bool = False
    breakdown: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "iterations": int(self.iterations),
            "residual_norm": float(self.residual_norm),
            "r0_norm": float(self.r0_norm),
            "rel_residual": float(self.rel_residual),
            "converged": bool(self.converged),
            "breakdown": bool(self.breakdown),
        }


def nystrom_pcg(
    A: MatOrMatVec,
    b: Tensor,
    x0: Optional[Tensor] = None,
    U: Optional[Tensor] = None,
    Lambda_hat: Optional[Tensor] = None,
    s: Optional[int] = None,
    mu: float = 1e-2,
    epsilon: float = 1e-16,
    M: int = 1000,
    *,
    preconditioner: Optional[NystromPreconditioner] = None,
    shift: bool = True,
    return_info: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Union[Tensor, Tuple[Tensor, NystromPCGInfo]]:
    """Algorithm 6: Nyström-preconditioned CG for ``(A + mu I) x = b``.

    Parameters
    ----------
    A:
        Symmetric operator (tensor or matvec callable, e.g. an HVP closure).
    b:
        Right-hand side, typically ``grad L(w_k)``.  Shape ``(n,)`` or ``(n, 1)``.
    x0:
        Warm start, typically the previous Newton step ``d_{k-1}``; ``None`` or a
        zero vector means a cold start.
    U, Lambda_hat, s:
        Nyström preconditioner factors from Algorithm 5.
    mu:
        Damping parameter.
    epsilon:
        CG relative tolerance; the iteration stops when
        ``||r_i|| <= epsilon * ||r_0||`` (or after ``M`` iterations).
    M:
        Maximum number of CG iterations.

    Returns
    -------
    The approximate solution ``x`` (same shape as ``b``).  With
    ``return_info=True``, ``(x, NystromPCGInfo)``.
    """
    shape = b.shape
    b_flat = b.reshape(-1).to(dtype=(dtype or b.dtype))
    device = b_flat.device
    if dtype is None:
        dtype = b_flat.dtype
    b_flat = b_flat.to(dtype=dtype)
    n = b_flat.numel()

    # assemble the preconditioner -------------------------------------------------
    if preconditioner is None:
        if U is None or Lambda_hat is None:
            raise ValueError("either `preconditioner` or (U, Lambda_hat) must be given")
        if s is not None and int(s) != U.shape[1]:
            U = U[:, : int(s)]
            Lambda_hat = Lambda_hat[: int(s)]
        preconditioner = NystromPreconditioner(
            U=U.to(dtype=dtype, device=device),
            Lambda_hat=Lambda_hat.to(dtype=dtype, device=device),
            mu=float(mu),
        )
    else:
        mu = float(preconditioner.mu)

    matvec = as_matvec(A)

    def _A_mu(v: Tensor) -> Tensor:
        out = matvec(v.reshape(-1)).reshape(-1).to(dtype=dtype, device=device)
        if shift:
            out = out + mu * v
        return out

    # warm start -------------------------------------------------------------------
    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
    else:
        x = x0.reshape(-1).to(dtype=dtype, device=device).clone()
        if not torch.isfinite(x).all():
            x = torch.zeros(n, dtype=dtype, device=device)

    info = NystromPCGInfo()
    r = b_flat - _A_mu(x)
    r0_norm = float(r.norm().detach())
    info.r0_norm = r0_norm
    tol = float(epsilon) * max(r0_norm, torch.finfo(dtype).tiny)

    if r0_norm == 0.0 or r0_norm <= tol:
        info.iterations = 0
        info.residual_norm = r0_norm
        info.rel_residual = 0.0
        info.converged = True
        out = x.reshape(shape)
        return (out, info) if return_info else out

    z = preconditioner.apply_inv(r)
    p = z.clone()
    gamma = float((r @ z).detach())

    for i in range(int(M)):
        if not math.isfinite(gamma) or gamma <= 0.0:
            info.breakdown = True
            break
        Ap = _A_mu(p)
        denom = float((p @ Ap).detach())
        if not math.isfinite(denom) or denom <= 0.0:
            info.breakdown = True
            break
        alpha = gamma / denom
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = float(r.norm().detach())
        info.iterations = i + 1
        info.residual_norm = r_norm
        if r_norm <= tol:
            info.converged = True
            break
        z = preconditioner.apply_inv(r)
        gamma_new = float((r @ z).detach())
        if not math.isfinite(gamma_new):
            info.breakdown = True
            break
        beta = gamma_new / gamma
        p = z + beta * p
        gamma = gamma_new

    info.rel_residual = (
        info.residual_norm / info.r0_norm if info.r0_norm > 0 else float("nan")
    )
    if not info.converged and info.iterations >= int(M):
        info.converged = info.residual_norm <= tol

    out = x.reshape(shape)
    return (out, info) if return_info else out


def NystromPCG(*args, **kwargs):  # pragma: no cover - paper-name alias
    """Alias of :func:`nystrom_pcg` using the paper's capitalisation."""
    return nystrom_pcg(*args, **kwargs)


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    torch.manual_seed(0)
    dtype = torch.float64
    n = 200
    Qm, _ = torch.linalg.qr(torch.randn(n, n, dtype=dtype))
    eigs = torch.logspace(1, -6, n, dtype=dtype)
    A = Qm @ torch.diag(eigs) @ Qm.T
    s = 30
    U, lam, info = randomized_nystrom_approximation(A, s, return_info=True)
    print("top-eig approx :", float(lam[0]), "true:", float(eigs[0]))
    print("info           :", info.as_dict())

    mu = 1e-2
    pre = NystromPreconditioner(U, lam, mu=mu)
    b = torch.randn(n, dtype=dtype)
    x, pcg_info = nystrom_pcg(
        lambda v: A @ v, b, None, preconditioner=pre, epsilon=1e-16, M=1000,
        return_info=True,
    )
    exact = torch.linalg.solve(A + mu * torch.eye(n, dtype=dtype), b)
    print("nystrom pcg    :", pcg_info.as_dict())
    print("rel err        :", float((x - exact).norm() / exact.norm()))
    # indefinite operator exercises the fail-safe branch
    A_ind = A - 5.0 * torch.eye(n, dtype=dtype)
    _, _, info_ind = randomized_nystrom_approximation(A_ind, 10, return_info=True)
    print("indef failsafe :", info_ind.as_dict())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
