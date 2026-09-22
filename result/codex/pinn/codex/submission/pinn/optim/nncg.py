"""NysNewton-CG (NNCG), Algorithms 4-7 of Appendix E.2.

NNCG is a damped Newton method in which the Newton step is computed with
``NystroemPCG``, a preconditioned conjugate gradient method designed for
matrices with fast spectral decay (Frangella et al., 2023).  The Hessian is only
accessed through Hessian-vector products.

Default hyper-parameters are those of the paper:

``eta=1, K=2000, s=60, F=20, eps=1e-16, M=1000, alpha=0.1, beta=0.5`` and the
damping ``mu`` tuned over ``[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]`` (``1e-2`` and
``1e-1`` work best in practice).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..metrics import l2_relative_error
from .objective import Objective


# ---------------------------------------------------------------------- #
# Algorithm 5: Randomized Nystroem approximation
# ---------------------------------------------------------------------- #
def randomized_nystrom_approximation(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    s: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(U, Lambda_hat)``: the top-``s`` approximate eigenpairs of ``M``."""
    S = torch.randn(dim, s, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(S, mode="reduced")
    Y = torch.stack([matvec(Q[:, j]) for j in range(s)], dim=1).to(torch.float64)
    nu = (dim**0.5) * float(torch.finfo(torch.float64).eps) * float(torch.linalg.norm(Y, 2))
    Ynu = Y + nu * Q
    lam = 0.0
    QtY = Q.T @ Ynu
    try:
        # torch.linalg.cholesky returns the lower triangular factor C with
        #     C C^T = Q^T Y_nu .
        # Algorithm 5 of the paper writes B = Y C^{-1} using the *upper*
        # triangular factor (the MATLAB `chol` convention, C^T C = Q^T Y_nu),
        # so with PyTorch's lower factor the same object is B = Y C^{-T}:
        #     B B^T = Y (C^T C)^{-1} Y^T = Y (Q^T Y_nu)^{-1} Y^T ,
        # which is the Nystroem approximation of the (shifted) matrix.
        C = torch.linalg.cholesky(QtY)
        # solve C X = Y^T for X, then B = X^T
        B = torch.linalg.solve_triangular(C, Y.T.contiguous(), upper=False).T
    except Exception:
        # fail-safe for indefinite Hessians (the red text of Algorithm 5)
        Gamma, W = torch.linalg.eigh(QtY)
        lam = float(Gamma.min())
        R = W @ torch.diag((Gamma + abs(lam)).clamp_min(1e-300).rsqrt()) @ W.mT
        B = Y @ R
    Vhat, Sigma, _ = torch.linalg.svd(B, full_matrices=False)
    Lam = (Sigma**2 - (nu + abs(lam))).clamp_min(0.0)
    return Vhat, Lam


# ---------------------------------------------------------------------- #
# Algorithm 6: NystroemPCG (damped Newton step)
# ---------------------------------------------------------------------- #
def nystrom_pcg(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor,
    U: torch.Tensor,
    Lam: torch.Tensor,
    s: int,
    mu: float,
    eps: float = 1e-16,
    max_iter: int = 1000,
    rel_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """Solve ``(A + mu I) x = b`` with the Nystroem preconditioner of Eq. (5).

    The stopping rule is the paper's absolute tolerance ``eps`` combined with a
    hard cap of ``max_iter`` CG iterations.  ``rel_tol`` optionally adds a
    relative criterion (``||r|| <= rel_tol * ||b||``); it is not part of the
    paper and defaults to ``None``.
    """
    lam_s = float(Lam[-1]) if Lam.numel() > 0 else 0.0
    b_norm = float(torch.linalg.norm(b))

    def pinv(r: torch.Tensor) -> torch.Tensor:
        Ut_r = U.T @ r
        low = (lam_s + mu) * (U @ ((Lam + mu).clamp_min(1e-300).reciprocal() * Ut_r))
        return low + (r - U @ Ut_r)

    r0 = b - (matvec(x0) + mu * x0)
    z0 = pinv(r0)
    p0 = z0.clone()
    k = 0
    while k < max_iter:
        r_norm = float(torch.linalg.norm(r0))
        if r_norm < eps:
            break
        if rel_tol is not None and r_norm <= rel_tol * max(b_norm, 1e-300):
            break
        v = matvec(p0) + mu * p0
        denom = torch.dot(p0, v)
        if abs(float(denom)) < 1e-300:
            break
        alpha = torch.dot(r0, z0) / denom
        x = x0 + alpha * p0
        r = r0 - alpha * v
        z = pinv(r)
        beta = torch.dot(r, z) / torch.dot(r0, z0)
        x0, r0, p0, z0 = x, r, z + beta * p0, z
        k += 1
    return x0, k


# ---------------------------------------------------------------------- #
# Algorithm 7: Armijo backtracking line search
# ---------------------------------------------------------------------- #
def armijo(
    objective: Objective,
    w: torch.Tensor,
    grad: torch.Tensor,
    direction: torch.Tensor,
    t: float,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_backtracks: int = 100,
) -> Tuple[float, int]:
    f0, gtd = float(objective.loss().detach()), float(torch.dot(grad, direction))
    n_back = 0
    while n_back < max_backtracks:
        objective.set_flat_params(w + t * direction)
        f = float(objective.loss().detach())
        if f <= f0 + alpha * t * gtd:
            return t, n_back
        t *= beta
        n_back += 1
    objective.set_flat_params(w)
    return 0.0, n_back


# ---------------------------------------------------------------------- #
# Algorithm 4: NysNewton-CG
# ---------------------------------------------------------------------- #
@dataclass
class NNCGConfig:
    eta: float = 1.0
    iters: int = 2000
    sketch_size: int = 60
    preconditioner_frequency: int = 20
    mu: float = 1e-2
    cg_tol: float = 1e-16
    cg_max_iter: int = 1000
    cg_rel_tol: Optional[float] = None
    alpha: float = 0.1
    beta: float = 0.5
    log_every: int = 10
    seed: int = 0


@dataclass
class NNCGResult:
    final_loss: float
    final_l2re: float
    final_grad_norm: float
    iterations: int
    wall_clock: float
    trace: Dict[str, List[float]] = field(default_factory=dict)
    cg_iterations: List[int] = field(default_factory=list)

    @property
    def seconds_per_iteration(self) -> float:
        return self.wall_clock / max(self.iterations, 1)


class NNCG:
    def __init__(self, objective: Objective, config: NNCGConfig, problem=None):
        self.objective = objective
        self.config = config
        self.problem = problem if problem is not None else objective.problem
        self.generator = torch.Generator().manual_seed(int(config.seed))

    def run(self, progress: bool = False) -> NNCGResult:
        cfg = self.config
        obj = self.objective
        trace: Dict[str, List[float]] = {
            "iteration": [],
            "loss": [],
            "grad_norm": [],
            "l2re": [],
            "l2re_iteration": [],
        }
        cg_iters: List[int] = []
        U: Optional[torch.Tensor] = None
        Lam: Optional[torch.Tensor] = None
        d_prev = torch.zeros(obj.n_params, dtype=torch.float64)

        t0 = time.time()
        w = obj.flat_params().detach().clone()
        for k in range(cfg.iters):
            if k % cfg.preconditioner_frequency == 0:
                U, Lam = randomized_nystrom_approximation(
                    obj.hvp, obj.n_params, cfg.sketch_size, generator=self.generator
                )
            loss, grad = obj.loss_and_grad()
            d_k, n_cg = nystrom_pcg(
                obj.hvp,
                grad.to(torch.float64),
                d_prev,
                U,
                Lam,
                cfg.sketch_size,
                cfg.mu,
                eps=cfg.cg_tol,
                max_iter=cfg.cg_max_iter,
                rel_tol=cfg.cg_rel_tol,
            )
            cg_iters.append(n_cg)
            eta_k, _ = armijo(obj, w, grad.to(torch.float64), -d_k, cfg.eta, cfg.alpha, cfg.beta)
            w = w - eta_k * d_k
            obj.set_flat_params(w)
            d_prev = d_k

            trace["iteration"].append(k)
            trace["loss"].append(loss)
            trace["grad_norm"].append(float(torch.linalg.norm(grad)))
            if k % cfg.log_every == 0 or k == cfg.iters - 1:
                trace["l2re_iteration"].append(k)
                trace["l2re"].append(l2_relative_error(obj.net, self.problem, obj.ds.X_eval))
            if progress and (k + 1) % 50 == 0:
                print(
                    f"    nncg {k + 1}/{cfg.iters} loss={loss:.4e} "
                    f"cg={n_cg} eta={eta_k:.3f}"
                )
        wall = time.time() - t0
        return NNCGResult(
            final_loss=obj.evaluate(),
            final_l2re=l2_relative_error(obj.net, self.problem, obj.ds.X_eval),
            final_grad_norm=float(torch.linalg.norm(obj.grad())),
            iterations=cfg.iters,
            wall_clock=wall,
            trace=trace,
            cg_iterations=cg_iters,
        )


def nncg_finetune(objective: Objective, config: NNCGConfig, problem=None) -> NNCGResult:
    """Section 7.3: continue an Adam+L-BFGS run with NNCG."""
    return NNCG(objective, config, problem=problem).run()


# ---------------------------------------------------------------------- #
# Gradient-descent baseline of Section 7.3
# ---------------------------------------------------------------------- #
def gd_finetune(
    objective: Objective,
    iters: int = 2000,
    lr: float = 1e-3,
    log_every: int = 10,
    problem=None,
    progress: bool = False,
) -> NNCGResult:
    """Plain gradient descent after Adam+L-BFGS (Table 2 / Figure 4)."""
    problem = problem if problem is not None else objective.problem
    trace: Dict[str, List[float]] = {
        "iteration": [],
        "loss": [],
        "grad_norm": [],
        "l2re": [],
        "l2re_iteration": [],
    }
    w = objective.flat_params().detach().clone()
    t0 = time.time()
    for k in range(iters):
        loss, grad = objective.loss_and_grad()
        w = w - lr * grad.to(w.dtype)
        objective.set_flat_params(w)
        trace["iteration"].append(k)
        trace["loss"].append(loss)
        trace["grad_norm"].append(float(torch.linalg.norm(grad)))
        if k % log_every == 0 or k == iters - 1:
            trace["l2re_iteration"].append(k)
            trace["l2re"].append(l2_relative_error(objective.net, problem, objective.ds.X_eval))
        if progress and (k + 1) % 200 == 0:
            print(f"    gd {k + 1}/{iters} loss={loss:.4e}")
    wall = time.time() - t0
    return NNCGResult(
        final_loss=objective.evaluate(),
        final_l2re=l2_relative_error(objective.net, problem, objective.ds.X_eval),
        final_grad_norm=float(torch.linalg.norm(objective.grad())),
        iterations=iters,
        wall_clock=wall,
        trace=trace,
    )
