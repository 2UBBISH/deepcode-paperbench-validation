"""Gradient-Damped Newton Descent (Algorithm 1 of the paper).

Algorithm 1 is the *theoretical* algorithm analysed in Section 8: a phase of
gradient descent followed by damped Newton steps
``w <- w - eta (H_L(w) + gamma I)^{-1} grad L(w)``.  Section 8 is out of scope
for the reproduction, but the algorithm itself is simple to provide and is used
as a reference point for NNCG: the damped Newton system is solved here with
plain conjugate gradient on Hessian-vector products, exactly as in
Algorithm 1 where the inverse is written out explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .objective import Objective


def conjugate_gradient(
    matvec,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> torch.Tensor:
    """Solve ``A x = b`` for symmetric positive definite ``A`` given by ``matvec``."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = torch.dot(r, r)
    for _ in range(max_iter):
        if float(torch.sqrt(rs)) <= tol:
            break
        Ap = matvec(p)
        denom = float(torch.dot(p, Ap))
        if abs(denom) < 1e-300:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


@dataclass
class GDNDConfig:
    k_gd: int = 1000  # number of gradient-descent iterations (Phase I)
    eta_gd: float = 1e-3  # gradient-descent learning rate
    k_dn: int = 50  # number of damped-Newton iterations (Phase II)
    eta_dn: float = 5.0 / 6.0  # damped-Newton step size (Theorem 8.5)
    gamma: float = 1e-2  # damping parameter
    cg_tol: float = 1e-12
    cg_max_iter: int = 500
    log_every: int = 10


@dataclass
class GDNDResult:
    final_loss: float
    iterations: int
    wall_clock: float
    trace: Dict[str, List[float]] = field(default_factory=dict)


def gdnd(objective: Objective, config: GDNDConfig, problem=None) -> GDNDResult:
    """Run Algorithm 1 on ``objective``."""
    trace: Dict[str, List[float]] = {"iteration": [], "loss": [], "grad_norm": []}
    t0 = time.time()
    w = objective.flat_params().detach().clone()

    def log(k: int, loss: float, grad: torch.Tensor) -> None:
        trace["iteration"].append(k)
        trace["loss"].append(loss)
        trace["grad_norm"].append(float(torch.linalg.norm(grad)))

    # ---- Phase I: gradient descent ---------------------------------- #
    for k in range(config.k_gd):
        loss, grad = objective.loss_and_grad()
        w = w - config.eta_gd * grad
        objective.set_flat_params(w)
        if k % config.log_every == 0:
            log(k, loss, grad)

    # ---- Phase II: damped Newton ------------------------------------ #
    for k in range(config.k_dn):
        loss, grad = objective.loss_and_grad()
        g64 = grad.to(torch.float64)
        d = conjugate_gradient(
            lambda v: (objective.hvp(v) + config.gamma * v).to(torch.float64),
            g64,
            tol=config.cg_tol,
            max_iter=config.cg_max_iter,
        )
        w = w - config.eta_dn * d.to(w.dtype)
        objective.set_flat_params(w)
        log(config.k_gd + k, loss, grad)

    return GDNDResult(
        final_loss=objective.evaluate(),
        iterations=config.k_gd + config.k_dn,
        wall_clock=time.time() - t0,
        trace=trace,
    )
