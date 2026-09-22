"""Evaluation metrics for PINNs.

Implements the metrics used throughout the paper:

* L2RE (Eq. 3): relative L2 error between the PINN prediction and the
  analytical solution, evaluated over the full evaluation grid (255x100)
  plus the initial-condition (257) and boundary-condition (101) points.
* Gradient norm ``||grad L(w)||_2`` of the (total) PINN loss.
* Condition number ``lambda_max / lambda_min`` of the Hessian of the loss.

The L2RE definition follows Eq. 3 of the paper::

    L2RE = sqrt( sum_i (y_i - y'_i)^2 / sum_i (y'_i)^2 )

where ``y_i`` is the PINN prediction and ``y'_i`` the exact solution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .data import make_eval_points
from .loss import compute_losses, pinn_loss


# ---------------------------------------------------------------------------
# L2 relative error (Eq. 3)
# ---------------------------------------------------------------------------
def l2_relative_error(
    model: nn.Module,
    pde,
    x: torch.Tensor,
    t: torch.Tensor,
) -> float:
    """Compute the L2 relative error (Eq. 3) on the given points.

    Parameters
    ----------
    model : nn.Module
        The PINN model mapping ``(x, t) -> u``.
    pde : object
        PDE object exposing an ``exact(x, t)`` method.
    x, t : torch.Tensor
        Evaluation points of shape ``(n, 1)``.

    Returns
    -------
    float
        The scalar L2RE value.
    """
    model.eval()
    with torch.no_grad():
        pred = model(x, t)
        exact = pde.exact(x, t)
        num = torch.sum((pred - exact) ** 2)
        den = torch.sum(exact ** 2)
        l2re = torch.sqrt(num / den)
    return float(l2re.item())


def l2_relative_error_from_data(
    model: nn.Module,
    pde,
    data,
) -> float:
    """Compute L2RE over the union of the eval grid, IC and BC points.

    This mirrors the paper's evaluation protocol: the error is measured on
    the full ``255 x 100`` grid together with the ``257`` initial-condition
    points and the ``101`` boundary points (per boundary).
    """
    x, t = make_eval_points(pde)
    device = next(model.parameters()).device
    x = x.to(device)
    t = t.to(device)
    return l2_relative_error(model, pde, x, t)


# ---------------------------------------------------------------------------
# Loss / gradient-norm helpers
# ---------------------------------------------------------------------------
def total_loss(
    model: nn.Module,
    pde,
    data,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
) -> float:
    """Return the scalar total PINN loss (Eq. 2) as a Python float."""
    loss = pinn_loss(model, pde, data, lambda_res, lambda_ic, lambda_bc)
    return float(loss.detach().item())


def loss_components(
    model: nn.Module,
    pde,
    data,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
) -> dict:
    """Return a dict of the per-component losses (residual, ic, bc, total)."""
    comps = compute_losses(model, pde, data, lambda_res, lambda_ic, lambda_bc)
    return {
        "total": float(comps.total.detach().item()),
        "residual": float(comps.residual.detach().item()),
        "ic": float(comps.ic.detach().item()),
        "bc": float(comps.bc.detach().item()),
    }


def gradient_norm(
    model: nn.Module,
    pde,
    data,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
) -> float:
    """Compute ``||grad L(w)||_2`` for the total PINN loss.

    The gradient is computed with autograd and the model's ``.grad`` buffers
    are left untouched (a fresh graph is built and discarded).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    loss = pinn_loss(model, pde, data, lambda_res, lambda_ic, lambda_bc)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    sq = sum(torch.sum(g ** 2) for g in grads)
    return float(torch.sqrt(sq).item())


# ---------------------------------------------------------------------------
# Condition number
# ---------------------------------------------------------------------------
def condition_number(
    hessian: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Condition number ``lambda_max / lambda_min`` of a (dense) Hessian.

    Parameters
    ----------
    hessian : torch.Tensor
        Square symmetric matrix of shape ``(p, p)``.
    eps : float
        Floor applied to ``|lambda_min|`` to avoid division by zero.

    Returns
    -------
    float
        The condition number.
    """
    evals = torch.linalg.eigvalsh(hessian.double())
    lam_max = float(evals.max().item())
    lam_min = float(evals.min().item())
    denom = max(abs(lam_min), eps)
    return abs(lam_max) / denom


def top_eigenvalue(hessian: torch.Tensor) -> float:
    """Largest eigenvalue of a symmetric matrix (used for Fig. 3/7 checks)."""
    evals = torch.linalg.eigvalsh(hessian.double())
    return float(evals.max().item())


def smallest_eigenvalue(hessian: torch.Tensor) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    evals = torch.linalg.eigvalsh(hessian.double())
    return float(evals.min().item())


# ---------------------------------------------------------------------------
# Convenience aggregate
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    pde,
    data,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
    compute_grad_norm: bool = True,
) -> dict:
    """Aggregate evaluation: loss, per-component losses, L2RE, gradient norm.

    Returns a dict with keys ``loss``, ``residual``, ``ic``, ``bc``, ``l2re``
    and (optionally) ``grad_norm``.
    """
    comps = loss_components(model, pde, data, lambda_res, lambda_ic, lambda_bc)
    out = {
        "loss": comps["total"],
        "residual": comps["residual"],
        "ic": comps["ic"],
        "bc": comps["bc"],
        "l2re": l2_relative_error_from_data(model, pde, data),
    }
    if compute_grad_norm:
        out["grad_norm"] = gradient_norm(
            model, pde, data, lambda_res, lambda_ic, lambda_bc
        )
    return out
