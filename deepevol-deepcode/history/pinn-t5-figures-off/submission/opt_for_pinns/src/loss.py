"""PINN loss definition (Eq. 2 of the paper).

The PINN loss is

    L(w) = (1 / (2 n_res)) sum_i ( D[u(x_r^i; w), x_r^i] )^2
         + (1 / (2 n_bc))  sum_j ( B[u(x_b^j; w), x_b^j] )^2

where ``D`` is the differential operator of the PDE (the residual) and ``B``
collects the boundary / initial condition operators.  The loss is zero if and
only if the network interpolates the PDE and its boundary/initial conditions at
the training points.

This module exposes both the *total* loss and the *per-component* losses
(residual, initial condition, boundary condition).  The per-component losses are
required for the spectral-density analysis of Figures 3 (bottom) and 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .data import PINNData


@dataclass
class LossComponents:
    """Container for the individual loss terms.

    Attributes
    ----------
    total:
        The full PINN loss ``L(w)`` (Eq. 2).
    residual:
        Mean-squared PDE residual term (first sum in Eq. 2).
    ic:
        Mean-squared initial-condition term.
    bc:
        Mean-squared boundary-condition term.
    """

    total: torch.Tensor
    residual: torch.Tensor
    ic: torch.Tensor
    bc: torch.Tensor


def _mse(values: torch.Tensor) -> torch.Tensor:
    """Mean of the squared values (the ``1/(2n) sum`` factor up to a constant).

    We use ``0.5 * mean(values**2)`` which matches the ``1/(2 n) sum`` form in
    Eq. 2 exactly.
    """

    return 0.5 * torch.mean(values ** 2)


def residual_loss(model: nn.Module, pde, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """First term of Eq. 2: the PDE residual loss."""

    u = model(x, t)
    r = pde.residual(u, x, t)
    return _mse(r)


def initial_loss(model: nn.Module, pde, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Initial-condition part of the boundary term of Eq. 2."""

    u = model(x, t)
    r = pde.initial_residual(u, x, t)
    loss = _mse(r)
    # Wave equation additionally has an initial-velocity condition du/dt(x,0)=0.
    if hasattr(pde, "initial_velocity_residual"):
        r_v = pde.initial_velocity_residual(u, x, t)
        loss = loss + _mse(r_v)
    return loss


def boundary_loss(model: nn.Module, pde, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Boundary-condition part of the boundary term of Eq. 2."""

    u = model(x, t)
    r = pde.boundary_residual(u, x, t)
    return _mse(r)


def compute_losses(
    model: nn.Module,
    pde,
    data: PINNData,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
) -> LossComponents:
    """Compute the total loss and its individual components.

    Parameters
    ----------
    model:
        The PINN network mapping ``(x, t) -> u``.
    pde:
        A PDE object exposing ``residual``, ``initial_residual`` and
        ``boundary_residual``.
    data:
        A :class:`~src.data.PINNData` container with the training points.
    lambda_res, lambda_ic, lambda_bc:
        Optional weights for each term (default 1.0, matching the paper).

    Returns
    -------
    LossComponents
        The total loss and the residual / IC / BC components.
    """

    res = residual_loss(model, pde, data.x_res, data.t_res)
    ic = initial_loss(model, pde, data.x_ic, data.t_ic)
    bc = boundary_loss(model, pde, data.x_bc, data.t_bc)

    total = lambda_res * res + lambda_ic * ic + lambda_bc * bc
    return LossComponents(total=total, residual=res, ic=ic, bc=bc)


def pinn_loss(
    model: nn.Module,
    pde,
    data: PINNData,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
) -> torch.Tensor:
    """Convenience wrapper returning only the total PINN loss (Eq. 2)."""

    return compute_losses(
        model, pde, data, lambda_res, lambda_ic, lambda_bc
    ).total


def make_loss_fn(
    model: nn.Module,
    pde,
    data: PINNData,
    lambda_res: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
):
    """Return a zero-argument closure computing the total loss.

    This is the form expected by PyTorch optimizers (``L-BFGS``) and by the
    Armijo line search (Algorithm 7), which need a loss oracle.
    """

    def loss_fn() -> torch.Tensor:
        return pinn_loss(model, pde, data, lambda_res, lambda_ic, lambda_bc)

    return loss_fn


def component_loss_fns(model: nn.Module, pde, data: PINNData):
    """Return zero-argument closures for each individual loss component.

    Used by the spectral-density analysis (Figures 3 bottom, 7) which studies
    the Hessian of each component separately.
    """

    def res_fn() -> torch.Tensor:
        return residual_loss(model, pde, data.x_res, data.t_res)

    def ic_fn() -> torch.Tensor:
        return initial_loss(model, pde, data.x_ic, data.t_ic)

    def bc_fn() -> torch.Tensor:
        return boundary_loss(model, pde, data.x_bc, data.t_bc)

    return {"residual": res_fn, "ic": ic_fn, "bc": bc_fn}
