"""PINN loss and L2 relative error (L2RE) metric.

Implements Section 2.1 / 2.2 of "Challenges in Training PINNs: A Loss Landscape
Perspective".

The PINN loss is

    L(w) = (1/(2 n_res)) * sum_i ( D[u(x_r^i; w)] )^2
         + (1/(2 n_bc))  * sum_j ( B[u(x_b^j; w)] )^2

where ``D`` is the PDE residual operator and ``B`` collects the initial and
boundary condition residuals.  The L2 relative error is

    L2RE = sqrt( sum_i (y_i - y'_i)^2 / sum_i (y'_i)^2 )

evaluated on the full evaluation grid (interior + IC + BC points).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .pdes import PDE


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------
def residual_loss(pde: PDE, u_fn, x_res: torch.Tensor, t_res: torch.Tensor) -> torch.Tensor:
    """Mean-squared PDE residual term (1/(2 n_res)) * sum D[u]^2."""
    r = pde.residual_operator(u_fn, x_res, t_res)
    return 0.5 * torch.mean(r ** 2)


def ic_loss(pde: PDE, u_fn, x_ic: torch.Tensor, t_ic: torch.Tensor) -> torch.Tensor:
    """Mean-squared initial-condition residual term."""
    r = pde.ic_residual(u_fn, x_ic, t_ic)
    return 0.5 * torch.mean(r ** 2)


def bc_loss(pde: PDE, u_fn, x_bc: torch.Tensor, t_bc: torch.Tensor) -> torch.Tensor:
    """Mean-squared boundary-condition residual term."""
    r = pde.bc_residual(u_fn, x_bc, t_bc)
    return 0.5 * torch.mean(r ** 2)


def pinn_loss(
    pde: PDE,
    u_fn,
    x_res: torch.Tensor,
    t_res: torch.Tensor,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    return_components: bool = False,
):
    """Total PINN loss.

    Parameters
    ----------
    pde : PDE
        The PDE problem instance.
    u_fn : callable
        The model, called as ``u_fn(x, t)``.
    x_res, t_res : torch.Tensor
        Interior collocation points.
    x_ic, t_ic : torch.Tensor
        Initial-condition points.
    x_bc, t_bc : torch.Tensor
        Boundary-condition points.
    return_components : bool
        If True, also return a dict with the individual loss components.

    Returns
    -------
    loss : torch.Tensor
        Scalar total loss.
    components : dict (optional)
        ``{"residual": ..., "ic": ..., "bc": ...}``.
    """
    l_res = residual_loss(pde, u_fn, x_res, t_res)
    l_ic = ic_loss(pde, u_fn, x_ic, t_ic)
    l_bc = bc_loss(pde, u_fn, x_bc, t_bc)
    total = l_res + l_ic + l_bc
    if return_components:
        return total, {"residual": l_res, "ic": l_ic, "bc": l_bc}
    return total


# ---------------------------------------------------------------------------
# L2 relative error
# ---------------------------------------------------------------------------
def l2_relative_error(
    pde: PDE,
    u_fn,
    x_eval: torch.Tensor,
    t_eval: torch.Tensor,
    y_exact: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L2 relative error on the provided evaluation points.

    L2RE = sqrt( sum (y - y')^2 / sum y'^2 )
    """
    with torch.no_grad():
        y_pred = u_fn(x_eval, t_eval)
        if y_exact is None:
            y_exact = pde.exact_solution(x_eval, t_eval)
        num = torch.sum((y_pred - y_exact) ** 2)
        den = torch.sum(y_exact ** 2)
        return torch.sqrt(num / den)


def evaluate(
    pde: PDE,
    u_fn,
    x_res: torch.Tensor,
    t_res: torch.Tensor,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    x_eval: torch.Tensor,
    t_eval: torch.Tensor,
    y_eval: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute loss components, total loss, L2RE and gradient norm.

    Returns a dict of python floats suitable for logging.
    """
    total, comps = pinn_loss(
        pde, u_fn, x_res, t_res, x_ic, t_ic, x_bc, t_bc, return_components=True
    )
    l2re = l2_relative_error(pde, u_fn, x_eval, t_eval, y_eval)

    # gradient norm of the total loss w.r.t. model parameters
    params = [p for p in u_fn.parameters() if p.requires_grad]
    grads = torch.autograd.grad(total, params, retain_graph=False, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is not None:
            sq = sq + torch.sum(g ** 2)
    grad_norm = torch.sqrt(sq) if isinstance(sq, torch.Tensor) else torch.tensor(0.0)

    return {
        "loss": float(total.detach().cpu()),
        "residual": float(comps["residual"].detach().cpu()),
        "ic": float(comps["ic"].detach().cpu()),
        "bc": float(comps["bc"].detach().cpu()),
        "l2re": float(l2re.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
    }
