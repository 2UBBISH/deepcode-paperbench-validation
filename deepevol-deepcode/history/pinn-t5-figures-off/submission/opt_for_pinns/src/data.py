"""Data generation and sampling for PINN training and evaluation.

Implements Section 2.2 of the paper:

- Residual (collocation) points: 10000 points randomly sampled from the interior
  of a 255 x 100 grid.
- Initial condition (IC) points: 257 equally spaced points along x at t=0.
- Boundary condition (BC) points: 101 equally spaced points per boundary.

Grids:
- convection / reaction: x in [0, 2*pi], t in [0, 1]
- wave: x in [0, 1], t in [0, 1]

The evaluation grid used for the L2RE metric is the full 255 x 100 grid plus the
257 IC points and the 101 BC points per boundary (see ``src/metrics.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Grid constants (Section 2.2)
# ---------------------------------------------------------------------------
N_X_GRID = 255          # number of x grid points for the evaluation grid
N_T_GRID = 100          # number of t grid points for the evaluation grid
N_RES = 10000           # number of residual / collocation points
N_IC = 257              # number of initial condition points
N_BC = 101              # number of boundary condition points per boundary


@dataclass
class PINNData:
    """Container holding all tensors required to train / evaluate a PINN.

    Attributes
    ----------
    x_res, t_res : Tensor
        Residual (collocation) points, shape ``(N_RES, 1)``.
    x_ic, t_ic : Tensor
        Initial condition points, shape ``(N_IC, 1)``.
    x_bc, t_bc : Tensor
        Boundary condition points, shape ``(2 * N_BC, 1)`` (both boundaries
        stacked together).  For periodic problems the two boundaries are the
        left and right edges; for the wave problem they are the two Dirichlet
        edges.
    x_grid, t_grid : Tensor
        Full evaluation grid, shape ``(N_X_GRID * N_T_GRID, 1)``.
    """

    x_res: torch.Tensor
    t_res: torch.Tensor
    x_ic: torch.Tensor
    t_ic: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    x_grid: torch.Tensor
    t_grid: torch.Tensor

    def to(self, device: torch.device) -> "PINNData":
        """Move every tensor in the container to ``device``."""
        return PINNData(
            x_res=self.x_res.to(device),
            t_res=self.t_res.to(device),
            x_ic=self.x_ic.to(device),
            t_ic=self.t_ic.to(device),
            x_bc=self.x_bc.to(device),
            t_bc=self.t_bc.to(device),
            x_grid=self.x_grid.to(device),
            t_grid=self.t_grid.to(device),
        )


def _domain_bounds(pde) -> Tuple[float, float, float, float]:
    """Return ``(x_min, x_max, t_min, t_max)`` for a PDE object."""
    x_min, x_max, t_min, t_max = pde.domain
    return float(x_min), float(x_max), float(t_min), float(t_max)


def make_eval_grid(pde, n_x: int = N_X_GRID, n_t: int = N_T_GRID,
                   device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the full ``n_x`` x ``n_t`` evaluation grid.

    Returns flattened ``(x, t)`` tensors of shape ``(n_x * n_t, 1)``.
    """
    x_min, x_max, t_min, t_max = _domain_bounds(pde)
    x = torch.linspace(x_min, x_max, n_x, device=device)
    t = torch.linspace(t_min, t_max, n_t, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return xx.reshape(-1, 1), tt.reshape(-1, 1)


def sample_residual_points(pde, n_res: int = N_RES, seed: Optional[int] = None,
                           device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample ``n_res`` interior collocation points.

    Points are drawn uniformly from the interior of the domain (matching the
    "randomly sampled from the 255 x 100 grid interior" description).
    """
    x_min, x_max, t_min, t_max = _domain_bounds(pde)
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    x = torch.rand(n_res, 1, generator=gen) * (x_max - x_min) + x_min
    t = torch.rand(n_res, 1, generator=gen) * (t_max - t_min) + t_min
    return x.to(device), t.to(device)


def make_ic_points(pde, n_ic: int = N_IC,
                   device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equally spaced initial condition points at ``t = t_min``."""
    x_min, x_max, t_min, _ = _domain_bounds(pde)
    x = torch.linspace(x_min, x_max, n_ic, device=device).reshape(-1, 1)
    t = torch.full_like(x, t_min)
    return x, t


def make_bc_points(pde, n_bc: int = N_BC,
                   device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equally spaced boundary points on both spatial boundaries.

    Returns stacked ``(2 * n_bc, 1)`` tensors: the first ``n_bc`` entries lie on
    the left boundary ``x = x_min`` and the remaining ``n_bc`` on the right
    boundary ``x = x_max``.
    """
    x_min, x_max, t_min, t_max = _domain_bounds(pde)
    t = torch.linspace(t_min, t_max, n_bc, device=device).reshape(-1, 1)
    x_left = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)
    x = torch.cat([x_left, x_right], dim=0)
    t = torch.cat([t, t], dim=0)
    return x, t


def build_data(pde, seed: Optional[int] = None, n_res: int = N_RES,
               n_ic: int = N_IC, n_bc: int = N_BC,
               n_x_grid: int = N_X_GRID, n_t_grid: int = N_T_GRID,
               device: Optional[torch.device] = None) -> PINNData:
    """Build the complete :class:`PINNData` container for a given PDE."""
    x_res, t_res = sample_residual_points(pde, n_res=n_res, seed=seed, device=device)
    x_ic, t_ic = make_ic_points(pde, n_ic=n_ic, device=device)
    x_bc, t_bc = make_bc_points(pde, n_bc=n_bc, device=device)
    x_grid, t_grid = make_eval_grid(pde, n_x=n_x_grid, n_t=n_t_grid, device=device)
    return PINNData(
        x_res=x_res, t_res=t_res,
        x_ic=x_ic, t_ic=t_ic,
        x_bc=x_bc, t_bc=t_bc,
        x_grid=x_grid, t_grid=t_grid,
    )


def make_eval_points(pde, n_x: int = N_X_GRID, n_t: int = N_T_GRID,
                     n_ic: int = N_IC, n_bc: int = N_BC,
                     device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the full set of evaluation points used for the L2RE metric.

    This is the union of the ``n_x`` x ``n_t`` grid, the ``n_ic`` IC points and
    the ``2 * n_bc`` boundary points, as described in Section 2.2 / Eq. 3.
    """
    x_grid, t_grid = make_eval_grid(pde, n_x=n_x, n_t=n_t, device=device)
    x_ic, t_ic = make_ic_points(pde, n_ic=n_ic, device=device)
    x_bc, t_bc = make_bc_points(pde, n_bc=n_bc, device=device)
    x = torch.cat([x_grid, x_ic, x_bc], dim=0)
    t = torch.cat([t_grid, t_ic, t_bc], dim=0)
    return x, t
