"""Data sampling utilities for PINN training and evaluation.

Following Section 2.2 of "Challenges in Training PINNs: A Loss Landscape Perspective":

- Interior residual points: n_res = 10000 random points drawn from a 255 x 100
  interior grid (255 points in x, 100 points in t).
- Initial condition points: 257 equally spaced points along x at t = t_min.
- Boundary condition points: 101 equally spaced points along t per boundary side.
- Full-batch training: all sampled points are used at every optimization step.
- L2RE is evaluated on the *full* 255 x 100 interior grid + 257 IC points +
  101 BC points per boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .pdes import PDE


# ---------------------------------------------------------------------------
# Grid sizes (paper Section 2.2)
# ---------------------------------------------------------------------------
N_X_GRID = 255          # interior grid resolution in x
N_T_GRID = 100          # interior grid resolution in t
N_RES = 10000           # number of sampled residual (collocation) points
N_IC = 257              # number of initial-condition points
N_BC = 101              # number of boundary-condition points per side


@dataclass
class PINNData:
    """Container holding all tensors needed for PINN training / evaluation."""

    # Training (sampled) points
    x_res: torch.Tensor
    t_res: torch.Tensor
    x_ic: torch.Tensor
    t_ic: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor

    # Full evaluation grid (used for L2RE)
    x_eval: torch.Tensor
    t_eval: torch.Tensor
    y_eval: torch.Tensor

    # Full interior grid (used for spectral density / heatmaps)
    x_grid: torch.Tensor
    t_grid: torch.Tensor

    def to(self, device, dtype: Optional[torch.dtype] = None) -> "PINNData":
        """Move all tensors to a device (and optionally cast dtype)."""
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return PINNData(
            x_res=self.x_res.to(**kwargs),
            t_res=self.t_res.to(**kwargs),
            x_ic=self.x_ic.to(**kwargs),
            t_ic=self.t_ic.to(**kwargs),
            x_bc=self.x_bc.to(**kwargs),
            t_bc=self.t_bc.to(**kwargs),
            x_eval=self.x_eval.to(**kwargs),
            t_eval=self.t_eval.to(**kwargs),
            y_eval=self.y_eval.to(**kwargs),
            x_grid=self.x_grid.to(**kwargs),
            t_grid=self.t_grid.to(**kwargs),
        )


def _linspace(a: float, b: float, n: int, dtype=torch.float32) -> torch.Tensor:
    return torch.linspace(a, b, n, dtype=dtype)


def make_interior_grid(pde: PDE, dtype=torch.float32) -> tuple:
    """Return the full 255 x 100 interior grid as flattened (x, t) tensors.

    The grid spans the *open* domain (endpoints excluded for the interior grid
    since boundaries are handled separately).
    """
    x = _linspace(pde.x_min, pde.x_max, N_X_GRID + 2, dtype=dtype)[1:-1]
    t = _linspace(pde.t_min, pde.t_max, N_T_GRID + 2, dtype=dtype)[1:-1]
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return xx.reshape(-1), tt.reshape(-1)


def sample_residual_points(pde: PDE, n_res: int = N_RES, seed: int = 0,
                           dtype=torch.float32) -> tuple:
    """Sample n_res random points uniformly from the interior grid.

    Points are drawn (with replacement) from the 255 x 100 interior grid, as
    described in Section 2.2.
    """
    x_grid, t_grid = make_interior_grid(pde, dtype=dtype)
    n_grid = x_grid.shape[0]
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, n_grid, (n_res,), generator=g)
    return x_grid[idx], t_grid[idx]


def make_ic_points(pde: PDE, n_ic: int = N_IC, dtype=torch.float32) -> tuple:
    """257 equally spaced initial-condition points at t = t_min."""
    x = _linspace(pde.x_min, pde.x_max, n_ic, dtype=dtype)
    t = torch.full_like(x, float(pde.t_min))
    return x, t


def make_bc_points(pde: PDE, n_bc: int = N_BC, dtype=torch.float32) -> tuple:
    """101 equally spaced boundary points per boundary side.

    For periodic problems (convection, reaction) the two sides are x_min and
    x_max. For the wave problem (Dirichlet) the two sides are x = 0 and x = 1.
    """
    t = _linspace(pde.t_min, pde.t_max, n_bc, dtype=dtype)
    xs = []
    ts = []
    for side in range(pde.n_bc_sides):
        if side == 0:
            x_side = float(pde.x_min)
        else:
            x_side = float(pde.x_max)
        xs.append(torch.full_like(t, x_side))
        ts.append(t)
    return torch.cat(xs, dim=0), torch.cat(ts, dim=0)


def make_eval_points(pde: PDE, dtype=torch.float32) -> tuple:
    """Full evaluation set: 255x100 interior grid + 257 IC + 101 BC per side.

    Returns (x_eval, t_eval, y_eval) where y_eval is the analytic solution.
    """
    x_grid, t_grid = make_interior_grid(pde, dtype=dtype)
    x_ic, t_ic = make_ic_points(pde, dtype=dtype)
    x_bc, t_bc = make_bc_points(pde, dtype=dtype)

    x_eval = torch.cat([x_grid, x_ic, x_bc], dim=0)
    t_eval = torch.cat([t_grid, t_ic, t_bc], dim=0)
    y_eval = pde.exact_solution(x_eval, t_eval)
    return x_eval, t_eval, y_eval


def build_data(pde: PDE, seed: int = 0, n_res: int = N_RES,
               dtype=torch.float32) -> PINNData:
    """Build the complete PINNData bundle for a given PDE and sampling seed."""
    x_res, t_res = sample_residual_points(pde, n_res=n_res, seed=seed, dtype=dtype)
    x_ic, t_ic = make_ic_points(pde, dtype=dtype)
    x_bc, t_bc = make_bc_points(pde, dtype=dtype)
    x_eval, t_eval, y_eval = make_eval_points(pde, dtype=dtype)
    x_grid, t_grid = make_interior_grid(pde, dtype=dtype)

    return PINNData(
        x_res=x_res, t_res=t_res,
        x_ic=x_ic, t_ic=t_ic,
        x_bc=x_bc, t_bc=t_bc,
        x_eval=x_eval, t_eval=t_eval, y_eval=y_eval,
        x_grid=x_grid, t_grid=t_grid,
    )
