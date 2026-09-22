"""Sampling of training points (Section 2.2).

"We use 10000 residual points randomly sampled from a 255 x 100 grid on the
interior of the problem domain.  We use 257 equally spaced points for the
initial conditions and 101 equally spaced points for each boundary condition."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .problems import Problem

N_RESIDUAL = 10000


@dataclass
class DataSet:
    """Collocation points of a single training run."""

    X_res: torch.Tensor
    X_ic: Dict[str, torch.Tensor] = field(default_factory=dict)
    X_bc: Dict[str, torch.Tensor] = field(default_factory=dict)
    # points used to report the L2RE (interior grid + IC points + BC points)
    X_eval: torch.Tensor = None  # type: ignore[assignment]


def sample_residual_points(
    problem: Problem,
    n_res: int = N_RESIDUAL,
    seed: int = 0,
    grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Randomly sample ``n_res`` points from the interior 255 x 100 grid.

    Sampling is done without replacement when enough grid points exist (this
    is the situation of every experiment in the paper) and with replacement
    otherwise.
    """
    grid = problem.interior_grid() if grid is None else grid
    n_grid = grid.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    if n_res <= n_grid:
        idx = torch.randperm(n_grid, generator=gen)[:n_res]
    else:
        idx = torch.randint(0, n_grid, (n_res,), generator=gen)
    return grid[idx].clone()


def build_dataset(
    problem: Problem,
    n_res: int = N_RESIDUAL,
    seed: int = 0,
    grid: torch.Tensor | None = None,
) -> DataSet:
    gen = torch.Generator().manual_seed(int(seed) + 991)
    ds = DataSet(X_res=sample_residual_points(problem, n_res=n_res, seed=seed, grid=grid))
    for term in problem.ic_terms():
        ds.X_ic[term.name] = term.sample(gen, term.n_points)
    for term in problem.bc_terms():
        ds.X_bc[term.name] = term.sample(gen, term.n_points)
    ds.X_eval = problem.evaluation_points()
    return ds


def dataset_eval_points(problem: Problem) -> torch.Tensor:
    """Points used for the L2RE, as a function of the problem only."""
    return problem.evaluation_points()
