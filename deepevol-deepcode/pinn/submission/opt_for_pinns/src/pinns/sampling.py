"""Collocation / initial / boundary point sampling for the PINN benchmarks.

Implements the sampling protocol of §2.2 of "Challenges in Training PINNs: A Loss
Landscape Perspective" (ICML 2024):

    "We use 10000 residual points randomly sampled from a 255 x 100 grid on the
     interior of the problem domain. We use 257 equally spaced points for the
     initial conditions and 101 equally spaced points for each boundary
     condition."

Additionally, this module provides the *full* evaluation grid used by the L2RE
metric: "We compute the L2RE using all points in the 255 x 100 grid on the
interior of the problem domain, along with the 257 and 101 points used for the
initial and boundary conditions."

Coordinate convention
---------------------
Every point tensor has shape ``(n, 2)`` with column 0 = spatial coordinate ``x``
and column 1 = time coordinate ``t`` (all three benchmarks in the paper have a
2-D space-time input, see §2.1 and Appendix A).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .problems import BoundaryCondition, PDEProblem, get_problem

__all__ = [
    "AXIS_X",
    "AXIS_T",
    "SamplingConfig",
    "PINNSampler",
    "domain_bounds",
    "spatial_time_grid",
    "interior_evaluation_grid",
    "condition_points",
    "condition_points_flat",
    "condition_point_counts",
    "sample_residual_points",
    "full_evaluation_points",
    "build_sampler",
    "sampler_from_config",
]

# ---------------------------------------------------------------------------
# axis bookkeeping
# ---------------------------------------------------------------------------
AXIS_X = 0  # spatial coordinate -> first column of a point tensor
AXIS_T = 1  # time coordinate    -> second column of a point tensor

_X_ALIASES = {"x", "0", "space", "spatial", "col0", "first"}
_T_ALIASES = {"t", "1", "time", "temporal", "col1", "second"}


def _as_axis_index(axis) -> int:
    """Map a problem's ``axis`` descriptor onto a column index (0 = x, 1 = t)."""
    if isinstance(axis, int):
        return axis if axis in (0, 1) else (0 if axis == 0 else 1)
    name = str(axis).strip().lower()
    if name in _T_ALIASES:
        return AXIS_T
    if name in _X_ALIASES:
        return AXIS_X
    # last resort: "t" anywhere in the name means time
    if "t" in name and "x" not in name:
        return AXIS_T
    return AXIS_X


def _free_axis(fixed_axis: int) -> int:
    return AXIS_T if fixed_axis == AXIS_X else AXIS_X


# ---------------------------------------------------------------------------
# domain bounds
# ---------------------------------------------------------------------------
def _coerce_range(value) -> Optional[Tuple[float, float]]:
    try:
        lo, hi = value
        return (float(lo), float(hi))
    except Exception:
        return None


def domain_bounds(problem: PDEProblem, device="cpu", dtype=None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the spatial and temporal domain bounds ``((x0, x1), (t0, t1))``.

    The bounds are read from ``problem.domain`` when possible; otherwise they are
    inferred from the problem's own interior grid helper (the first grid is the
    spatial one, the second the temporal one).
    """
    device = torch.device(device)
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    x_range: Optional[Tuple[float, float]] = None
    t_range: Optional[Tuple[float, float]] = None

    dom = getattr(problem, "domain", None)
    if dom is not None:
        if isinstance(dom, dict):
            for key, val in dom.items():
                rng = _coerce_range(val)
                if rng is None:
                    continue
                if _as_axis_index(key) == AXIS_T:
                    t_range = rng
                else:
                    x_range = rng
        elif isinstance(dom, (tuple, list)):
            if len(dom) == 2 and all(_coerce_range(v) is not None for v in dom):
                x_range = _coerce_range(dom[0])
                t_range = _coerce_range(dom[1])
            elif len(dom) == 4:
                x_range = _coerce_range((dom[0], dom[1]))
                t_range = _coerce_range((dom[2], dom[3]))
        else:
            for attr, target in (("x", "x"), ("t", "t"), ("space", "x"), ("time", "t")):
                if hasattr(dom, attr):
                    rng = _coerce_range(getattr(dom, attr))
                    if rng is not None:
                        if target == "x":
                            x_range = rng
                        else:
                            t_range = rng

    if x_range is None or t_range is None:
        gx, gt = problem.interior_grid_1d(device=device, dtype=dtype)
        gx = gx.detach().reshape(-1)
        gt = gt.detach().reshape(-1)
        if x_range is None:
            x_range = (float(gx[0]), float(gx[-1]))
        if t_range is None:
            t_range = (float(gt[0]), float(gt[-1]))

    return x_range, t_range


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
def spatial_time_grid(problem: PDEProblem, device="cpu", dtype=None) -> Tuple[Tensor, Tensor]:
    """The 255x100 interior grid: ``(x_grid (255,), t_grid (100,))``."""
    device = torch.device(device)
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    x, t = problem.interior_grid_1d(device=device, dtype=dtype)
    return x.reshape(-1).to(device=device, dtype=dtype), t.reshape(-1).to(device=device, dtype=dtype)


def interior_evaluation_grid(
    problem: PDEProblem, device="cpu", dtype=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """Flattened mesh of the full 255x100 interior grid.

    Returns ``(points, x_grid, t_grid)`` where ``points`` has shape
    ``(n_x * n_t, 2)`` (column 0 = x, column 1 = t).
    """
    x, t = spatial_time_grid(problem, device=device, dtype=dtype)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    points = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=-1)
    return points, x, t


# ---------------------------------------------------------------------------
# boundary / initial condition points
# ---------------------------------------------------------------------------
def _points_for_condition(
    problem: PDEProblem,
    cond: BoundaryCondition,
    device="cpu",
    dtype=None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[Tensor, ...]:
    """Build the point tensor(s) associated with a single condition.

    ``kind == "periodic"`` yields two tensors (the two ends of the periodic
    axis); every other kind yields a single tensor.  Each tensor has shape
    ``(cond.n_points, 2)`` with equally spaced points along the free axis.
    """
    device = torch.device(device)
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if bounds is None:
        bounds = domain_bounds(problem, device=device, dtype=dtype)

    fixed_axis = _as_axis_index(getattr(cond, "axis", AXIS_T))
    free_axis = _free_axis(fixed_axis)
    lo, hi = bounds[free_axis]

    n_points = int(getattr(cond, "n_points", problem.n_bc))
    n_points = max(n_points, 1)
    free = torch.linspace(lo, hi, n_points, device=device, dtype=dtype)

    kind = str(getattr(cond, "kind", "value")).strip().lower()
    if kind == "periodic":
        value0 = float(getattr(cond, "value", lo))
        value2 = getattr(cond, "value2", None)
        value1 = float(value2) if value2 is not None else float(hi)
    else:
        value0 = float(getattr(cond, "value", lo))
        value1 = value0

    def _make(value: float) -> Tensor:
        fixed = torch.full_like(free, value)
        if fixed_axis == AXIS_X:
            return torch.stack([fixed, free], dim=-1)
        return torch.stack([free, fixed], dim=-1)

    if kind == "periodic":
        return (_make(value0), _make(value1))
    return (_make(value0),)


def condition_points(
    problem: PDEProblem,
    device="cpu",
    dtype=None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> List[Tuple[Tensor, ...]]:
    """Per-condition point tensors, aligned with ``problem.conditions()`` order."""
    if bounds is None:
        bounds = domain_bounds(problem, device=device, dtype=dtype)
    out: List[Tuple[Tensor, ...]] = []
    for cond in problem.conditions():
        out.append(_points_for_condition(problem, cond, device=device, dtype=dtype, bounds=bounds))
    return out


def condition_points_flat(
    problem: PDEProblem, device="cpu", dtype=None
) -> Tuple[Tensor, List[int]]:
    """All condition points concatenated into one ``(N, 2)`` tensor.

    Returns ``(points, sizes)`` where ``sizes[i]`` is the number of points coming
    from the i-th condition (in the order of ``problem.conditions()``).
    """
    groups = condition_points(problem, device=device, dtype=dtype)
    flat: List[Tensor] = []
    sizes: List[int] = []
    for group in groups:
        for pts in group:
            pts = pts.reshape(-1, 2)
            flat.append(pts)
            sizes.append(int(pts.shape[0]))
    if not flat:
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        return torch.empty((0, 2), device=device, dtype=dtype), []
    return torch.cat(flat, dim=0), sizes


def condition_point_counts(problem: PDEProblem) -> Dict[str, int]:
    """Number of points used by each condition (residual excluded)."""
    counts: Dict[str, int] = {}
    for cond in problem.conditions():
        name = str(getattr(cond, "name", "cond"))
        n = int(getattr(cond, "n_points", 0))
        kind = str(getattr(cond, "kind", "value")).strip().lower()
        counts[name] = counts.get(name, 0) + (2 * n if kind == "periodic" else n)
    return counts


# ---------------------------------------------------------------------------
# residual collocation sampling
# ---------------------------------------------------------------------------
def sample_residual_points(
    problem: PDEProblem,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    device="cpu",
    dtype=None,
    generator: Optional[torch.Generator] = None,
    replace: bool = True,
) -> Tensor:
    """Sample ``n`` (default 10000) residual points from the 255x100 interior grid.

    The grid is flattened into ``n_x * n_t`` candidate points and ``n`` of them are
    drawn uniformly at random (with replacement by default), reproducing §2.2's
    "10000 residual points randomly sampled from a 255 x 100 grid".
    """
    device = torch.device(device)
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if n is None:
        n = int(getattr(problem, "n_residual", 10000))
    n = int(n)

    x_grid, t_grid = spatial_time_grid(problem, device=device, dtype=dtype)
    nx, nt = int(x_grid.numel()), int(t_grid.numel())
    n_candidates = nx * nt
    if n_candidates == 0 or n == 0:
        return torch.empty((0, 2), device=device, dtype=dtype)

    if generator is None and seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    if replace or n > n_candidates:
        idx = torch.randint(0, n_candidates, (n,), generator=generator)
    else:
        idx = torch.randperm(n_candidates, generator=generator)[:n]
    idx = idx.to(device)

    ix = idx // nt
    it = idx % nt
    points = torch.stack([x_grid[ix], t_grid[it]], dim=-1).to(device=device, dtype=dtype)
    return points


# ---------------------------------------------------------------------------
# full evaluation set (interior grid + condition points)
# ---------------------------------------------------------------------------
def full_evaluation_points(
    problem: PDEProblem, device="cpu", dtype=None
) -> Tuple[Tensor, List[Tuple[Tensor, ...]]]:
    """The complete point set used for L2RE (§2.2).

    Returns ``(interior_points, condition_point_groups)`` where
    ``interior_points`` is the flattened 255x100 interior grid and the second
    element is the per-condition list of point tensors.
    """
    interior, _, _ = interior_evaluation_grid(problem, device=device, dtype=dtype)
    return interior, condition_points(problem, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    """Sampling hyper-parameters (defaults follow §2.2)."""

    n_residual: int = 10000
    n_ic: int = 257
    n_bc: int = 101
    n_grid_x: int = 255
    n_grid_t: int = 100
    replace: bool = True  # sample residual points with replacement

    @classmethod
    def from_dict(cls, cfg: Optional[dict] = None) -> "SamplingConfig":
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


@dataclass
class PINNSampler:
    """Fixed sampling of residual / initial / boundary points for one problem run."""

    problem: PDEProblem
    residual_points: Tensor
    condition_point_groups: List[Tuple[Tensor, ...]]
    eval_interior_points: Tensor
    eval_grid_x: Tensor
    eval_grid_t: Tensor
    seed: Optional[int] = None

    # -- convenient views ---------------------------------------------------
    @property
    def n_residual(self) -> int:
        return int(self.residual_points.shape[0])

    @property
    def x_grid(self) -> Tensor:
        return self.eval_grid_x

    @property
    def t_grid(self) -> Tensor:
        return self.eval_grid_t

    def condition_points_flat(self) -> Tuple[Tensor, List[int]]:
        flat: List[Tensor] = []
        sizes: List[int] = []
        for group in self.condition_point_groups:
            for pts in group:
                pts = pts.reshape(-1, 2)
                flat.append(pts)
                sizes.append(int(pts.shape[0]))
        if not flat:
            return torch.empty((0, 2), device=self.residual_points.device, dtype=self.residual_points.dtype), []
        return torch.cat(flat, dim=0), sizes

    @property
    def n_condition_points(self) -> int:
        return int(sum(int(pts.shape[0]) for group in self.condition_point_groups for pts in group))

    @property
    def n_eval_interior(self) -> int:
        return int(self.eval_interior_points.shape[0])

    def condition_summary(self) -> Dict[str, int]:
        return condition_point_counts(self.problem)

    # -- device / resampling ------------------------------------------------
    def to(self, device) -> "PINNSampler":
        device = torch.device(device)
        return PINNSampler(
            problem=self.problem,
            residual_points=self.residual_points.to(device),
            condition_point_groups=[
                tuple(pts.to(device) for pts in group) for group in self.condition_point_groups
            ],
            eval_interior_points=self.eval_interior_points.to(device),
            eval_grid_x=self.eval_grid_x.to(device),
            eval_grid_t=self.eval_grid_t.to(device),
            seed=self.seed,
        )

    def resample_residual(self, seed: Optional[int] = None, replace: Optional[bool] = None) -> "PINNSampler":
        """Redraw the residual collocation points (everything else stays fixed)."""
        new_seed = self.seed if seed is None else int(seed)
        new_points = sample_residual_points(
            self.problem,
            n=self.n_residual,
            seed=new_seed,
            device=self.residual_points.device,
            dtype=self.residual_points.dtype,
            replace=self.residual_points is not None if replace is None else replace,
        )
        return PINNSampler(
            problem=self.problem,
            residual_points=new_points,
            condition_point_groups=self.condition_point_groups,
            eval_interior_points=self.eval_interior_points,
            eval_grid_x=self.eval_grid_x,
            eval_grid_t=self.eval_grid_t,
            seed=new_seed,
        )

    def summary(self) -> str:
        parts = [
            f"residual={self.n_residual}",
            f"condition_points={self.n_condition_points}",
            f"eval_interior={self.n_eval_interior}",
        ]
        for name, cnt in self.condition_summary().items():
            parts.append(f"{name}={cnt}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_sampler(
    problem: PDEProblem,
    seed: Optional[int] = None,
    device="cpu",
    dtype=None,
    n_residual: Optional[int] = None,
    replace: bool = True,
    resample: bool = True,
) -> PINNSampler:
    """Build a :class:`PINNSampler` for ``problem``.

    ``seed`` controls the random draw of the 10000 residual points; when
    ``resample`` is False the first ``n_residual`` grid points are used instead
    (deterministic, useful for tests).
    """
    device = torch.device(device)
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if n_residual is None:
        n_residual = int(getattr(problem, "n_residual", 10000))

    x_range, t_range = domain_bounds(problem, device=device, dtype=dtype)
    if resample:
        residual = sample_residual_points(
            problem,
            n=n_residual,
            seed=seed,
            device=device,
            dtype=dtype,
            replace=replace,
        )
    else:
        x, t = spatial_time_grid(problem, device=device, dtype=dtype)
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        grid = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=-1)
        residual = grid[: int(n_residual)].to(device=device, dtype=dtype)

    groups = condition_points(problem, device=device, dtype=dtype, bounds=(x_range, t_range))
    interior, x_grid, t_grid = interior_evaluation_grid(problem, device=device, dtype=dtype)

    return PINNSampler(
        problem=problem,
        residual_points=residual,
        condition_point_groups=groups,
        eval_interior_points=interior,
        eval_grid_x=x_grid,
        eval_grid_t=t_grid,
        seed=seed,
    )


def sampler_from_config(
    problem: PDEProblem,
    config: Optional[dict] = None,
    seed: Optional[int] = None,
    device="cpu",
    dtype=None,
) -> PINNSampler:
    """Build a sampler from a (YAML-derived) config dict for ``problem``.

    Recognised keys (nested under the problem name or at the top level):
    ``n_residual``, ``n_ic``, ``n_bc``, ``n_grid_x``, ``n_grid_t``, ``replace``,
    ``seed``.
    """
    cfg = dict(config or {})
    name = getattr(problem, "name", None)
    merged: Dict[str, object] = {}
    for key in ("sampling", None):
        if key in cfg and isinstance(cfg[key], dict):
            merged.update(cfg[key])
    merged.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
    if name is not None:
        for key in ("problems", name):
            scope = cfg.get(key)
            if isinstance(scope, dict):
                if "sampling" in scope and isinstance(scope["sampling"], dict):
                    merged.update(scope["sampling"])
                merged.update({k: v for k, v in scope.items() if not isinstance(v, dict)})

    s_cfg = SamplingConfig.from_dict(merged)
    problem.n_residual = s_cfg.n_residual  # keep the problem's own bookkeeping in sync
    s_seed = merged.get("seed", seed)
    s_seed = None if s_seed is None else int(s_seed)
    return build_sampler(
        problem,
        seed=s_seed,
        device=device,
        dtype=dtype,
        n_residual=s_cfg.n_residual,
        replace=bool(s_cfg.replace),
        resample=bool(s_cfg.replace),
    )
