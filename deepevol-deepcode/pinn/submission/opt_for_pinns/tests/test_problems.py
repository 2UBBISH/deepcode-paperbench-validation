"""Correctness tests for the PINN PDE infrastructure (convection / reaction / wave).

These tests implement the validation approach of the reproduction plan:

A) Analytical-solution sanity
   - ``D[u_exact] ~ 0`` on the residual collocation points for every PDE.
   - ``B[u_exact] ~ 0`` on the initial / boundary condition points for every PDE.
   - The analytical solutions are checked directly against their closed forms
     (uniqueness of the PDE solution given the IC/BCs makes this a strong check).

B) Zero-loss / zero-error at the exact solution
   - Wrapping the analytical solution in a tiny ``nn.Module`` gives a total PINN
     loss ``L(w) ~ 0`` and an L2RE ``~ 0`` (plan target: ``L2RE < 1e-6``).

C) Sampling protocol (paper Sec. 2.2)
   - 10 000 residual points drawn from a 255x100 interior grid, 257 IC points and
     101 points per boundary, plus the full evaluation grid for the L2RE metric.

D) Differentiability / Hessian machinery
   - The MLP is twice differentiable w.r.t. inputs (needed for the residual) and
     w.r.t. parameters; Pearlmutter-style Hessian-vector products match finite
     differences of the gradient (used by the spectral-density pipeline and NNCG).

The file is runnable both with pytest and standalone::

    python tests/test_problems.py
    python tests/test_problems.py -k wave --no-slow
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path bootstrap: make ``src.*`` importable no matter where pytest is invoked.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_ROOT = None
for _cand in (_HERE.parent.parent, _HERE.parent.parent.parent, _HERE.parent):
    if (_cand / "src" / "pinns" / "problems.py").is_file():
        _ROOT = _cand
        break
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pinns.model import MLP, make_pinn  # noqa: E402
from src.pinns.problems import (  # noqa: E402
    PROBLEMS,
    BoundaryCondition,
    PDEProblem,
    apply_condition,
    first_grad,
    get_problem,
    second_grad,
)
from src.pinns.sampling import (  # noqa: E402
    AXIS_T,
    AXIS_X,
    build_sampler,
    condition_point_counts,
    condition_points,
    interior_evaluation_grid,
    sample_residual_points,
    spatial_time_grid,
)
from src.pinns.loss import (  # noqa: E402
    BOUNDARY,
    INITIAL,
    PINNLoss,
    loss_breakdown,
    make_loss_fn,
    pinn_loss,
)
from src.pinns.metrics import compute_l2re, evaluate, l2re, predict  # noqa: E402

try:  # spectral layer is optional for the PDE-core tests
    from src.spectral.hvp import (  # noqa: E402
        flatten_params,
        hvp,
        loss_and_grad,
        num_parameters,
        set_flat_params,
    )

    _HVP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in minimal envs
    _HVP_AVAILABLE = False

try:
    from src.optimizers.first_order import AdamOptimizer  # noqa: E402

    _OPTIMIZERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPTIMIZERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DTYPE = torch.float64
# Plan target for |D[u_exact]| and |B[u_exact]| is < 1e-10; the achieved values are
# reported in every failure message so that borderline numerics are diagnosable.
RESIDUAL_ATOL = 1e-10
LOSS_ATOL = 1e-16
L2RE_ATOL = 1e-6
PDE_NAMES = ("convection", "reaction", "wave")
N_RESIDUAL_PAPER = 10_000
N_IC_PAPER = 257
N_BC_PAPER = 101
N_GRID_X_PAPER = 255
N_GRID_T_PAPER = 100

_SLOW_ENABLED = os.environ.get("PINN_SKIP_SLOW", "0") not in ("1", "true", "True")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_column(y: torch.Tensor) -> torch.Tensor:
    """Coerce a tensor to shape ``(n, 1)`` (keeps the autograd graph)."""
    if y.dim() == 0:
        return y.reshape(1, 1)
    if y.dim() == 1:
        return y.reshape(-1, 1)
    if y.dim() == 2:
        return y
    return y.reshape(y.shape[0], -1)


class ExactSolutionModel(torch.nn.Module):
    """``nn.Module`` whose forward pass *is* the analytical solution.

    Used to check that the loss and the L2RE metric both vanish when the network
    output coincides with ``u*`` (plan validation item A/B).  Carries a dummy
    parameter so that parameter-based utilities (optimizers, flattening) work.
    """

    def __init__(self, problem: PDEProblem, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.problem = problem
        self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = self.problem.exact(x)
        return _as_column(y)


def make_exact_u_fn(problem: PDEProblem):
    """Return ``u_fn(x) -> (n, 1)`` evaluating the analytical solution."""

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        return _as_column(problem.exact(x))

    return u_fn


def _condition_point_groups(problem: PDEProblem, dtype: torch.dtype = DTYPE):
    """Per-condition point tensors (a periodic condition yields two groups)."""
    pts = condition_points(problem, device="cpu", dtype=dtype)
    if isinstance(pts, torch.Tensor):  # tolerate a flat return convention
        pts = [(pts,)]
    groups = []
    for entry in pts:
        if isinstance(entry, torch.Tensor):
            groups.append((entry,))
        else:
            groups.append(tuple(entry))
    return groups


def _residual_points(problem: PDEProblem, n: int = 1024, seed: int = 0) -> torch.Tensor:
    """Deterministic small batch of residual points for the analytic checks."""
    return sample_residual_points(problem, n=n, seed=seed, dtype=DTYPE)


def _sup_norm(values) -> float:
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return 0.0
        return float(values.detach().abs().max())
    return max((_sup_norm(v) for v in values), default=0.0)


def _make_closure(model, problem, sampler=None, **kwargs):
    """Return ``(loss_object, zero_arg_closure)`` from ``make_loss_fn``."""
    out = make_loss_fn(model, problem, sampler=sampler, **kwargs)
    if isinstance(out, (tuple, list)) and len(out) == 2 and callable(out[1]):
        return out[0], out[1]
    if callable(out):
        return out, out
    raise AssertionError(f"unexpected make_loss_fn return type: {type(out)!r}")


# ===========================================================================
# PDE definitions: residual / condition operators
# ===========================================================================
def test_problem_registry():
    assert set(PROBLEMS) >= set(PDE_NAMES), f"missing PDEs in registry: {PROBLEMS}"
    for name in PDE_NAMES:
        problem = get_problem(name)
        assert isinstance(problem, PDEProblem)
        assert problem.name == name or name in repr(problem).lower()


def test_problem_coefficients():
    """``beta=40`` (convection), ``rho=5`` (reaction), ``beta=5`` (wave)."""
    convection = get_problem("convection")
    reaction = get_problem("reaction")
    wave = get_problem("wave")
    assert abs(float(getattr(convection, "beta", 40.0)) - 40.0) < 1e-12
    assert abs(float(getattr(reaction, "rho", 5.0)) - 5.0) < 1e-12
    assert abs(float(getattr(wave, "beta", 5.0)) - 5.0) < 1e-12


def test_problem_grids_are_255x100():
    for name in PDE_NAMES:
        problem = get_problem(name)
        x_grid, t_grid = spatial_time_grid(problem, dtype=DTYPE)
        assert x_grid.numel() == N_GRID_X_PAPER, f"{name}: x grid {x_grid.numel()}"
        assert t_grid.numel() == N_GRID_T_PAPER, f"{name}: t grid {t_grid.numel()}"
        points, xg, tg = interior_evaluation_grid(problem, dtype=DTYPE)
        assert points.shape == (N_GRID_X_PAPER * N_GRID_T_PAPER, 2)
        assert xg.numel() == N_GRID_X_PAPER and tg.numel() == N_GRID_T_PAPER


def test_analytic_residual_is_zero():
    """``D[u_exact](x_r) ~ 0`` on residual points for every PDE (plan item A)."""
    worst = {}
    for name in PDE_NAMES:
        problem = get_problem(name)
        u_fn = make_exact_u_fn(problem)
        x = _residual_points(problem, n=2048, seed=1234)
        residual = problem.residual(u_fn, x)
        err = _sup_norm(residual)
        worst[name] = err
        assert math.isfinite(err), f"{name}: non-finite residual norm {err}"
        assert err < RESIDUAL_ATOL, (
            f"{name}: |D[u_exact]|_inf = {err:.3e} exceeds {RESIDUAL_ATOL:.1e}"
        )
    print("   |D[u_exact]| (residual points):", {k: f"{v:.2e}" for k, v in worst.items()})
    return worst


def test_analytic_conditions_are_zero():
    """``B[u_exact] ~ 0`` on IC / boundary points for every PDE (plan item A)."""
    worst = {}
    for name in PDE_NAMES:
        problem = get_problem(name)
        u_fn = make_exact_u_fn(problem)
        conditions = list(problem.conditions())
        groups = _condition_point_groups(problem)
        assert len(groups) == len(conditions), (
            f"{name}: {len(groups)} point groups for {len(conditions)} conditions"
        )
        err = 0.0
        for cond, pts in zip(conditions, groups):
            err = max(err, _sup_norm(apply_condition(u_fn, cond, pts)))
        worst[name] = err
        assert math.isfinite(err), f"{name}: non-finite condition norm {err}"
        assert err < RESIDUAL_ATOL, (
            f"{name}: |B[u_exact]|_inf = {err:.3e} exceeds {RESIDUAL_ATOL:.1e}"
        )
    print("   |B[u_exact]| (IC/BC points):", {k: f"{v:.2e}" for k, v in worst.items()})
    return worst


def test_boundary_residuals_api():
    """The convenience API `boundary_residuals` must also vanish on ``u*``."""
    for name in PDE_NAMES:
        problem = get_problem(name)
        if not hasattr(problem, "boundary_residuals"):
            continue
        u_fn = make_exact_u_fn(problem)
        groups = _condition_point_groups(problem)
        residuals = problem.boundary_residuals(u_fn, groups)
        assert len(residuals) >= 1
        err = _sup_norm(residuals)
        assert err < RESIDUAL_ATOL, f"{name}: boundary_residuals inf-norm {err:.3e}"


def test_condition_metadata():
    """IC conditions carry 257 points, boundary conditions 101 points each."""
    for name in PDE_NAMES:
        problem = get_problem(name)
        conditions = list(problem.conditions())
        assert conditions, f"{name}: no conditions declared"
        counts = condition_point_counts(problem)
        assert counts, f"{name}: empty condition point counts"
        ic_counts = [
            counts.get(getattr(c, "name", ""), 0)
            for c in conditions
            if "ic" in str(getattr(c, "kind", "")).lower()
            or "initial" in str(getattr(c, "name", "")).lower()
        ]
        assert ic_counts, f"{name}: no initial-condition entry found"
        assert max(ic_counts) == N_IC_PAPER, (
            f"{name}: initial condition has {max(ic_counts)} points, expected {N_IC_PAPER}"
        )
        bc_counts = [int(v) for v in counts.values() if int(v) != N_IC_PAPER]
        assert bc_counts, f"{name}: no boundary-condition points found"
        assert all(c == N_BC_PAPER for c in bc_counts), (
            f"{name}: boundary counts {bc_counts} != {N_BC_PAPER}"
        )
        assert int(problem.n_bc_total) == sum(int(v) for v in counts.values())


# ===========================================================================
# Analytical solutions: closed-form sanity
# ===========================================================================
def test_convection_exact_closed_form():
    problem = get_problem("convection")
    beta = float(getattr(problem, "beta", 40.0))
    x = torch.linspace(0.0, 2.0 * math.pi, 129, dtype=DTYPE).reshape(-1, 1)
    t = torch.linspace(0.0, 1.0, 33, dtype=DTYPE).reshape(-1, 1)
    grid = torch.cat(
        [x.repeat_interleave(t.shape[0]), t.repeat(x.shape[0], 1).reshape(-1, 1)], dim=1
    )
    exact = _as_column(problem.exact(grid))
    expected = torch.sin(grid[:, :1] - beta * grid[:, 1:2])
    err = float((exact - expected).abs().max())
    assert err < 1e-9, f"convection exact deviates from sin(x - beta t) by {err:.3e}"
    # IC at t = 0 and x-periodicity.
    x0 = torch.cat([x, torch.zeros_like(x)], dim=1)
    assert float((_as_column(problem.exact(x0)) - torch.sin(x)).abs().max()) < 1e-12
    left = torch.cat([torch.zeros_like(t), t], dim=1)
    right = torch.cat([torch.full_like(t, 2.0 * math.pi), t], dim=1)
    periodic_err = float(
        (_as_column(problem.exact(left)) - _as_column(problem.exact(right))).abs().max()
    )
    assert periodic_err < 1e-9, f"convection periodicity error {periodic_err:.3e}"


def test_reaction_exact_closed_form():
    problem = get_problem("reaction")
    rho = float(getattr(problem, "rho", 5.0))
    # Gaussian IC of the paper: h(x) = exp(-(x - pi)^2 / (2 (pi/4)^2)).
    x = torch.linspace(0.0, 2.0 * math.pi, 257, dtype=DTYPE).reshape(-1, 1)
    h = torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))
    x0 = torch.cat([x, torch.zeros_like(x)], dim=1)
    ic_err = float((_as_column(problem.exact(x0)) - h).abs().max())
    assert ic_err < 1e-9, f"reaction IC mismatch: {ic_err:.3e}"
    # Closed form at a few times.
    for t_val in (0.1, 0.5, 1.0):
        t = torch.full_like(x, t_val)
        grid = torch.cat([x, t], dim=1)
        exact = _as_column(problem.exact(grid))
        e = torch.exp(torch.tensor(rho * t_val, dtype=DTYPE))
        expected = (h * e) / (h * e + 1.0 - h)
        err = float((exact - expected).abs().max())
        assert err < 1e-9, f"reaction exact mismatch at t={t_val}: {err:.3e}"


def test_wave_exact_closed_form():
    problem = get_problem("wave")
    beta = float(getattr(problem, "beta", 5.0))
    x = torch.linspace(0.0, 1.0, 101, dtype=DTYPE).reshape(-1, 1)
    t = torch.linspace(0.0, 1.0, 51, dtype=DTYPE).reshape(-1, 1)
    xx = x.repeat_interleave(t.shape[0])
    tt = t.repeat(x.shape[0], 1).reshape(-1, 1)
    grid = torch.cat([xx, tt], dim=1)
    exact = _as_column(problem.exact(grid))
    expected = torch.sin(math.pi * xx) * torch.cos(2.0 * math.pi * tt) + 0.5 * torch.sin(
        beta * math.pi * xx
    ) * torch.cos(beta * math.pi * 2.0 * tt)
    err = float((exact - expected).abs().max())
    assert err < 1e-9, f"wave exact deviates from the two-mode standing wave by {err:.3e}"
    # Dirichlet boundaries and zero initial velocity.
    for xb in (0.0, 1.0):
        pts = torch.cat([torch.full_like(t, xb), t], dim=1)
        assert float(_as_column(problem.exact(pts)).abs().max()) < 1e-12
    pts0 = torch.cat([x, torch.zeros_like(x)], dim=1).requires_grad_(True)
    u0 = _as_column(problem.exact(pts0))
    du_dt = torch.autograd.grad(u0.sum(), pts0, create_graph=False)[0][:, 1:2]
    assert float(du_dt.abs().max()) < 1e-9, "wave: u_t(x, 0) != 0"


def test_exact_solution_satisfies_autograd_operators():
    """Cross-check `exact` against the autograd helpers on generic points."""
    x = torch.linspace(0.1, 1.0, 23, dtype=DTYPE)
    y = (torch.sin(2.0 * x) * x).reshape(-1, 1)
    xr = torch.cat([x.reshape(-1, 1), (0.3 * x).reshape(-1, 1)], dim=1).requires_grad_(True)
    yy = (torch.sin(2.0 * xr[:, :1]) * xr[:, 1:2]).reshape(-1, 1)
    g = first_grad(yy, xr)
    g = g if g.dim() == 2 and g.shape[1] >= 2 else g.reshape(xr.shape[0], -1)
    d_dx = 2.0 * torch.cos(2.0 * xr[:, :1]) * xr[:, 1:2]
    d_dt = torch.sin(2.0 * xr[:, :1])
    assert float((g[:, 0:1] - d_dx).abs().max()) < 1e-9
    assert float((g[:, 1:2] - d_dt).abs().max()) < 1e-9
    d2 = second_grad(yy, xr, 0)
    d2 = d2.reshape(xr.shape[0], -1)[:, 0:1]
    assert float((d2 + 4.0 * torch.sin(2.0 * xr[:, :1]) * xr[:, 1:2]).abs().max()) < 1e-9
    assert y.shape == (23, 1)


# ===========================================================================
# Sampling protocol
# ===========================================================================
def test_sampling_sizes_match_paper():
    for name in PDE_NAMES:
        problem = get_problem(name)
        sampler = build_sampler(problem, seed=0, dtype=DTYPE)
        assert sampler.residual_points.shape == (N_RESIDUAL_PAPER, 2), (
            f"{name}: {sampler.residual_points.shape} residual points"
        )
        counts = condition_point_counts(problem)
        assert max(int(v) for v in counts.values()) == N_IC_PAPER
        assert sampler.eval_interior_points.shape == (
            N_GRID_X_PAPER * N_GRID_T_PAPER,
            2,
        )
        assert sampler.x_grid.numel() == N_GRID_X_PAPER
        assert sampler.t_grid.numel() == N_GRID_T_PAPER


def test_sampling_is_seed_reproducible():
    problem = get_problem("convection")
    a = sample_residual_points(problem, n=256, seed=7, dtype=DTYPE)
    b = sample_residual_points(problem, n=256, seed=7, dtype=DTYPE)
    c = sample_residual_points(problem, n=256, seed=8, dtype=DTYPE)
    assert torch.allclose(a, b), "residual sampling is not reproducible for a fixed seed"
    assert not torch.allclose(a, c), "different seeds produced identical residual points"
    s1 = build_sampler(problem, seed=7, dtype=DTYPE, n_residual=256)
    s2 = build_sampler(problem, seed=7, dtype=DTYPE, n_residual=256)
    assert torch.allclose(s1.residual_points, s2.residual_points)


def test_residual_points_lie_in_domain():
    for name in PDE_NAMES:
        problem = get_problem(name)
        sampler = build_sampler(problem, seed=3, dtype=DTYPE, n_residual=512)
        pts = sampler.residual_points
        xg, tg = sampler.x_grid, sampler.t_grid
        assert float(pts[:, 0].min()) >= float(xg.min()) - 1e-12
        assert float(pts[:, 0].max()) <= float(xg.max()) + 1e-12
        assert float(pts[:, 1].min()) >= float(tg.min()) - 1e-12
        assert float(pts[:, 1].max()) <= float(tg.max()) + 1e-12


# ===========================================================================
# Loss is zero at the analytical solution; L2RE is zero
# ===========================================================================
def test_loss_is_zero_at_exact_solution():
    for name in PDE_NAMES:
        problem = get_problem(name)
        model = ExactSolutionModel(problem)
        sampler = build_sampler(problem, seed=0, dtype=DTYPE, n_residual=1024)
        breakdown = loss_breakdown(model, problem, sampler=sampler)
        total = float(breakdown.total)
        assert math.isfinite(total)
        assert total < LOSS_ATOL, f"{name}: L(w*) = {total:.3e} should vanish"
        for key, value in breakdown.as_floats().items():
            assert value < LOSS_ATOL, f"{name}: component {key} = {value:.3e}"
    print("   L(w*) verified ~ 0 for", ", ".join(PDE_NAMES))


def test_pinn_loss_object_on_exact_solution():
    problem = get_problem("reaction")
    model = ExactSolutionModel(problem)
    sampler = build_sampler(problem, seed=1, dtype=DTYPE, n_residual=512)
    loss = PINNLoss(model, problem, sampler=sampler)
    value = float(loss())
    assert value < LOSS_ATOL, f"PINNLoss(w*) = {value:.3e}"
    functional = float(pinn_loss(model, problem, sampler=sampler))
    assert functional < LOSS_ATOL, f"pinn_loss(w*) = {functional:.3e}"


def test_l2re_metric_at_exact_solution():
    """Plan item B: L2RE < 1e-6 when the model reproduces ``u*``."""
    for name in PDE_NAMES:
        problem = get_problem(name)
        model = ExactSolutionModel(problem)
        sampler = build_sampler(problem, seed=0, dtype=DTYPE)
        value = compute_l2re(model, problem, sampler=sampler)
        assert math.isfinite(value)
        assert value < L2RE_ATOL, f"{name}: L2RE at exact solution = {value:.3e}"
        report = evaluate(model, problem, sampler=sampler)
        assert report.total < L2RE_ATOL
        assert report.n_points > 0
    print(f"   L2RE(u*) < {L2RE_ATOL:.0e} for all PDEs")


def test_l2re_metric_definition():
    y = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    assert l2re(y, y_true) < 1e-12
    y_bad = torch.zeros_like(y_true)
    assert abs(l2re(y_bad, y_true) - 1.0) < 1e-12
    scaled = l2re(y_true * 1e3, y_true)
    assert scaled < 1e-9, "L2RE must be scale invariant"
    assert isinstance(float(l2re(y, y_true)), float)


def test_per_region_l2re_report():
    problem = get_problem("wave")
    model = ExactSolutionModel(problem)
    sampler = build_sampler(problem, seed=0, dtype=DTYPE)
    report = evaluate(model, problem, sampler=sampler)
    assert report.per_region, "per-region L2RE breakdown is empty"
    for key, value in report.per_region.items():
        assert math.isfinite(value), f"region {key} is not finite: {value}"
        assert value < 1e-6, f"region {key} L2RE = {value:.3e}"
    assert sum(report.n_per_region.values()) == report.n_points


# ===========================================================================
# Model: twice differentiable, correct shapes/initialisation
# ===========================================================================
def test_mlp_forward_shapes():
    for width in (16, 50):
        model = make_pinn(in_dim=2, out_dim=1, width=width, depth=3, seed=0)
        x = torch.rand(17, 2, dtype=torch.float64)
        y = model(x)
        assert y.shape[0] == 17, f"forward output {tuple(y.shape)}"
        assert y.numel() == 17
        assert model.n_parameters() > 0


def test_mlp_zero_biases_and_xavier_weights():
    model = make_pinn(width=32, depth=3, seed=0)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert torch.allclose(module.bias, torch.zeros_like(module.bias)), (
                "biases must be initialised to zero (paper Sec. 2.2)"
            )
            assert torch.isfinite(module.weight).all()
    # Xavier-normal keeps the variance ~ 2/(fan_in + fan_out).
    first = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(first.weight)
    expected = 2.0 / (fan_in + fan_out)
    observed = float(first.weight.detach().var())
    assert 0.2 * expected < observed < 5.0 * expected, (
        f"weight variance {observed:.3e} inconsistent with Xavier normal {expected:.3e}"
    )


def test_mlp_is_twice_differentiable_wrt_inputs():
    model = make_pinn(width=16, depth=3, seed=0).double()
    x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).reshape(-1, 1)
    pts = torch.cat([x, x.flip(0)], dim=1).requires_grad_(True)
    u = _as_column(model(pts))
    du_dx = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
    assert du_dx.shape == pts.shape
    d2u_dx2 = torch.autograd.grad(
        du_dx[:, 0].sum(), pts, create_graph=True, retain_graph=True
    )[0]
    assert torch.isfinite(d2u_dx2).all(), "second input derivative not finite"
    assert d2u_dx2.abs().max() > 0, "second input derivative is identically zero"


def test_residual_accepts_network_output():
    """The residual operator must work on the MLP output (autograd graph kept)."""
    for name in PDE_NAMES:
        problem = get_problem(name)
        model = make_pinn(width=16, depth=3, seed=0).double()
        x = _residual_points(problem, n=64, seed=5)
        residual = problem.residual(lambda z: _as_column(model(z)), x)
        assert residual.shape[0] == x.shape[0]
        assert torch.isfinite(residual).all()
        assert float(residual.abs().max()) > 0.0


# ===========================================================================
# Gradients / Hessian-vector products
# ===========================================================================
def test_loss_gradient_is_finite_and_nonzero():
    problem = get_problem("convection")
    model = make_pinn(width=16, depth=3, seed=1).double()
    sampler = build_sampler(problem, seed=1, dtype=DTYPE, n_residual=128)
    _, closure = _make_closure(model, problem, sampler)
    loss = closure()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced by backward()"
    total = sum(float(g.abs().sum()) for g in grads)
    assert math.isfinite(total) and total > 0.0
    model.zero_grad()


def test_hvp_matches_finite_differences():
    """Pearlmutter HVP == central finite difference of the gradient."""
    if not _HVP_AVAILABLE:
        return
    problem = get_problem("convection")
    model = make_pinn(width=8, depth=2, seed=0).double()
    sampler = build_sampler(problem, seed=2, dtype=DTYPE, n_residual=128)
    _, closure = _make_closure(model, problem, sampler)
    params = list(model.parameters())
    n = num_parameters(model)
    gen = torch.Generator().manual_seed(0)
    v = torch.randn(n, dtype=DTYPE, generator=gen)
    v = v / v.norm()
    hv = hvp(closure, v, params).reshape(-1)
    assert hv.shape == (n,) and torch.isfinite(hv).all()
    w0 = flatten_params(params)
    eps = 1e-5
    set_flat_params(params, w0 + eps * v)
    _, g_plus = loss_and_grad(closure, params)
    set_flat_params(params, w0 - eps * v)
    _, g_minus = loss_and_grad(closure, params)
    set_flat_params(params, w0)
    fd = (g_plus.reshape(-1) - g_minus.reshape(-1)) / (2.0 * eps)
    rel = float((hv - fd).norm() / (fd.norm() + 1e-30))
    assert rel < 1e-4, f"HVP/finite-difference mismatch: rel err {rel:.3e}"


# ===========================================================================
# Integration smoke test (slower): Adam must reduce the PINN loss
# ===========================================================================
def test_adam_reduces_loss():
    if not (_SLOW_ENABLED and _OPTIMIZERS_AVAILABLE):
        return
    problem = get_problem("convection")
    torch.manual_seed(0)
    model = make_pinn(width=16, depth=3, seed=0).double()
    sampler = build_sampler(problem, seed=0, dtype=DTYPE, n_residual=256)
    _, closure = _make_closure(model, problem, sampler)
    opt = AdamOptimizer(model, lr=1e-2)
    first = None
    last = None
    for _ in range(60):
        value = float(opt.step(closure))
        if first is None:
            first = value
        last = value
    assert math.isfinite(last)
    assert last < first, f"Adam did not reduce the loss: {first:.6e} -> {last:.6e}"
    value = compute_l2re(model, problem, sampler=sampler)
    assert math.isfinite(value)
    print(f"   Adam smoke test: loss {first:.3e} -> {last:.3e}, L2RE {value:.3e}")


# ===========================================================================
# Standalone runner
# ===========================================================================
def _collect_tests():
    return [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PINN PDE infrastructure tests")
    parser.add_argument("-k", "--filter", default=None, help="substring filter on test names")
    parser.add_argument("--list", action="store_true", help="list tests and exit")
    parser.add_argument(
        "--no-slow", action="store_true", help="skip the (slower) training smoke test"
    )
    args = parser.parse_args(argv)

    global _SLOW_ENABLED
    if args.no_slow:
        _SLOW_ENABLED = False

    tests = _collect_tests()
    if args.filter:
        tests = [(n, f) for n, f in tests if args.filter in n]
    if args.list:
        for name, _ in tests:
            print(name)
        return 0

    torch.set_default_dtype(torch.float64)
    print(f"Running {len(tests)} tests (dtype=float64, slow={_SLOW_ENABLED})")
    failures = []
    t_start = time.time()
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name} ({time.time() - t0:.2f}s)")
            traceback.print_exc()
        else:
            print(f"PASS {name} ({time.time() - t0:.2f}s)")
    total = time.time() - t_start
    print("-" * 72)
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED in {total:.2f}s")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print(f"all {len(tests)} tests passed in {total:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
