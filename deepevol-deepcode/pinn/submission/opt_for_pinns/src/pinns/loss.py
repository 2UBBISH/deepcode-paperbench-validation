"""PINN training loss (Eq. (2) of the paper) and per-component losses.

The paper (Section 2.1) defines the non-linear least-squares problem

    minimize_w  L(w) := 1/(2 n_res) * sum_{i=1}^{n_res} ( D[u(x_r^i; w), x_r^i] )^2
                      + 1/(2 n_bc)  * sum_{i=1}^{n_bc}  ( B[u(x_b^i; w), x_b^i] )^2 ,

where D is the PDE residual operator, B collects the boundary/initial condition
operators, {x_r^i} are the residual collocation points and {x_b^i} the
boundary/initial points.

In addition to the total loss ``L`` this module exposes the three additive
"components" that the spectral-density study (Fig. 3 bottom / Fig. 7) analyses
separately:

  * ``residual`` : the PDE residual term,
  * ``initial``  : the initial-condition part of the B term,
  * ``boundary`` : the (spatial) boundary part of the B term,

Each component keeps the same ``1/(2 n)`` normalisation as Eq. (2), using the
number of points belonging to that component, so that components are directly
comparable in magnitude.  The total loss is exactly Eq. (2) (residual term +
sum over every condition group normalised by the total number of
boundary/initial points).

The loss is differentiable twice with respect to the weights (the residual is
built with ``create_graph=True`` in :mod:`src.pinns.problems`), so Hessian-vector
products / spectral density estimation can be built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .problems import BoundaryCondition, PDEProblem, apply_condition
from .sampling import (
    PINNSampler,
    build_sampler,
    condition_points as _condition_points,
)

__all__ = [
    "LossBreakdown",
    "PINNLoss",
    "pinn_loss",
    "loss_breakdown",
    "residual_values",
    "condition_values",
    "make_loss_fn",
    "classify_condition",
    "INITIAL",
    "BOUNDARY",
]

INITIAL = "initial"
BOUNDARY = "boundary"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_points_tuple(points) -> Tuple[Tensor, ...]:
    """Normalise a point-group into a tuple of tensors."""
    if isinstance(points, Tensor):
        return (points,)
    if isinstance(points, (list, tuple)):
        return tuple(p for p in points)
    raise TypeError(f"Unsupported point group type: {type(points)!r}")


def _half_mean_square(values: Tensor) -> Tensor:
    """``1/(2n) * sum_i values_i^2`` (Eq. (2) normalisation)."""
    flat = values.reshape(-1)
    n = flat.numel()
    if n == 0:
        return flat.new_zeros(())
    return 0.5 * torch.sum(flat ** 2) / n


def _sum_square(values: Tensor) -> Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return flat.new_zeros(())
    return torch.sum(flat ** 2)


def classify_condition(cond: BoundaryCondition) -> str:
    """Return ``"initial"`` or ``"boundary"`` for a condition.

    Initial conditions live on the temporal axis at the initial time (value
    conditions at ``t = t_0`` as well as time-derivative conditions); everything
    else (periodic or Dirichlet conditions on the spatial axis) is a boundary
    condition.
    """
    kind = str(getattr(cond, "kind", "")).lower()
    axis = str(getattr(cond, "axis", "")).lower()
    if "ic" in kind or kind in {"ic_derivative", "derivative"} and axis == "t":
        return INITIAL
    # any condition on the time axis with a temporal value/target is an IC
    if axis in {"t", "time"}:
        return INITIAL
    return BOUNDARY


# ---------------------------------------------------------------------------
# loss-value helpers (stateless variants, handy for tests / scripts)
# ---------------------------------------------------------------------------
def residual_values(model, problem: PDEProblem, points: Tensor) -> Tensor:
    """Raw residual values ``D[u(x; w), x]`` (flattened)."""
    return problem.residual(model, points).reshape(-1)


def condition_values(
    model,
    problem: PDEProblem,
    point_groups: Sequence[Sequence[Tensor]],
) -> List[Tensor]:
    """Raw condition values ``B[u(x; w), x]`` for every condition group."""
    conditions = problem.conditions()
    out: List[Tensor] = []
    for idx, cond in enumerate(conditions):
        pts = _as_points_tuple(point_groups[idx])
        out.append(apply_condition(model, cond, pts).reshape(-1))
    return out


def _condition_point_groups(
    problem: PDEProblem,
    sampler: Optional[PINNSampler],
    device,
    dtype,
) -> List[Tuple[Tensor, ...]]:
    """Resolve the boundary/initial point groups.

    Preference order: sampler bundle -> freshly generated condition points.
    """
    if sampler is not None:
        groups = getattr(sampler, "condition_point_groups", None)
        if groups:
            if hasattr(sampler, "to"):
                try:
                    sampler = sampler.to(device)
                    groups = sampler.condition_point_groups
                except Exception:  # pragma: no cover - defensive
                    pass
            return [_as_points_tuple(g) for g in groups]
    groups = _condition_points(problem, device=device, dtype=dtype)
    return [_as_points_tuple(g) for g in groups]


# ---------------------------------------------------------------------------
# loss container
# ---------------------------------------------------------------------------
@dataclass
class LossBreakdown:
    """Additive decomposition of the PINN loss."""

    total: Tensor
    residual: Tensor
    initial: Tensor
    boundary: Tensor
    conditions: Dict[str, Tensor] = field(default_factory=dict)
    n_residual: int = 0
    n_condition: int = 0
    condition_kinds: Dict[str, str] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    def component(self, name: str) -> Tensor:
        """Look up a component by name (``total``/``residual``/``initial``/``boundary``)."""
        name = name.lower()
        if name in {"total", "loss", "l"}:
            return self.total
        if name in {"residual", "res", "pde", "d"}:
            return self.residual
        if name in {"initial", "ic"}:
            return self.initial
        if name in {"boundary", "bc"}:
            return self.boundary
        if name in self.conditions:
            return self.conditions[name]
        raise KeyError(f"Unknown loss component: {name!r}")

    def as_floats(self) -> Dict[str, float]:
        out = {
            "loss": float(self.total.detach()),
            "residual": float(self.residual.detach()),
            "initial": float(self.initial.detach()),
            "boundary": float(self.boundary.detach()),
        }
        for k, v in self.conditions.items():
            out[f"cond/{k}"] = float(v.detach())
        return out

    def numpy_components(self) -> Dict[str, float]:
        """Alias of :meth:`as_floats` returning python floats."""
        return self.as_floats()

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.total.detach())


class PINNLoss:
    """Callable PINN loss ``L(w)`` (Eq. (2)) with component breakdown.

    Parameters
    ----------
    model:
        The PINN network ``u(x; w)``.
    problem:
        One of the :class:`PDEProblem` instances (convection / reaction / wave).
    sampler:
        Optional :class:`~src.pinns.sampling.PINNSampler` providing the fixed
        residual and condition point sets.  When omitted, points are generated
        with :func:`~src.pinns.sampling.build_sampler`.
    residual_points:
        Explicit residual collocation points; overrides the sampler.
    condition_point_groups:
        Explicit boundary/initial point groups; overrides the sampler.
    components:
        Which components to evaluate.  ``"all"`` (default) computes residual,
        initial and boundary terms; ``"residual"``/``"initial"``/``"boundary"``
        restrict the computation (useful to build component-wise spectral
        densities cheaply).
    residual_reduction:
        ``"paper"`` -> ``1/(2 n) sum``, ``"sum"`` -> plain ``sum``.
    """

    def __init__(
        self,
        model,
        problem: PDEProblem,
        sampler: Optional[PINNSampler] = None,
        residual_points: Optional[Tensor] = None,
        condition_point_groups: Optional[Sequence[Sequence[Tensor]]] = None,
        n_residual: Optional[int] = None,
        n_ic: Optional[int] = None,
        n_bc: Optional[int] = None,
        components: str = "all",
        seed: Optional[int] = None,
        device=None,
        dtype=None,
        residual_reduction: str = "paper",
    ) -> None:
        self.model = model
        self.problem = problem
        self.residual_reduction = residual_reduction
        self.components = (components or "all").lower()
        try:
            self.device = torch.device(device) if device is not None else next(model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            self.device = torch.device(device or "cpu")
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        if residual_points is None or condition_point_groups is None:
            if sampler is None:
                sampler = build_sampler(
                    problem,
                    seed=seed,
                    device=self.device,
                    dtype=self.dtype,
                    n_residual=n_residual,
                )
            self.sampler = sampler
        else:
            self.sampler = sampler

        if residual_points is not None:
            self.residual_points = residual_points.to(self.device)
        else:
            self.residual_points = sampler.residual_points.to(self.device)

        if condition_point_groups is not None:
            self.condition_point_groups = [_as_points_tuple(g) for g in condition_point_groups]
        else:
            self.condition_point_groups = _condition_point_groups(
                problem, sampler, self.device, self.dtype
            )

        self.conditions = problem.conditions()
        if len(self.condition_point_groups) < len(self.conditions):
            raise ValueError(
                "Number of condition point groups does not match the number of conditions "
                f"({len(self.condition_point_groups)} < {len(self.conditions)})."
            )

        # classification of every condition
        self.condition_kinds: List[str] = [classify_condition(c) for c in self.conditions]
        self.n_residual_points = int(self.residual_points.shape[0])
        self.n_condition_points = int(
            sum(int(p.shape[0]) for g in self.condition_point_groups for p in g)
        )

    # -- plumbing ---------------------------------------------------------
    def _want(self, name: str) -> bool:
        return self.components in {"all", "both", "full"} or self.components == name

    def to(self, device) -> "PINNLoss":
        self.device = torch.device(device)
        self.residual_points = self.residual_points.to(self.device)
        self.condition_point_groups = [
            tuple(p.to(self.device) for p in g) for g in self.condition_point_groups
        ]
        return self

    def resample_residual(self, seed: Optional[int] = None) -> "PINNLoss":
        """Re-draw the residual collocation points (fixed grid sampling)."""
        from .sampling import sample_residual_points

        self.residual_points = sample_residual_points(
            self.problem,
            n=self.n_residual_points,
            seed=seed,
            device=self.device,
            dtype=self.dtype,
        )
        return self

    # -- evaluation -------------------------------------------------------
    def breakdown(self, include_components: Optional[str] = None) -> LossBreakdown:
        """Evaluate the total loss and its component decomposition."""
        comps = (include_components or self.components).lower()

        residual_term = self.residual_points.new_zeros(())
        if comps in {"all", "both", "full", "residual"}:
            vals = problem_residual(self.model, self.problem, self.residual_points)
            if self.residual_reduction == "sum":
                residual_term = _sum_square(vals)
            else:
                residual_term = _half_mean_square(vals)

        # boundary / initial terms, one per condition group
        cond_terms: Dict[str, Tensor] = {}
        init_sum = self.residual_points.new_zeros(())
        init_count = 0
        bound_sum = self.residual_points.new_zeros(())
        bound_count = 0
        total_cond_sum = self.residual_points.new_zeros(())
        total_cond_count = 0

        if comps in {"all", "both", "full", "residual"} or True:
            # condition terms are cheap and are always needed for the total
            for idx, cond in enumerate(self.conditions):
                pts = self.condition_point_groups[idx]
                vals = apply_condition(self.model, cond, pts).reshape(-1)
                n_pts = int(vals.numel())
                if n_pts == 0:
                    continue
                name = f"{cond.name}#{idx}"
                if comps in {"all", "both", "full"} or comps == self.condition_kinds[idx]:
                    cond_terms[name] = _half_mean_square(vals)
                total_cond_sum = total_cond_sum + _sum_square(vals)
                total_cond_count += n_pts
                if self.condition_kinds[idx] == INITIAL:
                    init_sum = init_sum + _sum_square(vals)
                    init_count += n_pts
                else:
                    bound_sum = bound_sum + _sum_square(vals)
                    bound_count += n_pts

        if total_cond_count > 0:
            bc_term = 0.5 * total_cond_sum / total_cond_count
        else:  # pragma: no cover - no conditions
            bc_term = self.residual_points.new_zeros(())

        initial_term = (
            0.5 * init_sum / init_count if init_count > 0 else self.residual_points.new_zeros(())
        )
        boundary_term = (
            0.5 * bound_sum / bound_count if bound_count > 0 else self.residual_points.new_zeros(())
        )

        total = residual_term + bc_term

        return LossBreakdown(
            total=total,
            residual=residual_term,
            initial=initial_term,
            boundary=boundary_term,
            conditions=cond_terms,
            n_residual=self.n_residual_points,
            n_condition=total_cond_count,
            condition_kinds={
                f"{c.name}#{i}": self.condition_kinds[i] for i, c in enumerate(self.conditions)
            },
        )

    def __call__(self) -> Tensor:
        """Total PINN loss ``L(w)`` (differentiable w.r.t. the weights)."""
        return self.breakdown().total

    def forward(self) -> Tensor:  # pragma: no cover - alias
        return self()

    def closure(self) -> Tensor:
        """Zero-grad + re-evaluate; the interface L-BFGS/Armijo closures expect."""
        if hasattr(self.model, "zero_grad"):
            self.model.zero_grad(set_to_none=True)
        return self()

    def value(self) -> float:
        """Detached scalar value of the loss."""
        return float(self().detach())

    def components_dict(self) -> Dict[str, float]:
        return self.breakdown().as_floats()

    def grad(self) -> Tensor:
        """Gradient of the total loss w.r.t. the parameters (flat vector)."""
        loss = self()
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True, create_graph=False)
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros_like(p).reshape(-1))
            else:
                flat.append(g.reshape(-1).detach())
        if not flat:  # pragma: no cover
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        return torch.cat(flat)

    def grad_norm(self) -> float:
        g = self.grad()
        return float(torch.linalg.vector_norm(g))

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"problem={getattr(self.problem, 'name', '?')}, "
            f"n_residual={self.n_residual_points}, "
            f"n_condition={self.n_condition_points}, components={self.components}"
        )


def problem_residual(model, problem: PDEProblem, points: Tensor) -> Tensor:
    """Thin wrapper around ``problem.residual`` keeping the graph for autograd."""
    return problem.residual(model, points).reshape(-1)


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------
def loss_breakdown(
    model,
    problem: PDEProblem,
    sampler: Optional[PINNSampler] = None,
    residual_points: Optional[Tensor] = None,
    condition_point_groups: Optional[Sequence[Sequence[Tensor]]] = None,
    components: str = "all",
    seed: Optional[int] = None,
    device=None,
    dtype=None,
) -> LossBreakdown:
    """Functional form of :meth:`PINNLoss.breakdown`."""
    return PINNLoss(
        model=model,
        problem=problem,
        sampler=sampler,
        residual_points=residual_points,
        condition_point_groups=condition_point_groups,
        components=components,
        seed=seed,
        device=device,
        dtype=dtype,
    ).breakdown()


def pinn_loss(
    model,
    problem: PDEProblem,
    sampler: Optional[PINNSampler] = None,
    residual_points: Optional[Tensor] = None,
    condition_point_groups: Optional[Sequence[Sequence[Tensor]]] = None,
    return_components: bool = False,
    seed: Optional[int] = None,
    device=None,
    dtype=None,
    components: str = "all",
):
    """Evaluate the PINN loss (Eq. (2)).

    Parameters
    ----------
    return_components:
        When ``False`` (default) a scalar tensor (the total loss) is returned.
        When ``True`` a :class:`LossBreakdown` is returned, exposing ``total``,
        ``residual``, ``initial`` and ``boundary`` terms.
    """
    breakdown = loss_breakdown(
        model=model,
        problem=problem,
        sampler=sampler,
        residual_points=residual_points,
        condition_point_groups=condition_point_groups,
        components=components,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    if return_components:
        return breakdown
    return breakdown.total


def make_loss_fn(
    model,
    problem: PDEProblem,
    sampler: Optional[PINNSampler] = None,
    residual_points: Optional[Tensor] = None,
    condition_point_groups: Optional[Sequence[Sequence[Tensor]]] = None,
    components: str = "all",
    seed: Optional[int] = None,
    device=None,
    dtype=None,
):
    """Return a zero-argument closure evaluating the loss (for optimizers).

    The closure re-evaluates the loss on the *fixed* point set stored in the
    returned :class:`PINNLoss` object (L-BFGS / Armijo call the loss many times
    per step; keeping the points fixed matches the paper's fixed sample set).
    """
    objective = PINNLoss(
        model=model,
        problem=problem,
        sampler=sampler,
        residual_points=residual_points,
        condition_point_groups=condition_point_groups,
        components=components,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    return objective, objective.closure
