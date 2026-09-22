"""PDE problems studied in the paper.

Implements the differential operators ``D`` (PDE residual), the boundary/initial
operators ``B`` and the analytical solutions used only for error evaluation for
the three problems of Section 2.1 / Appendix A:

* convection :  u_t + beta u_x = 0,       (0, 2pi) x (0, 1), periodic BC,
                IC u(x, 0) = sin(x),     beta = 40,   exact u = sin(x - beta t)
* reaction   :  u_t - rho u (1 - u) = 0,  (0, 2pi) x (0, 1), periodic BC,
                IC u(x, 0) = exp(-(x - pi)^2 / (2 (pi / 4)^2)),  rho = 5,
                exact u = h e^{rho t} / (h e^{rho t} + 1 - h)
* wave       :  u_tt - 4 u_xx = 0,        (0, 1) x (0, 1), Dirichlet BC,
                IC u(x, 0)   = sin(pi x) + 0.5 sin(beta pi x),
                IC u_t(x, 0) = 0,        beta = 5,
                exact u = sin(pi x) cos(2 pi t) + 0.5 sin(beta pi x) cos(2 beta pi t)

``D`` is applied to the network output with ``torch.autograd.grad(..., create_graph=True)``
so that the resulting loss stays differentiable with respect to the weights ``w``
(needed for Hessian-vector products).

Source: §2.1 (Eqs. (1), (2)), Appendix A.1, A.2, A.3, §2.2 (sampling sizes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "BoundaryCondition",
    "PDEProblem",
    "Convection",
    "Reaction",
    "Wave",
    "PROBLEMS",
    "get_problem",
    "first_grad",
    "second_grad",
]

# --------------------------------------------------------------------------- #
# small autograd helpers
# --------------------------------------------------------------------------- #


def first_grad(y: Tensor, x: Tensor) -> Tensor:
    """First derivatives of ``y`` (n, 1) with respect to ``x`` (n, d) -> (n, d)."""
    return torch.autograd.grad(
        y.sum(), x, create_graph=True, retain_graph=True
    )[0]


def second_grad(y: Tensor, x: Tensor, index: int) -> Tensor:
    """Second derivative d^2 y / d x_index^2 -> (n, 1)."""
    g = first_grad(y, x)
    g_i = g[:, index : index + 1]
    gg = torch.autograd.grad(
        g_i.sum(), x, create_graph=True, retain_graph=True
    )[0]
    return gg[:, index : index + 1]


def _as_column(u: Tensor) -> Tensor:
    """Coerce a network output to shape (n, 1)."""
    if u.dim() == 1:
        return u.reshape(-1, 1)
    return u


def _network_u(u_fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Apply the ansatz and coerce to a column vector."""
    return _as_column(u_fn(x))


# --------------------------------------------------------------------------- #
# boundary / initial condition descriptors
# --------------------------------------------------------------------------- #


@dataclass
class BoundaryCondition:
    """Description of one additive term of the operator ``B``.

    Parameters
    ----------
    name : human readable name (``"ic"``, ``"dirichlet"``, ``"periodic"``, ...)
    kind : one of

        * ``"value"``          -> u(x_b) - target
        * ``"ic_derivative"``   -> du/dt (x_b) - target
        * ``"periodic"``        -> u(axis=value) - u(axis=value2)
    axis : ``"x"`` (points lie on a spatial boundary) or ``"t"`` (points lie on
        the initial time slice ``t = value``).
    value : coordinate of the boundary / initial slice.
    value2 : second boundary (only for ``kind="periodic"``).
    n_points : number of points used for this condition.
    target : callable mapping the free coordinate (shape (n,) or (n, 1)) to the
        target value; ``None`` means identically zero.
    """

    name: str
    kind: str
    axis: str
    value: float
    n_points: int
    target: Optional[Callable[[Tensor], Tensor]] = None
    value2: Optional[float] = None

    def target_values(self, x: Tensor) -> Tensor:
        """Evaluate the target on the batch ``x`` (n, 2) -> (n, 1)."""
        if self.target is None:
            return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        free = x[:, 0] if self.axis == "t" else x[:, 1]
        out = self.target(free)
        if not torch.is_tensor(out):
            out = torch.full_like(free, float(out))
        return _as_column(out)


def apply_condition(
    u_fn: Callable[[Tensor], Tensor],
    cond: BoundaryCondition,
    points: Sequence[Tensor],
) -> Tensor:
    """Evaluate the residual of a single boundary/initial condition.

    ``points`` contains one tensor for value/dirichlet/ic_derivative conditions
    and two tensors (paired boundary locations) for periodic conditions.
    """
    pts = [p.detach().clone().requires_grad_(True) for p in points]
    if cond.kind == "periodic":
        assert len(pts) == 2, "periodic conditions require the two boundary slabs"
        return _network_u(u_fn, pts[0]) - _network_u(u_fn, pts[1])
    if cond.kind == "ic_derivative":
        u = _network_u(u_fn, pts[0])
        u_t = first_grad(u, pts[0])[:, 1:2]
        return u_t - cond.target_values(pts[0])
    if cond.kind in ("value", "ic", "dirichlet"):
        u = _network_u(u_fn, pts[0])
        return u - cond.target_values(pts[0])
    raise ValueError(f"unknown boundary-condition kind: {cond.kind!r}")


# --------------------------------------------------------------------------- #
# base problem
# --------------------------------------------------------------------------- #


class PDEProblem:
    """Common interface for the three PDEs of the paper.

    Sub-classes set the domain, the coefficients and implement :meth:`residual`,
    :meth:`conditions` and :meth:`exact`.

    Sampling sizes (paper §2.2): ``n_residual = 10_000`` residual collocation
    points drawn from a ``255 x 100`` interior grid, ``n_ic = 257`` equally
    spaced initial-condition points, ``n_bc = 101`` equally spaced points per
    boundary.
    """

    name: str = "base"

    # domain
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # sampling / evaluation sizes
    n_residual: int = 10_000
    n_ic: int = 257
    n_bc: int = 101
    # fine grid used for sampling and for the L2RE evaluation
    n_grid_x: int = 255
    n_grid_t: int = 100

    #: input/output dimension of the ansatz (spatial + time -> scalar field)
    in_dim: int = 2
    out_dim: int = 1

    # -- geometry ---------------------------------------------------------- #
    @property
    def domain(self) -> tuple:
        return (self.x_min, self.x_max, self.t_min, self.t_max)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(x in ({self.x_min:g}, {self.x_max:g}), "
            f"t in ({self.t_min:g}, {self.t_max:g}))"
        )

    # -- to be implemented by sub-classes ---------------------------------- #
    def residual(self, u_fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
        """D[u(x; w), x] evaluated at the collocation points ``x`` (n, 2)."""
        raise NotImplementedError

    def conditions(self) -> List[BoundaryCondition]:
        """List of boundary/initial conditions defining the operator ``B``."""
        raise NotImplementedError

    def exact(self, x: Tensor) -> Tensor:
        """Analytical solution (used **only** for evaluation, never training)."""
        raise NotImplementedError

    # -- convenience ------------------------------------------------------- #
    @property
    def n_bc_total(self) -> int:
        """Total number of boundary/initial residuals (normalisation of ``B``)."""
        return int(sum(c.n_points for c in self.conditions()))

    def condition_summary(self) -> Dict[str, int]:
        return {c.name: c.n_points for c in self.conditions()}

    def boundary_residuals(
        self,
        u_fn: Callable[[Tensor], Tensor],
        points_per_condition: Sequence[Sequence[Tensor]],
    ) -> List[Tensor]:
        """Residual of every condition, stacked component-wise.

        ``points_per_condition[i]`` holds the point tensors of condition ``i``
        (one tensor, or two for a periodic condition).
        """
        conds = self.conditions()
        assert len(conds) == len(points_per_condition), (
            "expected one point tensor list per boundary condition"
        )
        return [
            apply_condition(u_fn, c, pts)
            for c, pts in zip(conds, points_per_condition)
        ]

    # -- grid helpers (used by sampling / metrics) ------------------------- #
    def interior_grid_1d(self, device="cpu", dtype=torch.float32):
        """1-D coordinates of the ``255 x 100`` interior grid."""
        x = torch.linspace(self.x_min, self.x_max, self.n_grid_x, device=device, dtype=dtype)
        t = torch.linspace(self.t_min, self.t_max, self.n_grid_t, device=device, dtype=dtype)
        return x, t

    def interior_grid(self, device="cpu", dtype=torch.float32) -> Tensor:
        """Full interior grid as an (n_grid_x * n_grid_t, 2) tensor."""
        x, t = self.interior_grid_1d(device=device, dtype=dtype)
        X, T = torch.meshgrid(x, t, indexing="ij")
        return torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)


# --------------------------------------------------------------------------- #
# A.1  Convection
# --------------------------------------------------------------------------- #


class Convection(PDEProblem):
    """u_t + beta u_x = 0 on (0, 2 pi) x (0, 1); exact u(x, t) = sin(x - beta t)."""

    name = "convection"

    def __init__(self, beta: float = 40.0):
        self.beta = float(beta)
        self.x_min = 0.0
        self.x_max = 2.0 * math.pi
        self.t_min = 0.0
        self.t_max = 1.0

    def residual(self, u_fn, x: Tensor) -> Tensor:
        u = _network_u(u_fn, x)
        g = first_grad(u, x)
        u_x = g[:, 0:1]
        u_t = g[:, 1:2]
        return u_t + self.beta * u_x

    def conditions(self) -> List[BoundaryCondition]:
        return [
            BoundaryCondition(
                name="ic",
                kind="value",
                axis="t",
                value=self.t_min,
                n_points=self.n_ic,
                target=lambda x: torch.sin(x),
            ),
            BoundaryCondition(
                name="periodic",
                kind="periodic",
                axis="x",
                value=self.x_min,
                value2=self.x_max,
                n_points=self.n_bc,
            ),
        ]

    def exact(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return torch.sin(xx - self.beta * tt)


# --------------------------------------------------------------------------- #
# A.2  Reaction
# --------------------------------------------------------------------------- #


class Reaction(PDEProblem):
    """u_t - rho u (1 - u) = 0 on (0, 2 pi) x (0, 1)."""

    name = "reaction"

    def __init__(self, rho: float = 5.0):
        self.rho = float(rho)
        self.x_min = 0.0
        self.x_max = 2.0 * math.pi
        self.t_min = 0.0
        self.t_max = 1.0

    # -- initial condition g(x) = exp(-(x - pi)^2 / (2 (pi/4)^2)) ----------- #
    @staticmethod
    def _gaussian(x: Tensor) -> Tensor:
        return torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))

    def residual(self, u_fn, x: Tensor) -> Tensor:
        u = _network_u(u_fn, x)
        g = first_grad(u, x)
        u_t = g[:, 1:2]
        return u_t - self.rho * u * (1.0 - u)

    def conditions(self) -> List[BoundaryCondition]:
        return [
            BoundaryCondition(
                name="ic",
                kind="value",
                axis="t",
                value=self.t_min,
                n_points=self.n_ic,
                target=self._gaussian,
            ),
            BoundaryCondition(
                name="periodic",
                kind="periodic",
                axis="x",
                value=self.x_min,
                value2=self.x_max,
                n_points=self.n_bc,
            ),
        ]

    def exact(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        h = self._gaussian(xx)
        e = torch.exp(self.rho * tt)
        return h * e / (h * e + 1.0 - h)


# --------------------------------------------------------------------------- #
# A.3  Wave
# --------------------------------------------------------------------------- #


class Wave(PDEProblem):
    """u_tt - 4 u_xx = 0 on (0, 1) x (0, 1) with Dirichlet and initial conditions."""

    name = "wave"

    def __init__(self, beta: float = 5.0, c2: float = 4.0):
        self.beta = float(beta)
        self.c2 = float(c2)
        self.x_min = 0.0
        self.x_max = 1.0
        self.t_min = 0.0
        self.t_max = 1.0

    def residual(self, u_fn, x: Tensor) -> Tensor:
        u = _network_u(u_fn, x)
        g = first_grad(u, x)
        u_x = g[:, 0:1]
        u_t = g[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x.sum(), x, create_graph=True, retain_graph=True
        )[0][:, 0:1]
        u_tt = torch.autograd.grad(
            u_t.sum(), x, create_graph=True, retain_graph=True
        )[0][:, 1:2]
        return u_tt - self.c2 * u_xx

    def _ic_value(self, x: Tensor) -> Tensor:
        return torch.sin(math.pi * x) + 0.5 * torch.sin(self.beta * math.pi * x)

    def conditions(self) -> List[BoundaryCondition]:
        return [
            BoundaryCondition(
                name="ic",
                kind="value",
                axis="t",
                value=self.t_min,
                n_points=self.n_ic,
                target=self._ic_value,
            ),
            BoundaryCondition(
                name="ic_derivative",
                kind="ic_derivative",
                axis="t",
                value=self.t_min,
                n_points=self.n_ic,
                target=None,  # du/dt(x, 0) = 0
            ),
            BoundaryCondition(
                name="dirichlet_left",
                kind="value",
                axis="x",
                value=self.x_min,
                n_points=self.n_bc,
                target=None,  # u(0, t) = 0
            ),
            BoundaryCondition(
                name="dirichlet_right",
                kind="value",
                axis="x",
                value=self.x_max,
                n_points=self.n_bc,
                target=None,  # u(1, t) = 0
            ),
        ]

    def exact(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return (
            torch.sin(math.pi * xx) * torch.cos(2.0 * math.pi * tt)
            + 0.5
            * torch.sin(self.beta * math.pi * xx)
            * torch.cos(2.0 * self.beta * math.pi * tt)
        )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

PROBLEMS: Dict[str, type] = {
    "convection": Convection,
    "reaction": Reaction,
    "wave": Wave,
}


def get_problem(name: str, **kwargs) -> PDEProblem:
    """Instantiate one of the three problems by name.

    Extra keyword arguments override the default coefficients
    (``beta=40`` for convection, ``rho=5`` for reaction, ``beta=5`` for wave).
    """
    key = str(name).strip().lower()
    if key not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[key](**kwargs)
