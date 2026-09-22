"""The differential equations studied in the paper (Appendix A).

The paper considers three problems:

* ``convection``  -- a 1D hyperbolic PDE,  ``u_t + beta u_x = 0``, beta = 40
* ``reaction``    -- a 1D non-linear ODE,  ``u_t - rho u (1 - u) = 0``, rho = 5
* ``wave``        -- a 1D hyperbolic PDE,  ``u_tt - 4 u_xx = 0``, beta = 5

Every problem exposes

``residual(net, X)``  the differential operator ``D[u(x; w), x]`` of Eq. (1),
``ic_terms()``        the list of initial condition operators ``B[u, x]``,
``bc_terms()``        the list of boundary condition operators ``B[u, x]``,
``exact(X)``          the analytical solution used for the L2RE.

Initial conditions are evaluated on ``N_IC = 257`` equally spaced points and
each boundary condition on ``N_BC = 101`` equally spaced points, matching
Section 2.2 of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch

# Number of collocation points used for each initial / boundary condition
# (Section 2.2: "257 equally spaced points for the initial conditions and
# 101 equally spaced points for each boundary condition").
N_IC = 257
N_BC = 101


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y / d x keeping the graph so that higher order derivatives work."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


@dataclass
class ConditionTerm:
    """A single initial/boundary condition ``B[u(x; w), x] = 0``."""

    name: str
    n_points: int
    fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    sample: Callable[[torch.Generator, int], torch.Tensor]


class Problem:
    """Base class for the three problems of Appendix A."""

    name: str = ""
    x_domain: Tuple[float, float] = (0.0, 1.0)
    t_domain: Tuple[float, float] = (0.0, 1.0)
    # number of equally spaced interior grid points used for the L2RE
    grid_x: int = 255
    grid_t: int = 100

    # ------------------------------------------------------------------ #
    # to be implemented by the subclasses
    # ------------------------------------------------------------------ #
    def residual(self, net: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ic_terms(self) -> List[ConditionTerm]:
        raise NotImplementedError

    def bc_terms(self) -> List[ConditionTerm]:
        raise NotImplementedError

    def exact(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def eq_points(self, n: int, axis: int, value: float) -> torch.Tensor:
        """``n`` equally spaced points on the closed domain with ``axis=value``.

        ``axis=1`` fixes ``t`` (initial conditions), ``axis=0`` fixes ``x``
        (boundary conditions).
        """
        lo, hi = self.x_domain if axis == 0 else self.t_domain
        other = self.t_domain if axis == 0 else self.x_domain
        s = torch.linspace(other[0], other[1], n)
        fixed = torch.full_like(s, float(value))
        if axis == 0:  # x is fixed, the free coordinate is t
            # keep the convention X = [x, t]
            return torch.stack([fixed, s], dim=1)
        return torch.stack([s, fixed], dim=1)

    def interior_grid(self) -> torch.Tensor:
        """The ``grid_x * grid_t`` interior grid used for L2RE (Section 2.2)."""
        x = torch.linspace(self.x_domain[0], self.x_domain[1], self.grid_x + 2)[1:-1]
        t = torch.linspace(self.t_domain[0], self.t_domain[1], self.grid_t + 2)[1:-1]
        X, T = torch.meshgrid(x, t, indexing="ij")
        return torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

    def evaluation_points(self) -> torch.Tensor:
        """Grid + initial-condition points + boundary points (Section 2.2)."""
        parts = [self.interior_grid()]
        for term in self.ic_terms():
            parts.append(term.sample(torch.Generator().manual_seed(0), term.n_points))
        for term in self.bc_terms():
            parts.append(term.sample(torch.Generator().manual_seed(0), term.n_points))
        # the wave problem evaluates two initial conditions on the same points,
        # and the corners of the IC/BC sets can coincide: keep unique locations
        return torch.unique(torch.cat(parts, dim=0), dim=0)


# ---------------------------------------------------------------------- #
# A.1 Convection
# ---------------------------------------------------------------------- #
class Convection(Problem):
    r"""``u_t + beta u_x = 0`` on ``(0, 2 pi) x (0, 1)`` with ``beta = 40``."""

    name = "convection"
    x_domain = (0.0, 2.0 * math.pi)
    t_domain = (0.0, 1.0)

    def __init__(self, beta: float = 40.0):
        self.beta = float(beta)

    def residual(self, net, X):
        X = X.clone().requires_grad_(True)
        u = net(X)
        g = _grad(u, X)
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        return u_t + self.beta * u_x

    def ic_terms(self):
        return [
            ConditionTerm(
                "initial_condition",
                N_IC,
                lambda net, X: net(X) - torch.sin(X[:, 0:1]),
                lambda g, n: self.eq_points(n, axis=1, value=0.0),
            )
        ]

    def bc_terms(self):
        lo, hi = self.x_domain

        def periodic(net, X):
            # X holds the t values; the periodic boundary wraps x around
            t = X[:, 1:2]
            xl = torch.zeros_like(t)
            xr = torch.full_like(t, hi)
            return net(torch.cat([xl, t], dim=1)) - net(torch.cat([xr, t], dim=1))

        return [
            ConditionTerm(
                "periodic_boundary",
                N_BC,
                periodic,
                # the periodic condition is imposed on 101 equally spaced
                # time values at x = 0 (and, by periodicity, at x = 2 pi)
                lambda g, n: self.eq_points(n, axis=0, value=0.0),
            )
        ]

    def exact(self, X):
        x, t = X[:, 0:1], X[:, 1:2]
        return torch.sin(x - self.beta * t)


# ---------------------------------------------------------------------- #
# A.2 Reaction
# ---------------------------------------------------------------------- #
class Reaction(Problem):
    r"""``u_t - rho u (1 - u) = 0`` with ``rho = 5``."""

    name = "reaction"
    x_domain = (0.0, 2.0 * math.pi)
    t_domain = (0.0, 1.0)

    def __init__(self, rho: float = 5.0):
        self.rho = float(rho)

    @staticmethod
    def h(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))

    def residual(self, net, X):
        X = X.clone().requires_grad_(True)
        u = net(X)
        u_t = _grad(u, X)[:, 1:2]
        return u_t - self.rho * u * (1.0 - u)

    def ic_terms(self):
        return [
            ConditionTerm(
                "initial_condition",
                N_IC,
                lambda net, X: net(X) - self.h(X[:, 0:1]),
                lambda g, n: self.eq_points(n, axis=1, value=0.0),
            )
        ]

    def bc_terms(self):
        _, hi = self.x_domain

        def periodic(net, X):
            t = X[:, 1:2]
            xl = torch.zeros_like(t)
            xr = torch.full_like(t, hi)
            return net(torch.cat([xl, t], dim=1)) - net(torch.cat([xr, t], dim=1))

        return [
            ConditionTerm(
                "periodic_boundary",
                N_BC,
                periodic,
                lambda g, n: self.eq_points(n, axis=0, value=0.0),
            )
        ]

    def exact(self, X):
        x, t = X[:, 0:1], X[:, 1:2]
        h = self.h(x)
        return h * torch.exp(self.rho * t) / (h * torch.exp(self.rho * t) + 1.0 - h)


# ---------------------------------------------------------------------- #
# A.3 Wave
# ---------------------------------------------------------------------- #
class Wave(Problem):
    r"""``u_tt - 4 u_xx = 0`` on ``(0, 1) x (0, 1)`` with ``beta = 5``."""

    name = "wave"
    x_domain = (0.0, 1.0)
    t_domain = (0.0, 1.0)

    def __init__(self, beta: float = 5.0):
        self.beta = float(beta)

    def residual(self, net, X):
        X = X.clone().requires_grad_(True)
        u = net(X)
        g = _grad(u, X)
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_tt = _grad(u_t, X)[:, 1:2]
        return u_tt - 4.0 * u_xx

    def _u0(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x) + 0.5 * torch.sin(self.beta * math.pi * x)

    def ic_terms(self):
        def ut(net, X):
            # d/dt u(x, 0) = 0
            Xg = X.clone().requires_grad_(True)
            u_t = _grad(net(Xg), Xg)[:, 1:2]
            return u_t

        def u0(net, X):
            return net(X) - self._u0(X[:, 0:1])

        return [
            ConditionTerm(
                "initial_condition_u",
                N_IC,
                u0,
                lambda g, n: self.eq_points(n, axis=1, value=0.0),
            ),
            ConditionTerm(
                "initial_condition_ut",
                N_IC,
                ut,
                lambda g, n: self.eq_points(n, axis=1, value=0.0),
            ),
        ]

    def bc_terms(self):
        def left(net, X):
            return net(X)  # u(0, t) = 0

        def right(net, X):
            return net(X)  # u(1, t) = 0

        return [
            ConditionTerm(
                "boundary_left", N_BC, left, lambda g, n: self.eq_points(n, axis=0, value=0.0)
            ),
            ConditionTerm(
                "boundary_right",
                N_BC,
                right,
                lambda g, n: self.eq_points(n, axis=0, value=1.0),
            ),
        ]

    def exact(self, X):
        x, t = X[:, 0:1], X[:, 1:2]
        return torch.sin(math.pi * x) * torch.cos(2.0 * math.pi * t) + 0.5 * torch.sin(
            self.beta * math.pi * x
        ) * torch.cos(2.0 * self.beta * math.pi * t)


PROBLEMS = {
    "convection": Convection,
    "reaction": Reaction,
    "wave": Wave,
}


def build_problem(name: str, **kwargs) -> Problem:
    if name not in PROBLEMS:
        raise ValueError(f"unknown problem {name!r}; expected one of {sorted(PROBLEMS)}")
    return PROBLEMS[name](**kwargs)
