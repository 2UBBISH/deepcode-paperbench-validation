"""
PDE definitions for the PINN loss-landscape study.

Implements three PDEs from Appendix A of
"Challenges in Training PINNs: A Loss Landscape Perspective":

  - Convection:  D[u] = u_t + beta * u_x
  - Reaction:    D[u] = u_t - rho * u * (1 - u)
  - Wave:        D[u] = u_tt - 4 * u_xx

Each PDE exposes:
  * residual_operator(u_fn, x, t)  -> residual values (autodiff)
  * ic_residual(u_fn, x)           -> initial-condition residual
  * bc_residual(u_fn, x, t)        -> boundary-condition residual
  * exact_solution(x, t)           -> analytic solution
  * domain / sampling metadata

All tensors are assumed to be shape (N, 1) unless stated otherwise.
"""

import math
import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class PDE:
    """Base class for a PDE problem."""

    name = "base"
    # spatial / temporal domain
    x_min = 0.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0
    # number of boundary sides (for BC sampling)
    n_bc_sides = 2

    def residual_operator(self, u_fn, x, t):
        """Return the PDE residual D[u] evaluated at (x, t)."""
        raise NotImplementedError

    def ic_residual(self, u_fn, x):
        """Return the initial-condition residual at t = t_min."""
        raise NotImplementedError

    def bc_residual(self, u_fn, x, t):
        """Return the boundary-condition residual."""
        raise NotImplementedError

    def exact_solution(self, x, t):
        """Return the analytic solution u*(x, t)."""
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _grad(y, x):
        """d y / d x with graph retained (for higher-order derivatives)."""
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]


# ---------------------------------------------------------------------------
# Convection:  u_t + beta u_x = 0
# ---------------------------------------------------------------------------
class Convection(PDE):
    name = "convection"
    x_min = 0.0
    x_max = 2.0 * math.pi
    t_min = 0.0
    t_max = 1.0
    n_bc_sides = 2  # periodic: x=0 and x=2pi

    def __init__(self, beta=40.0):
        self.beta = beta

    def residual_operator(self, u_fn, x, t):
        u = u_fn(x, t)
        u_t = self._grad(u, t)
        u_x = self._grad(u, x)
        return u_t + self.beta * u_x

    def ic_residual(self, u_fn, x):
        t0 = torch.zeros_like(x)
        u = u_fn(x, t0)
        return u - torch.sin(x)

    def bc_residual(self, u_fn, x, t):
        # periodic BC: u(0, t) = u(2pi, t)
        x0 = torch.full_like(x, self.x_min)
        x1 = torch.full_like(x, self.x_max)
        return u_fn(x0, t) - u_fn(x1, t)

    def exact_solution(self, x, t):
        return torch.sin(x - self.beta * t)


# ---------------------------------------------------------------------------
# Reaction:  u_t - rho u (1 - u) = 0
# ---------------------------------------------------------------------------
class Reaction(PDE):
    name = "reaction"
    x_min = 0.0
    x_max = 2.0 * math.pi
    t_min = 0.0
    t_max = 1.0
    n_bc_sides = 2  # periodic

    def __init__(self, rho=5.0):
        self.rho = rho

    @staticmethod
    def _h(x):
        # Gaussian initial condition
        return torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))

    def residual_operator(self, u_fn, x, t):
        u = u_fn(x, t)
        u_t = self._grad(u, t)
        return u_t - self.rho * u * (1.0 - u)

    def ic_residual(self, u_fn, x):
        t0 = torch.zeros_like(x)
        u = u_fn(x, t0)
        return u - self._h(x)

    def bc_residual(self, u_fn, x, t):
        x0 = torch.full_like(x, self.x_min)
        x1 = torch.full_like(x, self.x_max)
        return u_fn(x0, t) - u_fn(x1, t)

    def exact_solution(self, x, t):
        h = self._h(x)
        e = torch.exp(self.rho * t)
        return h * e / (h * e + 1.0 - h)


# ---------------------------------------------------------------------------
# Wave:  u_tt - 4 u_xx = 0
# ---------------------------------------------------------------------------
class Wave(PDE):
    name = "wave"
    x_min = 0.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0
    n_bc_sides = 2  # Dirichlet: x=0 and x=1

    def __init__(self, beta=5.0):
        self.beta = beta

    def residual_operator(self, u_fn, x, t):
        u = u_fn(x, t)
        u_t = self._grad(u, t)
        u_tt = self._grad(u_t, t)
        u_x = self._grad(u, x)
        u_xx = self._grad(u_x, x)
        return u_tt - 4.0 * u_xx

    def ic_residual(self, u_fn, x):
        # two ICs: u(x,0) and u_t(x,0); return stacked residual
        t0 = torch.zeros_like(x)
        u = u_fn(x, t0)
        u_t = self._grad(u, t0)
        ic_u = u - (torch.sin(math.pi * x) + 0.5 * torch.sin(self.beta * math.pi * x))
        ic_ut = u_t - torch.zeros_like(x)
        return torch.cat([ic_u, ic_ut], dim=0)

    def bc_residual(self, u_fn, x, t):
        x0 = torch.full_like(x, self.x_min)
        x1 = torch.full_like(x, self.x_max)
        return torch.cat([u_fn(x0, t), u_fn(x1, t)], dim=0)

    def exact_solution(self, x, t):
        return (
            torch.sin(math.pi * x) * torch.cos(2.0 * math.pi * t)
            + 0.5 * torch.sin(self.beta * math.pi * x) * torch.cos(2.0 * self.beta * math.pi * t)
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_pde(name, **kwargs):
    name = name.lower()
    if name == "convection":
        return Convection(**kwargs)
    if name == "reaction":
        return Reaction(**kwargs)
    if name == "wave":
        return Wave(**kwargs)
    raise ValueError(f"Unknown PDE: {name}")
