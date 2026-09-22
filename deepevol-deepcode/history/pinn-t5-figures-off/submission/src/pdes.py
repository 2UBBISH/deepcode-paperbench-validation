"""PDE definitions for the PINN experiments (Appendix A).

Each PDE class provides:
  - residual(u, x, t): the differential operator D[u] evaluated via autodiff
  - boundary_residual(u, x, t): boundary condition residual B[u]
  - initial_residual(u, x, t): initial condition residual
  - exact(x, t): analytical solution
  - domain: (x_min, x_max, t_min, t_max)
  - sample_residual_points / sample_ic_points / sample_bc_points

Conventions: x, t are tensors of shape (N, 1) requiring grad.
u is the network output of shape (N, 1).
"""

import math
import torch


def _grad(y, x):
    """Compute dy/dx, keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


class Convection:
    """du/dt + beta * du/dx = 0, x in (0, 2pi), t in (0, 1).

    IC: u(x, 0) = sin(x)
    Periodic BC: u(0, t) = u(2pi, t)
    Exact: u(x, t) = sin(x - beta t)
    """

    name = "convection"

    def __init__(self, beta=40.0):
        self.beta = beta
        self.domain = (0.0, 2.0 * math.pi, 0.0, 1.0)

    def residual(self, u, x, t):
        u_t = _grad(u, t)
        u_x = _grad(u, x)
        return u_t + self.beta * u_x

    def initial_residual(self, u, x, t):
        # u(x, 0) - sin(x)
        return u - torch.sin(x)

    def boundary_residual(self, u, x, t):
        # periodic: u(0, t) - u(2pi, t) handled by pairing points in data.py
        # Here we return u itself; the pairing is done by the loss using two sets.
        return u

    def exact(self, x, t):
        return torch.sin(x - self.beta * t)


class Reaction:
    """du/dt - rho * u * (1 - u) = 0, x in (0, 2pi), t in (0, 1).

    IC: u(x, 0) = exp(-(x - pi)^2 / (2 (pi/4)^2))
    Periodic BC.
    Exact: u = h(x) e^{rho t} / (h(x) e^{rho t} + 1 - h(x))
    """

    name = "reaction"

    def __init__(self, rho=5.0):
        self.rho = rho
        self.domain = (0.0, 2.0 * math.pi, 0.0, 1.0)

    def _h(self, x):
        return torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))

    def residual(self, u, x, t):
        u_t = _grad(u, t)
        return u_t - self.rho * u * (1.0 - u)

    def initial_residual(self, u, x, t):
        return u - self._h(x)

    def boundary_residual(self, u, x, t):
        return u

    def exact(self, x, t):
        h = self._h(x)
        e = torch.exp(self.rho * t)
        return h * e / (h * e + 1.0 - h)


class Wave:
    """d2u/dt2 - 4 * d2u/dx2 = 0, x in (0, 1), t in (0, 1).

    IC: u(x, 0) = sin(pi x) + 0.5 sin(beta pi x), du/dt(x, 0) = 0
    Dirichlet BC: u(0, t) = u(1, t) = 0
    Exact: u = sin(pi x) cos(2 pi t) + 0.5 sin(beta pi x) cos(2 beta pi t)
    """

    name = "wave"

    def __init__(self, beta=5.0):
        self.beta = beta
        self.domain = (0.0, 1.0, 0.0, 1.0)

    def residual(self, u, x, t):
        u_tt = _grad(_grad(u, t), t)
        u_xx = _grad(_grad(u, x), x)
        return u_tt - 4.0 * u_xx

    def initial_residual(self, u, x, t):
        return u - (torch.sin(math.pi * x) + 0.5 * torch.sin(self.beta * math.pi * x))

    def initial_velocity_residual(self, u, x, t):
        u_t = _grad(u, t)
        return u_t  # du/dt(x, 0) = 0

    def boundary_residual(self, u, x, t):
        # Dirichlet: u = 0 at x=0 and x=1
        return u

    def exact(self, x, t):
        return torch.sin(math.pi * x) * torch.cos(2.0 * math.pi * t) + 0.5 * torch.sin(
            self.beta * math.pi * x
        ) * torch.cos(2.0 * self.beta * math.pi * t)


def get_pde(name, **kwargs):
    name = name.lower()
    if name == "convection":
        return Convection(**kwargs)
    if name == "reaction":
        return Reaction(**kwargs)
    if name == "wave":
        return Wave(**kwargs)
    raise ValueError(f"Unknown PDE: {name}")
