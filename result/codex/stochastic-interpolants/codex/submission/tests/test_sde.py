"""Corollary 3.1: the forward/backward SDEs transport the base to the target.

The test uses a jointly Gaussian coupling, for which both the velocity and
the score are known in closed form, and checks that

    * the forward SDE from the base samples reproduces the target statistics,
    * the backward SDE from target samples reproduces the base statistics,
    * omitting the score term (eps_t = 0, the probability-flow ODE) still
      transports correctly, since eps_t >= 0 is arbitrary.
"""

import torch

from si_couplings.interpolants import GaussianCoupling
from si_couplings.solvers import (
    ConstantEpsilon,
    backward_sde_sample,
    forward_sde_sample,
    odeint,
)


def _coupling(d: int = 2, rho: float = 0.6, sigma: float = 0.5) -> GaussianCoupling:
    eye = torch.eye(d)
    return GaussianCoupling(
        mu_0=torch.zeros(d), mu_1=torch.zeros(d),
        C_00=(rho**2 + sigma**2) * eye, C_11=eye, C_01=rho * eye,
    )


def _schedule():
    # the standard stochastic interpolant: gamma_t = sqrt(2 t (1 - t)) and the
    # coupling x_0 = rho x_1 + sigma zeta on top of it
    from si_couplings.interpolants import build_interpolant

    return build_interpolant("linear")


def test_forward_sde_transports_base_to_target():
    torch.manual_seed(0)
    d, n, sigma = 2, 20_000, 0.5
    gc = _coupling(d, sigma=sigma)
    sch = _schedule()

    def velocity_fn(x, t):
        return gc.velocity_from_schedule(x, sch, float(t.reshape(-1)[0]))

    def log_score_fn(x, t):
        return gc.log_score_from_schedule(x, sch, float(t.reshape(-1)[0]))

    # base marginal: x_0 = rho x_1 + sigma zeta has variance rho^2 + sigma^2
    x0 = torch.randn(n, d) * (0.6**2 + sigma**2) ** 0.5
    x1 = forward_sde_sample(
        velocity_fn, log_score_fn, lambda t: torch.ones_like(t), x0,
        eps_fn=ConstantEpsilon(0.5), steps=500, log_score=True,
    )
    assert torch.allclose(x1.mean(0), torch.zeros(d), atol=0.08)
    assert torch.allclose(x1.var(dim=0).mean(), torch.tensor(1.0), atol=0.12)


def test_backward_sde_transports_target_to_base():
    torch.manual_seed(0)
    d, n, sigma = 2, 20_000, 0.5
    gc = _coupling(d, sigma=sigma)
    sch = _schedule()

    def velocity_fn(x, t):
        return gc.velocity_from_schedule(x, sch, float(t.reshape(-1)[0]))

    def log_score_fn(x, t):
        return gc.log_score_from_schedule(x, sch, float(t.reshape(-1)[0]))

    x1 = torch.randn(n, d)
    x0 = backward_sde_sample(
        velocity_fn, log_score_fn, lambda t: torch.ones_like(t), x1,
        eps_fn=ConstantEpsilon(0.5), steps=500, log_score=True,
    )
    assert torch.allclose(x0.var(dim=0).mean(), torch.tensor(0.6**2 + sigma**2), atol=0.1)


def test_zero_epsilon_recovers_probability_flow():
    """eps_t = 0 must reproduce the deterministic probability-flow ODE."""
    torch.manual_seed(0)
    d, n, sigma = 2, 5_000, 0.5
    gc = _coupling(d, sigma=sigma)
    sch = _schedule()

    def velocity_fn(x, t):
        return gc.velocity_from_schedule(x, sch, float(t.reshape(-1)[0]))

    def log_score_fn(x, t):
        return gc.log_score_from_schedule(x, sch, float(t.reshape(-1)[0]))

    x0 = torch.randn(n, d) * (0.6**2 + sigma**2) ** 0.5
    sde = forward_sde_sample(velocity_fn, log_score_fn, lambda t: torch.ones_like(t), x0,
                             eps_fn=ConstantEpsilon(0.0), steps=200, log_score=True)
    ode = odeint(velocity_fn, x0, method="euler", steps=200)
    assert torch.allclose(sde, ode, atol=1e-4)
