"""Numerical checks of Theorem 3.1: the velocity and the score identities.

For a jointly Gaussian coupling every object in Theorem 3.1 is available in
closed form, so the statements of the theorem can be verified numerically.
"""

import torch

from si_couplings.interpolants import GaussianCoupling, build_interpolant
from si_couplings.losses import flatten_sum, velocity_loss
from si_couplings.models.toy import MLPVelocity


def _gaussian_coupling(d: int = 2, rho: float = 0.6, sigma: float = 0.5) -> GaussianCoupling:
    """x_1 ~ N(0, Id), x_0 = rho * x_1 + sigma * zeta (a Gaussian coupling)."""
    eye = torch.eye(d)
    return GaussianCoupling(
        mu_0=torch.zeros(d),
        mu_1=torch.zeros(d),
        C_00=rho**2 * eye + sigma**2 * eye,
        C_11=eye,
        C_01=rho * eye,
    )


def test_score_identity_matches_closed_form_gaussian():
    """grad log rho_t(x) = -gamma_t^{-1} g_t(x) (eq. 6 / 28).

    For a jointly Gaussian coupling, rho_t is Gaussian with covariance
    Sigma_t, and the identity reduces to

        E[z | I_t = x] = gamma_t Sigma_t^{-1} (x - mu_t),
        -gamma_t^{-1} E[z | I_t = x] = -Sigma_t^{-1} (x - mu_t) = grad log rho_t(x).

    Both ingredients are checked numerically: the empirical marginal of I_t
    must match N(mu_t, Sigma_t), and a kernel-regression estimate of
    E[z | I_t ~ x] must match the closed form gamma_t Sigma_t^{-1} (x - mu_t).
    """
    torch.manual_seed(0)
    d, n = 2, 200_000
    gc = _gaussian_coupling(d)
    alpha, beta, gamma = 0.5, 0.5, 0.5
    x1 = torch.randn(n, d)
    x0 = 0.6 * x1 + 0.5 * torch.randn(n, d)
    z = torch.randn(n, d)
    i_t = alpha * x0 + beta * x1 + gamma * z

    # (i) the marginal of I_t is the Gaussian predicted by Theorem 3.1
    cov_i = gc.cov(alpha, beta, torch.tensor(gamma**2))
    assert torch.allclose(i_t.mean(0), torch.zeros(d), atol=0.02)
    assert torch.allclose(torch.cov(i_t.T), cov_i, atol=0.02)

    # (ii) kernel regression for g_t(x) = E[z | I_t = x] at a few test points
    x_test = torch.tensor([[0.5, -0.3], [-0.4, 0.2]])
    g_closed = gamma * (x_test @ torch.linalg.inv(cov_i).T)
    bandwidth = 0.25
    for k in range(x_test.shape[0]):
        dist2 = ((i_t - x_test[k]) ** 2).sum(-1)
        w = torch.exp(-dist2 / (2 * bandwidth**2))
        g_kernel = (w[:, None] * z).sum(0) / w.sum()
        assert torch.allclose(g_kernel, g_closed[k], atol=0.05)
        # and the score identity -gamma^{-1} g_t(x) = grad log rho_t(x)
        assert torch.allclose(-g_kernel / gamma, -(x_test[k] @ torch.linalg.inv(cov_i).T), atol=0.1)


def test_velocity_objective_minimised_by_conditional_expectation():
    """For a Gaussian coupling the optimal velocity is affine and equals
    E[I_dot_t | I_t = x]; a linear model trained with L_b must recover it."""
    torch.manual_seed(0)
    d, n = 2, 50_000
    rho, sigma = 0.6, 0.5
    gc = _gaussian_coupling(d, rho, sigma)
    alpha, beta, gamma = 0.5, 0.5, 0.0
    x1 = torch.randn(n, d)
    x0 = rho * x1 + sigma * torch.randn(n, d)
    i_t = alpha * x0 + beta * x1
    i_dot = -x0 + x1

    # least squares solution of E|b(I_t) - I_dot|^2 over affine models
    A = torch.cat([i_t, torch.ones(n, 1)], dim=1)
    theta = torch.linalg.lstsq(A, i_dot).solution

    # closed-form optimal velocity
    x_probe = torch.randn(64, d)
    b_closed = gc.optimal_velocity(x_probe, alpha, beta, -1.0, 1.0)
    b_fit = torch.cat([x_probe, torch.ones(64, 1)], dim=1) @ theta
    assert torch.allclose(b_closed, b_fit, atol=0.02)


def test_velocity_loss_is_mse_up_to_constant():
    """|b|^2 - 2 I_dot . b = |b - I_dot|^2 - |I_dot|^2."""
    torch.manual_seed(0)
    b = torch.randn(16, 4)
    i_dot = torch.randn(16, 4)
    lhs = flatten_sum(b**2) - 2 * flatten_sum(b * i_dot)
    rhs = flatten_sum((b - i_dot) ** 2) - flatten_sum(i_dot**2)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_transport_cost_bound_of_proposition_31():
    """int_0^1 E|I_dot_t|^2 dt bounds E|X_1(x_0) - x_0|^2.

    With the true (affine) velocity of a Gaussian coupling the flow is a
    deterministic linear map, so both sides can be computed accurately.
    """
    torch.manual_seed(0)
    d, sigma, n = 2, 0.5, 20_000
    rho = 1.0
    alpha, beta = 1.0, 0.0
    gc = _gaussian_coupling(d, rho, sigma)
    x1 = torch.randn(n, d)
    x0 = rho * x1 + sigma * torch.randn(n, d)

    # integrate the flow with the closed-form velocity using fine Euler steps
    x = x0.clone()
    steps = 2000
    for k in range(steps):
        t = k / steps
        x = x + (1.0 / steps) * gc.optimal_velocity(x, 1 - t, t, -1.0, 1.0)
    cost = ((x - x0) ** 2).sum(-1).mean()

    # int_0^1 E|I_dot_t|^2 dt = E|x_1 - x_0|^2 = d sigma^2 for this coupling
    bound = torch.tensor(d * sigma**2)
    assert cost <= bound + 1e-3
    # and the flow recovers the target marginal (variance 1)
    assert torch.allclose(x.var(dim=0).mean(), torch.tensor(1.0), atol=0.05)
