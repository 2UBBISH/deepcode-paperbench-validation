"""Perturbed prior score (Appendix B.2).

Several of the sequential methods in Section 3.2 require the score of the
*perturbed* prior ``grad_theta log p_t(theta_t)``, where
``p_t(theta_t) = int p_{t|0}(theta_t | theta_0) p(theta_0) d theta_0``.

For the VE SDE (``f = 0``, ``g(t) = tau_t``) this convolution is available in
closed form for two common choices of prior (Appendix B.2.1):

* uniform prior -- a product of Gaussian CDF differences;
* (mixture of) Gaussian prior -- a (mixture of) wider Gaussian(s).

When neither applies we fall back to estimating the prior score with a
dedicated score network, Algorithm 2 / Eq. (64).
"""

from __future__ import annotations


import torch
import torch.nn as nn

from .networks import ScoreNetwork
from .sde import VESDE


class PerturbedPriorScore(nn.Module):
    """Base class: maps ``(theta_t, t)`` to the perturbed prior score."""

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class UniformVEPriorScore(PerturbedPriorScore):
    """Closed-form perturbed prior score for ``U(a, b)`` under the VE SDE."""

    def __init__(self, low: torch.Tensor, high: torch.Tensor, sde: VESDE) -> None:
        super().__init__()
        self.register_buffer("low", low.clone())
        self.register_buffer("high", high.clone())
        self.sde = sde

    def log_prob(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = self.sde.std(t)
        if tau.dim() == 0:
            tau = tau.reshape(1)
        tau = tau.reshape(-1, 1)
        normal = torch.distributions.Normal(theta_t, tau)
        cdf_high = normal.cdf(self.high.to(theta_t.device))
        cdf_low = normal.cdf(self.low.to(theta_t.device))
        width = (self.high - self.low).to(theta_t.device)
        return (torch.log((cdf_high - cdf_low).clamp_min(1e-30)) - torch.log(width)).sum(-1)

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            theta_t = theta_t.detach().requires_grad_(True)
            lp = self.log_prob(theta_t, t).sum()
            (grad,) = torch.autograd.grad(lp, theta_t)
        return grad


class GaussianVEPriorScore(PerturbedPriorScore):
    """Closed-form perturbed prior score for a diagonal Gaussian prior."""

    def __init__(self, mean: torch.Tensor, var: torch.Tensor, sde: VESDE) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("var", var.clone())
        self.sde = sde

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau2 = self.sde.std(t).reshape(-1, 1) ** 2
        total = self.var.to(theta_t.device) + tau2
        return -(theta_t - self.mean.to(theta_t.device)) / total


class LearnedPriorScore(PerturbedPriorScore):
    """Score network trained on prior samples (Algorithm 2)."""

    def __init__(self, net: ScoreNetwork, standardizer, sde) -> None:
        super().__init__()
        self.net = net
        self.standardizer = standardizer
        self.sde = sde

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(theta_t, torch.zeros(theta_t.shape[0], 1, device=theta_t.device), t)


def estimate_prior_score(
    prior,
    dim_parameters: int,
    sde,
    num_samples: int = 10000,
    standardizer=None,
    lr: float = 1e-4,
    batch_size: int = 200,
    max_iters: int = 3000,
    val_fraction: float = 0.15,
    patience: int = 1000,
    loss_weight: str = "sigma^2",
    seed: int = 0,
    device: str = "cpu",
) -> LearnedPriorScore:
    """Algorithm 2: estimate ``grad log p_t(theta_t)`` with a score network."""
    if standardizer is None:
        from .normalization import Standardizer

        with torch.no_grad():
            theta = prior.sample((num_samples,)).to(device)
        standardizer = Standardizer.fit(theta)

    net = ScoreNetwork(dim_parameters=dim_parameters, dim_data=1).to(device)
    _train_prior_score(
        net,
        prior,
        standardizer,
        sde,
        num_samples=num_samples,
        lr=lr,
        batch_size=batch_size,
        max_iters=max_iters,
        val_fraction=val_fraction,
        patience=patience,
        loss_weight=loss_weight,
        seed=seed,
        device=device,
    )
    return LearnedPriorScore(net, standardizer, sde)


def _train_prior_score(
    net,
    prior,
    standardizer,
    sde,
    num_samples,
    lr,
    batch_size,
    max_iters,
    val_fraction,
    patience,
    loss_weight,
    seed,
    device,
):
    """Train ``net`` to predict the perturbed prior score on prior samples."""
    from .training import train_score_network

    z = prior.sample((num_samples,)).to(device)
    if not torch.is_tensor(z):
        z = torch.as_tensor(z)
    z = standardizer.transform(z)
    dummy_x = torch.zeros(z.shape[0], 1, device=device)
    net, info = train_score_network(
        net,
        sde,
        z,
        dummy_x,
        lr=lr,
        batch_size=batch_size,
        max_iters=max_iters,
        val_fraction=val_fraction,
        patience=patience,
        loss_weight=loss_weight,
        seed=seed,
    )
    return net, info
