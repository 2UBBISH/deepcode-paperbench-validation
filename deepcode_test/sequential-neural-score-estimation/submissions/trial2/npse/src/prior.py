"""Prior score handling for NPSE, NLSE, and SNPSE-C.

This module provides two complementary ways of obtaining the *perturbed prior
score* :math:`\\nabla_\\theta \\log p_t(\\theta_t)`:

1. **Analytic scores** for priors whose perturbed density is available in closed
   form under the SDE marginal transition :math:`p_{t|0}`.  We implement
   uniform priors and Gaussian-mixture priors.  The formulas are written for a
   general linear-Gaussian transition

   .. math::
       \\theta_t = \\alpha_t \\theta_0 + \\sqrt{\\beta_t}\\,\\varepsilon,
       \\qquad \\varepsilon \\sim \\mathcal{N}(0, I),

   which covers both the VE SDE (:math:`\\alpha_t = 1`) and the VP SDE
   (:math:`\\alpha_t = e^{-\\frac12 \\int_0^t \\beta_s ds}`).

2. **Learned prior scores** for implicit priors, trained by prior denoising
   score matching:

   .. math::
       J_\\mathrm{pri} = \\frac12 \\int_0^T \\lambda_t\\,
       \\mathbb{E}_{p(\\theta_0) p_{t|0}(\\theta_t|\\theta_0)}
       \\big[ \\| s_\\psi(\\theta_t, t) -
       \\nabla_\\theta \\log p_{t|0}(\\theta_t|\\theta_0) \\|^2 \\big]\\, dt.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import torch
from torch.distributions import MultivariateNormal

from .losses import prior_dsm_loss, sample_times
from .networks import PriorScoreNetwork, compute_standardization_stats
from .sde import SDE

__all__ = [
    "AnalyticPriorScore",
    "UniformPriorScore",
    "GaussianMixturePriorScore",
    "train_prior_score_network",
    "make_prior_score_fn",
    "uniform_perturbed_log_prob",
    "uniform_perturbed_score",
    "gaussian_mixture_perturbed_log_prob",
    "gaussian_mixture_perturbed_score",
    "normal_cdf",
]


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Element-wise standard-normal CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _as_tensor(value: Any, theta: torch.Tensor) -> torch.Tensor:
    """Convert a python/tensor value to a tensor on ``theta``'s device/dtype."""
    if torch.is_tensor(value):
        return value.to(device=theta.device, dtype=theta.dtype)
    return torch.tensor(value, device=theta.device, dtype=theta.dtype)


def _mean_scale(sde: SDE, t: Any, theta: torch.Tensor) -> torch.Tensor:
    """Return the marginal mean coefficient ``alpha_t`` broadcast to theta shape.

    ``sde.marginal_mean(theta0, t)`` is linear in ``theta0`` for both VE and VP,
    so passing an all-ones tensor returns ``alpha_t`` (element-wise).
    """
    ones = torch.ones_like(theta)
    return sde.marginal_mean(ones, t)


def _marginal_var_broadcast(sde: SDE, t: Any, theta: torch.Tensor) -> torch.Tensor:
    """Return the marginal variance broadcastable against ``theta``'s trailing dims."""
    var = _as_tensor(sde.marginal_var(t), theta)
    while var.dim() < theta.dim():
        var = var.unsqueeze(-1)
    return var


class AnalyticPriorScore:
    """Base class for analytic perturbed-prior score models.

    Subclasses implement ``_log_prob_flat`` for a flattened ``(N, d)`` theta
    tensor.  The score is obtained by reverse-mode automatic differentiation.
    """

    sde: SDE

    def _log_prob_flat(self, theta: torch.Tensor, t: Any) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, theta: torch.Tensor, t: Any) -> torch.Tensor:
        """Approximate/unnormalised perturbed-prior log density."""
        flat = theta.reshape(-1, theta.shape[-1])
        out = self._log_prob_flat(flat, t)
        return out.reshape(theta.shape[:-1])

    def score(self, theta: torch.Tensor, t: Any) -> torch.Tensor:
        """Perturbed-prior score ``grad_theta log p_t(theta_t)``."""
        original_shape = theta.shape
        flat = theta.reshape(-1, theta.shape[-1])
        if not flat.requires_grad:
            flat = flat.detach().requires_grad_(True)
        log_prob = self._log_prob_flat(flat, t)
        (grad,) = torch.autograd.grad(log_prob.sum(), flat, create_graph=True)
        return grad.reshape(original_shape)

    def __call__(self, theta: torch.Tensor, t: Any) -> torch.Tensor:
        return self.score(theta, t)


class UniformPriorScore(AnalyticPriorScore):
    """Analytic perturbed score for an independent uniform prior.

    For a uniform prior :math:`\\theta_0 \\sim U(a, b)` (per dimension) and a
    linear-Gaussian transition :math:`\\theta_t = \\alpha_t\\theta_0 +
    \\sqrt{\\beta_t}\\varepsilon`, the perturbed density is proportional to

    .. math::
        p_t(\\theta_t) \\propto \\prod_i \\Big[
            \\Phi\\big((\\alpha_t b_i - \\theta_{t,i})/\\sqrt{\\beta_t}\\big)
          - \\Phi\\big((\\alpha_t a_i - \\theta_{t,i})/\\sqrt{\\beta_t}\\big)
        \\Big].

    The omitted normalisation does not depend on :math:`\\theta_t`, so it does
    not affect the score.
    """

    def __init__(self, sde: SDE, low: Union[float, Any], high: Union[float, Any]):
        self.sde = sde
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)

    def _log_prob_flat(self, theta: torch.Tensor, t: Any) -> torch.Tensor:
        low = self.low.to(theta)
        high = self.high.to(theta)
        alpha = _mean_scale(self.sde, t, theta)
        std = torch.sqrt(_marginal_var_broadcast(self.sde, t, theta))
        a = alpha * low
        b = alpha * high
        diff = normal_cdf((b - theta) / std) - normal_cdf((a - theta) / std)
        return torch.log(diff.clamp_min(1e-12)).sum(dim=-1)


class GaussianMixturePriorScore(AnalyticPriorScore):
    """Analytic perturbed score for a Gaussian-mixture prior.

    If :math:`p(\\theta_0) = \\sum_i w_i \\mathcal{N}(\\mu_i, \\Sigma_i)` then
    under the linear-Gaussian transition

    .. math::
        p_t(\\theta_t) = \\sum_i w_i\\,
            \\mathcal{N}\\big(\\alpha_t\\mu_i,\\,
            \\alpha_t^2 \\Sigma_i + \\beta_t I\\big).
    """

    def __init__(
        self,
        sde: SDE,
        weights: Any,
        means: Any,
        covariances: Any,
    ):
        self.sde = sde
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.means = torch.as_tensor(means, dtype=torch.float32)
        cov = torch.as_tensor(covariances, dtype=torch.float32)
        if cov.dim() == 0:
            # A scalar variance is interpreted as an isotropic diagonal covariance.
            cov = cov.reshape(1)
        self.covariances = cov
        self.log_weights = torch.log(self.weights.clamp_min(1e-12))

    def _log_prob_flat(self, theta: torch.Tensor, t: Any) -> torch.Tensor:
        means = self.means.to(theta)
        covs = self.covariances.to(theta)
        log_weights = self.log_weights.to(theta)

        if means.dim() == 1:
            means = means.unsqueeze(0)
        if covs.dim() == 1:
            # A single diagonal covariance shared by all components (or one component).
            if covs.shape[0] == means.shape[-1]:
                covs = covs.unsqueeze(0).expand(means.shape[0], -1)
            else:
                covs = covs.unsqueeze(0)

        n, d = theta.shape
        alpha = _mean_scale(self.sde, t, theta)  # (N, d)
        var = _marginal_var_broadcast(self.sde, t, theta)  # (N, 1) or (1, 1)
        alpha_s = alpha[:, 0]  # (N,) -- isotropic coefficient
        alpha2 = alpha_s.pow(2)

        component_log_probs = []
        k = means.shape[0]
        for i in range(k):
            mu = alpha * means[i]
            cov_i = covs[i]
            if cov_i.dim() == 1:
                cov_diag = alpha2.unsqueeze(-1) * cov_i + var
                scale = torch.sqrt(cov_diag.clamp_min(1e-12))
                log_prob_i = -0.5 * (
                    ((theta - mu) / scale) ** 2
                    + torch.log(2.0 * math.pi * scale**2)
                ).sum(dim=-1)
            else:
                eye = torch.eye(d, device=theta.device, dtype=theta.dtype)
                cov_scaled = alpha2.unsqueeze(-1).unsqueeze(-1) * cov_i
                cov_full = cov_scaled + var.unsqueeze(-1) * eye
                dist = MultivariateNormal(mu, covariance_matrix=cov_full)
                log_prob_i = dist.log_prob(theta)
            component_log_probs.append(log_prob_i)

        stacked = torch.stack(component_log_probs, dim=-1)  # (N, K)
        return torch.logsumexp(stacked + log_weights, dim=-1)


def train_prior_score_network(
    prior_sampler: Callable[[int], torch.Tensor],
    theta_dim: int,
    sde: SDE,
    num_samples: int = 10000,
    hidden_dim: int = 256,
    time_embedding_dim: int = 64,
    lr: float = 1e-4,
    steps: int = 2000,
    batch_size: int = 256,
    patience: int = 200,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> PriorScoreNetwork:
    """Train a prior score network for an implicit prior by DSM.

    Parameters
    ----------
    prior_sampler:
        Callable returning ``(n, theta_dim)`` prior samples.
    theta_dim:
        Parameter dimensionality.
    sde:
        Forward SDE used for perturbation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    theta0 = prior_sampler(num_samples).to(device=device, dtype=dtype)
    if theta0.ndim == 1:
        theta0 = theta0.reshape(-1, theta_dim)

    net = PriorScoreNetwork(
        theta_dim=theta_dim,
        hidden_dim=hidden_dim,
        time_embedding_dim=time_embedding_dim,
    ).to(device=device, dtype=dtype)

    # Standardize perturbed parameters.
    t_stats = torch.rand(theta0.shape[0], device=device, dtype=dtype) * sde.T
    theta_t_stats = sde.transition_sample(theta0, t_stats)
    mean, std = compute_standardization_stats(theta_t_stats)
    net.set_standardization(mean, std)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for step in range(steps):
        idx = torch.randint(0, theta0.shape[0], (batch_size,), device=device)
        theta_batch = theta0[idx]
        t_batch = sample_times(batch_size, sde.T, device=device, dtype=dtype)
        loss = prior_dsm_loss(sde, net, theta_batch, t_batch, weighting="g2")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"[prior] early stop at step {step} loss={best_loss:.6f}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def make_prior_score_fn(
    prior: Union[str, AnalyticPriorScore, PriorScoreNetwork, Callable],
    sde: SDE,
    theta_dim: Optional[int] = None,
    **kwargs: Any,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """Factory returning a callable ``(theta, t) -> prior score``.

    ``prior`` may be one of:

    * an :class:`AnalyticPriorScore` instance;
    * a :class:`~npse.networks.PriorScoreNetwork` instance;
    * an arbitrary callable with signature ``(theta, t)``;
    * a string: ``"uniform"``, ``"gaussian"``, ``"gaussian_mixture"``, or
      ``"implicit"`` (requires ``prior_sampler`` in ``kwargs``).
    """
    if isinstance(prior, (AnalyticPriorScore, PriorScoreNetwork)):
        return prior
    if callable(prior) and not isinstance(prior, str):
        return prior

    if not isinstance(prior, str):
        raise TypeError(f"Unsupported prior specification: {type(prior)}")

    name = prior.lower()
    if name == "uniform":
        return UniformPriorScore(
            sde, low=kwargs.get("low", -1.0), high=kwargs.get("high", 1.0)
        )
    if name in ("gaussian", "normal"):
        if theta_dim is None:
            theta_dim = kwargs.get("dim")
        if theta_dim is None:
            raise ValueError("'dim' or 'theta_dim' is required for a Gaussian prior")
        mean = kwargs.get("mean", 0.0)
        var = kwargs.get("var", 1.0)
        means = torch.full((1, theta_dim), float(mean), dtype=torch.float32)
        covariances = torch.full((1, theta_dim), float(var), dtype=torch.float32)
        return GaussianMixturePriorScore(sde, [1.0], means, covariances)
    if name in ("gaussian_mixture", "gmm", "mixture"):
        return GaussianMixturePriorScore(
            sde,
            weights=kwargs["weights"],
            means=kwargs["means"],
            covariances=kwargs["covariances"],
        )
    if name == "implicit":
        prior_sampler = kwargs.get("prior_sampler")
        if prior_sampler is None:
            raise ValueError("'prior_sampler' is required for an implicit prior")
        if theta_dim is None:
            raise ValueError("'theta_dim' is required for an implicit prior")
        train_kwargs = {
            k: kwargs[k]
            for k in (
                "num_samples",
                "hidden_dim",
                "time_embedding_dim",
                "lr",
                "steps",
                "batch_size",
                "patience",
                "device",
                "dtype",
                "seed",
                "verbose",
            )
            if k in kwargs
        }
        return train_prior_score_network(
            prior_sampler, theta_dim, sde, **train_kwargs
        )

    raise ValueError(f"Unknown prior specification: {prior!r}")


def uniform_perturbed_log_prob(
    theta: torch.Tensor, t: Any, sde: SDE, low: Any, high: Any
) -> torch.Tensor:
    """Unnormalised perturbed log density for a uniform prior."""
    return UniformPriorScore(sde, low, high).log_prob(theta, t)


def uniform_perturbed_score(
    theta: torch.Tensor, t: Any, sde: SDE, low: Any, high: Any
) -> torch.Tensor:
    """Perturbed prior score for a uniform prior."""
    return UniformPriorScore(sde, low, high).score(theta, t)


def gaussian_mixture_perturbed_log_prob(
    theta: torch.Tensor,
    t: Any,
    sde: SDE,
    weights: Any,
    means: Any,
    covariances: Any,
) -> torch.Tensor:
    """Perturbed log density for a Gaussian-mixture prior."""
    return GaussianMixturePriorScore(sde, weights, means, covariances).log_prob(
        theta, t
    )


def gaussian_mixture_perturbed_score(
    theta: torch.Tensor,
    t: Any,
    sde: SDE,
    weights: Any,
    means: Any,
    covariances: Any,
) -> torch.Tensor:
    """Perturbed prior score for a Gaussian-mixture prior."""
    return GaussianMixturePriorScore(sde, weights, means, covariances).score(
        theta, t
    )
