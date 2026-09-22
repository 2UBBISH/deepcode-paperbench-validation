"""Forward / reverse stochastic differential equations used by the Simformer.

The Simformer is a score-based diffusion model (Song et al., 2021).  This module
implements the two SDEs that are used in the paper (Appendix A2.1):

* the Variance Exploding SDE  (VESDE):  ``f(x, t) = 0``,
  ``g(t) = sigma_min * (sigma_max / sigma_min) ** t
             * sqrt(2 * log(sigma_max / sigma_min))``
* the Variance Preserving SDE (VPSDE): ``f(x, t) = -0.5 * beta(t) * x``,
  ``g(t) = sqrt(beta(t))`` with ``beta(t) = beta_min + t * (beta_max - beta_min)``

Both are used on the time interval ``[t_min, 1]`` with ``t_min = 1e-5``,
``sigma_max = 15``, ``sigma_min = 1e-4``, ``beta_min = 0.01`` and
``beta_max = 10`` (paper Appendix A2.1).

All SDEs implemented here are *linear* (Gaussian) perturbations, i.e.

    x_t = mu(t) * x_0 + sigma(t) * eps,     eps ~ N(0, I)

which is what allows us to write the conditional score
``grad_{x_t} log p_t(x_t | x_0) = -(x_t - mu(t) x_0) / sigma(t)^2`` in closed
form  (Eq. 3 of the paper).
"""

from __future__ import annotations

import math

import torch


def _as_tensor(t, like: torch.Tensor) -> torch.Tensor:
    """Broadcast ``t`` to a column vector that can be multiplied with ``like``."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=like.dtype, device=like.device)
    t = t.to(dtype=like.dtype, device=like.device)
    while t.dim() < like.dim():
        t = t.unsqueeze(-1)
    return t


class SDE:
    """Base class for the noise processes of the Simformer."""

    name = "sde"
    t_min: float = 1e-5
    t_max: float = 1.0

    # ------------------------------------------------------------------ helpers
    def marginal_coeff(self, t: torch.Tensor):
        """Return ``(mu(t), sigma(t))`` of the perturbation kernel."""
        raise NotImplementedError

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient ``f(x, t)`` of the forward SDE."""
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient ``g(t)`` of the forward SDE."""
        raise NotImplementedError

    # ------------------------------------------------------------------- score
    def marginal_score(self, x_t: torch.Tensor, x_0: torch.Tensor, t) -> torch.Tensor:
        """``grad_{x_t} log p_t(x_t | x_0)`` (the denoising score-matching target)."""
        mu, sigma = self.marginal_coeff(t)
        mu = _as_tensor(mu, x_t)
        sigma = _as_tensor(sigma, x_t)
        return -(x_t - mu * x_0) / (sigma ** 2)

    def loss_weight(self, t) -> torch.Tensor:
        """Positive weighting function ``lambda(t)`` of the score-matching loss.

        We use the standard choice of the variance of the perturbation kernel,
        which for the (V)ESDE is the likelihood weighting of Song et al. (2021)
        ("denoising score matching with annealed Langevin dynamics").
        """
        _, sigma = self.marginal_coeff(t)
        return sigma ** 2

    # --------------------------------------------------------------- sampling
    def prior_sample(self, shape, device=None, dtype=torch.float32) -> torch.Tensor:
        """Sample from the terminal distribution ``p_T`` used to start the reverse SDE."""
        mu, sigma = self.marginal_coeff(torch.tensor(self.t_max))
        mu = float(mu) if not torch.is_tensor(mu) else float(mu)
        sigma = float(sigma) if not torch.is_tensor(sigma) else float(sigma)
        return mu + sigma * torch.randn(shape, device=device, dtype=dtype)

    @property
    def terminal_mean(self) -> float:
        mu, _ = self.marginal_coeff(torch.tensor(self.t_max))
        return float(mu)

    @property
    def terminal_std(self) -> float:
        _, sigma = self.marginal_coeff(torch.tensor(self.t_max))
        return float(sigma)

    def perturb(self, x0: torch.Tensor, t, eps: torch.Tensor | None = None):
        """Sample ``x_t = mu(t) x_0 + sigma(t) eps`` and return ``(x_t, eps, sigma)``."""
        if eps is None:
            eps = torch.randn_like(x0)
        mu, sigma = self.marginal_coeff(t)
        mu = _as_tensor(mu, x0)
        sigma = _as_tensor(sigma, x0)
        return mu * x0 + sigma * eps, eps, sigma

    # ---------------------------------------------------------------- guidance
    def guidance_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Scaling function ``s(t)`` used for (interval) guidance.

        The paper uses ``s(t) = 1 / sigma(t)^2``, i.e. a scaling that is inversely
        proportional to the variance of the marginal scores (Appendix A3.3) and
        which diverges as ``t -> 0``.
        """
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        _, sigma = self.marginal_coeff(t)
        return 1.0 / (sigma ** 2)


class VESDE(SDE):
    """Variance exploding SDE (Song et al., 2021)."""

    name = "vesde"

    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 15.0,
                 t_min: float = 1e-5, t_max: float = 1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.t_min = t_min
        self.t_max = t_max
        self._log_ratio = math.log(sigma_max / sigma_min)

    def marginal_coeff(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mu = torch.ones_like(sigma)
        return mu, sigma

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        _, sigma = self.marginal_coeff(t)
        return sigma * math.sqrt(2.0 * self._log_ratio)


class VPSDE(SDE):
    """Variance preserving SDE (Song et al., 2021)."""

    name = "vpsde"

    def __init__(self, beta_min: float = 0.01, beta_max: float = 10.0,
                 t_min: float = 1e-5, t_max: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_min = t_min
        self.t_max = t_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """``int_0^t beta(s) ds``."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def marginal_coeff(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        alpha = torch.exp(-0.5 * self._integral_beta(t))
        sigma = torch.sqrt(torch.clamp(1.0 - alpha ** 2, min=1e-12))
        return alpha, sigma

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        return -0.5 * _as_tensor(self.beta(t), x) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        return torch.sqrt(self.beta(t))

    def loss_weight(self, t) -> torch.Tensor:
        """Same variance weighting ``lambda(t) = sigma(t)^2`` as for the VESDE."""
        _, sigma = self.marginal_coeff(t)
        return sigma ** 2


def get_sde(name: str, **kwargs) -> SDE:
    """Factory: ``get_sde("vesde")`` / ``get_sde("vpsde")``."""
    name = name.lower()
    if name in ("vesde", "ve", "variance_exploding"):
        return VESDE(**kwargs)
    if name in ("vpsde", "vp", "variance_preserving"):
        return VPSDE(**kwargs)
    raise ValueError(f"Unknown SDE '{name}'.")
