"""Forward and reverse SDE machinery for Neural Posterior Score Estimation.

This module implements the two SDE families used in the paper:

* **Variance Exploding (VE / SMLD) SDE**

  .. math::
      d\\theta_t = \\sigma_{\\min} \\left(\\frac{\\sigma_{\\max}}
      {\\sigma_{\\min}}\\right)^t
      \\sqrt{2 \\log\\left(\\frac{\\sigma_{\\max}}{\\sigma_{\\min}}\\right)}
      \\, dW_t

  whose transition kernel is

  .. math::
      p_{t|0}(\\theta_t \\mid \\theta_0) =
      \\mathcal{N}\\left(\\theta_t; \\theta_0,
      \\sigma_{\\min}^2 (\\sigma_{\\max}/\\sigma_{\\min})^{2t} I\\right).

* **Variance Preserving (VP) SDE**

  .. math::
      d\\theta_t = -\\tfrac{1}{2}\\beta_t\\theta_t\\,dt
      + \\sqrt{\\beta_t}\\,dW_t,
      \\quad
      \\beta_t = \\beta_{\\min} + t(\\beta_{\\max}-\\beta_{\\min}),

  whose transition kernel is

  .. math::
      p_{t|0}(\\theta_t \\mid \\theta_0) =
      \\mathcal{N}\\left(\\theta_t; e^{-\\frac12\\int_0^t\\beta_s ds}
      \\theta_0, (1 - e^{-\\int_0^t\\beta_s ds}) I\\right).

Both SDEs expose the conditional (denoising) score
:math:`\\nabla_{\\theta_t}\\log p_{t|0}(\\theta_t|\\theta_0)` in closed form,
which is the regression target of all denoising-score-matching objectives.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


def ensure_tensor(
    t,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert a Python float/int or array to a torch tensor."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=dtype if dtype is not None else torch.float32)
    if device is not None:
        t = t.to(device)
    return t


class SDE(nn.Module):
    """Base class for score-based diffusion SDEs defined on :math:`[0, T]`.

    Parameters
    ----------
    T:
        Terminal time. Time flows from ``0`` (clean data) to ``T`` (prior).
    """

    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = float(T)

    # ------------------------------------------------------------------
    # Forward dynamics
    # ------------------------------------------------------------------
    def f(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient :math:`f(\\theta, t)`."""
        raise NotImplementedError

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient :math:`g(t)` (scalar, broadcastable)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transition kernel  p_{t|0}
    # ------------------------------------------------------------------
    def marginal_mean(
        self, theta0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Mean of :math:`p_{t|0}(\\theta_t \\mid \\theta_0)`."""
        raise NotImplementedError

    def marginal_var(self, t: torch.Tensor) -> torch.Tensor:
        """Isotropic marginal variance of :math:`p_{t|0}` at time ``t``."""
        raise NotImplementedError

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        return self.marginal_var(t).clamp_min(1e-12).sqrt()

    def transition_sample(
        self, theta0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Draw :math:`\\theta_t \\sim p_{t|0}(\\cdot \\mid \\theta_0)`."""
        mean = self.marginal_mean(theta0, t)
        std = self.marginal_std(t)
        noise = torch.randn_like(theta0)
        return mean + std * noise

    def transition_score(
        self, theta_t: torch.Tensor, theta0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Conditional score :math:`\\nabla_{\\theta_t}\\log p_{t|0}`."""
        mean = self.marginal_mean(theta0, t)
        var = self.marginal_var(t).clamp_min(1e-8)
        return -(theta_t - mean) / var

    # ------------------------------------------------------------------
    # Reverse-time SDE and probability-flow ODE drifts
    # ------------------------------------------------------------------
    def reverse_drift(
        self, theta: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Reverse SDE drift: :math:`f - g^2 \\nabla_\\theta\\log p_t`."""
        return self.f(theta, t) - (self.g(t) ** 2) * score

    def ode_velocity(
        self, theta: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Probability-flow ODE velocity: :math:`f - \\tfrac12 g^2 s`."""
        return self.f(theta, t) - 0.5 * (self.g(t) ** 2) * score

    # ------------------------------------------------------------------
    # Prior at time T
    # ------------------------------------------------------------------
    def prior_sample(
        self, n: int, d: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Draw samples from the SDE prior :math:`p_T(\\theta_T)`."""
        raise NotImplementedError

    def prior_logp(self, theta_T: torch.Tensor) -> torch.Tensor:
        """Evaluate :math:`\\log p_T(\\theta_T)` per sample."""
        raise NotImplementedError


class VESDE(SDE):
    """Variance-exploding SDE (SMLD / NCSN style).

    Parameters
    ----------
    sigma_min:
        Minimum noise scale. The paper uses ``0.01`` for two-dimensional tasks
        and SIR/Two Moons and ``0.05`` otherwise.
    sigma_max:
        Maximum noise scale. Chosen by Technique 1 of Song & Ermon (2020),
        i.e. the maximum pairwise Euclidean distance of the (perturbed)
        training samples.
    """

    def __init__(
        self,
        sigma_min: float = 0.05,
        sigma_max: float = 1.0,
        T: float = 1.0,
    ):
        super().__init__(T)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be strictly larger than sigma_min")
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        t = ensure_tensor(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def f(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(theta)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        t = ensure_tensor(t)
        return (
            self.sigma_min
            * (self.sigma_max / self.sigma_min) ** t
            * math.sqrt(2.0 * self._log_ratio)
        )

    def marginal_mean(
        self, theta0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return theta0

    def marginal_var(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) ** 2

    def prior_sample(
        self, n: int, d: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        std = self.sigma(torch.tensor(self.T, device=device))
        return std * torch.randn(n, d, device=device)

    def prior_logp(self, theta_T: torch.Tensor) -> torch.Tensor:
        var = self.marginal_var(torch.tensor(self.T, device=theta_T.device))
        return isotropic_gaussian_logp(theta_T, 0.0, var)


class VPSDE(SDE):
    """Variance-preserving SDE with a linear :math:`\\beta` schedule.

    Parameters
    ----------
    beta_min, beta_max:
        Endpoints of the linear noise schedule
        :math:`\\beta_t = \\beta_{\\min} + t(\\beta_{\\max}-\\beta_{\\min})`.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 11.0,
        T: float = 1.0,
    ):
        super().__init__(T)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t = ensure_tensor(t)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def beta_int(self, t: torch.Tensor) -> torch.Tensor:
        """Integral :math:`\\int_0^t \\beta_s\\,ds` for the linear schedule."""
        t = ensure_tensor(t)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def f(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta(t) * theta

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta(t).clamp_min(1e-6).sqrt()

    def marginal_mean(
        self, theta0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta_int(t)) * theta0

    def marginal_var(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-self.beta_int(t))

    def prior_sample(
        self, n: int, d: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        # At t=T the marginal is N(0, (1 - e^{-int beta}) I), which is very
        # close to N(0, I) for beta_max=11, T=1.  Use the exact variance.
        var = self.marginal_var(torch.tensor(self.T, device=device))
        return torch.sqrt(var) * torch.randn(n, d, device=device)

    def prior_logp(self, theta_T: torch.Tensor) -> torch.Tensor:
        var = self.marginal_var(torch.tensor(self.T, device=theta_T.device))
        return isotropic_gaussian_logp(theta_T, 0.0, var)


def isotropic_gaussian_logp(
    x: torch.Tensor,
    mean: torch.Tensor | float,
    var: torch.Tensor | float,
) -> torch.Tensor:
    """Log probability of an isotropic Gaussian, summed over the last dim.

    Parameters
    ----------
    x:
        Input tensor of shape ``(..., d)``.
    mean:
        Scalar mean.
    var:
        Scalar variance.

    Returns
    -------
        Per-sample log probabilities of shape ``x.shape[:-1]``.
    """
    d = x.shape[-1]
    var = ensure_tensor(var, device=x.device, dtype=x.dtype).clamp_min(1e-12)
    const = -0.5 * d * (math.log(2.0 * math.pi) + torch.log(var))
    return const - 0.5 * ((x - mean) ** 2).sum(dim=-1) / var


def estimate_sigma_max(
    theta_samples: torch.Tensor,
    max_samples: int = 2000,
) -> float:
    """Estimate ``sigma_max`` using Technique 1 of Song & Ermon (2020).

    Technique 1 sets the maximum noise scale to the maximum pairwise Euclidean
    distance between (a subsample of) the training data points.

    Parameters
    ----------
    theta_samples:
        Tensor of shape ``(n, d)``.
    max_samples:
        Cap the number of samples used for the pairwise-distance computation.

    Returns
    -------
        Estimated maximum noise scale as a Python float.
    """
    with torch.no_grad():
        x = theta_samples.detach().cpu()
        if x.shape[0] > max_samples:
            idx = torch.randperm(x.shape[0])[:max_samples]
            x = x[idx]
        if x.shape[0] < 2:
            return 1.0
        diff = x.unsqueeze(0) - x.unsqueeze(1)  # (n, n, d)
        dist = diff.norm(dim=-1)
        # Ignore diagonal zeros and return maximum.
        return float(dist.max().item())


def get_sde(sde_type: str, **kwargs) -> SDE:
    """Instantiate an SDE from its string identifier.

    Parameters
    ----------
    sde_type:
        ``"ve"`` or ``"vp"`` (case-insensitive).
    **kwargs:
        Forwarded to :class:`VESDE` or :class:`VPSDE`.

    Returns
    -------
        An :class:`SDE` instance.
    """
    sde_type = sde_type.lower().strip()
    if sde_type in ("ve", "vesde", "smld"):
        return VESDE(**kwargs)
    if sde_type in ("vp", "vpsde"):
        return VPSDE(**kwargs)
    raise ValueError(f"Unknown SDE type: {sde_type!r}")


def reverse_sde_step(
    sde: SDE,
    theta: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    score: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single Euler-Maruyama reverse-SDE step (backward in time).

    Useful for quick validation and for annealed reverse sampling where an ODE
    solver is not appropriate.  For the paper's primary sampling procedure the
    probability-flow ODE is used instead (see :mod:`npse.src.sampling`).
    """
    drift = sde.reverse_drift(theta, t, score)
    diffusion = sde.g(t)
    if noise is None:
        noise = torch.randn_like(theta)
    return theta - drift * dt + diffusion * torch.sqrt(dt.abs()) * noise


__all__ = [
    "SDE",
    "VESDE",
    "VPSDE",
    "get_sde",
    "estimate_sigma_max",
    "ensure_tensor",
    "isotropic_gaussian_logp",
    "reverse_sde_step",
]
