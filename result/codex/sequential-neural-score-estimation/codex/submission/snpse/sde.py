"""Forward noising processes (SDEs) used by (TS)NPSE.

Implements the two choices of forward noising process described in
Appendix E.3.1 of the paper:

* the variance exploding SDE (VE SDE),

      d theta_t = sigma_min (sigma_min / sigma_max)^t
                  sqrt(2 log(sigma_max / sigma_min)) dw_t,   t in (0, 1]

* the variance preserving SDE (VP SDE),

      d theta_t = -1/2 beta_t theta_t dt + sqrt(beta_t) dw_t,  t in (0, 1]
      beta_t = beta_min + t (beta_max - beta_min)

For both processes the transition density p_{t|0}(theta_t | theta_0) is
Gaussian with a closed-form mean and standard deviation, so the denoising score
matching target ``grad_{theta_t} log p_{t|0}(theta_t | theta_0)`` is available
in closed form (Eq. 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


class SDE:
    """Interface implemented by the forward noising processes.

    Subclasses must provide the transition density of the forward process
    ``p_{t|0}(theta_t | theta_0)`` (through ``transition``), the drift and
    diffusion coefficients of the forward SDE (through
    ``drift_and_diffusion``), the reference distribution at time ``T``, and the
    weighting function ``lambda_t`` of the score matching objective.
    """

    name: str = "sde"
    T: float = 1.0

    def transition(self, theta0: torch.Tensor, t: torch.Tensor):  # pragma: no cover
        raise NotImplementedError

    def transition_score(self, theta_t, theta0, t):  # pragma: no cover
        raise NotImplementedError

    def drift_and_diffusion(self, theta: torch.Tensor, t: torch.Tensor):  # pragma: no cover
        raise NotImplementedError

    def sample_reference(self, shape, device=None, dtype=None):  # pragma: no cover
        raise NotImplementedError

    def log_reference(self, theta: torch.Tensor):  # pragma: no cover
        raise NotImplementedError

    def loss_weight(self, t: torch.Tensor):  # pragma: no cover
        raise NotImplementedError


def compute_sigma_max(
    theta: torch.Tensor,
    technique: int = 1,
    num_pairwise: Optional[int] = None,
) -> float:
    """Choose ``sigma_max`` for the VE SDE.

    Technique 1 of Song & Ermon (2020) ("Improved Techniques for Training
    Score-Based Generative Models") sets ``sigma_max`` to the maximum pairwise
    distance between the training data points.  The addendum of this
    reproduction task specifies that, for the *sequential* methods, only the
    training data points available in the first round should be used to compute
    ``sigma_max``; that is handled by the caller (see ``TSNPSE``).

    Args:
        theta: (N, d) tensor of parameter samples.
        technique: only "Technique 1" (max pairwise distance) is implemented;
            kept as an explicit argument for clarity.
        num_pairwise: optionally subsample to this many points before
            computing pairwise distances (keeps the cost O(num_pairwise^2)).
    """
    if technique != 1:
        raise ValueError("Only Technique 1 (max pairwise distance) is supported")
    theta = theta.detach()
    if num_pairwise is not None and theta.shape[0] > num_pairwise:
        idx = torch.randperm(theta.shape[0])[:num_pairwise]
        theta = theta[idx]
    # pairwise distances
    diff = theta[:, None, :] - theta[None, :, :]
    dist = torch.sqrt((diff ** 2).sum(-1))
    return float(dist.max().item())


@dataclass
class VESDE:
    """Variance exploding SDE (Appendix E.3.1)."""

    sigma_min: float = 0.01
    sigma_max: float = 10.0
    T: float = 1.0
    name: str = "ve"

    def _ratio(self) -> float:
        return self.sigma_max / self.sigma_min

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """Marginal standard deviation of the transition density."""
        return self.sigma_min * self._ratio() ** t

    def mean_factor(self, t: torch.Tensor) -> torch.Tensor:
        """``mean = mean_factor(t) * theta_0``."""
        return torch.ones_like(t)

    def transition(self, theta0: torch.Tensor, t: torch.Tensor):
        """Return (mean, std) of ``p_{t|0}(.|theta_0)``, broadcastable to theta0."""
        t = t.reshape(-1, *([1] * (theta0.dim() - 1)))
        mean = self.mean_factor(t) * theta0
        std = self.std(t)
        return mean, std

    def transition_score(self, theta_t: torch.Tensor, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, std = self.transition(theta0, t)
        return -(theta_t - mean) / (std ** 2)

    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(theta)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, *([1] * 1))
        g = (
            self.sigma_min
            * self._ratio() ** t
            * math.sqrt(2.0 * math.log(self._ratio()))
        )
        return g

    def drift_and_diffusion(self, theta: torch.Tensor, t: torch.Tensor):
        """Return (f(theta, t), g(t)) with ``g`` broadcast to ``theta``'s shape."""
        t_b = t.reshape(-1, *([1] * (theta.dim() - 1)))
        f = torch.zeros_like(theta)
        g = (
            self.sigma_min
            * self._ratio() ** t_b
            * math.sqrt(2.0 * math.log(self._ratio()))
        )
        return f, g

    # -- reference distribution at t = T -----------------------------------
    def reference_std(self) -> torch.Tensor:
        return torch.tensor(self.sigma_max)

    def sample_reference(self, shape, device=None, dtype=None) -> torch.Tensor:
        return self.sigma_max * torch.randn(shape, device=device, dtype=dtype)

    def log_reference(self, theta: torch.Tensor) -> torch.Tensor:
        d = theta.shape[-1]
        var = self.sigma_max ** 2
        return -0.5 * d * math.log(2 * math.pi * var) - 0.5 * (theta ** 2).sum(-1) / var

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """lambda_t = sigma_t^2 (see ``snpse.diffusion`` for a discussion)."""
        return self.std(t) ** 2


@dataclass
class VPSDE:
    """Variance preserving SDE (Appendix E.3.1)."""

    beta_min: float = 0.1
    beta_max: float = 11.0
    T: float = 1.0
    name: str = "vp"

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """int_0^t beta_s ds = beta_min t + (beta_max-beta_min) t^2 / 2."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def mean_factor(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.integral_beta(t))

    def std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-self.integral_beta(t)), min=1e-12))

    def transition(self, theta0: torch.Tensor, t: torch.Tensor):
        t = t.reshape(-1, *([1] * (theta0.dim() - 1)))
        mean = self.mean_factor(t) * theta0
        std = self.std(t)
        return mean, std

    def transition_score(self, theta_t: torch.Tensor, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, std = self.transition(theta0, t)
        return -(theta_t - mean) / (std ** 2)

    def drift_and_diffusion(self, theta: torch.Tensor, t: torch.Tensor):
        t_b = t.reshape(-1, *([1] * (theta.dim() - 1)))
        beta = self.beta(t_b)
        f = -0.5 * beta * theta
        g = torch.sqrt(beta)
        return f, g

    def reference_std(self) -> torch.Tensor:
        return torch.tensor(1.0)

    def sample_reference(self, shape, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype)

    def log_reference(self, theta: torch.Tensor) -> torch.Tensor:
        d = theta.shape[-1]
        return -0.5 * d * math.log(2 * math.pi) - 0.5 * (theta ** 2).sum(-1)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """lambda_t = 1 - exp(-int beta) = sigma_t^2."""
        return self.std(t) ** 2


def get_sde(name: str, **kwargs):
    name = name.lower()
    if name in ("ve", "vesde", "variance_exploding"):
        return VESDE(**kwargs)
    if name in ("vp", "vpsde", "variance_preserving"):
        return VPSDE(**kwargs)
    raise ValueError(f"Unknown SDE: {name}")
