"""Forward noising SDEs for Sequential Neural Posterior Score Estimation (SNPSE).

Implements the two forward noising processes considered in the paper
(App. E.3.1, following Song et al. 2021):

* Variance Exploding SDE (VE SDE)
      d theta_t = sigma_min (sigma_min / sigma_max)^t sqrt(2 log(sigma_max/sigma_min)) dw_t
  with transition density p_{t|0} = N(theta_t | theta_0, sigma(t)^2 I) and
      sigma(t) = sigma_min (sigma_max / sigma_min)^t .

* Variance Preserving SDE (VP SDE)
      d theta_t = -1/2 beta_t theta_t dt + sqrt(beta_t) dw_t,  beta_t = beta_min + t (beta_max - beta_min)
  with transition density
      p_{t|0} = N(theta_t | theta_0 exp(-1/2 B(t)), (1 - exp(-B(t))) I),
      B(t) = int_0^t beta_s ds = beta_min t + 1/2 (beta_max - beta_min) t^2 .

NOTE on signs: the paper's Eq. (E.3.1-4) writes the mean as theta_0 e^{+1/2 int beta} and the
variance as I - I e^{+int beta}.  Those exponents carry a sign typo (they would make the process
variance-exploding, contradicting the name); we use the mathematically correct (Song et al. 2021)
form theta_0 e^{-1/2 int beta} and I (1 - e^{-int beta}), which is the unique form consistent with
the stated SDE (E.3.1-3).  All closed-form score targets are derived from these densities, so the
implementation is self-consistent.

Everything is in the paper's convention t in [0, T] with T = 1.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

T_FINAL = 1.0


def _as_column(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a time tensor of shape (B,) or (B, 1) so that it broadcasts against (B, d)."""
    if t.dim() == 1:
        t = t[:, None]
    if t.dim() < ndim:
        t = t.view(*t.shape, *([1] * (ndim - t.dim())))
    return t


class SDE:
    """Base class for the two forward noising SDEs of App. E.3.1."""

    name: str = "base"

    # ------------------------------------------------------------------ coefficients
    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(theta, t) of Eq. (E.3.1-1)/(E.3.1-3)."""
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) (scalar per sample), returned broadcastable to theta's shape."""
        raise NotImplementedError

    def diffusion_sq(self, t: torch.Tensor) -> torch.Tensor:
        g = self.diffusion(t)
        return g * g

    # ------------------------------------------------------------------ marginals
    def marginal_mean(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def marginal_std(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-coordinate standard deviation (scalar broadcastable to theta0)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def marginal_prob(self, theta0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.marginal_mean(theta0, t), self.marginal_std(theta0, t)

    def sample_perturbed(self, theta0: torch.Tensor, t: torch.Tensor,
                         noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """theta_t ~ p_{t|0}(. | theta0) = N(mu(t, theta0), sigma(t)^2 I)."""
        mean, std = self.marginal_prob(theta0, t)
        std = _as_column(std, theta0.dim()) if std.dim() < theta0.dim() else std
        if noise is None:
            noise = torch.randn_like(theta0)
        return mean + std * noise

    def score_target(self, theta_t: torch.Tensor, theta0: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        """Analytic target grad_{theta_t} log p_{t|0}(theta_t | theta0).

        Gaussian transition => -(theta_t - mu) / sigma^2.
        """
        mean, std = self.marginal_prob(theta0, t)
        std = _as_column(std, theta_t.dim()) if std.dim() < theta_t.dim() else std
        return -(theta_t - mean) / (std ** 2)

    def score_std_correction(self, t: torch.Tensor) -> torch.Tensor:
        """1 / sigma(t) -- used to convert the score into the unit-noise parameterisation."""
        return 1.0 / self.marginal_std(torch.zeros(1, 1), t)

    # ------------------------------------------------------------------ weights
    def weighting(self, t: torch.Tensor) -> torch.Tensor:
        """lambda_t of Eq. (7).

        The paper leaves lambda_t unspecified; we default to the SDE-specific weighting of
        Song et al. (2021) (their ``likelihood weighting``), which is lambda_t = g(t)^2.
        For the VE SDE g(t)^2 is proportional to sigma(t)^2 (the likelihood weighting);
        for the VP SDE g(t)^2 = beta_t.  See configs/sde_config.yaml.
        """
        return self.diffusion_sq(t)

    # ------------------------------------------------------------------ generative dynamics
    def probability_flow_drift(self, theta: torch.Tensor, t: torch.Tensor,
                               score: torch.Tensor) -> torch.Tensor:
        """RHS of the probability-flow ODE, Eq. (4):
        f(theta, t) - 1/2 g(t)^2 grad_theta log p_t(theta | x).
        """
        return self.drift(theta, t) - 0.5 * self.diffusion_sq(t) * score

    def reverse_sde_drift(self, theta: torch.Tensor, t: torch.Tensor,
                          score: torch.Tensor) -> torch.Tensor:
        """Drift of the reverse-time SDE in *forward-time* parameterisation, Eq. (3):
        f(theta, t) - g(t)^2 grad_theta log p_t(theta | x).
        Integrating this backwards in t (T -> 0) with diffusion g(t) reproduces Eq. (3).
        """
        return self.drift(theta, t) - self.diffusion_sq(t) * score


class VESDE(SDE):
    """Variance Exploding SDE, Eq. (E.3.1-1)."""

    name = "ve"

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0, T: float = T_FINAL):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.T = float(T)
        if not self.sigma_max > self.sigma_min > 0:
            raise ValueError(f"Require sigma_max > sigma_min > 0, got "
                             f"{sigma_max}, {sigma_min}")
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)

    # coefficients ---------------------------------------------------------------------
    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(theta)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_column(t, 2)
        g = self.sigma_min * (self.sigma_min / self.sigma_max) ** t \
            * math.sqrt(2.0 * self.log_ratio)
        return g

    # marginals ------------------------------------------------------------------------
    def marginal_mean(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return theta0

    def marginal_std(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t


class VPSDE(SDE):
    """Variance Preserving SDE, Eq. (E.3.1-3), beta_t = beta_min + t (beta_max - beta_min)."""

    name = "vp"

    def __init__(self, beta_min: float = 0.1, beta_max: float = 11.0, T: float = T_FINAL):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.T = float(T)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_column(t, 2)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        """B(t) = int_0^t beta_s ds."""
        t = torch.as_tensor(t)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    # coefficients ---------------------------------------------------------------------
    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta(t) * theta

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))

    # marginals ------------------------------------------------------------------------
    def marginal_mean(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _as_column(torch.as_tensor(t), theta0.dim())
        return theta0 * torch.exp(-0.5 * self.int_beta(t))

    def marginal_std(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        return torch.sqrt(1.0 - torch.exp(-self.int_beta(t)) + 1e-12)


# --------------------------------------------------------------------------- factories

def sigma_max_technique1(data: torch.Tensor, eps: float = 1e-8) -> float:
    """Song & Ermon (2020) *Technique 1* for selecting sigma_max of the VE SDE.

    The perturbation at the final level should be large enough that the noised data essentially
    fills the data support (and is close to its (near-)uniform stationary distribution), while
    still being of the order of the data scale.  Following the standard implementation of
    Technique 1 we set

        sigma_max = max_i || theta_i - mean(theta) ||_2 / sqrt(d) * sqrt(2 log(1/eps)),

    i.e. the largest per-dimension RMS displacement of a data point from the data mean, scaled by
    the usual sqrt(2 log(1/eps)) factor so that a sample at the final noise level is within the
    "bulk" of the perturbation kernel.  For standardised data this gives a value ~O(1).

    NOTE: the paper cites Song & Ermon (2020, Technique 1) without giving the formula; this is
    our documented reproduction of that heuristic (see configs/sde_config.yaml).
    """
    data = torch.as_tensor(data, dtype=torch.float64)
    if data.dim() == 1:
        data = data[:, None]
    d = data.shape[-1]
    centred = data - data.mean(dim=0, keepdim=True)
    max_rms = torch.sqrt((centred ** 2).sum(dim=-1) / d).max()
    factor = math.sqrt(2.0 * math.log(1.0 / eps))
    return float(max_rst := (max_rms * factor).item())


def get_sde(name: str, sigma_min: Optional[float] = None, sigma_max: Optional[float] = None,
            beta_min: float = 0.1, beta_max: float = 11.0,
            data: Optional[torch.Tensor] = None, dim: Optional[int] = None,
            eps: float = 1e-8) -> SDE:
    """Build one of the two SDEs of App. E.3.1.

    ``sigma_min`` defaults to 0.01 for the 2-dimensional experiments (SIR, Two Moons) and 0.05
    otherwise, as specified in App. E.3.1.  If ``sigma_max`` is None it is obtained from
    ``data`` with Song & Ermon (2020) Technique 1.
    """
    name = name.lower()
    if name in ("ve", "vesde", "variance_exploding"):
        if sigma_min is None:
            # App. E.3.1: sigma_min = 0.01 for the 2-d experiments (SIR, Two Moons), else 0.05.
            sigma_min = 0.01 if (dim is not None and dim == 2) else 0.05
        if sigma_max is None:
            if data is None:
                sigma_max = 1.0
            else:
                sigma_max = sigma_max_technique1(data, eps=eps)
            sigma_max = max(sigma_max, 10.0 * sigma_min)
        return VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
    if name in ("vp", "vpsde", "variance_preserving"):
        return VPSDE(beta_min=beta_min, beta_max=beta_max)
    raise ValueError(f"Unknown SDE '{name}' (expected 've' or 'vp').")
