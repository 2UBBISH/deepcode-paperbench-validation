"""Diffusion noise schedules and derived quantities.

Implements the quantities used throughout the paper (DPMs-ANT / ANT):

    beta_t          variance of the Gaussian noise added at step t
    alpha_t   = 1 - beta_t
    alpha_bar_t     = prod_{i=1..t} alpha_i          (cumulative signal retention)
    sigma_t         = eta * sqrt((1-ab_{t-1})/(1-ab_t)) * sqrt(1 - ab_t/ab_{t-1})
    sigma_hat_t     = (1 - alpha_bar_{t-1}) * sqrt(alpha_t / (1 - alpha_bar_t))

``sigma_hat_t`` is the coefficient appearing inside the similarity-guided loss
(Eq. (5) / Eq. (8)) and its derivation is given in Appendix A.2.

The default configuration follows Section 3 / Section 5.2: T = 1000 with a
linear beta schedule (Ho et al., 2020).  A cosine schedule (Nichol & Dhariwal,
2021) is also exposed and can be selected with ``schedule="cosine"``.

All quantities are returned as ``torch.Tensor`` of length ``T`` (1-indexed by
*step*).  ``alpha_bar[0]`` (i.e. the t = 0 entry) is set to 1.0 so that
``alpha_bar_prev(t) == alpha_bar[t - 1]`` works naturally with ``t >= 1``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# beta schedules
# ---------------------------------------------------------------------------
def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from Ho et al. (2020).  beta_1 .. beta_T, shape (T,)."""
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule of Nichol & Dhariwal (2021).

    beta_t = clip(1 - alpha_bar_t / alpha_bar_{t-1}, 0, 0.999)
    with alpha_bar_t = cos((t/T + s)/(1 + s) * pi/2)^2.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


def get_beta_schedule(name: str, num_timesteps: int, **kwargs) -> torch.Tensor:
    name = (name or "linear").lower()
    if name == "linear":
        return linear_beta_schedule(num_timesteps,
                                    kwargs.get("beta_start", 1e-4),
                                    kwargs.get("beta_end", 0.02))
    if name in ("cosine", "cos"):
        return cosine_beta_schedule(num_timesteps, kwargs.get("cosine_s", 0.008))
    if name == "scaled_linear":  # Stable-Diffusion style
        return linear_beta_schedule(num_timesteps,
                                    kwargs.get("beta_start", 0.00085) ** 0.5,
                                    kwargs.get("beta_end", 0.012) ** 0.5) ** 2
    raise ValueError(f"Unknown beta schedule: {name!r}")


# ---------------------------------------------------------------------------
# schedule container
# ---------------------------------------------------------------------------
class NoiseSchedule:
    """Holds every diffusion coefficient required by ANT.

    Parameters
    ----------
    num_timesteps:
        Number of diffusion steps ``T`` (default 1000).
    schedule:
        ``"linear"`` (default) or ``"cosine"``.
    eta:
        DDIM stochasticity for the reverse process (Eq. (3)).
        ``eta = 0`` -> deterministic DDIM sampling (Song et al., 2020),
        ``eta = 1`` -> DDPM sampling (Ho et al., 2020).
    device / dtype:
        Where the cached tensors live.
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "linear",
                 eta: float = 0.0, device=None, dtype: torch.dtype = torch.float32,
                 **schedule_kwargs):
        self.num_timesteps = int(num_timesteps)
        self.schedule_name = schedule
        self.eta = float(eta)
        self.dtype = dtype
        self.device = device

        betas = get_beta_schedule(schedule, self.num_timesteps, **schedule_kwargs)
        # prefix a dummy beta_0 = 0 so everything can be 1-indexed by step t
        betas = torch.cat([torch.zeros(1, dtype=torch.float64), betas], dim=0)  # (T+1,)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)          # ab_0 = 1, ab_1..ab_T

        # --- sigma_hat_t = (1 - ab_{t-1}) * sqrt(alpha_t / (1 - ab_t)) -------
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], dim=0)
        sigma_hat = (1.0 - ab_prev) * torch.sqrt(alphas / torch.clamp(1.0 - alphas_cumprod, min=1e-20))

        # --- sigma_t for the reverse process (Eq. (3)) ----------------------
        # sigma_t = eta * sqrt((1-ab_{t-1})/(1-ab_t)) * sqrt(1 - ab_t/ab_{t-1})
        sigma = (self.eta
                 * torch.sqrt(torch.clamp(1.0 - ab_prev, min=0.0) / torch.clamp(1.0 - alphas_cumprod, min=1e-20))
                 * torch.sqrt(torch.clamp(1.0 - alphas_cumprod / torch.clamp(ab_prev, min=1e-20), min=0.0)))

        self.betas = betas.to(dtype=dtype)
        self.alphas = alphas.to(dtype=dtype)
        self.alphas_cumprod = alphas_cumprod.to(dtype=dtype)
        self.alphas_cumprod_prev = ab_prev.to(dtype=dtype)
        self.sigma_hat = sigma_hat.to(dtype=dtype)             # (T+1,), index by t
        self.sigma = sigma.to(dtype=dtype)                     # (T+1,), index by t

        # convenience aliases matching the paper's notation
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - self.alphas_cumprod, min=0.0))
        self.sqrt_recip_alphas = torch.rsqrt(self.alphas)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    def to(self, device):
        self.device = device
        for name in ("betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
                     "sigma_hat", "sigma", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "sqrt_recip_alphas",
                     "sqrt_alphas_cumprod_prev"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ------------------------------------------------------------------
    # indexing helpers (t is a long tensor of diffusion steps in 1..T)
    # ------------------------------------------------------------------
    @staticmethod
    def _gather(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = arr.to(t.device)[t]
        while out.dim() < ndim:
            out = out.unsqueeze(-1)
        return out

    def ab(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """alpha_bar_t reshaped for broadcasting against an image of ``ndim`` dims."""
        return self._gather(self.alphas_cumprod, t, ndim)

    def ab_prev(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return self._gather(self.alphas_cumprod_prev, t, ndim)

    def alpha(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return self._gather(self.alphas, t, ndim)

    def get_sigma_hat(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return self._gather(self.sigma_hat, t, ndim)

    def get_sigma(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return self._gather(self.sigma, t, ndim)

    # ------------------------------------------------------------------
    # forward process q(x_t | x_0) -- Eq. (1)
    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (self.ab(t, x0.dim()) * x0
                + self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.dim()) * noise)

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                              eps: torch.Tensor) -> torch.Tensor:
        """Reparameterized x_0 estimate used by the reverse process (Eq. (3))."""
        return ((x_t - self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.dim()) * eps)
                / self._gather(self.sqrt_alphas_cumprod, t, x_t.dim()))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (f"NoiseSchedule(T={self.num_timesteps}, schedule={self.schedule_name!r}, "
                f"eta={self.eta})")


DEFAULT_SCHEDULE = NoiseSchedule(num_timesteps=1000, schedule="linear", eta=0.0)


def build_schedule(cfg: Optional[dict] = None, **overrides) -> NoiseSchedule:
    """Build a :class:`NoiseSchedule` from a (nested) config dict.

    Accepted keys: ``num_timesteps``/``T``, ``schedule``/``beta_schedule``,
    ``eta``, ``beta_start``, ``beta_end``, ``cosine_s``.
    """
    cfg = dict(cfg or {})
    cfg.update(overrides)
    T = cfg.get("num_timesteps", cfg.get("T", 1000))
    name = cfg.get("schedule", cfg.get("beta_schedule", "linear"))
    eta = cfg.get("eta", 0.0)
    kwargs = {k: cfg[k] for k in ("beta_start", "beta_end", "cosine_s") if k in cfg}
    return NoiseSchedule(num_timesteps=T, schedule=name, eta=eta, **kwargs)
