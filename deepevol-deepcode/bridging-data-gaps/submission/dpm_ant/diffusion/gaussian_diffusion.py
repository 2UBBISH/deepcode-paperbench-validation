"""Gaussian diffusion process (DDPM / DDIM) used by DPMs-ANT.

This module wraps :class:`dpm_ant.diffusion.schedule.NoiseSchedule` and implements
every diffusion-process equation the paper relies on:

* forward process ``q(x_t | x_0)`` and its reparameterisation
  ``x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps``          -- Eq. (1) (§3)
* the DDPM training loss ``|| eps - eps_theta(x_t, t) ||^2``          -- Eq. (2) (§3)
* the (DDIM/DDPM) reverse process                                     -- Eq. (3) (§3)
* the classifier-guided conditional reverse process                   -- Eq. (4) (§3)
* the similarity-guided training loss                                 -- Eq. (5) (§4.1)

Everything is expressed with a 1-indexed diffusion step ``t in {1, ..., T}`` and
tensors broadcastable against images of shape ``(B, C, H, W)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .schedule import NoiseSchedule, build_schedule

__all__ = [
    "GaussianDiffusion",
    "extract",
]


def extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Gather per-sample values from ``arr`` at indices ``t`` and broadcast them.

    Args:
        arr: 1-D tensor of shape ``(T + 1,)`` (index 0 is a dummy step).
        t: index tensor of shape ``(B,)`` with values in ``1..T``.
        ndim: rank of the target tensor (e.g. 4 for ``(B, C, H, W)``).

    Returns:
        Tensor of shape ``(B, 1, ..., 1)`` with ``ndim`` dimensions.
    """
    out = arr.to(t.device)[t.long()]
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


class GaussianDiffusion:
    """DDPM / DDIM forward and reverse processes.

    Args:
        schedule: an existing :class:`NoiseSchedule`, or ``None`` to build one.
        num_timesteps: ``T`` used when ``schedule`` is ``None``.
        schedule_name: beta-schedule name (``"linear"`` default, §3/§5.2).
        eta: DDIM stochasticity in Eq. (3); ``0`` -> DDIM, ``1`` -> DDPM.
        device / dtype: defaults for newly created tensors.
        **schedule_kwargs: forwarded to :func:`build_schedule`.
    """

    def __init__(
        self,
        schedule: Optional[NoiseSchedule] = None,
        num_timesteps: int = 1000,
        schedule_name: str = "linear",
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **schedule_kwargs,
    ) -> None:
        if schedule is None:
            schedule = build_schedule(
                {
                    "num_timesteps": num_timesteps,
                    "schedule": schedule_name,
                    "eta": eta,
                    **schedule_kwargs,
                },
                dtype=dtype,
                device=device,
            )
        self.schedule: NoiseSchedule = schedule
        self.num_timesteps: int = schedule.num_timesteps
        self.eta = eta

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def betas(self) -> torch.Tensor:
        return self.schedule.betas

    @property
    def alphas(self) -> torch.Tensor:
        return self.schedule.alphas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self.schedule.alphas_cumprod

    @property
    def sigma(self) -> torch.Tensor:
        """Reverse-process noise scale ``sigma_t`` of Eq. (3)."""
        return self.schedule.sigma

    @property
    def sigma_hat(self) -> torch.Tensor:
        r"""Coefficient ``\hat{\sigma}_t = (1 - \bar\alpha_{t-1}) sqrt(\alpha_t / (1-\bar\alpha_t))``.

        Used as the similarity-guidance scale in Eq. (5) / Eq. (8) (Appendix A.2).
        """
        return self.schedule.sigma_hat

    def to(self, device) -> "GaussianDiffusion":
        self.schedule.to(device)
        return self

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _to_tensor(self, value, t: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(t.device)
        return torch.as_tensor(value, device=t.device)

    def sqrt_alpha_bar(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        r"""``sqrt(\bar\alpha_t)`` broadcast to ``ndim`` dims."""
        return self.schedule.sqrt_alphas_cumprod[t.long()].view(-1, *([1] * (ndim - 1)))

    def sqrt_one_minus_alpha_bar(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        r"""``sqrt(1 - \bar\alpha_t)`` broadcast to ``ndim`` dims."""
        return self.schedule.sqrt_one_minus_alphas_cumprod[t.long()].view(
            -1, *([1] * (ndim - 1))
        )

    # ------------------------------------------------------------------ #
    # Forward process -- Eq. (1) of §3
    # ------------------------------------------------------------------ #
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Sample ``x_t = sqrt(\bar\alpha_t) x_0 + sqrt(1 - \bar\alpha_t) eps``.

        This is Eq. (1) of §3. ``noise`` defaults to ``eps ~ N(0, I)``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        ndim = x_start.dim()
        return self.sqrt_alpha_bar(t, ndim) * x_start + self.sqrt_one_minus_alpha_bar(
            t, ndim
        ) * noise

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        r"""Predicted ``x_0 = (x_t - sqrt(1 - \bar\alpha_t) eps) / sqrt(\bar\alpha_t)``.

        This is the "predicted x_0" term appearing in Eq. (3).
        """
        ndim = x_t.dim()
        x0 = (
            x_t - self.sqrt_one_minus_alpha_bar(t, ndim) * eps
        ) / self.sqrt_alpha_bar(t, ndim).clamp(min=1e-8)
        return x0

    def predict_noise_from_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor
    ) -> torch.Tensor:
        r"""Inverse of the reparameterisation: ``eps = (x_t - sqrt(\bar\alpha_t) x_0) / sqrt(1-\bar\alpha_t)``."""
        ndim = x_t.dim()
        return (x_t - self.sqrt_alpha_bar(t, ndim) * x_0) / self.sqrt_one_minus_alpha_bar(
            t, ndim
        ).clamp(min=1e-8)

    # ------------------------------------------------------------------ #
    # Posterior / reverse-process means
    # ------------------------------------------------------------------ #
    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""``q(x_{t-1} | x_t, x_0)`` mean/variance, i.e. ``\tilde\mu_t`` of Appendix A.2."""
        ndim = x_t.dim()
        ab = self.schedule.ab(t, ndim)
        ab_prev = self.schedule.ab_prev(t, ndim)
        alpha_t = self.schedule.alpha(t, ndim)
        beta_t = self._to_tensor(self.betas, t).view(-1, *([1] * (ndim - 1)))

        posterior_mean = (
            ab_prev.sqrt() * beta_t / (1.0 - ab).clamp(min=1e-8) * x_start
            + alpha_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab).clamp(min=1e-8) * x_t
        )
        posterior_var = beta_t * (1.0 - ab_prev) / (1.0 - ab).clamp(min=1e-8)
        return posterior_mean, posterior_var

    def p_mean_variance(
        self, eps_theta: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""``\mu_\theta(x_t, t) = 1/sqrt(\alpha_t) (x_t - (1-\alpha_t)/sqrt(1-\bar\alpha_t) eps_\theta)``.

        This is the reverse-process mean used in Eq. (3)/Eq. (4) of §3.
        """
        ndim = x_t.dim()
        alpha_t = self.schedule.alpha(t, ndim)
        one_minus_ab = (1.0 - self.schedule.ab(t, ndim)).clamp(min=1e-8)
        mean = (1.0 / alpha_t.sqrt()) * (x_t - (1.0 - alpha_t) / one_minus_ab.sqrt() * eps_theta)
        var = (self.schedule.get_sigma(t, ndim)) ** 2
        return mean, var

    # ------------------------------------------------------------------ #
    # Training losses
    # ------------------------------------------------------------------ #
    def ddpm_loss(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        **model_kwargs,
    ) -> torch.Tensor:
        r"""Plain DDPM objective ``E || eps - eps_theta(x_t, t) ||^2`` -- Eq. (2) of §3.

        Args:
            model: callable ``eps_theta(x_t, t, **kwargs)``.
            x_start: clean batch ``x_0`` (target samples).
            t: diffusion steps in ``1..T``.
            noise: optional pre-sampled ``eps``; sampled from ``N(0, I)`` otherwise.
            reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        eps_pred = model(x_t, t, **model_kwargs)
        return self.mse(eps_pred, noise, reduction=reduction)

    @staticmethod
    def mse(
        pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        if reduction == "none":
            return (pred - target) ** 2
        if reduction == "sum":
            return ((pred - target) ** 2).sum()
        return F.mse_loss(pred, target, reduction="mean")

    def similarity_guided_loss(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        classifier_grad: torch.Tensor,
        gamma: float = 5.0,
        noise: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        correct_sign: bool = True,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""Similarity-guided DPMs training loss -- Eq. (5) of §4.1 / Appendix A.2:

        .. math::
            \min_\theta \mathbb{E}_{t, x_0, \epsilon}\,
            \|\epsilon_t - \epsilon_\theta(x_t, t)
              - \hat\sigma_t^2 \gamma \nabla_{x_t} \log p_\phi(y=\mathcal{T} \mid x_t)\|^2

        with :math:`\hat\sigma_t = (1-\bar\alpha_{t-1})
        \sqrt{\alpha_t / (1-\bar\alpha_t)}`.

        Args:
            model: ``eps_theta(x_t, t, **kwargs)``.
            x_start: target domain samples ``x_0``.
            t: diffusion steps in ``1..T``.
            classifier_grad: ``\nabla_{x_t} \log p_\phi(y=\mathcal{T} \mid x_t)``
                evaluated at the *noised* ``x_t``; must be detached upstream
                (the classifier is frozen, §4.1).
            gamma: similarity-guidance strength (default 5, §5.2).
            correct_sign: if ``True`` the corrected target is
                ``eps - sigma_hat^2 * gamma * grad`` exactly as written in Eq. (5).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        eps_pred = model(x_t, t, **model_kwargs)

        ndim = x_t.dim()
        coef = (self.sch_sigma_hat(t, ndim)) ** 2 * float(gamma)
        correction = coef * classifier_grad.detach()
        target = noise - correction if correct_sign else noise + correction
        return self.mse(eps_pred, target, reduction=reduction)

    def sch_sigma_hat(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        r"""``\hat\sigma_t`` broadcast to ``ndim`` dims (Appendix A.2)."""
        return self.schedule.sigma_hat[t.long()].view(-1, *([1] * (ndim - 1)))

    # ------------------------------------------------------------------ #
    # Reverse process
    # ------------------------------------------------------------------ #
    def reverse_step(
        self,
        model,
        x_t: torch.Tensor,
        t: int,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""One step of the DDIM/DDPM reverse process -- Eq. (3) of §3:

        .. math::
            x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0
                    + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t, t)
                    + \sigma_t \epsilon_t

        ``eta`` follows Eq. (3): ``0`` -> DDIM, ``1`` -> DDPM; ``None`` uses the
        value stored on the diffusion object.
        """
        if eta is None:
            eta = self.eta
        t_batch = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
        ndim = x_t.dim()
        eps = model(x_t, t_batch, **model_kwargs)

        x0 = self.predict_x0_from_noise(x_t, t_batch, eps)
        ab_prev = self.schedule.ab_prev(t_batch, ndim)
        sigma_t = self._sigma_for_eta(eta, t_batch, ndim)

        coef = (1.0 - ab_prev - sigma_t**2).clamp(min=0.0)
        mean = ab_prev.sqrt() * x0 + coef.sqrt() * eps

        if float(eta) == 0.0:
            return mean
        noise = torch.randn(
            x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator
        )
        return mean + sigma_t * noise

    def _sigma_for_eta(self, eta: float, t: torch.Tensor, ndim: int) -> torch.Tensor:
        r"""``\sigma_t = eta sqrt((1-\bar\alpha_{t-1})/(1-\bar\alpha_t)) sqrt(1 - \bar\alpha_t/\bar\alpha_{t-1})``."""
        ab = self.schedule.ab(t, ndim)
        ab_prev = self.schedule.ab_prev(t, ndim)
        ratio = (1.0 - ab_prev) / (1.0 - ab).clamp(min=1e-8)
        inner = (1.0 - ab / ab_prev.clamp(min=1e-8)).clamp(min=0.0)
        return float(eta) * ratio.sqrt() * inner.sqrt()

    # ------------------------------------------------------------------ #
    # Classifier-guided (conditional) reverse process -- Eq. (4) of §3
    # ------------------------------------------------------------------ #
    def conditional_reverse_step(
        self,
        model,
        x_t: torch.Tensor,
        t: int,
        classifier_grad_fn,
        gamma: float = 5.0,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""Classifier-guided reverse step, Eq. (4) of §3:

        .. math::
            p_{\theta,\phi}(x_{t-1}|x_t, y) \approx
            \mathcal{N}\!\left(x_{t-1}; \mu_\theta(x_t,t)
            + \sigma_t^2 \gamma \nabla_{x_t} \log p_\phi(y|x_t), \sigma_t^2 I\right)

        Args:
            classifier_grad_fn: callable ``x_t -> \nabla_{x_t} log p_phi(y=T|x_t)``
                (already detached / stop-gradient, see
                :mod:`dpm_ant.models.classifier`).
        """
        if eta is None:
            eta = self.eta
        t_batch = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
        ndim = x_t.dim()
        eps = model(x_t, t_batch, **model_kwargs)

        mu, _ = self.p_mean_variance(eps, x_t, t_batch)
        sigma_t = self._sigma_for_eta(eta, t_batch, ndim)
        grad = classifier_grad_fn(x_t)
        mean = mu + (sigma_t**2) * float(gamma) * grad

        x0 = self.predict_x0_from_noise(x_t, t_batch, eps)
        ab_prev = self.schedule.ab_prev(t_batch, ndim)
        coef = (1.0 - ab_prev - sigma_t**2).clamp(min=0.0)
        out = ab_prev.sqrt() * x0 + coef.sqrt() * eps + (sigma_t**2) * float(gamma) * grad

        if float(eta) == 0.0:
            return out
        noise = torch.randn(
            x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator
        )
        return out + sigma_t * noise

    # ------------------------------------------------------------------ #
    # Constant used in Appendix A.2
    # ------------------------------------------------------------------ #
    def c2_coefficient(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        r"""``C_2 = beta_t^2 / (2 sigma_t^2 alpha_t (1 - \bar\alpha_t))`` (Appendix A.2).

        The paper drops this constant when defining the practical training loss.
        """
        ab = self.schedule.ab(t, ndim)
        alpha_t = self.schedule.alpha(t, ndim)
        beta_t = self._to_tensor(self.betas, t).view(-1, *([1] * (ndim - 1)))
        sigma_t = self.schedule.get_sigma(t, ndim)
        # Eq. (3) with eta = 1 gives sigma_t^2 = (1-ab_prev)/(1-ab) * (1 - ab/ab_prev).
        return beta_t**2 / (
            2.0 * (sigma_t**2).clamp(min=1e-12) * alpha_t * (1.0 - ab)
        )


def build_diffusion(cfg: Optional[dict] = None, **overrides) -> GaussianDiffusion:
    """Build a :class:`GaussianDiffusion` from a (possibly nested) config dict.

    Recognised keys: ``num_timesteps``/``T``, ``schedule``/``beta_schedule``,
    ``eta``, ``beta_start``, ``beta_end``, ``cosine_s``.
    """
    cfg = dict(cfg or {})
    diffusion_cfg = dict(cfg.get("diffusion", {}) or {})
    merged = {**diffusion_cfg, **overrides}

    num_timesteps = merged.pop("num_timesteps", merged.pop("T", 1000))
    schedule_name = merged.pop("schedule", merged.pop("beta_schedule", "linear"))
    eta = float(merged.pop("eta", 0.0))
    device = merged.pop("device", None)
    dtype = merged.pop("dtype", torch.float32)
    return GaussianDiffusion(
        num_timesteps=int(num_timesteps),
        schedule_name=schedule_name,
        eta=eta,
        device=device,
        dtype=dtype,
        **merged,
    )
