"""Reverse (denoising) process used at generation time.

Section 3 of the paper writes the reverse step as

    x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
              + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_theta(x_t, t)
              + sigma_t * eps_t

with ``eta = 0`` (DDIM, Song et al., 2020) and ``eta = 1`` (DDPM, Ho et al.,
2020).  Both are implemented here; ``eta=1`` is the default used by the paper.

Classifier guidance (Dhariwal & Nichol, 2021)
``p(x_{t-1} | x_t, y) ~ N(mu_theta + sigma_t^2 gamma grad log p_phi(y|x_t), sigma_t^2 I)``
can optionally be applied at every step (``guidance_fn``); the paper's transfer
method *does not* use it at sampling time -- it teaches the model the shift by
training instead -- but the option is kept for completeness/ablation.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch

from .schedules import DiffusionSchedule

ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GuidanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _split_model_output(model_output: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Drop the learned-variance half of the output if present (learn_sigma)."""
    if model_output.shape[1] == 2 * in_channels:
        return torch.split(model_output, in_channels, dim=1)[0]
    return model_output


@torch.no_grad()
def ddpm_step(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    eps_theta: torch.Tensor,
    eta: float = 1.0,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """One reverse step, exactly as written in Section 3 of the paper."""
    alpha_bar_prev = schedule._extract(schedule.alphas_cumprod_prev, timesteps, x_t)
    alpha_bar = schedule._extract(schedule.alphas_cumprod, timesteps, x_t)
    sigma_t = eta * torch.sqrt(
        torch.clamp((1 - alpha_bar_prev) / (1 - alpha_bar), min=1e-12)
    ) * torch.sqrt(torch.clamp(1 - alpha_bar / torch.clamp(alpha_bar_prev, min=1e-12), min=1e-12))

    x0_pred = schedule.predict_x0_from_eps(x_t, timesteps, eps_theta)
    if clip_denoised:
        x0_pred = x0_pred.clamp(-1, 1)
    direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * eps_theta
    mean = torch.sqrt(alpha_bar_prev) * x0_pred + direction
    if eta == 0.0:
        return mean
    noise = torch.randn_like(x_t)
    nonzero = (timesteps > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
    return mean + sigma_t * noise * nonzero


@torch.no_grad()
def ddim_step(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    timesteps_prev: torch.Tensor,
    eps_theta: torch.Tensor,
    eta: float = 0.0,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """DDIM (Song et al., 2020) update, i.e. the paper's step with ``eta=0``."""
    alpha_bar = schedule._extract(schedule.alphas_cumprod, timesteps, x_t)
    alpha_bar_prev = schedule._extract(schedule.alphas_cumprod_prev, timesteps_prev, x_t)
    x0_pred = schedule.predict_x0_from_eps(x_t, timesteps, eps_theta)
    if clip_denoised:
        x0_pred = x0_pred.clamp(-1, 1)
    sigma_t = eta * torch.sqrt(
        torch.clamp((1 - alpha_bar_prev) / (1 - alpha_bar), min=1e-12)
    ) * torch.sqrt(torch.clamp(1 - alpha_bar / torch.clamp(alpha_bar_prev, min=1e-12), min=1e-12))
    direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * eps_theta
    x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + direction
    if eta > 0:
        noise = torch.randn_like(x_t)
        nonzero = (timesteps_prev > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        x_prev = x_prev + sigma_t * noise * nonzero
    return x_prev


@torch.no_grad()
def sample_loop(
    schedule: DiffusionSchedule,
    shape: Sequence[int],
    model_fn: ModelFn,
    steps: Optional[torch.Tensor] = None,
    eta: float = 1.0,
    clip_denoised: bool = True,
    guidance_fn: Optional[GuidanceFn] = None,
    guidance_scale: float = 1.0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    progress: bool = False,
) -> torch.Tensor:
    """Full ancestral sampling loop.

    :param model_fn: callable ``(x_t, t) -> eps_theta``; ``t`` is a batch vector.
    :param steps: decreasing list of timesteps (default: all T..1).
    """
    device = torch.device(device)
    if steps is None:
        steps = torch.arange(schedule.num_timesteps - 1, -1, -1, device=device)
    else:
        steps = steps.to(device)

    x_t = torch.randn(*shape, device=device, dtype=dtype)
    iterator = range(len(steps))
    if progress:
        from .utils import progbar

        iterator = progbar(iterator, desc="sampling")

    for index in iterator:
        t = steps[index].reshape(1).expand(shape[0])
        eps_theta = model_fn(x_t, t)
        if guidance_fn is not None:
            eps_theta = eps_theta + guidance_scale * guidance_fn(x_t, t)
        if index == len(steps) - 1:
            # final step: jump to x_0 (no mean/noise mixture)
            x_t = schedule.predict_x0_from_eps(x_t, t, eps_theta)
            break
        if eta >= 1.0:
            x_t = ddpm_step(schedule, x_t, t, eps_theta, eta=1.0, clip_denoised=clip_denoised)
        else:
            t_prev = steps[index + 1].reshape(1).expand(shape[0])
            x_t = ddim_step(
                schedule, x_t, t, t_prev, eps_theta, eta=eta, clip_denoised=clip_denoised
            )
    return x_t


def make_guided_model_fn(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    schedule: DiffusionSchedule,
    in_channels: int = 3,
    target_class: int = 1,
    gamma: float = 5.0,
    classifier_scale: float = 1.0,
) -> GuidanceFn:
    """Classifier guidance term ``sigma_t^2 * gamma * grad log p(y=T|x_t)``."""

    def guidance_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        from .guidance import classifier_guidance

        return classifier_guidance(
            classifier=classifier,
            schedule=schedule,
            x_t=x_t,
            timesteps=timesteps,
            target_class=target_class,
            gamma=gamma,
            scale=classifier_scale,
        )

    return guidance_fn
