"""Adversarial noise selection (Section 4.2 of the paper).

The second challenge the paper identifies is that *fully random* Gaussian noise
has an unbalanced effect across images, which makes the transfer pace diverge
and forces many iterations.  Instead of sampling ``eps ~ N(0, I)``, the paper
solves the inner maximisation of a min-max problem (Eq. 6)

    min_theta max_eps E_{t,x0} || eps - eps_theta(x_t, t)
                                    - sigma_hat_t^2 gamma grad log p_phi(y=T|x_t) ||^2

approximately, with ``J`` steps of gradient ascent on the *un-guided*
residual (the "practically, the similarity-guided term is disregarded, as this
term is hard to compute differential and is almost unchanged in the process"):

    eps^{j+1} = Norm( eps^j
                      + omega * grad_{eps^j} || eps^j - eps_theta(
                            sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps^j, t) ||^2 )

``Norm(.)`` re-normalises the perturbation so that it approximately keeps mean
``0`` and standard deviation ``1`` (Eq. 7), ``omega`` is the ascent step size
(paper: ``0.02``) and ``J = 10``.

The result is the "worse-case" noise actually optimised in Eq. (8): minimising
it also minimises every Gaussian noise that is "better" than it, which corrects
the gradient computed from the few target images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .schedules import DiffusionSchedule


@dataclass
class AdversarialNoiseConfig:
    """Hyper-parameters of Eq. (7) (``J`` and ``omega`` in the paper)."""

    num_steps: int = 10          # J
    step_size: float = 0.02      # omega
    normalize: bool = True       # Norm(.)
    reduction: str = "sum"       # "sum" matches the paper's un-normalised norm
    detach_model: bool = True    # the model is a fixed target during the ascent
    # "per_sample" normalises every perturbation on its own (the natural choice
    # for images, where each sample has 3*256*256 elements); "global" uses the
    # statistics of the whole batch, which is what the 2-D toy experiment of
    # Section 5.1 needs (per-sample normalisation in 2-D would collapse every
    # perturbation onto (+-1, -+1) and no elliptical cloud could appear).
    normalization: str = "per_sample"


def normalize_noise(
    noise: torch.Tensor, eps: float = 1e-8, mode: str = "per_sample"
) -> torch.Tensor:
    """``Norm(.)``: per-sample zero mean and unit standard deviation.

    "a normalization function that approximately ensures the mean and standard
    deviation of ``eps^{j+1}`` is 0 and I, respectively" (Section 4.2).

    ``mode="global"`` uses a single mean/variance over the whole batch.
    """
    if mode == "global":
        return (noise - noise.mean()) / noise.var(unbiased=False).sqrt().clamp_min(eps)
    if mode != "per_sample":
        raise ValueError(f"unknown normalization mode: {mode}")
    dims = list(range(1, noise.ndim))
    mean = noise.mean(dim=dims, keepdim=True)
    std = noise.var(dim=dims, keepdim=True, unbiased=False).sqrt().clamp_min(eps)
    return (noise - mean) / std


def adversarial_noise_objective(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """``|| eps - eps_theta(sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps, t) ||^2``."""
    x_t = schedule.q_sample(x_start, timesteps, noise)
    model_output = model(x_t, timesteps)
    eps_pred = model_output
    if model_output.shape[1] == 2 * x_start.shape[1]:
        eps_pred = torch.split(model_output, x_start.shape[1], dim=1)[0]
    residual = noise - eps_pred
    per_sample = residual.reshape(residual.shape[0], -1).pow(2).mean(dim=1)
    return per_sample.sum() if reduction == "sum" else per_sample.mean()


def select_adversarial_noise(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    config: Optional[AdversarialNoiseConfig] = None,
    initial_noise: Optional[torch.Tensor] = None,
    return_history: bool = False,
):
    """Eq. (7): ``J``-step gradient ascent producing the "worse-case" noise.

    :param model: the *current* (adaptor-augmented) denoiser ``eps_theta``.
    :param x_start: target-domain images ``x_0`` (shape ``[N, C, H, W]``).
    :param timesteps: batch of timesteps ``t``.
    :returns: ``eps^J`` (the ``eps*`` of Eq. 8) and, optionally, every
        intermediate iterate (used to draw Figure 2a).
    """
    config = config or AdversarialNoiseConfig()
    if initial_noise is None:
        noise = torch.randn_like(x_start)
    else:
        noise = initial_noise.clone()

    history: List[torch.Tensor] = [noise.detach().clone()]
    for _ in range(config.num_steps):
        noise = noise.detach().requires_grad_(True)
        objective = adversarial_noise_objective(
            model, schedule, x_start, timesteps, noise, reduction=config.reduction
        )
        (grad,) = torch.autograd.grad(objective, noise)
        with torch.no_grad():
            noise = noise + config.step_size * grad
            if config.normalize:
                noise = normalize_noise(noise, mode=config.normalization)
        history.append(noise.detach().clone())

    noise = noise.detach()
    if return_history:
        return noise, history
    return noise


def noise_statistics(noise: torch.Tensor) -> Dict[str, float]:
    """Mean/std of a noise tensor and the anisotropy of its per-sample covariance.

    Useful to reproduce the claim of Figure 2a: the adversarial perturbations
    form an ellipse whose principal axis follows the model-parameter gradient
    direction (the isotropic Gaussian becomes anisotropic).
    """
    flat = noise.detach().reshape(noise.shape[0], -1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "per_sample_std_mean": float(flat.std(dim=1).mean()),
    }
