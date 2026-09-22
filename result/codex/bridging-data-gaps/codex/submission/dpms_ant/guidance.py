"""Similarity-guided training (Section 4.1 of the paper).

The paper replaces the "transfer direction" estimation of GAN-based few-shot
methods with a *similarity measure* obtained from a binary domain classifier
``p_phi(y | x_t)`` that is evaluated on the **noised** image ``x_t``:

    grad_{x_t} log p_phi(y=S | x_t)   and   grad_{x_t} log p_phi(y=T | x_t)

The KL divergence between the source and the target reverse processes reduces
to the squared difference of those two gradient fields (Eq. 4, derivation in
Appendix A.1), and the resulting training loss of Eq. 5 is

    min_theta E_{t,x0,eps} || eps_t - eps_theta(x_t, t)
                              - sigma_hat_t^2 * gamma * grad_{x_t} log p_phi(y=T | x_t) ||^2

with ``sigma_hat_t = (1 - alpha_bar_{t-1}) * sqrt(alpha_t / (1 - alpha_bar_t))``.

Note the sign: the classifier term is *subtracted* from the noise target.  This
is consistent with classifier guidance at sampling time, where the mean is
shifted by ``+ sigma_t^2 gamma grad log p(y=T|x_t)``: at convergence
``eps_theta = eps - sigma_hat^2 gamma grad log p_T``, and the reverse step uses
``-eps_theta``, hence moves the sample towards the target domain.

The classifier is frozen here; its gradient is an indicator (a "target
correction"), which is exactly how the paper uses it ("we employ a fixed
pre-trained binary classifier").
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .schedules import DiffusionSchedule
from .utils import mean_flat


def classifier_logits(
    classifier: torch.nn.Module, x_t: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Run a (guided-diffusion style) classifier and return its logits."""
    try:
        return classifier(x_t, timesteps)
    except TypeError:
        # plain image classifier (e.g. the toy MLP): ignore the timestep
        return classifier(x_t)


def classifier_guidance(
    classifier: torch.nn.Module,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    target_class: int = 1,
    gamma: float = 5.0,
    scale: float = 1.0,
    create_graph: bool = False,
    normalize: bool = False,
    target_rms: Optional[float] = None,
) -> torch.Tensor:
    """``sigma_hat_t^2 * gamma * grad_{x_t} log p_phi(y=target_class | x_t)``.

    Parameters
    ----------
    classifier:
        Frozen domain classifier ``p_phi`` (source vs. target, 2 logits).
    x_t:
        Noised images; the gradient is taken w.r.t. this tensor, which does not
        need ``requires_grad`` (it is enabled internally).
    schedule:
        Provides ``sigma_hat_t``.
    target_class:
        Index of the target domain (1 by convention: 0 = source, 1 = target).
    gamma:
        Similarity-guidance scale (paper: ``gamma = 5``).
    scale:
        Extra constant (classifier guidance scale, if guidance is used for
        generation as well).
    normalize:
        If True, the gradient is rescaled to unit RMS per sample.  The paper
        uses the raw gradient; the flag is provided for ablations.
    target_rms:
        Alternative to ``gamma``: rescale ``sigma_hat_t^2 * grad`` so that its
        RMS over the batch equals ``target_rms`` (in units of the noise std).
        Used by the 2-D toy study of Section 5.1, whose input dimensionality --
        and therefore whose per-element gradient magnitude -- is completely
        different from a 256x256 image, so that ``gamma=5`` (the image value)
        would make the correction dominate the noise.  ``None`` (the default)
        reproduces Eq. (5) literally.
    """
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        logits = classifier_logits(classifier, x_in, timesteps)
        log_prob = F.log_softmax(logits.float(), dim=-1)[:, target_class].sum()
        (grad,) = torch.autograd.grad(log_prob, x_in, create_graph=create_graph)
    grad = grad.detach() if not create_graph else grad
    if normalize:
        dims = list(range(1, grad.ndim))
        rms = grad.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
        grad = grad / rms
    sigma_hat = schedule.sigma_hat_t(timesteps, grad)
    correction = sigma_hat ** 2 * grad
    if target_rms is not None:
        rms = correction.pow(2).mean().sqrt().clamp_min(1e-12)
        return scale * correction * (target_rms / rms)
    return scale * gamma * correction


def similarity_guided_loss(
    eps_pred: torch.Tensor,
    eps_target: torch.Tensor,
    guidance: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq. (5)/(8): ``|| eps_t - eps_theta(x_t, t) - sigma_hat^2 gamma grad log p_T ||^2``.

    ``eps_target`` is ``eps_t`` (the sampled Gaussian noise) and ``guidance``
    is the correction term returned by :func:`classifier_guidance`; if the
    correction is ``None`` the loss degenerates to the vanilla DDPM loss
    (Eq. 1).
    """
    residual = eps_target - eps_pred
    if guidance is not None:
        residual = residual - guidance
    per_sample = mean_flat(residual ** 2)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"unknown reduction: {reduction}")


def vanilla_ddpm_loss(
    eps_pred: torch.Tensor, eps_target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Eq. (1): the standard DDPM epsilon-prediction loss."""
    return similarity_guided_loss(eps_pred, eps_target, guidance=None, reduction=reduction)


def split_eps_and_logvar(model_output: torch.Tensor, in_channels: int):
    """Split a ``learn_sigma`` model output into (eps, log-variance)."""
    if model_output.shape[1] == 2 * in_channels:
        eps, logvar = torch.split(model_output, in_channels, dim=1)
        return eps, logvar
    return model_output, None
