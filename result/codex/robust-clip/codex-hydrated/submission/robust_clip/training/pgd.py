"""Projected gradient descent for the inner maximisation of adversarial training.

The paper (Sec. 4, App. B.1) uses **10 steps of PGD** with a step size of
``1/255`` at an :math:`\\ell_\\infty` radius of ``2/255`` resp. ``4/255`` to
approximately solve the inner problems of Eq. (2) (TeCoA) and Eq. (3) (FARE).

Per the addendum the PGD implementation used in the paper features

* **gradient normalisation with elementwise sign** for the
  :math:`\\ell_\\infty` threat model,
* a **momentum factor of 0.9**,
* **initialisation with a uniform random perturbation**,
* and the :math:`\\ell_\\infty` ball is computed **around the non-normalised
  inputs**, i.e. in pixel space (the CLIP normalisation is applied inside the
  model, after the perturbation has been added).

All four properties are implemented here and are used both for training
(:mod:`robust_clip.training.train`) and for the evaluation attacks
(:mod:`robust_clip.attacks.apgd`) whenever the simpler PGD variant is required.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from ..utils.quant import quantize_pixels


def normalize_gradient(grad: torch.Tensor, mode: str = "elementwise_sign", eps: float = 1e-12):
    """Normalise a gradient for the :math:`\\ell_\\infty` threat model.

    ``elementwise_sign`` divides every entry by its own magnitude, which is the
    "gradient normalization with elementwise sign" of the paper's addendum and
    reduces to :func:`torch.sign` (the ``eps`` keeps zero entries at zero).
    """
    if mode in (None, "none"):
        return grad
    if mode in ("elementwise_sign", "sign"):
        return grad / (grad.abs() + eps)
    if mode == "mean_abs":
        return grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + eps)
    if mode == "l1":
        return grad / (grad.abs().sum(dim=(1, 2, 3), keepdim=True) + eps)
    if mode == "l2":
        return grad / (grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1) + eps)
    raise ValueError(f"unknown gradient normalisation '{mode}'")


def project_linf_ball(
    delta: torch.Tensor,
    x: torch.Tensor,
    eps: float,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Project ``x + delta`` back onto the feasible set.

    The feasible set is the intersection of the :math:`\\ell_\\infty` ball of
    radius ``eps`` around ``x`` (measured in *non-normalised* input space) and
    the valid image domain ``[clamp_min, clamp_max]``.
    """
    delta = delta.clamp(min=-eps, max=eps)
    x_adv = (x + delta).clamp(min=clamp_min, max=clamp_max)
    return x_adv - x


@torch.no_grad()
def uniform_perturbation_like(
    x: torch.Tensor, eps: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    return torch.empty_like(x).uniform_(-eps, eps, generator=generator)


def pgd_linf_maximize(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float,
    alpha: float,
    n_steps: int = 10,
    random_init: bool = True,
    momentum: float = 0.9,
    grad_normalization: str = "elementwise_sign",
    quantize_bits: Optional[int] = None,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    keep_best: bool = True,
    delta_init: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Maximise ``loss_fn`` inside an :math:`\\ell_\\infty` ball around ``x``.

    Parameters
    ----------
    loss_fn:
        callable mapping a batch of pixel-space images to a per-sample tensor of
        losses (``shape (B,)``) that are *maximised*.
    x:
        clean batch in pixel space (``[0, 1]``).
    eps, alpha, n_steps:
        radius, step size and number of iterations.
    quantize_bits:
        if set, the adversarial image is rounded onto a ``quantize_bits``-wide
        integer grid after every step (16 for half-precision attacks, 32 for
        single-precision attacks, see the addendum).

    Returns
    -------
    ``(x_adv, losses)`` where ``losses`` are the per-sample losses of the
    returned adversarial batch.
    """
    if delta_init is not None:
        delta = delta_init.clone()
    elif random_init:
        delta = uniform_perturbation_like(x, eps, generator=generator)
    else:
        delta = torch.zeros_like(x)
    delta = project_linf_ball(delta, x, eps, clamp_min, clamp_max)

    velocity = torch.zeros_like(x)
    x_best: Optional[torch.Tensor] = None
    loss_best: Optional[torch.Tensor] = None

    for _ in range(n_steps):
        x_adv = (x + delta).detach().requires_grad_(True)
        losses = loss_fn(x_adv)
        grad, = torch.autograd.grad(losses.sum(), x_adv, only_inputs=True)
        grad = normalize_gradient(grad.detach(), grad_normalization)

        velocity = momentum * velocity + grad
        delta = delta.detach() + alpha * velocity.sign()
        delta = project_linf_ball(delta, x, eps, clamp_min, clamp_max)
        if quantize_bits is not None:
            # Quantise the perturbation, then re-project so that the integer
            # perturbation never leaves the ball / image domain.
            delta = quantize_pixels(delta, quantize_bits)
            delta = project_linf_ball(delta, x, eps, clamp_min, clamp_max)

        if keep_best:
            with torch.no_grad():
                x_cand = x + delta
                cand_losses = loss_fn(x_cand).detach()
                if loss_best is None:
                    x_best, loss_best = x_cand.clone(), cand_losses.clone()
                else:
                    better = cand_losses > loss_best
                    if better.any():
                        x_best[better] = x_cand[better]
                        loss_best[better] = cand_losses[better]

    if keep_best and x_best is not None:
        return x_best.detach(), loss_best.detach()

    with torch.no_grad():
        x_adv = x + delta
        return x_adv.detach(), loss_fn(x_adv).detach()
