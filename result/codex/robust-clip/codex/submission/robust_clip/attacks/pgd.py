"""PGD attack used for the adversarial fine-tuning and the targeted attacks.

The implementation follows the description of the paper / addendum:

* the :math:`\\ell_\\infty` ball is computed around the **non-normalized** input,
  i.e. ``x`` lives in ``[0, 1]`` and normalization is applied inside the model
  wrapper (``forward_fn``),
* the initialization is a **uniform random perturbation** inside the ball,
* the gradient is **normalized** and we take the **elementwise sign** (normalized
  steepest descent for the :math:`\\ell_\\infty` threat model),
* a **momentum factor of 0.9** is used (``momentum=0`` reproduces the plain PGD
  of the jailbreaking attack of Qi et al. (2023), for which the addendum states
  "The PGD in the attacks doesn't use momentum"),
* every update is snapped to the grid of the attack precision (16-bit ints for
  half-precision attacks, 32-bit ints for single-precision attacks, see
  :mod:`robust_clip.utils.precision`).

The attack is written against two callables so that it can be reused for CLIP
embedding objectives (fine-tuning, App. C.4) and for generative LVLM objectives
(token-level losses, Sec. 4.2 / Sec. 4.4):

``forward_fn(x_adv)`` returns whatever ``loss_fn`` consumes (embeddings, logits,
...) and ``loss_fn(outputs, x_adv)`` returns a **per-sample** loss tensor.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

from ..utils.precision import PerturbationGrid, eps_to_int

__all__ = ["pgd_attack", "PGDAttack", "normalize_gradient"]


def normalize_gradient(gradient: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the gradient to unit :math:`\\ell_2` norm per sample.

    Together with the elementwise ``sign`` this yields the normalized steepest
    descent step for the :math:`\\ell_\\infty` ball
    ("gradient normalization with elementwise sign for l_infinity", addendum).
    """
    flat = gradient.flatten(1)
    norm = flat.norm(p=2, dim=1).clamp_min(eps)
    return (flat / norm[:, None]).view_as(gradient)


def pgd_attack(
    images: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], object],
    loss_fn: Callable[[object, torch.Tensor], torch.Tensor],
    eps: Union[str, float] = "2/255",
    alpha: Union[str, float] = "1/255",
    steps: int = 10,
    dtype: torch.dtype = torch.float32,
    momentum: float = 0.9,
    random_start: bool = True,
    maximize: bool = True,
    return_best: bool = True,
    x_min: float = 0.0,
    x_max: float = 1.0,
    generator: Optional[torch.Generator] = None,
    x_init: Optional[torch.Tensor] = None,
    track_best_every: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run PGD and return ``(x_adv, per_sample_loss)``.

    Parameters
    ----------
    images:
        clean images, float32 in ``[0, 1]`` (*not* normalized).
    forward_fn:
        differentiable function of the perturbed image.
    loss_fn:
        ``loss_fn(outputs, x_adv)`` -> per-sample loss ``[B]``.  For
        maximization (attacks) the loss is the quantity the attacker wants to
        increase; for the *targeted* attacks ``maximize=False`` minimizes the
        loss of the target string.
    eps, alpha:
        radius and step size, either as float or as ``'k/255'`` strings.
    dtype:
        ``torch.float16`` for the half-precision stage and ``torch.float32`` for
        the single-precision stage of the ensemble attack.
    x_init:
        optional warm start (used when the single-precision stage is initialized
        with the perturbation found by the half-precision stage).
    track_best_every:
        evaluate the loss of the current iterate every ``k`` steps to keep track
        of the best (worst-case for ``maximize=True``) point.  ``1`` evaluates
        every step, larger values reduce the number of extra forward passes for
        very long attacks (e.g. the 5000-step jailbreaking attack).
    """
    eps_int = eps_to_int(eps)
    alpha_int = eps_to_int(alpha)
    grid = PerturbationGrid(eps_int, dtype=dtype, alpha_int=alpha_int, x_min=x_min, x_max=x_max)

    x = images.detach().to(torch.float32).clamp(x_min, x_max)
    if x_init is not None:
        x_adv = grid.project(x_init.detach().to(torch.float32), x)
    elif random_start:
        x_adv = grid.random_start(x, generator=generator)
    else:
        x_adv = grid.snap(x.clone())

    velocity = torch.zeros_like(x_adv)
    best_x = x_adv.detach().clone()
    best_loss = None

    total_steps = max(0, steps)
    track_best_every = max(1, int(track_best_every))
    for step in range(total_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            outputs = forward_fn(x_adv)
            loss = loss_fn(outputs, x_adv)
            if loss.dim() == 0:
                loss = loss.expand(x.shape[0])
            gradient = torch.autograd.grad(loss.sum(), x_adv, retain_graph=False, create_graph=False)[0]

        gradient = normalize_gradient(gradient)
        if not maximize:
            gradient = -gradient

        velocity = momentum * velocity + gradient
        x_adv = x_adv.detach() + grid.alpha * velocity.sign()
        x_adv = grid.project(x_adv, x)

        is_last = step == total_steps - 1
        if (step + 1) % track_best_every == 0 or is_last or best_loss is None:
            with torch.no_grad():
                current = loss_fn(forward_fn(x_adv), x_adv).detach()
            if best_loss is None:
                best_loss = current
                best_x = x_adv.detach().clone()
            else:
                better = (current > best_loss) if maximize else (current < best_loss)
                best_loss = torch.where(better, current, best_loss)
                best_x[better] = x_adv.detach()[better]

    if return_best:
        with torch.no_grad():
            final_loss = loss_fn(forward_fn(best_x), best_x).detach()
        return best_x, final_loss
    with torch.no_grad():
        final_loss = loss_fn(forward_fn(x_adv), x_adv).detach()
    return x_adv.detach(), final_loss


class PGDAttack:
    """Object oriented wrapper around :func:`pgd_attack` storing the hyper-parameters."""

    def __init__(
        self,
        eps: Union[str, float] = "2/255",
        alpha: Union[str, float] = "1/255",
        steps: int = 10,
        dtype: torch.dtype = torch.float32,
        momentum: float = 0.9,
        random_start: bool = True,
        maximize: bool = True,
        return_best: bool = True,
    ):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.dtype = dtype
        self.momentum = momentum
        self.random_start = random_start
        self.maximize = maximize
        self.return_best = return_best

    def __call__(self, images, forward_fn, loss_fn, x_init=None, generator=None):
        return pgd_attack(
            images,
            forward_fn,
            loss_fn,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            dtype=self.dtype,
            momentum=self.momentum,
            random_start=self.random_start,
            maximize=self.maximize,
            return_best=self.return_best,
            x_init=x_init,
            generator=generator,
        )
