"""Optimality checks for the attacks used in the paper.

For a linear model the optimal :math:`\\ell_\\infty` attack can be written in
closed form, which makes it possible to verify that APGD actually solves the
inner maximisation instead of silently returning a weak perturbation (a bug that
would inflate every robustness number in the paper).

For ``f(x) = W x`` and the margin loss
``L(x) = max_t (W_t x) − W_y x`` the maximum over the ball of radius ``eps`` is

    max_t [ (W_t − W_y)·x + eps · ||W_t − W_y||_1 ]

so we can compare the loss APGD reaches with the exact optimum.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.attacks.apgd import APGDAttack  # noqa: E402


def _linear_problem(batch: int = 8, classes: int = 5, dim: int = 48, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(classes, dim, generator=generator)
    images = torch.rand(batch, dim, generator=generator).view(batch, 3, 4, 4)
    with torch.no_grad():
        logits = weight @ images.flatten(1).t()
        labels = logits.argmax(dim=0)
    return weight, images, labels


def _margin_loss_fn(weight: torch.Tensor, labels: torch.Tensor):
    def loss_fn(images: torch.Tensor) -> torch.Tensor:
        logits = images.flatten(1) @ weight.t()
        correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        other = logits.clone()
        other.scatter_(1, labels.view(-1, 1), -float("inf"))
        return other.max(dim=1).values - correct

    return loss_fn


def _optimal_margin(weight: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, eps: float):
    """Closed-form optimum of the margin loss under the ℓ∞ *and* box constraint.

    Per coordinate the best move is ``±eps``, capped by the distance to the
    image boundary (the feasible set is the intersection of the ball and
    ``[0, 1]^D``, exactly as in the paper).
    """
    flat = images.flatten(1)
    differences = weight.unsqueeze(0) - weight[labels].unsqueeze(1)   # B x K x D
    room = torch.where(differences > 0, (1.0 - flat).unsqueeze(1), flat.unsqueeze(1)).clamp(min=0)
    step = torch.clamp(room, max=eps)
    linear = torch.einsum("bkd,bd->bk", differences, flat)
    margins = linear + (differences.abs() * step).sum(dim=-1)
    margins.scatter_(1, labels.view(-1, 1), -float("inf"))
    return margins.max(dim=1).values


def test_apgd_matches_the_closed_form_optimum():
    weight, images, labels = _linear_problem()
    eps = 0.1
    loss_fn = _margin_loss_fn(weight, labels)
    attack = APGDAttack(eps=eps, n_iter=100, alpha=2 * eps, momentum=0.75)
    x_adv, achieved = attack.perturb(loss_fn, images, maximize=True)
    optimal = _optimal_margin(weight, images, labels, eps)

    # the attack must stay feasible ...
    assert (x_adv - images).abs().max() <= eps + 1e-6
    # ... and be essentially optimal on this problem (the margins can be
    # negative for images that can no longer be misclassified, hence the
    # absolute tolerance)
    gap = (achieved - optimal).abs()
    assert (gap <= 0.05 * optimal.abs().clamp(min=1.0)).all(), gap.max()
    # most samples are solved to machine precision; APGD's step-size schedule
    # can leave a small gap on individual problems
    assert float((gap <= 1e-3).float().mean()) >= 0.75


def test_apgd_flips_every_prediction_when_a_flip_exists():
    weight, images, labels = _linear_problem()
    eps = 0.1
    optimal = _optimal_margin(weight, images, labels, eps)
    loss_fn = _margin_loss_fn(weight, labels)
    attack = APGDAttack(eps=eps, n_iter=100, alpha=2 * eps)
    x_adv, _ = attack.perturb(loss_fn, images, maximize=True)
    with torch.no_grad():
        predictions = (x_adv.flatten(1) @ weight.t()).argmax(dim=1)
    flippable = optimal > 0
    assert flippable.sum() > 0
    assert (predictions[flippable] != labels[flippable]).all()


def test_pgd_targeted_perturbation_is_optimal_for_a_linear_model():
    """The training-time PGD also finds the exact optimum on a linear objective."""
    from robust_clip.training.pgd import pgd_linf_maximize

    weight, images, labels = _linear_problem(seed=3)
    eps = 0.08
    direction = weight[0] - weight[1]

    def objective(x_adv):
        return (x_adv.flatten(1) @ direction)

    x_adv, losses = pgd_linf_maximize(
        objective, images, eps=eps, alpha=eps / 4, n_steps=20, momentum=0.9, random_init=False
    )
    flat = images.flatten(1)
    room = torch.where(direction > 0, 1.0 - flat, flat).clamp(min=0)
    optimum = (flat @ direction) + (direction.abs() * torch.clamp(room, max=eps)).sum(dim=-1)
    assert torch.allclose(losses, optimum, atol=1e-4)
    assert (x_adv - images).abs().max() <= eps + 1e-6


if __name__ == "__main__":  # pragma: no cover
    test_apgd_matches_the_closed_form_optimum()
    test_apgd_flips_every_prediction_when_a_flip_exists()
    test_pgd_targeted_perturbation_is_optimal_for_a_linear_model()
    print("attack optimality tests OK")
