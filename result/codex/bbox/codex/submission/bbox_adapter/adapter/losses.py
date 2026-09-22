"""Ranking based Noise Contrastive Estimation (Section 3.2, Eqs. 2-3).

Eq. (2) is the listwise NCE objective obtained by minimising
``KL(q || p_theta)`` over the posterior that a sample is the positive one:

    max_theta  E_{p_data(x)} [ g_theta(x) - log sum_k exp(g_theta(x_k)) ]

Eq. (3) instantiates the gradient with positives sampled from the target domain
and negatives sampled from the adapted model ``p_theta`` and adds the
``alpha * g_theta^2`` term that plays the role of spectral normalisation, as
clarified in the addendum:

    grad = grad { -E_{y+~p_data}[g(x, y+)] + E_{y-~p_theta}[g(x, y-)]
                  + alpha E[g(x, y+)^2] + alpha E[g(x, y-)^2] }

The functions below implement exactly these two objectives on minibatches of
energies.  :func:`ranking_nce_loss` is the default used in the paper's
implementation (Eq. 3) and :func:`ranking_nce_softmax_loss` is the listwise
counterpart (Eq. 2).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

#: Coefficient alpha of the l2 regularisation of the energies.  The paper
#: states that spectral normalisation is realised through this term but does not
#: report the value; the default lives in ``AdapterConfig.alpha``.
DEFAULT_ALPHA = 0.01


def l2_energy_regularizer(*energies: torch.Tensor, alpha: float = DEFAULT_ALPHA) -> torch.Tensor:
    """``alpha * E[g^2]`` for every supplied energy tensor."""

    total = None
    for energy in energies:
        term = alpha * (energy ** 2).mean()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("At least one energy tensor is required")
    return total


def ranking_nce_loss(
    positive_energies: torch.Tensor,
    negative_energies: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pairwise ranking NCE loss, Eq. (3).

    ``positive_energies`` are ``g_theta(x, y_+)`` for ``y_+ ~ p_data`` and
    ``negative_energies`` are ``g_theta(x, y_-)`` for ``y_- ~ p_theta``.
    """

    if positive_energies.shape != negative_energies.shape:
        raise ValueError("positive and negative energies must have the same shape")
    per_example = (
        -positive_energies
        + negative_energies
        + alpha * positive_energies ** 2
        + alpha * negative_energies ** 2
    )
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.mean()


def ranking_nce_softmax_loss(energies: torch.Tensor, alpha: float = DEFAULT_ALPHA) -> torch.Tensor:
    """Listwise form of Eq. (2).

    ``energies`` has shape ``(batch, K)`` where index ``0`` is the positive
    sample drawn from the target domain and the remaining ``K - 1`` entries are
    negative samples drawn from ``p_theta``.
    """

    if energies.dim() != 2:
        raise ValueError("energies must have shape (batch, K)")
    positive = energies[:, 0]
    log_partition = torch.logsumexp(energies, dim=1)
    ranking = -(positive - log_partition).mean()
    return ranking + l2_energy_regularizer(energies, alpha=alpha)


def nce_accuracy(positive_energies: torch.Tensor, negative_energies: torch.Tensor) -> float:
    """Fraction of pairs for which the positive sample scores higher."""

    if positive_energies.numel() == 0:
        return float("nan")
    return (positive_energies > negative_energies).float().mean().item()


def energy_gap(positive_energies: torch.Tensor, negative_energies: torch.Tensor) -> float:
    return (positive_energies - negative_energies).mean().item()
