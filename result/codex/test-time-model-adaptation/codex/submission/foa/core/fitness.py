"""Unsupervised fitness function of FOA (Eqn. (5)).

.. math::

    \\mathcal{L}(f_\\Theta(\\mathbf{p}; \\mathcal{X}_t)) =
        \\sum_{\\mathbf{x} \\in \\mathcal{X}_t} \\sum_{c \\in \\mathcal{C}}
            -\\hat{y}_c \\log \\hat{y}_c
        + \\lambda \\sum_{i=1}^{N} \\left\\| \\boldsymbol{\\mu}_i(\\mathcal{X}_t)
            - \\boldsymbol{\\mu}_i^S \\right\\|_2
            + \\left\\| \\boldsymbol{\\sigma}_i(\\mathcal{X}_t)
            - \\boldsymbol{\\sigma}_i^S \\right\\|_2

The first term is the prediction entropy (summed over the batch, exactly as written in
the paper) and the second term is the activation distribution discrepancy between the
OOD testing CLS activations and the source in-distribution CLS activations.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .statistics import FeatureStatistics


def prediction_entropy(logits: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """``sum_{x in X_t} sum_c -y_c log y_c`` (Eqn. (5), left term)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=-1)
    if reduction == "sum":
        return ent.sum()
    if reduction == "mean":
        return ent.mean()
    if reduction == "none":
        return ent
    raise ValueError(reduction)


def activation_discrepancy(
    layer_features: Sequence[torch.Tensor],
    source: FeatureStatistics,
    layers: Optional[Sequence[int]] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """``sum_i ||mu_i(X_t) - mu_i^S||_2 + ||sigma_i(X_t) - sigma_i^S||_2``.

    Args:
        layer_features: ``[e_0^0, ..., e_N^0]`` of the current batch, each ``[B, d]``.
        source: source in-distribution statistics (same layer indexing).
        layers: which layers enter the sum; the paper sums ``i = 1 .. N``.
        reduce: ``"sum"`` (Eqn. (5)) or ``"mean"``.
    """
    layers = list(range(1, len(layer_features))) if layers is None else list(layers)
    total = None
    for i in layers:
        feats = layer_features[i]
        mu = feats.mean(dim=0)
        sigma = feats.std(dim=0, unbiased=False)
        d_mu = torch.linalg.vector_norm(mu.double() - source.means[i].to(mu.device).double())
        d_sigma = torch.linalg.vector_norm(
            sigma.double() - source.stds[i].to(sigma.device).double()
        )
        term = d_mu + d_sigma
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no layers selected for the activation discrepancy")
    if reduce == "mean":
        total = total / len(layers)
    total = total.to(layer_features[0].dtype)
    return total


def foa_fitness(
    logits: torch.Tensor,
    layer_features: Sequence[torch.Tensor],
    source: FeatureStatistics,
    lam: float,
    layers: Optional[Sequence[int]] = None,
    use_entropy: bool = True,
    use_activation_discrepancy: bool = True,
    normalize_entropy: bool = False,
) -> torch.Tensor:
    """Full Eqn. (5).  Lower is better (it is minimised by CMA-ES)."""
    value = None
    if use_entropy:
        value = prediction_entropy(
            logits, reduction="mean" if normalize_entropy else "sum"
        )
    if use_activation_discrepancy:
        disc = lam * activation_discrepancy(layer_features, source, layers=layers)
        value = disc if value is None else value + disc
    if value is None:
        raise ValueError("the fitness function needs at least one term")
    return value
