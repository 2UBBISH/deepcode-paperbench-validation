"""Forgetting diagnostics: expert log-likelihoods, PCA projections and
level-visitation densities.

* :func:`expert_log_likelihood` -- log-likelihood that the fine-tuned policy
  assigns to state-action pairs ``(s, a*)`` collected with the expert
  ``a* ~ pi_*(s)`` (Figure 8),
* :func:`pca_projection` -- 2D PCA projection used to visualise the
  log-likelihoods (Figure 8, bottom row),
* :func:`level_visitation_density` -- density of the maximum dungeon level
  against the number of turns (Figure 4, 16).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def expert_log_likelihood(
    policy_log_prob_fn: Callable[[Tensor, Tensor], Tensor],
    observations: Tensor,
    expert_actions: Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-sample log-likelihood ``log pi_theta(a* | s)`` for expert actions."""

    likelihoods = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            obs = observations[start : start + batch_size]
            actions = expert_actions[start : start + batch_size]
            likelihoods.append(policy_log_prob_fn(obs, actions).detach().cpu().numpy())
    return np.concatenate(likelihoods, axis=0)


def pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project ``features`` to ``n_components`` dimensions with PCA."""

    features = np.asarray(features, dtype=np.float64)
    features = features - features.mean(axis=0, keepdims=True)
    # SVD-based PCA (equivalent to sklearn's implementation for centred data).
    _, _, vt = np.linalg.svd(features, full_matrices=False)
    return features @ vt[:n_components].T


def level_visitation_density(
    max_levels: Sequence[int],
    turns: Sequence[int],
    bins: Tuple[int, int] = (30, 60),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of ``(turns, max_level)`` used for the density plots.

    Returns ``(histogram, turn_edges, level_edges)``.
    """

    histogram, turn_edges, level_edges = np.histogram2d(
        np.asarray(turns, dtype=np.float64),
        np.asarray(max_levels, dtype=np.float64),
        bins=bins,
    )
    return histogram, turn_edges, level_edges
