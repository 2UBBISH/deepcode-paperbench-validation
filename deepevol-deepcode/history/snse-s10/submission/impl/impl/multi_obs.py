'''Multiple-observation bridge score construction.'''

from __future__ import annotations

import torch
from torch import Tensor


def _num_observations(observations: Tensor, single_score: Tensor) -> int:
    if single_score.ndim == 3:
        return int(single_score.size(1))
    if observations.ndim == 3:
        return int(observations.size(1))
    if observations.ndim == 2 and single_score.ndim == 2 and observations.size(0) == single_score.size(0):
        return int(observations.size(0))
    if single_score.ndim == 2:
        return int(single_score.size(0))
    if observations.ndim == 2:
        return int(observations.size(0))
    return 1


def _broadcast_prior(prior: Tensor, target: Tensor) -> Tensor:
    if prior.shape == target.shape:
        return prior
    if target.ndim == 2 and prior.ndim == 1:
        return prior.unsqueeze(0).expand(target.shape[0], *prior.shape)
    if target.ndim == 1 and prior.ndim == 2 and prior.size(0) == 1:
        return prior.squeeze(0)
    try:
        return torch.broadcast_to(prior, target.shape)
    except RuntimeError:
        return prior


def multiple_observation_bridge_score(
    single_score: Tensor,
    prior_score: Tensor,
    observations: Tensor,
    t: float = 1.0,
) -> Tensor:
    '''Return the bridge score proposed in the paper for multiple observations.

    The bridge density is
    ``p_t^bridge(theta_t | x^1,...,x^n) proportional to
    p_t(theta_t)^(1-n) * product_i p_t(theta_t | x^i)``.
    The score is therefore
    ``(1-n) * prior_score + sum_i single_observation_scores_i``.
    The supplied ``single_score`` tensor is the already-evaluated single-observation
    score for every observation: shape ``(n, D)`` or ``(batch, n, D)``.
    '''

    del t  # kept for API compatibility with probability-flow ODE callers.

    single_score = torch.as_tensor(single_score, dtype=torch.float32)
    prior_score = torch.as_tensor(prior_score, dtype=torch.float32)
    observations = torch.as_tensor(observations, dtype=torch.float32)

    n = _num_observations(observations, single_score)

    if single_score.ndim == 3:
        summed = single_score.sum(dim=1)
    elif single_score.ndim == 2:
        if prior_score.ndim == 2 and prior_score.size(0) == single_score.size(0):
            summed = single_score
        else:
            summed = single_score.sum(dim=0)
    else:
        summed = single_score

    prior = _broadcast_prior(prior_score, summed)
    return (1.0 - float(n)) * prior + summed
