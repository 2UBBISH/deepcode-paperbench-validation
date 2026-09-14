'''Denoising score matching losses for NPSE and NLSE.

The two public functions compute Monte Carlo estimates of the conditional
denoising posterior score matching objective and the denoising likelihood score
matching objective.  The SDE objects are intentionally passed in rather than
imported so the same functions work for both VE and VP dynamics.
'''

from __future__ import annotations

import torch
from torch import Tensor


class _Dummy:  # pragma: no cover - kept to document the SDE protocol.
    pass


def _resolve_time(sde, t):
    if t is None:
        t = getattr(sde, '_current_t', None)
    if t is None:
        raise ValueError(
            'A diffusion time t is required to evaluate score matching losses. '
            'Pass t explicitly or set sde._current_t before calling the loss.'
        )
    return t


def denoising_posterior_score_matching_loss(
    score: Tensor,
    theta_t: Tensor,
    theta_0: Tensor,
    sde,
    t: Tensor | None = None,
) -> Tensor:
    '''Return a scalar Monte Carlo denoising posterior score matching loss.

    Parameters
    ----------
    score:
        Score network output for the diffused parameters.
    theta_t:
        Diffused parameter samples.
    theta_0:
        Original parameter samples.
    sde:
        Forward noising SDE with a ``score_target`` method.
    t:
        Diffusion times at which the samples were diffused. If omitted, the
        function reads ``sde._current_t`` so callers using the four-positional
        blueprint signature can set the current time before invocation.
    '''
    t = _resolve_time(sde, t)
    target = sde.score_target(theta_t, theta_0, t)
    residual = score - target
    return 0.5 * torch.mean(torch.sum(residual * residual, dim=-1))


def denoising_likelihood_score_matching_loss(
    score: Tensor,
    theta_t: Tensor,
    theta_0: Tensor,
    prior_score: Tensor,
    sde,
    t: Tensor | None = None,
) -> Tensor:
    '''Return a scalar denoising likelihood score matching loss.

    Here ``score`` is the likelihood score network output. The target for that
    network is ``sde.score_target(theta_t, theta_0, t) - prior_score``, which
    makes the sum ``score + prior_score`` approximate the posterior score.
    '''
    t = _resolve_time(sde, t)
    target = sde.score_target(theta_t, theta_0, t)
    residual = score + prior_score - target
    return 0.5 * torch.mean(torch.sum(residual * residual, dim=-1))
