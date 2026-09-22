'''HPR estimation and truncated proposal rejection sampling for TSNPSE.

The paper constructs each sequential proposal by estimating the
highest-probability region of the current approximate posterior and then
rejection-sampling the original prior restricted to that region.  This module
implements the required threshold estimation and rejection sampler.
'''

from __future__ import annotations

import torch
from torch import Tensor

from impl.sampler import (
    ProbabilityFlowODESampler,
    instantaneous_change_of_variables_log_density,
)


class TruncatedProposalSampler:
    '''Estimates an HPR threshold and draws truncated prior samples.

    Parameters
    ----------
    prior:
        The original prior. May be a callable ``prior(num_samples=n)`` or an
        object with a ``sample`` method.
    score_model:
        Current approximate posterior score network.
    sde:
        Forward noising SDE used by the score model.
    observation:
        Observed summary statistics.
    n_hpr_samples:
        Number of approximate posterior samples used to estimate the
        truncation threshold.
    epsilon:
        HPR quantile. The paper uses ``5e-4``.
    '''

    def __init__(
        self,
        prior,
        score_model,
        sde,
        observation,
        n_hpr_samples: int = 256,
        epsilon: float = 5e-4,
    ):
        self.prior = prior
        self.score_model = score_model
        self.sde = sde
        self.observation = torch.as_tensor(observation, dtype=torch.float32)
        self.n_hpr_samples = max(1, int(n_hpr_samples))
        self.epsilon = float(epsilon)
        self._threshold = None
        self._posterior_samples = None
        self._lower = None
        self._upper = None

    def _sample_prior(self, n: int) -> Tensor:
        prior = self.prior
        if callable(prior):
            try:
                samples = prior(n)
                return torch.as_tensor(samples, dtype=torch.float32)
            except TypeError:
                try:
                    samples = prior(num_samples=n)
                    return torch.as_tensor(samples, dtype=torch.float32)
                except TypeError as exc:
                    raise TypeError(
                        'prior callable must accept n or num_samples=n'
                    ) from exc

        if hasattr(prior, 'sample'):
            try:
                samples = prior.sample((n,))
                return torch.as_tensor(samples, dtype=torch.float32)
            except TypeError:
                try:
                    samples = prior.sample(n)
                    return torch.as_tensor(samples, dtype=torch.float32)
                except TypeError as exc:
                    raise TypeError(
                        'prior.sample must accept sample_shape or n'
                    ) from exc

        raise TypeError('prior must be callable or expose a sample method')

    def estimate_hpr_threshold(self) -> float:
        '''Estimate the epsilon quantile of approximate posterior log densities.'''
        sampler = ProbabilityFlowODESampler(
            self.score_model, self.sde, self.observation
        )
        samples = sampler.sample(self.n_hpr_samples)
        logs = instantaneous_change_of_variables_log_density(
            samples, self.score_model, self.sde, self.observation
        )
        self._posterior_samples = samples.detach()
        self._threshold = float(torch.quantile(logs, self.epsilon).detach().item())
        self._lower = samples.min(dim=0).values
        self._upper = samples.max(dim=0).values
        return self._threshold

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            self.estimate_hpr_threshold()
        return self._threshold

    def _inside_hypercube(self, candidates):
        if self._lower is None or self._upper is None:
            return torch.ones(candidates.size(0), dtype=torch.bool)
        lower = self._lower.to(candidates.device)
        upper = self._upper.to(candidates.device)
        return torch.all((candidates >= lower) & (candidates <= upper), dim=-1)

    def sample(self, n_samples: int) -> Tensor:
        '''Rejection-sample ``n_samples`` parameters from the truncated prior.'''
        n_samples = int(n_samples)
        if n_samples <= 0:
            dim = getattr(self.score_model, 'theta_dim', None)
            if dim is None:
                dim = self.sde.theta_dim if hasattr(self.sde, 'theta_dim') else 0
            return torch.empty((0, dim))

        if self._threshold is None:
            self.estimate_hpr_threshold()

        accepted_parts = []
        remaining = n_samples
        attempts = 0
        max_attempts = max(1000, min(20000, n_samples * 20))

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            draw_size = min(4096, max(128, remaining * 20))
            candidates = self._sample_prior(draw_size)
            inside = self._inside_hypercube(candidates)
            candidates = candidates[inside]
            if candidates.numel() == 0:
                continue

            for start in range(0, candidates.size(0), 128):
                chunk = candidates[start : start + 128]
                logs = instantaneous_change_of_variables_log_density(
                    chunk, self.score_model, self.sde, self.observation
                )
                mask = logs >= self._threshold
                accepted = chunk[mask]
                if accepted.size(0) > 0:
                    take = min(accepted.size(0), remaining)
                    accepted_parts.append(accepted[:take])
                    remaining -= take
                if remaining <= 0:
                    break

        if remaining > 0:
            sampler = ProbabilityFlowODESampler(
                self.score_model, self.sde, self.observation
            )
            accepted_parts.append(sampler.sample(remaining).cpu())

        return torch.cat(accepted_parts, dim=0).cpu()
