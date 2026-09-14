'''SNPSE-A post-hoc correction variant.

SNPSE-A starts from the same sequential posterior score estimation loop as
TSNPSE and applies a final sampling-importance-resampling style correction.
The correction uses a cheap multivariate Gaussian kernel density estimate over
the base posterior samples so the smoke path does not require additional
probability-flow ODE solves or autograd trace computations.
'''

from __future__ import annotations

import math

import torch
from torch import Tensor

from impl.tsnpse import Tsnpse


class SNPSEA(Tsnpse):
    '''SNPSE-A method with a post-hoc SIR correction after sequential rounds.'''

    @classmethod
    def from_task(cls, task, setting: str = '', device: str = 'cpu', params=None):
        del params
        return cls(
            task.get_prior(),
            task.get_simulator(),
            theta_dim=getattr(task, 'dim_parameters', None),
            x_dim=getattr(task, 'dim_data', None),
            setting=setting or getattr(task, 'name', ''),
            device=device,
            task=task,
        )

    def run_rounds(self, observation, params=None) -> Tensor:
        '''Run sequential rounds and resample outputs with cheap KDE-SIR weights.'''
        base_samples = super().run_rounds(observation, params)
        if base_samples.size(0) == 0:
            return base_samples

        target = int(base_samples.size(0))
        candidates = base_samples
        n = candidates.size(0)
        if n < 2:
            return candidates

        diffs = candidates.unsqueeze(0) - candidates.unsqueeze(1)
        dim = int(candidates.size(1))

        # Silverman rule-of-thumb bandwidth for a product Gaussian kernel.
        std = candidates.std(dim=0).clamp_min(1e-6)
        bandwidth = (
            (4.0 / (dim + 2.0)) ** (1.0 / (dim + 4.0))
            * std
            * (n ** (-1.0 / (dim + 4.0)))
        )
        inv_bw = 1.0 / bandwidth
        scaled = diffs * inv_bw
        log_kernel = -0.5 * (scaled * scaled).sum(dim=-1)
        log_density = torch.logsumexp(log_kernel, dim=1) - math.log(float(n))

        log_density = log_density - log_density.max()
        weights = torch.softmax(log_density, dim=0)
        weights = weights + 1e-12
        weights = weights / weights.sum()

        try:
            indices = torch.multinomial(
                weights.detach().cpu(), target, replacement=True
            ).to(candidates.device)
        except RuntimeError:
            indices = torch.randint(0, n, (target,), device=candidates.device)

        return candidates[indices]

    def run(self, observation, params=None) -> Tensor:
        '''Alias for :meth:`run_rounds`.'''
        return self.run_rounds(observation, params)
