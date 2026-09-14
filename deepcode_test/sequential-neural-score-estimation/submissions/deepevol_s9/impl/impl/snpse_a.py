"""SNPSE-A post-hoc correction variant.

SNPSE-A starts from the same sequential posterior score estimation loop as
TSNPSE and applies a final sampling-importance-resampling style correction.
The implementation inherits TSNPSE, runs its sequential rounds, then draws a
larger pool of approximate posterior samples and resamples them according to
their approximate posterior log densities.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from impl.npse import _get_param
from impl.sampler import instantaneous_change_of_variables_log_density
from impl.tsnpse import Tsnpse


class SNPSEA(Tsnpse):
    """SNPSE-A method with a post-hoc SIR correction after sequential rounds."""

    @classmethod
    def from_task(cls, task, setting: str = "", device: str = "cpu", params=None):
        del params
        return cls(
            task.get_prior(),
            task.get_simulator(),
            theta_dim=getattr(task, "dim_parameters", None),
            x_dim=getattr(task, "dim_data", None),
            setting=setting or getattr(task, "name", ""),
            device=device,
            task=task,
        )

    def run_rounds(self, observation, params=None) -> Tensor:
        """Run sequential rounds and resample outputs by approximate density."""
        base_samples = super().run_rounds(observation, params)
        if base_samples.size(0) == 0:
            return base_samples

        target = int(base_samples.size(0))
        n_candidates = max(target * 4, target)
        candidates = self._sample_posterior(n_candidates)

        logs = instantaneous_change_of_variables_log_density(
            candidates, self.model, self.sde, self.observation
        )
        if not torch.isfinite(logs).all():
            logs = torch.where(
                torch.isfinite(logs), logs, torch.full_like(logs, -1e30)
            )
        logs = logs - logs.max()
        weights = torch.softmax(logs, dim=0)
        weights = weights + 1e-12
        weights = weights / weights.sum()

        try:
            indices = torch.multinomial(weights, target, replacement=True)
        except RuntimeError:
            indices = torch.randint(0, candidates.size(0), (target,))

        return candidates[indices]

    def run(self, observation, params=None) -> Tensor:
        """Alias for :meth:`run_rounds`."""
        return self.run_rounds(observation, params)
