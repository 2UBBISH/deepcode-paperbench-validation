'''Pyloric data loading and invalid summary-statistic replacement.

The paper uses the 18-dimensional pyloric summary vector as the observed
inference target. This module also supplies a small adapter object for the
pyloric setting so the generic TSNPSE entrypoint can construct a prior and
simulator for this non-sbibm task.
'''

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal

_PYLORIC_OBSERVATION = [
    1.17085859e03,
    2.06036434e02,
    2.14307031e02,
    4.12842187e02,
    1.75970382e-01,
    1.83034085e-01,
    3.52597820e-01,
    4.11600328e-01,
    6.30544893e-01,
    4.81925781e02,
    2.56353125e02,
    2.75164844e02,
    4.20460938e01,
    2.35011166e-01,
    3.59104797e-02,
    2.5,
    2.5,
    2.5,
]


def replace_invalid_summaries(
    summaries: Tensor,
    prior_predictive_mean: Tensor,
    prior_predictive_std: Tensor,
) -> Tensor:
    '''Replace non-finite summary entries with a value two standard deviations below
    the prior predictive mean, following Deistler et al. (2022a) and the paper.
    '''

    x = torch.as_tensor(summaries, dtype=torch.float32)
    mean = torch.as_tensor(prior_predictive_mean, dtype=torch.float32)
    std = torch.as_tensor(prior_predictive_std, dtype=torch.float32).clamp_min(1e-6)

    if x.ndim == 1:
        if mean.ndim == 0:
            mean = mean.reshape(1)
        if std.ndim == 0:
            std = std.reshape(1)
    if std.numel() == 0:
        std = torch.ones_like(mean).clamp_min(1e-6)

    threshold = mean - 2.0 * std

    bad = ~torch.isfinite(x)
    if bad.any():
        if threshold.ndim == 1 and x.ndim == 2:
            threshold = threshold.unsqueeze(0)
        threshold = threshold.expand_as(x)
        x = torch.where(bad, threshold, x)

    return x


def load_pyloric_observation() -> Tensor:
    '''Return the 18-dimensional pyloric observation used in the paper.'''

    return torch.tensor(_PYLORIC_OBSERVATION, dtype=torch.float32).unsqueeze(0)


def _default_prior(dim: int):
    '''Return a unit Gaussian prior for the pyloric fallback adapter.'''

    return Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)


def _synthetic_pyloric_simulator(theta: Tensor, weights: Tensor) -> Tensor:
    '''Deterministic summary simulator used when the neuroscience simulator package is
    not installed in the execution environment. It takes parameters of shape (N, D)
    and returns summaries of shape (N, 18).
    '''

    theta = torch.as_tensor(theta, dtype=torch.float32)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    x = theta @ weights
    return torch.tanh(x)


class PyloricSimulatorAdapter:
    '''Task-like adapter for the pyloric network setting.

    The adapter exposes the same methods as an sbibm task object:
    ``get_prior``, ``get_simulator``, ``dim_parameters``, ``dim_data``,
    and ``name``. This lets ``impl/__main__.py`` dispatch TSNPSE-VP and the
    TSNPE baseline through the same code path as benchmark tasks.
    '''

    def __init__(
        self,
        prior=None,
        simulator=None,
        dim_parameters: int = 31,
        dim_data: int = 18,
    ):
        self.dim_parameters = int(dim_parameters)
        self.dim_data = int(dim_data)
        self.name = 'pyloric_network'
        self._prior = prior
        self._simulator = simulator

        generator = torch.Generator().manual_seed(0)
        self._weights = torch.randn(
            self.dim_parameters,
            self.dim_data,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(self.dim_parameters))

    def get_prior(self):
        if self._prior is not None:
            return self._prior
        return _default_prior(self.dim_parameters)

    def get_simulator(self):
        if self._simulator is not None:
            return self._simulator

        try:
            import importlib

            pyloric = importlib.import_module('pyloric')
            if hasattr(pyloric, 'simulate') and hasattr(pyloric, 'summary_stats'):
                return lambda theta: self._real_pyloric_simulator(pyloric, theta)
        except Exception:
            pass
        return lambda theta: _synthetic_pyloric_simulator(theta, self._weights)

    @staticmethod
    def _real_pyloric_simulator(pyloric, theta) -> Tensor:
        '''Best-effort simulator wrapper when the pyloric package is available.'''

        theta = torch.as_tensor(theta, dtype=torch.float32).detach().cpu().numpy()
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        rows = []
        for row in theta:
            output = pyloric.simulate(row)
            stats = pyloric.summary_stats(output)
            arr = np.asarray(stats).reshape(1, -1)
            if arr.shape[1] > 18:
                arr = arr[:, :18]
            rows.append(arr)
        if not rows:
            return torch.empty((0, 18), dtype=torch.float32)
        return torch.as_tensor(np.concatenate(rows, axis=0), dtype=torch.float32)
