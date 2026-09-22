"""Random Network Distillation (Burda et al., 2018).

Two networks with the same architecture: a randomly initialised *target* and a
trained *predictor*.  Both map an observation to a 512-dimensional vector.  The
prediction error is the intrinsic reward used to boost exploration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MontezumaConfig


def _orthogonal(layer: nn.Module, gain: float) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class RNDNetwork(nn.Module):
    """Small CNN used for both the RND target and predictor networks."""

    def __init__(self, feature_dim: int = 512, in_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flatten, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim)
        )
        self.apply(lambda m: _orthogonal(m, float(np.sqrt(2))))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.conv(x))


class _RunningMeanStd:
    """Running mean/variance with Welford's algorithm (as used in RND)."""

    def __init__(self, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total


class RND(nn.Module):
    """Random Network Distillation module.

    The target network is frozen; the predictor is trained to match it.  The
    intrinsic reward is the (normalised) prediction error.
    """

    def __init__(self, config: MontezumaConfig, feature_dim: int = 512) -> None:
        super().__init__()
        self.config = config
        self.target = RNDNetwork(feature_dim)
        self.predictor = RNDNetwork(feature_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimiser = torch.optim.Adam(self.predictor.parameters(), lr=config.learning_rate)

        # Running statistics used to whiten the intrinsic reward.
        self.reward_rms = _RunningMeanStd()
        self.obs_rms = _RunningMeanStd(shape=(1, config.preproc_height, config.preproc_width))

    def normalise_observation(self, obs: Tensor, update: bool = True) -> Tensor:
        if not self.config.use_norm:
            return obs
        if update:
            self.obs_rms.update(obs.detach().cpu().numpy())
        mean = torch.as_tensor(self.obs_rms.mean, dtype=obs.dtype, device=obs.device)
        var = torch.as_tensor(self.obs_rms.var, dtype=obs.dtype, device=obs.device)
        return (obs - mean) / torch.sqrt(var + 1e-8)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(intrinsic_reward, prediction_error)``."""

        with torch.no_grad():
            target = self.target(obs)
        prediction = self.predictor(obs)
        error = ((prediction - target) ** 2).mean(dim=1, keepdim=True)
        self.reward_rms.update(error.detach().cpu().numpy().ravel())
        intrinsic = error / torch.sqrt(
            torch.as_tensor(self.reward_rms.var, dtype=error.dtype, device=error.device) + 1e-8
        )
        return intrinsic, error

    def update(self, obs: Tensor) -> Tensor:
        """One predictor update on the sampled minibatch (``UpdateProportion``)."""

        with torch.no_grad():
            target = self.target(obs)
        prediction = self.predictor(obs)
        loss = F.mse_loss(prediction, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss.detach()


class RNDRewardWrapper:
    """Combines the extrinsic and intrinsic rewards: ``r = ext + int_coef * r_int``."""

    def __init__(self, rnd: RND, ext_coef: float = 2.0, int_coef: float = 1.0) -> None:
        self.rnd = rnd
        self.ext_coef = ext_coef
        self.int_coef = int_coef

    def __call__(self, obs: Tensor, extrinsic: Tensor) -> Tensor:
        intrinsic, _ = self.rnd(obs)
        return self.ext_coef * extrinsic + self.int_coef * intrinsic.squeeze(-1)
