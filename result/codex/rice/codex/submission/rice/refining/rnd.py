"""Random Network Distillation exploration bonus (Burda et al., 2018).

RICE adds ``lambda * ||f(s_{t+1}) - f_hat(s_{t+1})||^2`` to the task reward,
where ``f`` is a fixed randomly initialised network and ``f_hat`` is a
predictor trained to regress ``f``.  As the state coverage grows, the bonus
decays to zero and the agent recovers a purely extrinsic policy.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from rice.networks import init_orthogonal, mlp
from rice.running_stats import RunningMeanStd


class RND(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden: Iterable[int] = (64, 64),
        learning_rate: float = 1e-3,
        reward_scale: float = 1.0,
        normalize_reward: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        hidden = tuple(int(h) for h in hidden)
        self.obs_dim = int(obs_dim)
        self.target = mlp((obs_dim,) + hidden + (hidden[-1],), "relu")
        self.predictor = mlp((obs_dim,) + hidden + (hidden[-1],), "relu")
        init_orthogonal(self.target, gain=np.sqrt(2))
        init_orthogonal(self.predictor, gain=np.sqrt(2))
        for param in self.target.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.reward_scale = float(reward_scale)
        self.normalize_reward = bool(normalize_reward)
        self.reward_rms = RunningMeanStd(shape=())
        self.device = device

    @torch.no_grad()
    def error(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.atleast_2d(np.asarray(obs, dtype=np.float32)))
        obs_t = obs_t.to(self.device)
        diff = self.target(obs_t) - self.predictor(obs_t)
        err = diff.pow(2).sum(dim=-1).cpu().numpy()
        if self.normalize_reward:
            self.reward_rms.update(err)
            err = err / (self.reward_rms.std[()] + 1e-8)
        return err * self.reward_scale

    def update(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            target = self.target(obs_t)
        pred = self.predictor(obs_t)
        loss = nn.functional.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class RNDRewardShaper:
    """Adds the normalised RND bonus to the environment reward (Algorithm 2)."""

    def __init__(
        self,
        rnd: RND,
        coef: float,
        clip: Optional[float] = 5.0,
        normalize_obs=None,
    ):
        self.rnd = rnd
        self.coef = float(coef)
        self.clip = clip
        self.normalize_obs = normalize_obs or (lambda x: x)
        self._obs_buffer: list = []

    def shape(self, obs, next_obs, reward, done, info) -> float:
        next_obs = np.asarray(next_obs, dtype=np.float32)
        self._obs_buffer.append(next_obs.copy())
        bonus = float(self.rnd.error(next_obs)[0])
        if self.clip is not None:
            bonus = float(np.clip(bonus, -self.clip, self.clip))
        info["rnd_bonus"] = bonus
        return float(reward) + self.coef * bonus

    def after_iteration(self, buffer) -> dict:
        if not self._obs_buffer:
            return {}
        losses = [self.rnd.update(np.asarray(self._obs_buffer))]
        self._obs_buffer = []
        return {"rnd_loss": float(np.mean(losses))}
