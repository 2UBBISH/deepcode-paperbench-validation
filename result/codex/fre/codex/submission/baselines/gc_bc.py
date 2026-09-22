"""Goal-Conditioned Behavioural Cloning (GC-BC).

From the addendum:

  * Network architecture: an MLP with three hidden layers of size 512, ReLU
    activations, and layer normalisation applied before each activation.  The
    output layer predicts a Gaussian distribution over actions -- a linear mean
    and a log standard deviation clamped with a lower bound of -5.0;
  * Loss: maximum likelihood estimation (MLE),

        L_pi = -E_{(s, g, a) ~ D} log pi(a | s, g);

  * Training: hindsight relabelling where the goal is sampled from the dataset.
    For GC-BC *only geometric sampling* is used to sample goals from future
    states in the trajectory (no random goals and no goals equal to the current
    state);
  * Evaluation: the goal-conditioned agent is given the ground-truth goal of
    the evaluation task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fre.datasets import OfflineDataset
from fre.decoder import mlp


@dataclass
class GCBCConfig:
    """Configuration for GC-BC."""

    obs_dim: int
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 512)
    log_std_min: float = -5.0
    learning_rate: float = 1e-4
    geometric_p: float = 0.2
    # goals closer than this (in normalised observation space) are treated as
    # reached; used only for reporting success during evaluation
    goal_reached_threshold: float = 0.5

    @property
    def input_dim(self) -> int:
        return self.obs_dim * 2


class GaussianActor(nn.Module):
    """Goal-conditioned Gaussian policy: ``pi(a | s, g)``.

    Layer normalisation is applied before every ReLU activation; the output
    layer produces the action mean and log standard deviation.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (512, 512, 512), log_std_min: float = -5.0) -> None:
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 2 * action_dim))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.log_std_min = log_std_min

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([obs, goal], dim=-1))
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = log_std.clamp_min(self.log_std_min)
        return mean, log_std

    def log_prob(self, obs: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs, goal)
        dist = torch.distributions.Normal(mean, log_std.exp())
        return dist.log_prob(action).sum(dim=-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs, goal)
        return torch.tanh(mean)


class GeometricGoalSampler:
    """Samples goals from future states in the same trajectory geometrically."""

    def __init__(self, dataset: OfflineDataset, geometric_p: float = 0.2, seed: int = 0) -> None:
        self.dataset = dataset
        self.geometric_p = float(geometric_p)
        self.rng = np.random.default_rng(seed)

    def sample(self, indices: np.ndarray) -> np.ndarray:
        ds = self.dataset
        tids = ds.traj_ids[indices]
        offsets = ds.traj_offsets[tids]
        lengths = ds.traj_lengths[tids]
        pos = indices - offsets
        remaining = np.maximum(lengths - pos - 1, 0)
        draws = self.rng.geometric(self.geometric_p, size=remaining.shape) - 1
        sampled = pos + 1 + np.minimum(draws, np.maximum(remaining - 1, 0))
        sampled = np.where(remaining > 0, sampled, pos)
        return ds.observations[offsets + sampled]


class GCBCAgent:
    """Goal-conditioned behavioural cloning agent."""

    def __init__(self, config: GCBCConfig, device: torch.device = torch.device("cpu")) -> None:
        self.config = config
        self.device = device
        self.actor = GaussianActor(config.input_dim, config.action_dim, config.hidden_dims, config.log_std_min).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)

    def update(
        self, obs: torch.Tensor, goal: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, float]:
        log_prob = self.actor.log_prob(obs, goal, action)
        loss = -log_prob.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"bc_loss": float(loss.item())}

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        goal_t = torch.as_tensor(np.asarray(goal, dtype=np.float32), device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if goal_t.dim() == 1:
            goal_t = goal_t.unsqueeze(0)
        return self.actor.act(obs_t, goal_t).cpu().numpy()

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "config": self.config}

    def load_state_dict(self, state) -> None:
        self.actor.load_state_dict(state["actor"])
