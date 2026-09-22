"""Actor-critic network for Montezuma's Revenge (Appendix B.2)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


def orthogonal_init(module: nn.Module, gain: float = float(np.sqrt(2))) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class AtariActorCritic(nn.Module):
    """Shared CNN trunk with a policy head and a value head."""

    def __init__(self, num_actions: int = 18, hidden_dim: int = 512) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.trunk(torch.zeros(1, 4, 84, 84)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flatten, hidden_dim), nn.ReLU())
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.apply(orthogonal_init)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    # ------------------------------------------------------------------
    def features(self, obs: Tensor) -> Tensor:
        return self.fc(self.trunk(obs))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self.features(obs)
        return self.policy_head(h), self.value_head(h)

    def distribution(self, obs: Tensor) -> Categorical:
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def act(self, obs: Tensor) -> Tensor:
        return self.distribution(obs).sample()

    def evaluate_actions(self, obs: Tensor, actions: Tensor):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)


class RandomPolicy:
    """Uniform-random policy used in smoke tests and as a fallback."""

    def __init__(self, num_actions: int = 18) -> None:
        self.num_actions = num_actions

    def act(self, obs: Tensor) -> Tensor:
        return torch.randint(0, self.num_actions, (obs.shape[0],), device=obs.device)
