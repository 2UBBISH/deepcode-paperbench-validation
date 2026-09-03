"""Actor-critic networks for offline RL.

This module provides small MLP-based networks for implicit Q-learning and
related baselines. All networks optionally accept a *conditioning vector*
(`z` for FRE or `g` for goal-conditioned methods) that is concatenated with
the state (and action for Q-functions).

Implemented networks:
  - :class:`ValueNetwork`: V(s, c) scalar value.
  - :class:`QNetwork`: Q(s, a, c) scalar Q value.
  - :class:`GaussianPolicy`: diagonal Gaussian policy with tanh squashing.
  - :class:`DeterministicPolicy`: optional deterministic policy (used by some
    baselines and behavioral cloning variants).
  - :func:`soft_update`: Polyak target-network update.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    """Construct a simple feed-forward MLP."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _concat_condition(state: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
    """Concatenate an optional conditioning vector to a state tensor."""
    if condition is None:
        return state
    if condition.dim() == state.dim() - 1:
        # Allow (batch, cond_dim) to be broadcast over extra state dims.
        while condition.dim() < state.dim():
            condition = condition.unsqueeze(1)
    return torch.cat([state, condition], dim=-1)


class ValueNetwork(nn.Module):
    """State(-condition) value network V(s, c)."""

    def __init__(
        self,
        state_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.input_dim = state_dim + condition_dim
        self.mlp = _build_mlp(
            self.input_dim,
            1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )

    def forward(self, state: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = _concat_condition(state, condition)
        return self.mlp(x).squeeze(-1)


class QNetwork(nn.Module):
    """State-action(-condition) Q network Q(s, a, c)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.input_dim = state_dim + action_dim + condition_dim
        self.mlp = _build_mlp(
            self.input_dim,
            1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if condition is not None:
            if condition.dim() == state.dim() - 1:
                condition = condition.unsqueeze(1).expand(
                    *state.shape[:-1], -1
                )
            x = torch.cat([state, action, condition], dim=-1)
        else:
            x = torch.cat([state, action], dim=-1)
        return self.mlp(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with tanh action squashing.

    ``forward`` returns ``(mean, log_std)``. ``sample`` returns an action and
    its (squashing-corrected) log probability. ``log_prob`` evaluates a given
    action under the policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: nn.Module = nn.ReLU,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.input_dim = state_dim + condition_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.shared = _build_mlp(
            self.input_dim,
            hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden - 1,
            activation=activation,
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _concat_condition(state, condition)
        x = self.shared(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _distribution(
        self, state: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> Tuple[Normal, torch.Tensor]:
        mean, log_std = self.forward(state, condition)
        std = log_std.exp()
        return Normal(mean, std), mean

    def sample(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, mean)``.

        When ``deterministic`` is true the squashed mean is returned and the
        log-probability is ``None``.
        """
        dist, mean = self._distribution(state, condition)
        if deterministic:
            action = torch.tanh(mean)
            return action, None, mean

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        return action, log_prob, mean

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dist, _ = self._distribution(state, condition)
        clamped_action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(clamped_action)
        return self._squashed_log_prob(dist, raw_action, action)

    @staticmethod
    def _squashed_log_prob(
        dist: Normal, raw_action: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        # Tanh change-of-variables correction.
        log_prob = log_prob - torch.log(
            torch.clamp(1.0 - action.pow(2), min=1e-6)
        ).sum(dim=-1, keepdim=True)
        return log_prob

    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        action, _, _ = self.sample(state, condition, deterministic=deterministic)
        return action


class DeterministicPolicy(nn.Module):
    """Deterministic policy with optional tanh squashing.

    Useful for deterministic BC baselines and for some DDPG-style baselines.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: nn.Module = nn.ReLU,
        squashing: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.input_dim = state_dim + condition_dim
        self.squashing = squashing
        self.mlp = _build_mlp(
            self.input_dim,
            action_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )

    def forward(
        self, state: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = _concat_condition(state, condition)
        action = self.mlp(x)
        if self.squashing:
            action = torch.tanh(action)
        return action

    def get_action(
        self, state: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(state, condition)


def soft_update(
    target: nn.Module, source: nn.Module, tau: float = 0.005
) -> None:
    """In-place Polyak update ``target <- (1-tau)*target + tau*source``."""
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


__all__ = [
    "ValueNetwork",
    "QNetwork",
    "GaussianPolicy",
    "DeterministicPolicy",
    "soft_update",
]
