"""Shared IQL implementation used by the goal-conditioned baseline.

This is the same IQL update rule as ``fre.iql`` (the FRE agent conditions its
Q/V/policy on the latent ``z``; GC-IQL instead conditions on the goal state),
so both share a single implementation to guarantee that the comparison in
Table 1 is apples-to-apples: "FRE, GC-IQL, and GC-BC are implemented within the
same codebase and with the same network structure."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from fre.iql import IQLAgent, IQLConfig


@dataclass
class IQLCoreConfig:
    """IQL hyperparameters (identical defaults to the FRE agent)."""

    input_dim: int
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 512)
    discount: float = 0.88
    expectile: float = 0.8
    awr_temperature: float = 3.0
    target_update_rate: float = 0.001
    learning_rate: float = 1e-4
    max_advantage_weight: float = 100.0


class IQLCore:
    """IQL over ``(obs, conditioning)`` inputs, where conditioning is broadcast."""

    def __init__(self, config: IQLCoreConfig, device: torch.device = torch.device("cpu")) -> None:
        self.config = config
        self.device = device
        # ``input_dim`` is the total conditioning dimension fed to the networks
        # in addition to the action (i.e. obs_dim + goal_dim for GC-IQL).
        self.agent = IQLAgent(
            IQLConfig(
                obs_dim=config.input_dim,
                action_dim=config.action_dim,
                z_dim=0,
                hidden_dims=config.hidden_dims,
                discount=config.discount,
                expectile=config.expectile,
                awr_temperature=config.awr_temperature,
                target_update_rate=config.target_update_rate,
                learning_rate=config.learning_rate,
                max_advantage_weight=config.max_advantage_weight,
            ),
            device=device,
        )

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        mask: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> Dict[str, float]:
        """One IQL update step.

        ``obs`` / ``next_obs`` are the *base* observations; ``conditioning`` is
        concatenated to them (the goal state for GC-IQL).  ``mask`` is the
        terminal flag (1.0 where the goal was reached).
        """
        zero = torch.zeros(1, 0, device=self.device)
        zero = torch.zeros(obs.shape[0], 0, device=self.device)
        obs_in = torch.cat([obs, conditioning], dim=-1)
        next_in = torch.cat([next_obs, conditioning], dim=-1)
        stats = {}
        stats.update(self.agent.update_value(obs_in, action, zero))
        stats.update(self.agent.update_q(obs_in, action, reward, next_in, mask, zero))
        stats.update(self.agent.update_policy(obs_in, action, zero))
        self.agent.update_target()
        return stats

    @torch.no_grad()
    def act(self, obs: torch.Tensor, conditioning: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        zero = torch.zeros(obs.shape[0], 0, device=self.device)
        return self.agent.act(torch.cat([obs, conditioning], dim=-1), zero, deterministic=deterministic)

    @torch.no_grad()
    def value(self, obs: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(obs.shape[0], 0, device=self.device)
        return self.agent.value(torch.cat([obs, conditioning], dim=-1), zero)

    def state_dict(self):
        return self.agent.state_dict()

    def load_state_dict(self, state) -> None:
        self.agent.load_state_dict(state)
