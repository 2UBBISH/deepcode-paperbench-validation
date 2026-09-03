"""
Singleton Goal-Reaching Reward Function

Samples a goal state uniformly from the dataset and defines:
    r(s) = 0 if ||s - g||_2 < epsilon else -1

This is one of the three random unsupervised reward function families
used to pre-train the FRE encoder.
"""

from typing import Optional

import torch
import torch.nn as nn

from fre.reward_functions.base import RewardFunction


class SingletonRewardFunction(RewardFunction):
    """
    Goal-reaching reward: returns 0 when within epsilon of the goal state,
    -1 otherwise. The goal is sampled uniformly from a provided state dataset.

    Args:
        state_dim: Dimensionality of the state space.
        epsilon: Distance threshold for goal achievement.
        device: Torch device for tensors.
    """

    def __init__(
        self,
        state_dim: int,
        epsilon: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.epsilon = epsilon

        # Goal state will be set via sample_goal() before use
        self.register_buffer("goal_state", torch.zeros(state_dim))

    def sample_goal(self, states: torch.Tensor):
        """
        Sample a random goal state from a batch of dataset states.

        Args:
            states: Tensor of shape (N, state_dim) from which to sample.
        """
        idx = torch.randint(0, states.shape[0], (1,)).item()
        self.goal_state = states[idx].clone().to(self.device)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,) with values 0 or -1.
        """
        # Compute L2 distance between each state and the goal
        dist = torch.norm(states - self.goal_state.unsqueeze(0), dim=-1)

        # Reward: 0 if within epsilon, -1 otherwise
        rewards = torch.where(dist < self.epsilon, 0.0, -1.0)
        return rewards

    def get_info(self) -> dict:
        """Return metadata about this reward function."""
        return {
            "type": "singleton",
            "epsilon": self.epsilon,
            "goal_state": self.goal_state.cpu().tolist(),
        }