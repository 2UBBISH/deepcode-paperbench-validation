"""
Singleton (Goal-Reaching) Reward Function

Implements goal-conditioned reward functions where reward = -1 for every
timestep where the goal is not reached, and 0 otherwise. Goals are sampled
uniformly from the offline dataset states.

This is one of the three reward families in the FRE prior distribution.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base import RewardFunction


class SingletonReward(RewardFunction):
    """
    Goal-reaching singleton reward function.

    Given a goal state g (sampled from the offline dataset), the reward for
    any state s is:
        r(s) = 0 if ||s - g||_2 < threshold, else -1

    The threshold is typically set to a small value (e.g., 0.5 for AntMaze,
    or domain-specific). For exact singleton matching, threshold can be set
    to a very small epsilon.

    Attributes:
        goal: The goal state tensor of shape (state_dim,).
        threshold: L2 distance threshold for considering goal reached.
    """

    def __init__(
        self,
        state_dim: int,
        goal: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the singleton reward function.

        Args:
            state_dim: Dimensionality of the state space.
            goal: Goal state tensor of shape (state_dim,). If None, a random
                goal will be set later via set_goal().
            threshold: L2 distance threshold for goal-reaching success.
            device: Torch device for tensor operations.
        """
        super().__init__(state_dim=state_dim, device=device)
        self.threshold = threshold

        if goal is not None:
            self.goal = goal.to(self.device)
        else:
            # Placeholder; must be set via set_goal() before use
            self.goal = torch.zeros(state_dim, device=self.device)

    def set_goal(self, goal: torch.Tensor):
        """
        Set or update the goal state.

        Args:
            goal: Goal state tensor of shape (state_dim,) or (1, state_dim).
        """
        self.goal = goal.reshape(-1).to(self.device)
        # Ensure correct dimensionality
        if self.goal.shape[0] != self.state_dim:
            raise ValueError(
                f"Goal dimension {self.goal.shape[0]} does not match "
                f"state_dim {self.state_dim}"
            )

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute singleton rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,) with values -1 or 0.
        """
        states = states.to(self.device)
        goal = self.goal.to(self.device)

        # Compute L2 distance between each state and the goal
        # states: (B, D), goal: (D,) -> distances: (B,)
        distances = torch.norm(states - goal.unsqueeze(0), p=2, dim=-1)

        # Reward: 0 if within threshold, -1 otherwise
        rewards = torch.where(distances < self.threshold,
                              torch.zeros_like(distances),
                              -torch.ones_like(distances))

        return rewards

    def numpy_call(self, states: np.ndarray) -> np.ndarray:
        """
        NumPy-compatible reward computation.

        Args:
            states: NumPy array of shape (batch_size, state_dim).

        Returns:
            rewards: NumPy array of shape (batch_size,).
        """
        states_t = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            rewards_t = self.__call__(states_t)
        return rewards_t.cpu().numpy()

    @staticmethod
    def sample_goal_from_dataset(
        dataset_states: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Sample a goal state uniformly from a dataset of states.

        Args:
            dataset_states: Array of shape (N, state_dim) containing all
                states from the offline dataset.
            rng: Optional NumPy RandomState for reproducibility.

        Returns:
            goal: Array of shape (state_dim,) sampled uniformly.
        """
        if rng is None:
            rng = np.random
        idx = rng.randint(0, len(dataset_states))
        return dataset_states[idx].copy()

    def __repr__(self) -> str:
        return (
            f"SingletonReward(state_dim={self.state_dim}, "
            f"threshold={self.threshold}, "
            f"goal_norm={self.goal.norm().item():.3f})"
        )