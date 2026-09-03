"""
Abstract base class for reward functions used in FRE.

All reward functions (singleton goal-reaching, random linear, random MLP,
mixture, and evaluation rewards) inherit from this base class and implement
the __call__ method to compute scalar rewards for given states.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class RewardFunction(ABC, nn.Module):
    """
    Abstract base class for all reward functions.

    A reward function maps a batch of states to a batch of scalar rewards:
        r = eta(s),  where s has shape (batch_size, state_dim)
        and r has shape (batch_size,).

    Subclasses must implement:
        - forward(states) -> rewards: the core reward computation.
        - Optionally override __call__ for convenience.

    Inheriting from nn.Module allows reward functions with learnable
    parameters (e.g., random MLPs) to be treated as PyTorch modules,
    enabling easy device placement and parameter management.
    """

    def __init__(self, state_dim: int, device: Optional[str] = None):
        """
        Args:
            state_dim: Dimensionality of the state space.
            device: Torch device string (e.g., 'cuda', 'cpu'). If None,
                    uses the default device.
        """
        super().__init__()
        self.state_dim = state_dim
        if device is not None:
            self.to(device)

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,) containing scalar rewards.
        """
        pass

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper around forward(). Allows calling the reward
        function as eta(states).

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,).
        """
        return self.forward(states)

    def compute_on_dataset(
        self,
        states: torch.Tensor,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """
        Compute rewards for a large set of states in batches to avoid OOM.

        Args:
            states: Tensor of shape (N, state_dim).
            batch_size: Maximum batch size for processing.

        Returns:
            rewards: Tensor of shape (N,).
        """
        n = states.shape[0]
        rewards_list = []
        for i in range(0, n, batch_size):
            batch = states[i : i + batch_size]
            rewards_list.append(self.forward(batch))
        return torch.cat(rewards_list, dim=0)

    def reset(self):
        """
        Optional reset method for stateful reward functions (e.g., those
        that track episode progress). Default is no-op.
        """
        pass

    def get_info(self) -> dict:
        """
        Return a dictionary with information about this reward function
        (e.g., goal state, weight vector). Useful for logging and debugging.

        Returns:
            info: Dictionary of reward function metadata.
        """
        return {"type": self.__class__.__name__, "state_dim": self.state_dim}