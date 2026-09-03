"""
Abstract base class for reward functions used in Functional Reward Encodings (FRE).

All reward functions must implement the __call__ method, which takes a batch of
states and returns scalar rewards. This enables the FRE encoder to be trained on
diverse unsupervised reward functions sampled from a prior distribution.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class RewardFunction(ABC):
    """
    Abstract base class for all reward functions.

    A reward function η: S → R maps states to scalar rewards.
    Subclasses must implement the __call__ method.

    Attributes:
        state_dim: Dimensionality of the state space.
        device: Torch device for tensor operations.
    """

    def __init__(self, state_dim: int, device: Optional[torch.device] = None):
        """
        Initialize the reward function.

        Args:
            state_dim: Dimensionality of the state vectors.
            device: Torch device (CPU or CUDA). If None, defaults to CPU.
        """
        self.state_dim = state_dim
        self.device = device if device is not None else torch.device("cpu")

    @abstractmethod
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim) containing state vectors.

        Returns:
            Tensor of shape (batch_size,) containing scalar rewards.
        """
        pass

    def to(self, device: torch.device) -> "RewardFunction":
        """
        Move the reward function to the specified device.

        Args:
            device: Target torch device.

        Returns:
            Self, for method chaining.
        """
        self.device = device
        return self

    def numpy_call(self, states: np.ndarray) -> np.ndarray:
        """
        Convenience method to compute rewards from numpy arrays.

        Args:
            states: Numpy array of shape (batch_size, state_dim).

        Returns:
            Numpy array of shape (batch_size,) containing scalar rewards.
        """
        states_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            rewards = self(states_tensor)
        return rewards.cpu().numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_dim={self.state_dim})"