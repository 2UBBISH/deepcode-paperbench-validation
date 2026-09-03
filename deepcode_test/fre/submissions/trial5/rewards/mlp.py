"""
Random MLP Reward Function for FRE Prior Distribution.

Implements a 2-layer MLP with random (untrained) weights that maps states
to scalar rewards. Weights are initialized via Kaiming uniform and remain
fixed throughout training. This is one of the three reward families in the
FRE prior distribution (alongside singleton and linear rewards).

Architecture: Linear(in_dim, 256) -> ReLU -> Linear(256, 1)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from rewards.base import RewardFunction


class RandomMLPReward(RewardFunction):
    """
    A random MLP reward function: reward(s) = MLP(s) with fixed random weights.

    The MLP has architecture: Linear(state_dim, 256) -> ReLU -> Linear(256, 1).
    Weights are initialized via Kaiming uniform and never trained.

    Args:
        state_dim: Dimensionality of the state space.
        hidden_dim: Hidden layer dimension (default: 256).
        device: Torch device for tensor operations.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__(state_dim=state_dim, device=device)

        self.hidden_dim = hidden_dim

        # Build the MLP: Linear -> ReLU -> Linear
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights with Kaiming uniform
        self._init_weights()

        # Move to device
        self._mlp.to(self.device)

        # Freeze all parameters (no training)
        for param in self._mlp.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """Initialize all Linear layers with Kaiming uniform."""
        for module in self._mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    # Initialize bias uniformly in [-1/sqrt(fan_in), 1/sqrt(fan_in)]
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                        module.weight
                    )
                    bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 1.0
                    nn.init.uniform_(module.bias, -bound, bound)

    def resample_weights(self):
        """
        Re-initialize the MLP weights with fresh random values.
        Useful for sampling a new reward function from the MLP family.
        """
        self._init_weights()

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            rewards: Tensor of shape (batch_size, 1) or (1,) with scalar rewards.
        """
        # Handle single state input
        if states.dim() == 1:
            states = states.unsqueeze(0)

        states = states.to(self.device).float()

        with torch.no_grad():
            rewards = self._mlp(states)

        return rewards

    def numpy_call(self, states: np.ndarray) -> np.ndarray:
        """
        Compute rewards for numpy array input.

        Args:
            states: NumPy array of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            rewards: NumPy array of shape (batch_size,) or scalar.
        """
        states_tensor = torch.from_numpy(states).float()
        rewards_tensor = self.__call__(states_tensor)
        return rewards_tensor.cpu().numpy().squeeze(-1)

    def __repr__(self) -> str:
        return (
            f"RandomMLPReward(state_dim={self.state_dim}, "
            f"hidden_dim={self.hidden_dim}, device={self.device})"
        )