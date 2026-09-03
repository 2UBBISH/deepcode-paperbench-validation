"""
Random Linear Reward Function

Implements reward(s) = dot(w, s) where w ~ Uniform(-1, 1) with a sparse mask
(e.g., 80% zeros) to bias towards simple functions. This is one of the three
reward families in the FRE prior distribution.

Based on: "Functional Reward Encodings (FRE) for Zero-Shot Offline RL"
"""

from typing import Optional

import numpy as np
import torch

from rewards.base import RewardFunction


class RandomLinearReward(RewardFunction):
    """
    Random linear reward function: reward(s) = dot(w, s).

    The weight vector w is sampled uniformly from [-1, 1] and a sparse mask
    is applied to zero out a fraction of the weights, biasing the reward
    towards simple functions that depend on only a subset of state dimensions.

    Args:
        state_dim: Dimensionality of the state space.
        weights: Optional pre-defined weight vector. If None, random weights
            are sampled on construction.
        sparsity: Fraction of weights to set to zero (default 0.8, i.e., 80% zeros).
        device: Torch device for tensor operations.
    """

    def __init__(
        self,
        state_dim: int,
        weights: Optional[torch.Tensor] = None,
        sparsity: float = 0.8,
        device: Optional[torch.device] = None,
    ):
        super().__init__(state_dim=state_dim, device=device)

        self.sparsity = sparsity

        if weights is not None:
            self.weights = weights.to(self.device)
        else:
            self.weights = self._sample_weights(state_dim, sparsity, device)

    @staticmethod
    def _sample_weights(
        state_dim: int,
        sparsity: float = 0.8,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample a random sparse linear weight vector.

        Weights are drawn from Uniform(-1, 1), then a fraction (sparsity)
        are set to zero via a random binary mask.

        Args:
            state_dim: Dimensionality of the state space.
            sparsity: Fraction of weights to zero out.
            device: Torch device.

        Returns:
            Weight tensor of shape (state_dim,).
        """
        # Sample uniform weights in [-1, 1]
        w = torch.empty(state_dim, device=device).uniform_(-1.0, 1.0)

        # Create sparse mask: keep (1 - sparsity) fraction of weights
        num_nonzero = max(1, int(state_dim * (1.0 - sparsity)))
        mask = torch.zeros(state_dim, device=device)
        indices = torch.randperm(state_dim, device=device)[:num_nonzero]
        mask[indices] = 1.0

        w = w * mask
        return w

    def set_weights(self, weights: torch.Tensor) -> None:
        """
        Manually set the weight vector.

        Args:
            weights: New weight tensor of shape (state_dim,).
        """
        self.weights = weights.to(self.device)

    def resample_weights(self) -> None:
        """Resample a new random sparse weight vector."""
        self.weights = self._sample_weights(
            self.state_dim, self.sparsity, self.device
        )

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute linear reward for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Reward tensor of shape (batch_size,) or scalar.
        """
        states = states.to(self.device)
        # dot product: (batch, dim) @ (dim,) -> (batch,)
        rewards = torch.matmul(states, self.weights)
        return rewards

    def numpy_call(self, states: np.ndarray) -> np.ndarray:
        """
        Compute linear reward for numpy array states.

        Args:
            states: Numpy array of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Reward array of shape (batch_size,) or scalar.
        """
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            rewards = self.__call__(states_t)
        return rewards.cpu().numpy()

    def __repr__(self) -> str:
        nonzero = int((self.weights != 0).sum().item())
        return (
            f"RandomLinearReward(state_dim={self.state_dim}, "
            f"sparsity={self.sparsity:.2f}, nonzero_dims={nonzero})"
        )