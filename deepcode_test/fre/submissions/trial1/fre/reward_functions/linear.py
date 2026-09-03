"""
Random Linear Reward Function for FRE.

Samples a weight vector w ~ Uniform(-1, 1) of same dimensionality as state,
applies a sparse mask (randomly zeroing out a fraction of entries), and
computes reward as r(s) = dot(w, s).

This is one of the three random unsupervised reward function families used
to pre-train the FRE encoder (along with singleton goal-reaching and random MLPs).
"""

from typing import Optional

import torch
import torch.nn as nn

from fre.reward_functions.base import RewardFunction


class LinearRewardFunction(RewardFunction):
    """
    Random linear reward function with sparse masking.

    On each call to reset() (or on first forward), a new weight vector is
    sampled uniformly from [-1, 1] and a sparse mask is applied that zeros
    out a fraction (sparsity) of the entries. The reward is the dot product
    of the state with this masked weight vector.

    Args:
        state_dim: Dimensionality of the state space.
        sparsity: Fraction of weight entries to zero out (default 0.8).
        device: Torch device to place tensors on.
    """

    def __init__(
        self,
        state_dim: int,
        sparsity: float = 0.8,
        device: Optional[str] = None,
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.sparsity = sparsity

        # Register weight and mask as buffers (non-trainable but part of module state)
        self.register_buffer("weight", torch.zeros(state_dim, device=self.device))
        self.register_buffer("mask", torch.ones(state_dim, device=self.device))

        # Sample initial weights
        self.reset()

    def reset(self):
        """
        Sample a new random weight vector and sparse mask.

        Weight entries are drawn from Uniform(-1, 1). A fraction (sparsity)
        of entries are randomly selected and set to zero via the mask.
        """
        # Sample weights uniformly from [-1, 1]
        w = torch.empty(self.state_dim, device=self.device).uniform_(-1.0, 1.0)

        # Create sparse mask: randomly zero out sparsity fraction of entries
        num_nonzero = max(1, int(self.state_dim * (1.0 - self.sparsity)))
        mask = torch.zeros(self.state_dim, device=self.device)
        indices = torch.randperm(self.state_dim, device=self.device)[:num_nonzero]
        mask[indices] = 1.0

        self.weight.copy_(w)
        self.mask.copy_(mask)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,) with scalar rewards.
        """
        # Ensure states are on the correct device
        if states.device != self.device:
            states = states.to(self.device)

        # Apply mask to weights
        masked_weight = self.weight * self.mask

        # Compute dot product: r(s) = dot(w_masked, s)
        rewards = torch.matmul(states, masked_weight)

        return rewards

    def get_info(self) -> dict:
        """
        Return metadata about the current reward function.

        Returns:
            dict with keys: 'type', 'sparsity', 'num_active_dims'.
        """
        return {
            "type": "linear",
            "sparsity": self.sparsity,
            "num_active_dims": int(self.mask.sum().item()),
        }