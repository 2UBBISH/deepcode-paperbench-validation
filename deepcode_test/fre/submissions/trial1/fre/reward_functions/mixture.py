"""
Mixture reward function for FRE: uniformly samples one of the three
unsupervised reward function families (singleton goal-reaching, random linear,
random MLP) with probability 1/3 each.

This implements the prior reward distribution p(eta) described in the paper.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from fre.reward_functions.base import RewardFunction
from fre.reward_functions.singleton import SingletonRewardFunction
from fre.reward_functions.linear import LinearRewardFunction
from fre.reward_functions.mlp import MLPRewardFunction


class MixtureRewardFunction(RewardFunction):
    """
    Uniform mixture over three random reward function families:
      - Singleton goal-reaching (sparse reward)
      - Random linear functions with sparse mask
      - Random 2-layer MLPs

    At each call to reset() or forward(), one of the three types is sampled
    uniformly, and that function is used to compute rewards.
    """

    def __init__(
        self,
        state_dim: int,
        epsilon: float = 0.5,
        sparsity: float = 0.8,
        mlp_hidden_dim: int = 256,
        device: Optional[str] = None,
    ):
        """
        Args:
            state_dim: Dimensionality of the state space.
            epsilon: Threshold for singleton goal-reaching reward.
            sparsity: Fraction of linear weight entries to zero out.
            mlp_hidden_dim: Hidden dimension for the random MLP.
            device: Torch device.
        """
        super().__init__(state_dim, device)

        # Instantiate all three reward function types
        self.singleton = SingletonRewardFunction(
            state_dim=state_dim, epsilon=epsilon, device=device
        )
        self.linear = LinearRewardFunction(
            state_dim=state_dim, sparsity=sparsity, device=device
        )
        self.mlp = MLPRewardFunction(
            state_dim=state_dim, hidden_dim=mlp_hidden_dim, device=device
        )

        # Current active reward function type (index 0, 1, or 2)
        self._current_type: int = 0
        self._type_names = ["singleton", "linear", "mlp"]

        # Sample initial type
        self.reset()

    def reset(self):
        """
        Sample a new reward function type uniformly and reset the chosen
        function (e.g., sample new goal, new weights, new MLP).
        """
        self._current_type = torch.randint(0, 3, (1,)).item()

        # Reset the chosen function to generate a fresh random reward function
        if self._current_type == 0:
            self.singleton.reset()
        elif self._current_type == 1:
            self.linear.reset()
        else:
            self.mlp.reset()

    def _get_active_function(self) -> RewardFunction:
        """Return the currently active reward function."""
        if self._current_type == 0:
            return self.singleton
        elif self._current_type == 1:
            return self.linear
        else:
            return self.mlp

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for a batch of states using the currently active
        reward function type.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,).
        """
        active_fn = self._get_active_function()
        return active_fn(states)

    def get_info(self) -> Dict[str, Any]:
        """
        Return metadata about the current reward function.

        Returns:
            dict with keys: 'mixture_type', 'active_type', and the active
            function's info.
        """
        active_fn = self._get_active_function()
        info = {
            "mixture_type": "uniform",
            "active_type": self._type_names[self._current_type],
        }
        # Merge active function's info
        info.update(active_fn.get_info())
        return info

    def to(self, device: torch.device):
        """Move all sub-modules to the specified device."""
        super().to(device)
        self.singleton.to(device)
        self.linear.to(device)
        self.mlp.to(device)
        return self