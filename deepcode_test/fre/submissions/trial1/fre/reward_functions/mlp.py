"""
Random MLP Reward Function

Implements a random 2-layer MLP reward function for the FRE framework.
The MLP is initialized with random weights (Xavier uniform) and kept fixed
(not trained). It acts as a random nonlinear reward function, one of the
three unsupervised reward function families used to pre-train the encoder.

Architecture:
    - Hidden layer: 256 units, ReLU activation
    - Output: scalar reward
    - Weights initialized via Xavier uniform, biases uniform(-0.1, 0.1)
"""

from typing import Optional
import torch
import torch.nn as nn

from fre.reward_functions.base import RewardFunction


class MLPRewardFunction(RewardFunction):
    """
    A random 2-layer MLP that maps states to scalar rewards.

    The network weights are randomly initialized and frozen (not updated
    during training). Each call to reset() re-initializes the network
    with a fresh set of random weights, producing a different reward function.

    Architecture:
        Linear(state_dim -> 256) -> ReLU -> Linear(256 -> 1)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        activation: str = "relu",
        device: Optional[str] = None,
    ):
        """
        Initialize the random MLP reward function.

        Args:
            state_dim: Dimensionality of the state space.
            hidden_dim: Number of units in the hidden layer (default: 256).
            activation: Activation function name ('relu', 'tanh', etc.).
            device: Torch device string (e.g., 'cuda', 'cpu').
        """
        super().__init__(state_dim=state_dim, device=device)

        self.hidden_dim = hidden_dim
        self.activation_name = activation

        # Build the MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights with Xavier uniform and biases small
        self._init_weights()

        # Move to device
        if self.device is not None:
            self.to(self.device)

        # Freeze all parameters (this is a fixed random function)
        for param in self.net.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def reset(self):
        """
        Re-initialize the MLP with a fresh set of random weights.

        This produces a new random reward function, which is useful during
        training when sampling many different reward functions.
        """
        self._init_weights()
        # Ensure parameters remain frozen
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar rewards for a batch of states.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            rewards: Tensor of shape (batch_size,) with scalar rewards.
        """
        # Ensure states are on the correct device
        if states.device != self.device:
            states = states.to(self.device)

        # Forward through the MLP and squeeze to (batch_size,)
        rewards = self.net(states).squeeze(-1)
        return rewards

    def get_info(self) -> dict:
        """
        Return metadata about this reward function.

        Returns:
            dict with keys: 'type', 'hidden_dim', 'activation'.
        """
        return {
            "type": "mlp",
            "hidden_dim": self.hidden_dim,
            "activation": self.activation_name,
        }