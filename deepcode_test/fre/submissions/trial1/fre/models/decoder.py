"""
FRE Decoder: Feedforward reward decoder.

The decoder takes a state and a latent vector z (encoding a reward function)
and predicts the scalar reward r(s) = decoder(s, z).

Architecture: 2-3 hidden layers, 256 units each, ReLU activation.
Input: [s; z] concatenated, output: scalar predicted reward.
"""

import torch
import torch.nn as nn
from typing import Optional


class RewardDecoder(nn.Module):
    """
    Feedforward neural network that decodes a latent vector z and a state s
    into a predicted scalar reward.

    Input: state s (dim: state_dim) and latent z (dim: d_latent) concatenated.
    Output: scalar reward prediction.

    Args:
        state_dim: Dimensionality of the state space.
        d_latent: Dimensionality of the latent vector z.
        hidden_dims: List of hidden layer sizes (default: [256, 256]).
        activation: Activation function name ('relu', 'leaky_relu', 'gelu').
        dropout: Dropout rate applied after each hidden layer (default: 0.0).
    """

    def __init__(
        self,
        state_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.d_latent = d_latent
        self.hidden_dims = hidden_dims

        # Build activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        input_dim = state_dim + d_latent
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer: scalar reward
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict rewards for a batch of states given a latent vector z.

        Args:
            states: Tensor of shape (batch_size, state_dim) or
                    (batch_size, K, state_dim) for multiple states per z.
            z: Tensor of shape (batch_size, d_latent).

        Returns:
            Predicted rewards of shape (batch_size,) or (batch_size, K).
        """
        # Handle multi-state case: (batch_size, K, state_dim)
        if states.dim() == 3:
            batch_size, K, state_dim = states.shape
            # Expand z to match: (batch_size, K, d_latent)
            z_expanded = z.unsqueeze(1).expand(-1, K, -1)
            # Concatenate along last dim: (batch_size, K, state_dim + d_latent)
            combined = torch.cat([states, z_expanded], dim=-1)
            # Flatten for forward pass: (batch_size * K, input_dim)
            combined_flat = combined.reshape(-1, state_dim + self.d_latent)
            # Forward
            out = self.net(combined_flat)
            # Reshape back: (batch_size, K)
            return out.reshape(batch_size, K).squeeze(-1)
        else:
            # Single state per z: (batch_size, state_dim)
            combined = torch.cat([states, z], dim=-1)
            out = self.net(combined)
            return out.squeeze(-1)  # (batch_size,)

    def predict_batch(
        self, states: torch.Tensor, z: torch.Tensor, batch_size: int = 1024
    ) -> torch.Tensor:
        """
        Predict rewards for a large number of states in batches to avoid OOM.

        Args:
            states: Tensor of shape (num_states, state_dim).
            z: Tensor of shape (d_latent,) or (1, d_latent).
            batch_size: Maximum batch size for forward pass.

        Returns:
            Predicted rewards of shape (num_states,).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, d_latent)

        num_states = states.shape[0]
        all_rewards = []

        for i in range(0, num_states, batch_size):
            batch_states = states[i : i + batch_size]
            # Expand z to match batch
            z_batch = z.expand(batch_states.shape[0], -1)
            rewards = self.forward(batch_states, z_batch)
            all_rewards.append(rewards)

        return torch.cat(all_rewards, dim=0)


# Simple test
if __name__ == "__main__":
    # Test decoder
    state_dim = 29  # AntMaze state dim
    d_latent = 64
    batch_size = 32
    K = 32

    decoder = RewardDecoder(state_dim=state_dim, d_latent=d_latent)

    # Test single state per z
    states = torch.randn(batch_size, state_dim)
    z = torch.randn(batch_size, d_latent)
    rewards = decoder(states, z)
    print(f"Single state per z: input {states.shape}, z {z.shape} -> output {rewards.shape}")
    assert rewards.shape == (batch_size,), f"Expected ({batch_size},), got {rewards.shape}"

    # Test multiple states per z (K states)
    states_K = torch.randn(batch_size, K, state_dim)
    z_single = torch.randn(batch_size, d_latent)
    rewards_K = decoder(states_K, z_single)
    print(f"K states per z: input {states_K.shape}, z {z_single.shape} -> output {rewards_K.shape}")
    assert rewards_K.shape == (batch_size, K), f"Expected ({batch_size}, {K}), got {rewards_K.shape}"

    # Test predict_batch
    many_states = torch.randn(5000, state_dim)
    z_one = torch.randn(d_latent)
    rewards_many = decoder.predict_batch(many_states, z_one)
    print(f"Predict batch: input {many_states.shape}, z {z_one.shape} -> output {rewards_many.shape}")
    assert rewards_many.shape == (5000,), f"Expected (5000,), got {rewards_many.shape}"

    print("All decoder tests passed!")