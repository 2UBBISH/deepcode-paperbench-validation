"""
FRE Decoder: Feedforward MLP that predicts reward η(s) given state s and latent z.

Architecture:
- Input: state s (dim d_s) concatenated with latent z (dim d_z)
- Feedforward MLP: 2-3 hidden layers with ReLU activations
- Output: single scalar predicted reward η̂(s)

This decoder is trained jointly with the encoder as part of the FRE VAE.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardDecoder(nn.Module):
    """
    Feedforward MLP decoder that predicts a scalar reward given a state and latent vector.

    Architecture:
        Input: [s; z] concatenated (state_dim + latent_dim)
        Hidden layers: configurable number of layers with ReLU activations
        Output: single scalar (predicted reward)

    Args:
        state_dim: Dimension of the state space (d_s).
        latent_dim: Dimension of the latent vector z (d_z). Default: 64.
        hidden_dims: List of hidden layer dimensions. Default: [256, 256].
        activation: Activation function name. Default: "relu".
        dropout: Dropout rate applied after each hidden layer. Default: 0.0.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.input_dim = state_dim + latent_dim

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Build MLP layers
        layers = []
        in_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer: single scalar
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for given states and latent vector.

        Args:
            states: State tensor of shape (batch_size, state_dim) or
                    (batch_size, num_states, state_dim).
            z: Latent vector of shape (batch_size, latent_dim).

        Returns:
            Predicted rewards of shape (batch_size,) or (batch_size, num_states).
            Squeezes the last dimension (scalar output).
        """
        # Handle multi-state input: (batch, num_states, state_dim)
        if states.dim() == 3:
            batch_size, num_states, state_dim = states.shape
            # Expand z to match: (batch, num_states, latent_dim)
            z_expanded = z.unsqueeze(1).expand(-1, num_states, -1)
            # Concatenate: (batch, num_states, state_dim + latent_dim)
            combined = torch.cat([states, z_expanded], dim=-1)
            # Flatten for MLP: (batch * num_states, input_dim)
            combined_flat = combined.reshape(-1, self.input_dim)
            # Forward through MLP
            out = self.net(combined_flat)
            # Reshape back: (batch, num_states)
            out = out.reshape(batch_size, num_states)
            return out.squeeze(-1) if out.shape[-1] == 1 else out
        else:
            # Single state per batch item: (batch, state_dim)
            combined = torch.cat([states, z], dim=-1)
            out = self.net(combined)
            return out.squeeze(-1)

    def predict_batch(
        self, states: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method: predict rewards for a batch of states with a single z.

        Args:
            states: (batch_size, state_dim) or (batch_size, num_states, state_dim)
            z: (batch_size, latent_dim)

        Returns:
            Predicted rewards, same batch shape as states (minus last dim).
        """
        return self.forward(states, z)

    def predict_single(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict reward for a single state with a single z.

        Args:
            state: (state_dim,) or (1, state_dim)
            z: (latent_dim,) or (1, latent_dim)

        Returns:
            Scalar predicted reward.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.forward(state, z).squeeze()


def build_decoder(
    state_dim: int,
    latent_dim: int = 64,
    hidden_dims: Optional[List[int]] = None,
    activation: str = "relu",
    dropout: float = 0.0,
) -> RewardDecoder:
    """
    Factory function to build a RewardDecoder with default or custom settings.

    Args:
        state_dim: Dimension of state space.
        latent_dim: Dimension of latent vector z.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        dropout: Dropout rate.

    Returns:
        RewardDecoder instance.
    """
    return RewardDecoder(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with random data
    state_dim = 17
    latent_dim = 64
    batch_size = 8
    num_states = 32

    decoder = RewardDecoder(state_dim=state_dim, latent_dim=latent_dim)

    # Test single-state prediction
    states = torch.randn(batch_size, state_dim)
    z = torch.randn(batch_size, latent_dim)
    rewards = decoder(states, z)
    print(f"Single-state output shape: {rewards.shape}")  # Expected: (8,)

    # Test multi-state prediction
    states_multi = torch.randn(batch_size, num_states, state_dim)
    z = torch.randn(batch_size, latent_dim)
    rewards_multi = decoder(states_multi, z)
    print(f"Multi-state output shape: {rewards_multi.shape}")  # Expected: (8, 32)

    # Test predict_single
    state_single = torch.randn(state_dim)
    z_single = torch.randn(latent_dim)
    reward_single = decoder.predict_single(state_single, z_single)
    print(f"Single prediction: {reward_single.item():.4f}")

    # Test gradient flow
    states_grad = torch.randn(batch_size, state_dim, requires_grad=True)
    z_grad = torch.randn(batch_size, latent_dim, requires_grad=True)
    rewards_grad = decoder(states_grad, z_grad)
    loss = rewards_grad.sum()
    loss.backward()
    print(f"Gradient on states: {states_grad.grad is not None}")
    print(f"Gradient on z: {z_grad.grad is not None}")

    print("All decoder tests passed!")