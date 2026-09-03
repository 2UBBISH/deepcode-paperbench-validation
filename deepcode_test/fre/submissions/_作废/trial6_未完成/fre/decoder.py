"""
FRE Decoder: Feedforward reward predictor conditioned on latent z.

The decoder q_θ(η(s) | s, z) predicts the reward for a given state s
conditioned on the latent vector z produced by the FRE encoder.

Architecture:
    - Concatenate [s, z] -> input vector
    - MLP with 2-3 hidden layers and ReLU activations
    - Output: scalar predicted reward

This module is paired with the encoder in fre/fre_model.py for joint
VAE training (MSE reconstruction loss + KL divergence).
"""

from typing import List, Optional
import torch
import torch.nn as nn


class RewardDecoder(nn.Module):
    """
    Feedforward neural network that predicts reward η(s) given state s
    and latent conditioning vector z.

    Args:
        state_dim: Dimensionality of the state space.
        d_latent: Dimensionality of the latent vector z (from encoder).
        hidden_dims: List of hidden layer sizes (default: [256, 256]).
        activation: Activation function class (default: nn.ReLU).
        dropout: Dropout probability applied after each hidden layer (default: 0.0).
    """

    def __init__(
        self,
        state_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.d_latent = d_latent
        self.hidden_dims = hidden_dims

        input_dim = state_dim + d_latent

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer: scalar reward
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Kaiming uniform and zero biases."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict rewards for a batch of states conditioned on latent vectors.

        Args:
            states: Tensor of shape (batch_size, state_dim) or
                    (batch_size, num_states, state_dim).
            z: Tensor of shape (batch_size, d_latent).

        Returns:
            Predicted rewards of shape (batch_size, 1) or
            (batch_size, num_states, 1) depending on input shape.
        """
        # Handle multi-state input (batch, num_states, state_dim)
        if states.dim() == 3:
            batch_size, num_states, state_dim = states.shape
            # Expand z to match: (batch, num_states, d_latent)
            z_expanded = z.unsqueeze(1).expand(-1, num_states, -1)
            # Concatenate along last dim
            combined = torch.cat([states, z_expanded], dim=-1)
            # Flatten for MLP
            combined_flat = combined.reshape(batch_size * num_states, -1)
            out_flat = self.net(combined_flat)
            # Reshape back
            return out_flat.reshape(batch_size, num_states, 1)
        else:
            # Single state per z: (batch, state_dim)
            combined = torch.cat([states, z], dim=-1)
            return self.net(combined)

    def predict_batch(
        self, states: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method: predict rewards for a batch of decoder states.

        Args:
            states: (batch_size, K_prime, state_dim) decoder states.
            z: (batch_size, d_latent) latent vectors.

        Returns:
            (batch_size, K_prime, 1) predicted rewards.
        """
        return self.forward(states, z)


def create_reward_decoder(
    state_dim: int,
    d_latent: int = 64,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.0,
) -> RewardDecoder:
    """
    Factory function to create a RewardDecoder with given hyperparameters.

    Args:
        state_dim: Dimensionality of the state space.
        d_latent: Dimensionality of the latent vector.
        hidden_dims: Hidden layer sizes (default: [256, 256]).
        dropout: Dropout probability.

    Returns:
        RewardDecoder instance.
    """
    return RewardDecoder(
        state_dim=state_dim,
        d_latent=d_latent,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test
    state_dim = 29  # AntMaze state dim
    d_latent = 64
    batch_size = 4
    K_prime = 128  # number of decoder states

    decoder = RewardDecoder(state_dim=state_dim, d_latent=d_latent)

    # Test single-state prediction
    states_single = torch.randn(batch_size, state_dim)
    z = torch.randn(batch_size, d_latent)
    out_single = decoder(states_single, z)
    print(f"Single-state output shape: {out_single.shape}")  # (4, 1)

    # Test multi-state prediction (batch of decoder states)
    states_multi = torch.randn(batch_size, K_prime, state_dim)
    out_multi = decoder(states_multi, z)
    print(f"Multi-state output shape: {out_multi.shape}")  # (4, 128, 1)

    # Test predict_batch convenience method
    out_batch = decoder.predict_batch(states_multi, z)
    print(f"predict_batch output shape: {out_batch.shape}")  # (4, 128, 1)

    # Verify parameter count
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {num_params}")

    print("All decoder tests passed!")