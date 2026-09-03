"""
FRE Decoder: MLP-based reward decoder.

The decoder predicts the reward η(s) for a query state s given a latent vector z.
It is trained jointly with the encoder to minimize MSE reconstruction loss.

Architecture:
    - Input: concatenation of state s (dim: state_dim) and latent z (dim: d_z)
    - Hidden layers: 2-3 MLP layers with ReLU activations
    - Output: scalar reward prediction

Paper reference: Section 3.2, "Decoder"
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardDecoder(nn.Module):
    """
    MLP decoder that predicts reward η(s) from state s and latent z.

    Architecture:
        Input: [s; z] → Linear(state_dim + d_z, hidden_dim) → ReLU
              → Linear(hidden_dim, hidden_dim) → ReLU
              → Linear(hidden_dim, 1)

    Args:
        state_dim: Dimensionality of the state space.
        d_z: Dimensionality of the latent vector z.
        hidden_dims: List of hidden layer dimensions. Default: [256, 256].
        activation: Activation function. Default: ReLU.
        dropout: Dropout rate applied after each hidden layer. Default: 0.0.
    """

    def __init__(
        self,
        state_dim: int,
        d_z: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.d_z = d_z
        self.input_dim = state_dim + d_z

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Build MLP layers
        layers = []
        in_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output layer: scalar reward
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform for linear layers."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict rewards for query states given latent vector z.

        Args:
            states: Tensor of shape (batch_size, state_dim) or (batch_size, K', state_dim).
            z: Tensor of shape (batch_size, d_z).

        Returns:
            Predicted rewards of shape (batch_size,) or (batch_size, K').
        """
        # Handle both (B, state_dim) and (B, K', state_dim) shapes
        if states.dim() == 3:
            # states: (B, K', state_dim)
            B, K_prime, _ = states.shape
            # Expand z to match: (B, d_z) → (B, K', d_z)
            z_expanded = z.unsqueeze(1).expand(-1, K_prime, -1)
            # Concatenate along last dim: (B, K', state_dim + d_z)
            combined = torch.cat([states, z_expanded], dim=-1)
            # Flatten for MLP: (B * K', input_dim)
            combined_flat = combined.reshape(-1, self.input_dim)
            # Forward
            out_flat = self.net(combined_flat)
            # Reshape back: (B, K')
            out = out_flat.reshape(B, K_prime)
        else:
            # states: (B, state_dim)
            combined = torch.cat([states, z], dim=-1)  # (B, input_dim)
            out = self.net(combined).squeeze(-1)  # (B,)

        return out

    def predict_batch(
        self, states: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method: predict rewards for a batch of states and latents.

        Args:
            states: Tensor of shape (batch_size, state_dim) or (batch_size, K', state_dim).
            z: Tensor of shape (batch_size, d_z).

        Returns:
            Predicted rewards.
        """
        return self.forward(states, z)


def reconstruction_loss(
    decoder: RewardDecoder,
    states: torch.Tensor,
    true_rewards: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE reconstruction loss for the decoder.

    L_recon = (1/K') Σ (η(s_j) - q_θ(s_j, z))²

    Args:
        decoder: RewardDecoder module.
        states: Query states of shape (batch_size, K', state_dim).
        true_rewards: True rewards η(s_j) of shape (batch_size, K').
        z: Latent vector of shape (batch_size, d_z).

    Returns:
        Scalar MSE loss averaged over batch and K'.
    """
    pred_rewards = decoder(states, z)  # (batch_size, K')
    loss = F.mse_loss(pred_rewards, true_rewards)
    return loss


def create_decoder(
    state_dim: int,
    d_z: int = 64,
    hidden_dims: Optional[List[int]] = None,
    **kwargs,
) -> RewardDecoder:
    """
    Factory function to create a RewardDecoder with default settings.

    Args:
        state_dim: Dimensionality of the state space.
        d_z: Latent dimension (default: 64).
        hidden_dims: Hidden layer dimensions (default: [256, 256]).
        **kwargs: Additional arguments passed to RewardDecoder.

    Returns:
        RewardDecoder instance.
    """
    if hidden_dims is None:
        hidden_dims = [256, 256]
    return RewardDecoder(
        state_dim=state_dim,
        d_z=d_z,
        hidden_dims=hidden_dims,
        **kwargs,
    )