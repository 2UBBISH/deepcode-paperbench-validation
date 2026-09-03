"""
FRE Decoder: Predicts reward η(s) for a state s given latent encoding z.

Architecture:
    Feedforward neural network:
        input = concat(s, z) -> Linear -> ReLU -> Linear -> ReLU -> Linear -> scalar output.
    Hidden layers: [256, 256] (configurable).

The decoder is trained jointly with the encoder to minimize MSE between
predicted and true rewards on decoding states.

Reference: "Functional Reward Encodings (FRE) for Zero-Shot Offline RL"
"""

from typing import Optional
import torch
import torch.nn as nn


class FREDecoder(nn.Module):
    """
    Feedforward decoder that predicts scalar reward η(s) given state s and latent z.

    Architecture:
        concat(s, z) -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(1)

    Args:
        state_dim: Dimensionality of state vectors.
        latent_dim: Dimensionality of latent encoding z.
        hidden_dims: List of hidden layer sizes (default: [256, 256]).
        activation: Activation function (default: ReLU).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.hidden_dims = hidden_dims

        # Build MLP layers
        layers = []
        input_dim = state_dim + latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim

        # Output layer: scalar reward
        layers.append(nn.Linear(input_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Kaiming uniform (good for ReLU)."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for each state given latent encoding z.

        Args:
            states: Tensor of shape (batch_size, state_dim) or (K', state_dim).
            z: Tensor of shape (batch_size, latent_dim) or (latent_dim,).
               If z is 1D, it is broadcast to match states batch dimension.

        Returns:
            Predicted rewards of shape (batch_size, 1) or (K', 1).
        """
        # Handle broadcasting: if z is 1D, expand to match batch
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(states.size(0), -1)
        elif z.dim() == 2 and z.size(0) == 1 and states.size(0) > 1:
            z = z.expand(states.size(0), -1)

        # Concatenate state and latent
        x = torch.cat([states, z], dim=-1)

        # Forward through MLP
        reward_pred = self.net(x)

        return reward_pred

    def compute_loss(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
        true_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE reconstruction loss between predicted and true rewards.

        L_decoder = (1/K') Σ_{k} (η(s_k^d) - qθ(s_k^d, z))²

        Args:
            states: Decoding states of shape (K', state_dim).
            z: Latent encoding of shape (latent_dim,) or (1, latent_dim).
            true_rewards: True rewards η(s_k^d) of shape (K',) or (K', 1).

        Returns:
            Scalar MSE loss.
        """
        pred_rewards = self.forward(states, z)
        # Ensure shapes match
        if true_rewards.dim() == 1:
            true_rewards = true_rewards.unsqueeze(-1)
        loss = nn.functional.mse_loss(pred_rewards, true_rewards)
        return loss

    def get_reward_range_estimate(
        self,
        states: torch.Tensor,
        z_samples: torch.Tensor,
    ) -> tuple:
        """
        Estimate the range of predicted rewards for a set of states and latent samples.
        Useful for adaptive reward binning.

        Args:
            states: States of shape (N, state_dim).
            z_samples: Multiple latent samples of shape (M, latent_dim).

        Returns:
            (min_reward, max_reward) tuple of floats.
        """
        with torch.no_grad():
            all_preds = []
            for i in range(z_samples.size(0)):
                z = z_samples[i]
                preds = self.forward(states, z)
                all_preds.append(preds)
            all_preds = torch.cat(all_preds, dim=0)
            return all_preds.min().item(), all_preds.max().item()

    def __repr__(self) -> str:
        return (
            f"FREDecoder(state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )