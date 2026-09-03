"""
FRE Decoder: MLP for predicting rewards given state and latent z.
"""

import torch
import torch.nn as nn


class FREDecoder(nn.Module):
    """
    Functional Reward Decoder.

    Predicts reward for a state given latent encoding z.
    """

    def __init__(self, state_dim, latent_dim=128, hidden_dims=[512, 512, 512]):
        """
        Args:
            state_dim: Dimension of state space
            latent_dim: Dimension of latent embedding z
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Build MLP
        # Input: concatenation of state and z
        layers = []
        input_dim = state_dim + latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, states, z):
        """
        Predict rewards.

        Args:
            states: (batch_size, state_dim) or (batch_size, K', state_dim)
            z: (batch_size, latent_dim)

        Returns:
            rewards: (batch_size,) or (batch_size, K') predicted rewards
        """
        # Handle both 2D and 3D state inputs
        if states.ndim == 3:
            batch_size, K_prime, _ = states.shape
            # Expand z to match state dimensions
            z_expanded = z.unsqueeze(1).expand(batch_size, K_prime, -1)
            # Concatenate
            inputs = torch.cat([states, z_expanded], dim=-1)
            # Forward pass
            rewards = self.network(inputs).squeeze(-1)  # (batch_size, K')
        else:
            # Concatenate state and z
            inputs = torch.cat([states, z], dim=-1)
            # Forward pass
            rewards = self.network(inputs).squeeze(-1)  # (batch_size,)

        return rewards

    def compute_loss(self, states, z, true_rewards):
        """
        Compute MSE loss between predicted and true rewards.

        Args:
            states: (batch_size, K', state_dim)
            z: (batch_size, latent_dim)
            true_rewards: (batch_size, K')

        Returns:
            loss: scalar MSE loss
        """
        pred_rewards = self.forward(states, z)
        loss = nn.functional.mse_loss(pred_rewards, true_rewards)
        return loss
