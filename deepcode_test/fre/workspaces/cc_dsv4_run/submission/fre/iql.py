"""
Implicit Q-Learning (IQL) with FRE conditioning.

Implements the IQL algorithm (Kostrikov et al., 2021) adapted for
latent-conditioned policies, Q-functions, and value functions.

The latent vector z is concatenated to the observation state for all
RL components (policy, Q-network, V-network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import copy


def mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list = None,
    activation=nn.ReLU,
    output_activation=None,
    layer_norm: bool = False,
):
    """Build a configurable MLP."""
    if hidden_dims is None:
        hidden_dims = [512, 512, 512]

    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        if layer_norm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """
    Latent-conditioned Gaussian policy π(a | s, z).

    Outputs mean and log-standard-deviation for each action dimension.
    Following the paper: 3 hidden layers of size 512 with ReLU activations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim + latent_dim
        self.net = mlp(input_dim, hidden_dims[-1], hidden_dims[:-1], nn.ReLU, nn.ReLU)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, z], dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, z)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        action = mean + eps * std
        # Log probability of the sampled action
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-6)).pow(2) + 2 * log_std + math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def get_action(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean) for evaluation."""
        mean, _ = self.forward(state, z)
        return mean


import math


class LatentQFunction(nn.Module):
    """
    Q-function Q(s, a, z) conditioned on latent encoding z.
    z is concatenated to [s, a].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim + action_dim + latent_dim
        self.net = mlp(input_dim, 1, hidden_dims, nn.ReLU)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x)


class LatentValueFunction(nn.Module):
    """
    Value function V(s, z) conditioned on latent encoding z.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim + latent_dim
        self.net = mlp(input_dim, 1, hidden_dims, nn.ReLU)

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, z], dim=-1)
        return self.net(x)


class FREIQLAgent(nn.Module):
    """
    Full FRE-conditioned IQL agent.

    Components:
    - Q-function (critic): Q(s, a, z)
    - Value function: V(s, z)
    - Policy: π(a | s, z)
    - Target Q-function and target V-function for stable TD learning.

    IQL training objectives:
    - V: expectile regression on Q with state-value targets
    - Q: standard Bellman error with V as target
    - π: advantage-weighted regression (AWR)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
        expectile: float = 0.8,
        temperature: float = 3.0,
        discount: float = 0.88,
        target_update_rate: float = 0.001,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.qf = LatentQFunction(state_dim, action_dim, latent_dim, hidden_dims)
        self.vf = LatentValueFunction(state_dim, latent_dim, hidden_dims)
        self.policy = GaussianPolicy(state_dim, action_dim, latent_dim, hidden_dims)

        # Target networks
        self.target_qf = copy.deepcopy(self.qf)
        self.target_vf = copy.deepcopy(self.vf)

        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.latent_dim = latent_dim

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric L2 loss for expectile regression."""
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return weight * (diff ** 2)

    def update_targets(self):
        """Polyak-averaged target updates."""
        for param, target_param in zip(self.qf.parameters(), self.target_qf.parameters()):
            target_param.data.copy_(
                self.target_update_rate * param.data
                + (1 - self.target_update_rate) * target_param.data
            )
        for param, target_param in zip(self.vf.parameters(), self.target_vf.parameters()):
            target_param.data.copy_(
                self.target_update_rate * param.data
                + (1 - self.target_update_rate) * target_param.data
            )

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
    ) -> dict:
        """
        One training step of IQL with FRE conditioning.

        Args:
            states:      (B, state_dim)
            actions:     (B, action_dim)
            rewards:     (B, 1)
            next_states: (B, state_dim)
            dones:       (B, 1)
            z:           (B, latent_dim)
        Returns:
            Dictionary of losses.
        """
        # --- Value function update ---
        with torch.no_grad():
            target_q1 = self.target_qf(states, actions, z)
            target_q2 = self.target_qf(states, actions, z)  # second Q for double Q
            # For IQL with single Q, use the same

        q_val = self.qf(states, actions, z)
        v_val = self.vf(states, z)

        # Value loss: expectile regression of V toward Q
        value_loss = self._expectile_loss(q_val.detach() - v_val).mean()

        # --- Q-function update ---
        with torch.no_grad():
            next_v = self.target_vf(next_states, z)
            target = rewards + self.discount * (1 - dones) * next_v

        q_loss = F.mse_loss(q_val, target)

        # --- Policy update (AWR) ---
        with torch.no_grad():
            # Advantage = Q(s,a,z) - V(s,z)
            adv = q_val.detach() - v_val.detach()
            # Advantage-weighted regression weight
            exp_adv = torch.exp(adv / self.temperature)
            exp_adv = torch.clamp(exp_adv, max=100.0)

        _, log_prob, _ = self.policy.sample(states, z)
        policy_loss = -(exp_adv * log_prob).mean()

        # Compute combined Q+V loss
        critic_loss = value_loss + q_loss

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
        }