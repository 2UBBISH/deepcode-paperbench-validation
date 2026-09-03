"""
IQL (Implicit Q-Learning) with FRE conditioning.

Implements offline RL with Q, V, and policy networks conditioned on latent z.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class QNetwork(nn.Module):
    """Q-function network conditioned on latent z."""

    def __init__(self, state_dim, action_dim, latent_dim=128, hidden_dims=[512, 512, 512]):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim + latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states, actions, z):
        """
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            z: (batch_size, latent_dim)
        Returns:
            q_values: (batch_size,)
        """
        x = torch.cat([states, actions, z], dim=-1)
        return self.network(x).squeeze(-1)


class VNetwork(nn.Module):
    """Value function network conditioned on latent z."""

    def __init__(self, state_dim, latent_dim=128, hidden_dims=[512, 512, 512]):
        super().__init__()
        layers = []
        input_dim = state_dim + latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states, z):
        """
        Args:
            states: (batch_size, state_dim)
            z: (batch_size, latent_dim)
        Returns:
            values: (batch_size,)
        """
        x = torch.cat([states, z], dim=-1)
        return self.network(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Gaussian policy network conditioned on latent z."""

    def __init__(self, state_dim, action_dim, latent_dim=128,
                 hidden_dims=[512, 512, 512], log_std_min=-5.0):
        super().__init__()
        self.log_std_min = log_std_min

        layers = []
        input_dim = state_dim + latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(input_dim, action_dim)
        self.fc_logstd = nn.Linear(input_dim, action_dim)

    def forward(self, states, z):
        """
        Args:
            states: (batch_size, state_dim)
            z: (batch_size, latent_dim)
        Returns:
            mean: (batch_size, action_dim)
            logstd: (batch_size, action_dim)
        """
        x = torch.cat([states, z], dim=-1)
        features = self.backbone(x)
        mean = self.fc_mean(features)
        logstd = self.fc_logstd(features)
        logstd = torch.clamp(logstd, min=self.log_std_min)
        return mean, logstd

    def sample(self, states, z):
        """Sample action from policy."""
        mean, logstd = self.forward(states, z)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, states, actions, z):
        """Compute log probability of actions."""
        mean, logstd = self.forward(states, z)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(actions).sum(dim=-1)
        return log_prob


class FREIQL:
    """
    Implicit Q-Learning with FRE conditioning.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim=128,
                 hidden_dims=[512, 512, 512],
                 lr=1e-4,
                 gamma=0.88,
                 tau=0.001,
                 expectile=0.8,
                 temperature=3.0,
                 device='cuda'):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            latent_dim: Latent z dimension
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            tau: Target network update rate
            expectile: Expectile for value function
            temperature: Temperature for advantage-weighted regression
            device: Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.device = device

        # Q-networks (use two for stability)
        self.q1 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims).to(device)
        self.q2 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims).to(device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # Value network
        self.v = VNetwork(state_dim, latent_dim, hidden_dims).to(device)

        # Policy network
        self.policy = GaussianPolicy(state_dim, action_dim, latent_dim, hidden_dims).to(device)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def expectile_loss(self, diff, expectile):
        """Asymmetric L2 loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def update_v(self, states, actions, z):
        """Update value function using expectile regression."""
        with torch.no_grad():
            q1 = self.q1_target(states, actions, z)
            q2 = self.q2_target(states, actions, z)
            q = torch.min(q1, q2)

        v = self.v(states, z)
        loss = self.expectile_loss(q - v, self.expectile).mean()

        self.v_optimizer.zero_grad()
        loss.backward()
        self.v_optimizer.step()

        return loss.item()

    def update_q(self, states, actions, rewards, next_states, dones, z):
        """Update Q-functions using Bellman backup."""
        with torch.no_grad():
            next_v = self.v(next_states, z)
            target_q = rewards + self.gamma * (1 - dones) * next_v

        q1 = self.q1(states, actions, z)
        q2 = self.q2(states, actions, z)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q_loss.item()

    def update_policy(self, states, actions, z):
        """Update policy using advantage-weighted regression."""
        with torch.no_grad():
            v = self.v(states, z)
            q1 = self.q1_target(states, actions, z)
            q2 = self.q2_target(states, actions, z)
            q = torch.min(q1, q2)
            adv = q - v
            weights = torch.clamp(torch.exp(adv / self.temperature), max=100.0)

        log_prob = self.policy.log_prob(states, actions, z)
        loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

    def update_targets(self):
        """Soft update of target networks."""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_step(self, states, actions, rewards, next_states, dones, z):
        """
        Single training step.

        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            rewards: (batch_size,)
            next_states: (batch_size, state_dim)
            dones: (batch_size,)
            z: (batch_size, latent_dim)

        Returns:
            Dictionary of losses
        """
        v_loss = self.update_v(states, actions, z)
        q_loss = self.update_q(states, actions, rewards, next_states, dones, z)
        policy_loss = self.update_policy(states, actions, z)
        self.update_targets()

        return {
            'v_loss': v_loss,
            'q_loss': q_loss,
            'policy_loss': policy_loss
        }

    def select_action(self, state, z, deterministic=False):
        """Select action from policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            z = torch.FloatTensor(z).unsqueeze(0).to(self.device) if isinstance(z, np.ndarray) else z.unsqueeze(0)

            if deterministic:
                mean, _ = self.policy(state, z)
                action = mean
            else:
                action, _ = self.policy.sample(state, z)

            return action.cpu().numpy()[0]
