"""
Goal-Conditioned Behavioral Cloning (GC-BC).

Simple offline RL method that learns goal-reaching policies
by mimicking trajectories with hindsight relabeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCBCPolicy(nn.Module):
    """
    Goal-conditioned policy for behavioral cloning.

    Outputs Gaussian distribution over actions.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 512], log_std_min=-5.0):
        super().__init__()
        self.log_std_min = log_std_min

        layers = []
        # Input: state + goal
        input_dim = state_dim * 2

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Layer normalization
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output: mean and log_std for Gaussian distribution
        self.fc_mean = nn.Linear(input_dim, action_dim)
        self.fc_logstd = nn.Linear(input_dim, action_dim)

    def forward(self, states, goals):
        """
        Args:
            states: (batch_size, state_dim)
            goals: (batch_size, state_dim)

        Returns:
            mean: (batch_size, action_dim)
            logstd: (batch_size, action_dim)
        """
        x = torch.cat([states, goals], dim=-1)
        features = self.backbone(x)

        mean = self.fc_mean(features)
        logstd = self.fc_logstd(features)
        logstd = torch.clamp(logstd, min=self.log_std_min)

        return mean, logstd

    def sample(self, states, goals):
        """Sample action from policy."""
        mean, logstd = self.forward(states, goals)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        return action

    def log_prob(self, states, actions, goals):
        """Compute log probability of actions."""
        mean, logstd = self.forward(states, goals)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(actions).sum(dim=-1)
        return log_prob


class GCBC:
    """
    Goal-Conditioned Behavioral Cloning.

    Trains a policy using maximum likelihood estimation with hindsight relabeling.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims=[512, 512, 512],
                 lr=1e-4,
                 device='cuda'):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            device: Device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Policy network
        self.policy = GCBCPolicy(state_dim, action_dim, hidden_dims).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def sample_goal_geometric(self, trajectory_states, current_idx):
        """
        Sample goal from future states in trajectory using geometric distribution.

        Args:
            trajectory_states: States in the trajectory
            current_idx: Current step index

        Returns:
            goal: Sampled goal state
        """
        traj_length = len(trajectory_states)

        if current_idx >= traj_length - 1:
            # At end of trajectory, use current state
            return trajectory_states[current_idx]

        # Sample from future states using geometric distribution
        future_steps = traj_length - current_idx - 1
        # Geometric sampling with p=0.02 (decay parameter)
        idx_offset = np.random.geometric(p=0.02)
        future_idx = min(current_idx + idx_offset, traj_length - 1)

        return trajectory_states[future_idx]

    def train_step(self, states, actions, goals):
        """
        Training step using maximum likelihood estimation.

        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            goals: (batch_size, state_dim)

        Returns:
            Loss value
        """
        # Compute log probability of actions
        log_prob = self.policy.log_prob(states, actions, goals)

        # Maximum likelihood loss (negative log probability)
        loss = -log_prob.mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def select_action(self, state, goal, deterministic=False):
        """
        Select action given state and goal.

        Args:
            state: Current state
            goal: Goal state
            deterministic: If True, return mean action

        Returns:
            action: Selected action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)

            if deterministic:
                mean, _ = self.policy(state, goal)
                action = mean
            else:
                action = self.policy.sample(state, goal)

            return action.cpu().numpy()[0]
