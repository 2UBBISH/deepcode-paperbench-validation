"""
Goal-Conditioned IQL (GC-IQL).

IQL with hindsight relabeling for goal-reaching tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class GCQNetwork(nn.Module):
    """Q-function conditioned on goal."""

    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 512]):
        super().__init__()
        layers = []
        # Input: state + goal + action
        input_dim = state_dim * 2 + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states, actions, goals):
        x = torch.cat([states, actions, goals], dim=-1)
        return self.network(x).squeeze(-1)


class GCVNetwork(nn.Module):
    """Value function conditioned on goal."""

    def __init__(self, state_dim, hidden_dims=[512, 512, 512]):
        super().__init__()
        layers = []
        # Input: state + goal
        input_dim = state_dim * 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states, goals):
        x = torch.cat([states, goals], dim=-1)
        return self.network(x).squeeze(-1)


class GCPolicy(nn.Module):
    """Policy conditioned on goal."""

    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 512], log_std_min=-5.0):
        super().__init__()
        self.log_std_min = log_std_min

        layers = []
        # Input: state + goal
        input_dim = state_dim * 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(input_dim, action_dim)
        self.fc_logstd = nn.Linear(input_dim, action_dim)

    def forward(self, states, goals):
        x = torch.cat([states, goals], dim=-1)
        features = self.backbone(x)
        mean = self.fc_mean(features)
        logstd = self.fc_logstd(features)
        logstd = torch.clamp(logstd, min=self.log_std_min)
        return mean, logstd

    def sample(self, states, goals):
        mean, logstd = self.forward(states, goals)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, states, actions, goals):
        mean, logstd = self.forward(states, goals)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(actions).sum(dim=-1)
        return log_prob


class GCIQL:
    """
    Goal-Conditioned Implicit Q-Learning.

    Trains a goal-conditioned policy using hindsight relabeling.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims=[512, 512, 512],
                 lr=1e-4,
                 gamma=0.88,
                 tau=0.001,
                 expectile=0.8,
                 temperature=3.0,
                 device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.device = device

        # Networks
        self.q1 = GCQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q2 = GCQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        self.v = GCVNetwork(state_dim, hidden_dims).to(device)
        self.policy = GCPolicy(state_dim, action_dim, hidden_dims).to(device)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def sample_goal(self, trajectory_idx, step_idx, dataset, p_future=0.5, p_random=0.3, p_current=0.2):
        """
        Sample goal using hindsight relabeling.

        Args:
            trajectory_idx: Index of current trajectory
            step_idx: Current step in trajectory
            dataset: Dataset containing trajectories
            p_future: Probability of sampling future state
            p_random: Probability of sampling random state
            p_current: Probability of using current state

        Returns:
            goal: Sampled goal state
        """
        p = np.random.rand()

        if p < p_current:
            # Current state is goal
            return dataset['states'][trajectory_idx][step_idx]
        elif p < p_current + p_future:
            # Future state in trajectory (geometric sampling)
            traj_length = len(dataset['states'][trajectory_idx])
            if step_idx < traj_length - 1:
                # Sample geometrically from future states
                future_steps = traj_length - step_idx - 1
                # Geometric distribution
                idx_offset = np.random.geometric(p=0.02)
                future_idx = min(step_idx + idx_offset, traj_length - 1)
                return dataset['states'][trajectory_idx][future_idx]
            else:
                # At end of trajectory, sample random
                random_traj = np.random.randint(len(dataset['states']))
                random_step = np.random.randint(len(dataset['states'][random_traj]))
                return dataset['states'][random_traj][random_step]
        else:
            # Random state from dataset
            random_traj = np.random.randint(len(dataset['states']))
            random_step = np.random.randint(len(dataset['states'][random_traj]))
            return dataset['states'][random_traj][random_step]

    def compute_goal_reward(self, state, goal, threshold=0.1):
        """
        Compute goal-conditioned reward.
        0 if within threshold of goal, -1 otherwise.
        """
        distance = np.linalg.norm(state - goal)
        return 0.0 if distance < threshold else -1.0

    def expectile_loss(self, diff, expectile):
        """Asymmetric L2 loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def train_step(self, states, actions, next_states, goals, rewards, dones):
        """Training step for GC-IQL."""
        # Update V
        with torch.no_grad():
            q1 = self.q1_target(states, actions, goals)
            q2 = self.q2_target(states, actions, goals)
            q = torch.min(q1, q2)

        v = self.v(states, goals)
        v_loss = self.expectile_loss(q - v, self.expectile).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q
        with torch.no_grad():
            next_v = self.v(next_states, goals)
            target_q = rewards + self.gamma * (1 - dones) * next_v

        q1 = self.q1(states, actions, goals)
        q2 = self.q2(states, actions, goals)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        with torch.no_grad():
            v_val = self.v(states, goals)
            q1_val = self.q1_target(states, actions, goals)
            q2_val = self.q2_target(states, actions, goals)
            q_val = torch.min(q1_val, q2_val)
            adv = q_val - v_val
            weights = torch.clamp(torch.exp(adv / self.temperature), max=100.0)

        log_prob = self.policy.log_prob(states, actions, goals)
        policy_loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item()
        }

    def select_action(self, state, goal, deterministic=False):
        """Select action given state and goal."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)

            if deterministic:
                mean, _ = self.policy(state, goal)
                action = mean
            else:
                action, _ = self.policy.sample(state, goal)

            return action.cpu().numpy()[0]
