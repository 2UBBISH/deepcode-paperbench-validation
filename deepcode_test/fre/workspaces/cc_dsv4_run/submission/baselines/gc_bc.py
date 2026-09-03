"""
Goal-Conditioned Behavioral Cloning (GC-BC) baseline.

Implements the GC-BC baseline from the paper:
- MLP with 3 hidden layers of size 512
- ReLU activations with layer normalization before each activation
- Outputs Gaussian distribution (mean + log std)
- Trained via maximum likelihood on dataset actions
- Hindsight relabeling: goals sampled from future states using geometric distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple
import copy


class GCBCConditionalPolicy(nn.Module):
    """
    Goal-conditioned Gaussian policy for GC-BC.

    Input: [state, goal] concatenated
    Architecture: 3 × 512 hidden layers, ReLU + LayerNorm
    Output: Gaussian (μ, log σ)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = None,
        log_std_min: float = -5.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim * 2  # state + goal concatenated

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.LayerNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        self.log_std_min = log_std_min

    def forward(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, goal], dim=-1)
        x = self.backbone(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=self.log_std_min)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, goal)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        action = mean + eps * std
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-6)).pow(2) + 2 * log_std + math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def get_action(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean) for evaluation."""
        mean, _ = self.forward(state, goal)
        return mean


class GoalConditionedBC:
    """
    GC-BC training loop.

    Uses hindsight relabeling with geometric sampling for goals
    from future states within the trajectory.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = None,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.policy = GCBCConditionalPolicy(
            state_dim, action_dim, hidden_dims
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = torch.device(device)
        self.state_dim = state_dim

    def sample_goals_geometric(
        self,
        trajectory_states: list,
        current_indices: torch.Tensor,
        p_geometric: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample goals from future states in the same trajectory using
        geometric distribution.

        Per addendum: GC-BC uses ONLY geometric sampling from future states
        (no random goals or current-state goals).

        Args:
            trajectory_states: list of tensors, one per trajectory
            current_indices: which state index within each trajectory
            p_geometric: parameter for geometric distribution
        Returns:
            goals: (batch_size, state_dim)
        """
        batch_size = len(current_indices)
        goals = []

        for i in range(batch_size):
            traj = trajectory_states[i % len(trajectory_states)]
            t = current_indices[i].item()
            T = len(traj)

            if t >= T - 1:
                goals.append(traj[t])
            else:
                # Geometric distribution over future states
                future_steps = T - t - 1
                probs = np.array([
                    (1 - p_geometric) ** k * p_geometric
                    for k in range(future_steps)
                ])
                probs = probs / probs.sum()
                offset = np.random.choice(future_steps, p=probs)
                goals.append(traj[t + 1 + offset])

        return torch.stack(goals).to(self.device)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
    ) -> float:
        """
        Single training step of GC-BC.

        Loss: negative log-likelihood of dataset actions under the
        predicted Gaussian distribution.

        L_π = -E_{(s,g,a)~D} log π(a | s, g)
        """
        _, log_prob, _ = self.policy.sample(states, goals)
        loss = -log_prob.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Get action for evaluation — goal is the ground-truth task goal."""
        return self.policy.get_action(
            state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device),
            goal.unsqueeze(0).to(self.device) if goal.dim() == 1 else goal.to(self.device),
        ).squeeze(0)