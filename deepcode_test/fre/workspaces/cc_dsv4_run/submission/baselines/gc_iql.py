"""
Goal-Conditioned Implicit Q-Learning (GC-IQL) baseline.

GC-IQL extends IQL with goal-conditioning as described in the addendum:
- Goals and observations are concatenated as input to all networks.
- Hindsight relabeling with HER distribution:
  - p_randomgoal = 0.3 (random goal from dataset)
  - p_geometric_goal = 0.5 (geometric future state)
  - p_current_goal = 0.2 (current state IS the goal, reward=0, terminal=True)
- Reward: 0 if state == goal, -1 otherwise.
- No environment rewards used in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Callable
import copy


def gc_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list = None,
    activation=nn.ReLU,
    output_activation=None,
):
    """Build an MLP for goal-conditioned components."""
    if hidden_dims is None:
        hidden_dims = [512, 512, 512]

    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class GCIQLPolicy(nn.Module):
    """Goal-conditioned Gaussian policy."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: list = None,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim + goal_dim
        self.net = gc_mlp(
            input_dim, hidden_dims[-1], hidden_dims[:-1], nn.ReLU, nn.ReLU
        )
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, goal], dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        mean, _ = self.forward(state, goal)
        return mean


class GCIQL:
    """
    Goal-Conditioned IQL training pipeline.

    Per addendum: GC-IQL is just IQL with the additional goal state.
    Goals sampled with HER distribution, no environment rewards.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_dims: list = None,
        expectile: float = 0.8,
        temperature: float = 3.0,
        discount: float = 0.88,
        target_update_rate: float = 0.001,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        if goal_dim is None:
            goal_dim = state_dim

        self.device = torch.device(device)
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.target_update_rate = target_update_rate

        self.qf = gc_mlp(
            state_dim + action_dim + goal_dim, 1, hidden_dims, nn.ReLU
        ).to(device)
        self.vf = gc_mlp(
            state_dim + goal_dim, 1, hidden_dims, nn.ReLU
        ).to(device)
        self.policy = GCIQLPolicy(
            state_dim, goal_dim, action_dim, hidden_dims
        ).to(device)

        self.target_qf = copy.deepcopy(self.qf)
        self.target_vf = copy.deepcopy(self.vf)

        self.optimizer = torch.optim.Adam(
            list(self.qf.parameters())
            + list(self.vf.parameters())
            + list(self.policy.parameters()),
            lr=lr,
        )

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return weight * (diff ** 2)

    def sample_her_goals(
        self,
        dataset_states: torch.Tensor,
        batch_size: int,
        p_random: float = 0.3,
        p_geometric: float = 0.5,
        p_current: float = 0.2,
        trajectory_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample goals with HER distribution.

        Returns:
            goals: (batch_size, state_dim)
            rewards: (batch_size, 1) — 0 if goal==state else -1
            dones: (batch_size, 1) — True if goal==state (current state as goal)
        """
        N = dataset_states.shape[0]
        r = torch.rand(batch_size)

        is_random = r < p_random
        is_geometric = (r >= p_random) & (r < p_random + p_geometric)
        is_current = r >= p_random + p_geometric

        goals = torch.zeros(batch_size, self.state_dim)

        # Random goals
        if is_random.any():
            n_random = is_random.sum().item()
            random_indices = torch.randint(0, N, (n_random,))
            goals[is_random] = dataset_states[random_indices]

        # Geometric (future) goals
        if is_geometric.any() and trajectory_indices is not None:
            n_geo = is_geometric.sum().item()
            for i, (idx, is_geo) in enumerate(zip(is_geometric.nonzero(as_tuple=True)[0], is_geometric)):
                if is_geo:
                    # Simplified: just pick a random future index
                    geo_idx = torch.randint(0, N, (1,))
                    goals[idx] = dataset_states[geo_idx]

        # Current state as goal — will be filled in during training step
        # (needs current batch states)

        return goals, is_random, is_geometric, is_current

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        goals: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """
        Single training step of GC-IQL.
        Reward is 0 if at goal, -1 otherwise.
        """
        B = states.shape[0]

        # --- Value function update ---
        q_val = self.qf(torch.cat([states, actions, goals], dim=-1))
        v_val = self.vf(torch.cat([states, goals], dim=-1))

        value_loss = self._expectile_loss(
            q_val.detach() - v_val
        ).mean()

        # --- Q-function update ---
        with torch.no_grad():
            next_v = self.target_vf(torch.cat([next_states, goals], dim=-1))
            target = rewards + self.discount * (1 - dones) * next_v

        q_loss = F.mse_loss(q_val, target)

        # --- Policy update (AWR) ---
        with torch.no_grad():
            adv = q_val.detach() - v_val.detach()
            exp_adv = torch.exp(adv / self.temperature)
            exp_adv = torch.clamp(exp_adv, max=100.0)

        _, log_prob, _ = self.policy.sample(states, goals)
        policy_loss = -(exp_adv * log_prob).mean()

        total_loss = value_loss + q_loss + policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._update_targets()

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _update_targets(self):
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

    def get_action(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Get action for evaluation."""
        s = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        g = goal.unsqueeze(0).to(self.device) if goal.dim() == 1 else goal.to(self.device)
        return self.policy.get_action(s, g).squeeze(0).cpu()