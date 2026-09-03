"""
IQL Agent for FRE: Implicit Q-Learning agent conditioned on latent z.

Implements Q-function, Value function, and Policy networks that all take
state and latent encoding z as input. Uses IQL losses for offline RL training.

Reference: Kostrikov et al., 2021 (Implicit Q-Learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def _mlp(input_dim: int, hidden_dims: list, output_dim: int,
         activation: nn.Module = nn.ReLU, final_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """Build an MLP with specified hidden dimensions."""
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


def _init_weights(m: nn.Module):
    """Initialize weights using Kaiming uniform for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class QNetwork(nn.Module):
    """
    Q-function: Q(s, a, z) -> scalar.
    Input: concatenate(state, action, latent_z) -> MLP -> scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 hidden_dims: list = None, activation: nn.Module = nn.ReLU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + latent_dim
        self.net = _mlp(input_dim, hidden_dims, 1, activation)
        self.apply(_init_weights)

    def forward(self, states: torch.Tensor, actions: torch.Tensor,
                latent_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            latent_z: (batch_size, latent_dim) or (latent_dim,) for single z
        Returns:
            Q-values: (batch_size, 1)
        """
        # Handle broadcasting of latent_z
        if latent_z.dim() == 1:
            latent_z = latent_z.unsqueeze(0).expand(states.shape[0], -1)
        elif latent_z.shape[0] == 1 and states.shape[0] > 1:
            latent_z = latent_z.expand(states.shape[0], -1)

        x = torch.cat([states, actions, latent_z], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    Value function: V(s, z) -> scalar.
    Input: concatenate(state, latent_z) -> MLP -> scalar value.
    """

    def __init__(self, state_dim: int, latent_dim: int,
                 hidden_dims: list = None, activation: nn.Module = nn.ReLU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + latent_dim
        self.net = _mlp(input_dim, hidden_dims, 1, activation)
        self.apply(_init_weights)

    def forward(self, states: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim)
            latent_z: (batch_size, latent_dim) or (latent_dim,)
        Returns:
            V-values: (batch_size, 1)
        """
        if latent_z.dim() == 1:
            latent_z = latent_z.unsqueeze(0).expand(states.shape[0], -1)
        elif latent_z.shape[0] == 1 and states.shape[0] > 1:
            latent_z = latent_z.expand(states.shape[0], -1)

        x = torch.cat([states, latent_z], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Policy: π(a|s, z) -> Gaussian distribution over actions.
    Input: concatenate(state, latent_z) -> MLP -> mean and log_std.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 hidden_dims: list = None, activation: nn.Module = nn.ReLU,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        input_dim = state_dim + latent_dim
        self.net = _mlp(input_dim, hidden_dims, hidden_dims[-1], activation)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        self.apply(_init_weights)
        # Initialize log_std head with small values for stable start
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_head.bias, -1e-3, 1e-3)

    def forward(self, states: torch.Tensor, latent_z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (batch_size, state_dim)
            latent_z: (batch_size, latent_dim) or (latent_dim,)
        Returns:
            mean: (batch_size, action_dim)
            log_std: (batch_size, action_dim)
        """
        if latent_z.dim() == 1:
            latent_z = latent_z.unsqueeze(0).expand(states.shape[0], -1)
        elif latent_z.shape[0] == 1 and states.shape[0] > 1:
            latent_z = latent_z.expand(states.shape[0], -1)

        x = torch.cat([states, latent_z], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, states: torch.Tensor, latent_z: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        Returns:
            actions: (batch_size, action_dim)
            mean: (batch_size, action_dim)
            log_std: (batch_size, action_dim)
        """
        mean, log_std = self.forward(states, latent_z)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        actions = mean + eps * std
        return actions, mean, log_std

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor,
                 latent_z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions under the Gaussian policy.
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            latent_z: (batch_size, latent_dim) or (latent_dim,)
        Returns:
            log_probs: (batch_size, 1)
        """
        mean, log_std = self.forward(states, latent_z)
        std = torch.exp(log_std)
        var = std ** 2

        # Gaussian log probability
        log_probs = -0.5 * (
            ((actions - mean) ** 2) / (var + 1e-8) +
            2 * log_std +
            np.log(2 * np.pi)
        )
        return log_probs.sum(dim=-1, keepdim=True)


class IQLAgent(nn.Module):
    """
    Implicit Q-Learning agent conditioned on latent z.

    Contains Q-network, V-network, policy, and their target networks.
    Implements IQL losses: expectile regression for V, TD for Q, AWR for policy.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 hidden_dims: list = None,
                 expectile: float = 0.7,
                 temperature: float = 3.0,
                 discount: float = 0.99,
                 target_tau: float = 0.005,
                 log_std_min: float = -5.0,
                 log_std_max: float = 2.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.target_tau = target_tau

        # Q networks (two for clipped double Q-learning, though IQL typically uses one)
        self.q1 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims)

        # Target Q networks
        self.q1_target = QNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q2_target = QNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Value network and target
        self.v = ValueNetwork(state_dim, latent_dim, hidden_dims)
        self.v_target = ValueNetwork(state_dim, latent_dim, hidden_dims)
        self.v_target.load_state_dict(self.v.state_dict())

        # Policy
        self.policy = GaussianPolicy(state_dim, action_dim, latent_dim,
                                     hidden_dims, log_std_min=log_std_min,
                                     log_std_max=log_std_max)

    def _expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute expectile loss: L2_τ(u) = |τ - 1(u<0)| * u²
        Args:
            diff: (batch_size, 1) - difference (Q - V)
            expectile: τ parameter
        Returns:
            loss: scalar
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()

    def compute_value_loss(self, states: torch.Tensor, actions: torch.Tensor,
                           latent_z: torch.Tensor) -> torch.Tensor:
        """
        Value loss: L_V = E[ L2_τ(Q_target(s,a,z) - V(s,z)) ]
        Uses the minimum of the two target Q-values for stability.
        """
        with torch.no_grad():
            q1_target = self.q1_target(states, actions, latent_z)
            q2_target = self.q2_target(states, actions, latent_z)
            q_target = torch.min(q1_target, q2_target)

        v_pred = self.v(states, latent_z)
        diff = q_target - v_pred
        return self._expectile_loss(diff, self.expectile)

    def compute_q_loss(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
        """
        Q loss: L_Q = E[(r + γ * V_target(s',z) * (1-done) - Q(s,a,z))²]
        Computed for both Q networks.
        """
        with torch.no_grad():
            v_target_next = self.v_target(next_states, latent_z)
            target = rewards + self.discount * v_target_next * (1.0 - dones)

        q1_pred = self.q1(states, actions, latent_z)
        q2_pred = self.q2(states, actions, latent_z)

        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        return q1_loss + q2_loss

    def compute_policy_loss(self, states: torch.Tensor, actions: torch.Tensor,
                            latent_z: torch.Tensor) -> torch.Tensor:
        """
        Policy loss (AWR): L_π = E[ exp(β * (Q(s,a,z) - V(s,z))) * (-log π(a|s,z)) ]
        Uses advantage-weighted regression with clamped weights.
        """
        with torch.no_grad():
            q1 = self.q1(states, actions, latent_z)
            q2 = self.q2(states, actions, latent_z)
            q = torch.min(q1, q2)
            v = self.v(states, latent_z)
            advantage = q - v

            # Advantage-weighted regression weights
            # Clamp advantage for numerical stability
            adv_clamped = torch.clamp(advantage, -10.0, 10.0)
            weights = torch.exp(self.temperature * adv_clamped)
            # Clip weights to prevent extreme values
            weights = torch.clamp(weights, max=100.0)

        log_probs = self.policy.log_prob(states, actions, latent_z)
        # AWR minimizes negative log prob weighted by advantage
        policy_loss = -(weights * log_probs).mean()

        return policy_loss

    def update_targets(self):
        """Soft update target networks via Polyak averaging."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1 - self.target_tau) * target_param.data
            )
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1 - self.target_tau) * target_param.data
            )
        for target_param, param in zip(self.v_target.parameters(), self.v.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1 - self.target_tau) * target_param.data
            )

    def get_action(self, states: torch.Tensor, latent_z: torch.Tensor,
                   deterministic: bool = False) -> np.ndarray:
        """
        Get action for environment interaction.
        Args:
            states: (batch_size, state_dim) or (state_dim,)
            latent_z: (latent_dim,) or (batch_size, latent_dim)
            deterministic: if True, return mean action; else sample
        Returns:
            actions: numpy array (batch_size, action_dim)
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if deterministic:
            mean, _ = self.policy.forward(states, latent_z)
            actions = mean
        else:
            actions, _, _ = self.policy.sample(states, latent_z)

        return actions.detach().cpu().numpy()

    def forward(self, states: torch.Tensor, latent_z: torch.Tensor,
                deterministic: bool = False) -> np.ndarray:
        """Alias for get_action."""
        return self.get_action(states, latent_z, deterministic)