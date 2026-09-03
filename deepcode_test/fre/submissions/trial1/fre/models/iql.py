"""
IQL Networks: Q-function, Value function, and Policy, all conditioned on latent z.

All networks take latent vector z as additional input (concatenated to state or state-action)
to enable zero-shot generalization to novel reward functions.

Architecture follows the IQL paper (Kostrikov et al., 2021) with z-conditioning added:
- Q(s, a, z): 3-layer MLP, 256 hidden units, ReLU
- V(s, z): 3-layer MLP, 256 hidden units, ReLU
- pi(a|s, z): 3-layer MLP, 256 hidden units, ReLU, Gaussian policy with tanh squashing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list,
    activation: str = "relu",
    dropout: float = 0.0,
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a simple MLP with configurable activation and dropout."""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """
    Q-function: Q(s, a, z) -> scalar.
    
    Input: concatenation of state, action, and latent z.
    Architecture: 3 hidden layers, 256 units each, ReLU activation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_latent = d_latent
        self.input_dim = state_dim + action_dim + d_latent
        
        self.net = _build_mlp(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values.
        
        Args:
            states: (batch_size, state_dim) or (batch_size, K, state_dim)
            actions: (batch_size, action_dim) or (batch_size, K, action_dim)
            z: (batch_size, d_latent) or (batch_size, K, d_latent)
        
        Returns:
            Q-values: (batch_size,) or (batch_size, K)
        """
        # Handle multi-state case (K states per z)
        if states.dim() == 3:
            # states: (B, K, state_dim), actions: (B, K, action_dim), z: (B, K, d_latent)
            B, K, _ = states.shape
            # Concatenate along last dimension
            x = torch.cat([states, actions, z], dim=-1)  # (B, K, input_dim)
            # Reshape to (B*K, input_dim) for MLP
            x = x.reshape(B * K, self.input_dim)
            q = self.net(x)  # (B*K, 1)
            q = q.reshape(B, K)  # (B, K)
        elif states.dim() == 2:
            # Standard batch: (B, state_dim)
            # z may be (B, d_latent) - expand if needed
            if z.dim() == 2 and z.shape[0] == states.shape[0]:
                x = torch.cat([states, actions, z], dim=-1)
            else:
                raise ValueError(f"Shape mismatch: states {states.shape}, actions {actions.shape}, z {z.shape}")
            q = self.net(x).squeeze(-1)  # (B,)
        else:
            raise ValueError(f"Expected 2D or 3D states, got shape {states.shape}")
        
        return q


class ValueNetwork(nn.Module):
    """
    Value function: V(s, z) -> scalar.
    
    Input: concatenation of state and latent z.
    Architecture: 3 hidden layers, 256 units each, ReLU activation.
    """
    
    def __init__(
        self,
        state_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        self.state_dim = state_dim
        self.d_latent = d_latent
        self.input_dim = state_dim + d_latent
        
        self.net = _build_mlp(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute V-values.
        
        Args:
            states: (batch_size, state_dim) or (batch_size, K, state_dim)
            z: (batch_size, d_latent) or (batch_size, K, d_latent)
        
        Returns:
            V-values: (batch_size,) or (batch_size, K)
        """
        if states.dim() == 3:
            B, K, _ = states.shape
            x = torch.cat([states, z], dim=-1)  # (B, K, input_dim)
            x = x.reshape(B * K, self.input_dim)
            v = self.net(x)  # (B*K, 1)
            v = v.reshape(B, K)  # (B, K)
        elif states.dim() == 2:
            x = torch.cat([states, z], dim=-1)
            v = self.net(x).squeeze(-1)  # (B,)
        else:
            raise ValueError(f"Expected 2D or 3D states, got shape {states.shape}")
        
        return v


class GaussianPolicy(nn.Module):
    """
    Gaussian policy: pi(a|s, z).
    
    Input: concatenation of state and latent z.
    Output: mean and log_std of Gaussian distribution over actions.
    Actions are squashed through tanh to be in [-1, 1].
    
    Architecture: 3 hidden layers, 256 units each, ReLU activation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_latent = d_latent
        self.input_dim = state_dim + d_latent
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared trunk
        self.trunk = _build_mlp(
            input_dim=self.input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            dropout=dropout,
        )
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(
        self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            states: (batch_size, state_dim)
            z: (batch_size, d_latent)
            deterministic: if True, return mean action (no noise)
        
        Returns:
            actions: (batch_size, action_dim) in [-1, 1]
            mean: (batch_size, action_dim) pre-tanh mean
            log_std: (batch_size, action_dim) log standard deviation
        """
        if states.dim() == 3:
            # Handle multi-state case
            B, K, _ = states.shape
            x = torch.cat([states, z], dim=-1)  # (B, K, input_dim)
            x = x.reshape(B * K, self.input_dim)
            h = self.trunk(x)  # (B*K, hidden_dim)
            mean = self.mean_head(h)  # (B*K, action_dim)
            log_std = self.log_std_head(h)  # (B*K, action_dim)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            if deterministic:
                actions = torch.tanh(mean)
            else:
                std = torch.exp(log_std)
                noise = torch.randn_like(mean)
                actions = torch.tanh(mean + noise * std)
            
            # Reshape back
            mean = mean.reshape(B, K, self.action_dim)
            log_std = log_std.reshape(B, K, self.action_dim)
            actions = actions.reshape(B, K, self.action_dim)
        else:
            x = torch.cat([states, z], dim=-1)
            h = self.trunk(x)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            if deterministic:
                actions = torch.tanh(mean)
            else:
                std = torch.exp(log_std)
                noise = torch.randn_like(mean)
                actions = torch.tanh(mean + noise * std)
        
        return actions, mean, log_std
    
    def get_action(self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Convenience method to get only actions."""
        actions, _, _ = self.forward(states, z, deterministic=deterministic)
        return actions
    
    def log_prob(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions under the Gaussian policy (with tanh correction).
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim) in [-1, 1]
            z: (batch_size, d_latent)
        
        Returns:
            log_prob: (batch_size,) log probability
        """
        if states.dim() == 3:
            B, K, _ = states.shape
            x = torch.cat([states, z], dim=-1).reshape(B * K, self.input_dim)
            h = self.trunk(x)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
            # Reshape actions
            actions_flat = actions.reshape(B * K, self.action_dim)
            
            # Pre-tanh action (inverse tanh)
            # Clamp actions to avoid numerical issues with atanh
            actions_clamped = torch.clamp(actions_flat, -0.999, 0.999)
            pre_tanh = torch.atanh(actions_clamped)
            
            # Gaussian log prob
            log_prob_gaussian = -0.5 * (
                ((pre_tanh - mean) / std) ** 2
                + 2 * log_std
                + np.log(2 * np.pi)
            )
            log_prob_gaussian = log_prob_gaussian.sum(dim=-1)  # (B*K,)
            
            # Tanh correction: log(1 - tanh^2(x))
            log_prob_tanh_correction = torch.log(
                1.0 - actions_clamped ** 2 + 1e-6
            ).sum(dim=-1)  # (B*K,)
            
            log_prob = log_prob_gaussian - log_prob_tanh_correction
            log_prob = log_prob.reshape(B, K)  # (B, K)
        else:
            x = torch.cat([states, z], dim=-1)
            h = self.trunk(x)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
            # Pre-tanh action
            actions_clamped = torch.clamp(actions, -0.999, 0.999)
            pre_tanh = torch.atanh(actions_clamped)
            
            log_prob_gaussian = -0.5 * (
                ((pre_tanh - mean) / std) ** 2
                + 2 * log_std
                + np.log(2 * np.pi)
            )
            log_prob_gaussian = log_prob_gaussian.sum(dim=-1)  # (B,)
            
            log_prob_tanh_correction = torch.log(
                1.0 - actions_clamped ** 2 + 1e-6
            ).sum(dim=-1)  # (B,)
            
            log_prob = log_prob_gaussian - log_prob_tanh_correction
        
        return log_prob


class IQLNetworks(nn.Module):
    """
    Container for all IQL networks: two Q-networks, one V-network, one policy,
    and their target networks.
    
    Provides convenience methods for soft target updates and saving/loading.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        # Q-networks (double Q-learning)
        self.q1 = QNetwork(state_dim, action_dim, d_latent, hidden_dims, activation, dropout)
        self.q2 = QNetwork(state_dim, action_dim, d_latent, hidden_dims, activation, dropout)
        
        # Target Q-networks
        self.q1_target = QNetwork(state_dim, action_dim, d_latent, hidden_dims, activation, dropout)
        self.q2_target = QNetwork(state_dim, action_dim, d_latent, hidden_dims, activation, dropout)
        
        # Value network
        self.v = ValueNetwork(state_dim, d_latent, hidden_dims, activation, dropout)
        
        # Policy
        self.policy = GaussianPolicy(state_dim, action_dim, d_latent, hidden_dims, activation, dropout, log_std_min, log_std_max)
        
        # Initialize target networks to match online networks
        self._hard_update_targets()
    
    def _hard_update_targets(self):
        """Copy online network parameters to target networks."""
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
    
    def soft_update_targets(self, tau: float = 0.005):
        """
        Soft update target networks: target = tau * online + (1 - tau) * target.
        """
        with torch.no_grad():
            for target_param, online_param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            for target_param, online_param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def get_q_target(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Get minimum of two target Q-values (for value and policy updates)."""
        q1_target = self.q1_target(states, actions, z)
        q2_target = self.q2_target(states, actions, z)
        return torch.min(q1_target, q2_target)
    
    def get_q(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both Q-values."""
        return self.q1(states, actions, z), self.q2(states, actions, z)
    
    def get_v(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Get V-value."""
        return self.v(states, z)
    
    def get_action(self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from policy."""
        return self.policy.get_action(states, z, deterministic=deterministic)
    
    def get_trainable_parameters(self):
        """Return parameters of online networks (excluding targets)."""
        for p in self.q1.parameters():
            yield p
        for p in self.q2.parameters():
            yield p
        for p in self.v.parameters():
            yield p
        for p in self.policy.parameters():
            yield p


def test_iql_networks():
    """Quick test to verify IQL network shapes and forward passes."""
    batch_size = 64
    state_dim = 29
    action_dim = 8
    d_latent = 64
    
    networks = IQLNetworks(state_dim, action_dim, d_latent)
    
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    z = torch.randn(batch_size, d_latent)
    
    # Test Q-networks
    q1, q2 = networks.get_q(states, actions, z)
    assert q1.shape == (batch_size,), f"Q1 shape: {q1.shape}"
    assert q2.shape == (batch_size,), f"Q2 shape: {q2.shape}"
    
    # Test target Q
    q_target = networks.get_q_target(states, actions, z)
    assert q_target.shape == (batch_size,), f"Q target shape: {q_target.shape}"
    
    # Test V-network
    v = networks.get_v(states, z)
    assert v.shape == (batch_size,), f"V shape: {v.shape}"
    
    # Test policy
    sampled_actions, mean, log_std = networks.policy(states, z)
    assert sampled_actions.shape == (batch_size, action_dim), f"Actions shape: {sampled_actions.shape}"
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)
    
    # Test deterministic action
    det_actions = networks.get_action(states, z, deterministic=True)
    assert det_actions.shape == (batch_size, action_dim)
    
    # Test log prob
    log_prob = networks.policy.log_prob(states, sampled_actions, z)
    assert log_prob.shape == (batch_size,), f"Log prob shape: {log_prob.shape}"
    
    # Test soft update
    networks.soft_update_targets(tau=0.005)
    
    # Test multi-state case (K states per z)
    K = 32
    states_multi = torch.randn(batch_size, K, state_dim)
    actions_multi = torch.randn(batch_size, K, action_dim)
    z_multi = z.unsqueeze(1).expand(-1, K, -1)
    
    q_multi = networks.q1(states_multi, actions_multi, z_multi)
    assert q_multi.shape == (batch_size, K), f"Q multi shape: {q_multi.shape}"
    
    v_multi = networks.v(states_multi, z_multi)
    assert v_multi.shape == (batch_size, K), f"V multi shape: {v_multi.shape}"
    
    print("All IQL network tests passed!")
    return True


if __name__ == "__main__":
    test_iql_networks()