"""
Implicit Q-Learning (IQL) Agent with z-conditioning for Functional Reward Encodings.

Implements the IQL algorithm (Kostrikov et al., 2021) adapted for zero-shot
offline RL with latent reward representations. All networks (Q, V, policy) are
conditioned on the latent vector z produced by the FRE encoder.

Key components:
- Double Q-networks with target networks
- V-network with expectile regression
- Gaussian policy with advantage-weighted regression (AWR)
- Polyak averaging for target network updates
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


# ==============================================================================
# Network Modules
# ==============================================================================

def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """Build a simple MLP with configurable activation and optional dropout/layer norm."""
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """
    Q-network: Q(s, a, z) -> scalar Q-value.
    
    Input: state (dim d_s) concatenated with action (dim d_a) and latent z (dim d_z).
    Output: scalar Q-value.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + latent_dim
        
        self.net = _build_mlp(input_dim, 1, hidden_dims, activation, dropout)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            z: (batch, latent_dim)
        Returns:
            q_values: (batch, 1)
        """
        x = torch.cat([states, actions, z], dim=-1)
        return self.net(x)


class VNetwork(nn.Module):
    """
    V-network: V(s, z) -> scalar value.
    
    Input: state (dim d_s) concatenated with latent z (dim d_z).
    Output: scalar value.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + latent_dim
        
        self.net = _build_mlp(input_dim, 1, hidden_dims, activation, dropout)
    
    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch, state_dim)
            z: (batch, latent_dim)
        Returns:
            values: (batch, 1)
        """
        x = torch.cat([states, z], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy π(a|s, z): outputs mean and log standard deviation.
    
    Input: state (dim d_s) concatenated with latent z (dim d_z).
    Output: mean (dim d_a) and log_std (dim d_a).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        input_dim = state_dim + latent_dim
        
        self.net = _build_mlp(input_dim, 2 * action_dim, hidden_dims, activation, dropout)
    
    def forward(
        self, states: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (batch, state_dim)
            z: (batch, latent_dim)
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
        """
        x = torch.cat([states, z], dim=-1)
        output = self.net(x)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(
        self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            states: (batch, state_dim)
            z: (batch, latent_dim)
            deterministic: if True, return mean action (no noise)
        Returns:
            actions: (batch, action_dim)
            log_probs: (batch, 1) or None if deterministic
        """
        mean, log_std = self.forward(states, z)
        
        if deterministic:
            return mean, None
        
        std = torch.exp(log_std)
        # Reparameterization trick
        noise = torch.randn_like(mean)
        actions = mean + noise * std
        
        # Compute log probability
        log_probs = -0.5 * (
            ((actions - mean) / (std + 1e-6)) ** 2
            + 2 * log_std
            + np.log(2 * np.pi)
        )
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        
        return actions, log_probs
    
    def get_log_prob(
        self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of given actions under the policy.
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            z: (batch, latent_dim)
        Returns:
            log_probs: (batch, 1)
        """
        mean, log_std = self.forward(states, z)
        std = torch.exp(log_std)
        
        log_probs = -0.5 * (
            ((actions - mean) / (std + 1e-6)) ** 2
            + 2 * log_std
            + np.log(2 * np.pi)
        )
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        
        return log_probs


# ==============================================================================
# IQL Agent
# ==============================================================================

class IQLAgent:
    """
    Implicit Q-Learning (IQL) agent with z-conditioning.
    
    This agent learns a z-conditioned Q-function, V-function, and policy
    from an offline dataset. The latent vector z encodes the reward function
    and is produced by a frozen FRE encoder.
    
    Key hyperparameters:
        expectile (τ): Controls the expectile for V-function regression (default 0.7).
        temperature (α): Temperature for advantage-weighted regression (default 3.0).
        discount (γ): Discount factor (default 0.99).
        soft_target_update_rate (ρ): Polyak averaging coefficient (default 0.005).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        expectile: float = 0.7,
        temperature: float = 3.0,
        discount: float = 0.99,
        soft_target_update_rate: float = 0.005,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.soft_target_update_rate = soft_target_update_rate
        self.device = device
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Q-networks (double Q-learning)
        self.q1 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims, activation, dropout).to(device)
        self.q2 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims, activation, dropout).to(device)
        
        # Target Q-networks
        self.q1_target = deepcopy(self.q1).to(device)
        self.q2_target = deepcopy(self.q2).to(device)
        
        # V-network
        self.v = VNetwork(state_dim, latent_dim, hidden_dims, activation, dropout).to(device)
        
        # Target V-network
        self.v_target = deepcopy(self.v).to(device)
        
        # Policy
        self.policy = GaussianPolicy(
            state_dim, action_dim, latent_dim, hidden_dims, activation, dropout,
            log_std_min, log_std_max
        ).to(device)
        
        # Freeze target networks
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        for param in self.v_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        self.train()
    
    def train(self):
        """Set all networks to training mode."""
        self.q1.train()
        self.q2.train()
        self.q1_target.train()
        self.q2_target.train()
        self.v.train()
        self.v_target.train()
        self.policy.train()
    
    def eval(self):
        """Set all networks to evaluation mode."""
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        self.v.eval()
        self.v_target.eval()
        self.policy.eval()
    
    def _expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss: L2_τ(u) = |τ - 1(u<0)| * u^2.
        
        Args:
            diff: (batch, 1) - difference between target and prediction
            expectile: τ parameter
        Returns:
            loss: scalar
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()
    
    def update_v(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update V-network using expectile regression.
        
        L_V = E[ L2_τ( Q_target(s, a, z) - V(s, z) ) ]
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            z: (batch, latent_dim)
        Returns:
            dict with loss values
        """
        with torch.no_grad():
            # Use minimum of two Q-targets
            q1_target = self.q1_target(states, actions, z)
            q2_target = self.q2_target(states, actions, z)
            q_target = torch.min(q1_target, q2_target)
        
        v_pred = self.v(states, z)
        diff = q_target - v_pred
        v_loss = self._expectile_loss(diff, self.expectile)
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        return {"v_loss": v_loss.item()}
    
    def update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update Q-networks using TD learning.
        
        L_Q = E[ (r + γ * V_target(s', z) - Q(s, a, z))^2 ]
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            rewards: (batch, 1)
            next_states: (batch, state_dim)
            dones: (batch, 1)
            z: (batch, latent_dim)
        Returns:
            dict with loss values
        """
        with torch.no_grad():
            next_v = self.v_target(next_states, z)
            target = rewards + self.discount * (1 - dones) * next_v
        
        q1_pred = self.q1(states, actions, z)
        q2_pred = self.q2(states, actions, z)
        
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
        }
    
    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update policy using advantage-weighted regression (AWR).
        
        L_π = E[ exp( (Q(s,a,z) - V(s,z)) / α ) * (-log π(a|s,z)) ]
        
        This is equivalent to maximizing:
        E[ exp( A(s,a,z) / α ) * log π(a|s,z) ]
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            z: (batch, latent_dim)
        Returns:
            dict with loss values
        """
        with torch.no_grad():
            q1 = self.q1(states, actions, z)
            q2 = self.q2(states, actions, z)
            q = torch.min(q1, q2)
            v = self.v(states, z)
            advantage = q - v
            # Advantage weight: exp(A / α), clipped for stability
            exp_advantage = torch.exp(advantage / self.temperature)
            exp_advantage = torch.clamp(exp_advantage, max=100.0)
        
        log_probs = self.policy.get_log_prob(states, actions, z)
        
        # AWR loss: negative weighted log probability
        policy_loss = -(exp_advantage * log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "advantage_mean": advantage.mean().item(),
            "exp_advantage_mean": exp_advantage.mean().item(),
        }
    
    def update_targets(self):
        """Soft update target networks using Polyak averaging."""
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(
                    self.soft_target_update_rate * param.data
                    + (1 - self.soft_target_update_rate) * target_param.data
                )
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(
                    self.soft_target_update_rate * param.data
                    + (1 - self.soft_target_update_rate) * target_param.data
                )
            for param, target_param in zip(self.v.parameters(), self.v_target.parameters()):
                target_param.data.copy_(
                    self.soft_target_update_rate * param.data
                    + (1 - self.soft_target_update_rate) * target_param.data
                )
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        z: torch.Tensor,
        update_policy: bool = True,
    ) -> Dict[str, float]:
        """
        Perform one full IQL training step.
        
        Args:
            batch: dict with keys 'observations', 'actions', 'rewards',
                   'next_observations', 'terminals'
            z: (batch, latent_dim) - latent encoding of the reward function
            update_policy: whether to update the policy (can be delayed)
        Returns:
            dict with all loss values
        """
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        dones = batch["terminals"]
        
        # Ensure correct shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        # Update V
        v_info = self.update_v(states, actions, z)
        
        # Update Q
        q_info = self.update_q(states, actions, rewards, next_states, dones, z)
        
        # Update policy
        policy_info = {}
        if update_policy:
            policy_info = self.update_policy(states, actions, z)
        
        # Update target networks
        self.update_targets()
        
        return {**v_info, **q_info, **policy_info}
    
    def select_action(
        self,
        state: np.ndarray,
        z: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action for a single state.
        
        Args:
            state: (state_dim,) numpy array
            z: (latent_dim,) numpy array
            deterministic: if True, use mean action
        Returns:
            action: (action_dim,) numpy array
        """
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            z_t = torch.FloatTensor(z).unsqueeze(0).to(self.device)
            action_t, _ = self.policy.sample(state_t, z_t, deterministic=deterministic)
            action = action_t.squeeze(0).cpu().numpy()
        self.train()
        return action
    
    def select_action_batch(
        self,
        states: np.ndarray,
        z: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select actions for a batch of states.
        
        Args:
            states: (batch, state_dim) numpy array
            z: (batch, latent_dim) numpy array
            deterministic: if True, use mean actions
        Returns:
            actions: (batch, action_dim) numpy array
        """
        self.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            z_t = torch.FloatTensor(z).to(self.device)
            actions_t, _ = self.policy.sample(states_t, z_t, deterministic=deterministic)
            actions = actions_t.cpu().numpy()
        self.train()
        return actions
    
    def get_value(
        self, states: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        Get V(s, z) for given states.
        
        Args:
            states: (batch, state_dim) numpy array
            z: (batch, latent_dim) numpy array
        Returns:
            values: (batch, 1) numpy array
        """
        self.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            z_t = torch.FloatTensor(z).to(self.device)
            values = self.v(states_t, z_t).cpu().numpy()
        self.train()
        return values
    
    def get_q_value(
        self, states: np.ndarray, actions: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        Get Q(s, a, z) for given states and actions.
        
        Args:
            states: (batch, state_dim) numpy array
            actions: (batch, action_dim) numpy array
            z: (batch, latent_dim) numpy array
        Returns:
            q_values: (batch, 1) numpy array
        """
        self.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            z_t = torch.FloatTensor(z).to(self.device)
            q1 = self.q1(states_t, actions_t, z_t)
            q2 = self.q2(states_t, actions_t, z_t)
            q = torch.min(q1, q2).cpu().numpy()
        self.train()
        return q
    
    def save_checkpoint(self, path: str):
        """Save agent state to a checkpoint file."""
        checkpoint = {
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "v_state_dict": self.v.state_dict(),
            "v_target_state_dict": self.v_target.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "v_optimizer_state_dict": self.v_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "expectile": self.expectile,
            "temperature": self.temperature,
            "discount": self.discount,
            "soft_target_update_rate": self.soft_target_update_rate,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load agent state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        self.v.load_state_dict(checkpoint["v_state_dict"])
        self.v_target.load_state_dict(checkpoint["v_target_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.v_optimizer.load_state_dict(checkpoint["v_optimizer_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    
    def get_network_params(self) -> Dict[str, int]:
        """Get parameter counts for each network."""
        return {
            "q1_params": sum(p.numel() for p in self.q1.parameters()),
            "q2_params": sum(p.numel() for p in self.q2.parameters()),
            "v_params": sum(p.numel() for p in self.v.parameters()),
            "policy_params": sum(p.numel() for p in self.policy.parameters()),
            "total_params": (
                sum(p.numel() for p in self.q1.parameters())
                + sum(p.numel() for p in self.q2.parameters())
                + sum(p.numel() for p in self.v.parameters())
                + sum(p.numel() for p in self.policy.parameters())
            ),
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def build_iql_agent(
    state_dim: int,
    action_dim: int,
    latent_dim: int = 64,
    hidden_dims: Optional[List[int]] = None,
    activation: str = "relu",
    dropout: float = 0.0,
    expectile: float = 0.7,
    temperature: float = 3.0,
    discount: float = 0.99,
    soft_target_update_rate: float = 0.005,
    log_std_min: float = -5.0,
    log_std_max: float = 2.0,
    device: str = "cpu",
) -> IQLAgent:
    """
    Factory function to create an IQLAgent with specified hyperparameters.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        latent_dim: Dimension of latent z vector (default 64)
        hidden_dims: List of hidden layer dimensions (default [256, 256])
        activation: Activation function ("relu", "gelu", "tanh")
        dropout: Dropout rate (default 0.0)
        expectile: τ for expectile regression (default 0.7)
        temperature: α for AWR (default 3.0)
        discount: γ discount factor (default 0.99)
        soft_target_update_rate: ρ for Polyak averaging (default 0.005)
        log_std_min: Minimum log std for policy (default -5.0)
        log_std_max: Maximum log std for policy (default 2.0)
        device: Device to place networks on
    
    Returns:
        IQLAgent instance
    """
    return IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        expectile=expectile,
        temperature=temperature,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        device=device,
    )


# ==============================================================================
# Testing
# ==============================================================================

def test_iql_agent():
    """Quick test to verify IQL agent functionality."""
    print("Testing IQL Agent...")
    
    state_dim = 10
    action_dim = 4
    latent_dim = 64
    batch_size = 256
    device = "cpu"
    
    agent = build_iql_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        device=device,
    )
    
    print(f"  Total parameters: {agent.get_network_params()['total_params']}")
    
    # Create dummy batch
    batch = {
        "observations": torch.randn(batch_size, state_dim),
        "actions": torch.randn(batch_size, action_dim),
        "rewards": torch.randn(batch_size, 1),
        "next_observations": torch.randn(batch_size, state_dim),
        "terminals": torch.zeros(batch_size, 1),
    }
    z = torch.randn(batch_size, latent_dim)
    
    # Run training step
    info = agent.training_step(batch, z)
    print(f"  Training step losses: {info}")
    
    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    z_np = np.random.randn(latent_dim).astype(np.float32)
    action = agent.select_action(state, z_np)
    print(f"  Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test batch action selection
    states = np.random.randn(32, state_dim).astype(np.float32)
    z_batch = np.random.randn(32, latent_dim).astype(np.float32)
    actions = agent.select_action_batch(states, z_batch)
    print(f"  Batch actions shape: {actions.shape}")
    
    # Test value prediction
    values = agent.get_value(states, z_batch)
    print(f"  Values shape: {values.shape}, range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Test Q-value prediction
    q_values = agent.get_q_value(states, actions, z_batch)
    print(f"  Q-values shape: {q_values.shape}, range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    print("  IQL Agent test passed!")
    return True


if __name__ == "__main__":
    test_iql_agent()