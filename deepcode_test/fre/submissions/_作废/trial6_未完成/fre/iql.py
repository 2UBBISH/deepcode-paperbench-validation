"""
Implicit Q-Learning (IQL) agent conditioned on FRE latent vector z.

Implements the IQL algorithm (Kostrikov et al., 2021) with conditioning on
the latent vector z produced by the FRE encoder. This enables zero-shot
offline RL by encoding a reward function into z and then executing the
policy π(a|s,z) that was trained to maximize that encoded reward.

Architecture:
    - Q-function: Q(s, a, z) -> scalar Q-value
    - Value function: V(s, z) -> scalar value
    - Policy: π(a|s, z) -> action distribution (from fre.policy)

Losses:
    - Value loss: expectile regression L2_τ(Q(s,a,z) - V(s,z))
    - Q loss: TD learning with V as target (no max over actions)
    - Policy loss: Advantage-weighted regression (AWR)
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fre.policy import GaussianPolicy, DiscretePolicy, create_policy


# ==============================================================================
# Value Network: V(s, z) -> scalar
# ==============================================================================

class ValueNetwork(nn.Module):
    """
    State-value network conditioned on latent z.
    
    V(s, z) = MLP(concat(s, z)) -> scalar value.
    """
    
    def __init__(
        self,
        state_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_latent = d_latent
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        input_dim = state_dim + d_latent
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute V(s, z).
        
        Args:
            state: (batch, state_dim) or (batch, num_states, state_dim)
            z: (batch, d_latent)
        
        Returns:
            value: (batch, 1) or (batch, num_states, 1)
        """
        if state.dim() == 3:
            # Multi-state: (batch, num_states, state_dim)
            batch_size, num_states, state_dim = state.shape
            z_expanded = z.unsqueeze(1).expand(-1, num_states, -1)
            flat_state = state.reshape(-1, state_dim)
            flat_z = z_expanded.reshape(-1, self.d_latent)
            flat_input = torch.cat([flat_state, flat_z], dim=-1)
            flat_out = self.net(flat_input)
            return flat_out.reshape(batch_size, num_states, 1)
        else:
            # Single state: (batch, state_dim)
            inp = torch.cat([state, z], dim=-1)
            return self.net(inp)


# ==============================================================================
# Q-Network: Q(s, a, z) -> scalar
# ==============================================================================

class QNetwork(nn.Module):
    """
    State-action value network conditioned on latent z.
    
    Q(s, a, z) = MLP(concat(s, a, z)) -> scalar Q-value.
    Uses double Q-learning with two independent networks.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_latent = d_latent
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        input_dim = state_dim + action_dim + d_latent
        
        # Build two Q-networks for double Q-learning
        self.q1 = self._build_network(input_dim, hidden_dims, activation, dropout, use_layer_norm)
        self.q2 = self._build_network(input_dim, hidden_dims, activation, dropout, use_layer_norm)
    
    def _build_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: nn.Module,
        dropout: float,
        use_layer_norm: bool,
    ) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q1(s,a,z) and Q2(s,a,z).
        
        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)
            z: (batch, d_latent)
        
        Returns:
            q1: (batch, 1)
            q2: (batch, 1)
        """
        inp = torch.cat([state, action, z], dim=-1)
        return self.q1(inp), self.q2(inp)
    
    def get_min_q(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return min(Q1, Q2) for conservative estimate."""
        q1, q2 = self.forward(state, action, z)
        return torch.min(q1, q2)
    
    def get_q1(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return Q1 only."""
        inp = torch.cat([state, action, z], dim=-1)
        return self.q1(inp)


# ==============================================================================
# IQL Agent: Combines V, Q, and Policy with IQL losses
# ==============================================================================

class IQLAgent:
    """
    Implicit Q-Learning agent conditioned on FRE latent z.
    
    This agent implements the full IQL training loop:
    - Value function V(s,z) trained with expectile regression
    - Q-function Q(s,a,z) trained with TD learning
    - Policy π(a|s,z) trained with advantage-weighted regression
    
    The agent is designed to work with a frozen FRE encoder that produces
    latent z from (state, reward) pairs.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[List[int]] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
        discount: float = 0.99,
        policy_log_std_min: float = -5.0,
        policy_log_std_max: float = 2.0,
        discrete: bool = False,
        num_actions: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_latent = d_latent
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.discrete = discrete
        self.device = device
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Initialize networks
        self.vf = ValueNetwork(
            state_dim=state_dim,
            d_latent=d_latent,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        ).to(device)
        
        self.qf = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            d_latent=d_latent,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        ).to(device)
        
        self.policy = create_policy(
            state_dim=state_dim,
            action_dim=action_dim,
            d_latent=d_latent,
            hidden_dims=hidden_dims,
            discrete=discrete,
            num_actions=num_actions,
            log_std_min=policy_log_std_min,
            log_std_max=policy_log_std_max,
            dropout=dropout,
        ).to(device)
        
        # Target value network for stability
        self.vf_target = ValueNetwork(
            state_dim=state_dim,
            d_latent=d_latent,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        ).to(device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        for p in self.vf_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Training step counter
        self.total_steps = 0
        
        # Track losses
        self._vf_losses: List[float] = []
        self._qf_losses: List[float] = []
        self._policy_losses: List[float] = []
    
    def _expectile_loss(
        self, diff: torch.Tensor, expectile: float
    ) -> torch.Tensor:
        """
        Compute expectile loss: L2_τ(u) = |τ - 1(u<0)| * u^2.
        
        Args:
            diff: (batch, 1) - difference between Q and V
            expectile: τ parameter (0.5 = mean, >0.5 gives more weight to positive errors)
        
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
    ) -> float:
        """
        Update value function V(s,z) using expectile regression.
        
        L_V = E[ L2_τ(Q(s,a,z) - V(s,z)) ]
        where Q is the target (detached) and V is optimized.
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            z: (batch, d_latent)
        
        Returns:
            v_loss: scalar float
        """
        with torch.no_grad():
            q1, q2 = self.qf(states, actions, z)
            q = torch.min(q1, q2)
        
        v = self.vf(states, z)
        diff = q - v
        v_loss = self._expectile_loss(diff, self.expectile)
        
        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()
        
        self._vf_losses.append(v_loss.item())
        return v_loss.item()
    
    def update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
    ) -> float:
        """
        Update Q-function using TD learning with V as target.
        
        L_Q = E[ (r + γ * V(s', z) * (1 - done) - Q(s,a,z))^2 ]
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            rewards: (batch, 1)
            next_states: (batch, state_dim)
            dones: (batch, 1)
            z: (batch, d_latent)
        
        Returns:
            q_loss: scalar float
        """
        with torch.no_grad():
            next_v = self.vf_target(next_states, z)
            target = rewards + self.discount * next_v * (1.0 - dones)
        
        q1, q2 = self.qf(states, actions, z)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss
        
        self.qf_optimizer.zero_grad()
        q_loss.backward()
        self.qf_optimizer.step()
        
        self._qf_losses.append(q_loss.item())
        return q_loss.item()
    
    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> float:
        """
        Update policy using advantage-weighted regression (AWR).
        
        L_π = E[ exp( (Q(s,a,z) - V(s,z)) / α ) * (-log π(a|s,z)) ]
        where α is the temperature parameter.
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            z: (batch, d_latent)
        
        Returns:
            policy_loss: scalar float
        """
        with torch.no_grad():
            q1, q2 = self.qf(states, actions, z)
            q = torch.min(q1, q2)
            v = self.vf(states, z)
            advantage = q - v
            # Advantage-weighted regression weights
            exp_adv = torch.exp(advantage / self.temperature)
            # Clamp for numerical stability
            exp_adv = torch.clamp(exp_adv, max=100.0)
        
        # Get log probability of the actions under current policy
        log_prob = self.policy.get_log_prob(states, z, actions)
        
        # AWR loss: minimize -weight * log_prob = maximize weight * log_prob
        policy_loss = -(exp_adv * log_prob).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self._policy_losses.append(policy_loss.item())
        return policy_loss.item()
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
        update_policy: bool = True,
    ) -> Dict[str, float]:
        """
        Perform one full IQL training step: update V, Q, and optionally policy.
        
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim)
            rewards: (batch, 1)
            next_states: (batch, state_dim)
            dones: (batch, 1)
            z: (batch, d_latent)
            update_policy: whether to update policy this step
        
        Returns:
            dict with keys: 'v_loss', 'q_loss', 'policy_loss'
        """
        v_loss = self.update_v(states, actions, z)
        q_loss = self.update_q(states, actions, rewards, next_states, dones, z)
        
        policy_loss = 0.0
        if update_policy:
            policy_loss = self.update_policy(states, actions, z)
        
        # Update target network with soft update
        self._soft_update_target()
        
        self.total_steps += 1
        
        return {
            'v_loss': v_loss,
            'q_loss': q_loss,
            'policy_loss': policy_loss,
        }
    
    def _soft_update_target(self, tau: float = 0.005):
        """Soft update target V network: θ_target = τ*θ + (1-τ)*θ_target."""
        for target_param, param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )
    
    def get_action(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Get action from policy for given state and latent z.
        
        Args:
            state: (state_dim,) or (batch, state_dim)
            z: (d_latent,) or (batch, d_latent)
            deterministic: if True, return mean action
        
        Returns:
            action: numpy array (action_dim,) or (batch, action_dim)
        """
        was_single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            was_single = True
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        state = state.to(self.device)
        z = z.to(self.device)
        
        with torch.no_grad():
            if deterministic:
                if self.discrete:
                    action, _ = self.policy.sample(state, z, deterministic=True)
                else:
                    action = self.policy.get_action_mean(state, z)
            else:
                action, _ = self.policy.sample(state, z, deterministic=False)
        
        action = action.cpu().numpy()
        if was_single:
            action = action[0]
        return action
    
    def get_value(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Get V(s,z)."""
        return self.vf(state.to(self.device), z.to(self.device))
    
    def get_q_value(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Get min(Q1, Q2) for (s,a,z)."""
        return self.qf.get_min_q(
            state.to(self.device), action.to(self.device), z.to(self.device)
        )
    
    def get_recent_losses(self, window: int = 100) -> Dict[str, float]:
        """Get average losses over recent steps."""
        def avg(lst):
            if not lst:
                return 0.0
            return float(np.mean(lst[-window:]))
        
        return {
            'v_loss': avg(self._vf_losses),
            'q_loss': avg(self._qf_losses),
            'policy_loss': avg(self._policy_losses),
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get all network state dicts for checkpointing."""
        return {
            'vf': self.vf.state_dict(),
            'vf_target': self.vf_target.state_dict(),
            'qf': self.qf.state_dict(),
            'policy': self.policy.state_dict(),
            'vf_optimizer': self.vf_optimizer.state_dict(),
            'qf_optimizer': self.qf_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load all network state dicts from checkpoint."""
        self.vf.load_state_dict(state_dict['vf'])
        self.vf_target.load_state_dict(state_dict['vf_target'])
        self.qf.load_state_dict(state_dict['qf'])
        self.policy.load_state_dict(state_dict['policy'])
        self.vf_optimizer.load_state_dict(state_dict['vf_optimizer'])
        self.qf_optimizer.load_state_dict(state_dict['qf_optimizer'])
        self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self.total_steps = state_dict.get('total_steps', 0)
    
    def train(self):
        """Set all networks to training mode."""
        self.vf.train()
        self.qf.train()
        self.policy.train()
    
    def eval(self):
        """Set all networks to evaluation mode."""
        self.vf.eval()
        self.qf.eval()
        self.policy.eval()


# ==============================================================================
# IQL Trainer: Manages the IQL training loop with FRE encoder
# ==============================================================================

class IQLTrainer:
    """
    Trainer for Phase 2: IQL training with frozen FRE encoder.
    
    This trainer handles:
    - Sampling reward functions from the prior
    - Encoding them into z using the frozen FRE encoder
    - Sampling batches from the offline dataset
    - Computing rewards for those batches using the sampled reward function
    - Updating the IQL agent
    """
    
    def __init__(
        self,
        agent: IQLAgent,
        fre_model: 'FREModel',  # type: ignore
        prior: 'MixedPrior',  # type: ignore
        dataset: 'OfflineDataset',  # type: ignore
        device: torch.device = torch.device("cpu"),
        K_encoder: int = 32,
        batch_size: int = 256,
        clip_grad_norm: Optional[float] = None,
        policy_update_delay: int = 1,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.agent = agent
        self.fre_model = fre_model
        self.prior = prior
        self.dataset = dataset
        self.device = device
        self.K_encoder = K_encoder
        self.batch_size = batch_size
        self.clip_grad_norm = clip_grad_norm
        self.policy_update_delay = policy_update_delay
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Ensure FRE encoder is frozen
        self.fre_model.freeze_encoder()
        self.fre_model.eval()
        
        # Step counter
        self.step = 0
        
        # Loss tracking
        self._loss_history: List[Dict[str, float]] = []
    
    def _sample_reward_function(self) -> 'RewardFunction':  # type: ignore
        """Sample a random reward function from the prior."""
        return self.prior.sample(self.rng)
    
    def _encode_reward_function(
        self, reward_fn: 'RewardFunction'  # type: ignore
    ) -> torch.Tensor:
        """
        Encode a reward function into latent z using the frozen FRE encoder.
        
        Args:
            reward_fn: callable reward function
        
        Returns:
            z: (1, d_latent) latent vector
        """
        # Sample encoder states from dataset
        encoder_states = self.dataset.sample_random_norm_states(self.K_encoder)
        encoder_states_t = torch.FloatTensor(encoder_states).to(self.device)
        
        # Compute rewards for encoder states
        raw_states = self.dataset.sample_random_states(self.K_encoder)
        rewards = reward_fn(raw_states)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        
        # Encode using frozen FRE encoder (deterministic)
        with torch.no_grad():
            z = self.fre_model.encode_reward(
                encoder_states_t.unsqueeze(0),  # (1, K, state_dim)
                rewards_t.unsqueeze(0),          # (1, K)
                deterministic=True,
            )  # (1, d_latent)
        
        return z
    
    def sample_training_batch(
        self,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a training batch for IQL.
        
        Returns:
            states, actions, rewards, next_states, dones, z
        """
        # Sample reward function and encode
        reward_fn = self._sample_reward_function()
        z = self._encode_reward_function(reward_fn)  # (1, d_latent)
        
        # Sample batch from dataset
        batch = self.dataset.sample_batch(self.batch_size)
        
        states = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        next_states = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.FloatTensor(batch['terminals']).to(self.device)
        
        # Compute rewards using the sampled reward function on raw states
        raw_states = batch['observations_raw'] if 'observations_raw' in batch else batch['observations']
        rewards_np = reward_fn(raw_states)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        
        # Ensure correct shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        # Expand z to batch size
        z_batch = z.expand(self.batch_size, -1)  # (batch, d_latent)
        
        return states, actions, rewards, next_states, dones, z_batch
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one IQL training step.
        
        Returns:
            dict with loss values
        """
        states, actions, rewards, next_states, dones, z = self.sample_training_batch()
        
        update_policy = (self.step % self.policy_update_delay == 0)
        
        losses = self.agent.train_step(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            z=z,
            update_policy=update_policy,
        )
        
        # Clip gradients if configured
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.agent.vf.parameters(), self.clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.agent.qf.parameters(), self.clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.agent.policy.parameters(), self.clip_grad_norm
            )
        
        self.step += 1
        self._loss_history.append(losses)
        
        return losses
    
    def get_recent_losses(self, window: int = 100) -> Dict[str, float]:
        """Get average losses over recent steps."""
        if not self._loss_history:
            return {'v_loss': 0.0, 'q_loss': 0.0, 'policy_loss': 0.0}
        
        recent = self._loss_history[-window:]
        return {
            'v_loss': float(np.mean([l['v_loss'] for l in recent])),
            'q_loss': float(np.mean([l['q_loss'] for l in recent])),
            'policy_loss': float(np.mean([l['policy_loss'] for l in recent])),
        }
    
    def save_checkpoint(self, path: str):
        """Save IQL agent checkpoint."""
        torch.save(self.agent.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load IQL agent checkpoint."""
        self.agent.load_state_dict(torch.load(path, map_location=self.device))


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_iql_agent(
    state_dim: int,
    action_dim: int,
    d_latent: int = 64,
    hidden_dims: Optional[List[int]] = None,
    expectile: float = 0.7,
    temperature: float = 3.0,
    discount: float = 0.99,
    discrete: bool = False,
    num_actions: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> IQLAgent:
    """
    Create an IQL agent with default hyperparameters.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        d_latent: Dimension of FRE latent vector z
        hidden_dims: Hidden layer dimensions for all networks
        expectile: Expectile parameter τ for value loss
        temperature: Temperature α for AWR policy loss
        discount: Discount factor γ
        discrete: Whether action space is discrete
        num_actions: Number of discrete actions (if discrete=True)
        device: Torch device
    
    Returns:
        IQLAgent instance
    """
    return IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        d_latent=d_latent,
        hidden_dims=hidden_dims,
        expectile=expectile,
        temperature=temperature,
        discount=discount,
        discrete=discrete,
        num_actions=num_actions,
        device=device,
        **kwargs,
    )


def create_iql_trainer(
    agent: IQLAgent,
    fre_model: 'FREModel',  # type: ignore
    prior: 'MixedPrior',  # type: ignore
    dataset: 'OfflineDataset',  # type: ignore
    device: torch.device = torch.device("cpu"),
    K_encoder: int = 32,
    batch_size: int = 256,
    clip_grad_norm: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
) -> IQLTrainer:
    """
    Create an IQL trainer with default hyperparameters.
    
    Args:
        agent: IQLAgent instance
        fre_model: Trained FRE model (encoder will be frozen)
        prior: MixedPrior for sampling reward functions
        dataset: OfflineDataset for sampling transitions
        device: Torch device
        K_encoder: Number of encoder states for z computation
        batch_size: Batch size for IQL updates
        clip_grad_norm: Gradient clipping norm
        rng: Random state
    
    Returns:
        IQLTrainer instance
    """
    return IQLTrainer(
        agent=agent,
        fre_model=fre_model,
        prior=prior,
        dataset=dataset,
        device=device,
        K_encoder=K_encoder,
        batch_size=batch_size,
        clip_grad_norm=clip_grad_norm,
        rng=rng,
    )