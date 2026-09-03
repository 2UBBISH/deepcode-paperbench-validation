"""
Implicit Q-Learning (IQL) agent conditioned on latent vector z.

Implements the IQL algorithm from Kostrikov et al. (2021) with z-conditioning
for zero-shot offline RL. All networks (policy, Q, V) receive the latent
vector z concatenated to their inputs, enabling the agent to adapt its
behavior based on the encoded reward function.

Architecture:
    - Policy π(a|s, z): Gaussian with tanh squashing, MLP [256, 256]
    - Q-function Q(s, a, z): MLP [256, 256] → scalar
    - Value function V(s, z): MLP [256, 256] → scalar

Losses:
    - V loss: expectile regression with τ = 0.7
    - Q loss: standard TD learning with γ = 0.99
    - Policy loss: advantage-weighted regression with α = 3.0
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Network Components
# ---------------------------------------------------------------------------

def _init_weights(m: nn.Module):
    """Initialize linear layers with Kaiming uniform."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """Simple MLP with optional layer norm and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
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
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy π(a|s, z) with tanh squashing.

    Input: concatenation of state s and latent z.
    Output: mean and log_std of action distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_z: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_z = d_z
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        input_dim = state_dim + d_z
        self.backbone = MLP(
            input_dim=input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout=dropout,
        )
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        self.apply(_init_weights)

    def forward(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim)
            z:     (B, d_z)
        Returns:
            mean:    (B, action_dim)
            log_std: (B, action_dim)
        """
        x = torch.cat([state, z], dim=-1)
        h = self.backbone(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.

        Returns:
            action:     (B, action_dim)  -- squashed via tanh
            log_prob:   (B,)             -- log probability of sampled action
            pre_tanh:   (B, action_dim)  -- pre-tanh value (for entropy)
        """
        mean, log_std = self.forward(state, z)
        std = log_std.exp()
        # Reparameterization
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)

        # Log probability with tanh correction
        log_prob = self._log_prob(mean, log_std, pre_tanh)
        return action, log_prob, pre_tanh

    def _log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log π(a|s,z) with tanh squashing correction.

        log π(a|s) = log N(pre_tanh | μ, σ²) - Σ log(1 - tanh²(pre_tanh))
        """
        std = log_std.exp()
        var = std.pow(2)
        # Gaussian log density
        log_density = -0.5 * (
            ((pre_tanh - mean) / std).pow(2)
            + 2.0 * log_std
            + np.log(2 * np.pi)
        )
        log_density = log_density.sum(dim=-1)  # (B,)

        # Tanh correction: log(1 - tanh²(x))
        log_tanh_correction = 2.0 * (
            np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        log_tanh_correction = log_tanh_correction.sum(dim=-1)  # (B,)

        log_prob = log_density - log_tanh_correction
        return log_prob

    def get_action(
        self, state: torch.Tensor, z: torch.Tensor, deterministic: bool = False
    ) -> np.ndarray:
        """Get action for evaluation (numpy interface)."""
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state, z)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.sample(state, z)
        return action.cpu().numpy()


class QFunction(nn.Module):
    """
    Q-function Q(s, a, z): MLP mapping (state, action, z) → scalar value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_z: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        input_dim = state_dim + action_dim + d_z
        self.net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state:  (B, state_dim)
            action: (B, action_dim)
            z:      (B, d_z)
        Returns:
            q_value: (B, 1)
        """
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x)


class ValueFunction(nn.Module):
    """
    Value function V(s, z): MLP mapping (state, z) → scalar value.
    """

    def __init__(
        self,
        state_dim: int,
        d_z: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        input_dim = state_dim + d_z
        self.net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            z:     (B, d_z)
        Returns:
            v_value: (B, 1)
        """
        x = torch.cat([state, z], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# IQL Losses
# ---------------------------------------------------------------------------

def expectile_loss(
    diff: torch.Tensor, expectile: float = 0.7
) -> torch.Tensor:
    """
    Expectile regression loss: L(τ, x) = |τ - 1_{x < 0}| * x²

    Args:
        diff: (B, 1) -- difference tensor (e.g., Q - V)
        expectile: τ parameter (default 0.7)
    Returns:
        scalar loss
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * (diff ** 2)).mean()


def iql_value_loss(
    v_net: ValueFunction,
    target_q_net: QFunction,
    policy: GaussianPolicy,
    states: torch.Tensor,
    z: torch.Tensor,
    expectile: float = 0.7,
) -> torch.Tensor:
    """
    IQL Value loss:
        L_V = E_{(s,z)} [ L_τ²( Q_θ̂(s, a, z) - V_ψ(s, z) ) ]
    where a ~ π(·|s, z) and L_τ² is the expectile loss.

    Uses target Q-network (or current Q with no grad) for stability.
    """
    with torch.no_grad():
        # Sample actions from current policy
        actions, _, _ = policy.sample(states, z)
        # Target Q values (use target Q or detached Q)
        q_target = target_q_net(states, actions, z)

    v_pred = v_net(states, z)
    diff = q_target.detach() - v_pred
    return expectile_loss(diff, expectile)


def iql_q_loss(
    q_net: QFunction,
    target_v_net: ValueFunction,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    z: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    IQL Q loss (standard TD):
        L_Q = E[(r + γ * V_ψ(s', z) - Q_θ(s, a, z))²]

    Uses target V-network for the bootstrap value.
    """
    with torch.no_grad():
        next_v = target_v_net(next_states, z)
        target = rewards + gamma * (1.0 - dones.float()) * next_v

    q_pred = q_net(states, actions, z)
    return F.mse_loss(q_pred, target)


def iql_policy_loss(
    policy: GaussianPolicy,
    q_net: QFunction,
    v_net: ValueFunction,
    states: torch.Tensor,
    z: torch.Tensor,
    alpha: float = 3.0,
    clip_advantage: Optional[float] = None,
) -> torch.Tensor:
    """
    IQL Policy loss (advantage-weighted regression):
        L_π = E[ exp(α * (Q - V)) * (-log π(a|s, z)) ]

    The advantage is computed with detached Q and V.
    """
    with torch.no_grad():
        v = v_net(states, z)
        actions_sample, _, _ = policy.sample(states, z)
        q = q_net(states, actions_sample, z)
        advantage = q - v
        if clip_advantage is not None:
            advantage = torch.clamp(advantage, max=clip_advantage)
        weights = torch.exp(alpha * advantage).clamp(max=100.0)  # numerical stability

    # Compute log_prob of the sampled actions
    mean, log_std = policy.forward(states, z)
    log_prob = policy._log_prob(mean, log_std, actions_sample)

    # Weighted negative log-likelihood
    loss = -(weights * log_prob).mean()
    return loss


# ---------------------------------------------------------------------------
# IQL Agent
# ---------------------------------------------------------------------------

class IQLAgent:
    """
    Implicit Q-Learning agent conditioned on latent vector z.

    Maintains policy, Q-function, V-function networks and their target
    counterparts. Provides training step and action selection.

    Usage:
        agent = IQLAgent(state_dim, action_dim, d_z=64)
        # Training loop:
        for batch in replay_buffer:
            z = encoder.encode_deterministic(encoder_states, encoder_rewards)
            metrics = agent.train_step(batch, z)
        # Evaluation:
        action = agent.get_action(state, z)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_z: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,          # target network soft-update rate
        expectile: float = 0.7,
        alpha: float = 3.0,          # AWR temperature
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        clip_advantage: Optional[float] = None,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        dropout: float = 0.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_z = d_z
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.alpha = alpha
        self.clip_advantage = clip_advantage

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Networks
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            d_z=d_z,
            hidden_dims=hidden_dims,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout=dropout,
        ).to(device)

        self.q_net = QFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            d_z=d_z,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

        self.target_q_net = QFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            d_z=d_z,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.v_net = ValueFunction(
            state_dim=state_dim,
            d_z=d_z,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

        self.target_v_net = ValueFunction(
            state_dim=state_dim,
            d_z=d_z,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        self.target_v_net.load_state_dict(self.v_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)

        # Training step counter
        self.train_steps = 0

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        z: torch.Tensor,
        update_target: bool = True,
    ) -> Dict[str, float]:
        """
        Perform one IQL training step.

        Args:
            batch: dict with keys 'states', 'actions', 'rewards',
                   'next_states', 'dones' (all tensors on device).
            z: latent vector (B, d_z) on device.
            update_target: whether to soft-update target networks.

        Returns:
            dict of loss values for logging.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # --- Value update ---
        v_loss = iql_value_loss(
            v_net=self.v_net,
            target_q_net=self.target_q_net,
            policy=self.policy,
            states=states,
            z=z,
            expectile=self.expectile,
        )
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # --- Q update ---
        q_loss = iql_q_loss(
            q_net=self.q_net,
            target_v_net=self.target_v_net,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            z=z,
            gamma=self.gamma,
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Policy update ---
        pi_loss = iql_policy_loss(
            policy=self.policy,
            q_net=self.q_net,
            v_net=self.v_net,
            states=states,
            z=z,
            alpha=self.alpha,
            clip_advantage=self.clip_advantage,
        )
        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()

        # --- Target network soft update ---
        if update_target:
            self._soft_update(self.q_net, self.target_q_net, self.tau)
            self._soft_update(self.v_net, self.target_v_net, self.tau)

        self.train_steps += 1

        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'pi_loss': pi_loss.item(),
        }

    @torch.no_grad()
    def get_action(
        self,
        state: np.ndarray,
        z: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get action for a single state.

        Args:
            state: (state_dim,) numpy array
            z:     (d_z,) numpy array
            deterministic: if True, use mean action (no noise)

        Returns:
            action: (action_dim,) numpy array
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy.get_action(state_t, z_t, deterministic=deterministic)
        return action.squeeze(0)

    @torch.no_grad()
    def get_actions(
        self,
        states: np.ndarray,
        z: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get actions for a batch of states.

        Args:
            states: (B, state_dim) numpy array
            z:      (d_z,) numpy array (broadcast to all states)
            deterministic: if True, use mean action

        Returns:
            actions: (B, action_dim) numpy array
        """
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        B = states_t.shape[0]
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1)
        return self.policy.get_action(states_t, z_t, deterministic=deterministic)

    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """Polyak averaging: target = tau * source + (1 - tau) * target."""
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def state_dict(self) -> Dict[str, dict]:
        """Get all network state dicts for checkpointing."""
        return {
            'policy': self.policy.state_dict(),
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'v_net': self.v_net.state_dict(),
            'target_v_net': self.target_v_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'train_steps': self.train_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, dict]):
        """Load all network state dicts from checkpoint."""
        self.policy.load_state_dict(state_dict['policy'])
        self.q_net.load_state_dict(state_dict['q_net'])
        self.target_q_net.load_state_dict(state_dict['target_q_net'])
        self.v_net.load_state_dict(state_dict['v_net'])
        self.target_v_net.load_state_dict(state_dict['target_v_net'])
        self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self.q_optimizer.load_state_dict(state_dict['q_optimizer'])
        self.v_optimizer.load_state_dict(state_dict['v_optimizer'])
        self.train_steps = state_dict.get('train_steps', 0)

    def to(self, device: torch.device):
        """Move all networks to device."""
        self.device = device
        self.policy.to(device)
        self.q_net.to(device)
        self.target_q_net.to(device)
        self.v_net.to(device)
        self.target_v_net.to(device)
        return self

    def train(self):
        """Set all networks to training mode."""
        self.policy.train()
        self.q_net.train()
        self.target_q_net.train()
        self.v_net.train()
        self.target_v_net.train()

    def eval(self):
        """Set all networks to evaluation mode."""
        self.policy.eval()
        self.q_net.eval()
        self.target_q_net.eval()
        self.v_net.eval()
        self.target_v_net.eval()