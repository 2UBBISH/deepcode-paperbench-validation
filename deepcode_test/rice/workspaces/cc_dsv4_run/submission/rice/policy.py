"""
Policy networks for DRL agents compatible with RICE.

Provides standard actor-critic architectures that support the interface
required by RICE components (get_action_mean, get_action_std, get_value).

Architectures:
- MlpPolicy: Default Stable-Baselines3 style MLP policy for MuJoCo tasks
- CustomMLPPolicy: Configurable MLP for custom environments (selfish mining,
  cage challenge, autonomous driving, malware mutation)
- SACPolicy: Soft Actor-Critic policy (for SAC pre-trained agents)

All policies follow the black-box assumption: RICE's explanation and
refinement methods only interact with the policy through its action/value
interface, independent of internal architecture.
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def init_weights(layer: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization as used in Stable-Baselines3."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class MlpActor(nn.Module):
    """
    Actor network: outputs mean of Gaussian policy in continuous action space.
    Uses tanh activations following SB3 defaults.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        log_std_init: float = 0.0,
        action_scale: float = 1.0,
    ):
        super().__init__()
        layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            if activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_size = hs
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        self.action_scale = action_scale

        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def get_action_mean(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        return self.mean_head(features)

    def get_action_std(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        return torch.exp(self.log_std.clamp(-20, 2)).expand(batch_size, -1)


class MlpCritic(nn.Module):
    """Critic network: outputs state value V(s)."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            if activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MlpPolicy(nn.Module):
    """
    Standard MLP Actor-Critic policy (matches SB3 MlpPolicy).

    Compatible with RICE's interface requirements:
    - get_action_mean(state)
    - get_action_std(state)
    - get_value(state)

    Used for MuJoCo environments (Hopper, Walker2d, Reacher, HalfCheetah)
    and can be configured for custom environments by varying hidden_sizes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        log_std_init: float = 0.0,
        share_features: bool = False,
    ):
        """
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            hidden_sizes: Hidden layer sizes.
            activation: 'tanh' or 'relu'.
            log_std_init: Initial log standard deviation for Gaussian policy.
            share_features: Whether actor and critic share feature extractor.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.share_features = share_features

        self.actor = MlpActor(
            state_dim, action_dim, hidden_sizes, activation, log_std_init
        )
        if not share_features:
            self.critic = MlpCritic(state_dim, hidden_sizes, activation)
        else:
            # Shared feature extractor + separate heads
            layers = []
            prev_size = state_dim
            for hs in hidden_sizes:
                layers.append(nn.Linear(prev_size, hs))
                if activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
                prev_size = hs
            self.shared_features = nn.Sequential(*layers)
            self.actor_mean_head = nn.Linear(prev_size, action_dim)
            self.critic_head = nn.Linear(prev_size, 1)
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.apply(init_weights)

    def get_action_mean(self, state: torch.Tensor) -> torch.Tensor:
        if self.share_features:
            features = self.shared_features(state)
            return self.actor_mean_head(features)
        return self.actor.get_action_mean(state)

    def get_action_std(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        if self.share_features:
            return torch.exp(self.log_std.clamp(-20, 2)).expand(batch_size, -1)
        return self.actor.get_action_std(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        if self.share_features:
            features = self.shared_features(state)
            return self.critic_head(features)
        return self.critic.get_value(state)

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Get action for a single state."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mean = self.get_action_mean(state_t)
            if deterministic:
                return mean.squeeze(0).numpy()
            std = self.get_action_std(state_t)
            dist = Normal(mean, std)
            return dist.sample().squeeze(0).numpy()

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, value for PPO training."""
        mean = self.get_action_mean(state)
        std = self.get_action_std(state)
        dist = Normal(mean, std)
        action = dist.rsample()  # reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.get_value(state)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob, entropy, value for given actions."""
        mean = self.get_action_mean(state)
        std = self.get_action_std(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.get_value(state)
        return log_prob, entropy, value


class DiscreteMlpPolicy(nn.Module):
    """
    MLP Actor-Critic for discrete action spaces.

    Used for environments with discrete actions like:
    - Selfish Mining (3 actions)
    - Cage Challenge 2 (discrete actions)
    - Malware Mutation (16 actions)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        """
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            hidden_sizes: Hidden layer sizes.
            activation: 'tanh' or 'relu'.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor: outputs logits for categorical distribution
        actor_layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            actor_layers.append(nn.Linear(prev_size, hs))
            if activation == "tanh":
                actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.ReLU())
            prev_size = hs
        actor_layers.append(nn.Linear(prev_size, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            critic_layers.append(nn.Linear(prev_size, hs))
            if activation == "tanh":
                critic_layers.append(nn.Tanh())
            else:
                critic_layers.append(nn.ReLU())
            prev_size = hs
        critic_layers.append(nn.Linear(prev_size, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.apply(init_weights)

    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> int:
        """Get discrete action for a single state."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits = self.get_action_logits(state_t)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return int(dist.sample().item())

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, value for PPO training."""
        logits = self.get_action_logits(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        value = self.get_value(state)
        return action, log_prob, entropy, value


class SACPolicy(nn.Module):
    """
    Soft Actor-Critic policy network.

    Used as the pre-trained policy when the original agent was trained
    with SAC. In Experiment IV, RICE refines a SAC agent by:
    1. Using GAIL to learn an approximated PPO-compatible policy, OR
    2. Directly refining with RICE (which wraps the SAC actor).

    This provides the interface needed for GAIL imitation and RICE refinement.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            hidden_sizes: Hidden sizes (default larger for SAC).
            activation: 'relu' (SAC typically uses ReLU).
            log_std_min: Minimum log std for stability.
            log_std_max: Maximum log std.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Actor network
        actor_layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            actor_layers.append(nn.Linear(prev_size, hs))
            actor_layers.append(nn.ReLU())
            prev_size = hs
        self.actor_features = nn.Sequential(*actor_layers)
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std_head = nn.Linear(prev_size, action_dim)

        # Critic (Q-function) networks (two for SAC's double-Q trick)
        self.q1 = self._build_critic(state_dim, action_dim, hidden_sizes)
        self.q2 = self._build_critic(state_dim, action_dim, hidden_sizes)

        # Value network (for SAC)
        self.value_net = self._build_value(state_dim, hidden_sizes)

        self.apply(init_weights)

    def _build_critic(
        self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]
    ) -> nn.Module:
        layers = []
        prev_size = state_dim + action_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)

    def _build_value(
        self, state_dim: int, hidden_sizes: Tuple[int, ...]
    ) -> nn.Module:
        layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)

    def get_action_mean(self, state: torch.Tensor) -> torch.Tensor:
        features = self.actor_features(state)
        return self.mean_head(features)

    def get_action_std(self, state: torch.Tensor) -> torch.Tensor:
        features = self.actor_features(state)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return torch.exp(log_std)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_net(state)

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Get action for a single state."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mean = self.get_action_mean(state_t)
            if deterministic:
                return mean.squeeze(0).numpy()
            std = self.get_action_std(state_t)
            dist = Normal(mean, std)
            action = dist.rsample()
            return torch.tanh(action).squeeze(0).numpy()

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO compatibility."""
        mean = self.get_action_mean(state)
        std = self.get_action_std(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.get_value(state)
        return torch.tanh(action), log_prob, entropy, value


class GAILDiscriminator(nn.Module):
    """
    Discriminator for Generative Adversarial Imitation Learning (GAIL).

    Used in Experiment IV to learn an approximated policy network from
    a SAC pre-trained agent. GAIL trains a discriminator to distinguish
    expert (SAC) state-action pairs from policy-generated ones, and uses
    the discriminator's output as a reward signal for PPO training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (100, 100),
    ):
        super().__init__()
        layers = []
        prev_size = state_dim + action_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.Tanh())
            prev_size = hs
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """GAIL reward: -log(1 - D(s, a))."""
        with torch.no_grad():
            d = torch.sigmoid(self.forward(state, action))
            return -torch.log(1.0 - d + 1e-8)