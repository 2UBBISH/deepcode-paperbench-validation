"""IQL networks and losses for FRE-conditioned offline RL.

This module implements the implicit Q-learning (IQL) components described in
the Functional Reward Encodings paper:

* Twin Q critics mapping ``[state, action, z]`` to scalar values.
* A value network mapping ``[state, z]`` to scalar values.
* A squashed Gaussian policy mapping ``[state, z]`` to an action distribution.
* Expectile regression for the value network.
* Advantage-weighted regression for the policy.

The latent task code ``z`` is supplied by the frozen FRE encoder during phase-2
training and is treated as an additional conditioning vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    activation: nn.Module = nn.ReLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Construct a simple MLP.

    Args:
        input_dim: Input dimensionality.
        hidden_dims: Widths of the hidden layers.
        output_dim: Output dimensionality.
        activation: Activation module (class, not instance).
        dropout: Dropout probability applied after each hidden layer.

    Returns:
        An ``nn.Sequential`` network.
    """
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def expectile_loss(diff: torch.Tensor, expectile: float = 0.9) -> torch.Tensor:
    """Implicit expectile regression loss.

    Args:
        diff: Tensor of differences ``(target - prediction)``.
        expectile: Asymmetric weighting parameter in (0, 1). Values > 0.5
            penalize underestimation more heavily.

    Returns:
        Per-element expectile losses.
    """
    weight = torch.abs(expectile - (diff < 0.0).float())
    return weight * (diff ** 2)


def soft_update(
    target_net: nn.Module,
    source_net: nn.Module,
    tau: float = 0.005,
) -> None:
    """Polyak-averaging update of target network parameters."""
    with torch.no_grad():
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module) -> None:
    """Copy source network parameters into target network."""
    target_net.load_state_dict(source_net.state_dict())


class IQLValueNetwork(nn.Module):
    """State-value network ``V(s, z)``."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            state_dim + latent_dim,
            hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, z], dim=-1)
        return self.net(x)


class IQLQNetwork(nn.Module):
    """Twin Q-network ``Q(s, a, z)``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            state_dim + action_dim + latent_dim,
            hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x)


class SquashedGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing.

    Inputs are ``[state, z]``. The network outputs the mean and log-standard
    deviation of a normal distribution; samples are squashed with tanh to keep
    actions inside ``(-1, 1)``. The log-probability correction for tanh is
    included for advantage-weighted regression.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = build_mlp(
            state_dim + latent_dim,
            hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else 256,
            activation=activation,
        )
        trunk_out_dim = hidden_dims[-1] if hidden_dims else 256
        self.mean_head = nn.Linear(trunk_out_dim, action_dim)
        self.log_std_head = nn.Linear(trunk_out_dim, action_dim)

    def forward(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return its squashed log-probability.

        Args:
            state: State tensor ``(..., state_dim)``.
            z: Latent task code ``(..., latent_dim)``.

        Returns:
            Tuple ``(action, mean, log_std, log_prob)``.
        """
        x = torch.cat([state, z], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = Normal(mean, std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)

        # Squashed-Gaussian log probability.
        log_prob = dist.log_prob(normal_sample)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, mean, log_std, log_prob

    def sample(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sampled action and its log-probability."""
        action, _, _, log_prob = self.forward(state, z)
        return action, log_prob

    def get_action(
        self, state: torch.Tensor, z: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Return a policy action, deterministic by default."""
        if deterministic:
            x = torch.cat([state, z], dim=-1)
            h = self.trunk(x)
            mean = self.mean_head(h)
            return torch.tanh(mean)
        action, _, _, _ = self.forward(state, z)
        return action


class IQLNetworks(nn.Module):
    """Container for all IQL networks plus target Q copies.

    The module owns ``q1``, ``q2``, ``q1_target``, ``q2_target``, ``v``, and
    ``policy``. Target networks are initialized as exact copies and updated by
    calling :meth:`update_target_networks`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        expectile: float = 0.9,
        awr_temperature: float = 3.0,
        target_tau: float = 0.005,
        advantage_clip: Tuple[float, float] = (-5.0, 2.0),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.expectile = expectile
        self.awr_temperature = awr_temperature
        self.target_tau = target_tau
        self.advantage_clip = advantage_clip

        self.q1 = IQLQNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q2 = IQLQNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q1_target = IQLQNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q2_target = IQLQNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.v = IQLValueNetwork(state_dim, latent_dim, hidden_dims)
        self.policy = SquashedGaussianPolicy(
            state_dim, latent_dim, action_dim, hidden_dims
        )

        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        for param in self.q1_target.parameters():
            param.requires_grad_(False)
        for param in self.q2_target.parameters():
            param.requires_grad_(False)

    def update_target_networks(self) -> None:
        soft_update(self.q1_target, self.q1, self.target_tau)
        soft_update(self.q2_target, self.q2, self.target_tau)

    def compute_losses(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        z: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute IQL training losses on a batch.

        Args:
            states: ``(B, state_dim)``.
            actions: ``(B, action_dim)``.
            rewards: ``(B, 1)`` or ``(B,)``.
            next_states: ``(B, state_dim)``.
            z: Latent task code ``(B, latent_dim)``. The same code conditions
                both current and next states.
            dones: Optional done indicators ``(B, 1)`` or ``(B,)``.

        Returns:
            Dictionary with scalar tensors ``q_loss``, ``v_loss``,
            ``policy_loss``, and ``total_loss``.
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones is not None and dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        with torch.no_grad():
            q1_target = self.q1_target(states, actions, z)
            q2_target = self.q2_target(states, actions, z)
            q_target = torch.min(q1_target, q2_target)

            # Value target is the expectile of the target Q values.
            v_target = q_target

            next_v = self.v(next_states, z)
            q_backup = rewards + self.gamma * (1.0 - (dones if dones is not None else 0.0)) * next_v

        # Q loss (both critics).
        q1_pred = self.q1(states, actions, z)
        q2_pred = self.q2(states, actions, z)
        q_loss = F.mse_loss(q1_pred, q_backup) + F.mse_loss(q2_pred, q_backup)

        # V loss: asymmetric expectile regression toward Q target.
        v_pred = self.v(states, z)
        v_loss = expectile_loss(v_target.detach() - v_pred, self.expectile).mean()

        # Policy loss: advantage-weighted regression.
        sampled_action, _, _, log_prob = self.policy(states, z)
        with torch.no_grad():
            q_pi1 = self.q1(states, sampled_action, z)
            q_pi2 = self.q2(states, sampled_action, z)
            q_pi = torch.min(q_pi1, q_pi2)
            advantage = q_pi - v_pred.detach()
            advantage = torch.clamp(advantage, self.advantage_clip[0], self.advantage_clip[1])
            weight = torch.exp(advantage / self.awr_temperature)
        policy_loss = -(weight.detach() * log_prob).mean()

        total_loss = q_loss + v_loss + policy_loss

        return {
            "q_loss": q_loss,
            "v_loss": v_loss,
            "policy_loss": policy_loss,
            "total_loss": total_loss,
        }


def compute_iql_losses(
    q1: IQLQNetwork,
    q2: IQLQNetwork,
    q1_target: IQLQNetwork,
    q2_target: IQLQNetwork,
    v: IQLValueNetwork,
    policy: SquashedGaussianPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    z: torch.Tensor,
    gamma: float = 0.99,
    expectile: float = 0.9,
    awr_temperature: float = 3.0,
    advantage_clip: Tuple[float, float] = (-5.0, 2.0),
    dones: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Functional IQL loss helper for external callers.

    This mirrors :meth:`IQLNetworks.compute_losses` but accepts the individual
    network modules, which is convenient for unit tests and custom training
    loops that do not use the container class.
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
    if dones is not None and dones.dim() == 1:
        dones = dones.unsqueeze(-1)

    with torch.no_grad():
        q1_target_values = q1_target(states, actions, z)
        q2_target_values = q2_target(states, actions, z)
        q_target = torch.min(q1_target_values, q2_target_values)
        v_target = q_target

        next_v = v(next_states, z)
        q_backup = rewards + gamma * (1.0 - (dones if dones is not None else 0.0)) * next_v

    q1_pred = q1(states, actions, z)
    q2_pred = q2(states, actions, z)
    q_loss = F.mse_loss(q1_pred, q_backup) + F.mse_loss(q2_pred, q_backup)

    v_pred = v(states, z)
    v_loss = expectile_loss(v_target.detach() - v_pred, expectile).mean()

    sampled_action, _, _, log_prob = policy(states, z)
    with torch.no_grad():
        q_pi1 = q1(states, sampled_action, z)
        q_pi2 = q2(states, sampled_action, z)
        q_pi = torch.min(q_pi1, q_pi2)
        advantage = q_pi - v_pred.detach()
        advantage = torch.clamp(advantage, advantage_clip[0], advantage_clip[1])
        weight = torch.exp(advantage / awr_temperature)

    policy_loss = -(weight.detach() * log_prob).mean()
    total_loss = q_loss + v_loss + policy_loss

    return {
        "q_loss": q_loss,
        "v_loss": v_loss,
        "policy_loss": policy_loss,
        "total_loss": total_loss,
    }
