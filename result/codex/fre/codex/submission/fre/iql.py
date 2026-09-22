"""Implicit Q-Learning conditioned on the FRE latent ``z``.

Section 4.1: "In our experiments, we use implicit Q-learning (Kostrikov et al.,
2021) as the offline RL method to train our FRE-conditioned policy."

The RL components (Q-function, value function, policy) are all conditioned on
the latent encoding ``z``.  The addendum specifies how the conditioning is
performed: "the latent embedding is simply concatenated to the observation
state that is fed into the RL components."

All hyperparameters follow Appendix A:

    RL Network Layers   [512, 512, 512]
    Discount Factor     0.88
    Target Update Rate  0.001
    IQL Expectile       0.8
    AWR Temperature     3.0
    Optimizer           Adam, lr 1e-4

This follows the reference IQL implementation: the value function is regressed
onto the **target** Q-function with an expectile loss, the Q-function is
trained with a Bellman backup against ``V(s')``, and the policy is extracted
with an advantage-weighted regression (AWR) objective.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.decoder import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class TanhGaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy conditioned on ``(s, z)``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim + z_dim, hidden_dims, 2 * action_dim, activation="relu")
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([obs, z], dim=-1))
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs, z)
        if deterministic:
            action = torch.tanh(mean)
            return action, torch.zeros(obs.shape[0], device=obs.device)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        # Tanh correction from the standard SAC/IQL policy parameterisation.
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-density of an action sampled from the dataset under the policy."""
        mean, log_std = self.forward(obs, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # Invert the tanh squashing, guarding against saturation.
        action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        x = torch.atanh(action)
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """Q(s, a, z)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim + action_dim + z_dim, hidden_dims, 1, activation="relu")

    def forward(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action, z], dim=-1)).squeeze(-1)


class VNetwork(nn.Module):
    """V(s, z)."""

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim + z_dim, hidden_dims, 1, activation="relu")

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, z], dim=-1)).squeeze(-1)


def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    """Expectile regression loss used by IQL for the value function."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


@dataclass
class IQLConfig:
    """IQL hyperparameters (Appendix A)."""

    obs_dim: int
    action_dim: int
    z_dim: int = 128
    hidden_dims: Tuple[int, ...] = (512, 512, 512)
    discount: float = 0.88
    expectile: float = 0.8
    awr_temperature: float = 3.0
    target_update_rate: float = 0.001
    learning_rate: float = 1e-4
    max_advantage_weight: float = 100.0


class IQLAgent:
    """IQL with all networks conditioned on a FRE latent ``z``."""

    def __init__(self, config: IQLConfig, device: torch.device = torch.device("cpu")) -> None:
        self.config = config
        self.device = device
        self.q = QNetwork(config.obs_dim, config.action_dim, config.z_dim, config.hidden_dims).to(device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.v = VNetwork(config.obs_dim, config.z_dim, config.hidden_dims).to(device)
        self.policy = TanhGaussianPolicy(
            config.obs_dim, config.action_dim, config.z_dim, config.hidden_dims
        ).to(device)

        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    # -- updates -------------------------------------------------------------------
    def update_value(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> Dict[str, float]:
        """Regress V(s, z) onto the target Q(s, a, z) with an expectile loss."""
        with torch.no_grad():
            q = self.q_target(obs, action, z)
        v = self.v(obs, z)
        loss = expectile_loss(q - v, self.config.expectile).mean()
        self.v_optimizer.zero_grad()
        loss.backward()
        self.v_optimizer.step()
        return {"v_loss": loss.item(), "v_mean": v.mean().item()}

    def update_q(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """Bellman backup Q(s, a, z) <- r + gamma * V(s', z)."""
        with torch.no_grad():
            next_v = self.v(next_obs, z)
            target_q = reward + self.config.discount * (1.0 - done) * next_v
        q = self.q(obs, action, z)
        loss = F.mse_loss(q, target_q)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return {"q_loss": loss.item(), "q_mean": q.mean().item()}

    def update_policy(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> Dict[str, float]:
        """Advantage-weighted regression: weight dataset actions by exp(beta * A(s, a))."""
        with torch.no_grad():
            q = self.q(obs, action, z)
            v = self.v(obs, z)
            adv = q - v
            weight = torch.exp(self.config.awr_temperature * adv)
            weight = torch.clamp(weight, max=self.config.max_advantage_weight)
        log_prob = self.policy.log_prob(obs, z, action).squeeze(-1)
        loss = -(weight * log_prob).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return {"policy_loss": loss.item(), "adv_mean": adv.mean().item()}

    def update_target(self) -> None:
        """Polyak-average the target Q-network."""
        rate = self.config.target_update_rate
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                tp.mul_(1.0 - rate).add_(rate * p)

    # -- inference -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action, _ = self.policy.sample(obs, z, deterministic=deterministic)
        return action

    @torch.no_grad()
    def value(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.v(obs, z)

    @torch.no_grad()
    def q_value(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.q(obs, action, z)

    # -- checkpointing -------------------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "v": self.v.state_dict(),
            "policy": self.policy.state_dict(),
            "config": self.config,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q_target"])
        self.v.load_state_dict(state["v"])
        self.policy.load_state_dict(state["policy"])
