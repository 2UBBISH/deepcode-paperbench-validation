"""Implicit Q-Learning (IQL) losses and trainer for FRE-conditioned offline RL.

This module implements the three IQL objectives from the paper:

    L_V   = E[ L2^tau(Q_target(s, a, z) - V(s, z)) ]
    L_Q   = E[ (r(s) + gamma * V(s', z) - Q(s, a, z))^2 ]
    L_pi  = -E[ exp(beta * (Q(s, a, z) - V(s, z))) * log pi(a | s, z) ]

where ``z`` is either the FRE latent reward encoding or a goal vector, and the
expectile loss is::

    L2^tau(u) = |tau - 1{u < 0}| * u^2

All three networks are conditioned on the conditioning vector by concatenation.
The implementation is intentionally simple and mirrors the paper's single-Q
variant rather than the double-Q variant used in some other IQL codebases.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Union

import torch
import torch.nn as nn

from fre.config import IQLConfig
from fre.rl.networks import (
    GaussianPolicy,
    QNetwork,
    ValueNetwork,
    soft_update,
)

__all__ = ["expectile_loss", "IQL", "ImplicitQLearning"]


def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Compute the asymmetric squared expectile loss elementwise.

    Args:
        diff: Tensor of differences ``u`` (typically ``Q_target - V``).
        tau: Expectile parameter in ``(0, 1)``. ``tau=0.5`` recovers MSE,
            ``tau > 0.5`` penalizes negative differences more heavily, which
            biases the value function towards the upper expectile of Q.

    Returns:
        Elementwise loss with the same shape as ``diff``.
    """
    weight = torch.where(diff < 0.0, torch.tensor(1.0 - tau, device=diff.device),
                         torch.tensor(tau, device=diff.device))
    return weight * (diff ** 2)


def _cfg_value(cfg: Optional[IQLConfig], key: str, default):
    """Read a value from an IQLConfig with a safe fallback.

    The config dataclass may evolve between paper versions, so we also check a
    few common alias names. This keeps the trainer functional even when some
    fields use slightly different names.
    """
    if cfg is None:
        return default
    aliases = {
        "lr": ["lr", "learning_rate", "policy_lr"],
        "tau": ["tau", "expectile"],
        "beta": ["beta", "temperature", "advantage_temperature"],
        "gamma": ["gamma", "discount"],
        "target_tau": ["target_tau", "target_update_rate", "polyak_tau"],
        "hidden_dim": ["hidden_dim", "hidden_size", "network_hidden_dim"],
        "num_hidden": ["num_hidden", "num_layers", "network_depth"],
    }
    names = aliases.get(key, [key])
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value is not None:
                return value
    return default


class ImplicitQLearning(nn.Module):
    """Implicit Q-Learning agent with conditional value/Q/policy networks.

    Parameters
    ----------
    state_dim:
        Dimension of environment states.
    action_dim:
        Dimension of environment actions.
    condition_dim:
        Dimension of the conditioning vector ``z`` (or goal vector). Use 0 for
        an unconditioned agent.
    cfg:
        Optional :class:`IQLConfig`. Any missing field falls back to the paper
        defaults below.
    gamma:
        Discount factor.
    tau:
        Expectile parameter for the value update.
    beta:
        Advantage temperature for policy extraction.
    lr:
        Adam learning rate for all IQL networks.
    target_tau:
        Polyak averaging coefficient for the target Q network.
    hidden_dim:
        Width of hidden layers.
    num_hidden:
        Number of hidden layers in each network.
    device:
        Torch device string or object.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        condition_dim: int = 0,
        cfg: Optional[IQLConfig] = None,
        gamma: float = 0.99,
        tau: float = 0.7,
        beta: float = 3.0,
        lr: float = 3e-4,
        target_tau: float = 0.005,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.device = torch.device(device)

        self.gamma = float(_cfg_value(cfg, "gamma", gamma))
        self.tau = float(_cfg_value(cfg, "tau", tau))
        self.beta = float(_cfg_value(cfg, "beta", beta))
        self.target_tau = float(_cfg_value(cfg, "target_tau", target_tau))
        self.learning_rate = float(_cfg_value(cfg, "lr", lr))
        self.hidden_dim = int(_cfg_value(cfg, "hidden_dim", hidden_dim))
        self.num_hidden = int(_cfg_value(cfg, "num_hidden", num_hidden))

        # Networks.
        self.v_network = ValueNetwork(
            state_dim=state_dim,
            condition_dim=condition_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )
        self.q_target = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )

        self.optimizer = torch.optim.Adam(
            list(self.v_network.parameters())
            + list(self.q_network.parameters())
            + list(self.policy.parameters()),
            lr=self.learning_rate,
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def compute_value_loss(self, batch, condition: torch.Tensor) -> torch.Tensor:
        """Compute the expectile value loss."""
        with torch.no_grad():
            target_q = self.q_target(batch.states, batch.actions, condition)
        v = self.v_network(batch.states, condition)
        diff = target_q - v
        return expectile_loss(diff, self.tau).mean()

    def compute_q_loss(
        self,
        batch,
        condition: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Bellman Q loss.

        If ``rewards`` is not supplied, ``batch.rewards`` is used. A terminal
        mask is applied using ``batch.terminals`` when available so that
        terminal transitions do not bootstrap future value.
        """
        if rewards is None:
            rewards = batch.rewards
        if rewards is None:
            raise ValueError("Q loss requires rewards or batch.rewards.")

        rewards = rewards.float().view(batch.states.shape[0], 1)
        with torch.no_grad():
            next_v = self.v_network(batch.next_states, condition)

        if getattr(batch, "terminals", None) is not None:
            not_done = (1.0 - batch.terminals.float()).view(batch.states.shape[0], 1)
        else:
            not_done = torch.ones_like(rewards)

        target = rewards + self.gamma * not_done * next_v
        q = self.q_network(batch.states, batch.actions, condition)
        return ((q - target) ** 2).mean()

    def compute_policy_loss(self, batch, condition: torch.Tensor) -> torch.Tensor:
        """Compute the advantage-weighted policy extraction loss."""
        with torch.no_grad():
            q = self.q_network(batch.states, batch.actions, condition)
            v = self.v_network(batch.states, condition)
            advantage = q - v
            weights = torch.exp(self.beta * advantage).clamp_max(100.0)

        log_prob = self.policy.log_prob(batch.states, batch.actions, condition)
        log_prob = log_prob.view(batch.states.shape[0], 1)
        return -(weights * log_prob).mean()

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def train_step(
        self,
        batch,
        condition: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> dict:
        """Perform one full IQL update.

        Args:
            batch: A transition batch with ``states``, ``actions``,
                ``next_states`` and, unless ``rewards`` is supplied,
                ``rewards``.
            condition: Conditioning vectors of shape ``[B, condition_dim]``.
            rewards: Optional precomputed rewards of shape ``[B]`` or ``[B, 1]``.

        Returns:
            Dictionary containing the scalar losses for logging.
        """
        self.v_network.train()
        self.q_network.train()
        self.policy.train()

        # Value update.
        value_loss = self.compute_value_loss(batch, condition)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # Q update (target Q is already detached inside the loss).
        q_loss = self.compute_q_loss(batch, condition, rewards=rewards)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # Policy update (Q and V are detached inside the loss).
        policy_loss = self.compute_policy_loss(batch, condition)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Polyak target update.
        soft_update(self.q_target, self.q_network, self.target_tau)

        return {
            "value_loss": value_loss.detach().cpu().item(),
            "q_loss": q_loss.detach().cpu().item(),
            "policy_loss": policy_loss.detach().cpu().item(),
        }

    def update(self, batch, condition, rewards=None):
        """Alias for :meth:`train_step` to match trainer naming."""
        return self.train_step(batch, condition, rewards=rewards)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample an action from the policy conditioned on ``condition``."""
        self.policy.eval()
        return self.policy.get_action(
            state, condition=condition, deterministic=deterministic
        )

    @torch.no_grad()
    def value(self, state: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """Return V(s, condition) for visualization/analysis."""
        self.v_network.eval()
        return self.v_network(state, condition)

    @torch.no_grad()
    def q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        """Return Q(s, a, condition) for visualization/analysis."""
        self.q_network.eval()
        return self.q_network(state, action, condition)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the full agent state (networks, target, optimizer)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "condition_dim": self.condition_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "beta": self.beta,
            "target_tau": self.target_tau,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load a checkpoint previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"condition_dim={self.condition_dim}, gamma={self.gamma}, "
            f"tau={self.tau}, beta={self.beta}, lr={self.learning_rate}"
        )


# Public alias used throughout the repository.
IQL = ImplicitQLearning
