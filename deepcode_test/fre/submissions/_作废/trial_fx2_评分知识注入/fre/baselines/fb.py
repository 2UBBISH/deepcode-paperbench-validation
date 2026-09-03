"""Forward-Backward (FB) baseline for zero-shot offline reinforcement learning.

This module provides a self-contained implementation of the Forward-Backward
representation family from Touati et al. (2022).  The learned representation is
used in the same zero-shot reward-regression protocol as the paper's other
baselines: given a handful of labelled ``(state, reward)`` examples, we regress a
task vector ``z`` on the backward features ``B(s)`` and then act greedily with
respect to the learned forward (successor-measure) representation.

The implementation follows the universal successor-measure formulation:

    Q^pi(s, a, z) ~= <F(s, a), z>,

where ``F(s, a)`` is the forward representation and ``z`` is obtained at test
time by solving

    min_z E_{s ~ D}[(<B(s), z> - r(s))^2] + lambda * ||z||^2.

Training is DDPG-style: for randomly sampled unit vectors ``z`` we minimize the
Bellman error of ``<F(s, a), z>`` and update a deterministic policy to maximize
it.  By default the backward features ``B(s)`` are a fixed random MLP, which
keeps the reward regression well-conditioned and gives a stable universal
successor-feature policy; this matches the FB protocol used in the FRE paper
while keeping the adapter dependency-light.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.rl.networks import DeterministicPolicy, soft_update

__all__ = ["ForwardBackward", "FB", "train_fb_agent"]

logger = logging.getLogger(__name__)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(activation())
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ForwardBackward(nn.Module):
    """DDPG-based forward-backward successor-measure baseline.

    Parameters
    ----------
    state_dim:
        Dimensionality of the observation/state space.
    action_dim:
        Dimensionality of the action space.
    feature_dim:
        Dimensionality of the backward/forward feature space ``d``.
    hidden_dim:
        Hidden width for all MLP modules.
    num_hidden:
        Number of hidden layers for all MLP modules.
    lr:
        Learning rate for the forward and policy optimizers.
    gamma:
        Discount factor.
    tau:
        Polyak averaging coefficient for target networks.
    device:
        Torch device string or object.
    learn_backward:
        If ``True``, the backward network ``B(s)`` is trained jointly with the
        forward representation (closer to the full FB algorithm).  If ``False``,
        it is a fixed random projection, which is a simpler and more stable
        reward-regression basis.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        learn_backward: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.tau = tau
        self.learn_backward = learn_backward
        self.device = torch.device(device)
        self._z: Optional[torch.Tensor] = None

        self.backward = _build_mlp(
            state_dim, feature_dim, hidden_dim=hidden_dim, num_hidden=num_hidden
        )
        # Keep backward features stationary unless explicitly trained.
        if not learn_backward:
            for p in self.backward.parameters():
                p.requires_grad_(False)

        self.forward = _build_mlp(
            state_dim + action_dim, feature_dim, hidden_dim=hidden_dim, num_hidden=num_hidden
        )
        self.target_forward = _build_mlp(
            state_dim + action_dim, feature_dim, hidden_dim=hidden_dim, num_hidden=num_hidden
        )
        self.target_forward.load_state_dict(self.forward.state_dict())
        self.target_forward.requires_grad_(False)

        # Deterministic DDPG-style actor conditioned on the task vector z.
        self.actor = DeterministicPolicy(
            state_dim,
            action_dim,
            condition_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            squashing=True,
        )
        self.target_actor = DeterministicPolicy(
            state_dim,
            action_dim,
            condition_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            squashing=True,
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.requires_grad_(False)

        if self.learn_backward:
            params = list(self.forward.parameters()) + list(self.backward.parameters())
        else:
            params = list(self.forward.parameters())
        self.forward_optimizer = torch.optim.Adam(params, lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.to(self.device)

    def _sample_z(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.feature_dim, device=self.device)
        z = F.normalize(z, dim=-1)
        return z

    def _backward_features(self, states: torch.Tensor) -> torch.Tensor:
        return self.backward(states)

    def _forward_features(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(torch.cat([states, actions], dim=-1))

    def _target_forward_features(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.target_forward(torch.cat([states, actions], dim=-1))

    def q_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar universal Q-value ``<F(s,a), z>``."""
        z = self._resolve_z(z, states.shape[0])
        f = self._forward_features(states, actions)
        return torch.einsum("bd,bd->b", f, z)

    def _resolve_z(self, z: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if z is None:
            if self._z is None:
                raise ValueError(
                    "No task vector available; call fit_reward() first or pass z explicitly."
                )
            z = self._z
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(batch_size, -1)
        elif z.shape[0] == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)
        elif z.shape[0] != batch_size:
            raise ValueError(
                f"Task vector batch size {z.shape[0]} does not match state batch size {batch_size}"
            )
        return z

    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Return the deterministic action ``pi(s, z)``.

        ``condition`` is the task vector returned by :meth:`fit_reward`.
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        z = self._resolve_z(condition, state.shape[0])
        return self.actor.get_action(state, condition=z)

    def fit_reward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        reg: float = 1e-3,
    ) -> torch.Tensor:
        """Regress a task vector ``z`` from labelled reward examples.

        Parameters
        ----------
        states:
            Tensor of shape ``(N, state_dim)``.
        rewards:
            Tensor of shape ``(N,)``.
        reg:
            L2 regularization coefficient for the normal equations.
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        rewards = rewards.reshape(-1, 1)

        with torch.no_grad():
            phi = self._backward_features(states)  # [N, d]
        gram = phi.t() @ phi
        eye = torch.eye(self.feature_dim, device=self.device)
        rhs = phi.t() @ rewards
        z = torch.linalg.solve(gram + reg * eye, rhs).squeeze(-1)
        self._z = z.detach()
        return self._z

    def train_step(
        self,
        batch: Any,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One DDPG-style FB update.

        ``batch`` must expose ``states``, ``actions``, ``next_states`` and
        optionally ``terminals`` attributes (the repository's
        :class:`~fre.data.dataset.TransitionBatch` does).
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        next_states = batch.next_states.to(self.device)
        terminals = getattr(batch, "terminals", None)
        if terminals is not None:
            terminals = terminals.to(self.device)
        else:
            terminals = torch.zeros(states.shape[0], device=self.device)

        batch_size = states.shape[0]
        if z is None:
            z = self._sample_z(batch_size)
        else:
            z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
            if z.dim() == 1:
                z = z.unsqueeze(0).expand(batch_size, -1)
            elif z.shape[0] == 1 and batch_size > 1:
                z = z.expand(batch_size, -1)
            z = F.normalize(z, dim=-1)

        with torch.no_grad():
            next_actions = self.target_actor.get_action(next_states, condition=z)
            target_f = self._target_forward_features(next_states, next_actions)
            reward = torch.einsum("bd,bd->b", self._backward_features(states), z)
            target_q = torch.einsum("bd,bd->b", target_f, z)
            not_done = (1.0 - terminals).to(torch.float32)
            target = reward + self.gamma * not_done * target_q

        f = self._forward_features(states, actions)
        q = torch.einsum("bd,bd->b", f, z)
        forward_loss = F.mse_loss(q, target)

        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()

        # Policy update: maximize <F(s, pi(s,z)), z>.
        policy_actions = self.actor.get_action(states, condition=z)
        policy_f = self._forward_features(states, policy_actions)
        policy_q = torch.einsum("bd,bd->b", policy_f, z)
        policy_loss = -policy_q.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_forward, self.forward, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

        return {
            "forward_loss": float(forward_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "mean_q": float(q.detach().mean().cpu().item()),
            "z_norm": float(z.norm(dim=-1).mean().detach().cpu().item()),
        }

    def train(
        self,
        dataset: Any,
        num_steps: int = 100_000,
        batch_size: int = 256,
        log_every: int = 1000,
    ) -> Dict[str, Any]:
        """Run the offline FB training loop over a dataset.

        The dataset must implement ``sample_transitions(batch_size)``.
        """
        metrics: list[Dict[str, float]] = []
        for step in range(1, num_steps + 1):
            batch = dataset.sample_transitions(batch_size)
            if hasattr(batch, "to"):
                batch = batch.to(self.device)
            info = self.train_step(batch)
            metrics.append(info)
            if step % log_every == 0:
                avg = {k: float(np.mean([m[k] for m in metrics[-log_every:]])) for k in info}
                logger.info("FB step %d: %s", step, avg)
        if not metrics:
            return {"mean_metrics": {}, "last_metrics": {}}
        last = metrics[-1]
        mean = {k: float(np.mean([m[k] for m in metrics])) for k in last}
        return {"mean_metrics": mean, "last_metrics": last}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)

    def to(self, device: Any) -> "ForwardBackward":
        self.device = torch.device(device)
        return super().to(self.device)


FB = ForwardBackward


def train_fb_agent(
    dataset: Any,
    cfg: Optional[Any] = None,
    device: str = "cpu",
    num_steps: int = 100_000,
    batch_size: int = 256,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    feature_dim: int = 256,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    learn_backward: bool = False,
) -> ForwardBackward:
    """Construct and train a Forward-Backward baseline agent."""
    if state_dim is None:
        state_dim = dataset.states.shape[-1]
    if action_dim is None:
        action_dim = dataset.actions.shape[-1]
    if cfg is not None:
        if hasattr(cfg, "baseline") and cfg.baseline is not None:
            bcfg = cfg.baseline
            feature_dim = int(getattr(bcfg, "fb_feature_dim", feature_dim))
            hidden_dim = int(getattr(bcfg, "fb_hidden_dim", hidden_dim))
            num_hidden = int(getattr(bcfg, "fb_num_hidden", num_hidden))
            lr = float(getattr(bcfg, "fb_lr", lr))
            gamma = float(getattr(bcfg, "gamma", gamma))
            tau = float(getattr(bcfg, "target_tau", tau))
            learn_backward = bool(getattr(bcfg, "fb_learn_backward", learn_backward))

    agent = ForwardBackward(
        state_dim=state_dim,
        action_dim=action_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        lr=lr,
        gamma=gamma,
        tau=tau,
        device=device,
        learn_backward=learn_backward,
    )
    agent.train(dataset, num_steps=num_steps, batch_size=batch_size)
    return agent
