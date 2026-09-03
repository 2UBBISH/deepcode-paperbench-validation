"""Forward-Backward (FB) baseline for zero-shot offline RL.

This implementation follows the high-level recipe from the FRE paper:

* Learn a forward representation ``F(s, a, z)`` and a backward/state
  representation ``B(s)`` such that ``F(s, a, z)^T B(s')`` approximates the
  successor measure induced by the policy conditioned on ``z``.
* Train a DDPG-style deterministic policy and twin Q networks on top of the
  learned representation, using the intrinsic reward
  ``F(s, a, z)^T z``.
* At evaluation time, infer the task vector ``w`` by ridge-regressing
  ``B(state)`` onto reward samples (5120 by default) and evaluate
  ``pi(a | s, w)``.

The code is deliberately self-contained and only relies on
:mod:`baselines.baseline_utils`, so it can run with or without MuJoCo/D4RL
available.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.utils import to_torch
from .baseline_utils import (
    DeterministicPolicy,
    TwinQNetwork,
    build_mlp,
    hard_update,
    make_policy_fn_from_net,
    sample_reward_pairs,
    ridge_regression,
    soft_update,
)


class ForwardNet(nn.Module):
    """Forward representation ``F(s, a, z)``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        repr_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim + context_dim
        self.net = build_mlp(
            input_dim,
            hidden_dims,
            output_dim=repr_dim,
            activation=activation,
            output_activation=None,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x)


class BackwardNet(nn.Module):
    """Backward/state representation ``B(s)``."""

    def __init__(
        self,
        state_dim: int,
        repr_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            state_dim,
            hidden_dims,
            output_dim=repr_dim,
            activation=activation,
            output_activation=None,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class FB:
    """Forward-Backward zero-shot RL agent.

    Parameters
    ----------
    state_dim:
        Environment state dimensionality.
    action_dim:
        Environment action dimensionality.
    repr_dim:
        Dimension of the forward/backward representations.
    hidden_dims:
        Hidden sizes for all MLPs.
    lr:
        Adam learning rate.
    gamma:
        Discount factor.
    tau:
        Polyak averaging coefficient for target networks.
    batch_size:
        Default training batch size.
    eval_reward_samples:
        Number of reward samples used for task regression at evaluation.
    ridge_coeff:
        L2 regularization coefficient used for task regression.
    device:
        Torch device.
    max_action:
        Maximum absolute action (used by the deterministic policy).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        repr_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        eval_reward_samples: int = 5120,
        ridge_coeff: float = 1e-3,
        device: str = "cpu",
        max_action: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.repr_dim = repr_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eval_reward_samples = eval_reward_samples
        self.ridge_coeff = ridge_coeff
        self.device = torch.device(device)
        self.max_action = max_action

        self.f_net = ForwardNet(
            state_dim, action_dim, repr_dim, repr_dim, hidden_dims
        ).to(self.device)
        self.b_net = BackwardNet(state_dim, repr_dim, hidden_dims).to(self.device)
        self.policy = DeterministicPolicy(
            state_dim, repr_dim, action_dim, hidden_dims, max_action=max_action
        ).to(self.device)
        self.q_net = TwinQNetwork(
            state_dim, action_dim, repr_dim, hidden_dims
        ).to(self.device)

        self.f_target = copy.deepcopy(self.f_net).to(self.device)
        self.policy_target = copy.deepcopy(self.policy).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)

        self.fb_optimizer = torch.optim.Adam(
            list(self.f_net.parameters()) + list(self.b_net.parameters()), lr=lr
        )
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _sample_z(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.repr_dim, device=self.device)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, ...]:
        """Normalize offline dataset batches to (states, actions, next_states)."""
        if isinstance(batch, dict):
            states = to_torch(batch.get("states", batch.get("observations")), device=None)
            actions = to_torch(batch.get("actions"), device=None)
            next_states = to_torch(
                batch.get("next_states", batch.get("next_observations")), device=None
            )
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                states, actions, next_states = [to_torch(x, device=None) for x in batch]
            elif len(batch) >= 4:
                states, actions, next_states = [to_torch(x, device=None) for x in batch[:3]]
            else:
                raise ValueError(f"Unsupported batch tuple length: {len(batch)}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return states, actions, next_states

    def train_step(self, batch: Any, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Perform one FB/DDPG update from an offline batch."""
        batch_size = batch_size or self.batch_size
        states, actions, next_states = self._unpack_batch(batch)
        states = states.to(self.device).float()
        actions = actions.to(self.device).float()
        next_states = next_states.to(self.device).float()

        # Dataset samplers often return more elements than requested.
        n = min(states.shape[0], batch_size)
        if states.shape[0] > n:
            idx = torch.randperm(states.shape[0], device=self.device)[:n]
            states, actions, next_states = states[idx], actions[idx], next_states[idx]

        z = self._sample_z(states.shape[0])

        # ---- Forward-backward representation loss -------------------------
        f_sa = self.f_net(states, actions, z)
        b_s = self.b_net(states)
        b_snext = self.b_net(next_states)

        with torch.no_grad():
            next_actions = self.policy_target(next_states, z)
            f_next = self.f_target(next_states, next_actions, z)

        fb_next = (f_sa * b_snext).sum(dim=-1)
        fb_next_target = self.gamma * (f_next * b_snext).sum(dim=-1)
        fb_current = (f_sa * b_s).sum(dim=-1)

        fb_loss = F.mse_loss(fb_next, fb_next_target) + F.mse_loss(
            fb_current, torch.ones_like(fb_current)
        )
        # Mild regularization avoids the trivial or exploding solutions that
        # pure Bellman residuals can admit.
        fb_loss = fb_loss + 1e-4 * (f_sa.pow(2).mean() + b_s.pow(2).mean())

        # ---- DDPG-style critic update on the intrinsic value ---------------
        with torch.no_grad():
            r_int = (f_sa.detach() * z).sum(dim=-1, keepdim=True)
            next_actions_for_q = self.policy_target(next_states, z)
            q1_next, q2_next = self.q_target(next_states, next_actions_for_q, z)
            q_next = torch.min(q1_next, q2_next)
            q_target = r_int + self.gamma * q_next

        q1, q2 = self.q_net(states, actions, z)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # ---- Policy update ---------------------------------------------------
        policy_actions = self.policy(states, z)
        policy_q = self.q_net.q1(states, policy_actions, z)
        policy_loss = -policy_q.mean()

        self.fb_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_optimizer.step()

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        soft_update(self.f_target, self.f_net, self.tau)
        soft_update(self.q_target, self.q_net, self.tau)
        soft_update(self.policy_target, self.policy, self.tau)

        return {
            "fb_loss": float(fb_loss.detach().cpu().item()),
            "q_loss": float(q_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "fb_next_mean": float(fb_next.mean().detach().cpu().item()),
            "fb_current_mean": float(fb_current.mean().detach().cpu().item()),
        }

    def get_task_policy(
        self,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        state_pool: np.ndarray,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Infer a zero-shot policy for a downstream reward function.

        Exactly like the FB baseline in the FRE paper, this samples a large set
        of ``(state, reward)`` pairs (5120 by default), fits
        ``w = ridge(B(s), r)``, and conditions the learned policy on ``w``.
        """
        num_samples = num_samples or self.eval_reward_samples
        states, rewards = sample_reward_pairs(
            reward_fn, state_pool, num_samples=num_samples, seed=seed
        )
        states_t = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            features = self.b_net(states_t).cpu().numpy()
        w = ridge_regression(features, rewards, ridge=self.ridge_coeff)
        w_t = torch.from_numpy(w.astype(np.float32)).to(self.device)
        return make_policy_fn_from_net(
            self.policy,
            context=w_t,
            device=self.device,
            deterministic=True,
            state_is_tensor=False,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "f_net": self.f_net.state_dict(),
            "b_net": self.b_net.state_dict(),
            "policy": self.policy.state_dict(),
            "q_net": self.q_net.state_dict(),
            "f_target": self.f_target.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "q_target": self.q_target.state_dict(),
            "fb_optimizer": self.fb_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.f_net.load_state_dict(state_dict["f_net"])
        self.b_net.load_state_dict(state_dict["b_net"])
        self.policy.load_state_dict(state_dict["policy"])
        self.q_net.load_state_dict(state_dict["q_net"])
        if "f_target" in state_dict:
            self.f_target.load_state_dict(state_dict["f_target"])
        if "policy_target" in state_dict:
            self.policy_target.load_state_dict(state_dict["policy_target"])
        if "q_target" in state_dict:
            self.q_target.load_state_dict(state_dict["q_target"])
        if "fb_optimizer" in state_dict:
            self.fb_optimizer.load_state_dict(state_dict["fb_optimizer"])
        if "q_optimizer" in state_dict:
            self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        if "policy_optimizer" in state_dict:
            self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])

    def to(self, device: str) -> "FB":
        self.device = torch.device(device)
        self.f_net.to(self.device)
        self.b_net.to(self.device)
        self.policy.to(self.device)
        self.q_net.to(self.device)
        self.f_target.to(self.device)
        self.policy_target.to(self.device)
        self.q_target.to(self.device)
        return self
