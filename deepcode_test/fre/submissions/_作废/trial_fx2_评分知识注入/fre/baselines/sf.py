"""Successor Features (SF) baseline for zero-shot offline RL.

This module provides a self-contained SF implementation using Intrinsic
Curiosity Module (ICM) state features, as referenced in the FRE paper's
baseline comparison.  The method learns a universal successor-feature
representation ``Psi(s, a)`` with DDPG-style updates.  At evaluation time a
task-specific linear reward model is regressed from a modest number of
labelled reward examples, and the policy is conditioned on the resulting
task vector.

The public interface mirrors :class:`fre.baselines.fb.ForwardBackward` so the
unified baseline evaluator can use the same reward-regression protocol.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.rl.networks import DeterministicPolicy, soft_update

logger = logging.getLogger(__name__)

__all__ = ["SuccessorFeatures", "SF", "train_sf_agent"]


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """Defensively read a baseline configuration field.

    Configuration objects may be nested dataclasses, plain dictionaries, or
    ``None``.  This helper accepts all of those forms and falls back to the
    supplied default when the field is absent.
    """

    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    # Some configurations nest baseline fields under ``cfg.baseline``; the
    # caller may pass that sub-object directly, in which case the above
    # branch already succeeded.  Otherwise try common nested access.
    if hasattr(cfg, "baseline"):
        sub = getattr(cfg, "baseline")
        if isinstance(sub, dict) and key in sub:
            return sub[key]
        if hasattr(sub, key):
            return getattr(sub, key)
    return default


def _infer_dims(dataset: Any) -> Tuple[int, int]:
    """Return ``(state_dim, action_dim)`` for a dataset-like object."""

    states = getattr(dataset, "states", None)
    actions = getattr(dataset, "actions", None)

    if states is not None and actions is not None:
        state_dim = int(states.shape[-1])
        action_dim = int(actions.shape[-1])
        return state_dim, action_dim

    # Fallback to sampling a single transition and inspecting its shapes.
    batch = dataset.sample_transitions(1)
    state_dim = int(batch.states.shape[-1])
    action_dim = int(batch.actions.shape[-1])
    return state_dim, action_dim


class _MLP(nn.Module):
    """Small fully connected network with ReLU hidden activations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ICMFeatures(nn.Module):
    """Intrinsic Curiosity Module used as the SF feature extractor.

    The module learns a state representation ``phi(s)`` by jointly training
    an inverse dynamics head (predict the action from ``phi(s), phi(s')``)
    and a forward dynamics head (predict ``phi(s')`` from ``phi(s), a``).
    After pretraining, ``phi`` is frozen and used as the successor features
    basis.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = device

        self.phi = _MLP(state_dim, feature_dim, hidden_dim, num_hidden)
        self.inverse_head = _MLP(2 * feature_dim, action_dim, hidden_dim, 2)
        self.forward_head = _MLP(
            feature_dim + action_dim, feature_dim, hidden_dim, 2
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def features(self, states: torch.Tensor) -> torch.Tensor:
        """Return learned ICM state features ``phi(s)``."""

        return self.phi(states)

    def train_step(self, batch: Any) -> Dict[str, float]:
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        next_states = batch.next_states.to(self.device)

        phi_s = self.phi(states)
        phi_s_next = self.phi(next_states)

        pred_actions = self.inverse_head(torch.cat([phi_s, phi_s_next], dim=-1))
        inverse_loss = F.mse_loss(pred_actions, actions)

        pred_next_phi = self.forward_head(
            torch.cat([phi_s, actions], dim=-1)
        )
        forward_loss = F.mse_loss(pred_next_phi, phi_s_next.detach())

        loss = inverse_loss + forward_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        return {
            "icm_loss": float(loss.detach().cpu().item()),
            "icm_inverse_loss": float(inverse_loss.detach().cpu().item()),
            "icm_forward_loss": float(forward_loss.detach().cpu().item()),
        }

    def pretrain(
        self,
        dataset: Any,
        num_steps: int,
        batch_size: int = 256,
        log_every: int = 1000,
    ) -> Dict[str, Any]:
        self.train()
        history: list[Dict[str, float]] = []
        for step in range(num_steps):
            batch = dataset.sample_transitions(batch_size)
            metrics = self.train_step(batch)
            history.append(metrics)
            if log_every and (step % log_every == 0 or step == num_steps - 1):
                logger.info("ICM pretrain %d/%d %s", step + 1, num_steps, metrics)

        # Freeze the learned feature extractor before successor training.
        for p in self.phi.parameters():
            p.requires_grad_(False)
        self.phi.eval()

        mean_metrics = {
            k: float(np.mean([m[k] for m in history]))
            for k in history[0]
        } if history else {}
        return {"mean_metrics": mean_metrics, "last_metrics": history[-1] if history else {}}


class SuccessorFeatures(nn.Module):
    """Universal successor-feature policy trained with DDPG-style updates.

    Parameters match the forward-backward baseline where appropriate to keep
    comparison conditions as similar as possible.
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
        reg: float = 1e-3,
        icm_hidden_dim: int = 256,
        icm_num_hidden: int = 2,
        icm_lr: float = 3e-4,
        icm_pretrain_steps: int = 5_000,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.tau = tau
        self.reg = reg
        self.device = device
        self.icm_pretrain_steps = icm_pretrain_steps

        self.icm = ICMFeatures(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=icm_hidden_dim,
            num_hidden=icm_num_hidden,
            lr=icm_lr,
            device=device,
        )

        # Successor-feature vector network Psi(s, a).
        self.psi = _MLP(
            state_dim + action_dim, feature_dim, hidden_dim, num_hidden
        )
        self.psi_target = copy.deepcopy(self.psi)
        self.psi_target.requires_grad_(False)

        # Deterministic DDPG actor conditioned on the task vector.
        self.actor = DeterministicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            condition_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.requires_grad_(False)

        self.psi_optimizer = torch.optim.Adam(self.psi.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.to(device)

    def _batch_device_tensors(self, batch: Any) -> Tuple[torch.Tensor, ...]:
        return (
            batch.states.to(self.device),
            batch.actions.to(self.device),
            batch.next_states.to(self.device),
        )

    def features(self, states: torch.Tensor) -> torch.Tensor:
        return self.icm.features(states)

    def q_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``z^T Psi(s, a)``.

        If ``z`` is not provided, return the vector-valued successor features
        themselves.
        """

        psi = self.psi(torch.cat([states, actions], dim=-1))
        if z is None:
            return psi
        return (z * psi).sum(dim=-1)

    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Return the deterministic policy action conditioned on ``condition``.

        ``condition`` is the task vector estimated by :meth:`fit_reward`.
        """

        if condition is None:
            condition = torch.zeros(
                state.shape[:-1] + (self.feature_dim,),
                device=state.device,
                dtype=state.dtype,
            )
        return self.actor.get_action(state, condition=condition)

    def fit_reward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Regress a linear reward model ``w^T phi(s)`` using ridge regression.

        Returns the task vector used by the policy at evaluation time.
        """

        self.eval()
        reg = self.reg if reg is None else reg
        with torch.no_grad():
            phi = self.features(states.to(self.device))
            # phi: [N, D], rewards: [N]
            A = phi.t() @ phi + reg * torch.eye(
                self.feature_dim, device=phi.device, dtype=phi.dtype
            )
            b = phi.t() @ rewards.to(self.device).to(phi.dtype)
            w = torch.linalg.solve(A, b)
        return w

    def train_step(
        self, batch: Any, z: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Perform one successor-feature / policy update."""

        states, actions, next_states = self._batch_device_tensors(batch)
        batch_size = states.shape[0]

        # Freeze the ICM basis; it is only updated during pretraining.
        self.icm.eval()
        for p in self.icm.parameters():
            p.requires_grad_(False)

        if z is None:
            z = torch.randn(
                batch_size, self.feature_dim, device=self.device
            )
            z = F.normalize(z, dim=-1)
        else:
            z = z.to(self.device)
            if z.ndim == 1:
                z = z.unsqueeze(0).expand(batch_size, -1)

        terminals = getattr(batch, "terminals", None)
        if terminals is None:
            terminals = torch.zeros_like(states[:, 0])
        terminals = terminals.to(self.device)
        if terminals.ndim == 2:
            terminals = terminals[:, 0]

        with torch.no_grad():
            phi_s = self.icm.features(states)
            next_actions = self.actor_target(next_states, condition=z)
            next_psi = self.psi_target(
                torch.cat([next_states, next_actions], dim=-1)
            )
            not_done = (1.0 - terminals.float()).unsqueeze(-1)
            td_target = phi_s + self.gamma * not_done * next_psi

        psi = self.psi(torch.cat([states, actions], dim=-1))
        psi_loss = F.mse_loss(psi, td_target.detach())

        self.psi_optimizer.zero_grad()
        psi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.psi.parameters(), 10.0)
        self.psi_optimizer.step()

        policy_actions = self.actor(states, condition=z.detach())
        policy_q = (z.detach() * self.psi(
            torch.cat([states, policy_actions], dim=-1)
        )).sum(dim=-1)
        actor_loss = -policy_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        soft_update(self.psi_target, self.psi, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        return {
            "psi_loss": float(psi_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "policy_q": float(policy_q.detach().mean().cpu().item()),
        }

    def train(
        self,
        dataset: Any,
        num_steps: int = 100_000,
        batch_size: int = 256,
        log_every: int = 1_000,
    ) -> Dict[str, Any]:
        """Pretrain ICM features and then train the universal SF policy."""

        if self.icm_pretrain_steps > 0:
            logger.info("Pretraining ICM features for %d steps", self.icm_pretrain_steps)
            self.icm.pretrain(
                dataset,
                num_steps=self.icm_pretrain_steps,
                batch_size=batch_size,
                log_every=log_every,
            )

        self.train()
        history: list[Dict[str, float]] = []
        for step in range(num_steps):
            batch = dataset.sample_transitions(batch_size)
            metrics = self.train_step(batch)
            history.append(metrics)
            if log_every and (step % log_every == 0 or step == num_steps - 1):
                logger.info("SF train %d/%d %s", step + 1, num_steps, metrics)

        mean_metrics = {
            k: float(np.mean([m[k] for m in history]))
            for k in history[0]
        } if history else {}
        return {"mean_metrics": mean_metrics, "last_metrics": history[-1] if history else {}}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "feature_dim": self.feature_dim,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "tau": self.tau,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict)

    def to(self, device: Any) -> "SuccessorFeatures":
        self.device = str(device).split(":")[0] if not isinstance(device, str) else device
        return super().to(device)


# Public alias, matching the plan's naming conventions.
SF = SuccessorFeatures


def train_sf_agent(
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
    icm_pretrain_steps: int = 5_000,
) -> SuccessorFeatures:
    """Create and train a Successor Features agent.

    Hyperparameters are first read from ``cfg.baseline`` when available, then
    overridden by explicit function arguments.
    """

    state_dim = int(
        state_dim if state_dim is not None else _infer_dims(dataset)[0]
    )
    action_dim = int(
        action_dim if action_dim is not None else _infer_dims(dataset)[1]
    )

    feature_dim = int(_cfg_value(cfg, "sf_feature_dim", feature_dim))
    hidden_dim = int(_cfg_value(cfg, "sf_hidden_dim", hidden_dim))
    num_hidden = int(_cfg_value(cfg, "sf_num_hidden", num_hidden))
    lr = float(_cfg_value(cfg, "sf_lr", lr))
    gamma = float(_cfg_value(cfg, "gamma", gamma))
    tau = float(_cfg_value(cfg, "target_tau", tau))
    icm_pretrain_steps = int(
        _cfg_value(cfg, "sf_icm_pretrain_steps", icm_pretrain_steps)
    )

    agent = SuccessorFeatures(
        state_dim=state_dim,
        action_dim=action_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        lr=lr,
        gamma=gamma,
        tau=tau,
        device=device,
        icm_pretrain_steps=icm_pretrain_steps,
    )
    agent.train(dataset, num_steps=num_steps, batch_size=batch_size)
    return agent
