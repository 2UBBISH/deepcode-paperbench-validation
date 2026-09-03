"""Goal-conditioned Implicit Q-Learning (GC-IQL) baseline.

This module reuses the conditional IQL networks from :mod:`fre.rl.iql` but
conditions the value, Q, and policy functions on a goal state ``g`` instead of
the FRE latent ``z``.  The goal-conditioned reward used throughout is the
sparse reaching reward

    r(s, g) = 0       if ``||s - g|| < threshold``
    r(s, g) = -1      otherwise

As in the FRE paper, we use hindsight relabeling during training: a fraction
of training batches are labelled with future states sampled from the same
trajectory as goals, while the remaining fraction retains the original goal
(for D4RL tasks that provide one).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from fre.config import IQLConfig
from fre.rl.iql import IQL, ImplicitQLearning

__all__ = ["GoalConditionedIQL", "GC_IQL", "sparse_goal_reward", "train_gc_iql_agent"]


def sparse_goal_reward(
    states: torch.Tensor,
    goals: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Compute the sparse goal-reaching reward.

    Args:
        states: State tensor of shape ``(..., state_dim)``.
        goals: Goal tensor broadcastable against ``states``.
        threshold: Euclidean-distance threshold for considering a goal reached.

    Returns:
        Reward tensor with shape ``states.shape[:-1]`` containing ``0.0`` for
        reached goals and ``-1.0`` otherwise.
    """
    dist = torch.norm(states - goals, dim=-1)
    reached = dist < threshold
    return torch.where(reached, torch.zeros_like(dist), -torch.ones_like(dist))


def _as_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """Convert an array-like or tensor to a float tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class GoalConditionedIQL:
    """Goal-conditioned IQL agent.

    This is a thin wrapper around :class:`ImplicitQLearning` with
    ``condition_dim = state_dim``.  It additionally handles the sparse
    goal-reaching reward and hindsight-relabelled training batches.

    Parameters:
        state_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        cfg: Optional IQL hyperparameter config.
        threshold: Goal-reaching distance threshold.
        device: Torch device used for network construction.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: Optional[IQLConfig] = None,
        threshold: float = 1.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.threshold = float(threshold)
        self.device = device
        self.agent = ImplicitQLearning(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            condition_dim=self.state_dim,
            cfg=cfg,
            device=device,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------
    def get_action(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample or select an action conditioned on a goal state."""
        condition = goal if goal is not None else state
        return self.agent.get_action(state, condition=condition, deterministic=deterministic)

    def value(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        condition = goal if goal is not None else state
        return self.agent.value(state, condition=condition)

    def q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = goal if goal is not None else state
        return self.agent.q_value(state, action, condition=condition)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(
        self,
        batch: Any,
        goals: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Perform one IQL update conditioned on goals.

        If ``rewards`` is not supplied, the sparse goal-reaching reward is
        computed from ``batch.states`` and the goal tensor.
        """
        if goals is None:
            goals = getattr(batch, "goals", None)
            if goals is None:
                # Fallback: self-conditioning is only a defensive default and
                # should normally be replaced by hindsight relabelling.
                goals = batch.states

        condition = _as_tensor(goals, batch.states.device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(batch.states.shape[0], -1)
        elif condition.shape[0] == 1 and batch.states.shape[0] > 1:
            condition = condition.expand(batch.states.shape[0], -1)

        if rewards is None:
            thresh = self.threshold if threshold is None else float(threshold)
            rewards = sparse_goal_reward(batch.states, condition, threshold=thresh)

        return self.agent.train_step(batch, condition=condition, rewards=rewards)

    def train_hindsight_step(
        self,
        dataset: Any,
        batch_size: int,
        original_goal_prob: float = 0.3,
    ) -> Dict[str, float]:
        """Sample a hindsight-relabelled batch and run one IQL update.

        The dataset object is expected to provide either
        ``sample_hindsight_batch(batch_size, original_goal_prob)`` or
        ``sample_goal_transitions(batch_size, original_goal_prob)``.  If
        neither is available, a defensive fallback builds a batch from
        randomly sampled transitions and randomly sampled goals.
        """
        original_goal_prob = float(original_goal_prob)
        batch = None

        # Preferred dataset primitives for hindsight relabelling.
        for method_name in ("sample_hindsight_batch", "sample_goal_transitions"):
            if hasattr(dataset, method_name):
                try:
                    candidate = getattr(dataset, method_name)(
                        batch_size,
                        original_goal_prob=original_goal_prob,
                    )
                    if candidate is not None:
                        batch = candidate
                        break
                except TypeError:
                    try:
                        candidate = getattr(dataset, method_name)(batch_size)
                        if candidate is not None:
                            batch = candidate
                            break
                    except Exception:  # pragma: no cover - defensive fallback
                        continue
                except Exception:  # pragma: no cover - defensive fallback
                    continue

        if batch is None:
            batch = _fallback_hindsight_batch(dataset, batch_size, original_goal_prob)

        goals = getattr(batch, "goals", None)
        if goals is None:
            goals = batch.states

        return self.train_step(batch, goals=goals)

    def train(
        self,
        dataset: Any,
        num_steps: int = 100_000,
        batch_size: int = 256,
        original_goal_prob: float = 0.3,
        log_every: int = 1000,
    ) -> Dict[str, Any]:
        """Run a full goal-conditioned IQL training loop.

        Returns a dictionary of aggregated training metrics.
        """
        metrics_accum: Dict[str, list] = {"value_loss": [], "q_loss": [], "policy_loss": []}
        last = {}
        for step in range(int(num_steps)):
            last = self.train_hindsight_step(dataset, batch_size, original_goal_prob)
            for key, value in last.items():
                metrics_accum.setdefault(key, []).append(float(value))
            if int(log_every) > 0 and (step + 1) % int(log_every) == 0:
                means = {k: float(np.mean(v[-int(log_every):])) for k, v in metrics_accum.items() if v}
                logging.info("[GC-IQL] step %d: %s", step + 1, means)
        summary = {k: float(np.mean(v)) for k, v in metrics_accum.items() if v}
        summary["last_step"] = last
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the underlying IQL agent checkpoint."""
        self.agent.save(path)

    def load(self, path: str) -> None:
        """Load an IQL agent checkpoint."""
        self.agent.load(path)

    def to(self, device: str) -> "GoalConditionedIQL":
        self.device = device
        self.agent.to(device)
        return self


def _fallback_hindsight_batch(dataset: Any, batch_size: int, original_goal_prob: float):
    """Defensive hindsight batch builder when dataset primitives are absent.

    The returned object is a lightweight namespace with ``states``,
    ``actions``, ``next_states``, ``terminals``, ``timeouts``, and ``goals``.
    Goals are sampled uniformly from the dataset, which is not true hindsight
    but keeps the baseline runnable if dataset-specific helpers are missing.
    """
    from types import SimpleNamespace

    transitions = dataset.sample_transitions(batch_size)
    goals = dataset.sample_states(batch_size)
    if isinstance(goals, tuple):
        goals = goals[0]
    goals = _as_tensor(goals, transitions.states.device)
    return SimpleNamespace(
        states=transitions.states,
        actions=transitions.actions,
        next_states=transitions.next_states,
        rewards=transitions.rewards,
        terminals=transitions.terminals,
        timeouts=getattr(transitions, "timeouts", None),
        goals=goals,
    )


def train_gc_iql_agent(
    dataset: Any,
    cfg: Optional[IQLConfig] = None,
    device: str = "cpu",
    num_steps: int = 100_000,
    batch_size: int = 256,
    original_goal_prob: float = 0.3,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    threshold: float = 1.0,
) -> GoalConditionedIQL:
    """Convenience wrapper for constructing and training GC-IQL.

    State and action dimensionality are inferred from the dataset when not
    explicitly supplied.
    """
    if state_dim is None:
        state_dim = int(dataset.states.shape[-1])
    if action_dim is None:
        action_dim = int(dataset.actions.shape[-1])

    agent = GoalConditionedIQL(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg,
        threshold=threshold,
        device=device,
    )
    agent.train(
        dataset=dataset,
        num_steps=num_steps,
        batch_size=batch_size,
        original_goal_prob=original_goal_prob,
    )
    return agent


# Public alias matching the paper/plan naming.
GC_IQL = GoalConditionedIQL
