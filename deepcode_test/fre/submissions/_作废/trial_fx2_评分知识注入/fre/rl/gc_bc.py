"""Goal-conditioned behavioral cloning (GC-BC) baseline.

This module implements the goal-conditioned behavioral cloning baseline used in
the FRE paper comparisons. It trains a Gaussian policy ``pi(a | s, g)`` by
maximizing the log-likelihood of actions observed in the offline dataset.
Goals are obtained via hindsight relabeling: a future state from the same
trajectory is used as the conditioning goal, while the original dataset goal
is retained with a fixed probability.

The policy architecture matches the FRE policy architecture for fairness:
MLP with hidden size 256, two hidden layers, diagonal Gaussian output, and
tanh squashing for bounded action spaces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from fre.config import IQLConfig
from fre.data.dataset import OfflineDataset, TransitionBatch
from fre.rl.networks import GaussianPolicy, DeterministicPolicy


logger = logging.getLogger(__name__)


class GoalConditionedBC(nn.Module):
    """Goal-conditioned behavioral cloning agent.

    Parameters
    ----------
    state_dim:
        Dimension of the state/observation space.
    action_dim:
        Dimension of the action space.
    cfg:
        Optional IQLConfig/BaseConfig-like object used only to infer network
        width/depth and learning-rate defaults if not explicitly provided.
    hidden_dim:
        Width of each hidden MLP layer.
    num_hidden:
        Number of hidden MLP layers.
    lr:
        Learning rate for the policy optimizer.
    device:
        Torch device string.
    deterministic_policy:
        If True, use a deterministic policy (MSE loss) instead of a Gaussian
        policy (negative log-likelihood loss). Default is False (Gaussian).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: Optional[IQLConfig] = None,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        lr: float = 3e-4,
        device: str = "cpu",
        deterministic_policy: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        if cfg is not None:
            hidden_dim = getattr(cfg, "hidden_dim", hidden_dim) or hidden_dim
            num_hidden = getattr(cfg, "num_hidden", num_hidden) or num_hidden
            lr = getattr(cfg, "lr", lr) or lr

        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.lr = lr

        condition_dim = state_dim  # goal vector has same dimensionality as state
        if deterministic_policy:
            self.policy = DeterministicPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                condition_dim=condition_dim,
                hidden_dim=hidden_dim,
                num_hidden=num_hidden,
            )
            self.deterministic_policy = True
        else:
            self.policy = GaussianPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                condition_dim=condition_dim,
                hidden_dim=hidden_dim,
                num_hidden=num_hidden,
            )
            self.deterministic_policy = False

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.to(device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def get_action(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return an action for state conditioned on goal."""
        state = state.to(self.device)
        if goal is not None:
            goal = goal.to(self.device)
        if self.deterministic_policy:
            return self.policy.get_action(state, condition=goal)
        return self.policy.get_action(state, condition=goal, deterministic=deterministic)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(
        self,
        batch: TransitionBatch,
        goals: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Perform one supervised BC update on a provided transition batch.

        If ``goals`` is None, ``batch.goals`` is used. If neither is available,
        states are used as a fallback conditioning signal.
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)

        if goals is not None:
            cond = goals.to(self.device)
        elif getattr(batch, "goals", None) is not None:
            cond = batch.goals.to(self.device)
        else:
            cond = states

        # Broadcast a single goal to the full batch if necessary.
        if cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(states.shape[0], -1)
        elif cond.shape[0] != states.shape[0]:
            raise ValueError(
                f"Goal batch size {cond.shape[0]} does not match state batch size "
                f"{states.shape[0]}"
            )

        if self.deterministic_policy:
            pred_actions = self.policy(states, condition=cond)
            loss = nn.functional.mse_loss(pred_actions, actions)
        else:
            # Negative log-likelihood of the observed actions under the Gaussian policy.
            log_prob = self.policy.log_prob(states, actions, condition=cond)
            loss = -log_prob.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"bc_loss": float(loss.detach().cpu().item())}

    def _fallback_hindsight_batch(
        self,
        dataset: OfflineDataset,
        batch_size: int,
    ) -> TransitionBatch:
        """Fallback when the dataset has no native hindsight sampler.

        Samples random transitions and pairs each state with a randomly sampled
        dataset state. This is not true hindsight relabeling, but preserves
        training stability and is only used for datasets lacking episode
        boundary information.
        """
        batch = dataset.sample_transitions(batch_size)
        goals = dataset.sample_states(batch_size)
        if not isinstance(goals, torch.Tensor):
            goals = torch.as_tensor(goals, dtype=torch.float32, device=self.device)
        batch.goals = goals.to(self.device)
        return batch

    def train_hindsight_step(
        self,
        dataset: OfflineDataset,
        batch_size: int = 256,
        original_goal_prob: float = 0.3,
    ) -> Dict[str, float]:
        """Sample a hindsight-relabeled batch and perform one BC update."""
        if hasattr(dataset, "sample_hindsight_batch"):
            try:
                batch = dataset.sample_hindsight_batch(
                    batch_size, original_goal_prob=original_goal_prob
                )
            except TypeError:
                batch = dataset.sample_hindsight_batch(batch_size)
        elif hasattr(dataset, "sample_goal_transitions"):
            try:
                batch = dataset.sample_goal_transitions(
                    batch_size, original_goal_prob=original_goal_prob
                )
            except TypeError:
                batch = dataset.sample_goal_transitions(batch_size)
        else:
            batch = self._fallback_hindsight_batch(dataset, batch_size)

        return self.train_step(batch)

    def train(
        self,
        dataset: OfflineDataset,
        num_steps: int = 100_000,
        batch_size: int = 256,
        original_goal_prob: float = 0.3,
        log_every: int = 1000,
    ) -> Dict[str, Any]:
        """Run the goal-conditioned behavioral cloning training loop."""
        metrics_history: Dict[str, list[float]] = {"bc_loss": []}
        last_metrics: Dict[str, float] = {}

        for step in range(num_steps):
            metrics = self.train_hindsight_step(
                dataset,
                batch_size=batch_size,
                original_goal_prob=original_goal_prob,
            )
            for key, value in metrics.items():
                metrics_history.setdefault(key, []).append(value)
            last_metrics = metrics

            if step % log_every == 0 or step == num_steps - 1:
                log_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                logger.info("GC-BC step %d: %s", step, log_str)

        return {
            "mean_metrics": {k: float(np.mean(v)) for k, v in metrics_history.items()},
            "last_metrics": last_metrics,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_hidden": self.num_hidden,
                "deterministic_policy": self.deterministic_policy,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def to(self, device: str) -> "GoalConditionedBC":
        super().to(device)
        self.device = device
        return self


def train_gc_bc_agent(
    dataset: OfflineDataset,
    cfg: Optional[IQLConfig] = None,
    device: str = "cpu",
    num_steps: int = 100_000,
    batch_size: int = 256,
    original_goal_prob: float = 0.3,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    lr: float = 3e-4,
    deterministic_policy: bool = False,
) -> GoalConditionedBC:
    """Construct and train a GoalConditionedBC agent on the given dataset."""
    if state_dim is None:
        state_dim = dataset.state_dim
    if action_dim is None:
        action_dim = dataset.action_dim

    agent = GoalConditionedBC(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        lr=lr,
        device=device,
        deterministic_policy=deterministic_policy,
    )
    agent.train(
        dataset=dataset,
        num_steps=num_steps,
        batch_size=batch_size,
        original_goal_prob=original_goal_prob,
    )
    return agent


GC_BC = GoalConditionedBC

__all__ = ["GoalConditionedBC", "GC_BC", "train_gc_bc_agent"]
