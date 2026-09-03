"""Goal-Conditioned Behavioral Cloning (GC-BC) baseline.

This baseline learns a goal-conditioned policy purely by supervised cloning on
offline transitions whose goals are relabeled to states that were actually
reached (next states). During evaluation, the policy is conditioned on a target
goal vector and rolled out in the environment.

It intentionally keeps the same network sizes as the FRE policy (256x256 MLPs)
so that performance differences reflect the learning objective rather than
capacity.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.utils import to_torch
from .baseline_utils import GaussianPolicy, make_policy_fn_from_net

__all__ = ["GCBC"]


class GCBC:
    """Goal-conditioned behavioral cloning agent.

    Parameters
    ----------
    state_dim : int
        Dimensionality of environment observations/states.
    action_dim : int
        Dimensionality of actions.
    goal_dim : Optional[int]
        Dimensionality of goal vectors. Defaults to ``state_dim`` because the
        D4RL/ExORL goal-reaching tasks use full states as goals.
    hidden_dims : Sequence[int]
        Hidden sizes for the Gaussian policy MLP.
    lr : float
        Adam learning rate.
    batch_size : int
        Reference batch size (kept in metadata only).
    relabel_fraction : float
        Fraction of transitions for which the goal is relabeled to the
        transition's next state. For the remaining fraction a random state from
        the batch is used as goal, which injects a mild regularizer and makes
        the policy robust to unseen goal inputs.
    goal_epsilon : float
        Not used directly by BC training but retained for interface parity with
        GC-IQL and for reward definitions in evaluation.
    device : Union[str, torch.device]
        Torch device.
    max_action : float
        Action scale used by the squashed Gaussian policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        batch_size: int = 256,
        relabel_fraction: float = 0.5,
        goal_epsilon: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        max_action: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim if goal_dim is not None else state_dim
        self.hidden_dims = tuple(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.relabel_fraction = relabel_fraction
        self.goal_epsilon = goal_epsilon
        self.device = torch.device(device)
        self.max_action = max_action

        self.policy = GaussianPolicy(
            state_dim=state_dim,
            context_dim=self.goal_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
            max_action=max_action,
            activation="relu",
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    def to(self, device: Union[str, torch.device]) -> "GCBC":
        """Move all networks to ``device`` and return self."""
        self.device = torch.device(device)
        self.policy.to(self.device)
        return self

    def _as_tensor(
        self, x: Union[np.ndarray, torch.Tensor], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return to_torch(x, device=self.device, dtype=dtype)

    def train_step(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_states: Union[np.ndarray, torch.Tensor],
        dones: Optional[Union[np.ndarray, torch.Tensor]] = None,
        goals: Optional[Union[np.ndarray, torch.Tensor]] = None,
        goal_pool: Optional[Union[np.ndarray, torch.Tensor]] = None,
        relabel_fraction: Optional[float] = None,
    ) -> Dict[str, float]:
        """Perform one behavioral-cloning update.

        If explicit ``goals`` are not supplied, each transition receives a goal
        equal to its next state with probability ``relabel_fraction`` and a
        randomly sampled state otherwise. The random states are drawn from
        ``goal_pool`` when available, otherwise from the current batch.

        Returns
        -------
        Dict[str, float]
            Scalar training metrics.
        """
        s = self._as_tensor(states)
        a = self._as_tensor(actions)
        ns = self._as_tensor(next_states)

        if goals is not None:
            g = self._as_tensor(goals)
        else:
            frac = relabel_fraction if relabel_fraction is not None else self.relabel_fraction
            B = s.shape[0]
            if goal_pool is not None:
                pool = self._as_tensor(goal_pool)
                idx = torch.randint(0, pool.shape[0], (B,), device=self.device)
                random_goals = pool[idx]
            else:
                perm = torch.randperm(B, device=self.device)
                random_goals = s[perm]

            use_next = torch.rand(B, device=self.device) < frac
            g = torch.where(
                use_next.unsqueeze(-1), ns, random_goals
            )

        # GaussianPolicy.forward returns (action, mean, log_std, log_prob).
        _, _, _, log_prob = self.policy(s, g)
        loss = -log_prob.mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()

        return {"policy_loss": float(loss.detach().cpu().item())}

    def get_task_policy(
        self,
        goal: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ):
        """Return an observation -> action closure conditioned on ``goal``."""
        goal_t = self._as_tensor(goal)
        if goal_t.ndim == 1:
            goal_t = goal_t.unsqueeze(0)

        return make_policy_fn_from_net(
            self.policy,
            goal_t,
            device=self.device,
            deterministic=deterministic,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state, including optimizer state."""
        return {
            "policy_state_dict": copy.deepcopy(self.policy.state_dict()),
            "policy_optimizer_state_dict": copy.deepcopy(
                self.policy_optimizer.state_dict()
            ),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
            "hidden_dims": self.hidden_dims,
            "relabel_fraction": self.relabel_fraction,
            "goal_epsilon": self.goal_epsilon,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state from a dictionary returned by :meth:`state_dict`."""
        self.policy.load_state_dict(state_dict["policy_state_dict"])
        if "policy_optimizer_state_dict" in state_dict:
            self.policy_optimizer.load_state_dict(
                state_dict["policy_optimizer_state_dict"]
            )

    def save(self, path: str) -> None:
        """Save agent state to ``path``."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load agent state from ``path``."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)
