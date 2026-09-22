"""Rollout buffers for SAPG.

Each of the M policies owns its own rollout buffer ``D_j``.  A buffer stores the
transitions collected by a single policy over a fixed horizon and provides:

* n-step on-policy value targets (Eq. 5-6, n = 3 by default),
* GAE advantages for the on-policy PPO surrogate (Eq. 2),
* a flat view used for subsampling the off-policy batch ``D_1'`` (Eq. 3).

The buffers are deliberately simple (dense tensors on a single device) because
the number of transitions per policy is ``num_envs_per_policy * horizon`` which
is small compared to the model itself.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .losses import compute_gae, n_step_value_target


class RolloutBuffer:
    """Storage for the transitions collected by one policy.

    All tensors are stored with a leading time dimension ``T`` (the horizon) and
    a second dimension ``N`` (the number of parallel environments assigned to
    this policy).  Recurrent policies additionally store the LSTM hidden state
    at every timestep so that mini-batch updates can be performed with the
    correct initial state.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        recurrent: bool = False,
        lstm_hidden: int = 0,
    ) -> None:
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.recurrent = bool(recurrent)
        self.lstm_hidden = int(lstm_hidden)

        self.obs = torch.zeros(self.horizon, self.num_envs, self.obs_dim, device=device)
        self.actions = torch.zeros(self.horizon, self.num_envs, self.action_dim, device=device)
        self.log_probs = torch.zeros(self.horizon, self.num_envs, device=device)
        self.rewards = torch.zeros(self.horizon, self.num_envs, device=device)
        self.dones = torch.zeros(self.horizon, self.num_envs, device=device)
        self.values = torch.zeros(self.horizon, self.num_envs, device=device)

        # Optional recurrent state storage (h, c) at the *start* of each step.
        if self.recurrent and self.lstm_hidden > 0:
            self.lstm_h = torch.zeros(self.horizon, self.num_envs, self.lstm_hidden, device=device)
            self.lstm_c = torch.zeros(self.horizon, self.num_envs, self.lstm_hidden, device=device)
        else:
            self.lstm_h = None
            self.lstm_c = None

        self.ptr = 0
        self.full = False

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #
    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """Append one timestep of transitions for all environments."""
        if self.ptr >= self.horizon:
            raise RuntimeError(
                f"RolloutBuffer overflow: ptr={self.ptr} horizon={self.horizon}"
            )
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        if self.recurrent and lstm_state is not None and self.lstm_h is not None:
            self.lstm_h[t] = lstm_state[0]
            self.lstm_c[t] = lstm_state[1]
        self.ptr += 1
        if self.ptr == self.horizon:
            self.full = True

    # ------------------------------------------------------------------ #
    # Targets / advantages
    # ------------------------------------------------------------------ #
    def compute_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        tau: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE advantages and returns for the on-policy surrogate (Eq. 2)."""
        advantages, returns = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma=gamma, tau=tau
        )
        self.advantages = advantages
        self.returns = returns
        return advantages, returns

    def compute_n_step_targets(
        self,
        last_values: torch.Tensor,
        n: int = 3,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """On-policy n-step value targets (Eq. 5-6)."""
        targets = n_step_value_target(
            self.rewards, self.values, self.dones, last_values, n=n, gamma=gamma
        )
        self.value_targets = targets
        return targets

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def flat(self) -> Dict[str, torch.Tensor]:
        """Flatten the buffer into ``[T*N, ...]`` tensors for mini-batching."""
        out: Dict[str, torch.Tensor] = {
            "obs": self.obs.reshape(-1, self.obs_dim),
            "actions": self.actions.reshape(-1, self.action_dim),
            "log_probs": self.log_probs.reshape(-1),
            "rewards": self.rewards.reshape(-1),
            "dones": self.dones.reshape(-1),
            "values": self.values.reshape(-1),
        }
        if hasattr(self, "advantages"):
            out["advantages"] = self.advantages.reshape(-1)
            out["returns"] = self.returns.reshape(-1)
        if hasattr(self, "value_targets"):
            out["value_targets"] = self.value_targets.reshape(-1)
        if self.recurrent and self.lstm_h is not None:
            out["lstm_h"] = self.lstm_h.reshape(-1, self.lstm_hidden)
            out["lstm_c"] = self.lstm_c.reshape(-1, self.lstm_hidden)
        return out

    def size(self) -> int:
        return self.ptr * self.num_envs

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.size()


class MultiPolicyRollout:
    """Container holding one :class:`RolloutBuffer` per policy."""

    def __init__(
        self,
        num_policies: int,
        horizon: int,
        num_envs_per_policy: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        recurrent: bool = False,
        lstm_hidden: int = 0,
    ) -> None:
        self.num_policies = int(num_policies)
        self.buffers: List[RolloutBuffer] = [
            RolloutBuffer(
                horizon=horizon,
                num_envs=num_envs_per_policy,
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device,
                recurrent=recurrent,
                lstm_hidden=lstm_hidden,
            )
            for _ in range(self.num_policies)
        ]

    def __getitem__(self, j: int) -> RolloutBuffer:
        return self.buffers[j]

    def __len__(self) -> int:
        return len(self.buffers)

    def reset(self) -> None:
        for b in self.buffers:
            b.reset()

    def compute_targets(
        self,
        last_values: List[torch.Tensor],
        gamma: float = 0.99,
        tau: float = 0.95,
        n_step: int = 3,
    ) -> None:
        """Compute GAE advantages and n-step targets for every policy."""
        for j, buf in enumerate(self.buffers):
            buf.compute_advantages(last_values[j], gamma=gamma, tau=tau)
            buf.compute_n_step_targets(last_values[j], n=n_step, gamma=gamma)


def subsample_off_policy(
    buffers: List[RolloutBuffer],
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample ``num_samples`` transitions from the union of ``buffers``.

    Implements the subsampling step of the leader-follower aggregation
    (Sec. 4.3): the off-policy batch ``D_1'`` is drawn from
    ``union_{j=2..M} D_j`` so that its volume equals the leader's on-policy
    batch ``D_1``.
    """
    flats = [b.flat() for b in buffers]
    if len(flats) == 0:
        raise ValueError("subsample_off_policy requires at least one buffer")

    keys = list(flats[0].keys())
    merged: Dict[str, torch.Tensor] = {
        k: torch.cat([f[k] for f in flats], dim=0) for k in keys
    }
    total = merged["obs"].shape[0]
    if num_samples >= total:
        idx = torch.arange(total, device=merged["obs"].device)
    else:
        idx = torch.randperm(total, generator=generator, device=merged["obs"].device)[
            :num_samples
        ]
    return {k: v[idx] for k, v in merged.items()}
