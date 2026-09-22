"""Block-wise rollout buffer with GAE and off-policy importance weights for SAPG.

The buffer stores transitions collected by B environment blocks (each block runs its
own follower policy ``pi_b``).  For the leader update, *all* transitions from *all*
blocks are aggregated into a single buffer and treated as off-policy data with respect
to the leader policy.  Importance ratios are therefore computed against the behaviour
log-probs stored at collection time.

Layout
------
All tensors are stored with a leading ``num_blocks`` dimension and a ``horizon``
dimension::

    obs        : (B, T, obs_dim)
    actions    : (B, T, act_dim)
    logprobs   : (B, T)
    values     : (B, T)
    rewards    : (B, T)
    dones      : (B, T)
    worker_ids : (B, T)   long, index of the worker that produced the transition

The buffer is flattened to ``(B * T, ...)`` for mini-batch iteration.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch


class RolloutBuffer:
    """Block-wise rollout storage with GAE and off-policy importance weights.

    Parameters
    ----------
    num_blocks:
        Number of environment blocks ``B`` (each block = one follower worker).
    horizon:
        Rollout horizon ``T`` (16 for AllegroKuka, 8 for the hand tasks).
    num_envs_per_block:
        Number of parallel environments inside each block.  Used to derive the
        mini-batch size (``num_envs * 4``) when ``mini_batch_size`` is not given.
    obs_dim, action_dim:
        Dimensions of observations and actions.
    gamma:
        Discount factor (0.99).
    tau:
        GAE lambda (0.95).
    device:
        Torch device for the storage tensors.
    """

    def __init__(
        self,
        num_blocks: int,
        horizon: int,
        num_envs_per_block: int = 1,
        obs_dim: int = 1,
        action_dim: int = 1,
        gamma: float = 0.99,
        tau: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.horizon = int(horizon)
        self.num_envs_per_block = int(num_envs_per_block)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(device)

        self.reset()

    # ------------------------------------------------------------------ #
    # storage
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Allocate (or re-allocate) the storage tensors."""
        B, T = self.num_blocks, self.horizon
        dev = self.device
        self.obs = torch.zeros(B, T, self.obs_dim, device=dev)
        self.actions = torch.zeros(B, T, self.action_dim, device=dev)
        self.logprobs = torch.zeros(B, T, device=dev)
        self.values = torch.zeros(B, T, device=dev)
        self.rewards = torch.zeros(B, T, device=dev)
        self.dones = torch.zeros(B, T, device=dev)
        self.worker_ids = torch.zeros(B, T, dtype=torch.long, device=dev)

        # filled by compute_returns
        self.advantages = torch.zeros(B, T, device=dev)
        self.returns = torch.zeros(B, T, device=dev)

        self.step = 0
        self._computed = False

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one time-step of data for every block.

        All tensors are expected to have a leading ``num_blocks`` dimension.
        """
        t = self.step
        assert t < self.horizon, "rollout buffer overflow"
        self.obs[:, t] = obs
        self.actions[:, t] = actions
        self.logprobs[:, t] = logprobs
        self.values[:, t] = values
        self.rewards[:, t] = rewards
        self.dones[:, t] = dones
        if worker_ids is None:
            worker_ids = torch.arange(self.num_blocks, device=self.device)
        self.worker_ids[:, t] = worker_ids
        self.step += 1

    # ------------------------------------------------------------------ #
    # GAE
    # ------------------------------------------------------------------ #
    def compute_returns(
        self,
        last_values: torch.Tensor,
        last_dones: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_values:
            Value estimates for the state following the last stored step, shape
            ``(num_blocks,)``.
        last_dones:
            Optional done flags for the bootstrap state, shape ``(num_blocks,)``.
        """
        if last_dones is None:
            last_dones = torch.zeros(self.num_blocks, device=self.device)

        adv = torch.zeros(self.num_blocks, device=self.device)
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_value = last_values
                next_done = last_dones
            else:
                next_value = self.values[:, t + 1]
                next_done = self.dones[:, t + 1]

            not_done = 1.0 - next_done
            delta = self.rewards[:, t] + self.gamma * next_value * not_done - self.values[:, t]
            adv = delta + self.gamma * self.tau * not_done * adv
            self.advantages[:, t] = adv

        self.returns = self.advantages + self.values
        self._computed = True

    # ------------------------------------------------------------------ #
    # mini-batch iteration
    # ------------------------------------------------------------------ #
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *x.shape[2:])

    def mini_batch_size(self, multiplier: int = 4) -> int:
        """Mini-batch size = ``num_envs * multiplier`` (paper uses 4)."""
        total = self.num_blocks * self.horizon
        return max(1, min(total, self.num_blocks * self.num_envs_per_block * multiplier))

    def get_mini_batches(
        self,
        mini_batch_size: Optional[int] = None,
        num_mini_epochs: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches of flattened transitions.

        Each yielded dict contains ``obs, actions, old_logprobs, old_values,
        advantages, returns, worker_ids``.
        """
        assert self._computed, "call compute_returns() before iterating mini-batches"
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size()

        total = self.num_blocks * self.horizon
        flat = {
            "obs": self._flatten(self.obs),
            "actions": self._flatten(self.actions),
            "old_logprobs": self._flatten(self.logprobs),
            "old_values": self._flatten(self.values),
            "advantages": self._flatten(self.advantages),
            "returns": self._flatten(self.returns),
            "worker_ids": self._flatten(self.worker_ids),
        }

        for _ in range(num_mini_epochs):
            perm = torch.randperm(total, device=self.device, generator=generator)
            for start in range(0, total, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                yield {k: v[idx] for k, v in flat.items()}

    # ------------------------------------------------------------------ #
    # aggregation helpers
    # ------------------------------------------------------------------ #
    def select_blocks(self, block_ids) -> "RolloutBuffer":
        """Return a *view-like* buffer restricted to the given block indices.

        Used for the follower update (each follower trains on its own block data)
        and for the symmetric-aggregation ablation (each worker trains on all
        *other* workers' data).
        """
        if isinstance(block_ids, int):
            block_ids = [block_ids]
        block_ids = list(block_ids)
        sub = RolloutBuffer(
            num_blocks=len(block_ids),
            horizon=self.horizon,
            num_envs_per_block=self.num_envs_per_block,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            tau=self.tau,
            device=str(self.device),
        )
        idx = torch.as_tensor(block_ids, device=self.device, dtype=torch.long)
        sub.obs = self.obs[idx]
        sub.actions = self.actions[idx]
        sub.logprobs = self.logprobs[idx]
        sub.values = self.values[idx]
        sub.rewards = self.rewards[idx]
        sub.dones = self.dones[idx]
        sub.worker_ids = self.worker_ids[idx]
        sub.advantages = self.advantages[idx]
        sub.returns = self.returns[idx]
        sub.step = self.step
        sub._computed = self._computed
        return sub

    def aggregate(self, exclude: Optional[int] = None) -> "RolloutBuffer":
        """Aggregate all blocks into one buffer (optionally excluding one block).

        This is the core data-sharing primitive of SAPG: the leader (or, in the
        symmetric variant, every worker) is updated on the union of all blocks'
        transitions.
        """
        block_ids = [b for b in range(self.num_blocks) if b != exclude]
        return self.select_blocks(block_ids)

    def normalize_advantages(self, eps: float = 1e-8) -> None:
        """In-place advantage normalization over the whole buffer."""
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std + eps)

    def __len__(self) -> int:
        return self.num_blocks * self.horizon
