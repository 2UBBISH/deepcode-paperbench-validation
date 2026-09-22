"""Rollout buffer and off-policy data aggregation for SAPG.

This module implements the storage used by SAPG's split-and-aggregate scheme:

* Each *follower* (one per environment block) collects its own rollout into a
  :class:`RolloutBuffer`.  The buffer stores observations, actions, log
  probabilities, rewards, values and done flags, and can compute GAE
  advantages/returns.
* The *leader* is trained on the union of **all** transitions produced by all
  followers (off-policy data).  :class:`AggregatedBuffer` concatenates the
  per-follower buffers and exposes mini-batch iteration for the PPO update.

The design mirrors the paper: no transition is wasted -- every follower's data
is aggregated to update the leader, while each follower only uses its own data
for its own (on-policy) update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from .ppo import compute_gae


@dataclass
class RolloutBuffer:
    """Storage for a single follower's rollout.

    All tensors are stored on CPU by default and moved to the compute device
    lazily when mini-batches are produced.  Shapes follow ``(T, N, ...)`` where
    ``T`` is the horizon length and ``N`` the number of environments in the
    block.

    Attributes:
        horizon: Number of environment steps stored per environment.
        num_envs: Number of parallel environments in the block.
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        phi_dim: Dimensionality of the per-follower parameter vector ``phi_j``.
        use_lstm: Whether the policy is recurrent (AllegroKuka).
        lstm_seq_len: Sequence length used when flattening recurrent data.
        device: Torch device used when materialising mini-batches.
    """

    horizon: int
    num_envs: int
    obs_dim: int
    action_dim: int
    phi_dim: int = 0
    use_lstm: bool = False
    lstm_seq_len: int = 16
    device: str = "cpu"

    # Storage (allocated lazily on first insert).
    obs: Optional[torch.Tensor] = field(default=None, repr=False)
    actions: Optional[torch.Tensor] = field(default=None, repr=False)
    log_probs: Optional[torch.Tensor] = field(default=None, repr=False)
    rewards: Optional[torch.Tensor] = field(default=None, repr=False)
    values: Optional[torch.Tensor] = field(default=None, repr=False)
    dones: Optional[torch.Tensor] = field(default=None, repr=False)
    # Per-transition block id (which follower produced the transition).
    block_ids: Optional[torch.Tensor] = field(default=None, repr=False)
    # Per-transition phi_j used to produce the transition (needed for the
    # off-policy leader update, since the leader is conditioned on phi_j).
    phis: Optional[torch.Tensor] = field(default=None, repr=False)
    # LSTM hidden states at the start of each stored step (recurrent policies).
    lstm_states: Optional[torch.Tensor] = field(default=None, repr=False)

    # Bookkeeping.
    ptr: int = 0
    full: bool = False
    advantages: Optional[torch.Tensor] = field(default=None, repr=False)
    returns: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._allocate()

    # ------------------------------------------------------------------ #
    # Allocation / reset
    # ------------------------------------------------------------------ #
    def _allocate(self) -> None:
        T, N = self.horizon, self.num_envs
        self.obs = torch.zeros(T, N, self.obs_dim)
        self.actions = torch.zeros(T, N, self.action_dim)
        self.log_probs = torch.zeros(T, N)
        self.rewards = torch.zeros(T, N)
        self.values = torch.zeros(T, N)
        self.dones = torch.zeros(T, N)
        self.block_ids = torch.zeros(T, N, dtype=torch.long)
        if self.phi_dim > 0:
            self.phis = torch.zeros(T, N, self.phi_dim)
        if self.use_lstm:
            # Store the hidden state *entering* each step so the recurrent
            # forward pass can be replayed during the update.
            self.lstm_states = None  # populated on first insert

    def reset(self) -> None:
        """Zero the buffer and reset the write pointer."""
        self._allocate()
        self.ptr = 0
        self.full = False
        self.advantages = None
        self.returns = None

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        block_id: int = 0,
        phi: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """Store one timestep of data for all environments in the block."""
        if self.ptr >= self.horizon:
            raise RuntimeError(
                f"RolloutBuffer overflow: ptr={self.ptr} horizon={self.horizon}"
            )
        t = self.ptr
        self.obs[t] = obs.detach().cpu()
        self.actions[t] = actions.detach().cpu()
        self.log_probs[t] = log_probs.detach().cpu()
        self.rewards[t] = rewards.detach().cpu()
        self.values[t] = values.detach().cpu()
        self.dones[t] = dones.detach().cpu()
        self.block_ids[t] = block_id
        if self.phis is not None and phi is not None:
            phi_t = phi.detach().cpu()
            if phi_t.dim() == 1:
                phi_t = phi_t.unsqueeze(0).expand(self.num_envs, -1)
            self.phis[t] = phi_t
        if self.use_lstm and lstm_state is not None:
            h, c = lstm_state
            h = h.detach().cpu()
            c = c.detach().cpu()
            if self.lstm_states is None:
                self.lstm_states = torch.zeros(
                    self.horizon, self.num_envs, 2, h.shape[-1]
                )
            self.lstm_states[t, :, 0] = h.squeeze(0) if h.dim() == 3 else h
            self.lstm_states[t, :, 1] = c.squeeze(0) if c.dim() == 3 else c
        self.ptr += 1
        if self.ptr == self.horizon:
            self.full = True

    # ------------------------------------------------------------------ #
    # Advantage computation
    # ------------------------------------------------------------------ #
    def compute_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        tau: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns for the stored rollout.

        Args:
            last_values: Bootstrap value for the step after the horizon,
                shape ``(N,)``.
            last_dones: Done flags at the bootstrap step, shape ``(N,)``.
            gamma: Discount factor.
            tau: GAE lambda.

        Returns:
            ``(advantages, returns)`` each of shape ``(T, N)``.
        """
        if last_dones is None:
            last_dones = torch.zeros(self.num_envs)
        advantages, returns = compute_gae(
            self.rewards,
            self.values,
            self.dones,
            last_values.detach().cpu(),
            gamma=gamma,
            tau=tau,
        )
        self.advantages = advantages
        self.returns = returns
        return advantages, returns

    # ------------------------------------------------------------------ #
    # Mini-batch iteration
    # ------------------------------------------------------------------ #
    def _flatten(self) -> Dict[str, torch.Tensor]:
        """Flatten ``(T, N, ...)`` storage into ``(T*N, ...)`` tensors."""
        T, N = self.horizon, self.num_envs
        data = {
            "obs": self.obs.reshape(T * N, -1),
            "actions": self.actions.reshape(T * N, -1),
            "log_probs": self.log_probs.reshape(T * N),
            "values": self.values.reshape(T * N),
            "advantages": self.advantages.reshape(T * N),
            "returns": self.returns.reshape(T * N),
            "block_ids": self.block_ids.reshape(T * N),
        }
        if self.phis is not None:
            data["phis"] = self.phis.reshape(T * N, -1)
        if self.use_lstm and self.lstm_states is not None:
            data["lstm_states"] = self.lstm_states.reshape(T * N, 2, -1)
        return data

    def get_minibatches(
        self,
        num_minibatches: int,
        device: Optional[str] = None,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches of flattened transitions.

        Args:
            num_minibatches: Number of mini-batches to split the data into.
            device: Device to move each mini-batch to.
            shuffle: Whether to shuffle the flattened transitions.
            generator: Optional RNG for reproducible shuffling.

        Yields:
            Dict of tensors, each with leading dimension ``batch_size``.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "compute_advantages() must be called before get_minibatches()."
            )
        device = device or self.device
        data = self._flatten()
        total = self.horizon * self.num_envs
        indices = torch.randperm(total, generator=generator) if shuffle else torch.arange(total)
        batch_size = total // num_minibatches
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            if idx.numel() == 0:
                continue
            mb = {k: v[idx].to(device) for k, v in data.items()}
            yield mb

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.ptr * self.num_envs

    @property
    def num_transitions(self) -> int:
        return self.ptr * self.num_envs

    def to(self, device: str) -> "RolloutBuffer":
        """Move all stored tensors to ``device`` in place."""
        for name in (
            "obs",
            "actions",
            "log_probs",
            "rewards",
            "values",
            "dones",
            "block_ids",
            "phis",
            "lstm_states",
            "advantages",
            "returns",
        ):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        self.device = device
        return self


class AggregatedBuffer:
    """Concatenation of all followers' rollouts for the leader update.

    The leader is trained on **all** transitions produced by every follower
    (off-policy data).  This class concatenates the per-follower buffers and
    provides mini-batch iteration over the union, preserving the ``phi_j`` and
    ``block_id`` associated with each transition so the leader can be
    conditioned correctly and importance weights can be computed.
    """

    def __init__(
        self,
        buffers: List[RolloutBuffer],
        device: str = "cpu",
    ) -> None:
        if not buffers:
            raise ValueError("AggregatedBuffer requires at least one buffer.")
        self.buffers = buffers
        self.device = device
        self._data: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ #
    def _concat(self) -> Dict[str, torch.Tensor]:
        """Concatenate all buffers along the transition dimension."""
        keys = ["obs", "actions", "log_probs", "values", "advantages", "returns", "block_ids"]
        has_phi = all(b.phis is not None for b in self.buffers)
        has_lstm = all(b.lstm_states is not None for b in self.buffers)
        data: Dict[str, torch.Tensor] = {}
        for k in keys:
            parts = []
            for b in self.buffers:
                flat = getattr(b, k).reshape(-1, *getattr(b, k).shape[2:])
                parts.append(flat)
            data[k] = torch.cat(parts, dim=0)
        if has_phi:
            data["phis"] = torch.cat(
                [b.phis.reshape(-1, b.phis.shape[-1]) for b in self.buffers], dim=0
            )
        if has_lstm:
            data["lstm_states"] = torch.cat(
                [b.lstm_states.reshape(-1, 2, b.lstm_states.shape[-1]) for b in self.buffers],
                dim=0,
            )
        return data

    @property
    def num_transitions(self) -> int:
        return sum(b.num_transitions for b in self.buffers)

    def __len__(self) -> int:
        return self.num_transitions

    def get_minibatches(
        self,
        num_minibatches: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mini-batches over the union of all followers' transitions."""
        data = self._concat()
        total = data["obs"].shape[0]
        indices = (
            torch.randperm(total, generator=generator)
            if shuffle
            else torch.arange(total)
        )
        batch_size = max(1, total // num_minibatches)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            if idx.numel() == 0:
                continue
            yield {k: v[idx].to(self.device) for k, v in data.items()}

    def block_statistics(self) -> Dict[str, torch.Tensor]:
        """Return per-block transition counts (verifies no data is wasted)."""
        counts = {}
        for i, b in enumerate(self.buffers):
            counts[f"block_{i}"] = torch.tensor(b.num_transitions)
        return counts


def build_buffers(
    num_blocks: int,
    num_envs_per_block: int,
    horizon: int,
    obs_dim: int,
    action_dim: int,
    phi_dim: int = 0,
    use_lstm: bool = False,
    lstm_seq_len: int = 16,
    device: str = "cpu",
) -> List[RolloutBuffer]:
    """Create one :class:`RolloutBuffer` per environment block (follower)."""
    return [
        RolloutBuffer(
            horizon=horizon,
            num_envs=num_envs_per_block,
            obs_dim=obs_dim,
            action_dim=action_dim,
            phi_dim=phi_dim,
            use_lstm=use_lstm,
            lstm_seq_len=lstm_seq_len,
            device=device,
        )
        for _ in range(num_blocks)
    ]
