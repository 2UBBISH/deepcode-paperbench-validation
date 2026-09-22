"""Rollout buffer for SAPG / PPO.

Stores on-policy and off-policy transitions collected from one or more
"follower" policies (workers).  Supports:

* Fixed-horizon rollouts of shape ``(T, N, ...)`` where ``T`` is the rollout
  horizon and ``N`` is the number of parallel environments handled by a single
  worker.
* Recurrent (LSTM) policies: transitions are stored flat and can be reshaped
  into sequences of length ``seq_len`` for truncated back-propagation through
  time (BPTT).
* Off-policy aggregation: every transition is tagged with the ``worker_id`` of
  the follower that generated it, so the leader can compute importance-sampling
  ratios ``pi_leader(a|s) / pi_follower(a|s)``.

The buffer is intentionally simple: it accumulates tensors in Python lists and
materialises them on ``compute_returns_and_advantages`` / ``get_batches``.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch

from .utils import compute_gae


class RolloutBuffer:
    """Storage for a fixed-horizon rollout.

    Parameters
    ----------
    horizon:
        Number of environment steps stored per environment (``T``).
    num_envs:
        Number of parallel environments handled by this buffer (``N``).
    obs_dim:
        Dimensionality of the observation vector.
    action_dim:
        Dimensionality of the action vector.
    device:
        Torch device used to allocate the storage tensors.
    recurrent:
        Whether the policy is recurrent (LSTM).  When ``True`` the buffer also
        stores the initial hidden state for each rollout step so that the
        sequence can be replayed during the update.
    hidden_size:
        Size of the recurrent hidden state (only used when ``recurrent=True``).
    gamma, tau:
        Discount factor and GAE-lambda used by :func:`compute_gae`.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        recurrent: bool = False,
        hidden_size: int = 0,
        gamma: float = 0.99,
        tau: float = 0.95,
    ) -> None:
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.recurrent = bool(recurrent)
        self.hidden_size = int(hidden_size)
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.reset()

    # ------------------------------------------------------------------
    # storage management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all stored transitions."""
        T, N = self.horizon, self.num_envs
        dev = self.device

        self.obs = torch.zeros(T, N, self.obs_dim, device=dev)
        self.actions = torch.zeros(T, N, self.action_dim, device=dev)
        self.log_probs = torch.zeros(T, N, device=dev)
        self.rewards = torch.zeros(T, N, device=dev)
        self.dones = torch.zeros(T, N, device=dev)
        self.values = torch.zeros(T, N, device=dev)
        # worker id of the follower that produced each transition
        self.worker_ids = torch.zeros(T, N, dtype=torch.long, device=dev)

        # recurrent bookkeeping
        if self.recurrent:
            self.hidden_states = torch.zeros(
                T, N, self.hidden_size, device=dev
            )
            self.cell_states = torch.zeros(T, N, self.hidden_size, device=dev)
            self.masks = torch.ones(T, N, device=dev)
        else:
            self.hidden_states = None
            self.cell_states = None
            self.masks = None

        # filled in by compute_returns_and_advantages
        self.advantages = torch.zeros(T, N, device=dev)
        self.returns = torch.zeros(T, N, device=dev)

        self.step = 0
        self.full = False

    def __len__(self) -> int:
        return self.step * self.num_envs

    # ------------------------------------------------------------------
    # insertion
    # ------------------------------------------------------------------
    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one timestep of transitions for all environments."""
        if self.full:
            raise RuntimeError("RolloutBuffer is full; call reset() first.")

        t = self.step
        self.obs[t] = obs.to(self.device)
        self.actions[t] = actions.to(self.device)
        self.log_probs[t] = log_probs.to(self.device).reshape(self.num_envs)
        self.rewards[t] = rewards.to(self.device).reshape(self.num_envs)
        self.dones[t] = dones.to(self.device).reshape(self.num_envs)
        self.values[t] = values.to(self.device).reshape(self.num_envs)

        if worker_ids is not None:
            self.worker_ids[t] = worker_ids.to(self.device).reshape(self.num_envs)

        if self.recurrent:
            if hidden_state is not None:
                self.hidden_states[t] = hidden_state.to(self.device).reshape(
                    self.num_envs, self.hidden_size
                )
            if cell_state is not None:
                self.cell_states[t] = cell_state.to(self.device).reshape(
                    self.num_envs, self.hidden_size
                )
            if masks is not None:
                self.masks[t] = masks.to(self.device).reshape(self.num_envs)

        self.step += 1
        if self.step >= self.horizon:
            self.full = True

    # ------------------------------------------------------------------
    # advantage / return computation
    # ------------------------------------------------------------------
    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute GAE advantages and discounted returns in-place.

        ``last_values`` is the value bootstrap for the state following the last
        stored transition (shape ``(N,)``).
        """
        last_values = last_values.to(self.device).reshape(self.num_envs)
        if last_dones is None:
            last_dones = torch.zeros(self.num_envs, device=self.device)
        else:
            last_dones = last_dones.to(self.device).reshape(self.num_envs)

        advantages, returns = compute_gae(
            rewards=self.rewards,
            values=self.values,
            dones=self.dones,
            gamma=self.gamma,
            tau=self.tau,
            last_values=last_values,
            last_dones=last_dones,
        )
        self.advantages = advantages
        self.returns = returns

    # ------------------------------------------------------------------
    # batch iteration
    # ------------------------------------------------------------------
    def get_batches(
        self,
        num_mini_batches: int,
        seq_len: int = 1,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mini-batches of flattened transitions.

        For recurrent policies the flattened ``(T*N)`` transitions are grouped
        into contiguous sequences of length ``seq_len`` (per environment) so
        that the LSTM hidden state can be replayed.

        Each yielded dict contains:
            obs, actions, old_log_probs, old_values, advantages, returns,
            worker_ids, and (if recurrent) hidden_states, cell_states, masks.
        """
        T, N = self.horizon, self.num_envs
        total = T * N

        # flatten time-major -> (T*N, ...)
        flat = {
            "obs": self.obs.reshape(total, self.obs_dim),
            "actions": self.actions.reshape(total, self.action_dim),
            "old_log_probs": self.log_probs.reshape(total),
            "old_values": self.values.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
            "worker_ids": self.worker_ids.reshape(total),
        }
        if self.recurrent:
            flat["hidden_states"] = self.hidden_states.reshape(
                total, self.hidden_size
            )
            flat["cell_states"] = self.cell_states.reshape(total, self.hidden_size)
            flat["masks"] = self.masks.reshape(total)

        # normalise advantages (per-buffer, standard PPO practice)
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.recurrent:
            yield from self._recurrent_batches(flat, num_mini_batches, seq_len, shuffle, generator)
        else:
            yield from self._flat_batches(flat, num_mini_batches, shuffle, generator)

    # -- helpers -------------------------------------------------------
    def _flat_batches(
        self,
        flat: Dict[str, torch.Tensor],
        num_mini_batches: int,
        shuffle: bool,
        generator: Optional[torch.Generator],
    ) -> Iterator[Dict[str, torch.Tensor]]:
        total = flat["obs"].shape[0]
        batch_size = max(1, total // max(1, num_mini_batches))
        indices = torch.randperm(total, generator=generator) if shuffle else torch.arange(total)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            if idx.numel() == 0:
                continue
            yield {k: v[idx] for k, v in flat.items()}

    def _recurrent_batches(
        self,
        flat: Dict[str, torch.Tensor],
        num_mini_batches: int,
        seq_len: int,
        shuffle: bool,
        generator: Optional[torch.Generator],
    ) -> Iterator[Dict[str, torch.Tensor]]:
        T, N = self.horizon, self.num_envs
        seq_len = max(1, min(seq_len, T))
        num_seqs_per_env = T // seq_len
        num_seqs = num_seqs_per_env * N

        # reshape (T*N, ...) -> (num_seqs, seq_len, ...)
        def to_seq(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(T, N, *x.shape[1:]).permute(1, 0, *range(2, x.dim() + 1)).reshape(
                N, num_seqs_per_env, seq_len, *x.shape[1:]
            ).reshape(num_seqs, seq_len, *x.shape[1:])

        seq_data = {k: to_seq(v) for k, v in flat.items()}

        seq_indices = (
            torch.randperm(num_seqs, generator=generator)
            if shuffle
            else torch.arange(num_seqs)
        )
        batch_size = max(1, num_seqs // max(1, num_mini_batches))

        for start in range(0, num_seqs, batch_size):
            idx = seq_indices[start : start + batch_size]
            if idx.numel() == 0:
                continue
            yield {k: v[idx] for k, v in seq_data.items()}

    # ------------------------------------------------------------------
    # off-policy aggregation helpers
    # ------------------------------------------------------------------
    def get_worker_data(self, worker_id: int) -> Dict[str, torch.Tensor]:
        """Return all transitions generated by a specific follower.

        Useful for the SPLIT phase where each follower is updated only on its
        own on-policy data.
        """
        mask = (self.worker_ids == worker_id).reshape(-1)
        total = self.horizon * self.num_envs
        flat = {
            "obs": self.obs.reshape(total, self.obs_dim),
            "actions": self.actions.reshape(total, self.action_dim),
            "old_log_probs": self.log_probs.reshape(total),
            "old_values": self.values.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
            "worker_ids": self.worker_ids.reshape(total),
        }
        return {k: v[mask] for k, v in flat.items()}

    def all_data(self) -> Dict[str, torch.Tensor]:
        """Return every stored transition flattened (used by the leader)."""
        total = self.horizon * self.num_envs
        flat = {
            "obs": self.obs.reshape(total, self.obs_dim),
            "actions": self.actions.reshape(total, self.action_dim),
            "old_log_probs": self.log_probs.reshape(total),
            "old_values": self.values.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
            "worker_ids": self.worker_ids.reshape(total),
        }
        if self.recurrent:
            flat["hidden_states"] = self.hidden_states.reshape(total, self.hidden_size)
            flat["cell_states"] = self.cell_states.reshape(total, self.hidden_size)
            flat["masks"] = self.masks.reshape(total)
        return flat

    def to(self, device: torch.device) -> "RolloutBuffer":
        """Move all stored tensors to ``device`` in-place."""
        self.device = device
        for name in (
            "obs",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "values",
            "worker_ids",
            "advantages",
            "returns",
        ):
            setattr(self, name, getattr(self, name).to(device))
        if self.recurrent:
            self.hidden_states = self.hidden_states.to(device)
            self.cell_states = self.cell_states.to(device)
            self.masks = self.masks.to(device)
        return self


class MultiWorkerRolloutBuffer:
    """Convenience wrapper holding one :class:`RolloutBuffer` per worker.

    In SAPG the environments are split into ``B`` blocks; each block ``b`` runs
    follower ``pi_{phi_b}`` and collects its own rollout.  This wrapper keeps
    those rollouts separate (for the SPLIT update) while also exposing a merged
    view (for the AGGREGATE update of the leader).
    """

    def __init__(
        self,
        num_workers: int,
        horizon: int,
        envs_per_worker: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        recurrent: bool = False,
        hidden_size: int = 0,
        gamma: float = 0.99,
        tau: float = 0.95,
    ) -> None:
        self.num_workers = int(num_workers)
        self.horizon = int(horizon)
        self.envs_per_worker = int(envs_per_worker)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.recurrent = bool(recurrent)
        self.hidden_size = int(hidden_size)
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.buffers: List[RolloutBuffer] = [
            RolloutBuffer(
                horizon=horizon,
                num_envs=envs_per_worker,
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device,
                recurrent=recurrent,
                hidden_size=hidden_size,
                gamma=gamma,
                tau=tau,
            )
            for _ in range(self.num_workers)
        ]

    def __getitem__(self, idx: int) -> RolloutBuffer:
        return self.buffers[idx]

    def __len__(self) -> int:
        return self.num_workers

    def reset(self) -> None:
        for buf in self.buffers:
            buf.reset()

    def compute_returns_and_advantages(
        self, last_values: List[torch.Tensor]
    ) -> None:
        for buf, lv in zip(self.buffers, last_values):
            buf.compute_returns_and_advantages(lv)

    def merge(self) -> Dict[str, torch.Tensor]:
        """Concatenate all workers' transitions into a single flat dict.

        The merged ``worker_ids`` are offset so that worker ``b``'s transitions
        are tagged with ``b`` (matching the global embedding table).
        """
        merged: Dict[str, List[torch.Tensor]] = {}
        for b, buf in enumerate(self.buffers):
            data = buf.all_data()
            data["worker_ids"] = torch.full_like(data["worker_ids"], b)
            for k, v in data.items():
                merged.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0) for k, v in merged.items()}
