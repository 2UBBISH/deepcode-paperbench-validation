"""Rollout collection, block partitioning, and off-policy subsampling for SAPG.

This module implements the data-collection machinery described in Section 4.3
and 4.6 of the SAPG paper:

* The ``N`` parallel environments are split into ``M`` blocks of ``N / M``
  environments each.  Policy ``j`` (``j = 1 .. M``) collects data only from its
  own block, producing a per-policy buffer ``D_j``.
* Each policy collects ``H`` steps per environment instance (``H = 16`` for
  AllegroKuka, ``H = 8`` for the in-hand tasks).
* The leader (``i = 1``) aggregates follower data by *uniformly subsampling*
  ``|D_1|`` transitions from the union ``union_{j=2}^{M} D_j`` to form the
  off-policy batch ``D_1'``.  Matching the off-policy batch size to the
  on-policy batch size is a critical design choice (Section 4.6).

The buffers are stored time-major ``(T, B, ...)`` which matches the layout
expected by :mod:`sapg.gae` and :mod:`sapg.losses`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Per-policy rollout buffer
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """Storage for a single policy's on-policy rollout.

    All tensors are time-major with shape ``(T, B, ...)`` where ``T`` is the
    rollout horizon ``H`` and ``B`` is the number of environments in the
    policy's block (``N / M``).

    Attributes
    ----------
    obs : (T, B, obs_dim)
    actions : (T, B, act_dim)
    log_probs : (T, B)
    rewards : (T, B)
    values : (T, B)
    dones : (T, B)  -- 1.0 marks a terminal transition
    next_obs : (T, B, obs_dim) -- observation *after* the action (s')
    next_values : (T, B) -- V(s') under the behaviour policy (for off-policy target)
    lstm_states : optional list of (h, c) tuples per step, each (layers, B, hidden)
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    next_obs: Optional[torch.Tensor] = None
    next_values: Optional[torch.Tensor] = None
    lstm_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    # Extra bookkeeping
    policy_index: int = 0
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    # -- convenience -------------------------------------------------------
    @property
    def horizon(self) -> int:
        return int(self.obs.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.obs.shape[1])

    @property
    def size(self) -> int:
        """Total number of transitions in the buffer."""
        return self.horizon * self.num_envs

    def to(self, device: torch.device) -> "RolloutBuffer":
        """Move all tensors to ``device`` in place and return self."""
        for name in (
            "obs",
            "actions",
            "log_probs",
            "rewards",
            "values",
            "dones",
            "next_obs",
            "next_values",
            "advantages",
            "returns",
        ):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        if self.lstm_states is not None:
            self.lstm_states = [
                (h.to(device), c.to(device)) for (h, c) in self.lstm_states
            ]
        return self

    def flatten(self) -> Dict[str, torch.Tensor]:
        """Flatten the time dimension into the batch dimension.

        Returns a dict with tensors of shape ``(T * B, ...)``.  ``lstm_states``
        are returned as a list of per-step ``(h, c)`` tensors of shape
        ``(layers, B, hidden)`` (time dimension preserved).
        """
        out: Dict[str, torch.Tensor] = {}
        for name in (
            "obs",
            "actions",
            "log_probs",
            "rewards",
            "values",
            "dones",
            "next_obs",
            "next_values",
            "advantages",
            "returns",
        ):
            t = getattr(self, name)
            if t is not None:
                out[name] = t.reshape(-1, *t.shape[2:])
        if self.lstm_states is not None:
            out["lstm_states"] = self.lstm_states
        return out


# ---------------------------------------------------------------------------
# Block partitioning
# ---------------------------------------------------------------------------
def partition_blocks(num_envs: int, num_policies: int) -> List[Tuple[int, int]]:
    """Split ``num_envs`` into ``num_policies`` contiguous blocks.

    Returns a list of ``(start, end)`` index pairs (end exclusive).  If
    ``num_envs`` is not divisible by ``num_policies`` the remainder is
    distributed one-per-block to the first blocks.
    """
    if num_policies <= 0:
        raise ValueError("num_policies must be positive")
    if num_envs < num_policies:
        raise ValueError(
            f"num_envs ({num_envs}) must be >= num_policies ({num_policies})"
        )
    base = num_envs // num_policies
    remainder = num_envs % num_policies
    blocks: List[Tuple[int, int]] = []
    start = 0
    for j in range(num_policies):
        size = base + (1 if j < remainder else 0)
        blocks.append((start, start + size))
        start += size
    return blocks


def block_env_ids(num_envs: int, num_policies: int, policy_index: int) -> List[int]:
    """Return the environment indices belonging to ``policy_index``'s block."""
    blocks = partition_blocks(num_envs, num_policies)
    start, end = blocks[policy_index]
    return list(range(start, end))


# ---------------------------------------------------------------------------
# Off-policy subsampling
# ---------------------------------------------------------------------------
def subsample_off_policy(
    follower_buffers: List[RolloutBuffer],
    target_size: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Uniformly subsample ``target_size`` transitions from the union of
    follower buffers ``D_2 .. D_M``.

    Parameters
    ----------
    follower_buffers : list of RolloutBuffer
        The buffers of the follower policies (policy indices ``2 .. M``).
    target_size : int
        Number of transitions to sample.  In SAPG this equals ``|D_1|`` (the
        leader's on-policy batch size) -- matching the off-policy batch size to
        the on-policy batch size is a critical design choice (Section 4.6).
    generator : optional torch.Generator
        For reproducible sampling.

    Returns
    -------
    dict with keys ``obs, actions, log_probs, rewards, values, dones,
    next_obs, next_values`` (and ``lstm_states`` if present), each of shape
    ``(target_size, ...)``.  ``log_probs`` are the *behaviour* log-probabilities
    ``log pi_j(a|s)`` used to form the importance ratio ``r = pi_i / pi_j``.
    """
    if not follower_buffers:
        raise ValueError("follower_buffers must be non-empty")

    # Concatenate all follower transitions along the flattened batch dimension.
    keys = ["obs", "actions", "log_probs", "rewards", "values", "dones"]
    optional_keys = ["next_obs", "next_values"]

    cat: Dict[str, torch.Tensor] = {}
    for k in keys:
        cat[k] = torch.cat([getattr(b, k).reshape(-1, *getattr(b, k).shape[2:]) for b in follower_buffers], dim=0)
    for k in optional_keys:
        if all(getattr(b, k) is not None for b in follower_buffers):
            cat[k] = torch.cat(
                [getattr(b, k).reshape(-1, *getattr(b, k).shape[2:]) for b in follower_buffers],
                dim=0,
            )

    total = cat["obs"].shape[0]
    if target_size > total:
        # Sample with replacement if the union is smaller than requested.
        replace = True
    else:
        replace = False

    device = cat["obs"].device
    idx = torch.randint(
        0, total, (target_size,), generator=generator, device=device, dtype=torch.long
    ) if replace else torch.randperm(total, generator=generator, device=device)[:target_size]

    out: Dict[str, torch.Tensor] = {}
    for k, v in cat.items():
        out[k] = v[idx]

    # LSTM states: subsample the per-step hidden states along the batch dim.
    if all(b.lstm_states is not None for b in follower_buffers):
        # Build a flat list of (h, c) per step across followers, then index.
        # Each follower buffer has the same horizon H (enforced by caller).
        horizons = {b.horizon for b in follower_buffers}
        if len(horizons) == 1:
            H = horizons.pop()
            # offsets into the concatenated batch for each follower
            offsets = []
            running = 0
            for b in follower_buffers:
                offsets.append(running)
                running += b.num_envs
            # For each sampled flat index, recover (follower, step, env)
            # flat index layout: step-major within each follower, then concat.
            # We instead reconstruct per-step hidden states by concatenating
            # followers along the batch dim at each step.
            step_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for t in range(H):
                hs = torch.cat([b.lstm_states[t][0] for b in follower_buffers], dim=1)
                cs = torch.cat([b.lstm_states[t][1] for b in follower_buffers], dim=1)
                step_states.append((hs, cs))
            # Map flat index -> (step, env_within_concat)
            step_of = idx // running
            env_of = idx % running
            sampled_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for t in range(H):
                mask = step_of == t
                if mask.any():
                    sel = env_of[mask]
                    sampled_states.append(
                        (step_states[t][0][:, sel], step_states[t][1][:, sel])
                    )
            out["lstm_states"] = sampled_states
    return out


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_rollout(
    env,
    actor_critic,
    policy_index: int,
    num_envs_block: int,
    horizon: int,
    obs: torch.Tensor,
    lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    deterministic: bool = False,
) -> Tuple[RolloutBuffer, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Collect ``horizon`` steps of experience for a single policy.

    Parameters
    ----------
    env : environment wrapper
        Must expose ``step(actions) -> (obs, rewards, dones, infos)`` and
        ``reset()``.  ``obs`` is expected to be a tensor of shape
        ``(num_envs_block, obs_dim)``.
    actor_critic : MultiPolicyActorCritic
        Shared actor-critic; ``policy_index`` selects ``phi_j``.
    policy_index : int
        Index of the policy collecting data (0-based internally).
    num_envs_block : int
        Number of environments in this policy's block.
    horizon : int
        Number of steps ``H`` to collect.
    obs : (num_envs_block, obs_dim)
        Current observation for the block.
    lstm_state : optional (h, c)
        Recurrent state carried across iterations (AllegroKuka only).
    device : optional torch.device
    deterministic : bool
        If True, use the distribution mean (evaluation).

    Returns
    -------
    buffer : RolloutBuffer
    next_obs : (num_envs_block, obs_dim)
    next_lstm_state : optional (h, c)
    """
    if device is None:
        device = obs.device

    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []
    next_obs_buf = []
    next_val_buf = []
    lstm_buf: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
        [] if lstm_state is not None else None
    )

    for _t in range(horizon):
        if lstm_state is not None:
            lstm_buf.append((lstm_state[0].clone(), lstm_state[1].clone()))

        actions, log_probs, values, lstm_state = actor_critic.act(
            obs, policy_index, deterministic=deterministic, lstm_state=lstm_state
        )

        next_obs, rewards, dones, _infos = env.step(actions)

        # Value of the next observation under the behaviour policy (for the
        # 1-step off-policy critic target, Eq. 6).
        next_values = actor_critic.value(next_obs, policy_index, lstm_state=lstm_state)

        obs_buf.append(obs)
        act_buf.append(actions)
        logp_buf.append(log_probs)
        rew_buf.append(rewards)
        val_buf.append(values)
        done_buf.append(dones)
        next_obs_buf.append(next_obs)
        next_val_buf.append(next_values)

        obs = next_obs

    buffer = RolloutBuffer(
        obs=torch.stack(obs_buf, dim=0),
        actions=torch.stack(act_buf, dim=0),
        log_probs=torch.stack(logp_buf, dim=0),
        rewards=torch.stack(rew_buf, dim=0),
        values=torch.stack(val_buf, dim=0),
        dones=torch.stack(done_buf, dim=0),
        next_obs=torch.stack(next_obs_buf, dim=0),
        next_values=torch.stack(next_val_buf, dim=0),
        lstm_states=lstm_buf,
        policy_index=policy_index,
    )
    return buffer, obs, lstm_state


def collect_all_blocks(
    env,
    actor_critic,
    num_policies: int,
    horizon: int,
    obs: torch.Tensor,
    lstm_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[RolloutBuffer], torch.Tensor, Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
    """Collect one rollout for every policy block.

    ``obs`` is the full ``(N, obs_dim)`` observation tensor; it is sliced into
    ``M`` blocks and each block is rolled out with its own policy ``phi_j``.
    Returns the list of per-policy buffers, the updated full observation, and
    the updated per-policy LSTM states.
    """
    num_envs = obs.shape[0]
    blocks = partition_blocks(num_envs, num_policies)
    buffers: List[RolloutBuffer] = []
    new_obs = obs.clone()
    new_lstm: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    for j, (start, end) in enumerate(blocks):
        block_obs = obs[start:end]
        lstm_state = None if lstm_states is None else lstm_states[j]
        buf, next_obs, next_lstm = collect_rollout(
            env,
            actor_critic,
            policy_index=j,
            num_envs_block=end - start,
            horizon=horizon,
            obs=block_obs,
            lstm_state=lstm_state,
            device=device,
        )
        buffers.append(buf)
        new_obs[start:end] = next_obs
        new_lstm.append(next_lstm)

    return buffers, new_obs, (new_lstm if lstm_states is not None else None)


__all__ = [
    "RolloutBuffer",
    "partition_blocks",
    "block_env_ids",
    "subsample_off_policy",
    "collect_rollout",
    "collect_all_blocks",
]
