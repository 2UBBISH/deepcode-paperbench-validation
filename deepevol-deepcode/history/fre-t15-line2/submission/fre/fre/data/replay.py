"""Offline (unlabeled) replay buffer for FRE.

The paper's Algorithm 1 (Source: Section 4.3, Algorithm 1 "Functional Reward
Encodings (FRE)") takes as input an **unlabeled** offline dataset
:math:`\\mathcal{D}` and samples from it:

* :math:`K` states for the encoder, :math:`\\{s_k^{e}\\} \\sim \\mathcal{D}` (K = 32),
* :math:`K'` states for the decoder, :math:`\\{s_k^{d}\\} \\sim \\mathcal{D}` (K' = 8),
* a batch of state-action pairs :math:`(s, a)` used by IQL
  ("At each training iteration, a batch of state-action pairs :math:`(s, a)` are
  selected from the offline dataset", Source: Section 4.3).

Nothing about the dataset is task-specific: the reward functions are drawn from
the prior :math:`p(\\eta)` (Section 4.2) and evaluated on the sampled states, so
this module only has to provide *uniform* state sampling and uniform
transition sampling, plus enough bookkeeping (trajectory id / timestep) for the
hindsight-relabelling goal-reaching sampler of Appendix B ("a future state
within the trajectory with a 0.5 chance").

The buffer is deliberately environment-agnostic; the benchmark-specific loaders
(`antmaze_dataset.py`, `exorl_dataset.py`, `kitchen_dataset.py`) build one and
attach normalization statistics.

Optional ``encoder_observations``: for ExORL the encoder receives physics
augmented states while the RL components use the raw environment state
(Source: Appendix C.2 -- "The above auxiliary information is neccessary only for
the encoder network"), so the buffer can hold two parallel state arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Batch",
    "Trajectory",
    "ReplayBuffer",
    "UniformBatchSampler",
    "make_replay_buffer",
]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """A single offline trajectory.

    ``observations`` holds ``T + 1`` states (the terminal observation included)
    so that ``s'`` of the last transition is well defined without leaking across
    trajectory boundaries.
    """

    observations: np.ndarray  # (T + 1, obs_dim)
    actions: np.ndarray  # (T, act_dim)
    rewards: Optional[np.ndarray] = None  # (T,) -- not used by FRE (unlabeled)
    terminals: Optional[np.ndarray] = None  # (T,) -- done mask
    timeouts: Optional[np.ndarray] = None  # (T,) -- episode cut by step limit
    encoder_observations: Optional[np.ndarray] = None  # (T + 1, enc_obs_dim)
    infos: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        T = self.actions.shape[0]
        if self.rewards is None:
            self.rewards = np.zeros(T, dtype=np.float32)
        if self.terminals is None:
            self.terminals = np.zeros(T, dtype=np.float32)

    @property
    def length(self) -> int:
        """Number of transitions ``T`` in the trajectory."""
        return int(self.actions.shape[0])

    @property
    def num_states(self) -> int:
        return int(self.observations.shape[0])


@dataclass
class Batch:
    """A minibatch of transitions sampled uniformly from the offline dataset."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    traj_index: Optional[np.ndarray] = None
    step_index: Optional[np.ndarray] = None
    encoder_observations: Optional[np.ndarray] = None
    encoder_next_observations: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def to_torch(self, device=None, dtype=None):
        """Return the batch as a dict of ``torch`` tensors (``s, a, s2, r, done``)."""
        import torch

        if dtype is None:
            dtype = torch.float32

        def _t(x):
            if x is None:
                return None
            return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)

        out = {
            "observations": _t(self.observations),
            "actions": _t(self.actions),
            "next_observations": _t(self.next_observations),
            "rewards": _t(self.rewards),
            "terminals": _t(self.terminals),
        }
        if self.encoder_observations is not None:
            out["encoder_observations"] = _t(self.encoder_observations)
        if self.encoder_next_observations is not None:
            out["encoder_next_observations"] = _t(self.encoder_next_observations)
        return out


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Unlabeled offline dataset of trajectories with uniform sampling helpers.

    All sampling is uniform over *states* (or transitions) in the dataset, which
    is exactly what Algorithm 1 prescribes (Source: Section 4.3:
    ":math:`\\{s_k^{e}\\} \\sim \\mathcal{D}`" and ":math:`\\{s_k^{d}\\} \\sim
    \\mathcal{D}`").
    """

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        seed: int = 0,
        exclude_final_states: bool = True,
    ) -> None:
        if len(trajectories) == 0:
            raise ValueError("ReplayBuffer requires at least one trajectory")
        self.trajectories: List[Trajectory] = list(trajectories)
        self.rng = np.random.default_rng(seed)

        self.observations = np.concatenate(
            [np.asarray(t.observations, dtype=np.float32) for t in self.trajectories], axis=0
        )
        self._traj_index = np.concatenate(
            [
                np.full(t.num_states, i, dtype=np.int64)
                for i, t in enumerate(self.trajectories)
            ]
        )
        self._step_index = np.concatenate(
            [np.arange(t.num_states, dtype=np.int64) for t in self.trajectories]
        )

        # ExORL-style physics augmentation for the encoder only.
        if all(t.encoder_observations is not None for t in self.trajectories):
            self.encoder_observations = np.concatenate(
                [np.asarray(t.encoder_observations, dtype=np.float32) for t in self.trajectories],
                axis=0,
            )
        else:
            self.encoder_observations = None

        # For each state: index of a valid transition starting there (or -1 if it
        # is the terminal observation of its trajectory).
        self._transition_pointer = np.full(self._traj_index.shape[0], -1, dtype=np.int64)
        offset = 0
        pointers = []
        for t in self.trajectories:
            n_states = t.num_states
            ptr = np.arange(n_states, dtype=np.int64)
            ptr[-1] = -1  # terminal observation: no outgoing transition
            pointers.append(ptr)
            offset += n_states
        self._transition_pointer = np.concatenate(pointers)

        # State sampling mask: the paper samples states from the dataset; the
        # terminal observation of each trajectory duplicates information and is
        # excluded by default (Source: not specified in the paper).
        if exclude_final_states:
            mask = self._transition_pointer >= 0
            if mask.sum() == 0:  # degenerate: single-state trajectories
                mask = np.ones_like(mask, dtype=bool)
            self._state_mask = mask
        else:
            self._state_mask = np.ones(self._traj_index.shape[0], dtype=bool)
        self._state_indices = np.nonzero(self._state_mask)[0]

        self.num_transitions = int(self._state_mask.sum())
        self._flat_valid = self._state_indices

    # -- basic properties ---------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def encoder_obs_dim(self) -> int:
        if self.encoder_observations is None:
            return self.obs_dim
        return int(self.encoder_observations.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.trajectories[0].actions.shape[-1])

    @property
    def num_states(self) -> int:
        return int(self._state_indices.shape[0])

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    def __len__(self) -> int:
        return self.num_transitions

    # -- uniform state sampling --------------------------------------------
    def sample_state_indices(self, num_samples: int, rng: Optional[np.random.Generator] = None):
        rng = self.rng if rng is None else rng
        idx = rng.integers(0, self.num_states, size=int(num_samples))
        return self._flat_valid[idx]

    def sample_states(
        self,
        num_samples: int,
        rng: Optional[np.random.Generator] = None,
        encoder_input: bool = False,
    ) -> np.ndarray:
        """Uniformly sample ``num_samples`` states from the offline dataset.

        Returns an array of shape ``(num_samples, obs_dim)`` (or
        ``(num_samples, encoder_obs_dim)`` when ``encoder_input=True`` and the
        buffer carries ExORL physics-augmented states).
        """
        flat = self.sample_state_indices(num_samples, rng=rng)
        source = self.encoder_observations if (encoder_input and self.encoder_observations is not None) else self.observations
        return np.asarray(source[flat], dtype=np.float32)

    def sample_states_with_metadata(
        self, num_samples: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like :meth:`sample_states` but also returns ``(traj_index, step_index)``.

        Needed by the hindsight-relabelling goal-reaching prior of Appendix B,
        which samples a *future state within the same trajectory*.
        """
        flat = self.sample_state_indices(num_samples, rng=rng)
        return (
            np.asarray(self.observations[flat], dtype=np.float32),
            np.asarray(self._traj_index[flat], dtype=np.int64),
            np.asarray(self._step_index[flat], dtype=np.int64),
        )

    def sample_encoder_and_decoder_states(
        self,
        batch_size: int,
        num_encoder_states: int,
        num_decoder_states: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample Algorithm 1's encoder/decoder state sets.

        Returns
        -------
        enc_states : (batch_size, K, encoder_obs_dim)
            The :math:`K` states labeled with :math:`\\eta` for the encoder.
        dec_states : (batch_size, K', obs_dim)
            The :math:`K'` ``decoding`` states, guaranteed disjoint from the
            encoder states in the same row (Source: Section 5 -- "Crucially, the
            states sampled for decoding are different than those used for
            encoding.").
        dec_traj, dec_step : (batch_size, K')
            Metadata for the decoder states (kept for completeness / logging).
        enc_traj, enc_step : (batch_size, K)
            Metadata for the encoder states, used by the goal-reaching prior to
            place the goal in the same trajectory as an encoding state.
        """
        rng = self.rng if rng is None else rng
        batch_size = int(batch_size)
        enc_traj, enc_step, enc_flat = self._sample_flat_with_metadata(
            batch_size * num_encoder_states, rng
        )
        # Sample extra decoder candidates so disjointness can be enforced.
        extra = 4
        dec_traj, dec_step, dec_flat = self._sample_flat_with_metadata(
            batch_size * num_decoder_states * extra, rng
        )
        enc_flat = enc_flat.reshape(batch_size, num_encoder_states)
        enc_traj = enc_traj.reshape(batch_size, num_encoder_states)
        enc_step = enc_step.reshape(batch_size, num_encoder_states)
        dec_flat = dec_flat.reshape(batch_size, num_decoder_states * extra)
        dec_traj = dec_traj.reshape(batch_size, num_decoder_states * extra)
        dec_step = dec_step.reshape(batch_size, num_decoder_states * extra)

        chosen_flat = np.empty((batch_size, num_decoder_states), dtype=np.int64)
        chosen_traj = np.empty((batch_size, num_decoder_states), dtype=np.int64)
        chosen_step = np.empty((batch_size, num_decoder_states), dtype=np.int64)
        for b in range(batch_size):
            taken = set(enc_flat[b].tolist())
            picked = 0
            for j in range(dec_flat.shape[1]):
                if picked >= num_decoder_states:
                    break
                cand = int(dec_flat[b, j])
                if cand in taken:
                    continue
                taken.add(cand)
                chosen_flat[b, picked] = cand
                chosen_traj[b, picked] = dec_traj[b, j]
                chosen_step[b, picked] = dec_step[b, j]
                picked += 1
            if picked < num_decoder_states:
                # Dataset too small to guarantee disjointness; fall back to the
                # remaining candidates (documented deviation).
                for j in range(dec_flat.shape[1]):
                    if picked >= num_decoder_states:
                        break
                    chosen_flat[b, picked] = dec_flat[b, j]
                    chosen_traj[b, picked] = dec_traj[b, j]
                    chosen_step[b, picked] = dec_step[b, j]
                    picked += 1

        enc_source = self.encoder_observations if self.encoder_observations is not None else self.observations
        enc_states = np.asarray(enc_source[enc_flat], dtype=np.float32)
        dec_states = np.asarray(self.observations[chosen_flat], dtype=np.float32)
        return enc_states, dec_states, enc_traj, enc_step, chosen_traj, chosen_step

    def _sample_flat_with_metadata(self, num_samples: int, rng: np.random.Generator):
        flat = self.sample_state_indices(num_samples, rng=rng)
        return self._traj_index[flat], self._step_index[flat], flat

    # -- trajectory access (for the goal-reaching prior) --------------------
    def trajectory_states(self, traj_index: int, encoder_input: bool = True) -> np.ndarray:
        t = self.trajectories[int(traj_index)]
        if encoder_input and t.encoder_observations is not None:
            return np.asarray(t.encoder_observations, dtype=np.float32)
        return np.asarray(t.observations, dtype=np.float32)

    def state_at(self, traj_index: int, step_index: int) -> np.ndarray:
        return np.asarray(self.trajectories[int(traj_index)].observations[int(step_index)], dtype=np.float32)

    def trajectory_length(self, traj_index: int) -> int:
        return self.trajectories[int(traj_index)].num_states

    # -- transition sampling (IQL phase) ------------------------------------
    def sample_transitions(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        with_encoder_inputs: bool = False,
    ) -> Batch:
        """Uniformly sample ``batch_size`` transitions ``(s, a, s', r, done)``."""
        rng = self.rng if rng is None else rng
        flat = self.sample_state_indices(batch_size, rng=rng)
        traj = self._traj_index[flat]
        step = self._step_index[flat]

        next_flat = np.empty_like(flat)
        # next state inside the same trajectory where possible
        for i, (tj, st, f) in enumerate(zip(traj, step, flat)):
            t = self.trajectories[int(tj)]
            if st + 1 < t.num_states:
                next_flat[i] = f + 1
            else:
                next_flat[i] = f

        rewards = np.empty(batch_size, dtype=np.float32)
        terminals = np.empty(batch_size, dtype=np.float32)
        actions = np.empty((batch_size, self.act_dim), dtype=np.float32)
        for i, (tj, st) in enumerate(zip(traj, step)):
            t = self.trajectories[int(tj)]
            st = min(int(st), t.length - 1)
            actions[i] = t.actions[st]
            rewards[i] = float(t.rewards[st]) if t.rewards is not None else 0.0
            terminals[i] = float(t.terminals[st]) if t.terminals is not None else 0.0

        batch = Batch(
            observations=np.asarray(self.observations[flat], dtype=np.float32),
            actions=actions,
            next_observations=np.asarray(self.observations[next_flat], dtype=np.float32),
            rewards=rewards,
            terminals=terminals,
            traj_index=traj,
            step_index=step,
        )
        if with_encoder_inputs and self.encoder_observations is not None:
            batch.encoder_observations = np.asarray(self.encoder_observations[flat], dtype=np.float32)
            batch.encoder_next_observations = np.asarray(
                self.encoder_observations[next_flat], dtype=np.float32
            )
        return batch

    # -- statistics ---------------------------------------------------------
    def state_statistics(self, encoder_input: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the dataset states (per dimension)."""
        source = self.encoder_observations if (encoder_input and self.encoder_observations is not None) else self.observations
        states = np.asarray(source[self._state_indices], dtype=np.float64)
        mean = states.mean(axis=0)
        std = states.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Iterable sampler used by the IQL training loop
# ---------------------------------------------------------------------------
class UniformBatchSampler:
    """Infinite-ish iterator yielding uniform transition batches.

    The IQL phase of Algorithm 1 runs for a fixed number of *gradient steps*
    (150k encoder steps then 850k policy steps on AntMaze, 1M/1M on
    ExORL/Kitchen; Source: Appendix A, Table 3), so this sampler is sized in
    steps rather than epochs.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 512,
        num_steps: Optional[int] = None,
        seed: int = 0,
        with_encoder_inputs: bool = False,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = int(batch_size)
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)
        self.with_encoder_inputs = with_encoder_inputs

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        while self.num_steps is None or step < self.num_steps:
            yield self.replay_buffer.sample_transitions(
                self.batch_size, rng=self.rng, with_encoder_inputs=self.with_encoder_inputs
            )
            step += 1

    def __len__(self) -> int:
        if self.num_steps is None:
            raise TypeError("UniformBatchSampler has no fixed length when num_steps is None")
        return int(self.num_steps)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _split_lengths(total: int, ends: np.ndarray) -> List[int]:
    lengths: List[int] = []
    prev = 0
    for e in ends:
        e = int(e) + 1
        lengths.append(e - prev)
        prev = e
    if prev < total:
        lengths.append(total - prev)
    return [l for l in lengths if l > 0]


def make_replay_buffer(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: Optional[np.ndarray] = None,
    terminals: Optional[np.ndarray] = None,
    timeouts: Optional[np.ndarray] = None,
    next_observations: Optional[np.ndarray] = None,
    ends: Optional[np.ndarray] = None,
    trajectory_lengths: Optional[Sequence[int]] = None,
    encoder_observations: Optional[np.ndarray] = None,
    seed: int = 0,
    exclude_final_states: bool = True,
) -> ReplayBuffer:
    """Build a :class:`ReplayBuffer` from flat dataset arrays.

    Parameters
    ----------
    observations : (N, obs_dim)
        States (for D4RL-style arrays these are ``observations`` concatenated over
        trajectories; ``next_observations`` supplies the final state of each
        trajectory when available).
    actions : (T, act_dim)
        Actions aligned with the *transitions*.
    ends : optional (num_trajectories,)
        Indices (into the flat arrays) of the last transition of each trajectory,
        e.g. mark `done` or timeout steps. When ``None`` the whole dataset is
        treated as a single trajectory (only appropriate for tiny fixtures).
    trajectory_lengths : optional
        Alternative to ``ends`` when lengths are known directly.
    encoder_observations : optional (N, enc_obs_dim)
        Physics-augmented states used *only* by the encoder (ExORL, Appendix C.2).
    """
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    N = int(observations.shape[0])
    T = int(actions.shape[0])
    if rewards is None:
        rewards = np.zeros(T, dtype=np.float32)
    if terminals is None:
        terminals = np.zeros(T, dtype=np.float32)

    if trajectory_lengths is not None:
        lengths = [int(l) for l in trajectory_lengths]
    elif ends is not None:
        ends = np.asarray(ends, dtype=np.int64).reshape(-1)
        lengths = _split_lengths(T, ends)
    else:
        lengths = [T]

    trajectories: List[Trajectory] = []
    cursor = 0
    for length in lengths:
        if length <= 0:
            continue
        obs = observations[cursor : cursor + length]
        act = actions[cursor : cursor + length]
        rew = np.asarray(rewards[cursor : cursor + length], dtype=np.float32)
        term = np.asarray(terminals[cursor : cursor + length], dtype=np.float32)
        # Append the terminal observation so s' is well defined.
        if cursor + length < N:
            final_obs = observations[cursor + length : cursor + length + 1]
        elif next_observations is not None:
            final_obs = np.asarray(next_observations[-1:], dtype=np.float32)
        else:
            final_obs = obs[-1:]
        obs_seq = np.concatenate([obs, final_obs], axis=0)

        enc_seq = None
        if encoder_observations is not None:
            encoder_observations = np.asarray(encoder_observations, dtype=np.float32)
            enc_seq = encoder_observations[cursor : cursor + length]
            if cursor + length < encoder_observations.shape[0]:
                enc_final = encoder_observations[cursor + length : cursor + length + 1]
            else:
                enc_final = enc_seq[-1:]
            enc_seq = np.concatenate([enc_seq, enc_final], axis=0)

        to = None
        if timeouts is not None:
            to = np.asarray(timeouts[cursor : cursor + length], dtype=np.float32)

        trajectories.append(
            Trajectory(
                observations=obs_seq,
                actions=act,
                rewards=rew,
                terminals=term,
                timeouts=to,
                encoder_observations=enc_seq,
            )
        )
        cursor += length

    return ReplayBuffer(
        trajectories, seed=seed, exclude_final_states=exclude_final_states
    )
