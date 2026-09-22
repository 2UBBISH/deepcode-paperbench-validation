"""Offline dataset / replay buffer for FRE.

The FRE trainer (Algorithm 1 of the paper) needs two different kinds of access to
the *unlabeled* offline dataset ``D``:

*   ``sample_states(K)`` -- an unordered set of ``K`` states used by the encoder
    and the ``K'`` states used by the decoder (Section 4.1 / 4.3, Algorithm 1).
*   ``sample_transitions(B)`` -- ``(s, a, r, s', done)`` tuples used by the
    ``z``-conditioned IQL update (Section 4.3).
*   ``sample_trajectories(B)`` -- short state trajectories used by the goal
    reaching prior, which samples goals with the HER distribution
    ``p(current)=0.2, p(future)=0.5, p(random)=0.3`` (Appendix B).

All rewards stored here are *irrelevant* for FRE training (the reward function
``eta`` is sampled from the prior and evaluated on the fly), but they are kept
because the same buffers are re-used by the GC-IQL / GC-BC baselines and because
value-based diagnostics use them.

The buffer is deliberately numpy-only at storage level so that it can be
constructed without torch, and it optionally exposes torch tensors through
``to_torch``/``as_tensors``.  This mirrors the duck-typed dataset API expected by
``fre/fre/prior.py`` (``sample_states``, ``sample_trajectories``, ``.states``,
``.state_std``).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "DatasetStats",
    "ReplayBuffer",
    "Episode",
    "stack_episodes",
    "make_replay_buffer",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return _as_float32(x)


class DatasetStats(object):
    """Summary statistics of a :class:`ReplayBuffer` (numpy scalars)."""

    __slots__ = ("num_transitions", "obs_dim", "action_dim", "obs_mean", "obs_std",
                 "action_mean", "action_std", "reward_min", "reward_max",
                 "reward_mean", "reward_std", "num_trajectories", "max_episode_steps")

    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))

    def as_dict(self) -> Dict[str, object]:
        return {key: getattr(self, key) for key in self.__slots__}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        items = ", ".join(
            "{}={}".format(k, np.round(v, 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in self.as_dict().items()
        )
        return "DatasetStats({})".format(items)


class Episode(object):
    """A single contiguous trajectory of an offline dataset."""

    __slots__ = ("observations", "actions", "rewards", "next_observations", "terminals")

    def __init__(self, observations: np.ndarray, actions: Optional[np.ndarray] = None,
                 rewards: Optional[np.ndarray] = None,
                 next_observations: Optional[np.ndarray] = None,
                 terminals: Optional[np.ndarray] = None):
        self.observations = _as_2d(observations)
        T = len(self.observations)
        self.actions = None if actions is None else _as_2d(actions)
        self.rewards = None if rewards is None else _as_float32(np.asarray(rewards).reshape(T))
        if next_observations is None:
            # roll the observations forward; the final next-observation is undefined
            # for terminals but is only ever consumed by the baselines.
            nxt = np.concatenate([self.observations[1:], self.observations[-1:]], axis=0)
            self.next_observations = nxt
        else:
            self.next_observations = _as_2d(next_observations)
        if terminals is None:
            self.terminals = np.zeros(T, dtype=np.float32)
            self.terminals[-1] = 1.0
        else:
            self.terminals = _as_float32(np.asarray(terminals).reshape(T))

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def length(self) -> int:
        return len(self.observations)


def stack_episodes(episodes: Sequence[Episode]) -> Dict[str, np.ndarray]:
    """Concatenate a list of :class:`Episode` into flat arrays."""
    if len(episodes) == 0:
        raise ValueError("stack_episodes requires at least one episode")
    keys = ["observations", "next_observations", "terminals"]
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        out[key] = np.concatenate([getattr(e, key) for e in episodes], axis=0)
    if all(e.actions is not None for e in episodes):
        out["actions"] = np.concatenate([e.actions for e in episodes], axis=0)
    if all(e.rewards is not None for e in episodes):
        out["rewards"] = np.concatenate([e.rewards for e in episodes], axis=0)
    out["episode_ids"] = np.concatenate(
        [np.full(len(e), i, dtype=np.int64) for i, e in enumerate(episodes)], axis=0
    )
    out["episode_lengths"] = np.asarray([len(e) for e in episodes], dtype=np.int64)
    return out


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer(object):
    """Flat offline dataset with state / transition / trajectory sampling.

    Parameters
    ----------
    observations:
        ``(N, obs_dim)`` state array (this is the *policy* observation space).
    actions:
        ``(N, action_dim)`` action array (optional; only needed for RL training).
    rewards:
        ``(N,)`` dataset rewards (not used by FRE, needed by baselines).
    next_observations:
        ``(N, obs_dim)`` successor states, defaults to a shifted copy.
    terminals / timeouts:
        ``(N,)`` float masks.  ``terminals`` is ``True`` on the last transition of
        an episode that does *not* continue (D4RL ``terminals``); ``timeouts`` is
        ``True`` on truncation by the time limit.  Either signal ends a
        trajectory for the purpose of trajectory bookkeeping.
    physics:
        ``(N, physics_dim)`` optional auxiliary physically meaningful quantities
        appended for *encoder training only* (Appendix C.2: Walker
        ``horizontal_velocity/torso_upright/torso_height``, Cheetah ``speed``).
    trajectory_ids:
        ``(N,)`` int array mapping each transition to its episode index.  If not
        given it is inferred from ``terminals``/``timeouts``.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        next_observations: Optional[np.ndarray] = None,
        terminals: Optional[np.ndarray] = None,
        timeouts: Optional[np.ndarray] = None,
        physics: Optional[np.ndarray] = None,
        trajectory_ids: Optional[np.ndarray] = None,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        name: str = "offline",
        seed: Optional[int] = None,
        clip_actions: Optional[float] = None,
        normalize_actions: bool = False,
    ):
        self.name = name
        self.observations = _as_2d(observations)
        if actions is not None:
            self.actions = _as_2d(actions)
        else:
            self.actions = np.zeros((len(self.observations), 0), dtype=np.float32)
        if next_observations is not None:
            self.next_observations = _as_2d(next_observations)
        else:
            self.next_observations = np.concatenate(
                [self.observations[1:], self.observations[-1:]], axis=0
            )
        n = len(self.observations)
        if rewards is None:
            self.rewards = np.zeros(n, dtype=np.float32)
        else:
            self.rewards = _as_float32(np.asarray(rewards).reshape(n))
        if terminals is None:
            self.terminals = np.zeros(n, dtype=np.float32)
            self.terminals[-1] = 1.0
        else:
            self.terminals = _as_float32(np.asarray(terminals).reshape(n))
        if timeouts is None:
            self.timeouts = np.zeros(n, dtype=np.float32)
        else:
            self.timeouts = _as_float32(np.asarray(timeouts).reshape(n))
        self.physics = None if physics is None else _as_2d(physics)

        if len(self.actions) and len(self.actions) != n:
            raise ValueError("actions and observations must share the first dimension")
        for arr, nm in ((self.next_observations, "next_observations"),
                        (self.rewards, "rewards"),
                        (self.terminals, "terminals"),
                        (self.timeouts, "timeouts")):
            if len(arr) != n:
                raise ValueError("{} has length {}, expected {}".format(nm, len(arr), n))
        if self.physics is not None and len(self.physics) != n:
            raise ValueError("physics must have the same length as observations")

        if trajectory_ids is None:
            trajectory_ids = self._infer_trajectory_ids(self.terminals, self.timeouts)
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(n)
        self.num_trajectories = int(self.trajectory_ids.max()) + 1 if n > 0 else 0
        self._trajectory_slices = self._build_trajectory_index()

        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self.clip_actions = clip_actions
        if clip_actions is not None and len(self.actions):
            self.actions = np.clip(self.actions, -clip_actions, clip_actions)
        if normalize_actions and len(self.actions):
            a_std = np.maximum(self.actions.std(axis=0), 1e-6)
            self.actions = _as_float32(self.actions / a_std)

        max_len = 0
        for start, end in self._trajectory_slices:
            max_len = max(max_len, end - start)
        self.max_episode_steps = int(max_len)

        self.rng = np.random.RandomState(0 if seed is None else int(seed))
        self._stats: Optional[DatasetStats] = None

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_trajectory_ids(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
        """Split a flat transition array into episodes at done/truncation signals."""
        n = len(terminals)
        ends = np.where((terminals > 0.5) | (timeouts > 0.5))[0]
        ids = np.zeros(n, dtype=np.int64)
        if len(ends) == 0:
            return ids
        # every index after an end belongs to the next episode
        starts = np.concatenate([[0], ends[:-1] + 1])
        for traj, (s, e) in enumerate(zip(starts, ends)):
            ids[s:e + 1] = traj
        # trailing transitions that never terminated
        last_end = ends[-1]
        if last_end + 1 < n:
            ids[last_end + 1:] = len(ends)
        return ids

    def _build_trajectory_index(self) -> List[Tuple[int, int]]:
        slices: List[Tuple[int, int]] = []
        order = np.argsort(self.trajectory_ids, kind="stable")
        sorted_ids = self.trajectory_ids[order]
        if len(sorted_ids) == 0:
            return slices
        boundaries = np.where(np.diff(sorted_ids) != 0)[0] + 1
        groups = np.split(order, boundaries)
        for g in groups:
            slices.append((int(g.min()), int(g.max()) + 1))
        return slices

    @classmethod
    def from_transitions(cls, data: Dict[str, np.ndarray], trajectory_length: Optional[int] = None,
                         **kwargs) -> "ReplayBuffer":
        """Build a buffer from a dictionary (e.g. the output of ``d4rl``).

        Keys understood: ``observations``/``states``, ``actions``, ``rewards``,
        ``next_observations``, ``terminals``, ``timeouts``, ``physics``.
        ``trajectory_length`` splits the flat array into fixed-length episodes
        (D4RL antmaze/kitchen/ExORL all use 1000-step episodes).
        """
        obs = data.get("observations", data.get("states"))
        if obs is None:
            raise KeyError("data must contain 'observations' or 'states'")
        n = len(obs)
        if trajectory_length is not None and "terminals" not in data and "timeouts" not in data:
            trajectory_ids = np.arange(n, dtype=np.int64) // int(trajectory_length)
        else:
            trajectory_ids = data.get("trajectory_ids")
        return cls(
            observations=obs,
            actions=data.get("actions"),
            rewards=data.get("rewards"),
            next_observations=data.get("next_observations"),
            terminals=data.get("terminals"),
            timeouts=data.get("timeouts"),
            physics=data.get("physics", data.get("extra")),
            trajectory_ids=trajectory_ids,
            **kwargs
        )

    @classmethod
    def from_episodes(cls, episodes: Sequence[Episode], physics: Optional[Sequence[np.ndarray]] = None,
                      **kwargs) -> "ReplayBuffer":
        flat = stack_episodes(episodes)
        if physics is not None:
            flat["physics"] = np.concatenate([_as_2d(p) for p in physics], axis=0)
        return cls(
            observations=flat["observations"],
            actions=flat.get("actions"),
            rewards=flat.get("rewards"),
            next_observations=flat["next_observations"],
            terminals=flat["terminals"],
            physics=flat.get("physics"),
            trajectory_ids=flat["episode_ids"],
            **kwargs
        )

    @classmethod
    def from_config(cls, config, **kwargs) -> "ReplayBuffer":
        """Load the offline dataset described by a ``fre.config`` object.

        The heavyweight ``d4rl`` / ExORL readers live in ``fre.envs.d4rl_loader``
        and are imported lazily so that this module stays import-light.
        """
        from fre.envs.d4rl_loader import load_offline_dataset  # local import (heavy deps)

        dataset = load_offline_dataset(getattr(config, "domain", "antmaze"), config=config, **kwargs)
        return dataset

    # ------------------------------------------------------------------ #
    # basic properties
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(len(self.observations))

    @property
    def num_transitions(self) -> int:
        return int(len(self.observations))

    @property
    def num_states(self) -> int:
        return int(len(self.observations))

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def physics_dim(self) -> int:
        return 0 if self.physics is None else int(self.physics.shape[1])

    @property
    def has_actions(self) -> bool:
        return self.action_dim > 0

    @property
    def has_physics(self) -> bool:
        return self.physics is not None

    # ------------------------------------------------------------------ #
    # duck-typed view used by ``fre.fre.prior``
    # ------------------------------------------------------------------ #
    @property
    def states(self) -> np.ndarray:
        """All states of the offline dataset, shape ``(N, obs_dim)``."""
        return self.observations

    @property
    def state_mean(self) -> np.ndarray:
        return self.observations.mean(axis=0)

    @property
    def state_std(self) -> np.ndarray:
        return np.maximum(self.observations.std(axis=0), 1e-6)

    # alias used by some prior implementations
    @property
    def obs_std(self) -> np.ndarray:
        return self.state_std

    @property
    def action_mean(self) -> np.ndarray:
        if not self.has_actions:
            return np.zeros(0, dtype=np.float32)
        return self.actions.mean(axis=0)

    @property
    def action_std(self) -> np.ndarray:
        if not self.has_actions:
            return np.zeros(0, dtype=np.float32)
        return np.maximum(self.actions.std(axis=0), 1e-6)

    @property
    def action_low(self) -> np.ndarray:
        if not self.has_actions:
            return np.zeros(0, dtype=np.float32)
        return self.actions.min(axis=0)

    @property
    def action_high(self) -> np.ndarray:
        if not self.has_actions:
            return np.zeros(0, dtype=np.float32)
        return self.actions.max(axis=0)

    # ------------------------------------------------------------------ #
    # physics augmentation (Appendix C.2)
    # ------------------------------------------------------------------ #
    def augmented_states(self, use_physics: bool = True) -> np.ndarray:
        """Encoder input: ``[observation, physics]`` when physics is available."""
        if use_physics and self.physics is not None:
            return np.concatenate([self.observations, self.physics], axis=-1)
        return self.observations

    @property
    def encoder_state_dim(self) -> int:
        return self.obs_dim + self.physics_dim

    def with_physics(self, physics: np.ndarray, inplace: bool = False) -> "ReplayBuffer":
        """Attach (or replace) the auxiliary physics features used by the encoder."""
        physics = _as_2d(physics)
        if inplace:
            self.physics = physics
            self._stats = None
            return self
        return ReplayBuffer(
            observations=self.observations,
            actions=self.actions if self.has_actions else None,
            rewards=self.rewards,
            next_observations=self.next_observations,
            terminals=self.terminals,
            timeouts=self.timeouts,
            physics=physics,
            trajectory_ids=self.trajectory_ids,
            reward_scale=self.reward_scale,
            reward_shift=self.reward_shift,
            name=self.name,
            clip_actions=self.clip_actions,
        )

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def sample_indices(self, batch_size: int, replace: bool = True,
                       rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        return rng.randint(0, len(self), size=int(batch_size))

    def sample_states(self, num: int, replace: bool = True,
                      rng: Optional[np.random.RandomState] = None,
                      device=None, to_torch: bool = False,
                      use_physics: bool = False):
        """Uniformly sample ``num`` states from the dataset.

        Used for both the ``K=32`` encoder states and the ``K'=8`` decoder states
        (Algorithm 1).  Returns a numpy ``(num, state_dim)`` array by default;
        with ``use_physics=True`` the physics-augmented encoder state is returned.
        """
        idx = self.sample_indices(num, replace=replace, rng=rng)
        states = self.augmented_states(use_physics) if use_physics else self.observations
        out = states[idx]
        if to_torch:
            return self.as_tensor(out, device=device)
        return out

    def sample_state_tensor(self, num: int, device=None, use_physics: bool = False):
        import torch  # local import keeps the module torch-free at import time

        out = self.sample_states(num, use_physics=use_physics)
        return torch.as_tensor(out, dtype=torch.float32, device=device)

    def sample_transitions(self, batch_size: int, replace: bool = True,
                           rng: Optional[np.random.RandomState] = None,
                           as_torch: bool = False, device=None,
                           use_physics: bool = False) -> Dict[str, Union[np.ndarray, "object"]]:
        """Sample a batch of ``(s, a, r, s', done)`` transitions."""
        idx = self.sample_indices(batch_size, replace=replace, rng=rng)
        obs = self.augmented_states(use_physics) if use_physics else self.observations
        batch = {
            "observations": obs[idx],
            "actions": self.actions[idx] if self.has_actions else np.zeros((len(idx), 0), np.float32),
            "rewards": self.rewards[idx] * self.reward_scale + self.reward_shift,
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
            "timeouts": self.timeouts[idx],
            "indices": idx,
        }
        if as_torch:
            return self.as_tensors(batch, device=device)
        return batch

    def sample_trajectories(self, num: int,
                            rng: Optional[np.random.RandomState] = None,
                            device=None, to_torch: bool = False,
                            use_physics: bool = False,
                            max_length: Optional[int] = None,
                            return_lengths: bool = False):
        """Sample ``num`` full trajectories (used by the HER goal prior).

        Returns a padded ``(num, T, state_dim)`` array where ``T`` is the length
        of the longest sampled trajectory (D4RL / ExORL episodes are all 1000
        steps, so in practice no padding occurs).  Shorter trajectories are
        padded by repeating their last state so that any downstream
        ``uniform``/``argmax`` indexing remains valid.
        """
        rng = self.rng if rng is None else rng
        n_traj = self.num_trajectories
        if n_traj == 0:
            raise RuntimeError("replay buffer contains no trajectories")
        chosen = rng.randint(0, n_traj, size=int(num))
        traj_states = []
        states = self.augmented_states(use_physics) if use_physics else self.observations
        for t in chosen:
            start, end = self._trajectory_slices[int(t)]
            traj_states.append(states[start:end])
        lengths = np.asarray([len(x) for x in traj_states], dtype=np.int64)
        T = int(lengths.max()) if max_length is None else int(max_length)
        T = max(T, 1)
        out = np.zeros((len(traj_states), T, states.shape[1]), dtype=np.float32)
        for i, traj in enumerate(traj_states):
            take = min(T, len(traj))
            out[i, :take] = traj[:take]
            if take < T:
                out[i, take:] = traj[take - 1]
        if to_torch:
            out = self.as_tensor(out, device=device)
        if return_lengths:
            return out, lengths
        return out

    def sample_trajectory_indices(self, num: int, rng: Optional[np.random.RandomState] = None
                                  ) -> np.ndarray:
        rng = self.rng if rng is None else rng
        return rng.randint(0, self.num_trajectories, size=int(num))

    def sample_indices_from_trajectory(self, traj_ids: np.ndarray,
                                       rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Random transition index inside each of the given trajectories."""
        rng = self.rng if rng is None else rng
        out = np.empty(len(traj_ids), dtype=np.int64)
        for i, t in enumerate(np.asarray(traj_ids).ravel()):
            start, end = self._trajectory_slices[int(t)]
            out[i] = rng.randint(start, end)
        return out

    # ------------------------------------------------------------------ #
    # tensors / device
    # ------------------------------------------------------------------ #
    @staticmethod
    def as_tensor(array: np.ndarray, device=None, dtype=None):
        import torch

        if dtype is None:
            dtype = torch.float32
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

    def as_tensors(self, batch: Dict[str, np.ndarray], device=None) -> Dict[str, object]:
        import torch

        out = {}
        for key, value in batch.items():
            if key in ("indices",):
                out[key] = torch.as_tensor(np.asarray(value), dtype=torch.long, device=device)
            else:
                out[key] = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
        return out

    def to(self, device):  # pragma: no cover - compatibility shim
        """No-op device move: arrays are converted on demand."""
        self._device = device
        return self

    # ------------------------------------------------------------------ #
    # statistics / utilities
    # ------------------------------------------------------------------ #
    def statistics(self, recompute: bool = False) -> DatasetStats:
        if self._stats is not None and not recompute:
            return self._stats
        self._stats = DatasetStats(
            num_transitions=self.num_transitions,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            obs_mean=self.observations.mean(axis=0),
            obs_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            reward_min=float(self.rewards.min()) if len(self) else 0.0,
            reward_max=float(self.rewards.max()) if len(self) else 0.0,
            reward_mean=float(self.rewards.mean()) if len(self) else 0.0,
            reward_std=float(self.rewards.std()) if len(self) else 0.0,
            num_trajectories=int(self.num_trajectories),
            max_episode_steps=int(self.max_episode_steps),
        )
        return self._stats

    def normalization(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(mean, std)`` used to standardize states for goal distances.

        Appendix C.2: "Each state dimension is normalized according to the
        standard deviation along that dimension within the offline dataset."
        """
        return self.observations.mean(axis=0), self.state_std

    def subset(self, indices: np.ndarray, name: Optional[str] = None) -> "ReplayBuffer":
        """Return a buffer containing only the given transition indices."""
        indices = np.asarray(indices, dtype=np.int64)
        physics = None if self.physics is None else self.physics[indices]
        return ReplayBuffer(
            observations=self.observations[indices],
            actions=self.actions[indices] if self.has_actions else None,
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminals=self.terminals[indices],
            timeouts=self.timeouts[indices],
            physics=physics,
            reward_scale=self.reward_scale,
            reward_shift=self.reward_shift,
            name=self.name if name is None else name,
            clip_actions=self.clip_actions,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return ("ReplayBuffer(name={}, transitions={}, trajectories={}, obs_dim={}, "
                "action_dim={}, physics_dim={})").format(
            self.name, self.num_transitions, self.num_trajectories,
            self.obs_dim, self.action_dim, self.physics_dim)


def make_replay_buffer(config, state_dim: Optional[int] = None, **kwargs) -> ReplayBuffer:
    """Convenience constructor used by ``fre/main.py`` and the training scripts.

    ``state_dim`` is accepted for interface symmetry with the model factories; it
    is validated against the loaded dataset only when provided.
    """
    buffer = ReplayBuffer.from_config(config, **kwargs)
    if state_dim is not None and buffer.obs_dim != int(state_dim):
        raise ValueError(
            "dataset obs_dim={} does not match the requested state_dim={}".format(
                buffer.obs_dim, state_dim)
        )
    return buffer
