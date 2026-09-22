"""Offline dataset containers, state normalisation, and HER goal sampling.

FRE trains on an *unlabeled* dataset of trajectories
``(s_0, a_0, s_1, a_1, ..., s_T)`` with no reward annotations (Section 3).
Two things are needed from the dataset:

  1. random states, used as encoding / decoding states for the FRE objective;
  2. goals, sampled with the hindsight relabelling distribution from Appendix B
     (0.2 current state, 0.5 future state within the trajectory, 0.3 uniform
     random dataset state).

Dataset loaders for the three domains are provided:

  * AntMaze: ``antmaze-large-diverse-v2`` from D4RL (Appendix C.1).
  * ExORL: ``cheetah`` / ``walker`` RND datasets (Appendix C.2, addendum).
  * Kitchen: the D4RL Kitchen environment (Appendix C.3).

The loaders import ``gym``/``d4rl``/``h5py`` lazily so that this module (and
the tests) can be imported in environments where those packages are absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
class RunningNormalizer:
    """Mean/std normalisation of observations, computed over the offline dataset."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, data: np.ndarray) -> "RunningNormalizer":
        data = np.asarray(data, dtype=np.float64)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean, std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean

    def to(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.mean, device=device),
            torch.as_tensor(self.std, device=device),
        )

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}


# --------------------------------------------------------------------------------------
# Dataset container
# --------------------------------------------------------------------------------------
@dataclass
class OfflineDataset:
    """Container for offline trajectories plus pre-flattened transitions.

    Attributes:
        observations: ``(N, obs_dim)`` all states in the dataset.
        actions: ``(N, act_dim)`` all actions.
        next_observations: ``(N, obs_dim)`` successor states.
        terminals: ``(N,)`` done flags (per timestep, not per trajectory).
        traj_ids: ``(N,)`` index of the trajectory each transition belongs to.
        traj_offsets: ``(num_traj + 1,)`` start index of each trajectory inside
            the flattened arrays, used for the HER future-state sampling.
        encoder_observations: optional per-transition auxiliary features that
            are appended for *encoder training only* (Appendix C.2: the ExORL
            physics values that the true reward functions depend on).
    """

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    traj_ids: np.ndarray
    traj_lengths: np.ndarray
    traj_offsets: np.ndarray
    encoder_observations: Optional[np.ndarray] = None
    next_encoder_observations: Optional[np.ndarray] = None
    normalizer: Optional[RunningNormalizer] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    # Optional fixed goal states (ExORL goal-reaching tasks use five states
    # sampled from the offline dataset and kept fixed across evaluation).
    goal_states: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.next_observations = np.asarray(self.next_observations, dtype=np.float32)
        self.terminals = np.asarray(self.terminals, dtype=np.float32)
        self.traj_ids = np.asarray(self.traj_ids, dtype=np.int64)
        self.traj_lengths = np.asarray(self.traj_lengths, dtype=np.int64)
        self.traj_offsets = np.asarray(self.traj_offsets, dtype=np.int64)
        if self.encoder_observations is None:
            self.encoder_observations = self.observations
        if self.next_encoder_observations is None:
            self.next_encoder_observations = self.encoder_observations

    # -- shapes ----------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def encoder_obs_dim(self) -> int:
        return int(self.encoder_observations.shape[1])

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_trajectories(self) -> int:
        return int(len(self.traj_lengths))

    def __len__(self) -> int:
        return self.size

    # -- sampling --------------------------------------------------------------
    def random_states(self, n: int, rng: Optional[np.random.Generator] = None,
                      encoder: bool = True) -> torch.Tensor:
        """Uniformly sample ``n`` states from the offline dataset."""
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=n)
        source = self.encoder_observations if encoder else self.observations
        return torch.as_tensor(np.asarray(source[idx], dtype=np.float32))

    def sample_goals(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        p_current: float = 0.2,
        p_future: float = 0.5,
        p_random: float = 0.3,
        encoder: bool = True,
    ) -> torch.Tensor:
        """Sample goals with the HER distribution from Appendix B.

        Given a randomly selected state, that state is used as the goal with
        probability ``p_current``, a future state within the same trajectory
        with probability ``p_future``, and a completely random dataset state
        with probability ``p_random``.
        """
        rng = rng or np.random.default_rng()
        source = self.encoder_observations if encoder else self.observations
        base = rng.integers(0, self.size, size=n)
        choice = rng.random(n)
        goals = np.empty((n, source.shape[1]), dtype=np.float32)

        current_mask = choice < p_current
        future_mask = (choice >= p_current) & (choice < p_current + p_future)
        random_mask = choice >= p_current + p_future

        # current state
        goals[current_mask] = source[base[current_mask]]
        # uniform random dataset state
        if random_mask.any():
            rand_idx = rng.integers(0, self.size, size=int(random_mask.sum()))
            goals[random_mask] = source[rand_idx]
        # future state inside the same trajectory
        if future_mask.any():
            rows = base[future_mask]
            tids = self.traj_ids[rows]
            offsets = self.traj_offsets[tids]
            lengths = self.traj_lengths[tids]
            pos = rows - offsets
            # Geometric-ish "future" sampling: pick uniformly among indices
            # strictly after the current position within the trajectory.
            remaining = np.maximum(lengths - pos - 1, 0)
            sampled = np.where(
                remaining > 0,
                pos + 1 + (rng.random(remaining.shape) * remaining).astype(np.int64),
                pos,
            )
            goals[future_mask] = source[offsets + sampled]
        return torch.as_tensor(goals)

    def goal_sampler(self, rng: Optional[np.random.Generator] = None):
        """Return a callable ``n -> (n, state_dim)`` tensor sampler for the prior."""
        rng = rng or np.random.default_rng()

        def _sample(n: int) -> torch.Tensor:
            return self.sample_goals(n, rng=rng)

        return _sample

    def sample_transitions(
        self, batch_size: int, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample an i.i.d. batch of ``(s, a, s', done)`` transitions.

        Both the base observation (used by the RL networks) and the encoder
        observation (base observation plus any auxiliary physics features, used
        by the reward functions and the FRE encoder) are returned.
        """
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[idx]),
            "encoder_observations": torch.as_tensor(self.encoder_observations[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "next_observations": torch.as_tensor(self.next_observations[idx]),
            "next_encoder_observations": torch.as_tensor(self.encoder_observations[idx] if
                                                         self.next_encoder_observations is None
                                                         else self.next_encoder_observations[idx]),
            "terminals": torch.as_tensor(self.terminals[idx]),
        }


def build_trajectory_index(traj_lengths: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-transition trajectory ids and per-trajectory offsets."""
    lengths = np.asarray(traj_lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    traj_ids = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    return traj_ids, offsets


def from_trajectories(
    trajectories: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    rewards: Optional[Sequence[np.ndarray]] = None,
    encoder_features: Optional[Sequence[np.ndarray]] = None,
    normalizer: Optional[RunningNormalizer] = None,
) -> OfflineDataset:
    """Assemble an :class:`OfflineDataset` from a list of trajectories.

    Each trajectory of length ``T`` yields ``T - 1`` transitions (plus a
    terminal transition at the end, matching the D4RL convention).
    """
    obs_list, act_list, next_list, term_list = [], [], [], []
    enc_list = []
    next_enc_list = []
    lengths = []
    for i, (obs, act) in enumerate(zip(trajectories, actions)):
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        T = obs.shape[0]
        if T < 2:
            continue
        obs_list.append(obs[:-1])
        act_list.append(act[:-1])
        next_list.append(obs[1:])
        terminal = np.zeros(T - 1, dtype=np.float32)
        terminal[-1] = 1.0
        term_list.append(terminal)
        if encoder_features is not None:
            enc = np.asarray(encoder_features[i], dtype=np.float32)
            enc_list.append(enc[:-1])
            next_enc_list.append(enc[1:])
        lengths.append(T - 1)

    observations = np.concatenate(obs_list, axis=0)
    actions_arr = np.concatenate(act_list, axis=0)
    next_observations = np.concatenate(next_list, axis=0)
    terminals = np.concatenate(term_list, axis=0)
    encoder_obs = np.concatenate(enc_list, axis=0) if enc_list else observations
    next_encoder_obs = np.concatenate(next_enc_list, axis=0) if next_enc_list else next_observations

    traj_ids, offsets = build_trajectory_index(lengths)

    if normalizer is None:
        normalizer = RunningNormalizer.fit(observations)

    return OfflineDataset(
        observations=observations,
        actions=actions_arr,
        next_observations=next_observations,
        terminals=terminals,
        traj_ids=traj_ids,
        traj_lengths=np.asarray(lengths),
        traj_offsets=offsets,
        encoder_observations=encoder_obs,
        next_encoder_observations=next_encoder_obs,
        normalizer=normalizer,
    )


# --------------------------------------------------------------------------------------
# Domain loaders
# --------------------------------------------------------------------------------------
def load_d4rl_dataset(env_name: str, max_episodes: Optional[int] = None) -> OfflineDataset:
    """Load a D4RL dataset (AntMaze ``antmaze-large-diverse-v2``, Kitchen, ...).

    Requires ``gym`` and ``d4rl`` (``pip install git+https://github.com/Farama-Foundation/d4rl``).
    The addendum recommends using a D4RL revision from before June 2024 for
    reproducibility.
    """
    try:
        import d4rl  # noqa: F401
        import gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "D4RL/gym are required to load AntMaze or Kitchen datasets. "
            "Install d4rl from a pre-June-2024 commit (see README)."
        ) from exc

    env = gym.make(env_name)
    raw = env.get_dataset()
    N = raw["observations"].shape[0]
    terminals = raw["terminals"]
    timeouts = raw.get("timeouts", np.zeros_like(terminals))
    dones = np.logical_or(terminals, timeouts)
    episode_ends = np.nonzero(dones)[0] + 1
    start = 0
    trajectories, actions, rewards = [], [], []
    for i, end in enumerate(episode_ends):
        if max_episodes is not None and i >= max_episodes:
            break
        trajectories.append(raw["observations"][start:end])
        actions.append(raw["actions"][start:end])
        rewards.append(raw["rewards"][start:end])
        start = end
    return from_trajectories(trajectories, actions, rewards)


def load_exorl_dataset(
    domain: str,
    algo: str = "rnd",
    data_dir: Optional[str] = None,
    max_episodes: Optional[int] = None,
    append_physics: bool = True,
) -> OfflineDataset:
    """Load an ExORL dataset (``walker`` or ``cheetah``, RND protocol).

    ExORL ships one ``.npz`` file per episode under
    ``<data_dir>/<domain>/<algo>/buffer/`` (Yarats et al., 2022), each holding
    ``observation``, ``action``, ``physics`` (the raw MuJoCo state),
    ``discount`` and rewards.  The addendum specifies that all ExORL runs use
    the **RND** dataset for each domain, and Appendix C.2 requires the
    physics-derived features to be appended to the observations *for encoder
    training only*; ``append_physics=True`` performs that augmentation and
    stores it in ``OfflineDataset.encoder_observations``.

    A single concatenated ``.npz`` (keys ``observation`` / ``action`` /
    ``physics``) is accepted as a fallback.
    """
    from fre.tasks.exorl import augment_with_physics

    if data_dir is None:
        data_dir = os.environ.get("EXORL_DATA_DIR", os.path.expanduser("~/.exorl"))

    buffer_dir = os.path.join(data_dir, domain, algo, "buffer")
    trajectories: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    physics_list: List[np.ndarray] = []

    if os.path.isdir(buffer_dir):
        episode_files = sorted(f for f in os.listdir(buffer_dir) if f.endswith(".npz"))
        if max_episodes is not None:
            episode_files = episode_files[:max_episodes]
        if not episode_files:
            raise FileNotFoundError(f"No episode .npz files found in {buffer_dir}")
        for fname in episode_files:
            with np.load(os.path.join(buffer_dir, fname)) as payload:
                episode = {k: payload[k] for k in payload.keys()}
            trajectories.append(np.asarray(episode["observation"], dtype=np.float32))
            actions_list.append(np.asarray(episode["action"], dtype=np.float32))
            if "physics" in episode:
                physics_list.append(np.asarray(episode["physics"], dtype=np.float64))
    else:
        candidates = [
            os.path.join(data_dir, f"{domain}_{algo}.npz"),
            os.path.join(data_dir, domain, f"{algo}.npz"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                f"Could not find an ExORL dataset for domain={domain} algo={algo}. "
                f"Expected per-episode files under {buffer_dir} or an npz at {candidates}. "
                "Set EXORL_DATA_DIR or pass data_dir=."
            )
        with np.load(path) as payload:
            observations = np.asarray(payload["observation"], dtype=np.float32)
            actions = np.asarray(payload["action"], dtype=np.float32)
            physics = np.asarray(payload["physics"], dtype=np.float64) if "physics" in payload else None
        if physics is not None and observations.shape[0] == physics.shape[0]:
            episode_ends = payload["episode_end"] if "episode_end" in payload else None
            if episode_ends is None:
                trajectories = [observations]
                actions_list = [actions]
                physics_list = [physics]
            else:
                starts = np.concatenate([[0], np.asarray(episode_ends)[:-1]])
                for s, e in zip(starts, np.asarray(episode_ends)):
                    trajectories.append(observations[s : e + 1])
                    actions_list.append(actions[s : e + 1])
                    physics_list.append(physics[s : e + 1])
        else:
            trajectories = [observations]
            actions_list = [actions]

    # ExORL episode arrays hold T + 1 entries: index 0 is the reset state and
    # ``action`` carries a dummy value in its first slot (see the ExORL replay
    # buffer, which samples ``idx in [1, T]``).
    norm_traj, norm_act = [], []
    for obs, act in zip(trajectories, actions_list):
        if obs.shape[0] == act.shape[0]:
            obs, act = obs[1:], act[1:]
        elif obs.shape[0] == act.shape[0] + 1:
            obs = obs[1:]
        norm_traj.append(obs)
        norm_act.append(act)

    encoder_features = None
    if append_physics and physics_list:
        # Align physics with the trimmed trajectories.
        trimmed_physics = []
        for phys, obs in zip(physics_list, norm_traj):
            trimmed_physics.append(phys[phys.shape[0] - obs.shape[0] :])
        encoder_features = [
            augment_with_physics(domain, obs, phys)
            for obs, phys in zip(norm_traj, trimmed_physics)
        ]

    return from_trajectories(
        norm_traj,
        norm_act,
        encoder_features=encoder_features,
    )


def load_kitchen_dataset(env_name: str = "kitchen-complete-v0") -> OfflineDataset:
    """Load the D4RL Kitchen dataset used for the Kitchen evaluation tasks."""
    return load_d4rl_dataset(env_name)
