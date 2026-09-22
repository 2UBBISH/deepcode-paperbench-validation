"""Montezuma's Revenge: M2 behavioral-cloning pretraining and fine-tuning.

Reproduces the M1 -> M2 pipeline of Wołczyk et al. (2024), Section 3 and Appendix B.2:

* M1 is a PPO + Random Network Distillation agent trained from scratch until the
  episode cumulative reward reaches ~7000 (``src.montezuma.m1_train``).
* More than 500 trajectories are collected with M1.  In the main-text setting the
  pre-training distribution covers "rooms from a certain room onward", i.e. the
  trajectories are truncated to **Room 7 and subsequent rooms** (FAR states),
  while the preceding rooms are CLOSE.
* M2 is obtained by behavioral cloning on those trajectories.  Following
  Appendix C.2 of the paper the behavioral-cloning objective is the KL
  divergence between the student action distribution and the pre-trained
  (teacher) distribution:

      L_BC(theta) = E_{s ~ B}[ D_KL^s(pi_theta || pi_*) ].

  M2 therefore plays the role of ``pi_*`` for the fine-tuning stage.
* Finally M2 is fine-tuned on the **whole game** (starting from Room 1) with PPO
  + RND, optionally with the BC distillation loss as an auxiliary actor-only
  term (weighted by a tunable KL coefficient, ablated in Figure 13).  The
  from-scratch baseline never sees BC data.

A room is considered *completed* when at least one of the following happens:
a coin is earned, a new item is acquired, or the room is exited through a
different passage than the one used to enter it (Appendix B.2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional heavy dependencies (kept importable for docs / smoke tests)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on environment
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover - depends on environment
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical as _Categorical
    from torch.distributions import kl_divergence as _kl_divergence

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _Categorical = None  # type: ignore
    _kl_divergence = None  # type: ignore
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "Montezuma M2 behavioral cloning requires PyTorch. "
            "Install torch to run this module."
        )


# ---------------------------------------------------------------------------
# Environment / model / agent imports (defensive: several import layouts)
# ---------------------------------------------------------------------------
def _import_montezuma_env():
    try:  # packaged layout
        from src.montezuma import env as env_mod  # type: ignore

        return env_mod
    except Exception:
        pass
    try:  # relative layout
        from . import env as env_mod  # type: ignore

        return env_mod
    except Exception:
        pass
    try:
        import env as env_mod  # type: ignore  # noqa: F401

        return env_mod
    except Exception:  # pragma: no cover
        return None


def _import_ppo_rnd():
    try:
        from src.montezuma import ppo_rnd as mod  # type: ignore

        return mod
    except Exception:
        pass
    try:
        from . import ppo_rnd as mod  # type: ignore

        return mod
    except Exception:
        pass
    try:
        import ppo_rnd as mod  # type: ignore  # noqa: F401

        return mod
    except Exception:  # pragma: no cover
        return None


def _import_m1_train():
    try:
        from src.montezuma import m1_train as mod  # type: ignore

        return mod
    except Exception:
        pass
    try:
        from . import m1_train as mod  # type: ignore

        return mod
    except Exception:
        pass
    try:
        import m1_train as mod  # type: ignore  # noqa: F401

        return mod
    except Exception:  # pragma: no cover
        return None


def _import_retention():
    """Import the shared retention losses (EWC used for the Montezuma ablation)."""
    out: Dict[str, Any] = {}
    for key, module_name, attrs in (
        ("ewc", "src.retention.ewc", ("EWC", "ewc_coef_for")),
        (
            "behavioral_cloning",
            "src.retention.behavioral_cloning",
            ("BehavioralCloning", "bc_coef_for"),
        ),
    ):
        mod = None
        for path in (module_name, "." + module_name.split(".")[-1]):
            try:
                if path.startswith("."):
                    from importlib import import_module

                    mod = import_module(path, __package__)
                else:
                    from importlib import import_module

                    mod = import_module(path)
                break
            except Exception:
                mod = None
        if mod is not None:
            out[key] = mod
            for attr in attrs:
                if hasattr(mod, attr):
                    out[attr] = getattr(mod, attr)
    return out


_ENV = _import_montezuma_env()
_PPO = _import_ppo_rnd()
_M1 = _import_m1_train()
_RET = _import_retention()


def _env_attr(name: str, default: Any = None) -> Any:
    return getattr(_ENV, name, default) if _ENV is not None else default


ROOM7 = int(_env_attr("ROOM7", 7))
NUM_ACTIONS = int(_env_attr("NUM_ACTIONS", 18))
OBS_SHAPE = tuple(_env_attr("OBS_SHAPE", (4, 84, 84)))
MAX_STEPS_PER_EPISODE = int(_env_attr("MAX_STEPS_PER_EPISODE", 4500))

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------
NUM_BC_TRAJECTORIES = 500          # Appendix B.2: "more than 500 trajectories"
MIN_ROOM = ROOM7                   # main text: pre-training starts from Room 7
DEFAULT_BC_KL_WEIGHT = 1.0         # chosen via the Figure 13 sweep
KL_WEIGHT_GRID: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)
DEFAULT_BC_EPOCHS = 5
DEFAULT_BC_BATCH_SIZE = 256
DEFAULT_BC_LR = 1e-4
DEFAULT_BC_UPDATES = 10_000
DEFAULT_FINETUNE_STEPS = 200_000_000
ROOM7_EVAL_EVERY = 5_000_000      # room-7 success rate logged every 5M steps
DEFAULT_EVAL_EPISODES = 100
BC_FREE_SCRATCH = "scratch"

__all__ = [
    "M2BCConfig",
    "BCDataset",
    "M2PretrainResult",
    "M2FinetuneResult",
    "build_bc_dataset",
    "build_bc_dataset_from_m1",
    "truncate_to_min_room",
    "room_from_transition",
    "bc_kl_loss",
    "bc_cross_entropy_loss",
    "make_bc_aux_loss",
    "pretrain_m2",
    "finetune_m2",
    "train_from_scratch",
    "evaluate_room7_success_rate",
    "sweep_kl_weight",
    "run_pipeline",
    "main",
    "build_parser",
    "NUM_BC_TRAJECTORIES",
    "MIN_ROOM",
    "DEFAULT_BC_KL_WEIGHT",
    "KL_WEIGHT_GRID",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class M2BCConfig:
    """Hyperparameters of the M2 behavioral-cloning / fine-tuning pipeline."""

    # --- data collection ---
    num_trajectories: int = NUM_BC_TRAJECTORIES
    min_room: int = MIN_ROOM
    truncate_to_far: bool = True
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE
    max_bc_samples: Optional[int] = None
    store_teacher_logits: bool = False

    # --- behavioral cloning ---
    bc_epochs: int = DEFAULT_BC_EPOCHS
    bc_updates: int = DEFAULT_BC_UPDATES
    bc_batch_size: int = DEFAULT_BC_BATCH_SIZE
    bc_lr: float = DEFAULT_BC_LR
    bc_loss_type: str = "kl"        # "kl" (distillation, Appendix C.2) or "ce"
    bc_grad_clip: Optional[float] = 10.0

    # --- fine-tuning ---
    method: str = "bc"              # scratch | none | bc | ewc
    kl_weight: float = DEFAULT_BC_KL_WEIGHT
    ewc_coef: float = 1.0
    total_steps: int = DEFAULT_FINETUNE_STEPS
    eval_every: int = ROOM7_EVAL_EVERY
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    save_every: int = 25_000_000

    # --- misc ---
    device: str = "cpu"
    seed: int = 0
    stub: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def with_overrides(self, **overrides: Any) -> "M2BCConfig":
        cfg = M2BCConfig(**self.to_dict())
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "M2BCConfig":
        """Build config from a loaded YAML mapping (``configs/montezuma.yaml``)."""
        kwargs: Dict[str, Any] = {}

        def _get(container: Any, *keys: str) -> Any:
            if container is None:
                return None
            for key in keys:
                if isinstance(container, dict) and key in container:
                    return container[key]
                if hasattr(container, key):
                    return getattr(container, key)
            return None

        data_blocks = [
            _get(cfg, "bc", "m2_bc", "behavioral_cloning"),
            _get(cfg, "montezuma"),
            cfg,
        ]
        aliases = {
            "num_trajectories": ("num_trajectories", "trajectories", "num_trajs"),
            "min_room": ("min_room", "start_room", "far_room"),
            "truncate_to_far": ("truncate_to_far", "truncate"),
            "bc_epochs": ("epochs", "bc_epochs"),
            "bc_updates": ("updates", "bc_updates", "train_steps"),
            "bc_batch_size": ("batch_size", "bc_batch_size"),
            "bc_lr": ("lr", "learning_rate", "bc_lr"),
            "bc_loss_type": ("loss_type", "bc_loss_type"),
            "kl_weight": ("kl_weight", "kl_coef", "bc_kl_weight", "coef"),
            "method": ("method",),
            "total_steps": ("total_steps", "num_steps", "finetune_steps"),
            "eval_every": ("eval_every", "room7_every"),
            "eval_episodes": ("eval_episodes",),
            "save_every": ("save_every",),
            "device": ("device",),
            "seed": ("seed",),
            "stub": ("stub",),
        }
        for field_name, keys in aliases.items():
            for block in data_blocks:
                value = _get(block, *keys)
                if value is not None:
                    kwargs[field_name] = value
                    break
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Trajectory / transition helpers
# ---------------------------------------------------------------------------
def _transition_field(transition: Any, key: str, default: Any = None) -> Any:
    """Fetch a field from a dict / namedtuple / attribute-carrying object."""
    if transition is None:
        return default
    if isinstance(transition, dict):
        if key in transition:
            return transition[key]
        info = transition.get("info")
        if isinstance(info, dict) and key in info:
            return info[key]
        return default
    if hasattr(transition, key):
        return getattr(transition, key)
    info = getattr(transition, "info", None)
    if isinstance(info, dict) and key in info:
        return info[key]
    if isinstance(transition, (tuple, list)):
        index = {"obs": 0, "observation": 0, "action": 1, "reward": 2,
                 "next_obs": 3, "done": 4}.get(key)
        if index is not None and len(transition) > index:
            return transition[index]
    return default


def room_from_transition(transition: Any, default: Optional[int] = None) -> Optional[int]:
    """Room index stored on a transition (RAM-derived ``info['room']``)."""
    value = _transition_field(transition, "room", None)
    if value is None:
        value = _transition_field(transition, "room_number", None)
    if value is None:
        value = _transition_field(transition, "dlvl", None)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def truncate_to_min_room(
    trajectory: Sequence[Any], min_room: int = MIN_ROOM
) -> List[Any]:
    """Keep only the suffix of ``trajectory`` whose rooms are ``>= min_room``.

    This implements the paper's "pre-train on a part of the environment that
    includes only rooms from a certain room onward" (Section 3 / Appendix B.2).
    Transitions with unknown room information are kept only if at least one
    previous transition in the same trajectory was already at ``>= min_room``.
    """
    kept: List[Any] = []
    started = False
    for transition in trajectory:
        room = room_from_transition(transition, None)
        if room is None:
            if started:
                kept.append(transition)
            continue
        if room >= min_room:
            started = True
        if started:
            kept.append(transition)
    return kept


def _stack_obs(obs_list: Sequence[Any]) -> Any:
    """Stack observations into a contiguous array (uint8 when possible)."""
    if _np is None:
        return list(obs_list)
    array = _np.asarray(obs_list)
    if array.dtype == object:  # heterogeneous (e.g. dict observations)
        return _np.stack([_np.asarray(o, dtype=_np.float32) for o in obs_list])
    return array


class BCDataset:
    """Buffer ``B`` of pre-training states/actions used by behavioral cloning.

    Stores observations (stacked frames), the actions taken by M1 and, optionally,
    pre-computed teacher logits so that the KL distillation loss can be evaluated
    without keeping the teacher in memory.
    """

    def __init__(
        self,
        observations: Any = None,
        actions: Any = None,
        rooms: Any = None,
        trajectory_ids: Any = None,
        teacher_logits: Any = None,
        teacher_probs: Any = None,
        rewards: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.rooms = rooms
        self.trajectory_ids = trajectory_ids
        self.teacher_logits = teacher_logits
        self.teacher_probs = teacher_probs
        self.rewards = rewards
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.device = device

    # -- construction -------------------------------------------------------
    @classmethod
    def from_trajectories(
        cls,
        trajectories: Sequence[Sequence[Any]],
        min_room: Optional[int] = None,
        truncate: bool = True,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BCDataset":
        obs_list: List[Any] = []
        action_list: List[int] = []
        room_list: List[int] = []
        traj_list: List[int] = []
        reward_list: List[float] = []

        for t_idx, trajectory in enumerate(trajectories):
            if trajectory is None:
                continue
            segment = (
                truncate_to_min_room(trajectory, min_room)
                if (truncate and min_room is not None)
                else list(trajectory)
            )
            for transition in segment:
                obs = _transition_field(
                    transition, "obs", _transition_field(transition, "observation", None)
                )
                if obs is None:
                    continue
                action = _transition_field(transition, "action", None)
                if action is None:
                    continue
                obs_list.append(obs)
                action_list.append(int(action))
                room_list.append(int(room_from_transition(transition, 0) or 0))
                traj_list.append(t_idx)
                reward_list.append(
                    float(_transition_field(transition, "reward", 0.0) or 0.0)
                )

        if max_samples is not None and 0 < max_samples < len(obs_list):
            step = max(1, len(obs_list) // max_samples)
            idx = list(range(0, len(obs_list), step))[:max_samples]
            obs_list = [obs_list[i] for i in idx]
            action_list = [action_list[i] for i in idx]
            room_list = [room_list[i] for i in idx]
            traj_list = [traj_list[i] for i in idx]
            reward_list = [reward_list[i] for i in idx]

        meta = dict(metadata or {})
        meta.setdefault("num_trajectories", len(trajectories))
        meta.setdefault("min_room", min_room)
        meta.setdefault("truncated", bool(truncate))
        meta.setdefault("num_samples", len(obs_list))

        dataset = cls(
            observations=_stack_obs(obs_list) if obs_list else None,
            actions=_np.asarray(action_list, dtype=_np.int64)
            if (_np is not None and action_list)
            else action_list,
            rooms=_np.asarray(room_list, dtype=_np.int64)
            if (_np is not None and room_list)
            else room_list,
            trajectory_ids=_np.asarray(traj_list, dtype=_np.int64)
            if (_np is not None and traj_list)
            else traj_list,
            rewards=_np.asarray(reward_list, dtype=_np.float32)
            if (_np is not None and reward_list)
            else reward_list,
            metadata=meta,
            device=device,
        )
        return dataset

    # -- basic container API ------------------------------------------------
    def __len__(self) -> int:
        if self.observations is None:
            return 0
        try:
            return int(len(self.observations))
        except Exception:  # pragma: no cover
            return 0

    @property
    def num_samples(self) -> int:
        return len(self)

    def room_histogram(self) -> Dict[int, int]:
        if self.rooms is None or _np is None:
            return {}
        values, counts = _np.unique(_np.asarray(self.rooms), return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def filter_room(self, min_room: int) -> "BCDataset":
        """Return a copy restricted to observations with room ``>= min_room``."""
        if self.rooms is None or _np is None or len(self) == 0:
            return self
        mask = _np.asarray(self.rooms) >= int(min_room)
        if not mask.any():
            return BCDataset(metadata=dict(self.metadata))
        return BCDataset(
            observations=_np.asarray(self.observations)[mask],
            actions=_np.asarray(self.actions)[mask] if self.actions is not None else None,
            rooms=_np.asarray(self.rooms)[mask],
            trajectory_ids=_np.asarray(self.trajectory_ids)[mask]
            if self.trajectory_ids is not None
            else None,
            teacher_logits=_np.asarray(self.teacher_logits)[mask]
            if self.teacher_logits is not None
            else None,
            teacher_probs=_np.asarray(self.teacher_probs)[mask]
            if self.teacher_probs is not None
            else None,
            rewards=_np.asarray(self.rewards)[mask] if self.rewards is not None else None,
            metadata=dict(self.metadata),
            device=self.device,
        )

    def subsample(self, n: int, seed: int = 0) -> "BCDataset":
        total = len(self)
        if n >= total or total == 0:
            return self
        rng = _np.random.default_rng(seed) if _np is not None else random.Random(seed)
        idx = rng.choice(total, size=n, replace=False) if hasattr(rng, "choice") else []
        idx = _np.sort(idx) if _np is not None else idx
        return BCDataset(
            observations=_np.asarray(self.observations)[idx],
            actions=_np.asarray(self.actions)[idx] if self.actions is not None else None,
            rooms=_np.asarray(self.rooms)[idx] if self.rooms is not None else None,
            trajectory_ids=_np.asarray(self.trajectory_ids)[idx]
            if self.trajectory_ids is not None
            else None,
            teacher_logits=_np.asarray(self.teacher_logits)[idx]
            if self.teacher_logits is not None
            else None,
            teacher_probs=_np.asarray(self.teacher_probs)[idx]
            if self.teacher_probs is not None
            else None,
            rewards=_np.asarray(self.rewards)[idx] if self.rewards is not None else None,
            metadata=dict(self.metadata),
            device=self.device,
        )

    # -- batching -----------------------------------------------------------
    def sample_indices(self, batch_size: int, generator: Any = None) -> Any:
        total = len(self)
        if total == 0:
            raise ValueError("Cannot sample from an empty BCDataset")
        batch = min(int(batch_size), total)
        if _np is not None:
            rng = generator if generator is not None else _np.random.default_rng(0)
            if hasattr(rng, "choice"):
                return rng.choice(total, size=batch, replace=batch > total)
        rng = generator if generator is not None else random.Random(0)
        return [rng.randrange(total) for _ in range(batch)]

    def sample_tensors(
        self,
        batch_size: int,
        generator: Any = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a device-ready dict of observation/action (and teacher) tensors."""
        _require_torch()
        idx = self.sample_indices(batch_size, generator)
        obs = _to_obs_tensor(self.observations[idx] if _np is not None else [self.observations[i] for i in idx])
        device = device or self.device or "cpu"
        obs = obs.to(device)
        out: Dict[str, Any] = {"obs": obs}
        if self.actions is not None:
            actions = self.actions[idx] if _np is not None else [self.actions[i] for i in idx]
            out["actions"] = torch.as_tensor(
                _np.asarray(actions), dtype=torch.long, device=device
            )
        if self.teacher_logits is not None:
            logits = (
                self.teacher_logits[idx]
                if _np is not None
                else [self.teacher_logits[i] for i in idx]
            )
            out["teacher_logits"] = torch.as_tensor(
                _np.asarray(logits), dtype=torch.float32, device=device
            )
        elif self.teacher_probs is not None:
            probs = (
                self.teacher_probs[idx]
                if _np is not None
                else [self.teacher_probs[i] for i in idx]
            )
            out["teacher_probs"] = torch.as_tensor(
                _np.asarray(probs), dtype=torch.float32, device=device
            )
        out["indices"] = idx
        return out

    def minibatches(
        self,
        batch_size: int,
        generator: Any = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        """Iterate over the dataset in mini-batches (one epoch)."""
        _require_torch()
        total = len(self)
        if total == 0:
            return
        order = list(range(total))
        if shuffle:
            rng = generator if generator is not None else random.Random(0)
            rng.shuffle(order)
        for start in range(0, total, batch_size):
            chunk = order[start : start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            obs = _to_obs_tensor(
                _np.asarray(self.observations)[chunk] if _np is not None
                else [self.observations[i] for i in chunk]
            )
            batch: Dict[str, Any] = {"obs": obs}
            if self.actions is not None:
                batch["actions"] = torch.as_tensor(
                    _np.asarray(self.actions)[chunk], dtype=torch.long
                )
            if self.teacher_logits is not None:
                batch["teacher_logits"] = torch.as_tensor(
                    _np.asarray(self.teacher_logits)[chunk], dtype=torch.float32
                )
            yield batch

    # -- teacher precomputation --------------------------------------------
    def precompute_teacher(
        self,
        teacher: Any,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> "BCDataset":
        """Store teacher logits for every state in the buffer (optional)."""
        _require_torch()
        if len(self) == 0:
            return self
        device = device or self.device or "cpu"
        logits_chunks: List[Any] = []
        with torch.no_grad():
            for start in range(0, len(self), batch_size):
                chunk = _np.asarray(self.observations)[start : start + batch_size]
                obs = _to_obs_tensor(chunk).to(device)
                out = _teacher_logits(teacher, obs)
                logits_chunks.append(out.detach().cpu().numpy())
        self.teacher_logits = _np.concatenate(logits_chunks, axis=0)
        return self

    # -- persistence --------------------------------------------------------
    def state_dict(self, include_observations: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "metadata": self.metadata,
            "num_samples": len(self),
        }
        if include_observations and self.observations is not None:
            payload["observations"] = self.observations
        for key in ("actions", "rooms", "trajectory_ids", "rewards", "teacher_logits",
                    "teacher_probs"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    def save(self, path: str, include_observations: bool = True) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        if path.endswith(".npz") and _np is not None:
            arrays = {}
            for key in ("observations", "actions", "rooms", "trajectory_ids",
                        "rewards", "teacher_logits", "teacher_probs"):
                value = getattr(self, key)
                if value is not None:
                    arrays[key] = _np.asarray(value)
            _np.savez_compressed(path, **arrays)
            meta_path = path[:-4] + ".json"
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(self.metadata, handle, indent=2)
            return path
        if _HAS_TORCH:
            torch.save(self.state_dict(include_observations), path)
        else:  # pragma: no cover
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"metadata": self.metadata}, handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "BCDataset":
        if path.endswith(".npz") and _np is not None:
            data = _np.load(path, allow_pickle=True)
            meta_path = path[:-4] + ".json"
            meta: Dict[str, Any] = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            return cls(
                observations=data.get("observations"),
                actions=data.get("actions"),
                rooms=data.get("rooms"),
                trajectory_ids=data.get("trajectory_ids"),
                rewards=data.get("rewards"),
                teacher_logits=data.get("teacher_logits"),
                teacher_probs=data.get("teacher_probs"),
                metadata=meta,
                device=device,
            )
        _require_torch()
        state = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            observations=state.get("observations"),
            actions=state.get("actions"),
            rooms=state.get("rooms"),
            trajectory_ids=state.get("trajectory_ids"),
            rewards=state.get("rewards"),
            teacher_logits=state.get("teacher_logits"),
            teacher_probs=state.get("teacher_probs"),
            metadata=state.get("metadata", {}),
            device=device,
        )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def _to_obs_tensor(obs: Any) -> Any:
    """Convert observations to a float tensor, normalising uint8 frames."""
    _require_torch()
    if _HAS_TORCH and isinstance(obs, torch.Tensor):
        tensor = obs.float()
    else:
        array = _np.asarray(obs) if _np is not None else None
        if array is None:  # pragma: no cover
            raise TypeError("Cannot convert observations to a tensor without numpy")
        tensor = torch.as_tensor(array)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
    tensor = tensor.float()
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor


def _policy_logits(policy: Any, obs: Any) -> Any:
    """Forward a Montezuma policy and extract the action logits."""
    _require_torch()
    if policy is None:
        raise ValueError("A policy is required to compute logits")
    out = None
    for method_name in ("distribution", "logits"):
        fn = getattr(policy, method_name, None)
        if method_name == "logits" and not callable(fn):
            continue
        if callable(fn) and method_name == "logits":
            try:
                out = fn(obs)
                if hasattr(out, "logits"):
                    return out.logits
                return out
            except Exception:
                out = None
    if hasattr(policy, "forward"):
        try:
            result = policy(obs)
        except Exception:
            result = None
        if result is not None:
            if isinstance(result, dict):
                for key in ("logits", "policy_logits", "pi_logits"):
                    if key in result:
                        return result[key]
            elif isinstance(result, (tuple, list)) and result:
                first = result[0]
                if hasattr(first, "logits"):
                    return first.logits
                return first
            elif hasattr(result, "logits"):
                return result.logits
            else:
                return result
    dist_fn = getattr(policy, "distribution", None)
    if callable(dist_fn):
        try:
            distribution = dist_fn(obs)
            return distribution.logits
        except Exception:
            pass
    raise TypeError("Unable to obtain action logits from the supplied policy")


def _teacher_logits(teacher: Any, obs: Any) -> Any:
    """Frozen teacher logits (no gradient tracking, eval mode preferred)."""
    _require_torch()
    if hasattr(teacher, "policy"):
        teacher = teacher.policy
    was_training = getattr(teacher, "training", None)
    if was_training:
        teacher.eval()  # type: ignore[union-attr]
    with torch.no_grad():
        logits = _policy_logits(teacher, obs)
    if was_training:
        teacher.train(  # type: ignore[union-attr]
        )
    return logits.detach()


# ---------------------------------------------------------------------------
# Behavioral-cloning losses (Appendix C.2)
# ---------------------------------------------------------------------------
def bc_kl_loss(
    student_logits: Any,
    teacher_logits: Any = None,
    actions: Any = None,
    reduction: str = "mean",
    temperature: float = 1.0,
) -> Any:
    """``D_KL^s(pi_theta || pi_*)`` for categorical action distributions.

    When ``teacher_logits`` is ``None`` the recorded expert ``actions`` are used
    as one-hot targets (standard maximum-likelihood behavioral cloning); otherwise
    the analytic KL between the student and the teacher categorical distribution
    is minimised, as specified in Appendix C.2.
    """
    _require_torch()
    student_logits = student_logits / float(temperature)
    log_student = F.log_softmax(student_logits, dim=-1)
    if teacher_logits is not None:
        teacher_logits = teacher_logits / float(temperature)
        log_teacher = F.log_softmax(teacher_logits, dim=-1)
        student = torch.distributions.Categorical(logits=student_logits)
        teacher = torch.distributions.Categorical(logits=teacher_logits)
        per_sample = torch.distributions.kl_divergence(student, teacher)
    elif actions is not None:
        per_sample = F.nll_loss(
            log_student, actions.long(), reduction="none"
        )
    else:
        raise ValueError("bc_kl_loss requires either teacher_logits or actions")
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "batchmean":
        return per_sample.mean() / max(1, int(per_sample.numel()) + 1e-12) * 1.0
    return per_sample.mean()


def bc_cross_entropy_loss(student_logits: Any, actions: Any, reduction: str = "mean") -> Any:
    """Cross-entropy (negative log-likelihood) behavioral cloning loss."""
    _require_torch()
    return F.cross_entropy(student_logits, actions.long(), reduction=reduction)


def make_bc_aux_loss(
    teacher: Any,
    dataset: Optional[BCDataset] = None,
    kl_weight: float = DEFAULT_BC_KL_WEIGHT,
    batch_size: int = DEFAULT_BC_BATCH_SIZE,
    device: Optional[str] = None,
    generator: Any = None,
    seed: int = 0,
    loss_type: str = "kl",
    max_batches_per_step: int = 1,
) -> Callable[..., Any]:
    """Build the auxiliary actor-only BC term used during M2 fine-tuning.

    The returned callable matches the hook expected by
    :class:`src.montezuma.ppo_rnd.PPORNDAgent`::

        aux_loss_fn(policy=..., obs=..., step=...) -> scalar tensor

    The loss is ``kl_weight * E_{s ~ B}[ D_KL^s(pi_theta || pi_*) ]`` evaluated on
    a mini-batch sampled from the pre-training buffer ``B`` (states from Room 7
    onward).  When ``obs`` is supplied by the PPO trainer the KL is computed on
    those online observations instead, still distilling the pre-trained policy.
    """
    _require_torch()
    device = device or "cpu"
    rng = generator if generator is not None else _np.random.default_rng(seed) if _np is not None else random.Random(seed)
    if dataset is not None and len(dataset) > 0:
        dataset.device = device
        if dataset.device is None:
            dataset.device = device

    def aux_loss_fn(policy: Any = None, obs: Any = None, step: Any = None, **kwargs: Any) -> Any:
        student = policy
        if student is None:
            raise ValueError("aux_loss_fn requires the current policy")
        target = teacher
        if hasattr(target, "policy"):
            target = target.policy
        total = None
        for _ in range(max(1, int(max_batches_per_step))):
            if dataset is not None and len(dataset) > 0:
                batch = dataset.sample_tensors(batch_size, generator=rng, device=device)
                batch_obs = batch["obs"]
                teacher_logits = batch.get("teacher_logits")
                actions = batch.get("actions")
            else:
                batch_obs = _to_obs_tensor(obs).to(device) if obs is not None else None
                teacher_logits, actions = None, None
            if batch_obs is None:
                break
            logits = _policy_logits(student, batch_obs)
            if loss_type == "ce" and teacher_logits is None and actions is not None:
                loss = bc_cross_entropy_loss(logits, actions)
            else:
                if teacher_logits is None:
                    with torch.no_grad():
                        teacher_logits = _teacher_logits(target, batch_obs)
                loss = bc_kl_loss(logits, teacher_logits=teacher_logits, actions=actions)
            total = loss if total is None else total + loss
        if total is None:
            params = getattr(student, "parameters", None)
            if callable(params):
                return sum(p.sum() * 0.0 for p in student.parameters())
            return torch.zeros((), device=device)
        return float(kl_weight) * (total / float(max(1, int(max_batches_per_step))))

    aux_loss_fn.dataset = dataset  # type: ignore[attr-defined]
    aux_loss_fn.kl_weight = float(kl_weight)  # type: ignore[attr-defined]
    return aux_loss_fn


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------
def build_bc_dataset(
    agent: Any,
    env: Any = None,
    num_trajectories: int = NUM_BC_TRAJECTORIES,
    min_room: int = MIN_ROOM,
    truncate: bool = True,
    seed: int = 0,
    stub: bool = False,
    deterministic: bool = False,
    max_steps: Optional[int] = None,
    max_samples: Optional[int] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
    record_all: bool = False,
) -> BCDataset:
    """Collect M1 trajectories and build the behavioral-cloning buffer ``B``.

    Uses :func:`src.montezuma.env.collect_trajectories` when available (which
    already supports Room-7-onward truncation) and falls back to a local rollout
    loop otherwise.
    """
    env_collect = _env_attr("collect_trajectories", None)
    trajectories: List[Any] = []
    if callable(env_collect):
        attempts = (
            dict(
                policy=agent,
                env=env,
                num_trajectories=int(num_trajectories),
                min_room=int(min_room) if truncate else None,
                max_steps=max_steps,
                seed=seed,
                deterministic=deterministic,
                stub=stub,
                record_all=record_all,
                progress_fn=progress_fn,
            ),
            dict(
                policy=agent,
                num_trajectories=int(num_trajectories),
                min_room=int(min_room),
                seed=seed,
                stub=stub,
            ),
            dict(policy=agent, num_trajectories=int(num_trajectories), seed=seed),
        )
        for kwargs in attempts:
            try:
                trajectories = list(env_collect(**kwargs))  # type: ignore[misc]
                break
            except TypeError:
                continue
            except Exception:
                trajectories = []
                break
    if not trajectories:
        trajectories = _rollout_trajectories(
            agent,
            env=env,
            num_trajectories=int(num_trajectories),
            seed=seed,
            stub=stub,
            deterministic=deterministic,
            max_steps=max_steps,
            progress_fn=progress_fn,
        )

    dataset = BCDataset.from_trajectories(
        trajectories,
        min_room=int(min_room) if truncate else None,
        truncate=truncate,
        max_samples=max_samples,
        metadata={
            "stage": "M2_BC",
            "agent": "M1",
            "min_room": int(min_room),
            "truncate_to_far": bool(truncate),
            "num_trajectories": int(len(trajectories)),
            "deterministic": bool(deterministic),
        },
    )
    return dataset


def build_bc_dataset_from_m1(
    checkpoint: Optional[str] = None,
    cfg: Any = None,
    num_trajectories: Optional[int] = None,
    seed: Optional[int] = None,
    stub: Optional[bool] = None,
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[BCDataset, Any]:
    """Load M1 from a checkpoint, collect Room-7-onward trajectories, build ``B``."""
    bc_cfg = M2BCConfig.from_config(cfg, **{k: v for k, v in kwargs.items() if v is not None})
    if num_trajectories is not None:
        bc_cfg.num_trajectories = int(num_trajectories)
    if seed is not None:
        bc_cfg.seed = int(seed)
    if stub is not None:
        bc_cfg.stub = bool(stub)

    agent = None
    if _M1 is not None and checkpoint is not None:
        loader = getattr(_M1, "load_m1_agent", None)
        if callable(loader):
            agent = loader(checkpoint, stub=bc_cfg.stub, device=bc_cfg.device)
    dataset = build_bc_dataset(
        agent,
        env=None,
        num_trajectories=bc_cfg.num_trajectories,
        min_room=bc_cfg.min_room,
        truncate=bc_cfg.truncate_to_far,
        seed=bc_cfg.seed,
        stub=bc_cfg.stub,
        max_steps=bc_cfg.max_steps_per_episode,
        max_samples=max_samples if max_samples is not None else bc_cfg.max_bc_samples,
    )
    return dataset, agent


def _rollout_trajectories(
    agent: Any,
    env: Any = None,
    num_trajectories: int = NUM_BC_TRAJECTORIES,
    seed: int = 0,
    stub: bool = False,
    deterministic: bool = False,
    max_steps: Optional[int] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> List[List[Dict[str, Any]]]:
    """Local rollout loop used when ``env.collect_trajectories`` is unavailable."""
    maker = _env_attr("make_env", None)
    if env is None and callable(maker):
        env = maker(seed=seed, stub=stub)
    if env is None:
        raise RuntimeError(
            "Cannot collect Montezuma trajectories: neither an environment nor "
            "src.montezuma.env.make_env is available."
        )
    max_steps = int(max_steps or MAX_STEPS_PER_EPISODE)
    trajectories: List[List[Dict[str, Any]]] = []
    obs = _reset_env(env, seed)
    for episode in range(int(num_trajectories)):
        trajectory: List[Dict[str, Any]] = []
        steps = 0
        while steps < max_steps:
            action = _select_action(agent, obs, deterministic=deterministic)
            step_result = env.step(action)
            next_obs, reward, terminated, truncated, info = _split_step(step_result)
            trajectory.append(
                {
                    "obs": obs,
                    "action": int(action),
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "done": bool(terminated or truncated),
                    **(info if isinstance(info, dict) else {}),
                }
            )
            obs = next_obs
            steps += 1
            if terminated or truncated:
                obs = _reset_env(env, None)
                break
        trajectories.append(trajectory)
        if progress_fn is not None:
            try:
                progress_fn(episode + 1, int(num_trajectories))
            except Exception:
                pass
    return trajectories


def _split_step(step_result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(step_result, (tuple, list)):
        raise TypeError("environment step() must return a tuple")
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
    elif len(step_result) == 4:
        obs, reward, done, info = step_result
        terminated, truncated = bool(done), False
    else:  # pragma: no cover
        raise ValueError(f"Unexpected step() output of length {len(step_result)}")
    return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})


def _reset_env(env: Any, seed: Optional[int]) -> Any:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _select_action(agent: Any, obs: Any, deterministic: bool = False) -> int:
    selector = _env_attr("select_action", None)
    if callable(selector):
        try:
            return int(selector(agent, obs, deterministic=deterministic))
        except Exception:
            pass
    for name in ("act", "sample_action", "select_action"):
        fn = getattr(agent, name, None)
        if callable(fn):
            result = fn(obs, deterministic=deterministic)
            if isinstance(result, (tuple, list)) and result:
                return int(_scalar(result[0]))
            if isinstance(result, dict):
                for key in ("action", "actions"):
                    if key in result:
                        return int(_scalar(result[key]))
            return int(_scalar(result))
    if callable(agent):
        return int(_scalar(agent(obs)))
    raise TypeError("Unable to select an action with the supplied agent")


def _scalar(value: Any) -> Any:
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if _np is not None and hasattr(value, "reshape"):
        try:
            return int(_np.asarray(value).reshape(-1)[0])
        except Exception:  # pragma: no cover
            return int(value)
    return value


# ---------------------------------------------------------------------------
# M2 pretraining by behavioral cloning
# ---------------------------------------------------------------------------
@dataclass
class M2PretrainResult:
    updates: int = 0
    epochs: int = 0
    final_loss: float = float("nan")
    initial_loss: float = float("nan")
    dataset_size: int = 0
    num_trajectories: int = 0
    checkpoint: Optional[str] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    elapsed: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["history"] = list(self.history)
        return data


def pretrain_m2(
    cfg: Any = None,
    agent: Any = None,
    dataset: Optional[BCDataset] = None,
    teacher: Any = None,
    checkpoint: Optional[str] = None,
    dataset_path: Optional[str] = None,
    updates: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    output_dir: Optional[str] = None,
    logger: Any = None,
    verbose: bool = True,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Any, M2PretrainResult]:
    """Train M2 by behavioral cloning on trajectories collected with M1.

    ``M2`` is initialised from M1 (``teacher``); when no agent/dataset is given the
    function loads M1 from ``checkpoint`` and collects the Room-7-onward buffer.
    The returned agent becomes the pre-trained policy ``pi_*`` for fine-tuning.
    """
    _require_torch()
    bc_cfg = M2BCConfig.from_config(cfg)
    if updates is not None:
        bc_cfg.bc_updates = int(updates)
    if epochs is not None:
        bc_cfg.bc_epochs = int(epochs)
    if batch_size is not None:
        bc_cfg.bc_batch_size = int(batch_size)
    if lr is not None:
        bc_cfg.bc_lr = float(lr)

    start = time.time()
    result = M2PretrainResult()

    # --- resolve teacher (M1) -------------------------------------------------
    if teacher is None and agent is not None:
        teacher = agent
    if teacher is None and checkpoint is not None and _M1 is not None:
        loader = getattr(_M1, "load_m1_agent", None)
        if callable(loader):
            teacher = loader(checkpoint, stub=bc_cfg.stub, device=bc_cfg.device)

    # --- resolve dataset ------------------------------------------------------
    if dataset is None and dataset_path is not None and os.path.exists(dataset_path):
        dataset = BCDataset.load(dataset_path, device=bc_cfg.device)
    if dataset is None:
        dataset = build_bc_dataset(
            teacher,
            num_trajectories=bc_cfg.num_trajectories,
            min_room=bc_cfg.min_room,
            truncate=bc_cfg.truncate_to_far,
            seed=bc_cfg.seed,
            stub=bc_cfg.stub,
            max_steps=bc_cfg.max_steps_per_episode,
            max_samples=bc_cfg.max_bc_samples,
            progress_fn=progress_fn,
        )
    dataset.device = bc_cfg.device
    result.dataset_size = len(dataset)
    result.num_trajectories = int(dataset.metadata.get("num_trajectories", 0))

    # --- resolve student (M2 = copy of M1) ------------------------------------
    m2_agent = agent
    if m2_agent is None:
        m2_agent = _clone_agent(teacher, cfg=cfg, device=bc_cfg.device, stub=bc_cfg.stub)
    policy = getattr(m2_agent, "policy", m2_agent)
    if hasattr(policy, "train"):
        policy.train()

    if len(dataset) == 0:
        if verbose:
            _log(logger, "M2 BC: empty dataset, returning the initialised agent")
        result.final_loss = float("nan")
        result.elapsed = time.time() - start
        return m2_agent, result

    optimizer = torch.optim.Adam(policy.parameters(), lr=bc_cfg.bc_lr)
    generator = _np.random.default_rng(bc_cfg.seed) if _np is not None else random.Random(bc_cfg.seed)
    precomputed = dataset.teacher_logits is not None or dataset.teacher_probs is not None

    total_updates = max(1, int(bc_cfg.bc_updates))
    epoch = 0
    update = 0
    last_loss = float("nan")
    while update < total_updates:
        epoch += 1
        batches = list(
            dataset.minibatches(bc_cfg.bc_batch_size, generator=generator, shuffle=True)
        )
        if not batches:
            break
        for batch in batches:
            obs = batch["obs"].to(bc_cfg.device)
            logits = _policy_logits(policy, obs)
            teacher_logits = None
            if batch.get("teacher_logits") is not None:
                teacher_logits = batch["teacher_logits"].to(bc_cfg.device)
            if teacher_logits is None and not precomputed:
                teacher_logits = _teacher_logits(teacher, obs)
            if bc_cfg.bc_loss_type == "ce" and teacher_logits is None:
                loss = bc_cross_entropy_loss(logits, batch["actions"].to(bc_cfg.device))
            else:
                loss = bc_kl_loss(
                    logits,
                    teacher_logits=teacher_logits,
                    actions=batch.get("actions"),
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if bc_cfg.bc_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float(bc_cfg.bc_grad_clip)
                )
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            if result.initial_loss != result.initial_loss:  # NaN check
                result.initial_loss = last_loss
            update += 1
            if update % max(1, total_updates // 20) == 0 or update == total_updates:
                result.history.append({"update": update, "loss": last_loss})
                if verbose:
                    _log(logger, f"M2 BC update {update}/{total_updates} loss={last_loss:.4f}")
                if progress_fn is not None:
                    try:
                        progress_fn(update, total_updates)
                    except Exception:
                        pass
            if update >= total_updates:
                break

    result.updates = update
    result.epochs = epoch
    result.final_loss = last_loss
    if lr is None and cfg is None:
        pass

    # --- optional persistence -------------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "m2_pretrained.pt")
        try:
            m2_agent.save(path)
            result.checkpoint = path
        except Exception:
            state = {"policy": policy.state_dict(), "metadata": {"stage": "M2_BC"}}
            torch.save(state, path)
            result.checkpoint = path
        with open(os.path.join(output_dir, "m2_bc_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(result.as_dict(), fh, indent=2)

    result.elapsed = time.time() - start
    return m2_agent, result


def _clone_agent(teacher: Any, cfg: Any = None, device: str = "cpu", stub: bool = False) -> Any:
    """Build an M2 agent initialised from the M1 parameters (if available)."""
    _require_torch()
    if teacher is None:
        # No M1 available: build a fresh agent from the config.
        if _PPO is None:
            raise RuntimeError("src.montezuma.ppo_rnd is required to build an M2 agent")
        return _PPO.PPORNDAgent(config=_PPO.PPORNDConfig.from_config(cfg), device=device)
    builder = getattr(_PPO, "PPORNDAgent", None) if _PPO is not None else None
    if builder is None:
        return teacher
    try:
        config = getattr(teacher, "config", None)
        if config is None and _PPO is not None:
            config = _PPO.PPORNDConfig.from_config(cfg)
        agent = builder(config=config, device=device)
    except Exception:
        return teacher
    try:
        teacher_state = teacher.state_dict()
        agent.load_state_dict(teacher_state, load_optimizer=False)
    except Exception:
        try:
            agent.policy.load_state_dict(teacher.policy.state_dict())
        except Exception:
            pass
    return agent


# ---------------------------------------------------------------------------
# M2 fine-tuning on the whole game
# ---------------------------------------------------------------------------
@dataclass
class M2FinetuneResult:
    method: str = "bc"
    kl_weight: float = DEFAULT_BC_KL_WEIGHT
    steps: int = 0
    final_return: float = float("nan")
    best_return: float = float("nan")
    room7_success_rate: float = float("nan")
    max_room: float = float("nan")
    checkpoint: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def finetune_m2(
    cfg: Any = None,
    agent: Any = None,
    checkpoint: Optional[str] = None,
    dataset: Optional[BCDataset] = None,
    teacher: Any = None,
    method: Optional[str] = None,
    kl_weight: Optional[float] = None,
    total_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    logger: Any = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Tuple[Any, M2FinetuneResult]:
    """Fine-tune M2 on the whole game with PPO + RND (+ optional BC auxiliary loss).

    ``method`` is one of ``none`` (vanilla fine-tuning), ``bc`` (fine-tuning + BC
    distillation with weight ``kl_weight``) or ``ewc``.  The BC loss is applied to
    the actor only, as required by the paper.
    """
    _require_torch()
    bc_cfg = M2BCConfig.from_config(cfg)
    if method is not None:
        bc_cfg.method = str(method)
    if kl_weight is not None:
        bc_cfg.kl_weight = float(kl_weight)
    if total_steps is not None:
        bc_cfg.total_steps = int(total_steps)

    result = M2FinetuneResult(method=bc_cfg.method, kl_weight=float(bc_cfg.kl_weight))
    start = time.time()

    if _PPO is None:
        raise RuntimeError("src.montezuma.ppo_rnd is required to fine-tune M2")

    if agent is None:
        load_path = checkpoint or bc_cfg.__dict__.get("checkpoint")
        if load_path:
            loader = getattr(_M1, "load_m1_agent", None) if _M1 is not None else None
            if callable(loader):
                agent = loader(load_path, stub=bc_cfg.stub, device=bc_cfg.device)

    aux_loss_fn = None
    if bc_cfg.method == "bc":
        if teacher is None:
            teacher = agent
        aux_loss_fn = make_bc_aux_loss(
            teacher,
            dataset=dataset,
            kl_weight=bc_cfg.kl_weight,
            batch_size=bc_cfg.bc_batch_size,
            device=bc_cfg.device,
            seed=bc_cfg.seed,
        )
    elif bc_cfg.method == "ewc":
        ewc_cls = _RET.get("EWC")
        if ewc_cls is not None and agent is not None and dataset is not None and len(dataset) > 0:
            try:
                policy = getattr(agent, "policy", agent)
                ewc = ewc_cls(
                    policy,
                    fisher_diag=None,
                    coef=float(bc_cfg.ewc_coef),
                    name="ewc",
                )
                aux_loss_fn = lambda policy=None, obs=None, step=None, **_: ewc.penalty_loss(  # noqa: E731
                    actor=policy
                )
            except Exception:
                aux_loss_fn = None

    trainer_fn = getattr(_PPO, "train_ppo_rnd", None)
    if not callable(trainer_fn):
        raise RuntimeError("src.montezuma.ppo_rnd.train_ppo_rnd is unavailable")

    out = trainer_fn(
        config=_ppo_config(cfg, bc_cfg),
        total_steps=bc_cfg.total_steps,
        aux_loss_fn=aux_loss_fn,
        output_dir=output_dir,
        logger=logger,
        method=bc_cfg.method if bc_cfg.method in ("none", "bc", "ewc") else "none",
        init_checkpoint=checkpoint,
        agent=agent,
        stub=bc_cfg.stub,
        seed=bc_cfg.seed,
        eval_every=bc_cfg.eval_every,
        eval_episodes=bc_cfg.eval_episodes,
        save_every=bc_cfg.save_every,
        **{k: v for k, v in kwargs.items() if v is not None},
    )

    result.steps = int(out.get("steps", 0) or 0)
    result.final_return = float(out.get("episode_return", out.get("return_mean", float("nan"))))
    result.best_return = float(out.get("best_episode_return", result.final_return))
    result.room7_success_rate = float(out.get("room7_success_rate", float("nan")))
    result.max_room = float(out.get("max_room", float("nan")))
    result.history = list(out.get("history", []))
    result.checkpoint = out.get("checkpoint")
    result.elapsed = time.time() - start
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"m2_{bc_cfg.method}_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(result.as_dict(), fh, indent=2)
    return out.get("agent", agent), result


def _ppo_config(cfg: Any, bc_cfg: M2BCConfig) -> Any:
    """Build the PPO+RND config used for the fine-tuning run."""
    if _PPO is None:  # pragma: no cover
        return None
    try:
        ppo_cfg = _PPO.PPORNDConfig.from_config(cfg)
    except Exception:  # pragma: no cover
        ppo_cfg = _PPO.PPORNDConfig()
    try:
        ppo_cfg.method = bc_cfg.method
        ppo_cfg.kl_weight = float(bc_cfg.kl_weight)
        ppo_cfg.total_steps = int(bc_cfg.total_steps)
        ppo_cfg.eval_every = int(bc_cfg.eval_every)
        ppo_cfg.eval_episodes = int(bc_cfg.eval_episodes)
        ppo_cfg.save_every = int(bc_cfg.save_every)
        ppo_cfg.seed = int(bc_cfg.seed)
        ppo_cfg.device = bc_cfg.device
        ppo_cfg.stub = bool(bc_cfg.stub)
    except Exception:
        pass
    return ppo_cfg


def train_from_scratch(
    cfg: Any = None,
    total_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Tuple[Any, M2FinetuneResult]:
    """From-scratch baseline: M2 trained on the whole game *without* BC data."""
    _require_torch()
    bc_cfg = M2BCConfig.from_config(cfg)
    if total_steps is not None:
        bc_cfg.total_steps = int(total_steps)
    if _PPO is None:  # pragma: no cover
        raise RuntimeError("src.montezuma.ppo_rnd is required for from-scratch training")
    start = time.time()
    out = _PPO.train_ppo_rnd(
        config=_ppo_config(cfg, bc_cfg),
        total_steps=bc_cfg.total_steps,
        aux_loss_fn=None,
        output_dir=output_dir,
        logger=logger,
        method="none",
        init_checkpoint=None,
        stub=bc_cfg.stub,
        seed=bc_cfg.seed,
        **{k: v for k, v in kwargs.items() if v is not None},
    )
    result = M2FinetuneResult(
        method=BC_FREE_SCRATCH,
        kl_weight=0.0,
        steps=int(out.get("steps", 0) or 0),
        final_return=float(out.get("episode_return", out.get("return_mean", float("nan")))),
        best_return=float(out.get("best_episode_return", float("nan"))),
        room7_success_rate=float(out.get("room7_success_rate", float("nan"))),
        max_room=float(out.get("max_room", float("nan"))),
        checkpoint=out.get("checkpoint"),
        history=list(out.get("history", [])),
        elapsed=time.time() - start,
    )
    return out.get("agent"), result


def evaluate_room7_success_rate(
    agent: Any,
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    stub: bool = False,
    far_room: int = ROOM7,
) -> float:
    """Room-7 success rate of a (fine-tuned) policy, as plotted in Figure 6."""
    fn = _env_attr("room7_success_rate", None)
    policy = getattr(agent, "policy", agent)
    if callable(fn):
        for kwargs in (
            dict(policy=policy, num_episodes=num_episodes, seed=seed, stub=stub, far_room=far_room),
            dict(policy=policy, num_episodes=num_episodes, seed=seed, stub=stub),
            dict(policy, num_episodes=num_episodes, seed=seed),
        ):
            try:
                return float(fn(**kwargs))  # type: ignore[misc]
            except TypeError:
                continue
            except Exception:
                break
    evaluator = _env_attr("evaluate_policy", None)
    if callable(evaluator):
        out = evaluator(
            policy, num_episodes=num_episodes, seed=seed, stub=stub, far_room=far_room
        )
        if isinstance(out, dict):
            for key in ("room7_success_rate", "success_rate"):
                if key in out:
                    return float(out[key])
    return float("nan")


# ---------------------------------------------------------------------------
# KL-weight sweep (Figure 13)
# ---------------------------------------------------------------------------
def sweep_kl_weight(
    cfg: Any = None,
    weights: Sequence[float] = KL_WEIGHT_GRID,
    dataset: Optional[BCDataset] = None,
    checkpoint: Optional[str] = None,
    seeds: Sequence[int] = (0,),
    total_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    stub: Optional[bool] = None,
    include_baselines: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Figure 13: average return vs. BC KL weight during fine-tuning."""
    bc_cfg = M2BCConfig.from_config(cfg)
    if stub is not None:
        bc_cfg.stub = bool(stub)
    steps = int(total_steps if total_steps is not None else bc_cfg.total_steps)

    results: Dict[str, Any] = {"weights": [float(w) for w in weights], "runs": [], "steps": steps}
    for weight in weights:
        for seed in seeds:
            run_cfg = bc_cfg.with_overrides(kl_weight=float(weight), seed=int(seed))
            out_dir = (
                os.path.join(str(output_dir), f"bc_w{weight}_seed{seed}")
                if output_dir
                else None
            )
            _, res = finetune_m2(
                cfg=run_cfg,
                dataset=dataset,
                checkpoint=checkpoint,
                method="bc",
                kl_weight=float(weight),
                total_steps=steps,
                output_dir=out_dir,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
            record = res.as_dict()
            record["seed"] = int(seed)
            results["runs"].append(record)

    if include_baselines:
        for method in ("none", "scratch"):
            for seed in seeds:
                run_cfg = bc_cfg.with_overrides(method=method, seed=int(seed))
                out_dir = (
                    os.path.join(str(output_dir), f"{method}_seed{seed}")
                    if output_dir
                    else None
                )
                if method == "scratch":
                    _, res = train_from_scratch(
                        cfg=run_cfg, total_steps=steps, output_dir=out_dir
                    )
                else:
                    _, res = finetune_m2(
                        cfg=run_cfg,
                        dataset=None,
                        checkpoint=checkpoint,
                        method="none",
                        kl_weight=0.0,
                        total_steps=steps,
                        output_dir=out_dir,
                    )
                record = res.as_dict()
                record["seed"] = int(seed)
                results["runs"].append(record)

    results["mean_return"] = _mean_by_key(results["runs"], "kl_weight")
    if output_dir:
        os.makedirs(str(output_dir), exist_ok=True)
        with open(os.path.join(str(output_dir), "kl_weight_sweep.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return results


def _mean_by_key(runs: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for run in runs:
        label = (
            f"{run.get('method')}_w{run.get(key)}"
            if key in run
            else str(run.get("method"))
        )
        value = run.get("final_return", float("nan"))
        if value != value:
            continue
        buckets.setdefault(label, []).append(float(value))
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    cfg: Any = None,
    *,
    methods: Sequence[str] = ("none", "bc", "ewc", "scratch"),
    seeds: Sequence[int] = (0,),
    m1_checkpoint: Optional[str] = None,
    train_m1: bool = False,
    num_trajectories: Optional[int] = None,
    output_dir: Optional[str] = None,
    bc_updates: Optional[int] = None,
    total_steps: Optional[int] = None,
    stub: Optional[bool] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Full M1 -> dataset -> M2 -> fine-tuning pipeline.

    Steps:
      1. (optionally) train M1 with PPO + RND until ~7000 return;
      2. collect >500 Room-7-onward trajectories and build the BC buffer;
      3. train M2 (= ``pi_*``) by behavioral cloning;
      4. fine-tune M2 on the whole game for each requested method, logging the
         Room-7 success rate; the ``scratch`` baseline never sees BC data.
    """
    bc_cfg = M2BCConfig.from_config(cfg)
    if stub is not None:
        bc_cfg.stub = bool(stub)
    if num_trajectories is not None:
        bc_cfg.num_trajectories = int(num_trajectories)
    if total_steps is not None:
        bc_cfg.total_steps = int(total_steps)
    steps = int(bc_cfg.total_steps)
    root = str(output_dir) if output_dir else None

    summary: Dict[str, Any] = {
        "config": bc_cfg.to_dict(),
        "method_runs": [],
        "stages": {},
    }

    # --- 1. M1 ---------------------------------------------------------------
    m1_agent = None
    if train_m1 and _M1 is not None:
        m1_agent, m1_result = _M1.train_m1(
            cfg=cfg,
            stub=bc_cfg.stub,
            seed=bc_cfg.seed,
            output_dir=os.path.join(root, "M1") if root else None,
            verbose=verbose,
        )
        summary["stages"]["M1"] = m1_result.as_dict()
        m1_checkpoint = m1_result.checkpoint
    elif m1_checkpoint and _M1 is not None:
        loader = getattr(_M1, "load_m1_agent", None)
        if callable(loader):
            m1_agent = loader(m1_checkpoint, stub=bc_cfg.stub, device=bc_cfg.device)

    # --- 2. BC dataset -------------------------------------------------------
    dataset_path = os.path.join(root, "bc_dataset.npz") if root else None
    if dataset_path and os.path.exists(dataset_path):
        dataset = BCDataset.load(dataset_path, device=bc_cfg.device)
    else:
        dataset = build_bc_dataset(
            m1_agent,
            num_trajectories=bc_cfg.num_trajectories,
            min_room=bc_cfg.min_room,
            truncate=bc_cfg.truncate_to_far,
            seed=bc_cfg.seed,
            stub=bc_cfg.stub,
            max_steps=bc_cfg.max_steps_per_episode,
            max_samples=bc_cfg.bc_max_samples if hasattr(bc_cfg, "bc_max_samples") else bc_cfg.max_bc_samples,
        )
        if dataset_path:
            os.makedirs(root, exist_ok=True)
            dataset.save(dataset_path, include_observations=True)
    summary["stages"]["dataset"] = {
        "samples": len(dataset),
        "num_trajectories": dataset.metadata.get("num_trajectories"),
        "min_room": dataset.metadata.get("min_room"),
        "room_histogram": {str(k): v for k, v in dataset.room_histogram().items()},
    }

    # --- 3. M2 BC ------------------------------------------------------------
    m2_agent, m2_result = pretrain_m2(
        cfg=cfg,
        agent=m1_agent,
        dataset=dataset,
        updates=bc_updates,
        output_dir=os.path.join(root, "M2") if root else None,
        verbose=verbose,
    )
    summary["stages"]["M2"] = m2_result.as_dict()
    m2_checkpoint = m2_result.checkpoint

    # --- 4. fine-tuning variants --------------------------------------------
    for method in methods:
        for seed in seeds:
            if method == "scratch":
                _, res = train_from_scratch(
                    cfg=bc_cfg.with_overrides(seed=int(seed)),
                    total_steps=steps,
                    output_dir=os.path.join(root, f"scratch_seed{seed}") if root else None,
                    stub=bc_cfg.stub,
                    seed=int(seed),
                )
            else:
                _, res = finetune_m2(
                    cfg=bc_cfg.with_overrides(seed=int(seed)),
                    agent=None,
                    checkpoint=m2_checkpoint or m1_checkpoint,
                    dataset=dataset if method == "bc" else None,
                    method=method,
                    total_steps=steps,
                    output_dir=os.path.join(root, f"{method}_seed{seed}") if root else None,
                    stub=bc_cfg.stub,
                    seed=int(seed),
                    eval_episodes=eval_episodes,
                )
            record = res.as_dict()
            record["seed"] = int(seed)
            summary["method_runs"].append(record)

    summary["mean_return"] = _mean_by_key(summary["method_runs"], "seed")
    if root:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "montezuma_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Montezuma's Revenge M2 behavioral-cloning pretraining and fine-tuning "
            "(Section 3 / Appendix B.2)"
        )
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="config overrides key=value")
    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["pipeline", "collect", "pretrain", "finetune",
                                 "scratch", "sweep-kl"],
                        help="which stage of the M1/M2 pipeline to run")
    parser.add_argument("--methods", type=str, default="none,bc,ewc,scratch")
    parser.add_argument("--m1-checkpoint", type=str, default=None)
    parser.add_argument("--m2-checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="pre-built BC dataset path")
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--bc-updates", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--kl-weights", type=str, default=None,
                        help="comma separated KL weights for the Figure 13 sweep")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-m1", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg: Any = None
    if args.config:
        try:
            from src.common.config import load_config  # type: ignore

            cfg = load_config(args.config, args.overrides)
        except Exception:
            try:
                from ..common.config import load_config  # type: ignore

                cfg = load_config(args.config, args.overrides)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Unable to load config {args.config}: {exc}")

    bc_cfg = M2BCConfig.from_config(cfg)
    if args.device:
        bc_cfg.device = args.device
    if args.kl_weight is not None:
        bc_cfg.kl_weight = float(args.kl_weight)
    if args.total_steps is not None:
        bc_cfg.total_steps = int(args.total_steps)
    if args.stub or args.smoke_test:
        bc_cfg.stub = True
    if args.smoke_test:
        bc_cfg.total_steps = min(bc_cfg.total_steps, 2048)
        bc_cfg.bc_updates = min(bc_cfg.bc_updates or 10, 10)
        bc_cfg.num_trajectories = min(bc_cfg.num_trajectories, 4)
        bc_cfg.eval_episodes = min(bc_cfg.eval_episodes, 2)

    seeds = [int(args.seed)]
    if args.seeds:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    mode = args.mode
    dataset = BCDataset.load(args.dataset, device=bc_cfg.device) if args.dataset else None

    if mode == "collect":
        ds = build_bc_dataset(
            None,
            num_trajectories=bc_cfg.num_trajectories,
            min_room=bc_cfg.min_room,
            truncate=bc_cfg.truncate_to_far,
            seed=bc_cfg.seed,
            stub=bc_cfg.stub,
        )
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            ds.save(os.path.join(args.output_dir, "bc_dataset.npz"))
        print(json.dumps({"samples": len(ds), "metadata": ds.metadata}, indent=2))
        return 0

    if mode == "pretrain":
        agent = None
        if args.m1_checkpoint and _M1 is not None:
            loader = getattr(_M1, "load_m1_agent", None)
            if callable(loader):
                agent = loader(args.m1_checkpoint, stub=bc_cfg.stub, device=bc_cfg.device)
        _, res = pretrain_m2(
            cfg=bc_cfg,
            agent=agent,
            dataset=dataset,
            checkpoint=args.m1_checkpoint,
            updates=args.bc_updates,
            output_dir=args.output_dir,
        )
        print(json.dumps(res.as_dict(), indent=2))
        return 0

    if mode == "finetune":
        _, res = finetune_m2(
            cfg=bc_cfg,
            checkpoint=args.m2_checkpoint or args.m1_checkpoint,
            dataset=dataset,
            method="bc" if bc_cfg.method == "bc" else bc_cfg.method,
            kl_weight=bc_cfg.kl_weight,
            total_steps=bc_cfg.total_steps,
            output_dir=args.output_dir,
            stub=bc_cfg.stub,
            eval_episodes=args.eval_episodes,
        )
        print(json.dumps(res.as_dict(), indent=2))
        return 0

    if mode == "scratch":
        _, res = train_from_scratch(
            cfg=bc_cfg, total_steps=bc_cfg.total_steps, output_dir=args.output_dir
        )
        print(json.dumps(res.as_dict(), indent=2))
        return 0

    if mode == "sweep-kl":
        weights = (
            [float(w) for w in str(args.kl_weights).split(",")]
            if args.kl_weights
            else list(KL_WEIGHT_GRID)
        )
        out = sweep_kl_weight(
            cfg=bc_cfg,
            weights=weights,
            dataset=dataset,
            checkpoint=args.m2_checkpoint or args.m1_checkpoint,
            seeds=seeds,
            total_steps=bc_cfg.total_steps,
            output_dir=args.output_dir,
            stub=bc_cfg.stub,
        )
        print(json.dumps(out, indent=2))
        return 0

    summary = run_pipeline(
        cfg=bc_cfg,
        methods=[m.strip() for m in str(args.methods).split(",") if m.strip()],
        seeds=seeds,
        m1_checkpoint=args.m1_checkpoint,
        train_m1=args.train_m1,
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        bc_updates=args.bc_updates,
        total_steps=bc_cfg.total_steps,
        stub=bc_cfg.stub,
        eval_episodes=args.eval_episodes,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _log(logger: Any, message: str) -> None:
    if logger is not None and hasattr(logger, "info"):
        try:
            logger.info(message)
            return
        except Exception:
            pass
    print(message, flush=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
