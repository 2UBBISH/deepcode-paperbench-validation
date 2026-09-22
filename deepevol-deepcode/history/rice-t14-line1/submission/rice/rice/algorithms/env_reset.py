"""Simulator state save / restore for RICE (Go-Explore style reset).

The paper states (§C.1 Implementation Details, verbatim):

    "We implement the environment reset function similar to Ecoffet et al. (2019)
     to restore the environment to selected critical states."

This module implements that reset machinery: it captures the *complete internal
simulator state* (not merely the observation) of an environment, so that the
refining loop (Algorithm 2) can start an episode exactly from a previously
identified critical state ``s_t``.

Why the observation is not enough
---------------------------------
For Hopper-v3 / Walker2d-v3 / HalfCheetah-v3 the observation deliberately omits
the forward ``x`` position (and for Reacher-v2 the random target is stored in a
python attribute, ``env.goal``).  Restoring ``s_t`` from an observation alone is
therefore lossy, so we snapshot the underlying simulator objects:

* ``mujoco_py`` environments       -> ``env.unwrapped.sim.get_state()`` (MjSimState)
* ``mujoco>=2.3`` / gymnasium      -> ``env.unwrapped.data.{qpos,qvel,time}``
* classic-control style envs       -> deepcopy of ``env.unwrapped.state``
* custom envs                      -> their own ``get_state()`` / ``set_state()``

In addition, three families of bookkeeping are captured/restored because they
otherwise leak across episodes:

1. episode counters (``_elapsed_steps``, ``steps_beyond_done``, ``current_step``…),
2. task-specific randomness attributes (e.g. Reacher's ``goal`` / ``goallist``),
3. the environment RNG state.

Public API
----------
``Snapshot``            -- picklable container describing a simulator state.
``capture_state``       -- snapshot an environment (Go-Explore "save").
``set_state``           -- restore an environment from a snapshot ("restore").
``supports_state_restore`` -- feature detection.
``EnvStateManager``     -- convenience wrapper with a state *pool*.
``SnapshotPool`` / ``StateBuffer`` -- store / sample / persist snapshots.
``StateRestoreWrapper`` -- gym wrapper adding ``snapshot()``/``restore()``.
``make_state_manager``  -- factory.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import gym  # type: ignore
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import gymnasium  # type: ignore
except Exception:  # pragma: no cover
    gymnasium = None  # type: ignore


__all__ = [
    "Snapshot",
    "StateBuffer",
    "SnapshotPool",
    "EnvStateManager",
    "StateRestoreWrapper",
    "capture_state",
    "set_state",
    "supports_state_restore",
    "make_state_manager",
]


# --------------------------------------------------------------------------- #
# environment traversal helpers
# --------------------------------------------------------------------------- #
#: Episode-counter attributes that are reset by ``env.reset()`` and therefore
#: must be restored together with the simulator state.
_COUNTER_ATTRS: Tuple[str, ...] = (
    "_elapsed_steps",
    "steps_beyond_done",
    "current_step",
    "step_count",
    "episode_steps",
    "elapsed_steps",
    "_step_count",
    "total_steps",
)

#: Task-specific randomness attributes used by some environments (Reacher's
#: target, BipedalWalker's terrain ...) that are *not* part of the simulator
#: state but change the dynamics/reward.
_TASK_ATTRS: Tuple[str, ...] = (
    "goal",
    "_target",
    "target",
    "goallist",
    "curr_path",
    "prev_shaping",
    "has_reset",
)


def iter_env_chain(env: Any) -> List[Any]:
    """Return ``[env, env.env, ..., core]`` (wrapper chain, outermost first)."""
    chain: List[Any] = []
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        chain.append(cur)
        seen.add(id(cur))
        nxt = getattr(cur, "env", None)
        if nxt is None:
            break
        cur = nxt
    return chain


def unwrap_env(env: Any) -> Any:
    """Unwrap nested gym wrappers (and the first sub-env of vector envs)."""
    chain = iter_env_chain(env)
    core = chain[-1] if chain else env
    # vectorised environments: behave like the first sub-environment
    inner = getattr(core, "envs", None)
    if isinstance(inner, (list, tuple)) and len(inner) > 0:
        return unwrap_env(inner[0])
    return core


def _np_random_state(core: Any) -> Any:
    rng = getattr(core, "np_random", None)
    if rng is None:
        return None
    bitgen = getattr(rng, "bit_generator", None)
    if bitgen is None:
        return None
    try:
        return copy.deepcopy(bitgen.state)
    except Exception:  # pragma: no cover - exotic RNG
        return None


def _restore_np_random(core: Any, state: Any) -> None:
    if state is None:
        return
    rng = getattr(core, "np_random", None)
    if rng is None or getattr(rng, "bit_generator", None) is None:
        return
    try:
        rng.bit_generator.state = state
    except Exception:  # pragma: no cover
        pass


def current_observation(env: Any) -> Optional[np.ndarray]:
    """Best-effort read of the current observation without stepping."""
    core = unwrap_env(env)
    for candidate in (core, env):
        get_obs = getattr(candidate, "_get_obs", None)
        if callable(get_obs):
            try:
                return np.asarray(get_obs(), dtype=np.float32)
            except Exception:
                pass
    state = getattr(core, "state", None)
    if isinstance(state, (np.ndarray, list, tuple)):
        try:
            return np.asarray(state, dtype=np.float32).ravel()
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Snapshot container
# --------------------------------------------------------------------------- #
@dataclass
class Snapshot:
    """A complete, picklable simulator state.

    Only one of the ``kind``-specific payload fields is populated, chosen by
    :func:`capture_state`.

    Attributes
    ----------
    kind : str
        ``"sim_state"`` (mujoco_py MjSimState), ``"mujoco_data"`` (mujoco>=2.3
        ``data.{qpos,qvel,time}``), ``"state_attr"`` (deepcopy of
        ``env.unwrapped.state``), ``"custom"`` (env provided state objects) or
        ``"observation"`` (degraded fallback: observation only).
    observation : np.ndarray or None
        The observation ``s_t`` at capture time.
    counters, task_attrs : dict
        Episode counters / task-specific randomness attributes, keyed by
        ``"{chain_index}:{attribute}"``.
    """

    kind: str = "observation"
    observation: Optional[np.ndarray] = None
    sim_state: Any = None
    qpos: Optional[np.ndarray] = None
    qvel: Optional[np.ndarray] = None
    time: Optional[float] = None
    env_state: Any = None
    custom_state: Any = None
    counters: Dict[str, Any] = field(default_factory=dict)
    task_attrs: Dict[str, Any] = field(default_factory=dict)
    rng_state: Any = None
    episode_step: int = 0
    episode_return: float = 0.0

    # -- serialisation ----------------------------------------------------- #
    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "observation": None if self.observation is None else np.asarray(self.observation),
            "episode_step": self.episode_step,
            "episode_return": self.episode_return,
        }

    def clone(self) -> "Snapshot":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- #
# capture / restore
# --------------------------------------------------------------------------- #
def _capture_layers(env: Any, attrs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx, layer in enumerate(iter_env_chain(env)):
        for name in attrs:
            if not hasattr(layer, name):
                continue
            try:
                out[f"{idx}:{name}"] = copy.deepcopy(getattr(layer, name))
            except Exception:
                pass
    return out


def _restore_layers(env: Any, values: Dict[str, Any]) -> None:
    chain = iter_env_chain(env)
    for key, value in values.items():
        try:
            idx_str, name = key.split(":", 1)
            idx = int(idx_str)
        except ValueError:
            continue
        if idx >= len(chain):
            continue
        try:
            setattr(chain[idx], name, copy.deepcopy(value))
        except Exception:
            pass


def capture_state(
    env: Any,
    observation: Optional[np.ndarray] = None,
    episode_step: int = 0,
    episode_return: float = 0.0,
) -> Snapshot:
    """Snapshot ``env`` (Go-Explore style state save, Ecoffet et al. 2019)."""
    core = unwrap_env(env)
    snap = Snapshot(
        observation=(
            np.asarray(observation, dtype=np.float32)
            if observation is not None
            else current_observation(env)
        ),
        counters=_capture_layers(env, _COUNTER_ATTRS),
        task_attrs=_capture_layers(env, _TASK_ATTRS),
        rng_state=_np_random_state(core),
        episode_step=int(episode_step),
        episode_return=float(episode_return),
    )

    # 1) custom env API
    get_state = getattr(core, "get_state", None)
    if callable(get_state) and not isinstance(core, EnvStateManager):
        try:
            snap.custom_state = copy.deepcopy(get_state())
            snap.kind = "custom"
            return snap
        except Exception:
            snap.custom_state = None

    # 2) mujoco_py
    sim = getattr(core, "sim", None)
    if sim is not None and hasattr(sim, "get_state"):
        try:
            sim_state = sim.get_state()
            snap.sim_state = copy.deepcopy(sim_state)
            snap.qpos = np.asarray(getattr(sim_state, "qpos", np.zeros(0)), dtype=np.float64).copy()
            snap.qvel = np.asarray(getattr(sim_state, "qvel", np.zeros(0)), dtype=np.float64).copy()
            snap.time = float(getattr(sim_state, "time", 0.0))
            snap.kind = "sim_state"
            return snap
        except Exception:
            pass

    # 3) mujoco >= 2.3 / gymnasium
    data = getattr(core, "data", None)
    if data is not None and hasattr(data, "qpos"):
        try:
            snap.qpos = np.array(data.qpos, dtype=np.float64, copy=True)
            snap.qvel = np.array(getattr(data, "qvel", np.zeros(0)), dtype=np.float64, copy=True)
            snap.time = float(getattr(data, "time", 0.0))
            snap.kind = "mujoco_data"
            return snap
        except Exception:
            pass

    # 4) envs exposing a mutable ``state`` attribute (classic control)
    if hasattr(core, "state"):
        try:
            snap.env_state = copy.deepcopy(getattr(core, "state"))
            snap.kind = "state_attr"
            return snap
        except Exception:
            pass

    snap.kind = "observation"
    return snap


def set_state(env: Any, snapshot: Snapshot) -> Optional[np.ndarray]:
    """Restore ``env`` to ``snapshot`` and return the resulting observation.

    Mirrors Ecoffet et al. (2019): the simulator is put back in the exact state
    it had, including episode counters and task randomness, after which the
    environment can be stepped normally.
    """
    if snapshot is None:
        obs, _ = _call_reset(env)
        return obs

    core = unwrap_env(env)
    _restore_layers(env, snapshot.counters)
    _restore_layers(env, snapshot.task_attrs)
    _restore_np_random(core, snapshot.rng_state)

    kind = snapshot.kind
    if kind == "custom":
        setter = getattr(core, "set_state", None)
        if callable(setter):
            try:
                setter(copy.deepcopy(snapshot.custom_state))
            except Exception:
                pass
    elif kind == "sim_state" and snapshot.sim_state is not None:
        restored = False
        sim = getattr(core, "sim", None)
        if sim is not None and hasattr(sim, "set_state"):
            try:
                sim.set_state(snapshot.sim_state)
                if hasattr(sim, "forward"):
                    sim.forward()
                restored = True
            except Exception:
                restored = False
        if not restored and hasattr(core, "set_state"):
            try:
                core.set_state(snapshot.qpos, snapshot.qvel)
                restored = True
            except Exception:
                pass
    elif kind == "mujoco_data":
        try:
            core.data.qpos[:] = snapshot.qpos
            if snapshot.qvel is not None and hasattr(core.data, "qvel"):
                core.data.qvel[:] = snapshot.qvel
            if snapshot.time is not None and hasattr(core.data, "time"):
                core.data.time = snapshot.time
            try:  # keep derived quantities (xpos / com) consistent
                import mujoco  # type: ignore

                mujoco.mj_forward(core.model, core.data)
            except Exception:
                pass
        except Exception:
            pass
    elif kind == "state_attr":
        try:
            setattr(core, "state", copy.deepcopy(snapshot.env_state))
        except Exception:
            pass

    obs = current_observation(env)
    if obs is None:
        obs = snapshot.observation
    return obs


def _call_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Any]:
    """``env.reset`` supporting both gym (1-tuple) and gymnasium (2-tuple)."""
    if seed is not None:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
            out = env.reset()
    else:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def supports_state_restore(env: Any) -> bool:
    """True when :func:`capture_state` finds a real (non-degraded) state."""
    snap = capture_state(env)
    return snap.kind != "observation"


# --------------------------------------------------------------------------- #
# snapshot pools
# --------------------------------------------------------------------------- #
class StateBuffer:
    """A simple append-only list of :class:`Snapshot` objects."""

    def __init__(self, snapshots: Optional[Iterable[Snapshot]] = None):
        self.snapshots: List[Snapshot] = list(snapshots) if snapshots else []

    def add(self, snapshot: Snapshot) -> Snapshot:
        self.snapshots.append(snapshot)
        return snapshot

    append = add

    def extend(self, snapshots: Iterable[Snapshot]) -> None:
        self.snapshots.extend(snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index):
        return self.snapshots[index]

    def __iter__(self):
        return iter(self.snapshots)

    def clear(self) -> None:
        self.snapshots.clear()

    def sample(
        self, rng: Optional[Any] = None, index: Optional[int] = None
    ) -> Optional[Snapshot]:
        if not self.snapshots:
            return None
        if index is not None:
            return self.snapshots[int(index) % len(self.snapshots)]
        if rng is None:
            idx = random.randrange(len(self.snapshots))
        elif hasattr(rng, "integers"):
            idx = int(rng.integers(0, len(self.snapshots)))
        elif hasattr(rng, "choice"):
            idx = int(rng.choice(len(self.snapshots)))
        else:  # pragma: no cover
            idx = random.randrange(len(self.snapshots))
        return self.snapshots[idx]

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.snapshots, handle)
        return path

    def load(self, path: str) -> "StateBuffer":
        with open(path, "rb") as handle:
            self.snapshots = list(pickle.load(handle))
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"snapshots": list(self.snapshots)}

    def load_state_dict(self, state: Dict[str, Any]) -> "StateBuffer":
        self.snapshots = list(state.get("snapshots", []))
        return self


#: The paper (and Ecoffet et al. 2019) refer to a collection of resettable
#: states as a "cell archive"; we expose the same object under two names.
SnapshotPool = StateBuffer


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #
class EnvStateManager:
    """Save / restore manager for a single (or vectorised) environment.

    Duck-typed interface expected by ``rice.algorithms.mixed_init`` and
    ``rice.algorithms.critical_state``:
    ``snapshot`` / ``save_state`` / ``save`` / ``get_state`` / ``state_dict``
    for saving and
    ``restore`` / ``load_state`` / ``load`` / ``set_state`` / ``restore_state``
    for restoring, plus ``current_observation``.

    Parameters
    ----------
    env :
        Gym / gymnasium environment (wrappers allowed).  ``env.envs[0]`` is used
        automatically for vectorised environments.
    rng :
        Optional RNG used by :meth:`sample_state`.
    max_snapshots :
        Cap on the internal pool (oldest entries dropped first) so long refines
        do not leak memory.
    """

    def __init__(self, env: Any, rng: Optional[Any] = None, max_snapshots: int = 100000):
        self.env = env
        self.rng = rng
        self.max_snapshots = int(max_snapshots)
        self.pool = StateBuffer()
        self.episode_step = 0
        self.episode_return = 0.0
        self.restore_count = 0
        self.supported = supports_state_restore(env)

    # -- basic env interaction ------------------------------------------- #
    def reset(self, seed: Optional[int] = None) -> Any:
        obs, _ = _call_reset(self.env, seed=seed)
        self.episode_step = 0
        self.episode_return = 0.0
        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = _call_step_legacy(self.env, action)
        self.episode_step += 1
        self.episode_return += float(reward)
        return obs, float(reward), bool(done), bool(info.pop("_truncated", False)), info

    # -- state API -------------------------------------------------------- #
    def snapshot(self) -> Snapshot:
        """Capture the current simulator state (picklable)."""
        return capture_state(
            self.env,
            observation=current_observation(self.env),
            episode_step=self.episode_step,
            episode_return=self.episode_return,
        )

    # aliases used across the code base
    save_state = snapshot
    get_state = snapshot
    save = snapshot

    def restore(self, snapshot: Optional[Snapshot]) -> Optional[np.ndarray]:
        """Restore a snapshot; returns the observation at that state."""
        self.restore_count += 1
        if snapshot is None:
            return self.reset()
        obs = set_state(self.env, snapshot)
        self.episode_step = int(getattr(snapshot, "episode_step", 0))
        self.episode_return = float(getattr(snapshot, "episode_return", 0.0))
        return obs

    load_state = restore
    restore_state = restore
    load = restore

    def set_state(self, snapshot_or_env_state: Any, *args: Any) -> Any:
        """Same as :meth:`restore`, but also accepts ``(qpos, qvel)``."""
        if isinstance(snapshot_or_env_state, Snapshot):
            return self.restore(snapshot_or_env_state)
        marker = Snapshot(kind="mujoco_data", qpos=np.asarray(snapshot_or_env_state))
        if args:
            marker.qvel = np.asarray(args[0])
        return self.restore(marker)

    def current_observation(self) -> Optional[np.ndarray]:
        return current_observation(self.env)

    # -- pool ------------------------------------------------------------- #
    def add_snapshot(self, snapshot: Optional[Snapshot] = None) -> Snapshot:
        snap = snapshot if snapshot is not None else self.snapshot()
        self.pool.add(snap)
        if len(self.pool) > self.max_snapshots:
            del self.pool.snapshots[0]
        return snap

    def sample_state(self, index: Optional[int] = None) -> Optional[Snapshot]:
        return self.pool.sample(rng=self.rng, index=index)

    def restore_sample(self, index: Optional[int] = None) -> Optional[np.ndarray]:
        return self.restore(self.sample_state(index=index))

    def __len__(self) -> int:
        return len(self.pool)

    # -- persistence ------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "pool": self.pool.state_dict(),
            "episode_step": self.episode_step,
            "episode_return": self.episode_return,
            "restore_count": self.restore_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "EnvStateManager":
        self.pool.load_state_dict(state.get("pool", {}))
        self.episode_step = int(state.get("episode_step", 0))
        self.episode_return = float(state.get("episode_return", 0.0))
        self.restore_count = int(state.get("restore_count", 0))
        return self

    def save_pool(self, path: str) -> str:
        return self.pool.save(path)

    def load_pool(self, path: str) -> "EnvStateManager":
        self.pool.load(path)
        return self


def _call_step_legacy(env: Any, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Normalise gym (4-tuple) / gymnasium (5-tuple) ``step`` returns."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        info = dict(info or {})
        info["_truncated"] = bool(truncated)
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out  # type: ignore[misc]
    return obs, float(reward), bool(done), dict(info or {})


# --------------------------------------------------------------------------- #
# gym wrapper + factory
# --------------------------------------------------------------------------- #
if gym is not None:  # pragma: no cover - depends on optional dependency

    class StateRestoreWrapper(gym.Wrapper):  # type: ignore[misc]
        """Adds Go-Explore style ``snapshot``/``restore`` to any gym env."""

        def snapshot(self) -> Snapshot:
            return capture_state(self.env, observation=current_observation(self.env))

        save_state = snapshot
        get_state = snapshot

        def restore(self, snapshot: Optional[Snapshot]):
            obs = set_state(self.env, snapshot)
            return obs

        restore_state = restore
        set_state = restore
        load_state = restore

        def current_observation(self) -> Optional[np.ndarray]:
            return current_observation(self.env)

else:  # pragma: no cover

    class StateRestoreWrapper:  # type: ignore[no-redef]
        """Fallback when gym is unavailable (gymnasium-only installs)."""

        def __init__(self, env: Any):
            self.env = env

        def __getattr__(self, item: str) -> Any:
            return getattr(self.env, item)

        def snapshot(self) -> Snapshot:
            return capture_state(self.env, observation=current_observation(self.env))

        save_state = snapshot
        get_state = snapshot

        def restore(self, snapshot: Optional[Snapshot]):
            return set_state(self.env, snapshot)

        restore_state = restore
        set_state = restore
        load_state = restore

        def current_observation(self) -> Optional[np.ndarray]:
            return current_observation(self.env)


def make_state_manager(env: Any, **kwargs: Any) -> EnvStateManager:
    """Factory: build an :class:`EnvStateManager` for ``env``."""
    return EnvStateManager(env, **kwargs)
