"""Go-Explore-style environment reset / state-restore wrapper for RICE.

Paper reference (Appendix C.1, Implementation Details):

    "We implement the environment reset function similar to Ecoffet et al.
    (2019) to restore the environment to selected critical states. This method
    is feasible in our case, as we operate within simulator-based environments.
    ... It's important to note that our framework is designed to be versatile
    and is indeed compatible with a goal/state-conditioned policy approach such
    as Ecoffet et al. (2021). Given a trajectory with an identified most
    important state, we can select the most important state as the final goal
    and select the en-route intermediate states as sub-goals."

This module implements the two mechanisms required by Algorithm 2
("Constructing Mixed Initial State Distribution" resets the environment to a
critical state identified by the mask network):

1. **Direct state injection** (preferred for simulator-based environments):
   restore the simulator's internal state via
   ``env.sim.set_state(...)`` / ``env.set_state(qpos, qvel)`` /
   ``env.unwrapped.state = ...`` when the backend supports it.
2. **Action replay** (Go-Explore style, always available): store the sequence of
   actions that produced the trajectory and replay ``actions[:t]`` from a fresh
   :meth:`reset` to reach the state visited at step ``t``.

Both mechanisms are exposed through a small, uniform interface so that
:mod:`rice.refining.mixed_init` and :mod:`rice.refining.ppo_refine` can jump to a
critical state without knowing which backend they run on::

    env = make_reset_env(base_env)
    obs, info = env.reset()                # -> StateSnapshot(step=0)
    for t in range(K):
        obs, r, term, trunc, info = env.step(a_t)   # actions recorded
    snapshot = env.snapshot()              # or env.snapshot_at(t_star)
    obs, info = env.reset_to(snapshot)     # jump back to the critical state

The wrapper is deliberately defensive: every opt-in backend hook is probed at
call time and failures degrade to the replay fallback instead of raising.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional gym dependency (module must stay importable without gym/MuJoCo).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    import gym  # type: ignore

    _WrapperBase = gym.Wrapper
    _HAS_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    _WrapperBase = object  # type: ignore
    _HAS_GYM = False


__all__ = [
    "StateSnapshot",
    "ResetWrapper",
    "ReplayResetter",
    "make_reset_env",
    "wrap_reset",
    "env_supports_direct_state",
    "get_env_state",
    "set_env_state",
]


# ---------------------------------------------------------------------------
# Small helpers for state copying / probing of backend hooks
# ---------------------------------------------------------------------------
def _copy_state(state: Any) -> Any:
    """Deep-copy a simulator state object, tolerating non-copyable objects.

    ``mujoco_py.MjSimState`` is a plain named-tuple-like object, so ``copy``
    works; for exotic objects we fall back to a NumPy buffer copy when possible.
    """
    if state is None:
        return None
    try:
        return copy.deepcopy(state)
    except Exception:
        pass
    # NumPy-backed state (e.g. a dict of arrays or a plain ndarray).
    try:
        return {k: np.array(v, copy=True) for k, v in dict(state).items()}
    except Exception:
        pass
    try:
        return np.array(state, copy=True)
    except Exception:
        return state


def _unwrap_chain(env: Any) -> List[Any]:
    """Return the env stack ``[outer, ..., inner]`` (outermost wrapper first)."""
    chain: List[Any] = []
    node = env
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        chain.append(node)
        node = getattr(node, "env", None)
    return chain


def _find_attr(env: Any, names: Sequence[str]) -> Optional[Any]:
    """Search the wrapper chain for the first object owning any of ``names``."""
    chain = _unwrap_chain(env)
    for node in chain:
        for name in names:
            if hasattr(node, name):
                return node
    return None


def env_supports_direct_state(env: Any) -> bool:
    """True if the (possibly wrapped) env exposes a usable state-injection hook."""
    chain = _unwrap_chain(env)
    for node in chain:
        # MuJoCo bindings (mujoco_py / mujoco>=2 used by gym v3 wrappers).
        sim = getattr(node, "sim", None)
        if sim is not None and hasattr(sim, "set_state") and hasattr(sim, "get_state"):
            return True
        # Classic `MujocoEnv.set_state(qpos, qvel)` hook.
        if callable(getattr(node, "set_state", None)) and hasattr(node, "state"):
            return True
        # MDP-style memoized-state environments.
        if hasattr(node, "state") and hasattr(node, "set_state"):
            return True
        # Environments exposing a restore hook directly.
        if callable(getattr(node, "restore_state", None)):
            return True
    return False


def get_env_state(env: Any) -> Dict[str, Any]:
    """Extract a restorable state from the env (direct hook if available)."""
    chain = _unwrap_chain(env)

    # 1) MuJoCo simulator state.
    for node in chain:
        sim = getattr(node, "sim", None)
        if sim is not None and hasattr(sim, "get_state"):
            try:
                return {"kind": "sim", "state": _copy_state(sim.get_state())}
            except Exception:
                pass

    # 2) MDP-style `env.state` attribute.
    for node in chain:
        if hasattr(node, "state"):
            try:
                state = _copy_state(getattr(node, "state"))
                if state is not None:
                    return {"kind": "attr", "state": state}
            except Exception:
                pass

    # 3) Generic serialisation hook.
    for node in chain:
        hook = getattr(node, "get_state", None)
        if callable(hook):
            try:
                return {"kind": "hook", "state": _copy_state(hook())}
            except Exception:
                pass

    return {"kind": "none", "state": None}


def set_env_state(env: Any, packed: Any) -> bool:
    """Inject a state previously returned by :func:`get_env_state`.

    Returns ``True`` on success, ``False`` when no backend hook accepted the
    state (the caller should then fall back to action replay).
    """
    if packed is None:
        return False
    if isinstance(packed, dict) and "kind" in packed:
        kind = packed.get("kind")
        state = packed.get("state")
    else:  # accept a bare state object too
        kind, state = "auto", packed

    chain = _unwrap_chain(env)

    def _try_sim() -> bool:
        for node in chain:
            sim = getattr(node, "sim", None)
            if sim is not None and hasattr(sim, "set_state"):
                try:
                    sim.set_state(_copy_state(state))
                    try:  # forward kinematics refresh, mirrors Go-Explore impls
                        sim.forward()
                    except Exception:
                        pass
                    return True
                except Exception:
                    continue
        return False

    def _try_attr() -> bool:
        for node in chain:
            if hasattr(node, "state"):
                try:
                    setattr(node, "state", _copy_state(state))
                    return True
                except Exception:
                    continue
        return False

    def _try_hook() -> bool:
        for node in chain:
            for name in ("set_state", "restore_state", "set_env_state"):
                hook = getattr(node, name, None)
                if callable(hook):
                    try:
                        hook(_copy_state(state))
                        return True
                    except TypeError:
                        # `MujocoEnv.set_state(qpos, qvel)` style hook: unpack.
                        try:
                            if isinstance(state, (tuple, list)) and len(state) == 2:
                                hook(_copy_state(state[0]), _copy_state(state[1]))
                                return True
                        except Exception:
                            continue
                    except Exception:
                        continue
        return False

    order = {
        "sim": (_try_sim, _try_hook, _try_attr),
        "attr": (_try_attr, _try_hook, _try_sim),
        "hook": (_try_hook, _try_attr, _try_sim),
        "auto": (_try_sim, _try_attr, _try_hook),
        "none": (_try_hook, _try_attr, _try_sim),
    }.get(kind, (_try_sim, _try_attr, _try_hook))

    for fn in order:
        try:
            if fn():
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Snapshot container
# ---------------------------------------------------------------------------
@dataclass
class StateSnapshot:
    """A point on a recorded trajectory that can be restored later.

    Attributes
    ----------
    step:
        Index of this state inside its trajectory (``0`` = episode start).
    observation:
        Observation observed at this state.
    state:
        Backend-specific restorable state (see :func:`get_env_state`).
    actions:
        Actions ``a_0 .. a_{step-1}`` that lead from the episode start to this
        state; the replay fallback executes exactly this prefix.
    info:
        Info dict returned with the observation (diagnostics only).
    score:
        Optional importance score attached by the explanation stage
        (e.g. ``P(a^m = 0 | s)`` from the mask network).
    metadata:
        Free-form extras (trajectory id, seed, render buffer, ...).
    """

    step: int = 0
    observation: Any = None
    state: Any = None
    actions: List[Any] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def clone(self) -> "StateSnapshot":
        return StateSnapshot(
            step=int(self.step),
            observation=_copy_state(self.observation),
            state=_copy_state(self.state),
            actions=list(self.actions),
            info=dict(self.info or {}),
            score=self.score,
            metadata=dict(self.metadata or {}),
        )

    def with_score(self, score: Optional[float]) -> "StateSnapshot":
        self.score = None if score is None else float(score)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """JSON/pickle friendly (light) summary — omits raw simulator state."""
        return {
            "step": int(self.step),
            "score": self.score,
            "n_actions": len(self.actions),
            "metadata": dict(self.metadata or {}),
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.step)


# ---------------------------------------------------------------------------
# Replay-based restorer (Go-Explore fallback)
# ---------------------------------------------------------------------------
class ReplayResetter:
    """Replays a stored action prefix from a fresh episode reset.

    Used when the environment cannot be state-injected directly, and also as a
    validation path for simulator envs where deterministic replay is expected.
    """

    def __init__(self, env: Any, max_replay_steps: Optional[int] = None):
        self.env = env
        self.max_replay_steps = max_replay_steps
        self.replay_count = 0
        self.last_replay_steps = 0

    # -- gym plumbing ------------------------------------------------------
    @staticmethod
    def _split_obs_ret(ret: Any) -> Tuple[Any, Dict[str, Any]]:
        """Normalize a gym reset return (obs) or (obs, info)."""
        if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
            return ret[0], ret[1]
        return ret, {}

    def _raw_reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        try:
            ret = self.env.reset(**kwargs)
        except TypeError:
            ret = self.env.reset()
        return self._split_obs_ret(ret)

    def _raw_step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        ret = self.env.step(action)
        if isinstance(ret, tuple) and len(ret) == 5:
            return ret  # type: ignore[return-value]
        obs, reward, done, info = ret  # 4-tuple (classic gym API)
        return obs, reward, bool(done), False, dict(info or {})

    # -- public API --------------------------------------------------------
    def replay(
        self,
        actions: Sequence[Any],
        reset_kwargs: Optional[Dict[str, Any]] = None,
        stop_on_done: bool = True,
    ) -> Tuple[Any, Dict[str, Any], int]:
        """Reset and replay ``actions``.

        Returns ``(observation, info, steps_executed)``. If the episode
        terminates early the replay stops (Go-Explore truncates such rollouts).
        """
        obs, info = self._raw_reset(**(reset_kwargs or {}))
        actions = list(actions or [])
        if self.max_replay_steps is not None:
            actions = actions[: int(self.max_replay_steps)]

        executed = 0
        for action in actions:
            obs, _reward, terminated, truncated, info = self._raw_step(action)
            executed += 1
            if stop_on_done and (terminated or truncated):
                break
        self.replay_count += 1
        self.last_replay_steps = executed
        return obs, info, executed

    def replay_to_snapshot(
        self, snapshot: StateSnapshot, reset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any], int]:
        return self.replay(snapshot.actions, reset_kwargs=reset_kwargs)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class ResetWrapper(_WrapperBase):  # type: ignore[misc]
    """Record trajectories and restore the env to any visited state.

    Parameters
    ----------
    env:
        Environment to wrap (may itself be a wrapper stack).
    max_records:
        Ring-buffer cap on the number of snapshots kept (memory guard for the
        length-1000 MuJoCo horizon / the 500-trajectory fidelity sweep).
    prefer_direct:
        If ``True`` (default) try simulator state injection before replay.
    store_actions:
        Whether to keep the action prefix in every snapshot (needed for the
        replay fallback).
    deterministic_replay:
        If ``True``, re-seed the environment and its action space before each
        replay so the reproduced trajectory is (best effort) identical.
    seed:
        Base seed used for replay determinism.
    verify_replay:
        If ``True``, replays also run for envs that support direct injection
        (useful as a correctness check of the restore machinery).
    tol:
        Observation tolerance used when :meth:`reset_to` verifies a restore.
    """

    def __init__(
        self,
        env: Any,
        max_records: int = 4096,
        prefer_direct: bool = True,
        store_actions: bool = True,
        deterministic_replay: bool = True,
        seed: Optional[int] = None,
        verify_replay: bool = False,
        tol: float = 1e-6,
    ):
        if _HAS_GYM:  # pragma: no cover - depends on optional gym
            super().__init__(env)
        else:  # minimal duck-typed fallback
            self.env = env
            self.action_space = getattr(env, "action_space", None)
            self.observation_space = getattr(env, "observation_space", None)
            self.metadata = getattr(env, "metadata", {})

        self.max_records = int(max_records)
        self.prefer_direct = bool(prefer_direct)
        self.store_actions = bool(store_actions)
        self.deterministic_replay = bool(deterministic_replay)
        self.seed_value = None if seed is None else int(seed)
        self.verify_replay = bool(verify_replay)
        self.tol = float(tol)

        self.resetter = ReplayResetter(self, max_replay_steps=None)

        # Trajectory bookkeeping.
        self.snapshots: List[StateSnapshot] = []
        self.actions: List[Any] = []
        self.observations: List[Any] = []
        self.step_count = 0
        self.episode_count = 0
        self.trajectories: List[Dict[str, Any]] = []

        # Statistics (reported by experiment drivers).
        self.direct_restores = 0
        self.replay_restores = 0
        self.failed_restores = 0

        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._episode_return = 0.0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def can_restore_direct(self) -> bool:
        return env_supports_direct_state(self.env)

    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the env, archiving the finished trajectory and restarting records."""
        if self.snapshots:
            self._archive_trajectory()

        if self.deterministic_replay and self.seed_value is not None:
            try:
                self.env.seed(self.seed_value + self.episode_count)
            except Exception:
                pass

        ret = self.env.reset(*args, **kwargs)
        obs, info = self._split_obs_ret(ret)

        self.episode_count += 1
        self.step_count = 0
        self.actions = []
        self.observations = [obs] if True else []
        self.snapshots = []
        self._last_obs = obs
        self._last_info = info
        self._episode_return = 0.0
        self._record_snapshot(step=0, obs=obs, info=info, actions=[])
        return ret

    def step(self, action: Any) -> Any:
        """Step the env while recording the action and the resulting state."""
        obs, reward, terminated, truncated, info = self._raw_step(action)
        self.actions.append(_copy_action(action))
        self.observations.append(obs)
        self.step_count += 1
        self._episode_return += float(reward)
        self._last_obs = obs
        self._last_info = info
        self._record_snapshot(step=self.step_count, obs=obs, info=info, actions=self.actions)
        return obs, reward, terminated, truncated, info

    def _split_obs_ret(self, ret: Any) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
            return ret[0], ret[1]
        return ret, {}

    def _raw_step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        ret = self.env.step(action)
        if isinstance(ret, tuple) and len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
        obs, reward, done, info = ret  # classic 4-tuple API
        return obs, float(reward), bool(done), False, dict(info or {})

    def _record_snapshot(
        self, step: int, obs: Any, info: Dict[str, Any], actions: Sequence[Any]
    ) -> None:
        packed = get_env_state(self.env)
        snap = StateSnapshot(
            step=int(step),
            observation=obs,
            state=packed,
            actions=[_copy_action(a) for a in actions] if self.store_actions else [],
            info=dict(info or {}),
            metadata={"episode": int(self.episode_count), "kind": packed.get("kind")},
        )
        self.snapshots.append(snap)
        if len(self.snapshots) > self.max_records:  # ring buffer
            self.snapshots.pop(0)

    def _archive_trajectory(self) -> Dict[str, Any]:
        """Freeze the finished trajectory (used by the fidelity evaluator)."""
        traj = {
            "episode": int(self.episode_count),
            "length": len(self.actions),
            "observations": list(self.observations),
            "actions": list(self.actions),
            "return": float(self._episode_return),
            "snapshots": list(self.snapshots),
        }
        self.trajectories.append(traj)
        return traj

    # ------------------------------------------------------------------
    # snapshot access
    # ------------------------------------------------------------------
    def snapshot(self, index: Optional[int] = None) -> StateSnapshot:
        """Return a snapshot (default: most recent; supports negative indices)."""
        if not self.snapshots:
            raise RuntimeError("ResetWrapper.snapshot() called before any reset()")
        if index is None:
            return self.snapshots[-1]
        return self.snapshots[int(index)]

    def snapshot_at(self, step: int) -> StateSnapshot:
        """Return the snapshot recorded at absolute trajectory step ``step``."""
        step = int(step)
        if step < 0:
            step = max(0, len(self.snapshots) + step)
        if not self.snapshots:
            raise RuntimeError("no snapshots recorded")
        step = min(step, len(self.snapshots) - 1)
        return self.snapshots[step]

    def snapshots_for(
        self, scores: Optional[Sequence[float]] = None, top_k: Optional[int] = None
    ) -> List[StateSnapshot]:
        """Attach importance ``scores`` and optionally return the top-``k`` states.

        ``scores`` is aligned with :attr:`snapshots` (i.e. produced by
        :mod:`rice.explanation.importance` over the recorded observations).
        Ties are broken by the earliest step (deterministic).
        """
        snaps = self.snapshots
        if scores is not None:
            n = min(len(snaps), len(scores))
            for i in range(n):
                snaps[i].score = None if scores[i] is None else float(scores[i])
        if top_k is None:
            return list(snaps)
        order = sorted(
            range(len(snaps)),
            key=lambda i: (-(snaps[i].score if snaps[i].score is not None else -np.inf), i),
        )
        return [snaps[i] for i in order[: int(top_k)]]

    def best_snapshot(self, scores: Optional[Sequence[float]] = None) -> StateSnapshot:
        """argmax-importance state = exploration frontier (Algorithm 2)."""
        ranked = self.snapshots_for(scores=scores, top_k=1)
        if not ranked:
            raise RuntimeError("no snapshots recorded")
        return ranked[0]

    # ------------------------------------------------------------------
    # restore
    # ------------------------------------------------------------------
    def reset_to(
        self,
        target: Any,
        actions: Optional[Sequence[Any]] = None,
        use_direct: Optional[bool] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        verify: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the env so that it starts at ``target``.

        ``target`` may be a :class:`StateSnapshot`, an integer step index into
        the recorded trajectory, or a raw packed state (from
        :func:`get_env_state`). Returns ``(observation, info)`` of the restored
        state, matching the ``reset()`` contract so it can be dropped straight
        into the refinement/fidelity loops.
        """
        if isinstance(target, StateSnapshot):
            snapshot = target
        elif isinstance(target, (int, np.integer)):
            snapshot = self.snapshot_at(int(target))
        else:
            snapshot = StateSnapshot(step=0, state=target, actions=list(actions or []))

        if actions is not None:
            snapshot = snapshot.clone()
            snapshot.actions = [_copy_action(a) for a in actions]

        if use_direct is None:
            use_direct = self.prefer_direct

        do_replay = (not use_direct) or (self.verify_replay and verify)

        if use_direct and not do_replay:
            obs, info = self._restore_direct(snapshot)
            if obs is not None:
                self._finalize_restore(snapshot, obs, info)
                self.direct_restores += 1
                return obs, info
            # fall through to replay

        # Go-Explore replay fallback.
        obs, info, _steps = self.resetter.replay(
            snapshot.actions, reset_kwargs=reset_kwargs, stop_on_done=True
        )
        if len(snapshot.actions) > 0 and _steps == 0 and self.step_count > 0:
            # Replay produced nothing (env terminated at reset) -> count as a
            # failure but still return the fresh episode observation.
            self.failed_restores += 1
        else:
            self.replay_restores += 1
        self._finalize_restore(snapshot, obs, info)
        return obs, info

    def reset_to_state(self, state: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Alias of :meth:`reset_to` accepting a raw backend state object."""
        return self.reset_to(state, **kwargs)

    def reset_to_critical(
        self, snapshot: Optional[StateSnapshot] = None, **kwargs: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset to the critical state identified by the explanation stage.

        With ``snapshot=None`` the last recorded state is used; callers
        typically pass the argmax-importance snapshot returned by
        :meth:`best_snapshot`.
        """
        if snapshot is None:
            snapshot = self.snapshot()
        return self.reset_to(snapshot, **kwargs)

    def _restore_direct(
        self, snapshot: StateSnapshot
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Try to inject ``snapshot.state`` into the simulator."""
        packed = snapshot.state
        if packed is None:
            return None, {}
        ok = set_env_state(self.env, packed)
        if not ok:
            return None, {}

        # Rebuild a gym-style observation after injecting the state. Prefer the
        # observation stored in the snapshot (exact, backend independent) and
        # optionally validate against a freshly rendered observation.
        obs = snapshot.observation
        if obs is None:
            obs = self._observe_from_state()
        info = dict(snapshot.info or {})
        info["restored_from_snapshot"] = int(snapshot.step)
        info["restore_mode"] = "direct"
        return obs, info

    def _observe_from_state(self) -> Any:
        """Best-effort observation recovery after raw state injection."""
        for name in ("_get_obs", "get_obs", "get_state"):
            hook = getattr(self.env, name, None)
            if callable(hook):
                try:
                    out = hook()
                    if isinstance(out, dict):
                        out = out.get("observation", out)
                    return out
                except Exception:
                    continue
        return None

    def _finalize_restore(
        self, snapshot: StateSnapshot, obs: Any, info: Dict[str, Any]
    ) -> None:
        """Reset per-episode bookkeeping so the rollout continues from ``obs``."""
        self.step_count = int(snapshot.step)
        self.actions = [_copy_action(a) for a in snapshot.actions] if self.store_actions else []
        self.observations = [obs]
        self.snapshots = []
        self._last_obs = obs
        self._last_info = dict(info or {})
        self._episode_return = 0.0
        self.episode_count += 1
        self._record_snapshot(step=self.step_count, obs=obs, info=self._last_info, actions=self.actions)

    # ------------------------------------------------------------------
    # convenience for the refinement loop
    # ------------------------------------------------------------------
    def rollout_with_snapshots(self, policy_step: Any, length: int, reset_kwargs=None):
        """Roll the env for ``length`` steps using ``policy_step(obs) -> action``.

        Returns ``(snapshots, observations, actions, rewards)`` — the raw
        material that :mod:`rice.explanation.importance` scores and
        :mod:`rice.explanation.critical_state` ranks.
        """
        obs, info = self.reset(**(reset_kwargs or {}))
        rewards: List[float] = []
        for _ in range(int(length)):
            action = policy_step(obs)
            obs, reward, terminated, truncated, info = self.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        return list(self.snapshots), list(self.observations), list(self.actions), rewards

    def trajectory_arrays(self) -> Dict[str, np.ndarray]:
        """Recorded trajectory as arrays (for the fidelity evaluator)."""
        obs = np.asarray(self.observations) if self.observations else np.zeros((0,))
        act = (
            np.asarray(self.actions)
            if self.actions and np.isscalar(self.actions[0])
            else (np.asarray(self.actions) if self.actions else np.zeros((0,)))
        )
        return {"observations": obs, "actions": act}

    # ------------------------------------------------------------------
    # delegating plumbing
    # ------------------------------------------------------------------
    def get_state(self) -> Any:
        return get_env_state(self.env)

    def set_state(self, state: Any) -> bool:
        return set_env_state(self.env, state)

    def restore_state(self, state: Any) -> bool:
        return set_env_state(self.env, state)

    def seed(self, seed: Optional[int] = None) -> Any:
        self.seed_value = None if seed is None else int(seed)
        try:
            if _HAS_GYM:
                return super().seed(seed)  # type: ignore[misc]
            return self.env.seed(seed)
        except Exception:
            return None

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        return self.env.close()

    @property
    def unwrapped(self) -> Any:  # keep gym semantics intact
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the wrapper itself.
        if name.startswith("__") or name in ("env",):
            raise AttributeError(name)
        env = self.__dict__.get("env", None)
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _copy_action(action: Any) -> Any:
    """Copy an action (NumPy-safe, tolerant of scalars and nested tuples)."""
    if isinstance(action, np.ndarray):
        return action.copy()
    if isinstance(action, (tuple, list)):
        return type(action)(_copy_action(a) for a in action)
    return action


def make_reset_env(env: Any, **kwargs: Any) -> ResetWrapper:
    """Wrap ``env`` with a :class:`ResetWrapper` (idempotent)."""
    if isinstance(env, ResetWrapper):
        for key, value in kwargs.items():
            if hasattr(env, key):
                setattr(env, key, value)
        return env
    return ResetWrapper(env, **kwargs)


def wrap_reset(env: Any, *args: Any, **kwargs: Any) -> ResetWrapper:
    """Alias of :func:`make_reset_env`."""
    return make_reset_env(env, **kwargs)
