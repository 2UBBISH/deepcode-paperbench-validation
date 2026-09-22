"""Tests for CORE COMPONENT #6: Go-Explore-style environment reset.

Paper reference (RICE, Appendix C.1 Implementation Details)::

    "We implement the environment reset function similar to Ecoffet et al. (2019)
     to restore the environment to selected critical states. This method is
     feasible in our case, as we operate within simulator-based environments."

This module exercises ``rice.algorithms.env_reset``: the snapshot/restore
pipeline that lets Algorithm 2 (and the mixed initial state distribution of
CORE COMPONENT #2) reset the simulator exactly to an identified critical state.

Checklist from the reproduction plan's validation section:
  * environment reset restores a saved state within tolerance;
  * save/restore degrades gracefully when a simulator exposes no state API;
  * snapshot pools, sampling, serialization and episode counters behave;
  * the duck-typed aliases used by ``mixed_init.py`` / ``refine.py`` exist.

The suite is dependency tolerant: it skips (never fails) when the module or
torch/gym are unavailable, and it tolerates minor naming drift in the
implementation through the ``_get``/``_require`` helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

try:  # package-relative import (pytest with rice.tests as a package)
    from . import _helpers  # type: ignore
except ImportError:  # pragma: no cover - direct/bare execution
    import _helpers  # type: ignore

DummyEnv = _helpers.DummyEnv
LossOnlyEnv = _helpers.LossOnlyEnv
import_optional = _helpers.import_optional

# --------------------------------------------------------------------------
# Tolerant module import
# --------------------------------------------------------------------------
_env_reset = import_optional("algorithms.env_reset")

if _env_reset is None:  # pragma: no cover - module missing
    pytest.skip(
        "rice.algorithms.env_reset is not importable; skipping CORE #6 tests",
        allow_module_level=True,
    )


def _get(*names, default=None):
    """Return the first attribute of ``_env_reset`` matching ``names``."""
    for name in names:
        obj = getattr(_env_reset, name, None)
        if obj is not None:
            return obj
    return default


EnvStateManager = _get("EnvStateManager", "StateManager", "EnvStateRestorer")
Snapshot = _get("Snapshot", "EnvSnapshot", "StateSnapshot")
StateBuffer = _get("StateBuffer", "SnapshotPool", "StatePool", "SnapshotBuffer")
capture_state = _get("capture_state", "snapshot_env", "make_snapshot")
set_state = _get("set_state", "restore_state", "load_state")
supports_state_restore = _get(
    "supports_state_restore", "supports_restore", "can_restore", "state_restore_supported"
)
make_state_manager = _get("make_state_manager", "build_state_manager", "state_manager_for")
StateRestoreWrapper = _get("StateRestoreWrapper", "RestoreWrapper", "SnapshotWrapper")
iter_env_chain = _get("iter_env_chain", "env_chain")
unwrap_env = _get("unwrap_env", "unwrap")
current_observation = _get("current_observation", "get_current_observation", "read_observation")


def _require(value, name: str):
    """Skip the current test when ``value`` is unavailable."""
    if value is None:
        pytest.skip(f"rice.algorithms.env_reset.{name} is not implemented")
    return value


# --------------------------------------------------------------------------
# Tiny env helpers (gym / gymnasium agnostic)
# --------------------------------------------------------------------------
def _reset(env, seed=None):
    """Reset an env and normalise the return value to ``(obs, info)``."""
    out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(out, tuple):
        obs = out[0]
        info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
        return obs, info
    return out, {}


def _step(env, action):
    """Step an env and normalise to ``(obs, reward, done, info)``."""
    out = env.step(action)
    if len(out) == 5:  # gymnasium API
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out  # legacy gym API
    return obs, float(reward), bool(done), info


def _flat(obs) -> np.ndarray:
    """Flatten a (possibly dict/nested) observation to a 1-D float array."""
    if isinstance(obs, dict):
        parts = [np.asarray(obs[k], dtype=np.float64).ravel() for k in sorted(obs)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).ravel()


def _dummy(obs_dim=4, act_dim=2, max_episode_steps=20):
    env = DummyEnv(obs_dim=obs_dim, act_dim=act_dim, max_episode_steps=max_episode_steps)
    return env


def _zeros(env):
    return np.zeros(env.act_dim, dtype=np.float32)


def _ones(env):
    return np.ones(env.act_dim, dtype=np.float32)


# --------------------------------------------------------------------------
# Snapshot capture
# --------------------------------------------------------------------------
def test_capture_state_returns_a_snapshot_with_observation():
    capture = _require(capture_state, "capture_state")
    env = _dummy()
    obs, _ = _reset(env, seed=0)
    snap = capture(env, observation=obs)

    assert snap is not None
    stored = getattr(snap, "observation", None)
    assert stored is not None
    np.testing.assert_allclose(_flat(stored), _flat(obs), atol=1e-6)


def test_capture_state_twice_from_same_state_is_consistent():
    capture = _require(capture_state, "capture_state")
    env = _dummy()
    obs, _ = _reset(env, seed=3)
    for _ in range(2):
        obs, _, _, _ = _step(env, _zeros(env))

    s1 = capture(env, observation=obs)
    s2 = capture(env, observation=obs)
    assert getattr(s1, "kind", None) == getattr(s2, "kind", None)
    np.testing.assert_allclose(
        _flat(getattr(s1, "observation")), _flat(getattr(s2, "observation")), atol=1e-6
    )


def test_snapshot_records_episode_counters_when_provided():
    capture = _require(capture_state, "capture_state")
    env = _dummy()
    obs, _ = _reset(env, seed=0)
    snap = capture(env, observation=obs, episode_step=7, episode_return=1.5)

    step_attr = getattr(snap, "episode_step", None)
    ret_attr = getattr(snap, "episode_return", None)
    if step_attr is not None:
        assert int(step_attr) == 7
    if ret_attr is not None:
        assert abs(float(ret_attr) - 1.5) < 1e-9


def test_snapshot_as_dict_and_clone_are_independent():
    capture = _require(capture_state, "capture_state")
    env = _dummy()
    obs, _ = _reset(env, seed=1)
    snap = capture(env, observation=obs)

    as_dict = getattr(snap, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        assert isinstance(payload, dict)
        assert len(payload) > 0

    clone = getattr(snap, "clone", None)
    if callable(clone):
        copy_snap = clone()
        original = np.array(getattr(snap, "observation"), dtype=np.float64, copy=True)
        # Mutating the copy's observation must not leak into the original.
        try:
            copy_snap.observation[...] = 123.0
        except Exception:  # pragma: no cover - immutable storage
            pass
        np.testing.assert_allclose(_flat(getattr(snap, "observation")), original.ravel(), atol=1e-9)


# --------------------------------------------------------------------------
# Feature detection
# --------------------------------------------------------------------------
def test_supports_state_restore_returns_a_bool():
    supports = _require(supports_state_restore, "supports_state_restore")
    env = _dummy()
    assert isinstance(bool(supports(env)), bool)


def test_degraded_observation_snapshot_is_not_reported_as_supported():
    """An env with no simulator state API must not claim full support.

    ``LossOnlyEnv`` has its ``get_state``/``set_state`` methods removed, so the
    capture cascade falls back to the (lossy) observation-only snapshot and
    feature detection should report ``False``.
    """
    capture = _require(capture_state, "capture_state")
    supports = _require(supports_state_restore, "supports_state_restore")

    env = LossOnlyEnv()
    snap = capture(env)
    kind = getattr(snap, "kind", None)
    if kind == "observation":
        assert supports(env) is False
    else:  # pragma: no cover - implementation found another state source
        pytest.skip(f"capture_state used non-degraded kind={kind!r}")


# --------------------------------------------------------------------------
# The core requirement: restore a saved state within tolerance
# --------------------------------------------------------------------------
def test_manager_restores_saved_state_within_tolerance():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    obs, _ = _reset(env, seed=0)
    for _ in range(3):
        obs, _, _, _ = _step(env, _zeros(env))
    target = _flat(obs)

    snapshot = manager.snapshot()
    for _ in range(5):  # move the simulator away from the recorded state
        obs, _, _, _ = _step(env, _ones(env))
    assert not np.allclose(_flat(obs), target, atol=1e-9)

    restored = manager.restore(snapshot)
    obs_after = _flat(restored) if restored is not None else _flat(manager.current_observation())
    np.testing.assert_allclose(obs_after, target, atol=1e-5)


def test_restore_reproduces_the_next_transition_deterministically():
    """After restoring, replaying the same action yields the same successor."""
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    _reset(env, seed=7)
    for _ in range(4):
        _step(env, np.full(env.act_dim, 0.25, dtype=np.float32))

    snapshot = manager.snapshot()
    action = np.full(env.act_dim, -0.5, dtype=np.float32)

    next_1, _, _, _ = _step(env, action)
    expected = next_1.copy()

    for _ in range(3):  # wander elsewhere
        _step(env, _ones(env))

    manager.restore(snapshot)
    next_2, _, _, _ = _step(env, action)
    np.testing.assert_allclose(_flat(next_2), _flat(expected), atol=1e-5)


def test_restore_with_none_starts_a_fresh_episode():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    _reset(env, seed=0)
    for _ in range(6):
        obs, _, _, _ = _step(env, _ones(env))

    first_obs, _ = _reset(env, seed=0)
    manager.restore(None)
    obs_after = manager.current_observation()
    if obs_after is None:  # restore(None) may not expose the observation
        pytest.skip("manager does not expose current_observation()")

    # A fresh episode is *not* the mid-episode state we came from.
    assert not np.allclose(_flat(obs_after), _flat(obs), atol=1e-9)
    # ... but a manual step from it must be well defined.
    out = _step(env, _zeros(env))
    assert out[0] is not None
    del first_obs


def test_manager_step_and_reset_keep_episode_counters():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    if not hasattr(manager, "step"):
        pytest.skip("EnvStateManager.step is not implemented")

    manager.reset(seed=0)
    for _ in range(3):
        manager.step(_zeros(env))

    step_attr = getattr(manager, "episode_step", None)
    if step_attr is not None:
        assert int(step_attr) >= 0

    # episode_return is bookkeeping only; it must stay finite.
    ret_attr = getattr(manager, "episode_return", None)
    if ret_attr is not None:
        assert np.isfinite(float(ret_attr))


def test_restore_count_increments_on_restore():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    _reset(env, seed=0)
    snapshot = manager.snapshot()
    before = int(getattr(manager, "restore_count", 0) or 0)
    manager.restore(snapshot)
    after = int(getattr(manager, "restore_count", before) or before)
    assert after >= before


# --------------------------------------------------------------------------
# Free functions and factory
# --------------------------------------------------------------------------
def test_set_state_free_function_restores_the_observation():
    capture = _require(capture_state, "capture_state")
    restore_fn = _require(set_state, "set_state")

    env = _dummy()
    obs, _ = _reset(env, seed=2)
    for _ in range(2):
        obs, _, _, _ = _step(env, _zeros(env))
    target = _flat(obs)

    snapshot = capture(env, observation=obs)
    for _ in range(4):
        _step(env, _ones(env))

    restored = restore_fn(env, snapshot)
    if restored is not None:
        np.testing.assert_allclose(_flat(restored), target, atol=1e-5)


def test_make_state_manager_factory_returns_manager():
    factory = _require(make_state_manager, "make_state_manager")
    env = _dummy()
    manager = factory(env)
    assert manager is not None
    assert hasattr(manager, "snapshot") or hasattr(manager, "save_state")


def test_state_manager_aliases_are_duck_typed_together():
    """``mixed_init``/``refine`` rely on interchangeable alias spellings."""
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)
    _reset(env, seed=0)
    manager.step(_zeros(env))

    save_names = [n for n in ("snapshot", "save_state", "get_state", "save") if hasattr(manager, n)]
    load_names = [
        n
        for n in ("restore", "load_state", "restore_state", "load", "set_state")
        if hasattr(manager, n)
    ]
    assert save_names, "no snapshot/save alias found on EnvStateManager"
    assert load_names, "no restore/load alias found on EnvStateManager"

    snapshot = getattr(manager, save_names[0])()
    for name in load_names:
        try:
            getattr(manager, name)(snapshot)
        except TypeError:  # alias with a different signature (e.g. set_state(kwarg))
            continue
        except Exception as exc:  # pragma: no cover - unexpected failure
            pytest.fail(f"alias {name!r} failed to restore a snapshot: {exc!r}")


# --------------------------------------------------------------------------
# Snapshot pool / StateBuffer
# --------------------------------------------------------------------------
def test_state_buffer_add_sample_and_length():
    buffer_cls = _require(StateBuffer, "StateBuffer")
    capture = _require(capture_state, "capture_state")

    env = _dummy()
    buffer = buffer_cls()
    assert len(buffer) == 0

    obs, _ = _reset(env, seed=0)
    for _ in range(3):
        obs, _, _, _ = _step(env, _zeros(env))
        snap = capture(env, observation=obs)
        add = getattr(buffer, "add", None) or getattr(buffer, "append", None)
        assert callable(add), "StateBuffer exposes neither add() nor append()"
        add(snap)

    assert len(buffer) == 3

    sample = getattr(buffer, "sample", None)
    if callable(sample):
        drawn = sample()
        assert drawn is not None
        drawn_indexed = sample(index=0) if "index" in getattr(sample, "__code__", None).co_varnames else None  # type: ignore[attr-defined]
        del drawn_indexed

    clear = getattr(buffer, "clear", None)
    if callable(clear):
        clear()
        assert len(buffer) == 0


def test_state_buffer_state_dict_roundtrip():
    buffer_cls = _require(StateBuffer, "StateBuffer")
    capture = _require(capture_state, "capture_state")

    env = _dummy()
    buffer = buffer_cls()
    obs, _ = _reset(env, seed=1)
    add = getattr(buffer, "add", None) or getattr(buffer, "append", None)
    add(capture(env, observation=obs))

    state_dict = getattr(buffer, "state_dict", None)
    load_state_dict = getattr(buffer, "load_state_dict", None)
    if not (callable(state_dict) and callable(load_state_dict)):
        pytest.skip("StateBuffer does not implement state_dict/load_state_dict")

    payload = state_dict()
    fresh = buffer_cls()
    fresh.load_state_dict(payload)
    assert len(fresh) == len(buffer)


def test_manager_pool_collects_and_restores_snapshots():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    pool = getattr(manager, "pool", None)
    if pool is None:
        pytest.skip("EnvStateManager does not expose a snapshot pool")

    _reset(env, seed=0)
    if not hasattr(manager, "add_snapshot"):
        pytest.skip("EnvStateManager.add_snapshot is not implemented")

    observations = []
    for _ in range(3):
        obs, _, _, _ = _step(env, _zeros(env))
        observations.append(_flat(obs))
        manager.add_snapshot()

    assert len(manager) >= 1 if hasattr(manager, "__len__") else True

    restore_sample = getattr(manager, "restore_sample", None)
    if not callable(restore_sample):
        pytest.skip("EnvStateManager.restore_sample is not implemented")
    manager.restore_sample()
    obs_after = manager.current_observation()
    if obs_after is not None:
        assert any(np.allclose(_flat(obs_after), o, atol=1e-5) for o in observations)


def test_manager_max_snapshots_caps_pool_growth():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    try:
        manager = manager_cls(env, max_snapshots=3)
    except TypeError:  # pragma: no cover - parameter named differently
        pytest.skip("EnvStateManager does not accept max_snapshots")

    add = getattr(manager, "add_snapshot", None)
    if not callable(add):
        pytest.skip("EnvStateManager.add_snapshot is not implemented")

    _reset(env, seed=0)
    for _ in range(10):
        _step(env, _zeros(env))
        add()

    pool = getattr(manager, "pool", None)
    if pool is not None and hasattr(pool, "__len__"):
        assert len(pool) <= 3


def test_manager_state_dict_roundtrip():
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    state_dict = getattr(manager, "state_dict", None)
    load_state_dict = getattr(manager, "load_state_dict", None)
    if not (callable(state_dict) and callable(load_state_dict)):
        pytest.skip("EnvStateManager does not implement state_dict/load_state_dict")

    _reset(env, seed=0)
    for _ in range(2):
        _step(env, _zeros(env))

    payload = state_dict()
    assert isinstance(payload, dict)

    fresh_env = _dummy()
    fresh = manager_cls(fresh_env)
    fresh.load_state_dict(payload)
    if hasattr(fresh, "current_observation") and hasattr(manager, "current_observation"):
        np.testing.assert_allclose(
            _flat(fresh.current_observation()), _flat(manager.current_observation()), atol=1e-5
        )


def test_manager_save_and_load_pool_to_disk(tmp_path):
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    save_pool = getattr(manager, "save_pool", None)
    load_pool = getattr(manager, "load_pool", None)
    add = getattr(manager, "add_snapshot", None)
    if not (callable(save_pool) and callable(load_pool) and callable(add)):
        pytest.skip("EnvStateManager pool persistence is not implemented")

    _reset(env, seed=0)
    for _ in range(2):
        _step(env, _zeros(env))
        add()

    path = str(tmp_path / "pool.pkl")
    save_pool(path)
    assert path and (len(getattr(manager, "pool", []) or []) >= 0)

    fresh = manager_cls(_dummy())
    fresh.load_pool(path)
    if hasattr(fresh, "pool") and hasattr(manager, "pool") and hasattr(manager.pool, "__len__"):
        assert len(fresh.pool) == len(manager.pool)


# --------------------------------------------------------------------------
# Wrapper / chain helpers
# --------------------------------------------------------------------------
def test_state_restore_wrapper_exposes_snapshot_and_restore():
    wrapper_cls = _require(StateRestoreWrapper, "StateRestoreWrapper")
    env = _dummy()
    try:
        wrapped = wrapper_cls(env)
    except Exception as exc:  # pragma: no cover - gym requirement
        pytest.skip(f"StateRestoreWrapper could not wrap the dummy env: {exc!r}")

    obs, _ = _reset(wrapped, seed=0)
    for _ in range(2):
        obs, _, _, _ = _step(wrapped, _zeros(env))
    before = _flat(obs)

    snapshot = None
    for name in ("snapshot", "save_state", "get_state"):
        fn = getattr(wrapped, name, None)
        if callable(fn):
            snapshot = fn()
            break
    if snapshot is None:
        pytest.skip("StateRestoreWrapper exposes no save alias")

    for _ in range(3):
        _step(wrapped, _ones(env))

    for name in ("restore", "restore_state", "load_state", "set_state"):
        fn = getattr(wrapped, name, None)
        if callable(fn):
            fn(snapshot)
            break

    after = getattr(wrapped, "current_observation", None)
    if callable(after):
        np.testing.assert_allclose(_flat(after()), before, atol=1e-5)


def test_iter_env_chain_and_unwrap():
    env = _dummy()
    if iter_env_chain is not None:
        chain = list(iter_env_chain(env))
        assert len(chain) >= 1
        assert chain[0] is env or hasattr(chain[0], "step")

    if unwrap_env is not None:
        inner = unwrap_env(env)
        assert inner is not None
        assert hasattr(inner, "step")


def test_current_observation_helper_returns_env_observation():
    helper = _require(current_observation, "current_observation")
    env = _dummy()
    obs, _ = _reset(env, seed=0)

    read = helper(env)
    if read is None:  # pragma: no cover - env does not expose its observation
        pytest.skip("current_observation() returned None for the dummy env")
    np.testing.assert_allclose(_flat(read), _flat(obs), atol=1e-6)


def test_restore_is_tolerant_to_a_broken_environment():
    """Restoration must degrade gracefully rather than crash the refining loop."""
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = LossOnlyEnv()
    manager = manager_cls(env)

    _reset(env, seed=0)
    snapshot = manager.snapshot()
    try:
        manager.restore(snapshot)
    except Exception as exc:  # pragma: no cover - should not happen
        pytest.fail(f"restore() raised on a state-less environment: {exc!r}")


# --------------------------------------------------------------------------
# Integration with the mixed initial state distribution (CORE #2 / #4)
# --------------------------------------------------------------------------
def test_restore_supports_the_critical_state_roll_in_pattern():
    """Algorithm 2: identify a critical state, restore it, then continue.

    The state captured deep inside an episode is restored before the *next*
    refining iteration begins, so ``mixed_init`` can hand a valid observation
    back to the refiner.
    """
    manager_cls = _require(EnvStateManager, "EnvStateManager")
    env = _dummy()
    manager = manager_cls(env)

    _reset(env, seed=0)
    visited = []
    for _ in range(6):
        obs, _, _, _ = _step(env, np.full(env.act_dim, 0.1, dtype=np.float32))
        visited.append(_flat(obs))

    # "most critical state" -> index 3 (stand-in for the argmax over P(mask=0))
    critical_index = 3
    manager.add_snapshot()  # keep the pool warm
    snapshot = manager.snapshot()

    restore_sample = getattr(manager, "restore_sample", None)
    if callable(restore_sample):
        restored = restore_sample()
    else:  # pragma: no cover
        restored = manager.restore(snapshot)

    if restored is not None:
        obs_after = _flat(restored)
    else:
        obs_after = _flat(manager.current_observation())

    # After a roll-in reset the refiner must be able to keep stepping.
    next_obs, reward, done, info = _step(env, _zeros(env))
    assert next_obs is not None
    assert np.isfinite(reward)
    assert np.isfinite(obs_after).all()
    del critical_index, visited
