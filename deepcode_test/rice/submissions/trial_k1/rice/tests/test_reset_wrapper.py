"""Unit tests for the mixed-initial-state reset wrapper (ResettableEnv)."""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pytest

from rice.envs.resettable_env import (
    CriticalStateBuffer,
    ResettableEnv,
    make_resettable,
)


class DummyEnv(gym.Env):
    """Minimal deterministic environment for testing reset behaviour."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._state = np.zeros(4, dtype=np.float32)
        self._step_count = 0
        self._seed = seed

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._state = np.ones(4, dtype=np.float32) * (seed if seed is not None else self._seed)
        self._step_count = 0
        return self._state.copy(), {}

    def step(self, action: int):
        self._state += float(action)
        self._step_count += 1
        reward = float(action)
        terminated = self._step_count >= 10
        truncated = False
        return self._state.copy(), reward, terminated, truncated, {"step": self._step_count}

    def get_simulator_state(self):
        return {"state": self._state.copy(), "step": self._step_count}

    def set_simulator_state(self, state: Dict[str, Any]):
        self._state = state["state"].copy()
        self._step_count = state["step"]


@pytest.fixture
def buffer() -> CriticalStateBuffer:
    return CriticalStateBuffer(capacity=10)


def test_buffer_add_and_sample(buffer: CriticalStateBuffer):
    state = {"obs": np.ones(3), "simulator_state": {"x": 1}}
    buffer.add(state)
    assert len(buffer) == 1
    sampled = buffer.sample()
    assert sampled is not None
    assert np.allclose(sampled["obs"], state["obs"])


def test_buffer_capacity_fifo(buffer: CriticalStateBuffer):
    for i in range(15):
        buffer.add({"obs": np.array([i]), "simulator_state": {"x": i}})
    assert len(buffer) == 10
    # oldest entries should have been evicted
    obs_values = [entry["obs"][0] for entry in buffer._entries]
    assert obs_values == list(range(5, 15))


def test_buffer_top_k(buffer: CriticalStateBuffer):
    for i in range(10):
        buffer.add({"obs": np.array([i]), "mask_score": float(i) / 10.0})
    top = buffer.top_k(3)
    assert len(top) == 3
    # highest scores first
    assert [t["mask_score"] for t in top] == [0.9, 0.8, 0.7]


def test_buffer_save_load(buffer: CriticalStateBuffer):
    for i in range(5):
        buffer.add({"obs": np.array([i]), "simulator_state": {"x": i}})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "buffer.pkl"
        buffer.save(path)
        loaded = CriticalStateBuffer.load(path)
        assert len(loaded) == len(buffer)
        for orig, new in zip(buffer._entries, loaded._entries):
            assert np.allclose(orig["obs"], new["obs"])


def test_resettable_env_default_reset():
    env = DummyEnv(seed=42)
    wrapped = ResettableEnv(env, critical_buffer=None, p=0.0)
    obs, info = wrapped.reset(seed=42)
    assert np.allclose(obs, np.ones(4) * 42)
    assert not wrapped.last_reset_from_critical


def test_resettable_env_critical_reset_reproduces_state():
    env = DummyEnv(seed=0)
    wrapped = ResettableEnv(env, critical_buffer=None, p=1.0)

    # Run a few steps and capture a critical state.
    obs, _ = wrapped.reset(seed=0)
    for _ in range(3):
        obs, _, _, _, _ = wrapped.step(1)
    critical_state = {
        "obs": obs.copy(),
        "simulator_state": env.get_simulator_state(),
        "mask_score": 0.9,
    }

    buffer = CriticalStateBuffer(capacity=10)
    buffer.add(critical_state)
    wrapped = ResettableEnv(DummyEnv(seed=0), critical_buffer=buffer, p=1.0)

    reset_obs, _ = wrapped.reset(seed=999)
    assert wrapped.last_reset_from_critical
    assert np.allclose(reset_obs, critical_state["obs"])
    # The internal simulator state should match the stored one.
    assert env.get_simulator_state()["step"] == critical_state["simulator_state"]["step"]


def test_resettable_env_mixed_distribution():
    env = DummyEnv(seed=0)
    buffer = CriticalStateBuffer(capacity=1)
    buffer.add({"obs": np.zeros(4), "simulator_state": {"state": np.zeros(4), "step": 0}})
    wrapped = ResettableEnv(env, critical_buffer=buffer, p=0.5)

    counts = {"critical": 0, "default": 0}
    for _ in range(100):
        wrapped.reset(seed=0)
        if wrapped.last_reset_from_critical:
            counts["critical"] += 1
        else:
            counts["default"] += 1

    # With p=0.5 over 100 trials we expect roughly 50/50; allow wide tolerance.
    assert counts["critical"] > 20
    assert counts["default"] > 20


def test_make_resettable_factory():
    env = gym.make("CartPole-v1")
    wrapped = make_resettable(env, p=0.25)
    assert isinstance(wrapped, ResettableEnv)
    assert wrapped._p == 0.25


def test_resettable_env_add_critical_state():
    env = DummyEnv(seed=0)
    wrapped = ResettableEnv(env, critical_buffer=CriticalStateBuffer(capacity=2), p=0.0)
    obs, _ = wrapped.reset(seed=0)
    for _ in range(2):
        obs, _, _, _, _ = wrapped.step(1)
    wrapped.add_critical_state(obs, mask_score=0.8)
    assert len(wrapped.critical_buffer) == 1
    assert wrapped.critical_buffer.top_k(1)[0]["mask_score"] == pytest.approx(0.8)


def test_resettable_env_save_load_buffer():
    env = DummyEnv(seed=0)
    wrapped = ResettableEnv(env, critical_buffer=CriticalStateBuffer(capacity=2), p=0.0)
    obs, _ = wrapped.reset(seed=0)
    wrapped.add_critical_state(obs, mask_score=0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "crit_buf.pkl"
        wrapped.save_buffer(path)
        # Create a fresh wrapper and load the buffer.
        wrapped2 = make_resettable(DummyEnv(seed=0), p=1.0, buffer_path=path)
        assert len(wrapped2.critical_buffer) == 1
        reset_obs, _ = wrapped2.reset(seed=12345)
        assert wrapped2.last_reset_from_critical
        assert np.allclose(reset_obs, obs)


@pytest.mark.skipif(
    not any(
        env_id.startswith(prefix)
        for prefix in ("Hopper", "Walker2d", "HalfCheetah", "Reacher")
        for env_id in gym.envs.registry.keys()
    ),
    reason="MuJoCo environments not available",
)
def test_mujoco_state_restore():
    """Best-effort check that MuJoCo simulator state can be saved and restored."""
    env = gym.make("Hopper-v4")
    wrapped = ResettableEnv(env, critical_buffer=CriticalStateBuffer(capacity=1), p=1.0)
    obs, _ = wrapped.reset(seed=0)
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = wrapped.step(action)
        if terminated or truncated:
            obs, _ = wrapped.reset(seed=0)

    critical_state = {
        "obs": obs.copy(),
        "simulator_state": env.unwrapped.get_state(),
        "mask_score": 1.0,
    }
    wrapped.critical_buffer.add(critical_state)

    reset_obs, _ = wrapped.reset(seed=999)
    assert wrapped.last_reset_from_critical
    # Observations should be very close after restoring simulator state.
    assert np.allclose(reset_obs, critical_state["obs"], atol=1e-5)


def test_resettable_env_info_flag():
    env = DummyEnv(seed=0)
    buffer = CriticalStateBuffer(capacity=1)
    buffer.add({"obs": np.zeros(4), "simulator_state": {"state": np.zeros(4), "step": 0}})
    wrapped = ResettableEnv(env, critical_buffer=buffer, p=1.0)
    _, info = wrapped.reset(seed=0)
    assert info.get("reset_from_critical") is True


def test_buffer_clear(buffer: CriticalStateBuffer):
    buffer.add({"obs": np.zeros(2)})
    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.sample() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
