"""Unit-test suite for the RICE refinement module.

This module exercises the critical-state buffer, mixed-reset wrapper, RND bonus,
and the end-to-end refinement trainer on tiny synthetic environments so the
pipeline can be validated quickly without heavy task dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for older gym installs
    import gym
    from gym import spaces  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rice.agents import PPOConfig
from rice.agents.target_policy import MLPActorCritic, TorchTargetPolicy
from rice.refine import (
    CriticalState,
    CriticalStateBuffer,
    MixedResetEnv,
    RefineTrainer,
    RNDBonus,
    RNDRewardWrapper,
    build_critical_buffer_from_trajectories,
    default_restore_state,
    refine_policy,
)
from rice.refine.mixed_reset_env import default_fallback_reset


class DummyEnv(gym.Env):
    """Minimal continuous-control environment for refinement tests."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, max_steps: int = 10):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim), high=np.ones(obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.ones(act_dim), high=np.ones(act_dim), dtype=np.float32
        )
        self._state: Optional[np.ndarray] = None
        self._steps: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._state = self.observation_space.sample()
        self._steps = 0
        return self._state.astype(np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        self._state = self._state + 0.1 * np.asarray(action, dtype=np.float32)
        self._state = np.clip(self._state, self.observation_space.low, self.observation_space.high)
        reward = float(np.sum(self._state))
        terminated = bool(self._steps >= self.max_steps)
        truncated = False
        info = {"steps": self._steps}
        return self._state.astype(np.float32), reward, terminated, truncated, info

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        self._state = np.asarray(state, dtype=np.float32)


def _make_target_policy(obs_dim: int, act_dim: int, discrete: bool = False) -> TorchTargetPolicy:
    model = MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_sizes=(16, 16),
        discrete=discrete,
    )
    return TorchTargetPolicy(
        model=model,
        observation_space=spaces.Box(low=-np.ones(obs_dim), high=np.ones(obs_dim), dtype=np.float32),
        action_space=(
            spaces.Discrete(act_dim)
            if discrete
            else spaces.Box(low=-np.ones(act_dim), high=np.ones(act_dim), dtype=np.float32)
        ),
        device="cpu",
    )


def _make_trajectories(n: int = 5, length: int = 10, obs_dim: int = 4) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    trajectories: List[Dict[str, Any]] = []
    for _ in range(n):
        observations = rng.uniform(-1, 1, size=(length, obs_dim)).astype(np.float32)
        xi_values = rng.uniform(0, 1, size=length).astype(np.float32)
        trajectories.append({"observations": observations, "xi": xi_values})
    return trajectories


# --------------------------------------------------------------------------- #
# CriticalStateBuffer tests
# --------------------------------------------------------------------------- #


def test_critical_state_buffer_add_and_sample():
    buffer = CriticalStateBuffer(capacity=10)
    obs = np.ones(4, dtype=np.float32)
    for i in range(5):
        buffer.add_state(CriticalState(observation=obs * i, xi=float(i) / 5.0))

    assert len(buffer) == 5
    sample = buffer.sample()
    assert isinstance(sample, CriticalState)
    assert sample.observation.shape == (4,)


def test_critical_state_buffer_top_p_selection():
    trajectories = _make_trajectories(n=1, length=20, obs_dim=4)
    buffer = build_critical_buffer_from_trajectories(
        trajectories, capacity=None, selection_mode="top_p", top_p=0.25
    )
    # 25% of 20 = 5 states should be retained.
    assert len(buffer) == 5
    # All retained states should have xi >= the 75th percentile of the original.
    all_xi = trajectories[0]["xi"]
    cutoff = np.percentile(all_xi, 75)
    for state in buffer.get_all():
        assert state.xi >= cutoff - 1e-6


def test_critical_state_buffer_threshold_selection():
    trajectories = _make_trajectories(n=2, length=10, obs_dim=4)
    buffer = build_critical_buffer_from_trajectories(
        trajectories, selection_mode="threshold", threshold=0.7
    )
    for state in buffer.get_all():
        assert state.xi >= 0.7 - 1e-6


def test_critical_state_buffer_capacity_eviction():
    buffer = CriticalStateBuffer(capacity=3)
    for i in range(10):
        buffer.add_state(CriticalState(observation=np.ones(2, dtype=np.float32), xi=float(i) / 10.0))
    assert len(buffer) == 3
    # Only the highest-xi states should remain.
    min_xi = min(state.xi for state in buffer.get_all())
    assert min_xi >= 0.7 - 1e-6


def test_critical_state_buffer_save_load(tmp_path: Path):
    buffer = CriticalStateBuffer(capacity=5)
    for i in range(5):
        buffer.add_state(
            CriticalState(observation=np.arange(3, dtype=np.float32) + i, xi=float(i) / 5.0)
        )
    path = tmp_path / "buffer.npz"
    buffer.save(str(path))
    loaded = CriticalStateBuffer.load(str(path))
    assert len(loaded) == len(buffer)


# --------------------------------------------------------------------------- #
# MixedResetEnv tests
# --------------------------------------------------------------------------- #


def test_mixed_reset_env_default_reset():
    base = DummyEnv()
    buffer = CriticalStateBuffer()
    buffer.add_state(CriticalState(observation=np.ones(4, dtype=np.float32) * 0.5, xi=0.9))

    env = MixedResetEnv(base, buffer, p=0.0)
    obs, info = env.reset(seed=0)
    assert env.last_reset_source == "default"
    assert not np.allclose(obs, 0.5 * np.ones(4))


def test_mixed_reset_env_critical_reset():
    base = DummyEnv()
    base.reset(seed=0)
    critical_obs = np.ones(4, dtype=np.float32) * 0.42
    buffer = CriticalStateBuffer()
    buffer.add_state(CriticalState(observation=critical_obs, xi=0.9))

    env = MixedResetEnv(base, buffer, p=1.0, restore_fn=default_restore_state)
    obs, info = env.reset(seed=1)
    assert env.last_reset_source == "critical"
    assert np.allclose(obs, critical_obs, atol=1e-5)


def test_mixed_reset_env_set_p():
    base = DummyEnv()
    buffer = CriticalStateBuffer()
    env = MixedResetEnv(base, buffer, p=0.25)
    env.set_p(0.75)
    assert env.p == 0.75


def test_default_restore_state_uses_set_state():
    base = DummyEnv()
    base.reset(seed=0)
    critical_obs = np.ones(4, dtype=np.float32) * 0.33
    state = CriticalState(observation=critical_obs, xi=0.8)
    obs = default_restore_state(base, state)
    assert np.allclose(obs, critical_obs, atol=1e-5)


def test_default_fallback_reset():
    base = DummyEnv()
    base.reset(seed=0)
    state = CriticalState(observation=np.zeros(4, dtype=np.float32), xi=0.5)
    obs = default_fallback_reset(base, state)
    assert obs.shape == (4,)


# --------------------------------------------------------------------------- #
# RND bonus tests
# --------------------------------------------------------------------------- #


def test_rnd_bonus_shape_and_normalization():
    obs_dim = 4
    rnd = RNDBonus(obs_dim=obs_dim, hidden_sizes=(16, 16), output_dim=8, device="cpu")
    obs = np.random.randn(10, obs_dim).astype(np.float32)
    bonus = rnd.compute_bonus(obs, normalize=True)
    assert bonus.shape == (10,)
    assert np.all(np.isfinite(bonus))


def test_rnd_bonus_predictor_update():
    obs_dim = 4
    rnd = RNDBonus(obs_dim=obs_dim, hidden_sizes=(16, 16), output_dim=8, lr=1e-2, device="cpu")
    obs = np.random.randn(32, obs_dim).astype(np.float32)
    stats_before = rnd.state_dict()
    loss_info = rnd.update(obs)
    assert "loss" in loss_info
    stats_after = rnd.state_dict()
    # Predictor parameters should have changed; target parameters should not.
    for key in stats_before["predictor"].keys():
        assert not torch.allclose(stats_before["predictor"][key], stats_after["predictor"][key])
    for key in stats_before["target"].keys():
        assert torch.allclose(stats_before["target"][key], stats_after["target"][key])


def test_rnd_reward_wrapper():
    base = DummyEnv()
    rnd = RNDBonus(obs_dim=base.obs_dim, hidden_sizes=(8, 8), output_dim=4, device="cpu")
    env = RNDRewardWrapper(base, rnd, lambda_rnd=0.1, update_every=1)
    obs, _ = env.reset(seed=0)
    obs_next, reward, terminated, truncated, info = env.step(np.zeros(base.act_dim))
    assert "rnd_bonus" in info
    assert "rnd_reward" in info
    assert np.isfinite(reward)


# --------------------------------------------------------------------------- #
# RefineTrainer tests
# --------------------------------------------------------------------------- #


def test_refine_trainer_short_run():
    base = DummyEnv(obs_dim=4, act_dim=2, max_steps=10)
    target_policy = _make_target_policy(4, 2)

    trajectories = _make_trajectories(n=3, length=10, obs_dim=4)
    buffer = build_critical_buffer_from_trajectories(trajectories, top_p=0.3)

    config = PPOConfig(
        n_steps=32,
        n_epochs=2,
        batch_size=8,
        learning_rate=3e-4,
        device="cpu",
    )
    trainer = RefineTrainer(
        env=base,
        target_policy=target_policy,
        critical_buffer=buffer,
        p=0.5,
        lambda_rnd=0.01,
        ppo_config=config,
        device="cpu",
    )
    stats = trainer.learn(total_timesteps=64, log_interval=1)
    assert "mean_reward" in stats
    assert stats["total_timesteps"] >= 64


def test_refine_policy_factory():
    base = DummyEnv(obs_dim=3, act_dim=1, max_steps=8)
    target_policy = _make_target_policy(3, 1)
    trajectories = _make_trajectories(n=2, length=8, obs_dim=3)
    buffer = build_critical_buffer_from_trajectories(trajectories, top_p=0.5)

    config = PPOConfig(
        n_steps=16,
        n_epochs=1,
        batch_size=8,
        learning_rate=3e-4,
        device="cpu",
    )
    trainer = refine_policy(
        env=base,
        target_policy=target_policy,
        critical_buffer=buffer,
        total_timesteps=32,
        p=0.5,
        lambda_rnd=0.0,
        ppo_config=config,
        device="cpu",
    )
    assert isinstance(trainer, RefineTrainer)
    assert trainer.refined_policy is not None


def test_refine_trainer_save_load(tmp_path: Path):
    base = DummyEnv(obs_dim=4, act_dim=2, max_steps=10)
    target_policy = _make_target_policy(4, 2)
    trajectories = _make_trajectories(n=2, length=10, obs_dim=4)
    buffer = build_critical_buffer_from_trajectories(trajectories, top_p=0.5)

    config = PPOConfig(n_steps=16, n_epochs=1, batch_size=8, device="cpu")
    trainer = RefineTrainer(
        env=base,
        target_policy=target_policy,
        critical_buffer=buffer,
        p=0.5,
        lambda_rnd=0.01,
        ppo_config=config,
        device="cpu",
    )
    trainer.learn(total_timesteps=16, log_interval=1)

    save_dir = tmp_path / "refine_ckpt"
    trainer.save(str(save_dir))
    assert (save_dir / "policy.pt").exists()
    assert (save_dir / "rnd.pt").exists()

    # Loading should restore a functional policy.
    loaded_policy = TorchTargetPolicy.load(str(save_dir / "policy.pt"))
    obs = base.observation_space.sample()
    action, _ = loaded_policy.predict(obs, deterministic=True)
    assert action.shape == base.action_space.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
