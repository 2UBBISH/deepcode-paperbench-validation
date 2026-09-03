"""Unit tests for the RND exploration bonus module.

Tests verify that:
- The RND bonus is positive for novel states.
- The bonus decreases for visited states after predictor updates.
- Running normalization keeps bonuses stable.
- The RND module can be saved and loaded.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pytest
import torch

from rice.agents.rnd_network import RNDModule, RNDRewardWrapper, make_rnd_module


class DummyBoxEnv(gym.Env):
    """Minimal Box-observation environment for RND tests."""

    def __init__(self, obs_dim: int = 4) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._state = np.zeros(obs_dim, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = rng.uniform(-1, 1, size=(self.obs_dim,)).astype(np.float32)
        else:
            self._state = np.zeros(self.obs_dim, dtype=np.float32)
        return self._state, {}

    def step(self, action: Any):
        self._state += np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        self._state = np.clip(self._state, -1.0, 1.0)
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        return self._state, reward, terminated, truncated, info


@pytest.fixture(scope="module")
def rnd_module() -> RNDModule:
    """Module-scoped RND module for reuse across tests."""
    obs_space = gym.spaces.Box(
        low=-10.0, high=10.0, shape=(8,), dtype=np.float32
    )
    return make_rnd_module(
        observation_space=obs_space,
        output_dim=32,
        hidden_sizes=(64, 64),
        normalize_inputs=True,
        device="cpu",
    )


def test_rnd_bonus_positive_for_novel_states(rnd_module: RNDModule) -> None:
    """RND bonus should be positive for states never seen by the predictor."""
    obs = np.random.randn(8).astype(np.float32)
    bonus = rnd_module.compute_bonus(obs)
    assert bonus > 0.0, f"Expected positive RND bonus for novel state, got {bonus}"


def test_rnd_bonus_decreases_after_training(rnd_module: RNDModule) -> None:
    """After fitting the predictor on a state, its bonus should decrease."""
    obs = np.random.randn(8).astype(np.float32)

    initial_bonus = rnd_module.compute_bonus(obs)

    # Fit predictor on the same state many times.
    optimizer = torch.optim.Adam(rnd_module.predictor.parameters(), lr=1e-3)
    obs_tensor = rnd_module.normalize_obs(
        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    )
    for _ in range(200):
        optimizer.zero_grad()
        loss = rnd_module.predictor_loss(obs_tensor)
        loss.backward()
        optimizer.step()

    final_bonus = rnd_module.compute_bonus(obs)
    assert final_bonus < initial_bonus, (
        f"Expected RND bonus to decrease after training, "
        f"initial={initial_bonus:.4f}, final={final_bonus:.4f}"
    )


def test_rnd_bonus_remains_positive_for_unseen_states(rnd_module: RNDModule) -> None:
    """A state far from the training distribution should still yield a positive bonus."""
    train_obs = np.random.randn(8).astype(np.float32)

    optimizer = torch.optim.Adam(rnd_module.predictor.parameters(), lr=1e-3)
    obs_tensor = rnd_module.normalize_obs(
        torch.as_tensor(train_obs, dtype=torch.float32).unsqueeze(0)
    )
    for _ in range(200):
        optimizer.zero_grad()
        loss = rnd_module.predictor_loss(obs_tensor)
        loss.backward()
        optimizer.step()

    novel_obs = train_obs + 5.0 * np.ones(8, dtype=np.float32)
    bonus = rnd_module.compute_bonus(novel_obs)
    assert bonus > 0.0, (
        f"Expected positive RND bonus for out-of-distribution state, got {bonus}"
    )


def test_rnd_observation_normalization() -> None:
    """RND should update running observation statistics and normalize inputs."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    rnd = make_rnd_module(obs_space, output_dim=16, normalize_inputs=True, device="cpu")

    obs1 = np.ones(4, dtype=np.float32)
    obs2 = -np.ones(4, dtype=np.float32)

    rnd.update_obs_stats(obs1)
    rnd.update_obs_stats(obs2)

    assert rnd.obs_mean is not None
    assert rnd.obs_var is not None
    np.testing.assert_allclose(rnd.obs_mean, np.zeros(4), atol=1e-6)
    np.testing.assert_allclose(rnd.obs_var, np.ones(4), atol=1e-6)


def test_rnd_bonus_normalization() -> None:
    """RND should normalize bonuses using running statistics."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    rnd = make_rnd_module(obs_space, output_dim=16, normalize_inputs=False, device="cpu")

    bonuses = []
    for _ in range(50):
        obs = np.random.randn(4).astype(np.float32)
        b = rnd.compute_bonus(obs, normalize=False)
        bonuses.append(b)
        rnd.update_bonus_stats(b)

    normalized = [rnd.normalize_bonus(b) for b in bonuses]
    # Normalized bonuses should have smaller variance than raw bonuses.
    assert np.std(normalized) < np.std(bonuses) * 1.5


def test_rnd_reward_wrapper() -> None:
    """RNDRewardWrapper should augment environment rewards with the RND bonus."""
    env = DummyBoxEnv(obs_dim=4)
    rnd = make_rnd_module(
        env.observation_space, output_dim=16, normalize_inputs=True, device="cpu"
    )
    wrapper = RNDRewardWrapper(rnd_module=rnd, lambda_coef=0.1, device="cpu")

    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    augmented = wrapper.augment_reward(obs, next_obs, reward, info)
    assert augmented > reward, (
        f"Expected augmented reward > original reward, got {augmented} vs {reward}"
    )
    assert "rnd_bonus" in info
    assert info["rnd_bonus"] > 0.0
    assert "original_reward" in info
    assert info["original_reward"] == reward


def test_rnd_save_load(tmp_path: Path) -> None:
    """RND module should be saveable and loadable with consistent bonuses."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    rnd = make_rnd_module(obs_space, output_dim=16, normalize_inputs=True, device="cpu")

    obs = np.random.randn(4).astype(np.float32)
    bonus_before = rnd.compute_bonus(obs)

    save_path = tmp_path / "rnd.pt"
    rnd.save(save_path)

    loaded = RNDModule(obs_space, output_dim=16, normalize_inputs=True, device="cpu")
    loaded.load(save_path)

    bonus_after = loaded.compute_bonus(obs)
    np.testing.assert_allclose(bonus_before, bonus_after, rtol=1e-5)


def test_rnd_module_config_dict(rnd_module: RNDModule) -> None:
    """RND module should expose a configuration dictionary."""
    cfg = rnd_module.config_dict()
    expected_keys = {"output_dim", "hidden_sizes", "activation", "normalize_inputs"}
    assert expected_keys.issubset(cfg.keys())
    assert cfg["output_dim"] == 32
    assert cfg["hidden_sizes"] == [64, 64]


def test_rnd_predictor_loss_decreases() -> None:
    """Predictor loss should decrease when trained on a fixed observation."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    rnd = make_rnd_module(obs_space, output_dim=16, normalize_inputs=False, device="cpu")

    obs = np.random.randn(4).astype(np.float32)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = rnd.predictor_loss(obs_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Expected predictor loss to decrease, initial={losses[0]:.4f}, "
        f"final={losses[-1]:.4f}"
    )
