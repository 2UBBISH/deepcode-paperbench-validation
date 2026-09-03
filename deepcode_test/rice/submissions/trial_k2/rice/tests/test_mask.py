"""Unit tests for the RICE MaskNet explanation module."""

import numpy as np
import pytest
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

from rice.agents import PPOConfig
from rice.agents.target_policy import MLPActorCritic, TorchTargetPolicy
from rice.masknet import (
    MaskIntrinsicReward,
    MaskNetwork,
    MaskTrainer,
    mask_reward,
)
from rice.masknet.mask_network import build_mask_network, match_target_mask_network
from rice.masknet.masked_env import MaskedEnv


class DummyEnv(gym.Env):
    """A tiny continuous-control environment for testing."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, max_steps: int = 10):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self._state = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        self._state = self.observation_space.sample()
        self._step_count = 0
        return self._state.astype(np.float32), {}

    def step(self, action):
        self._state = self.observation_space.sample()
        self._step_count += 1
        reward = float(np.linalg.norm(action))
        terminated = self._step_count >= self.max_steps
        truncated = False
        return (
            self._state.astype(np.float32),
            reward,
            terminated,
            truncated,
            {"step": self._step_count},
        )


def _make_target_policy(obs_dim: int, act_dim: int, discrete: bool = False):
    ac = MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_sizes=(32, 32),
        discrete=discrete,
    )
    return TorchTargetPolicy(
        model=ac,
        observation_space=gym.spaces.Box(-1, 1, shape=(obs_dim,), dtype=np.float32),
        action_space=(
            gym.spaces.Discrete(act_dim)
            if discrete
            else gym.spaces.Box(-1, 1, shape=(act_dim,), dtype=np.float32)
        ),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# MaskNetwork tests
# ---------------------------------------------------------------------------


def test_mask_network_output_range():
    obs_dim = 5
    net = MaskNetwork(obs_dim=obs_dim, hidden_sizes=(16, 16))
    obs = torch.randn(10, obs_dim)
    xi = net(obs)
    assert xi.shape == (10, 1)
    assert torch.all((xi > 0) & (xi < 1))


def test_mask_network_predict_single():
    obs_dim = 3
    net = MaskNetwork(obs_dim=obs_dim, hidden_sizes=(8,))
    obs = np.random.randn(obs_dim).astype(np.float32)
    xi = net.predict(obs)
    assert isinstance(xi, np.ndarray)
    assert xi.shape == (1,)
    assert 0.0 < xi.item() < 1.0


def test_build_mask_network_from_space():
    space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    net = build_mask_network(space, hidden_sizes=(12, 12))
    assert net.obs_dim == 7
    xi = net.predict(space.sample())
    assert 0.0 < xi.item() < 1.0


def test_match_target_mask_network():
    target = _make_target_policy(obs_dim=4, act_dim=2)
    mask_net = match_target_mask_network(target, hidden_sizes=(32, 32))
    assert mask_net.obs_dim == 4
    xi = mask_net.predict(np.random.randn(4).astype(np.float32))
    assert 0.0 < xi.item() < 1.0


# ---------------------------------------------------------------------------
# Intrinsic reward tests
# ---------------------------------------------------------------------------


def test_mask_reward_scalar():
    r = mask_reward(env_reward=1.0, xi=0.8, alpha=1e-4)
    expected = 1.0 + 1e-4 * (1.0 - 0.8)
    assert np.isclose(r, expected)


def test_mask_reward_array():
    env_r = np.array([1.0, 2.0, 3.0])
    xi = np.array([0.5, 0.9, 0.1])
    r = mask_reward(env_r, xi, alpha=0.01)
    expected = env_r + 0.01 * (1.0 - xi)
    assert np.allclose(r, expected)


def test_mask_intrinsic_reward_callable():
    bonus = MaskIntrinsicReward(alpha=0.1)
    r = bonus(env_reward=2.0, xi=0.7)
    assert np.isclose(r, 2.0 + 0.1 * 0.3)


# ---------------------------------------------------------------------------
# MaskedEnv tests
# ---------------------------------------------------------------------------


def test_masked_env_action_space_is_binary():
    env = DummyEnv()
    target = _make_target_policy(env.obs_dim, env.act_dim)
    mask_net = MaskNetwork(obs_dim=env.obs_dim, hidden_sizes=(8, 8))
    wrapped = MaskedEnv(env, target, mask_net, alpha=1e-4, device="cpu")
    assert isinstance(wrapped.action_space, gym.spaces.Discrete)
    assert wrapped.action_space.n == 2


def test_masked_env_step_target_action():
    env = DummyEnv(max_steps=5)
    target = _make_target_policy(env.obs_dim, env.act_dim)
    mask_net = MaskNetwork(obs_dim=env.obs_dim, hidden_sizes=(8, 8))
    wrapped = MaskedEnv(env, target, mask_net, alpha=1e-4, device="cpu")

    obs, info = wrapped.reset(seed=0)
    assert obs.shape == (env.obs_dim,)

    obs2, reward, terminated, truncated, info = wrapped.step(0)
    assert obs2.shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert "env_reward" in info
    assert "mask_reward" in info
    assert "xi" in info
    assert "mask_action" in info


def test_masked_env_step_random_action():
    env = DummyEnv(max_steps=5)
    target = _make_target_policy(env.obs_dim, env.act_dim)
    mask_net = MaskNetwork(obs_dim=env.obs_dim, hidden_sizes=(8, 8))
    wrapped = MaskedEnv(env, target, mask_net, alpha=1e-4, device="cpu")

    obs, _ = wrapped.reset(seed=1)
    obs2, reward, terminated, truncated, info = wrapped.step(1)
    assert obs2.shape == (env.obs_dim,)
    assert info["mask_action"] == 1


# ---------------------------------------------------------------------------
# MaskTrainer tests
# ---------------------------------------------------------------------------


def test_mask_trainer_train_short():
    env = DummyEnv(max_steps=5)
    target = _make_target_policy(env.obs_dim, env.act_dim)
    trainer = MaskTrainer(
        env=env,
        target_policy=target,
        alpha=1e-4,
        ppo_config=PPOConfig(
            n_steps=32,
            n_epochs=2,
            batch_size=8,
            learning_rate=3e-4,
            device="cpu",
        ),
        device="cpu",
    )
    stats = trainer.train(total_timesteps=64, log_interval=1)
    assert "total_timesteps" in stats
    assert stats["total_timesteps"] >= 64


def test_mask_trainer_collect_critical_states():
    env = DummyEnv(max_steps=5)
    target = _make_target_policy(env.obs_dim, env.act_dim)
    trainer = MaskTrainer(
        env=env,
        target_policy=target,
        alpha=1e-4,
        ppo_config=PPOConfig(
            n_steps=16,
            n_epochs=1,
            batch_size=8,
            learning_rate=3e-4,
            device="cpu",
        ),
        device="cpu",
    )
    # Train briefly so the mask network has non-trivial outputs.
    trainer.train(total_timesteps=64, log_interval=1)
    critical = trainer.collect_critical_states(
        n_trajectories=5, top_p=0.5, max_steps_per_episode=5
    )
    assert len(critical) > 0
    assert all("observation" in s for s in critical)
    assert all("xi" in s for s in critical)
    assert all(0.0 <= s["xi"] <= 1.0 for s in critical)


def test_mask_trainer_save_load(tmp_path):
    env = DummyEnv(max_steps=5)
    target = _make_target_policy(env.obs_dim, env.act_dim)
    trainer = MaskTrainer(
        env=env,
        target_policy=target,
        alpha=1e-4,
        ppo_config=PPOConfig(
            n_steps=16,
            n_epochs=1,
            batch_size=8,
            learning_rate=3e-4,
            device="cpu",
        ),
        device="cpu",
    )
    trainer.train(total_timesteps=32, log_interval=1)

    save_path = tmp_path / "mask.pt"
    trainer.save(str(save_path))
    assert save_path.exists()

    obs = np.random.randn(env.obs_dim).astype(np.float32)
    xi_before = trainer.mask_network.predict(obs).copy()

    trainer2 = MaskTrainer(
        env=env,
        target_policy=target,
        alpha=1e-4,
        ppo_config=PPOConfig(device="cpu"),
        device="cpu",
    )
    trainer2.load(str(save_path))
    xi_after = trainer2.mask_network.predict(obs)
    assert np.allclose(xi_before, xi_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
