"""Environment and state-restore smoke tests."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import ENV_SPECS, make_env  # noqa: E402


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except Exception:
        return False


class SelfishMiningTest(unittest.TestCase):
    def test_dynamics_and_restore(self):
        env = make_env("SelfishMining", seed=0)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        for _ in range(5):
            env.step(env.random_action())
        snapshot = env.get_state()
        obs_at_snapshot = env.current_obs().copy()
        actions = [env.random_action() for _ in range(5)]
        forward = [env.step(a)[0] for a in actions]

        # restoring must reproduce both the observation and the dynamics
        env.set_state(snapshot)
        np.testing.assert_allclose(env.current_obs(), obs_at_snapshot, atol=1e-7)
        replayed = [env.step(a)[0] for a in actions]
        np.testing.assert_allclose(np.asarray(replayed), np.asarray(forward), atol=1e-7)

    def test_reward_is_finite(self):
        env = make_env("SelfishMining", seed=1)
        env.reset(seed=1)
        rewards = [env.step(env.random_action())[1] for _ in range(50)]
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertTrue(all(r >= -10 for r in rewards))


@unittest.skipUnless(_mujoco_available(), "mujoco is not installed")
class MujocoTest(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        env = make_env("Hopper", seed=0)
        env.reset(seed=0)
        for _ in range(3):
            env.step(env.random_action())
        snapshot = env.get_state()
        obs_before = env.current_obs().copy()
        for _ in range(4):
            env.step(env.random_action())
        env.set_state(snapshot)
        np.testing.assert_allclose(obs_before, env.current_obs(), rtol=0, atol=1e-6)

    def test_sparse_reward(self):
        env = make_env("SparseHopper", seed=0)
        env.reset(seed=0)
        obs, reward, *_ = env.step(env.random_action())
        self.assertEqual(float(reward), 0.0)  # x < 0.6 at the beginning

    def test_registry_entries(self):
        for name in ("Hopper", "Walker2d", "Reacher", "HalfCheetah"):
            spec = ENV_SPECS[name]
            self.assertIsNotNone(spec.gym_id)


if __name__ == "__main__":
    unittest.main()
