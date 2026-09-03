"""Smoke tests for the RICE package.

These tests verify that core classes can be instantiated and that the main
pipelines run for a small number of steps. They are intended for quick local
validation, not for exhaustive correctness checks.
"""
import unittest

import gymnasium as gym
import numpy as np
import torch

from rice.baselines import ppo_finetune
from rice.env_utils import make_env
from rice.explanations import MaskExplanation, RandomExplanation
from rice.fidelity import sample_trajectory
from rice.mask_network import MaskNetwork, MaskNetworkTrainer
from rice.refining import RICERefiningEnv, refine_rice
from rice.rnd import RNDBonus


class TestMaskNetwork(unittest.TestCase):
    def test_forward(self) -> None:
        net = MaskNetwork(obs_dim=4)
        obs = torch.randn(8, 4)
        logits, value = net(obs)
        self.assertEqual(logits.shape, (8,))
        self.assertEqual(value.shape, (8,))


class TestRND(unittest.TestCase):
    def test_bonus(self) -> None:
        rnd = RNDBonus(obs_dim=4)
        obs = np.random.randn(10, 4).astype(np.float32)
        bonus, loss = rnd.compute_and_update(obs)
        self.assertEqual(bonus.shape, (10,))


class TestExplanations(unittest.TestCase):
    def test_random_explanation(self) -> None:
        expl = RandomExplanation(seed=0)
        traj = np.random.randn(20, 4).astype(np.float32)
        scores = expl.explain(traj)
        self.assertEqual(scores.shape, (20,))


class TestRefiningEnv(unittest.TestCase):
    def test_env_runs(self) -> None:
        env = make_env("Hopper-v3", seed=0)
        policy = object()  # placeholder; we won't actually step far
        mask_net = MaskNetwork(obs_dim=env.observation_space.shape[0])
        rice_env = RICERefiningEnv(
            env,
            policy=policy,
            mask_net=mask_net,
            p=0.0,  # always default reset to avoid policy calls
        )
        obs, _ = rice_env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = rice_env.step(action)
        self.assertIn("rnd_bonus", info)


class TestSmokeIntegration(unittest.TestCase):
    def test_mask_trainer_one_iter(self) -> None:
        env = make_env("Hopper-v3", seed=0)
        from stable_baselines3 import PPO

        policy = PPO("MlpPolicy", env, n_steps=64, batch_size=64, verbose=0, seed=0)
        policy.learn(total_timesteps=128)
        trainer = MaskNetworkTrainer(
            env=env,
            target_policy=policy,
            obs_dim=env.observation_space.shape[0],
            alpha=0.0001,
        )
        logs = trainer.train(total_timesteps=64, steps_per_iter=64)
        self.assertGreater(len(logs), 0)


if __name__ == "__main__":
    unittest.main()
