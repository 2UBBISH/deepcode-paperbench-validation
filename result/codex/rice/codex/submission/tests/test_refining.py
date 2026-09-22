"""Algorithm 2 and the baselines: end-to-end smoke tests (tiny budgets)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import make_env  # noqa: E402
from rice.explanation.mask_trainer import MaskTrainingConfig, train_mask_network  # noqa: E402
from rice.networks import ActorCritic  # noqa: E402
from rice.policies import TorchPolicy  # noqa: E402
from rice.ppo_core import PPOConfig  # noqa: E402
from rice.refining.methods import (  # noqa: E402
    refine_jsrl,
    refine_ppo_finetune,
    refine_rice,
    refine_statemask_r,
)
from rice.refining.trainer import RefineConfig  # noqa: E402


class RefiningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        cls.env_name = "SelfishMining"
        cls.policy = ActorCritic(8, 3, hidden=(16, 16), discrete=True)
        env = make_env(cls.env_name, seed=0)
        mask_config = MaskTrainingConfig(
            iterations=5,
            rollout_length=100,
            max_samples=500,
            seed=0,
            ppo=PPOConfig(learning_rate=1e-3, n_epochs=1, batch_size=32),
        )
        cls.mask = train_mask_network(
            env, TorchPolicy(cls.policy), mask_config, verbose=False
        ).mask_net

    def refine_config(self, **kwargs) -> RefineConfig:
        config = RefineConfig(
            total_steps=1200,
            n_steps=200,
            seed=0,
            eval_interval=600,
            eval_episodes=2,
            p=0.5,
            rnd_lambda=0.01,
            rollin_length=32,
            ppo=PPOConfig(learning_rate=1e-3, n_epochs=2, batch_size=32),
            jsrl_max_rollin=20,
        )
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config

    def test_rice(self):
        env = make_env(self.env_name, seed=0)
        eval_env = make_env(self.env_name, seed=1)
        result = refine_rice(
            env,
            ActorCritic(8, 3, hidden=(16, 16), discrete=True),
            self.mask,
            config=self.refine_config(),
            eval_env=eval_env,
            verbose=False,
        )
        self.assertGreater(result.total_steps, 0)
        self.assertTrue(np.isfinite(result.final_eval))
        self.assertGreater(result.provider_stats["critical_resets"], 0)

    def test_ppo_finetune(self):
        env = make_env(self.env_name, seed=0)
        result = refine_ppo_finetune(
            env,
            ActorCritic(8, 3, hidden=(16, 16), discrete=True),
            config=self.refine_config(),
            eval_env=make_env(self.env_name, seed=1),
            verbose=False,
        )
        self.assertEqual(result.method, "ppo_finetune")
        self.assertTrue(np.isfinite(result.final_eval))

    def test_statemask_r(self):
        env = make_env(self.env_name, seed=0)
        result = refine_statemask_r(
            env,
            ActorCritic(8, 3, hidden=(16, 16), discrete=True),
            self.mask,
            config=self.refine_config(),
            eval_env=make_env(self.env_name, seed=1),
            verbose=False,
        )
        self.assertEqual(result.provider_stats["default_resets"], 0)

    def test_jsrl(self):
        env = make_env(self.env_name, seed=0)
        result = refine_jsrl(
            env,
            ActorCritic(8, 3, hidden=(16, 16), discrete=True),
            config=self.refine_config(),
            eval_env=make_env(self.env_name, seed=1),
            verbose=False,
        )
        self.assertEqual(result.method, "jsrl")

    def test_rnd_decays_bonus(self):
        from rice.refining.rnd import RND

        rnd = RND(8, hidden=(8, 8), device="cpu")
        obs = np.zeros((1, 8), dtype=np.float32)
        first = float(rnd.error(obs)[0])
        for _ in range(50):
            rnd.update(np.random.randn(16, 8).astype(np.float32))
        last = float(rnd.error(np.random.randn(1, 8).astype(np.float32))[0])
        self.assertTrue(np.isfinite(first) and np.isfinite(last))


if __name__ == "__main__":
    unittest.main()
