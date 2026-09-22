"""Algorithm 1 / StateMask smoke tests on the (fast) selfish mining task."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import make_env  # noqa: E402
from rice.explanation.critical_states import (  # noqa: E402
    importance_scores,
    most_critical_state,
    most_critical_window,
    topk_critical_states,
)
from rice.explanation.mask_trainer import (  # noqa: E402
    MaskTrainingConfig,
    train_mask_network,
)
from rice.explanation.state_mask import (  # noqa: E402
    StateMaskTrainingConfig,
    train_state_mask,
)
from rice.networks import ActorCritic, MaskNet  # noqa: E402
from rice.policies import TorchPolicy  # noqa: E402
from rice.ppo_core import PPOConfig  # noqa: E402


def tiny_config(cls=MaskTrainingConfig) -> MaskTrainingConfig:
    return cls(
        iterations=10,
        rollout_length=200,
        alpha=0.01,
        max_samples=2000,
        seed=0,
        log_interval=100,
        ppo=PPOConfig(learning_rate=1e-3, n_epochs=2, batch_size=64),
    )


class MaskTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.env = make_env("SelfishMining", seed=0)
        net = ActorCritic(8, 3, hidden=(16, 16), discrete=True)
        self.policy = TorchPolicy(net)

    def test_mask_training_runs(self):
        result = train_mask_network(
            self.env, self.policy, tiny_config(), verbose=False
        )
        self.assertEqual(result.samples, 2000)
        self.assertGreater(result.wall_time, 0.0)
        scores = importance_scores(result.mask_net, np.zeros((4, 8), dtype=np.float32))
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))
        self.assertTrue(np.isfinite(result.history[-1]["policy_loss"]))

    def test_statemask_baseline_runs(self):
        result = train_state_mask(
            self.env, self.policy, tiny_config(StateMaskTrainingConfig), verbose=False
        )
        self.assertGreater(result.samples, 0)
        self.assertIn("dual", result.history[-1])
        self.assertGreaterEqual(result.history[-1]["dual"], 0.0)

    def test_critical_state_helpers(self):
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        self.assertEqual(most_critical_state(scores), 1)
        np.testing.assert_array_equal(topk_critical_states(scores, 2), [1, 3])
        start, end = most_critical_window(scores, 2)
        self.assertEqual((start, end), (1, 3))

    def test_mask_net_probabilities(self):
        net = MaskNet(8, hidden=(8,))
        obs = np.zeros((5, 8), dtype=np.float32)
        np.testing.assert_allclose(
            net.importance(obs) + net.mask_prob(obs), np.ones(5), atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
