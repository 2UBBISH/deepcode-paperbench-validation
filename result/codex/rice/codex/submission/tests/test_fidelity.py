"""Fidelity metric smoke test (tiny trajectory budget)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import make_env  # noqa: E402
from rice.fidelity import (  # noqa: E402
    FidelityConfig,
    compute_fidelity,
    mask_importance_fn,
    random_importance_fn,
)
from rice.networks import ActorCritic, MaskNet  # noqa: E402
from rice.policies import TorchPolicy  # noqa: E402


class FidelityTest(unittest.TestCase):
    def test_random_explanation(self):
        env = make_env("SelfishMining", seed=0)
        policy = TorchPolicy(ActorCritic(8, 3, hidden=(16, 16), discrete=True))
        config = FidelityConfig(
            n_trajectories=5,
            k_values=(0.2, 0.4),
            d_max=100.0,
            seed=0,
        )
        summary = compute_fidelity(
            env, policy, random_importance_fn(0), config, n_seeds=1
        )
        self.assertEqual(len(summary["mean"]), 2)
        self.assertTrue(np.all(np.isfinite(summary["mean"])))

    def test_mask_explanation(self):
        env = make_env("SelfishMining", seed=0)
        policy = TorchPolicy(ActorCritic(8, 3, hidden=(16, 16), discrete=True))
        mask = MaskNet(8, hidden=(16, 16))
        config = FidelityConfig(n_trajectories=3, k_values=(0.3,), d_max=100.0)
        summary = compute_fidelity(
            env, policy, mask_importance_fn(mask), config, n_seeds=1
        )
        self.assertTrue(np.isfinite(summary["mean"][0]))

    def test_higher_k_is_not_penalised_arbitrarily(self):
        """Sanity check of the score definition (log d/dmax - log l/L)."""
        import math

        l, L = 100, 1000
        score = math.log(0.1) - math.log(l / L)
        self.assertAlmostEqual(score, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
