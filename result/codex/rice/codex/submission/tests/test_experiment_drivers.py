"""The experiment drivers of Sections 4.2 (tiny budgets, selfish mining)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import ENV_SPECS  # noqa: E402
from rice.experiments import common as exp_common  # noqa: E402
from rice.experiments.explanation import run_experiment_i  # noqa: E402
from rice.experiments.refining import run_experiment_ii_and_iii  # noqa: E402
from rice.networks import ActorCritic  # noqa: E402


class DriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tmpdir = tempfile.mkdtemp(prefix="rice-test-")
        exp_common.DEFAULT_ROOT = cls.tmpdir
        # shrink every budget of the application
        spec = ENV_SPECS["SelfishMining"]
        cls.original = (spec.mask_samples, spec.mask_iterations, spec.refine_steps)
        spec.mask_samples = 600
        spec.mask_iterations = 3
        spec.refine_steps = 800
        # a (weak) pre-trained target agent
        cls.agent_path = os.path.join(cls.tmpdir, "agent.pt")
        ActorCritic(8, 3, hidden=(16, 16), discrete=True).save(cls.agent_path)

    @classmethod
    def tearDownClass(cls):
        spec = ENV_SPECS["SelfishMining"]
        spec.mask_samples, spec.mask_iterations, spec.refine_steps = cls.original

    def test_experiment_i_train_and_measure(self):
        result = run_experiment_i(
            "SelfishMining",
            self.agent_path,
            n_trajectories=2,
            k_values=(0.2,),
            n_seeds=1,
            seed=0,
            train_if_missing=True,
            verbose=False,
        )
        # fidelity was computed for the two explanation methods + random
        self.assertIn("ours", result.results)
        self.assertIn("statemask", result.results)
        self.assertIn("random", result.results)
        # both mask networks were trained with the same sample budget -> the
        # efficiency table of the paper can be filled in
        self.assertIn("ours", result.efficiency)
        self.assertIn("statemask", result.efficiency)
        self.assertEqual(result.efficiency["ours"]["samples"], 600)
        self.assertIn("time_reduction_percent", result.efficiency)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmpdir, "results", "experiment_i",
                             "SelfishMining_fidelity.json")
            )
        )

    def test_experiment_ii_and_iii(self):
        masks = {
            "ours": os.path.join(
                self.tmpdir, "checkpoints", "masks", "SelfishMining_ours_seed0.pt"
            ),
            "statemask": os.path.join(
                self.tmpdir, "checkpoints", "masks", "SelfishMining_statemask_seed0.pt"
            ),
        }
        if not all(os.path.exists(p) for p in masks.values()):
            # Experiment I also trains the mask networks used here
            run_experiment_i(
                "SelfishMining",
                self.agent_path,
                n_trajectories=1,
                k_values=(0.2,),
                n_seeds=1,
                verbose=False,
            )
        for path in masks.values():
            self.assertTrue(os.path.exists(path), "mask checkpoints missing")
        result = run_experiment_ii_and_iii(
            "SelfishMining",
            self.agent_path,
            masks,
            seeds=[0],
            methods=("ppo", "ours"),
            explanations=("random", "ours"),
            verbose=False,
        )
        self.assertIn("ppo", result.vary_refine)
        self.assertIn("ours", result.vary_refine)
        self.assertEqual(set(result.vary_explanation), {"random", "ours"})
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmpdir, "results", "experiment_ii_iii",
                             "SelfishMining_refining.json")
            )
        )


if __name__ == "__main__":
    unittest.main()
