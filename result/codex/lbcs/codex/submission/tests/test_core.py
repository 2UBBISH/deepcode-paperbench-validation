"""Tests of the RCS objectives, the mask bookkeeping and Algorithm 1."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from lbcs.data import DatasetBundle, inject_symmetric_noise, make_class_imbalanced
from lbcs.inner_loop import InnerLoopConfig, InnerLoopTrainer
from lbcs.lbcs import (LBCS, LBCSConfig, TargetTrainingConfig,
                       evaluate_coreset, mask_to_continuous, random_mask)
from lbcs.lexiflow import LexiFlowConfig
from lbcs.models import ConvNet, build_model, count_parameters
from lbcs.objectives import BilevelObjective, ObjectiveConfig
from lbcs.utils import discretize


def _tiny_bundle(n=40, num_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DatasetBundle("tiny", x, y, x.clone(), y.clone(), num_classes)


class TestDiscretisation(unittest.TestCase):
    def test_matches_appendix_a(self):
        m = torch.tensor([-1.0, -0.5, -1e-6, 0.0, 0.5, 1.0])
        self.assertEqual(discretize(m).tolist(), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def test_continuous_mask_of_random_mask(self):
        mask = random_mask(50, 10, seed=0)
        self.assertEqual(int(mask.sum().item()), 10)
        continuous = mask_to_continuous(mask)
        self.assertTrue(torch.equal(discretize(continuous), mask))


class TestObjectives(unittest.TestCase):
    def setUp(self):
        self.bundle = _tiny_bundle()
        self.trainer = InnerLoopTrainer(
            lambda: ConvNet(output_dim=3, base_hid=4),
            InnerLoopConfig(epochs=1, batch_size=8, optimizer="sgd", lr=0.01,
                            warm_start=False),
            torch.device("cpu"))
        self.objective = BilevelObjective(self.bundle, self.trainer,
                                          torch.device("cpu"),
                                          ObjectiveConfig(eval_batch_size=16))

    def test_f2_is_the_coreset_size(self):
        mask = random_mask(self.bundle.n, 7, seed=0)
        value = self.objective.evaluate(mask_to_continuous(mask))
        self.assertEqual(float(value[1]), 7.0)

    def test_repeated_queries_are_cached(self):
        mask = mask_to_continuous(random_mask(self.bundle.n, 7, seed=0))
        first = self.objective.evaluate(mask)
        n_trainings = self.objective.num_trainings
        second = self.objective.evaluate(mask)
        self.assertTrue(torch.allclose(first, second))
        self.assertEqual(self.objective.num_trainings, n_trainings)

    def test_grouping_shares_one_mask_entry(self):
        grouped = BilevelObjective(self.bundle, self.trainer,
                                   torch.device("cpu"),
                                   ObjectiveConfig(eval_batch_size=16),
                                   group_size=4)
        mask = torch.ones(self.bundle.n)
        out = grouped.group_indices(mask)
        self.assertEqual(out.numel(), self.bundle.n)


class TestLBCS(unittest.TestCase):
    def test_algorithm_1_shape_and_size(self):
        bundle = _tiny_bundle()
        cfg = LBCSConfig(
            k=10, epsilon=0.2, T=3, seed=0,
            inner=InnerLoopConfig(epochs=1, batch_size=8, optimizer="sgd",
                                  lr=0.01, warm_start=True),
            objective=ObjectiveConfig(eval_batch_size=16),
            lexiflow=LexiFlowConfig(max_steps=3, step_decay_patience=2, seed=0))
        runner = LBCS(bundle, lambda: ConvNet(output_dim=3, base_hid=4),
                      torch.device("cpu"), cfg)
        info = runner.select()
        self.assertEqual(info["init_mask"].sum().item(), 10)
        self.assertEqual(info["mask"].numel(), bundle.n)
        self.assertLessEqual(info["coreset_size"], 10)
        self.assertIn("f1", info)

    def test_moderate_initialisation(self):
        bundle = _tiny_bundle()
        init = torch.arange(12)
        cfg = LBCSConfig(k=10, T=1, init_indices=init, init_label="moderate",
                         inner=InnerLoopConfig(epochs=1, batch_size=8,
                                               optimizer="sgd", lr=0.01),
                         lexiflow=LexiFlowConfig(max_steps=1,
                                                 step_decay_patience=1))
        runner = LBCS(bundle, lambda: ConvNet(output_dim=3, base_hid=4),
                      torch.device("cpu"), cfg)
        mask = runner.initial_mask()
        self.assertEqual(int(mask[init].sum().item()), 12)


class TestData(unittest.TestCase):
    def test_symmetric_noise_rate(self):
        y = torch.arange(1000) % 10
        noisy = inject_symmetric_noise(y, 0.3, 10, seed=0)
        changed = (noisy != y).sum().item()
        self.assertAlmostEqual(changed / len(y), 0.3, delta=0.02)

    def test_class_imbalance_ratio(self):
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1000, 1, 4, 4, generator=g)
        y = torch.arange(1000) % 10
        xb, yb = make_class_imbalanced(x, y, 10, 0.01, seed=0)
        counts = torch.bincount(yb, minlength=10).float()
        ratio = float(counts.max() / counts.min())
        self.assertGreater(ratio, 5.0)
        self.assertLess(xb.size(0), 1000)


class TestModels(unittest.TestCase):

    def test_evaluate_coreset_uses_the_discrete_mask(self):
        """A ``{0, 1}`` mask must select exactly the ones it contains."""
        bundle = _tiny_bundle(n=40, num_classes=3)
        mask = torch.zeros(bundle.n)
        mask[:7] = 1.0
        out = evaluate_coreset(
            bundle, mask, lambda: ConvNet(output_dim=3, base_hid=4),
            torch.device("cpu"),
            TargetTrainingConfig(epochs=1, batch_size=8, optimizer="sgd",
                                 lr=0.01))
        self.assertEqual(out["coreset_size"], 7)

    def test_forward_shapes(self):
        cases = [
            ("convnet", (2, 1, 28, 28), {}),
            ("lenet", (2, 1, 28, 28), {}),
            ("svhn_cnn_inner", (2, 3, 32, 32), {}),
            ("svhn_cnn_target", (2, 3, 32, 32), {}),
            ("cifar10_cnn_inner", (2, 3, 32, 32), {}),
            ("resnet18", (2, 3, 32, 32), {}),
            ("wideresnet", (2, 3, 32, 32), {"depth": 16, "widen_factor": 2}),
            ("vit", (2, 3, 32, 32), {"depth": 2, "num_heads": 2,
                                    "embed_dim": 32}),
        ]
        for name, shape, kwargs in cases:
            with self.subTest(model=name):
                model = build_model(name, **kwargs)
                out = model(torch.rand(*shape))
                self.assertEqual(out.shape, (2, 10))
                self.assertGreater(count_parameters(model), 0)


if __name__ == "__main__":
    unittest.main()
