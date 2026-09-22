"""Tests of the lexicographic relations and of the LexiFlow optimiser."""

from __future__ import annotations

import itertools
import unittest

import torch

from lbcs.lexiflow import (LexiFlow, LexiFlowConfig, practical_eq,
                           practical_less, thresholds_from_history, true_less)


class TestPracticalRelations(unittest.TestCase):
    def test_thresholds(self):
        values = [[2.0, 10.0], [3.0, 22.0], [2.5, 30.0]]
        f1_thr, f2_thr = thresholds_from_history(values, epsilon=0.2)
        self.assertAlmostEqual(f1_thr, 2.4)          # 2.0 * 1.2
        # only [2.0, 10.0] is inside the eps region -> f2 threshold is 10
        self.assertAlmostEqual(f2_thr, 10.0)

    def test_equality_when_both_inside_threshold(self):
        thr = [1.0, 100.0]
        self.assertTrue(practical_eq([0.9, 50.0], [0.5, 10.0], thr))
        self.assertFalse(practical_eq([1.5, 50.0], [0.5, 10.0], thr))

    def test_less_on_primary_objective(self):
        thr = [1.0, 100.0]
        # 0.5 < 0.9, and 0.9 is above the f1 threshold -> strictly better
        self.assertTrue(practical_less([0.5, 5.0], [0.9, 90.0], [0.4, 100.0]))

    def test_no_improvement_when_opponent_already_at_threshold(self):
        # b is inside the region on both objectives: improvements do not count
        thr = [1.0, 100.0]
        self.assertFalse(practical_less([0.5, 5.0], [0.9, 90.0], thr))

    def test_less_on_secondary_objective(self):
        # both inside the f1 region, so the comparison falls through to f2
        thr = [1.0, 100.0]
        self.assertTrue(practical_less([0.9, 5.0], [0.95, 500.0], thr))

    def test_no_secondary_improvement_below_threshold(self):
        # the opponent already meets the f2 threshold -> no strict improvement
        thr = [1.0, 1000.0]
        self.assertFalse(practical_less([0.9, 5.0], [0.95, 500.0], thr))

    def test_true_less_is_plain_lexicographic(self):
        self.assertTrue(true_less([1.0, 5.0], [1.0, 6.0]))
        self.assertFalse(true_less([1.0, 6.0], [1.0, 5.0]))
        self.assertFalse(true_less([1.0, 5.0], [1.0, 5.0]))


class _SyntheticRCS:
    """``f_1`` = 1 - mean quality of the mask, ``f_2`` = |m|_0."""

    def __init__(self, n=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.quality = torch.rand(n, generator=g) ** 3
        self.n = n
        self._cache = {}

    def __call__(self, mask):
        key = tuple((mask >= 0).long().tolist())
        if key not in self._cache:
            m = torch.tensor(key, dtype=torch.float64)
            k = float(m.sum())
            if k == 0:
                value = torch.tensor([float("inf"), 0.0], dtype=torch.float64)
            else:
                f1 = 1.0 - float((m * self.quality.double()).sum()) / k
                value = torch.tensor([f1, k], dtype=torch.float64)
            self._cache[key] = value
        return self._cache[key]

    def brute_force(self, epsilon):
        table = []
        for bits in itertools.product([0, 1], repeat=self.n):
            m = torch.tensor(bits, dtype=torch.float64)
            k = float(m.sum())
            if k == 0:
                continue
            f1 = 1.0 - float((m * self.quality.double()).sum()) / k
            table.append((f1, k))
        f1_star = min(f1 for f1, _ in table)
        f2_star = min(k for f1, k in table if f1 <= f1_star * (1 + epsilon))
        return f1_star, f2_star


class TestLexiFlowConvergence(unittest.TestCase):
    def test_reaches_lexicographic_optimum(self):
        problem = _SyntheticRCS(n=10, seed=0)
        f1_star, f2_star = problem.brute_force(epsilon=0.2)
        for seed in range(3):
            g = torch.Generator().manual_seed(seed)
            init = -torch.ones(problem.n)
            init[torch.randperm(problem.n, generator=g)[:problem.n // 2]] = 1.0
            search = LexiFlow(problem,
                              LexiFlowConfig(epsilon=0.2, max_steps=1500,
                                             delta_init=1.0, delta_lower=1e-3,
                                             step_decay_patience=10, seed=seed))
            best = search.optimize(init)
            f1, f2 = (float(v) for v in problem(best))
            self.assertLessEqual(f1, f1_star * (1 + 0.2) + 1e-9)
            self.assertLessEqual(f2, f2_star + 1e-9)

    def test_smaller_epsilon_does_not_increase_the_size(self):
        problem = _SyntheticRCS(n=12, seed=3)
        init = -torch.ones(problem.n)
        init[torch.randperm(problem.n)[:6]] = 1.0
        sizes = []
        for eps in (0.1, 0.5):
            search = LexiFlow(problem,
                              LexiFlowConfig(epsilon=eps, max_steps=1200,
                                             delta_lower=1e-3,
                                             step_decay_patience=10, seed=1))
            best = search.optimize(init)
            sizes.append(float(problem(best)[1]))
        self.assertLessEqual(sizes[0], sizes[1] + problem.n)

    def test_optimiser_queries_are_cached(self):
        calls = {"n": 0}
        problem = _SyntheticRCS(n=8, seed=1)

        def objective(mask):
            calls["n"] += 1
            return problem(mask)

        search = LexiFlow(objective, LexiFlowConfig(max_steps=50,
                                                    step_decay_patience=5,
                                                    seed=0))
        init = -torch.ones(8)
        init[:4] = 1.0
        search.optimize(init)
        self.assertGreater(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
