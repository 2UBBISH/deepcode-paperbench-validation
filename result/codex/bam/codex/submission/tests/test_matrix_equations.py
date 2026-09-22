"""Appendix B: the quadratic matrix equation ``X U X + X = V``.

* Lemma B.1: closed-form solution ``X = 2 V [I + (I + 4 U V)^{1/2}]^{-1}``;
* Lemma B.2: the solution is symmetric and positive definite, also when ``U`` is
  singular;
* Lemma B.3: the low-rank solver agrees with the general one for ``U = Q Q^T``;
* Lemma B.4 (monotonicity): ``T >= U >= 0`` with ``X T X + X = Y U Y + Y = V``
  implies ``X <= Y``.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.linalg import (  # noqa: E402
    solve_quadratic_matrix_eq,
    solve_quadratic_matrix_eq_low_rank,
    symmetrize,
)


def random_psd(rng, D, K):
    Q = rng.normal(size=(D, K))
    return Q @ Q.T, Q


def random_pd(rng, D):
    A = rng.normal(size=(D, D))
    return symmetrize(A @ A.T) + 0.1 * np.eye(D)


class TestQuadraticMatrixEquation(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.D = 8
        self.V = random_pd(self.rng, self.D)

    def test_solution_satisfies_equation_and_is_pd(self):
        for K in (1, 3, 8, 20):  # includes U of full rank and of rank 1
            U, _ = random_psd(self.rng, self.D, K)
            X = solve_quadratic_matrix_eq(U, self.V)
            self.assertLess(np.abs(X @ U @ X + X - self.V).max(), 1e-8)
            self.assertLess(np.abs(X - X.T).max(), 1e-12)
            self.assertGreater(np.linalg.eigvalsh(symmetrize(X)).min(), 0.0)

    def test_low_rank_solver_matches_general_solver(self):
        for K in (1, 2, 4):
            U, Q = random_psd(self.rng, self.D, K)
            X_general = solve_quadratic_matrix_eq(U, self.V)
            X_low_rank = solve_quadratic_matrix_eq_low_rank(Q, self.V)
            self.assertLess(np.abs(X_general - X_low_rank).max(), 1e-8)

    def test_monotonicity_lemma(self):
        """Lemma B.4: a smaller ``T`` (i.e. ``T >= U``) gives a smaller solution."""
        Q_small = self.rng.normal(size=(self.D, 2)) * 0.5
        Q_large = np.concatenate([Q_small, self.rng.normal(size=(self.D, 2))], axis=1)
        U = Q_small @ Q_small.T          # U (smaller matrix)
        T = Q_large @ Q_large.T          # T >= U
        X = solve_quadratic_matrix_eq(T, self.V)   # solution for T
        Y = solve_quadratic_matrix_eq(U, self.V)   # solution for U
        self.assertGreaterEqual(np.linalg.eigvalsh(symmetrize(Y - X)).min(), -1e-10)

    def test_small_U_limit(self):
        """With ``U -> 0`` the solution tends to ``V`` (as in the limit ``lambda -> 0``)."""
        U = 1e-10 * np.eye(self.D)
        X = solve_quadratic_matrix_eq(U, self.V)
        self.assertLess(np.abs(X - self.V).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
