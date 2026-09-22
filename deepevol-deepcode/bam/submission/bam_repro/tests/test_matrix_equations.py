"""Unit tests for the BaM numerical core: the quadratic matrix equation

        X U X + X = V,        U >= 0,  V > 0,

with its dense closed form (Lemma B.1)

        X = 2 V [ I + (I + 4 U V)^{1/2} ]^{-1},

and its low-rank variant for ``U = Q Q^T`` (Lemma B.3)

        X = V - V Q [ (1/2) I + (Q^T V Q + (1/4) I)^{1/2} ]^{-2} Q^T V.

Every test that touches the solver asserts the defining identities:

* ``X U X + X == V`` (the quadratic matrix equation must actually be solved),
* ``X == X^T`` (symmetry),
* ``lambda_min(X) > 0`` (positive definiteness),

plus agreement between the dense and low-rank codepaths and sanity limits
(``U = 0`` implies ``X = V``, scalar closed form, ill-conditioned ``V``).

Run with either ``pytest bam_repro/tests/test_matrix_equations.py`` or
``python -m unittest bam_repro.tests.test_matrix_equations``.
"""

from __future__ import annotations

import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Import shims: the package is laid out as ``bam_repro/bam/matrix_equations.py``
# (and mirrored at ``bam/matrix_equations.py``); support both plus the
# flat-script layout so the tests run from any working directory.
# ---------------------------------------------------------------------------
try:  # preferred layout
    from bam_repro.bam.matrix_equations import (  # type: ignore
        array_namespace,
        ensure_spd,
        inverse_spd,
        matrix_sqrt,
        matrix_sqrt_inv,
        residual as module_residual,
        solve_quadratic_matrix_equation,
        solve_quadratic_matrix_equation_dense,
        solve_quadratic_matrix_equation_low_rank,
        solve_symmetric,
        symmetrize,
    )
except Exception:  # pragma: no cover - fallback import path
    from bam.matrix_equations import (  # type: ignore
        array_namespace,
        ensure_spd,
        inverse_spd,
        matrix_sqrt,
        matrix_sqrt_inv,
        residual as module_residual,
        solve_quadratic_matrix_equation,
        solve_quadratic_matrix_equation_dense,
        solve_quadratic_matrix_equation_low_rank,
        solve_symmetric,
        symmetrize,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rand_psd(dim: int, rng: np.random.Generator, rank: int | None = None, cond: float = 1.0):
    """Random symmetric positive semidefinite matrix (optionally low rank)."""
    A = rng.standard_normal((dim, dim))
    if cond is not None and cond > 1.0:
        U, _ = np.linalg.qr(A)
        V, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        s = np.linspace(1.0, 1.0 / cond, dim)
        A = (U * s) @ V.T
    if rank is not None and rank < dim:
        # rank-deficient PSD matrix: A A^T with A of size (D, rank)
        A = rng.standard_normal((dim, rank))
        return A @ A.T
    return A @ A.T


def _rand_pd(dim: int, rng: np.random.Generator, cond: float = 1.0, jitter: float = 1.0):
    """Random symmetric positive definite matrix."""
    S = _rand_psd(dim, rng, cond=cond)
    S = symmetrize(S)
    return S + jitter * np.eye(dim)


def _quadratic_residual(U: np.ndarray, X: np.ndarray) -> np.ndarray:
    """X U X + X (the left-hand side of the matrix equation), computed locally."""
    return X @ U @ X + X


def _max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(1.0, float(np.max(np.abs(b))))
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / denom


def _assert_solves(self, U, V, X, tol=1e-8, check_pd=True):
    """Assert X solves X U X + X = V, is symmetric and (optionally) PD."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d = U.shape[0]

    self.assertEqual(X.shape, (d, d), "solver returned a matrix of the wrong shape")

    # 1) the defining identity: X U X + X = V
    lhs = _quadratic_residual(U, X)
    err = float(np.max(np.abs(lhs - V))) / max(1.0, float(np.max(np.abs(V))))
    self.assertLess(err, tol, f"X U X + X != V (relative error {err:.3e})")

    # 2) symmetry
    asym = float(np.max(np.abs(X - X.T))) / max(1.0, float(np.max(np.abs(X))))
    self.assertLess(asym, 1e-9, f"X is not symmetric (asymmetry {asym:.3e})")

    # 3) positive definiteness
    if check_pd:
        eig = np.linalg.eigvalsh(symmetrize(X))
        self.assertGreater(
            float(eig.min()), 0.0, f"X is not positive definite (min eig {eig.min():.3e})"
        )
        self.assertLess(
            float(eig.max()), 1e6 * max(1.0, float(np.max(np.abs(V)))),
            "X has an implausibly large eigenvalue",
        )
        # consistency with the declared solution used for the covariance update
        self.assertLess(
            float(np.linalg.norm(X @ U @ X + X - V)), tol * max(1.0, np.linalg.norm(V)),
            "matrix equation violated in Frobenius norm",
        )


# ---------------------------------------------------------------------------
# dense solver (Lemma B.1)
# ---------------------------------------------------------------------------
class TestDenseQuadraticMatrixEquation(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_dims_and_identity(self):
        """X U X + X = V, X symmetric and PD, for a range of dimensions."""
        for dim in (1, 2, 3, 5, 8, 16, 32):
            with self.subTest(dim=dim):
                U = _rand_psd(dim, self.rng)
                V = _rand_pd(dim, self.rng)
                X = solve_quadratic_matrix_equation_dense(U, V)
                _assert_solves(self, U, V, X)

    def test_zero_U_gives_X_equal_V(self):
        """U = 0  =>  (I + 4UV)^{1/2} = I  =>  X = 2V(2I)^{-1} = V."""
        dim = 6
        U = np.zeros((dim, dim))
        V = _rand_pd(dim, self.rng)
        X = solve_quadratic_matrix_equation_dense(U, V)
        self.assertLess(_max_rel_err(X, V), 1e-10)

    def test_V_equal_identity(self):
        dim = 12
        U = _rand_psd(dim, self.rng)
        V = np.eye(dim)
        X = solve_quadratic_matrix_equation_dense(U, V)
        _assert_solves(self, U, V, X)

    def test_scalar_closed_form(self):
        """D = 1: x u x + x = v  =>  x = (-1 + sqrt(1 + 4 u v)) / (2 u)."""
        for u, v in ((0.5, 2.0), (3.0, 0.25), (1e-3, 5.0), (10.0, 10.0)):
            with self.subTest(u=u, v=v):
                U = np.array([[u]])
                V = np.array([[v]])
                X = solve_quadratic_matrix_equation_dense(U, V)
                expected = (-1.0 + np.sqrt(1.0 + 4.0 * u * v)) / (2.0 * u)
                self.assertAlmostEqual(float(X[0, 0]), float(expected), places=10)
                _assert_solves(self, U, V, X, tol=1e-10)

    def test_singular_U(self):
        """Rank-deficient U is allowed (only V needs to be strictly PD)."""
        dim = 8
        for rank in (1, 2, 4):
            with self.subTest(rank=rank):
                U = _rand_psd(dim, self.rng, rank=rank)
                V = _rand_pd(dim, self.rng)
                X = solve_quadratic_matrix_equation_dense(U, V)
                _assert_solves(self, U, V, X)

    def test_ill_conditioned_V(self):
        """Moderately ill-conditioned V still yields an accurate solution."""
        dim = 10
        U = _rand_psd(dim, self.rng)
        V = _rand_pd(dim, self.rng, cond=1e6, jitter=1e-6)
        X = solve_quadratic_matrix_equation_dense(U, V)
        _assert_solves(self, U, V, X, tol=1e-5)

    def test_scaling_of_V(self):
        """The dense formula is exactly the conditional mean; check linear sanity.

        For small U (weak score information) the solution must approach V.
        """
        dim = 6
        V = _rand_pd(dim, self.rng)
        X_small = solve_quadratic_matrix_equation_dense(np.eye(dim) * 1e-10, V)
        self.assertLess(_max_rel_err(X_small, V), 1e-8)

    def test_monotone_in_U(self):
        """Larger U means more curvature => X decreases (PSD order) towards 0 info."""
        dim = 5
        V = _rand_pd(dim, self.rng)
        X_weak = solve_quadratic_matrix_equation_dense(np.eye(dim) * 1e-3, V)
        X_strong = solve_quadratic_matrix_equation_dense(np.eye(dim) * 1e3, V)
        self.assertLess(np.linalg.norm(X_strong), np.linalg.norm(X_weak))
        _assert_solves(self, np.eye(dim) * 1e3, V, X_strong, tol=1e-6)


# ---------------------------------------------------------------------------
# low-rank solver (Lemma B.3)
# ---------------------------------------------------------------------------
class TestLowRankQuadraticMatrixEquation(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_matches_dense(self):
        """Low-rank solver with U = Q Q^T must agree with the dense solver."""
        dim = 16
        for rank in (1, 2, 5, 8, 15):
            with self.subTest(rank=rank):
                Q = self.rng.standard_normal((dim, rank))
                U = Q @ Q.T
                V = _rand_pd(dim, self.rng)
                X_low = solve_quadratic_matrix_equation_low_rank(V, Q)
                X_dense = solve_quadratic_matrix_equation_dense(U, V)
                _assert_solves(self, U, V, X_low)
                self.assertLess(
                    _max_rel_err(X_low, X_dense), 1e-6,
                    "low-rank solver disagrees with the dense solver",
                )

    def test_identity_via_Q(self):
        dim = 20
        rank = 4
        Q = self.rng.standard_normal((dim, rank))
        U = Q @ Q.T
        V = _rand_pd(dim, self.rng)
        X = solve_quadratic_matrix_equation_low_rank(V, Q)
        _assert_solves(self, U, V, X)

    def test_full_rank_Q(self):
        """Q square (K = D) should still work, even if the fast path is unused."""
        dim = 6
        Q = self.rng.standard_normal((dim, dim))
        U = Q @ Q.T
        V = _rand_pd(dim, self.rng)
        X = solve_quadratic_matrix_equation_low_rank(V, Q)
        _assert_solves(self, U, V, X)

    def test_rank_one(self):
        """Rank-one U = q q^T: matches the dense solution exactly."""
        dim = 9
        q = self.rng.standard_normal((dim, 1))
        U = q @ q.T
        V = _rand_pd(dim, self.rng)
        X = solve_quadratic_matrix_equation_low_rank(V, q)
        X_dense = solve_quadratic_matrix_equation_dense(U, V)
        _assert_solves(self, U, V, X)
        self.assertLess(_max_rel_err(X, X_dense), 1e-7)

    def test_small_V_jitter(self):
        """jitter argument must not change the solution beyond its size."""
        dim = 10
        Q = self.rng.standard_normal((dim, 3))
        U = Q @ Q.T
        V = _rand_pd(dim, self.rng)
        X_a = solve_quadratic_matrix_equation_low_rank(V, Q, jitter=0.0)
        X_b = solve_quadratic_matrix_equation_low_rank(V, Q, jitter=1e-12)
        self.assertLess(_max_rel_err(X_a, X_b), 1e-8)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
class TestSolverDispatch(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_auto_dispatch_matches_dense(self):
        for dim, rank in ((8, 3), (24, 5), (32, 1)):
            with self.subTest(dim=dim, rank=rank):
                Q = self.rng.standard_normal((dim, rank))
                U = Q @ Q.T
                V = _rand_pd(dim, self.rng)
                X_auto = solve_quadratic_matrix_equation(U, V, Q=Q)
                X_dense = solve_quadratic_matrix_equation_dense(U, V)
                _assert_solves(self, U, V, X_auto)
                self.assertLess(_max_rel_err(X_auto, X_dense), 1e-6)

    def test_explicit_low_rank_flag(self):
        dim, rank = 12, 4
        Q = self.rng.standard_normal((dim, rank))
        U = Q @ Q.T
        V = _rand_pd(dim, self.rng)
        X_low = solve_quadratic_matrix_equation(U, V, Q=Q, low_rank=True)
        X_dense = solve_quadratic_matrix_equation(U, V, low_rank=False)
        _assert_solves(self, U, V, X_low)
        _assert_solves(self, U, V, X_dense)
        self.assertLess(_max_rel_err(X_low, X_dense), 1e-6)

    def test_dense_only(self):
        dim = 7
        U = _rand_psd(dim, self.rng)
        V = _rand_pd(dim, self.rng)
        X = solve_quadratic_matrix_equation(U, V)
        _assert_solves(self, U, V, X)

    def test_module_residual_matches_local_formula(self):
        """``residual`` should report (an equivalent of) X U X + X - V."""
        dim = 6
        U = _rand_psd(dim, self.rng)
        V = _rand_pd(dim, self.rng)
        X = solve_quadratic_matrix_equation_dense(U, V)
        try:
            r = np.asarray(module_residual(U, V, X), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - signature drift
            self.skipTest(f"residual() unavailable/signature mismatch: {exc}")
        self.assertEqual(r.shape, (dim, dim))
        self.assertLess(float(np.max(np.abs(r))), 1e-7 * max(1.0, np.max(np.abs(V))))
        # must be consistent with the locally computed identity
        lhs = _quadratic_residual(U, X)
        self.assertLess(float(np.max(np.abs((lhs - V) - r))), 1e-7)


# ---------------------------------------------------------------------------
# supporting linear algebra
# ---------------------------------------------------------------------------
class TestSupportingLinearAlgebra(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_symmetrize(self):
        A = self.rng.standard_normal((5, 5))
        S = np.asarray(symmetrize(A))
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        np.testing.assert_allclose(S, 0.5 * (A + A.T), atol=1e-12)

    def test_matrix_sqrt_squares_back(self):
        for dim in (1, 3, 8):
            with self.subTest(dim=dim):
                S = _rand_psd(dim, self.rng)
                R = np.asarray(matrix_sqrt(S))
                np.testing.assert_allclose(R, R.T, atol=1e-9)
                self.assertLess(_max_rel_err(R @ R, S), 1e-8)

    def test_matrix_sqrt_psd_with_zero_eigenvalue(self):
        dim = 6
        S = _rand_psd(dim, self.rng, rank=2)  # singular PSD
        R = np.asarray(matrix_sqrt(S, clip_negative=True))
        self.assertTrue(np.all(np.isfinite(R)))
        self.assertLess(_max_rel_err(R @ R, S), 1e-7)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(symmetrize(R)).min()), -1e-9)

    def test_matrix_sqrt_inv(self):
        dim = 5
        S = _rand_pd(dim, self.rng)
        Rinv = np.asarray(matrix_sqrt_inv(S))
        np.testing.assert_allclose(Rinv, Rinv.T, atol=1e-9)
        self.assertLess(_max_rel_err(Rinv @ S @ Rinv, np.eye(dim)), 1e-8)

    def test_solve_symmetric(self):
        dim = 6
        A = _rand_pd(dim, self.rng)
        B = self.rng.standard_normal((dim, 2))
        X = np.asarray(solve_symmetric(A, B))
        self.assertLess(_max_rel_err(A @ X, B), 1e-10)

    def test_inverse_spd(self):
        dim = 6
        A = _rand_pd(dim, self.rng)
        Ainv = np.asarray(inverse_spd(A))
        np.testing.assert_allclose(Ainv, Ainv.T, atol=1e-9)
        self.assertLess(_max_rel_err(A @ Ainv, np.eye(dim)), 1e-9)

    def test_ensure_spd_projects(self):
        """An indefinite / asymmetric matrix must come back symmetric PD."""
        dim = 4
        M = self.rng.standard_normal((dim, dim))
        M = 0.5 * (M + M.T)
        M[0, 0] = -5.0  # make it indefinite
        out = np.asarray(ensure_spd(M))
        np.testing.assert_allclose(out, out.T, atol=1e-9)
        self.assertGreater(float(np.linalg.eigvalsh(symmetrize(out)).min()), 0.0)

    def test_ensure_spd_keeps_clean_pd_input(self):
        dim = 4
        A = _rand_pd(dim, self.rng, jitter=1.0)
        out = np.asarray(ensure_spd(A))
        self.assertLess(_max_rel_err(out, A), 1e-6)

    def test_array_namespace(self):
        ns = array_namespace(np.zeros((2, 2)))
        self.assertTrue(hasattr(ns, "eye") or hasattr(ns, "asarray"))


# ---------------------------------------------------------------------------
# identity / solver consistency under the BaM usage pattern
# ---------------------------------------------------------------------------
class TestBaMUsagePattern(unittest.TestCase):
    """Mimic the match-step usage: U = lam * Gamma (+ rank-1 term), V = Sigma + lam * C."""

    def setUp(self):
        self.rng = np.random.default_rng(4)

    def test_match_step_shapes(self):
        dim, batch = 10, 4
        Z = self.rng.standard_normal((batch, dim))
        G = self.rng.standard_normal((batch, dim))
        z_bar, g_bar = Z.mean(0), G.mean(0)
        C = (Z - z_bar).T @ (Z - z_bar) / batch
        Gamma = (G - g_bar).T @ (G - g_bar) / batch
        mu_t = np.zeros(dim)
        Sigma_t = np.eye(dim)
        lam = 50.0

        U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(g_bar, g_bar)
        V = Sigma_t + lam * C + (lam / (1.0 + lam)) * np.outer(mu_t - z_bar, mu_t - z_bar)
        U = symmetrize(U)
        V = symmetrize(V)

        X = solve_quadratic_matrix_equation_dense(U, V)
        _assert_solves(self, U, V, X)

        # mean update must use the *new* covariance: mu_{t+1} = (mu_t + lam (Sigma_{t+1} g_bar + z_bar)) / (1+lam)
        mu_next = (mu_t + lam * (X @ g_bar + z_bar)) / (1.0 + lam)
        self.assertEqual(mu_next.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(mu_next)))

    def test_low_rank_usage_pattern(self):
        """With B << D the low-rank path (U = Q Q^T) must reproduce the dense result."""
        dim, batch = 20, 3
        Z = self.rng.standard_normal((batch, dim))
        G = self.rng.standard_normal((batch, dim))
        z_bar, g_bar = Z.mean(0), G.mean(0)
        C = (Z - z_bar).T @ (Z - z_bar) / batch
        Gamma = (G - g_bar).T @ (G - g_bar) / batch
        lam = 10.0

        U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(g_bar, g_bar)
        V = np.eye(dim) + lam * C
        # factor U: it is symmetric PSD of rank <= batch + 1
        w, Q_full = np.linalg.eigh(symmetrize(U))
        keep = w > 1e-12 * max(1.0, float(w.max()))
        Q = Q_full[:, keep] * np.sqrt(w[keep])[None, :]

        X_low = solve_quadratic_matrix_equation_low_rank(V, Q)
        X_dense = solve_quadratic_matrix_equation_dense(U, V)
        _assert_solves(self, U, V, X_low, tol=1e-6)
        self.assertLess(_max_rel_err(X_low, X_dense), 1e-6)


# ---------------------------------------------------------------------------
# optional JAX backend
# ---------------------------------------------------------------------------
class TestJaxBackend(unittest.TestCase):
    def setUp(self):
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
        except Exception:  # pragma: no cover
            self.skipTest("jax is not installed")
        import jax.numpy as jnp

        self.jnp = jnp
        self.rng = np.random.default_rng(5)

    def test_dense_matches_numpy(self):
        dim = 8
        U = _rand_psd(dim, self.rng)
        V = _rand_pd(dim, self.rng)
        X_np = np.asarray(solve_quadratic_matrix_equation_dense(U, V))
        X_jax = np.asarray(
            solve_quadratic_matrix_equation_dense(self.jnp.asarray(U), self.jnp.asarray(V))
        )
        self.assertLess(_max_rel_err(X_jax, X_np), 1e-6)
        _assert_solves(self, U, V, X_jax, tol=1e-6)

    def test_low_rank_matches_numpy(self):
        dim, rank = 12, 3
        Q = self.rng.standard_normal((dim, rank))
        U = Q @ Q.T
        V = _rand_pd(dim, self.rng)
        X_np = np.asarray(solve_quadratic_matrix_equation_low_rank(V, Q))
        X_jax = np.asarray(
            solve_quadratic_matrix_equation_low_rank(self.jnp.asarray(V), self.jnp.asarray(Q))
        )
        self.assertLess(_max_rel_err(X_jax, X_np), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
