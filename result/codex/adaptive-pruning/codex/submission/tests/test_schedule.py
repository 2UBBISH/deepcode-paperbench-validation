"""Schedule tests (Appendix A)."""

from apt.schedule import SparsitySchedule, adjustment_steps, linear_rank, mu_schedule


def test_cubic_schedule_endpoints_and_monotonicity():
    s = SparsitySchedule(target_sparsity=0.6, total_steps=100)
    assert abs(s.sparsity(0) - 0.0) < 1e-9
    assert abs(s.sparsity(100) - 0.6) < 1e-9
    values = [s.sparsity(t) for t in range(0, 101, 5)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    # "early pruning": more than half of the pruning budget is spent in the
    # first third of training
    assert s.sparsity(33) > 0.5 * 0.6


def test_density_formula_matches_the_paper():
    """gamma_t = gamma_T + (1 - gamma_T)(1 - t/T)^3 with gamma = retained ratio."""
    s = SparsitySchedule(target_sparsity=0.6, total_steps=100)
    gamma_T = 0.4
    for t in (0, 25, 50, 75, 100):
        expected = gamma_T + (1 - gamma_T) * (1 - t / 100.0) ** 3
        assert abs(s.density(t) - expected) < 1e-9


def test_mu_schedule():
    assert mu_schedule(0, 0, 100) == 0.0
    assert mu_schedule(50, 0, 100) == 0.5
    assert mu_schedule(100, 0, 100) == 1.0
    assert mu_schedule(120, 0, 100) == 1.0
    assert mu_schedule(5, 10, 100) == 0.0


def test_rank_and_adjustment_helpers():
    assert linear_rank(64, 8, 0, 100) == 8
    assert linear_rank(64, 8, 100, 100) == 64
    assert adjustment_steps(100, 25) == [25, 50, 75, 100]
