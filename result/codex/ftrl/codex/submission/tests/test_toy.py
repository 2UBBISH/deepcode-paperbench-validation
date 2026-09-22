"""Tests for the toy environments (Appendix A)."""

from __future__ import annotations

import numpy as np
import pytest

from fpc.toy.apple_retrieval import AppleRetrieval, LinearPolicy, pretrain_on_phase2, reinforce
from fpc.toy.two_state_mdp import (
    build_imperfect_cloning_gap,
    build_state_coverage_gap,
    imperfect_cloning_gap_f,
    paper_value_function,
    run_paper_scenarios,
    state_coverage_gap_f,
)


def test_two_state_mdp_values_are_finite_and_positive():
    mdp = build_state_coverage_gap()
    v0, v1 = mdp.values(0.5)
    assert np.isfinite(v0) and np.isfinite(v1)
    assert v1 > 0


def test_state_coverage_gap_parametrisation_is_continuous():
    eps = 0.02
    left = state_coverage_gap_f(1.0 - eps / 2 - 1e-9, eps)
    right = state_coverage_gap_f(1.0 - eps / 2 + 1e-9, eps)
    assert left == pytest.approx(right, abs=1e-6)


def test_imperfect_cloning_gap_hits_one_at_theta_one():
    assert imperfect_cloning_gap_f(1.0) == pytest.approx(1.0)
    assert imperfect_cloning_gap_f(0.0) == pytest.approx(1.0)


def test_paper_value_function_optimum_is_one_over_one_minus_gamma():
    value = paper_value_function(theta=1.0, f_theta=1.0, r0=0.0, r1=-1.0, gamma=0.9)
    assert value == pytest.approx(10.0)


def test_reported_suboptimal_fixed_point_matches_paper():
    """Appendix A.1 reports convergence to theta = 0.11 with value 2.22."""

    scenario = [s for s in run_paper_scenarios(steps=20_000) if s.name == "reported_suboptimal_fixed_point"][0]
    assert scenario.converged_theta == pytest.approx(0.1111, abs=0.01)
    assert scenario.converged_value == pytest.approx(2.2222, abs=0.05)
    assert scenario.optimal_value == pytest.approx(10.0, abs=1e-6)


def test_apple_retrieval_phase_sign_convention():
    """Phase 1 rewards moving right, Phase 2 rewards moving left."""

    env = AppleRetrieval(M=5, c=1.0)
    right_policy = LinearPolicy(c=1.0, w=-10.0, b=0.0)  # high p(right) in phase 1
    result = env.rollout(right_policy, phase=0, greedy=True)
    assert result["reached_goal"]
    assert all(r == 1.0 for r in result["returns"][:5])


def test_apple_retrieval_forgetting_depends_on_c():
    """Small c -> bias-dominated pre-trained policy -> forgetting (Figure 11)."""

    small = AppleRetrieval(M=30, c=0.1, seed=0)
    policy, _ = pretrain_on_phase2(small, episodes=500, lr=1e-2, seed=0)
    assert policy.wb_ratio > 1.0  # bias dominates
    reinforce(small, policy, episodes=1000, lr=1e-2, phase=None, seed=0)
    small_success = small.evaluate(policy, episodes=50, phase=1)

    large = AppleRetrieval(M=30, c=5.0, seed=0)
    policy2, _ = pretrain_on_phase2(large, episodes=500, lr=1e-2, seed=0)
    reinforce(large, policy2, episodes=1000, lr=1e-2, phase=None, seed=0)
    large_success = large.evaluate(policy2, episodes=50, phase=1)

    assert small_success <= large_success
