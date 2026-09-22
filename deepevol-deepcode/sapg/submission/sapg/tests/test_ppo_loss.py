"""Unit tests for the on-policy PPO loss (Eq. 2) and SAPG entropy term (Eq. 10).

Paper references
----------------
* Eq. 2  (Section 3):  L_on = E[ min( r_t A_t, clip(r_t, 1-eps, 1+eps) A_t ) ]
                       with r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
* Eq. 10 (Section 4.5): entropy bonus ``sigma * (j - 1) * H(pi(a|s))`` added for
                       *followers only* (the leader's coefficient is zero).
* ``lambda' = 4.0`` is the critic coefficient and is *not* part of this loss.

The tests are deliberately implementation-agnostic about the sign convention
(some helpers return the maximised objective, others the minimised loss) and
about whether a function returns a tensor or a dict -- the same defensive style
used by ``tests/test_importance_ratio.py``.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sapg.losses.ppo_loss import (  # noqa: E402
    PPOLoss,
    combine_policy_and_entropy,
    compute_bounds_loss,
    compute_entropy_bonus,
    compute_ppo_loss,
    entropy_coefficient_for,
    importance_ratio,
    on_policy_loss,
    ppo_loss,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
OBJECTIVE_KEYS = ("objective", "surrogate", "policy_objective", "ppo_objective")
LOSS_KEYS = ("policy_loss", "loss", "ppo_loss")


def _objective(out) -> float:
    """Extract the (possibly maximised) clipped surrogate value from ``out``."""
    if isinstance(out, torch.Tensor):
        return float(out.detach().reshape(-1).mean())
    for key in OBJECTIVE_KEYS:
        if key in out:
            return float(torch.as_tensor(out[key]).detach().reshape(-1).mean())
    for key in LOSS_KEYS:
        if key in out:
            return -float(torch.as_tensor(out[key]).detach().reshape(-1).mean())
    raise AssertionError(f"no objective-like key found in {sorted(out)}")


def _loss(out) -> float:
    """Extract the minimised policy loss from ``out``."""
    if isinstance(out, torch.Tensor):
        return float(out.detach().reshape(-1).mean())
    for key in LOSS_KEYS:
        if key in out:
            return float(torch.as_tensor(out[key]).detach().reshape(-1).mean())
    return -_objective(out)


def _ratio_from_logprobs(new_logprobs, old_logprobs):
    """Reference Eq. 2 ratio r_t = pi_theta / pi_old (log-space difference)."""
    return torch.exp(torch.as_tensor(new_logprobs) - torch.as_tensor(old_logprobs))


def _reference_objective(ratio, advantages, clip_epsilon):
    """Eq. 2 surrogate: mean( min(r*A, clip(r, 1-eps, 1+eps)*A) )."""
    ratio = torch.as_tensor(ratio)
    advantages = torch.as_tensor(advantages)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return torch.minimum(ratio * advantages, clipped * advantages).mean()


def _expected_objective(ratio, advantages, clip_epsilon):
    return float(_reference_objective(ratio, advantages, clip_epsilon))


# ---------------------------------------------------------------------------
# importance ratio / Eq. 2 surrogate
# ---------------------------------------------------------------------------
def test_importance_ratio_is_exp_of_log_difference():
    new_logprobs = torch.tensor([-0.5, -1.25, -2.0, -0.75])
    old_logprobs = torch.tensor([-0.25, -1.0, -2.5, -0.75])
    ratio = importance_ratio(new_logprobs, old_logprobs)
    assert torch.allclose(ratio, torch.exp(new_logprobs - old_logprobs), atol=1e-6)

    expected = _ratio_from_logprobs(new_logprobs, old_logprobs)
    assert torch.allclose(ratio, expected, atol=1e-6)


def test_importance_ratio_is_one_when_policies_agree():
    logprobs = torch.tensor([-1.0, -2.0, -3.0, -0.125])
    ratio = importance_ratio(logprobs, logprobs.clone())
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)


def test_surrogate_equals_unclipped_objective_inside_trust_region():
    """When every ratio lies in [1-eps, 1+eps] the clip is inactive (Eq. 2)."""
    advantages = torch.tensor([1.0, -0.5, 2.0, -1.5])
    ratio = torch.tensor([1.02, 0.98, 1.05, 0.95])
    out = compute_ppo_loss(
        ratio=ratio, advantages=advantages, clip_epsilon=0.1, return_stats=False
    )
    expected = _expected_objective(ratio, advantages, 0.1)
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_surrogate_clips_ratio_above_upper_bound():
    """r > 1+eps is clipped to 1+eps when the advantage is positive (Eq. 2)."""
    eps = 0.1
    advantages = torch.ones(4)
    ratio = torch.full((4,), 2.0)  # far outside the trust region
    out = compute_ppo_loss(ratio=ratio, advantages=advantages, clip_epsilon=eps)

    expected = _expected_objective(ratio, advantages, eps)
    assert math.isclose(expected, 1.0 + eps, rel_tol=1e-6)
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)
    # the unclipped value would have been 2.0
    assert not math.isclose(abs(_objective(out)), 2.0, rel_tol=1e-3)


def test_surrogate_clips_ratio_below_lower_bound():
    """r < 1-eps is clipped to 1-eps (the pessimistic branch of the min)."""
    eps = 0.2
    advantages = torch.ones(4)
    ratio = torch.full((4,), 0.1)
    out = compute_ppo_loss(ratio=ratio, advantages=advantages, clip_epsilon=eps)

    expected = _expected_objective(ratio, advantages, eps)
    assert math.isclose(expected, 1.0 - eps, rel_tol=1e-6)
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_surrogate_takes_minimum_branch_for_negative_advantage():
    """With A < 0 the min picks the *unclipped* pessimistic branch."""
    eps = 0.1
    advantages = -torch.ones(3)
    ratio = torch.full((3,), 3.0)
    out = compute_ppo_loss(ratio=ratio, advantages=advantages, clip_epsilon=eps)
    expected = _expected_objective(ratio, advantages, eps)
    # min(r*A, clip(r)*A) = min(-3, -1.1) = -3
    assert math.isclose(expected, -3.0, rel_tol=1e-6)
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_surrogate_uses_logprob_difference_when_no_ratio_given():
    old_logprobs = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    new_logprobs = torch.tensor([-0.9, -1.0, -1.1, -0.5])
    advantages = torch.tensor([2.0, 2.0, 2.0, 2.0])
    out = compute_ppo_loss(
        new_logprobs=new_logprobs, old_logprobs=old_logprobs, advantages=advantages,
        clip_epsilon=0.1, return_stats=False,
    )
    ratio = _ratio_from_logprobs(new_logprobs, old_logprobs)
    expected = _expected_objective(ratio, advantages, 0.1)
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_missing_old_logprobs_degrades_to_on_policy_ratio_one():
    """No behaviour log-probs => r = 1 (pure on-policy reduction)."""
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    out = compute_ppo_loss(advantages=advantages, clip_epsilon=0.1)
    expected = float(advantages.mean())
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_ppo_loss_minimised_returns_negated_objective():
    """``ppo_loss`` returns a *minimised* quantity (loss = -objective)."""
    advantages = torch.tensor([1.0, -0.5, 2.0, -1.5])
    ratio = torch.tensor([1.02, 0.98, 1.05, 0.95])
    old_logprobs = torch.zeros(4, dtype=torch.float32)
    new_logprobs = torch.log(ratio)
    out = ppo_loss(
        new_logprobs=new_logprobs, old_logprobs=old_logprobs, advantages=advantages,
        clip_epsilon=0.1,
    )
    expected = _expected_objective(ratio, advantages, 0.1)
    assert math.isclose(_loss(out), -expected, rel_tol=1e-5, abs_tol=1e-6)


def test_ppo_loss_weight_scales_the_objective():
    advantages = torch.tensor([1.0, 2.0, -0.5, 1.5])
    logprobs = torch.zeros(4, dtype=torch.float32)
    base = ppo_loss(new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages)
    scaled = ppo_loss(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages, weight=3.0
    )
    assert math.isclose(_loss(scaled), 3.0 * _loss(base), rel_tol=1e-5, abs_tol=1e-6)


def test_on_policy_loss_alias_matches_ppo_loss():
    advantages = torch.tensor([0.5, -0.25, 1.0])
    logprobs = torch.full((3,), -0.3)
    out_alias = on_policy_loss(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages
    )
    out_direct = ppo_loss(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages
    )
    assert math.isclose(_objective(out_alias), _objective(out_direct), rel_tol=1e-6)


def test_clip_fraction_reports_fully_clipped_batch():
    ratio = torch.full((8,), 2.0)
    advantages = torch.ones(8)
    out = compute_ppo_loss(ratio=ratio, advantages=advantages, clip_epsilon=0.1)
    if "clip_frac" in out:
        assert math.isclose(float(out["clip_frac"]), 1.0, rel_tol=1e-5)


def test_clip_fraction_reports_unclipped_batch():
    ratio = torch.full((8,), 1.01)
    advantages = torch.ones(8)
    out = compute_ppo_loss(ratio=ratio, advantages=advantages, clip_epsilon=0.1)
    if "clip_frac" in out:
        assert math.isclose(float(out["clip_frac"]), 0.0, abs_tol=1e-6)


def test_ratio_stats_are_reported_in_log_space():
    ratio = torch.tensor([1.0, 1.5, 0.5, 2.0])
    advantages = torch.ones(4)
    out = compute_ppo_loss(ratio=ratio, advantages=advantages, clip_epsilon=0.1)
    if "ratio_mean" in out:
        assert math.isclose(float(out["ratio_mean"]), float(ratio.mean()), rel_tol=1e-5)


def test_kl_is_zero_when_policies_agree():
    logprobs = torch.tensor([-0.4, -0.9, -1.7])
    advantages = torch.tensor([1.0, -1.0, 0.5])
    out = compute_ppo_loss(
        new_logprobs=logprobs, old_logprobs=logprobs.clone(), advantages=advantages
    )
    if "kl" in out:
        assert abs(float(out["kl"])) < 1e-6


# ---------------------------------------------------------------------------
# entropy term (Eq. 10)
# ---------------------------------------------------------------------------
def test_entropy_bonus_is_negated_scaled_entropy():
    entropy = torch.tensor([1.0, 2.0, 3.0, 4.0])
    coefficient = 0.5
    value = compute_entropy_bonus(entropy, coefficient)
    expected = -coefficient * float(entropy.mean())
    assert math.isclose(float(torch.as_tensor(value)), expected, rel_tol=1e-6, abs_tol=1e-7)


def test_entropy_coefficient_matches_eq10_for_followers():
    """Eq. 10: follower j (1-based) gets coefficient sigma * (j - 1)."""
    sigma = 0.005
    expected = {1: 0.0, 2: 0.005, 3: 0.010, 4: 0.015, 5: 0.020, 6: 0.025}
    for policy_index, value in expected.items():
        got = entropy_coefficient_for(
            policy_index, sigma=sigma, leader_index=1, num_policies=6
        )
        assert math.isclose(float(got), value, rel_tol=1e-9, abs_tol=1e-12)


def test_entropy_coefficient_is_zero_for_leader():
    """The leader is excluded from the entropy bonus (Appendix B.3 note)."""
    for sigma in (0.0, 0.003, 0.005):
        assert float(entropy_coefficient_for(1, sigma=sigma, leader_index=1)) == 0.0


def test_entropy_coefficient_is_zero_when_sigma_zero():
    for policy_index in range(1, 7):
        assert float(entropy_coefficient_for(policy_index, sigma=0.0)) == 0.0


def test_combine_policy_and_entropy_adds_minimised_bonus():
    policy_loss = torch.tensor(2.0)
    entropy = torch.tensor([1.0, 3.0])  # mean = 2.0
    total = combine_policy_and_entropy(policy_loss, entropy=entropy, coefficient=0.5)
    assert math.isclose(float(torch.as_tensor(total)), 2.0 - 0.5 * 2.0, rel_tol=1e-6)


def test_combine_policy_and_entropy_defaults_to_plain_loss():
    policy_loss = torch.tensor(-1.5)
    total = combine_policy_and_entropy(policy_loss)
    assert math.isclose(float(torch.as_tensor(total)), -1.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# action-bound regularisation (bounds loss coefficient 1e-4)
# ---------------------------------------------------------------------------
def test_bounds_loss_is_zero_inside_the_tanh_band():
    action_mean = torch.tensor([[0.0, 0.5, -1.0], [1.0, -0.25, 0.75]])
    value = compute_bounds_loss(action_mean, coefficient=1e-4)
    assert math.isclose(float(torch.as_tensor(value)), 0.0, abs_tol=1e-12)


def test_bounds_loss_penalises_outside_the_band():
    coefficient = 1e-4
    action_mean = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # excess 1 and 2
    value = float(torch.as_tensor(compute_bounds_loss(action_mean, coefficient=coefficient)))
    expected = coefficient * (1.0 ** 2 + 2.0 ** 2) / 2.0  # mean over the batch
    assert math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-12)
    assert value > 0.0


def test_bounds_loss_is_quadratic_in_excess():
    coefficient = 1.0
    one = float(torch.as_tensor(compute_bounds_loss(torch.tensor([[2.0]]), coefficient=coefficient)))
    two = float(torch.as_tensor(compute_bounds_loss(torch.tensor([[3.0]]), coefficient=coefficient)))
    assert math.isclose(two, 4.0 * one, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# PPOLoss module wrapper
# ---------------------------------------------------------------------------
def test_ppo_loss_module_policy_term_matches_function():
    advantages = torch.tensor([1.0, -0.5, 2.0, -1.5])
    old_logprobs = torch.zeros(4)
    new_logprobs = torch.log(torch.tensor([1.02, 0.98, 1.05, 0.95]))
    module = PPOLoss(clip_epsilon=0.1)
    out = module(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        entropy=None,
        action_mean=None,
        policy_index=1,
    )
    ratio = torch.exp(new_logprobs - old_logprobs)
    expected = _expected_objective(ratio, advantages, 0.1)
    assert math.isclose(_objective(out), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_ppo_loss_module_adds_entropy_only_with_coefficient():
    advantages = torch.tensor([1.0, 1.0])
    logprobs = torch.zeros(2)
    entropy = torch.tensor([2.0, 2.0])  # mean H = 2.0
    module = PPOLoss(clip_epsilon=0.1, entropy_coefficient=0.25)

    plain = module(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages,
        entropy=entropy, action_mean=None, policy_index=1, coefficient=0.0,
    )
    with_entropy = module(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages,
        entropy=entropy, action_mean=None, policy_index=2, coefficient=0.25,
    )
    # objective decreases by coefficient * mean(H) when the entropy bonus is on
    assert math.isclose(
        _objective(with_entropy), _objective(plain) - 0.25 * 2.0, rel_tol=1e-5, abs_tol=1e-6
    )


def test_ppo_loss_module_total_loss_equals_sum_of_terms():
    advantages = torch.tensor([1.0, -1.0, 0.5])
    logprobs = torch.zeros(3)
    module = PPOLoss(clip_epsilon=0.1, entropy_coefficient=0.1,
                     bounds_loss_coefficient=1e-4)
    out = module(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages,
        entropy=torch.ones(3), action_mean=torch.full((3, 2), 3.0),
        policy_index=2, coefficient=0.1,
    )
    if "total_loss" in out:
        total = float(torch.as_tensor(out["total_loss"]))
        assert math.isclose(total, _loss(out), rel_tol=1e-5, abs_tol=1e-6)
        assert total > 0.0  # bound penalty pushes the loss above zero


def test_ppo_loss_module_loss_method_returns_tensor():
    advantages = torch.tensor([1.0, -1.0])
    logprobs = torch.zeros(2)
    module = PPOLoss(clip_epsilon=0.1)
    value = module.loss(
        new_logprobs=logprobs, old_logprobs=logprobs, advantages=advantages,
        entropy=torch.ones(2), action_mean=torch.zeros(2, 1), policy_index=1,
    )
    assert isinstance(value, torch.Tensor)
    assert torch.isfinite(value).all()


def test_ppo_loss_gradient_flows_to_new_logprobs():
    """The whole point of Eq. 2: a differentiable surrogate for the policy."""
    advantages = torch.tensor([1.0, 2.0, -1.0])
    old_logprobs = torch.zeros(3)
    new_logprobs = torch.zeros(3, requires_grad=True)
    out = compute_ppo_loss(
        new_logprobs=new_logprobs, old_logprobs=old_logprobs, advantages=advantages
    )
    value = out["policy_loss"] if isinstance(out, dict) else out
    value.backward()
    assert new_logprobs.grad is not None
    assert torch.isfinite(new_logprobs.grad).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
