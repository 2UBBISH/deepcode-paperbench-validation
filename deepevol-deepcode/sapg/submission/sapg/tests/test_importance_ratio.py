"""Unit tests for the SAPG importance ratio and mu-corrected clipping bounds.

These tests cover Section 4.1 of the paper:

    r_pi_i(s, a) = pi_i(s, a) / pi_j(s, a)                    (Eq. 3 ratio)
    mu           = pi_{i,old}(s, a) / pi_j(s, a)              (Eq. 3 correction)

and the requirement (planned unit test #1) that the ratio reduces to the
on-policy update when the target policy and the behaviour policy coincide
(``i == j``), in which case ``r = mu = 1`` and the off-policy surrogate
degenerates to the plain on-policy PPO surrogate.

Run with::

    python -m pytest sapg/tests/test_importance_ratio.py -q
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sapg.losses.off_policy_loss import (  # noqa: E402
    combine_on_off_loss,
    importance_ratio,
    mu_from_logprobs,
    off_policy_loss,
    off_policy_surrogate,
)
from sapg.losses.ppo_loss import importance_ratio as on_policy_importance_ratio  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _call_surrogate(**kwargs):
    """Call ``off_policy_surrogate`` and normalise the optional stats tuple."""
    out = off_policy_surrogate(**kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _sign_convention() -> float:
    """Determine the sign convention used by ``off_policy_surrogate``.

    Both conventions appear in the wild (maximised objective vs. minimised
    loss), so the tests are written against whichever convention the module
    actually uses.
    """
    probe = _call_surrogate(
        new_logprobs=torch.zeros(4),
        behaviour_logprobs=torch.zeros(4),
        advantages=torch.ones(4),
        reduction="mean",
    )
    sign = float(torch.sign(probe.reshape(-1)[0]))
    return sign if sign != 0.0 else 1.0


def _expected_objective(ratio, advantages, mu, clip_epsilon):
    """Eq. 3 surrogate value (maximised convention)."""
    unclipped = ratio * advantages
    lower = mu * (1.0 - clip_epsilon)
    upper = mu * (1.0 + clip_epsilon)
    clipped_ratio = torch.clamp(ratio, lower, upper)
    return torch.minimum(unclipped, clipped_ratio * advantages).mean()


# ---------------------------------------------------------------------------
# raw ratio / mu computation
# ---------------------------------------------------------------------------
def test_importance_ratio_equals_exp_of_log_difference():
    new = torch.tensor([0.0, -0.5, 1.25, 3.0])
    old = torch.tensor([0.0, 0.5, 0.25, -1.0])
    ratio = importance_ratio(new, old)
    expected = torch.exp(new - old)
    assert torch.allclose(ratio, expected, atol=1e-6)


def test_importance_ratio_matches_on_policy_variant():
    new = torch.tensor([0.1, -0.2, 0.75])
    old = torch.tensor([-0.3, 0.4, 0.75])
    assert torch.allclose(
        importance_ratio(new, old), on_policy_importance_ratio(new, old), atol=1e-6
    )


def test_importance_ratio_is_one_when_policies_agree():
    """i == j  =>  pi_i == pi_j  =>  r = 1 (on-policy reduction)."""
    logprobs = torch.tensor([-1.5, 0.0, 2.25, -4.0])
    ratio = importance_ratio(logprobs, logprobs.clone())
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)


def test_mu_is_one_when_target_old_matches_behaviour():
    logprobs = torch.tensor([-0.9, 0.1, 1.4, -2.0])
    mu = mu_from_logprobs(logprobs, logprobs.clone())
    assert torch.allclose(mu, torch.ones_like(mu), atol=1e-6)


def test_mu_from_logprobs_formula():
    target_old = torch.tensor([0.0, 1.0, -1.0, 2.5])
    behaviour = torch.tensor([0.0, 0.5, 0.5, 0.0])
    mu = mu_from_logprobs(target_old, behaviour)
    assert torch.allclose(mu, torch.exp(target_old - behaviour), atol=1e-6)


# ---------------------------------------------------------------------------
# mu-scaled clipping bounds (Eq. 3)
# ---------------------------------------------------------------------------
def test_surrogate_reduces_to_on_policy_when_i_equals_j():
    """When behaviour == target-old == new policy, r = mu = 1 and the
    surrogate is simply the mean advantage."""
    advantages = torch.tensor([1.0, 2.0, -1.0, 0.5])
    logprobs = torch.tensor([0.2, -0.4, 0.9, 1.1])
    sign = _sign_convention()
    value = _call_surrogate(
        new_logprobs=logprobs,
        behaviour_logprobs=logprobs.clone(),
        target_old_logprobs=logprobs.clone(),
        advantages=advantages,
        clip_epsilon=0.1,
        reduction="mean",
    )
    scalar = value.reshape(-1).mean() if value.ndim > 0 else value
    assert math.isclose(float(scalar), sign * float(advantages.mean()), abs_tol=1e-5)


def test_surrogate_clips_ratio_above_upper_mu_bound():
    """r above mu(1+eps) must clamp the ratio to mu(1+eps)."""
    eps = 0.1
    mu = torch.full((4,), 1.0)
    # r = exp(2.0) ~ 7.39 >> 1.1
    new = torch.full((4,), 2.0)
    behaviour = torch.zeros(4)
    target_old = torch.zeros(4)
    advantages = torch.ones(4)
    ratio = torch.exp(new - behaviour)
    expected = _expected_objective(ratio, advantages, mu, eps)
    sign = _sign_convention()
    value = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=target_old,
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    scalar = value.reshape(-1).mean() if value.ndim > 0 else value
    assert math.isclose(float(scalar), sign * float(expected), rel_tol=1e-5)
    # and the clipped value is the mu-scaled upper bound, not 1+eps
    assert math.isclose(float(scalar), sign * (1.0 + eps), rel_tol=1e-5)


def test_surrogate_clips_ratio_below_lower_mu_bound():
    eps = 0.1
    # r = exp(-2.0) ~ 0.135 < 0.9 -> clamp keeps r (min(r*A, clip*A) = r*A)
    new = torch.full((4,), -2.0)
    behaviour = torch.zeros(4)
    ratio = torch.exp(new - behaviour)
    advantages = torch.ones(4)
    mu = torch.ones(4)
    expected = _expected_objective(ratio, advantages, mu, eps)
    sign = _sign_convention()
    value = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=behaviour.clone(),
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    scalar = value.reshape(-1).mean() if value.ndim > 0 else value
    assert math.isclose(float(scalar), sign * float(expected), rel_tol=1e-5)


def test_clipping_bounds_scale_with_mu():
    """A non-unit ``mu`` widens/narrows the clipping window proportionally."""
    eps = 0.1
    mu = torch.full((4,), 2.0)
    # r = exp(2.0) >> mu * 1.1 = 2.2 -> clipped to 2.2 * A
    new = torch.full((4,), 2.0)
    behaviour = torch.zeros(4)
    target_old = torch.log(mu)  # mu = exp(target_old - behaviour) = 1.0 * ...
    advantages = torch.ones(4)
    ratio = torch.exp(new - behaviour)
    expected = _expected_objective(ratio, advantages, mu, eps)
    sign = _sign_convention()
    value = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=target_old,
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    scalar = value.reshape(-1).mean() if value.ndim > 0 else value
    assert math.isclose(float(scalar), sign * float(expected), rel_tol=1e-5)


def test_explicit_mu_argument_is_used():
    """Passing ``mu`` directly must override the value derived from log-probs."""
    eps = 0.1
    new = torch.full((4,), 2.0)
    behaviour = torch.zeros(4)
    advantages = torch.ones(4)
    value_derived = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=torch.zeros(4),
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    value_explicit = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        advantages=advantages,
        mu=torch.full((4,), 3.0),
        clip_epsilon=eps,
        reduction="mean",
    )
    # derived: clipped to 1.1 ; explicit: clipped to 3.3
    a = value_derived.reshape(-1).mean() if value_derived.ndim else value_derived
    b = value_explicit.reshape(-1).mean() if value_explicit.ndim else value_explicit
    assert not math.isclose(float(a), float(b), rel_tol=1e-3)
    assert math.isclose(abs(float(b)) / abs(float(a)), 3.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# per-source reduction (Eq. 3 averages over X = {2..M})
# ---------------------------------------------------------------------------
def test_per_source_reduction_averages_source_means():
    """Eq. 3 averages E_{pi_j}[.] per source policy j, not per transition."""
    # Two sources with very different advantage magnitudes and equal size.
    advantages = torch.tensor([1.0, 1.0, 4.0, 4.0])
    source = torch.tensor([0, 0, 1, 1])
    logprobs = torch.zeros(4)
    sign = _sign_convention()

    per_source = _call_surrogate(
        new_logprobs=logprobs,
        behaviour_logprobs=logprobs.clone(),
        target_old_logprobs=logprobs.clone(),
        advantages=advantages,
        source_policy=source,
        clip_epsilon=0.1,
        reduction="source",
    )
    flat = _call_surrogate(
        new_logprobs=logprobs,
        behaviour_logprobs=logprobs.clone(),
        target_old_logprobs=logprobs.clone(),
        advantages=advantages,
        source_policy=source,
        clip_epsilon=0.1,
        reduction="mean",
    )
    per_source_scalar = float(
        per_source.reshape(-1).mean() if per_source.ndim else per_source
    )
    flat_scalar = float(flat.reshape(-1).mean() if flat.ndim else flat)
    # source-mean average = (1 + 4)/2 = 2.5 ; flat mean = 2.5 as well here
    assert math.isclose(per_source_scalar, 2.5 * sign, rel_tol=1e-5)
    assert math.isclose(flat_scalar, 2.5 * sign, rel_tol=1e-5)

    # Now make the sources unequal in size: flat mean != source-mean average.
    advantages2 = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    source2 = torch.tensor([0, 1, 1, 1, 1])
    per_source2 = _call_surrogate(
        new_logprobs=torch.zeros(5),
        behaviour_logprobs=torch.zeros(5),
        target_old_logprobs=torch.zeros(5),
        advantages=advantages2,
        source_policy=source2,
        clip_epsilon=0.1,
        reduction="source",
    )
    flat2 = _call_surrogate(
        new_logprobs=torch.zeros(5),
        behaviour_logprobs=torch.zeros(5),
        target_old_logprobs=torch.zeros(5),
        advantages=advantages2,
        source_policy=source2,
        clip_epsilon=0.1,
        reduction="mean",
    )
    ps2 = float(per_source2.reshape(-1).mean() if per_source2.ndim else per_source2)
    fl2 = float(flat2.reshape(-1).mean() if flat2.ndim else flat2)
    assert math.isclose(ps2, 2.5 * sign, rel_tol=1e-5)  # (1 + 4) / 2
    assert math.isclose(fl2, 3.4 * sign, rel_tol=1e-5)  # (1 + 4*4) / 5


# ---------------------------------------------------------------------------
# full Eq. 3 loss dict + Eq. 4 combination
# ---------------------------------------------------------------------------
def test_off_policy_loss_negative_of_surrogate_by_default():
    advantages = torch.ones(6)
    logprobs = torch.zeros(6)
    result = off_policy_loss(
        new_logprobs=logprobs,
        behaviour_logprobs=logprobs.clone(),
        target_old_logprobs=logprobs.clone(),
        advantages=advantages,
        clip_epsilon=0.1,
    )
    assert isinstance(result, dict)
    assert "off_policy_loss" in result
    loss = result["off_policy_loss"]
    assert torch.is_tensor(loss)
    assert float(loss.reshape(-1)[0]) < 0.0  # minimised form is negative


def test_off_policy_loss_on_policy_reduction_is_zero_gradient_free():
    """With pi_i,old == pi_j the surrogate is mean(A); its negative is -mean(A)."""
    advantages = torch.tensor([1.0, -1.0, 2.0, 0.0])
    logprobs = torch.zeros(4)
    result = off_policy_loss(
        new_logprobs=logprobs,
        behaviour_logprobs=logprobs.clone(),
        target_old_logprobs=logprobs.clone(),
        advantages=advantages,
        clip_epsilon=0.1,
    )
    scalar = float(result["off_policy_loss"].reshape(-1).mean())
    sign = _sign_convention()
    assert math.isclose(scalar, -sign * float(advantages.mean()), abs_tol=1e-5)


def test_combine_on_off_loss_applies_lambda():
    on = torch.tensor(3.0)
    off = torch.tensor(2.0)
    combined = combine_on_off_loss(on, off, lam=1.0)
    assert math.isclose(float(combined), 5.0, rel_tol=1e-6)
    combined_half = combine_on_off_loss(on, off, lam=0.5)
    assert math.isclose(float(combined_half), 4.0, rel_tol=1e-6)
    combined_zero = combine_on_off_loss(on, off, lam=0.0)
    assert math.isclose(float(combined_zero), 3.0, rel_tol=1e-6)


def test_combine_on_off_loss_accepts_sequence_of_terms():
    on = torch.tensor(1.0)
    terms = [torch.tensor(2.0), torch.tensor(4.0)]
    combined = combine_on_off_loss(on, terms, lam=1.0)
    assert math.isclose(float(combined), 4.0, rel_tol=1e-6)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
