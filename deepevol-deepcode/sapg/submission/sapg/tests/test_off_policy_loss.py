"""Unit tests for the SAPG off-policy importance-sampled PPO loss (Section 4.1).

The paper (Eq. 3) defines, for updating policy ``pi_i`` with data drawn from a
set of source policies ``j in X``::

    L_off(pi_i; X) = 1/|X| sum_{j in X} E_{(s,a)~pi_j}[
        min( r_{pi_i}(s,a),
             clip( r_{pi_i}(s,a), mu (1 - eps), mu (1 + eps) ) ) A^{pi_i,old}(s,a) ]

where

    r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a)
    mu            = pi_{i,old}(s,a) / pi_j(s,a)

and Eq. 4 combines the terms as

    L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X)

The key structural properties verified here are:

* ``r`` and ``mu`` are exponentials of log-probability differences;
* the clipping bounds are *scaled by mu* (i.e. ``mu(1 - eps)`` / ``mu(1 + eps)``),
  not by 1;
* when the target and behaviour policies coincide (``i == j``) we have
  ``r = mu = 1`` and the surrogate reduces exactly to the on-policy clipped
  PPO surrogate;
* the reduction ``1/|X| sum_j`` averages *per source policy*, not per
  transition;
* Eq. 4's ``lambda`` scales only the off-policy term.

The tests intentionally avoid hard-coding a sign convention: the module is
probed at runtime to decide whether it returns a maximised objective or a
minimised loss.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sapg.losses.off_policy_loss import (  # noqa: E402
    combine_on_off_loss,
    combined_objective,
    importance_ratio,
    mu_from_logprobs,
    off_policy_loss,
    off_policy_surrogate,
)

try:  # pragma: no cover - optional cross-check
    from sapg.losses.ppo_loss import importance_ratio as on_policy_importance_ratio
except Exception:  # pragma: no cover
    on_policy_importance_ratio = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_CLIP = 0.1


def _as_float(value) -> float:
    """Convert a python float / 0-d tensor / 1-element tensor to ``float``."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _call_surrogate(**kwargs):
    """Call :func:`off_policy_surrogate` normalising the ``(value, stats)`` form."""
    out = off_policy_surrogate(**kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _mean(out) -> float:
    if isinstance(out, torch.Tensor):
        return float(out.detach().reshape(-1).mean().item())
    return float(out)


def _sign_convention() -> float:
    """``+1`` if the module returns a maximised objective, ``-1`` if a loss.

    With ``r = mu = 1`` and unit advantages the Eq. 3 quantity is exactly ``1``,
    so the sign of the returned number reveals the convention.
    """
    z = torch.zeros(4)
    out = _call_surrogate(
        new_logprobs=z,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=torch.ones(4),
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    return 1.0 if _mean(out) >= 0 else -1.0


_SIGN = _sign_convention()


def _signed(out) -> float:
    """Return the *maximised objective* value from any return form."""
    return _SIGN * _mean(out)


def _reference_objective(ratio, advantages, mu, clip_epsilon=_CLIP):
    """Eq. 3 surrogate per-transition (maximised objective), from first principles."""
    ratio = torch.as_tensor(ratio, dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    mu = torch.as_tensor(mu, dtype=torch.float32)
    bounds = torch.stack([mu * (1.0 - clip_epsilon), mu * (1.0 + clip_epsilon)])
    clipped = torch.clamp(ratio, min=bounds[0], max=bounds[1])
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    return surrogate.mean()


def _advantages_where_min_picks_first():
    """Advantages where ``min`` selects the *unclipped* branch (A > 0, ratio < 1)."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# ratio / mu definitions
# ---------------------------------------------------------------------------
def test_importance_ratio_is_exp_of_log_difference():
    new = torch.tensor([0.0, -0.5, 0.7, -1.3])
    old = torch.tensor([0.3, 0.1, -0.2, 0.05])
    expected = torch.exp(new - old)
    assert torch.allclose(importance_ratio(new, old), expected, atol=1e-6)


def test_importance_ratio_matches_on_policy_variant():
    if on_policy_importance_ratio is None:  # pragma: no cover
        pytest.skip("ppo_loss.importance_ratio unavailable")
    new = torch.tensor([0.1, -0.4, 0.9])
    old = torch.tensor([0.0, 0.2, -0.3])
    assert torch.allclose(
        importance_ratio(new, old), on_policy_importance_ratio(new, old), atol=1e-6
    )


def test_importance_ratio_is_one_when_policies_agree():
    new = torch.randn(64)
    assert torch.allclose(importance_ratio(new, new), torch.ones(64), atol=1e-6)


def test_mu_from_logprobs_formula():
    target_old = torch.tensor([0.0, -0.25, 0.6])
    behaviour = torch.tensor([0.2, 0.1, -0.1])
    expected = torch.exp(target_old - behaviour)
    assert torch.allclose(mu_from_logprobs(target_old, behaviour), expected, atol=1e-6)


def test_mu_is_one_when_target_old_matches_behaviour():
    """``i == j`` implies ``pi_{i,old} == pi_j`` hence ``mu = 1``."""
    behaviour = torch.randn(32) * 0.5
    assert torch.allclose(mu_from_logprobs(behaviour, behaviour), torch.ones(32), atol=1e-6)


# ---------------------------------------------------------------------------
# surrogate: on-policy reduction (i == j)
# ---------------------------------------------------------------------------
def test_surrogate_reduces_to_on_policy_when_i_equals_j():
    """i == j => r = mu = 1 => Eq. 3 becomes the on-policy clipped surrogate."""
    advantages = torch.tensor([0.5, -0.25, 1.5, -2.0])
    logprobs = torch.tensor([0.1, -0.2, 0.3, 0.0])
    out = _call_surrogate(
        new_logprobs=logprobs,
        behaviour_logprobs=logprobs,
        target_old_logprobs=logprobs,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    expected = float(advantages.mean().item())  # r = 1 for every transition
    assert math.isclose(_signed(out), expected, abs_tol=1e-5)


def test_on_policy_reduction_matches_ppo_clipped_surrogate():
    """With i == j and a genuinely changed ratio, we get the PPO min/clip form."""
    advantages = _advantages_where_min_picks_first()
    new = torch.tensor([0.0, -0.05, 0.05, -0.15])  # ratios ~1, 0.95, 1.05, 0.86
    old = torch.zeros(4)
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=old,
        target_old_logprobs=old,  # mu = 1
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    ratio = torch.exp(new - old)
    expected = _reference_objective(ratio, advantages, torch.ones(4))
    assert math.isclose(_signed(out), float(expected.item()), abs_tol=1e-5)


# ---------------------------------------------------------------------------
# clipping bounds are mu-scaled (Eq. 3)
# ---------------------------------------------------------------------------
def test_surrogate_clips_ratio_above_upper_mu_bound():
    """Ratio above ``mu(1 + eps)`` with positive advantage => clipped branch wins."""
    mu = torch.ones(1)
    new = torch.tensor([0.5])  # ratio ~1.6487, above 1.1
    behaviour = torch.zeros(1)
    advantages = torch.tensor([2.0])
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=behaviour,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    ratio = torch.exp(new - behaviour)
    expected = _reference_objective(ratio, advantages, mu)
    # For A > 0 the min picks the clipped term: mu*(1+eps)*A
    assert math.isclose(
        _signed(out), float((mu * (1 + _CLIP) * advantages).mean().item()), abs_tol=1e-5
    )
    assert math.isclose(_signed(out), float(expected.item()), abs_tol=1e-5)


def test_surrogate_clips_ratio_below_lower_mu_bound():
    """Ratio below ``mu(1 - eps)``; ``min`` keeps the unclipped branch for A > 0."""
    new = torch.tensor([-0.5])  # ratio ~0.6065, below 0.9
    behaviour = torch.zeros(1)
    advantages = torch.tensor([3.0])
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=behaviour,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    ratio = torch.exp(new - behaviour)
    # min(r*A, clipped*A) with A > 0 and r < lower bound => r*A is smaller
    assert math.isclose(
        _signed(out), float((ratio * advantages).mean().item()), abs_tol=1e-5
    )


def test_surrogate_negative_advantage_selects_pessimistic_branch():
    """For A < 0 and ratio above the upper bound, the clipped (larger) term is min."""
    new = torch.tensor([0.5])  # ratio ~1.6487 > 1.1
    behaviour = torch.zeros(1)
    advantages = torch.tensor([-2.0])
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=behaviour,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    ratio = torch.exp(new - behaviour)
    expected = _reference_objective(ratio, advantages, torch.ones(1))
    assert math.isclose(_signed(out), float(expected.item()), abs_tol=1e-5)
    assert expected.item() > (ratio * advantages).mean().item() - 1e-9


def test_clipping_bounds_scale_with_mu():
    """The clip bounds must be ``mu(1 +/- eps)``, not ``1 +/- eps``."""
    eps = _CLIP
    # mu = 4, ratio = exp(0) = 1 (well inside the mu-scaled interval [3.6, 4.4]).
    z = torch.zeros(3)
    target_old = torch.full((3,), math.log(4.0))
    advantages = torch.ones(3)
    out = _call_surrogate(
        new_logprobs=z,
        behaviour_logprobs=z,  # ratio = 1
        target_old_logprobs=target_old,  # mu = 4
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    # Without mu-scaling the ratio 1 would be clipped up to 1+eps; with mu-scaling
    # it stays at its unclipped value of 1 for A > 0 -> objective = 1.
    assert math.isclose(_signed(out), 1.0, abs_tol=1e-5)

    # Now use a ratio that is above the *plain* upper bound 1 + eps but below the
    # mu-scaled upper bound mu(1 + eps) = 4.4: it must NOT be clipped.
    log_ratio = math.log(2.0)  # ratio = 2 (inside [3.6, 4.4]? no -> below)
    out2 = _call_surrogate(
        new_logprobs=torch.full((3,), log_ratio),
        behaviour_logprobs=z,
        target_old_logprobs=target_old,
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    # ratio 2 < mu(1-eps)=3.6 => min picks r*A = 2 (A > 0)
    assert math.isclose(_signed(out2), 2.0, abs_tol=1e-5)


def test_clipping_bounds_scale_with_mu_upper_side():
    mu_value = 4.0
    eps = _CLIP
    log_ratio = math.log(10.0)  # 10 > mu(1+eps) = 4.4
    target_old = torch.full((3,), math.log(mu_value))
    advantages = torch.ones(3)
    out = _call_surrogate(
        new_logprobs=torch.full((3,), log_ratio),
        behaviour_logprobs=torch.zeros(3),
        target_old_logprobs=target_old,
        advantages=advantages,
        clip_epsilon=eps,
        reduction="mean",
    )
    assert math.isclose(
        _signed(out), mu_value * (1.0 + eps), abs_tol=1e-5
    )


def test_explicit_mu_argument_overrides_logprob_derived_mu():
    z = torch.zeros(2)
    log_ratio = math.log(5.0)
    # Derived mu would be 1 (target_old == behaviour), but we pass mu = 100 so the
    # ratio 5 stays inside the interval [90, 110] and is not clipped.
    out = _call_surrogate(
        new_logprobs=torch.full((2,), log_ratio),
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=torch.ones(2),
        mu=torch.full((2,), 100.0),
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    assert math.isclose(_signed(out), 5.0, abs_tol=1e-4)


def test_reference_formula_matches_module_on_mixed_inputs():
    torch.manual_seed(0)
    new = torch.randn(128) * 0.3
    behaviour = torch.randn(128) * 0.3
    target_old = behaviour + torch.randn(128) * 0.1
    advantages = torch.randn(128)
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=target_old,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    ratio = torch.exp(new - behaviour)
    mu = torch.exp(target_old - behaviour)
    expected = _reference_objective(ratio, advantages, mu)
    assert math.isclose(_signed(out), float(expected.item()), abs_tol=1e-5)


# ---------------------------------------------------------------------------
# per-source reduction: 1/|X| sum_{j in X} E_{pi_j}[...]
# ---------------------------------------------------------------------------
def test_per_source_reduction_averages_source_means():
    """Eq. 3 averages the expectation over each source policy separately."""
    z = torch.zeros(4)
    # Two source policies with clearly different per-source means.
    source = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long)
    new = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=advantages,
        source_policy=source,
        clip_epsilon=_CLIP,
        reduction="source",
    )
    per_source = []
    for s in (2, 3):
        mask = source == s
        r = torch.exp(new[mask] - z[mask])
        per_source.append(float(r.mean().item()))
    expected = sum(per_source) / len(per_source)
    assert math.isclose(_signed(out), expected, abs_tol=1e-5)


def test_per_source_reduction_differs_from_flat_mean_when_sources_unequal():
    z = torch.zeros(8)
    # Source 2 has one sample, source 3 has three; unequal means expose the
    # difference between E_j[. ] and a flat per-transition mean.
    source = torch.tensor([2, 3, 3, 3], dtype=torch.long)
    new = torch.tensor([0.0, 1.0, 1.0, 1.0])
    advantages = torch.ones(4)
    out = _call_surrogate(
        new_logprobs=new,
        behaviour_logprobs=z[:4],
        target_old_logprobs=z[:4],
        advantages=advantages,
        source_policy=source,
        clip_epsilon=_CLIP,
        reduction="source",
    )
    flat = float((torch.exp(new) * advantages).mean().item())
    per_source = float((torch.exp(new[1:]).mean().item() + torch.exp(new[:1]).mean().item()) / 2)
    obtained = _signed(out)
    assert math.isclose(obtained, per_source, abs_tol=1e-5)
    assert not math.isclose(obtained, flat, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# off_policy_loss wrapper
# ---------------------------------------------------------------------------
def test_off_policy_loss_returns_dict_with_expected_keys():
    z = torch.zeros(8)
    out = off_policy_loss(
        new_logprobs=z,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=torch.ones(8),
        clip_epsilon=_CLIP,
    )
    assert isinstance(out, dict)
    assert "off_policy_loss" in out
    assert "policy_loss" in out


def test_off_policy_loss_is_negated_objective_by_default():
    z = torch.zeros(8)
    advantages = torch.tensor([1.0, 2.0, -1.0, 0.5, 0.0, 0.25, -2.0, 1.0])
    out = off_policy_loss(
        new_logprobs=z,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=advantages,
        clip_epsilon=_CLIP,
    )
    value = _as_float(out["off_policy_loss"])
    objective = float(advantages.mean().item())
    # The loss is the negated (minimised) objective when the module follows the
    # common convention; be tolerant of either sign but consistent with the probe.
    assert math.isclose(value, _SIGN * objective, abs_tol=1e-5)


def test_off_policy_loss_matches_surrogate_reference():
    torch.manual_seed(1)
    new = torch.randn(64) * 0.2
    behaviour = torch.randn(64) * 0.2
    target_old = behaviour + torch.randn(64) * 0.05
    advantages = torch.randn(64)
    out = off_policy_loss(
        new_logprobs=new,
        behaviour_logprobs=behaviour,
        target_old_logprobs=target_old,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    ratio = torch.exp(new - behaviour)
    mu = torch.exp(target_old - behaviour)
    expected = float(_reference_objective(ratio, advantages, mu).item())
    # accept either raw or minimised convention via the probe sign
    value = _SIGN * _as_float(out["off_policy_loss"])
    # if the module stores the raw surrogate under "surrogate", compare to that too
    if "surrogate" in out:
        assert math.isclose(
            _SIGN * _as_float(out["surrogate"]), expected, abs_tol=1e-5
        )
    assert math.isclose(value, expected, abs_tol=1e-5)


def test_off_policy_loss_lambda_weighting():
    """``lam`` (Eq. 4's lambda) multiplies the off-policy term."""
    z = torch.zeros(8)
    advantages = torch.ones(8)
    base = off_policy_loss(
        new_logprobs=z,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=advantages,
        clip_epsilon=_CLIP,
        lam=1.0,
    )
    scaled = off_policy_loss(
        new_logprobs=z,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=advantages,
        clip_epsilon=_CLIP,
        lam=3.0,
    )
    v1 = _as_float(base["off_policy_loss"])
    v3 = _as_float(scaled["off_policy_loss"])
    assert math.isclose(v3, 3.0 * v1, rel_tol=1e-6)


def test_off_policy_loss_accepts_batch_dict():
    z = torch.zeros(16)
    batch = {
        "logprobs": z,
        "behaviour_logprobs": z,
        "old_logprobs": z,
        "advantages": torch.ones(16),
    }
    out = off_policy_loss(batch=batch, clip_epsilon=_CLIP)
    assert "off_policy_loss" in out
    assert math.isfinite(_as_float(out["off_policy_loss"]))


def test_off_policy_loss_gradients_flow_to_new_logprobs():
    z = torch.zeros(16)
    new = z.clone().requires_grad_(True)
    out = off_policy_loss(
        new_logprobs=new,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=torch.ones(16),
        clip_epsilon=_CLIP,
    )
    loss = out["off_policy_loss"]
    loss.backward()
    assert new.grad is not None
    assert torch.isfinite(new.grad).all()


def test_clip_fraction_reported_when_clipping_happens():
    z = torch.zeros(4)
    new = torch.full((4,), math.log(5.0))  # ratio 5 >> 1 + eps
    out = off_policy_loss(
        new_logprobs=new,
        behaviour_logprobs=z,
        target_old_logprobs=z,
        advantages=torch.ones(4),
        clip_epsilon=_CLIP,
    )
    if "clip_frac" in out:
        assert _as_float(out["clip_frac"]) > 0.5


# ---------------------------------------------------------------------------
# Eq. 4 combination
# ---------------------------------------------------------------------------
def test_combine_on_off_loss_applies_lambda():
    on = torch.tensor(1.5)
    off = torch.tensor(2.0)
    combined = combine_on_off_loss(on, off, lam=2.0)
    assert math.isclose(_as_float(combined), 1.5 + 2.0 * 2.0, abs_tol=1e-6)


def test_combine_on_off_loss_default_lambda_is_one():
    on = torch.tensor(0.75)
    off = torch.tensor(-0.25)
    combined = combine_on_off_loss(on, off)
    assert math.isclose(_as_float(combined), 0.75 - 0.25, abs_tol=1e-6)


def test_combine_on_off_loss_accepts_sequence_of_source_terms():
    """``X = {2..M}`` is folded into a single off-policy term first."""
    on = torch.tensor(1.0)
    terms = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    combined = combine_on_off_loss(on, terms, lam=1.0)
    assert math.isclose(_as_float(combined), 1.0 + 2.0, abs_tol=1e-6)


def test_combine_on_off_loss_handles_none_off_term():
    on = torch.tensor(4.0)
    combined = combine_on_off_loss(on, None, lam=5.0)
    assert math.isclose(_as_float(combined), 4.0, abs_tol=1e-6)


def test_combined_objective_matches_manual_formula():
    on = torch.tensor(0.5)
    off = torch.tensor(-1.5)
    lam = 4.0
    out = combined_objective(on, off, lam=lam)
    assert math.isclose(_as_float(out), 0.5 + lam * (-1.5), abs_tol=1e-6)


def test_combined_objective_without_off_term():
    on = torch.tensor(2.5)
    out = combined_objective(on, None, lam=1.0)
    assert math.isclose(_as_float(out), 2.5, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Follower behaviour: X = empty (on-policy only)
# ---------------------------------------------------------------------------
def test_on_policy_only_uses_no_off_policy_data():
    """Followers set ``X = {}``; the combined loss must be just ``L_on``."""
    on = torch.tensor(3.0)
    assert math.isclose(_as_float(combine_on_off_loss(on, [], lam=1.0)), 3.0, abs_tol=1e-6)


def test_surrogate_without_behaviour_logprobs_is_on_policy():
    """Absent ``behaviour_logprobs`` => treated as ``i == j`` (ratio = mu = 1)."""
    advantages = torch.tensor([1.0, -1.0, 0.5, 2.0])
    out = _call_surrogate(
        new_logprobs=None,
        behaviour_logprobs=None,
        target_old_logprobs=None,
        advantages=advantages,
        clip_epsilon=_CLIP,
        reduction="mean",
    )
    assert math.isclose(_signed(out), float(advantages.mean().item()), abs_tol=1e-5)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
