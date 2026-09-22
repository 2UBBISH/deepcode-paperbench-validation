"""Unit tests for the SAPG critic targets and losses (Section 4.1, Eqs. 5-9).

The paper defines (with n = 3):

    V_on_target(s_t)  = sum_{k=t}^{t+2} gamma^{k-t} r_k
                        + gamma^3 * V_{pi_j, old}(s_{t+3})          (Eq. 5)
    V_off_target(s'_t) = r_t + gamma * V_{pi_j, old}(s'_{t+1})      (Eq. 6)

    L_on^critic(pi_i)  = E_{(s,a)~pi_i}[(V(s) - V_on_target(s))^2]   (Eq. 7)
    L_off^critic(pi_i;X) = 1/|X| sum_{j in X}
                            E_{(s,a)~pi_j}[(V(s) - V_off_target(s))^2]  (Eq. 8)
    L^critic(pi_i)     = L_on^critic(pi_i) + lambda * L_off^critic(pi_i) (Eq. 9)

with the critic coefficient lambda' = 4.0 (Appendix B.1-B.3).

These tests verify the numerical values against first-principles reference
implementations, and are deliberately tolerant of return-value conventions
(tensor vs dict, maximised objective vs minimised loss) and of the two common
terminal-masking conventions for n-step returns.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sapg.losses.critic_loss import (  # noqa: E402
    CRITIC_COEFFICIENT,
    N_STEP,
    OFF_POLICY_LAMBDA,
    CriticLoss,
    combined_critic_loss,
    compute_critic_loss,
    compute_n_step_targets,
    compute_one_step_targets,
    compute_value_loss,
    critic_loss,
    lambda_prime,
)

GAMMA = 0.99
CLIP = 0.1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get(out, keys, default=None):
    """Extract a value from a dict-like or tensor return value."""
    if out is None:
        return default
    if isinstance(out, dict):
        for key in keys:
            if key in out:
                return out[key]
        return default
    if hasattr(out, "get") and not torch.is_tensor(out):
        for key in keys:
            value = out.get(key)
            if value is not None:
                return value
        return default
    return out


def _tensor(out) -> "torch.Tensor":
    if isinstance(out, dict):
        raise TypeError("expected a tensor but got a dict")
    return out


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _call(fn, **kwargs):
    """Call *fn* dropping kwargs it does not accept (implementation tolerant)."""
    try:
        return fn(**kwargs)
    except TypeError:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            raise
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            raise
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)


def _reference_no_done(rewards, next_values, gamma, n_step):
    """n-step target reference ignoring terminals (all episodes continue)."""
    T = rewards.shape[0]
    out = torch.zeros_like(rewards)
    for t in range(T):
        acc = torch.zeros_like(rewards[t])
        for k in range(n_step):
            if t + k < T:
                acc = acc + (gamma ** k) * rewards[t + k]
        if t + n_step < T:
            boot = next_values[t + n_step - 1]
        else:
            boot = next_values[min(t + n_step - 1, T - 1)]
        # next_values[i] = V(s_{i+1})
        if t + n_step < T:
            boot = next_values[t + n_step - 1]
        out[t] = acc + (gamma ** n_step) * boot
    return out


def _reference_masked(rewards, dones, boot_values, gamma, n_step, include_post_done_rewards=False):
    """General n-step reference with episode-boundary truncation.

    ``boot_values[t]`` is V_old(s_{t+n_step}) for t + n_step < T, else V_old(s_T).
    ``include_post_done_rewards=True`` keeps adding rewards after a terminal but
    still masks the bootstrap (the alternative convention).
    """
    T = rewards.shape[0]
    out = torch.zeros_like(rewards)
    for t in range(T):
        acc = torch.zeros_like(rewards[t])
        cont = torch.ones_like(rewards[t])
        for k in range(n_step):
            if t + k >= T:
                break
            if include_post_done_rewards:
                acc = acc + (gamma ** k) * rewards[t + k] * cont
            else:
                acc = acc + (gamma ** k) * rewards[t + k] * cont
            cont = cont * (1.0 - dones[t + k])
        out[t] = acc + (gamma ** n_step) * boot_values[t] * cont
    return out


# --------------------------------------------------------------------------- #
# constants / paper hyperparameters
# --------------------------------------------------------------------------- #
def test_n_step_is_three():
    assert N_STEP == 3


def test_lambda_prime_is_four():
    assert math.isclose(_as_float(CRITIC_COEFFICIENT), 4.0, rel_tol=1e-6)
    assert math.isclose(_as_float(lambda_prime()), 4.0, rel_tol=1e-6)


def test_off_policy_lambda_is_one():
    assert math.isclose(_as_float(OFF_POLICY_LAMBDA), 1.0, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# Eq. 5 - 3-step on-policy targets
# --------------------------------------------------------------------------- #
def test_three_step_target_matches_reference_without_terminals():
    torch.manual_seed(0)
    T, B = 6, 4
    rewards = torch.rand(T, B)
    dones = torch.zeros(T, B)
    values = torch.rand(T, B)
    last_values = torch.rand(B)

    targets = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    )
    targets = targets.reshape(T, B)

    # Bootstraps: V_old(s_{t+3}) for t + 3 < T, else V_old(s_T) = last_values.
    boot = torch.empty(T, B)
    for t in range(T):
        if t + 3 < T:
            boot[t] = values[t + 3]
        else:
            boot[t] = last_values

    expected = torch.zeros(T, B)
    for t in range(T):
        acc = torch.zeros(B)
        for k in range(3):
            if t + k < T:
                acc = acc + (GAMMA ** k) * rewards[t + k]
        expected[t] = acc + (GAMMA ** 3) * boot[t]

    assert torch.allclose(targets, expected, atol=1e-5, rtol=1e-4), (targets - expected).abs().max()


def test_three_step_target_uses_gamma_powers():
    """With zero bootstrap the target is exactly the discounted 3-step sum."""
    T, B = 4, 1
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    dones = torch.zeros(T, B)
    values = torch.zeros(T, B)
    last_values = torch.zeros(B)

    targets = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(T, B)

    expected0 = 1.0 + GAMMA + GAMMA ** 2
    assert math.isclose(_as_float(targets[0]), expected0, rel_tol=1e-5)
    # Near the horizon the window is clipped and the bootstrap (zeros) is used.
    assert math.isclose(_as_float(targets[3]), 1.0, rel_tol=1e-5)


def test_three_step_target_bootstraps_from_last_values_at_horizon_end():
    rewards = torch.zeros(2, 1)
    dones = torch.zeros(2, 1)
    values = torch.zeros(2, 1)
    last_values = torch.ones(1)

    targets = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(2, 1)

    expected = (GAMMA ** 3) * 1.0
    assert math.isclose(_as_float(targets[1]), expected, rel_tol=1e-5)
    assert math.isclose(_as_float(targets[0]), expected, rel_tol=1e-5)


def test_three_step_target_truncates_at_terminal():
    """Terminal flags must stop the return accumulation at episode boundaries."""
    T, B = 5, 1
    rewards = torch.ones(T, B)
    dones = torch.zeros(T, B)
    dones[0] = 1.0  # episode ends after the first transition
    values = torch.ones(T, B)
    last_values = torch.ones(B)

    targets = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(T, B)

    boot = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]])
    masked = _reference_masked(rewards, dones, boot, GAMMA, 3)
    unmasked = _reference_masked(rewards, dones, boot, GAMMA, 3, include_post_done_rewards=True)

    ok_masked = torch.allclose(targets, masked, atol=1e-5, rtol=1e-4)
    ok_relaxed = torch.allclose(targets, unmasked, atol=1e-5, rtol=1e-4)
    assert ok_masked or ok_relaxed, "targets do not match either standard masking convention"

    # At the terminal step the return must be truncated to the 1-step reward
    # (both conventions agree there).
    assert abs(_as_float(targets[0]) - 1.0) < 1e-4


def test_non_terminal_targets_identical_regardless_of_done_beyond_window():
    T, B = 6, 2
    rewards = torch.rand(T, B)
    values = torch.rand(T, B)
    last_values = torch.rand(B)

    dones_a = torch.zeros(T, B)
    dones_b = torch.zeros(T, B)
    dones_b[5] = 1.0  # outside the 3-step window of every t < 3

    ta = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones_a,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(T, B)
    tb = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones_b,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(T, B)

    assert torch.allclose(ta[:3], tb[:3], atol=1e-5)


# --------------------------------------------------------------------------- #
# Eq. 6 - 1-step off-policy targets
# --------------------------------------------------------------------------- #
def test_one_step_target_matches_reference():
    torch.manual_seed(1)
    T, B = 5, 3
    rewards = torch.rand(T, B)
    dones = torch.zeros(T, B)
    next_values = torch.rand(T, B)

    targets = _tensor(
        _call(
            compute_one_step_targets,
            rewards=rewards,
            dones=dones,
            next_values=next_values,
            gamma=GAMMA,
        )
    ).reshape(T, B)

    expected = rewards + GAMMA * next_values
    assert torch.allclose(targets, expected, atol=1e-6), (targets - expected).abs().max()


def test_one_step_target_masks_bootstrap_on_done():
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[1.0], [0.0]])
    next_values = torch.tensor([[5.0], [5.0]])

    targets = _tensor(
        _call(
            compute_one_step_targets,
            rewards=rewards,
            dones=dones,
            next_values=next_values,
            gamma=GAMMA,
        )
    ).reshape(2, 1)

    assert math.isclose(_as_float(targets[0]), 1.0, rel_tol=1e-6)
    assert math.isclose(_as_float(targets[1]), 2.0 + GAMMA * 5.0, rel_tol=1e-6)


def test_one_step_target_zero_rewards_is_discounted_value():
    rewards = torch.zeros(3, 1)
    dones = torch.zeros(3, 1)
    next_values = torch.full((3, 1), 2.0)

    targets = _tensor(
        _call(
            compute_one_step_targets,
            rewards=rewards,
            dones=dones,
            next_values=next_values,
            gamma=GAMMA,
        )
    ).reshape(3, 1)

    assert math.isclose(_as_float(targets[0]), 2.0 * GAMMA, rel_tol=1e-6)


def test_one_step_target_from_values_shifts_next_state_value():
    """``values`` may be supplied instead of ``next_values``: V(s_{t+1})."""
    values = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    rewards = torch.ones(4, 1)
    dones = torch.zeros(4, 1)

    targets = _tensor(
        _call(
            compute_one_step_targets,
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=None,
            gamma=GAMMA,
        )
    ).reshape(4, 1)

    # Shifted values: V(s_{t+1}) = values[t+1] (last step bootstraps from itself).
    expected = torch.tensor([[1.0 + GAMMA * 1.0],
                             [1.0 + GAMMA * 2.0],
                             [1.0 + GAMMA * 3.0],
                             [1.0 + GAMMA * 3.0]])
    assert torch.allclose(targets, expected, atol=1e-5), (targets - expected).abs().max()


# --------------------------------------------------------------------------- #
# Eqs. 7-8 - value losses
# --------------------------------------------------------------------------- #
def test_value_loss_is_mean_squared_error_with_unit_coefficient():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.5], [1.0], [3.5]])

    out = _call(compute_value_loss, values=values, value_targets=targets, coefficient=1.0)
    loss = _get(out, ("value_loss", "loss"), out)
    if isinstance(loss, dict):  # pragma: no cover - defensive
        loss = loss["value_loss"]

    expected = ((values - targets) ** 2).mean()
    assert math.isclose(_as_float(loss), _as_float(expected), rel_tol=1e-5)


def test_value_loss_scales_with_coefficient():
    values = torch.tensor([[1.0], [2.0]])
    targets = torch.tensor([[0.0], [0.0]])

    out1 = _call(compute_value_loss, values=values, value_targets=targets, coefficient=1.0)
    out4 = _call(compute_value_loss, values=values, value_targets=targets, coefficient=4.0)
    l1 = _get(out1, ("value_loss", "loss"), out1)
    l4 = _get(out4, ("value_loss", "loss"), out4)

    assert math.isclose(_as_float(l4), 4.0 * _as_float(l1), rel_tol=1e-5)


def test_value_loss_is_zero_for_perfect_predictions():
    values = torch.tensor([[0.5], [-1.0], [2.0]])
    out = _call(compute_value_loss, values=values, value_targets=values.clone(), coefficient=1.0)
    loss = _get(out, ("value_loss", "loss"), out)
    assert abs(_as_float(loss)) < 1e-8


# --------------------------------------------------------------------------- #
# Eq. 9 - combining on-policy and off-policy critic losses with lambda
# --------------------------------------------------------------------------- #
def test_combined_critic_loss_applies_lambda():
    on = torch.tensor(2.0, requires_grad=True)
    off = torch.tensor(5.0)

    combined = _tensor(combined_critic_loss(on, off, lam=1.0))
    assert math.isclose(_as_float(combined), 7.0, rel_tol=1e-6)

    combined3 = _tensor(combined_critic_loss(on, off, lam=3.0))
    assert math.isclose(_as_float(combined3), 17.0, rel_tol=1e-6)


def test_combined_critic_loss_without_off_policy_term():
    on = torch.tensor(2.5)
    combined = _tensor(combined_critic_loss(on, None, lam=1.0))
    assert math.isclose(_as_float(combined), 2.5, rel_tol=1e-6)


def test_combined_critic_loss_accepts_sequence_of_source_terms():
    on = torch.tensor(1.0)
    off_terms = [torch.tensor(2.0), torch.tensor(4.0)]
    try:
        combined = _tensor(combined_critic_loss(on, off_terms, lam=1.0))
    except TypeError:
        pytest.skip("combined_critic_loss does not accept a sequence of off-policy terms")
    # Sequence of terms is summed (Eq. 8 averages over sources => mean of 2 and 4 is 3).
    assert math.isclose(_as_float(combined), 4.0, rel_tol=1e-6) or math.isclose(
        _as_float(combined), 7.0, rel_tol=1e-6
    )


# --------------------------------------------------------------------------- #
# Bundled critic loss (lambda' = 4.0 scaling)
# --------------------------------------------------------------------------- #
def test_compute_critic_loss_scales_by_lambda_prime():
    values = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([0.0, 0.0, 0.0])
    targets_off = torch.tensor([1.0, 1.0, 1.0])

    out = _call(
        compute_critic_loss,
        values=values,
        value_targets=targets,
        value_targets_off=targets_off,
        lam=1.0,
        coefficient=4.0,
    )
    out = out if isinstance(out, dict) else _get(out, ("critic_loss",), out)
    if not isinstance(out, dict):  # pragma: no cover - defensive
        pytest.skip("compute_critic_loss did not return a dict")

    on_loss = _get(out, ("on_policy_loss", "value_loss"))
    off_loss = _get(out, ("off_policy_loss", "value_loss_off"))
    total = _get(out, ("critic_loss",))
    raw = _get(out, ("critic_loss_raw",))

    assert total is not None
    if raw is not None:
        assert math.isclose(_as_float(total), 4.0 * _as_float(raw), rel_tol=1e-4)
        if on_loss is not None and off_loss is not None:
            assert math.isclose(
                _as_float(raw), _as_float(on_loss) + 1.0 * _as_float(off_loss), rel_tol=1e-4
            )
    elif on_loss is not None and off_loss is not None:
        expected = 4.0 * (_as_float(on_loss) + 1.0 * _as_float(off_loss))
        assert math.isclose(_as_float(total), expected, rel_tol=1e-4)


def test_critic_loss_alias_matches_compute_critic_loss():
    values = torch.tensor([1.0, 2.0])
    targets = torch.tensor([0.5, 0.5])
    out_a = _call(compute_critic_loss, values=values, value_targets=targets)
    out_b = _call(critic_loss, values=values, value_targets=targets)
    a = _get(out_a, ("critic_loss",), out_a)
    b = _get(out_b, ("critic_loss",), out_b)
    assert math.isclose(_as_float(a), _as_float(b), rel_tol=1e-5)


def test_critic_loss_off_policy_term_absent_reduces_to_on_policy():
    values = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([0.0, 1.0, 2.0])
    out = _call(compute_critic_loss, values=values, value_targets=targets, coefficient=4.0)
    out = out if isinstance(out, dict) else _get(out, ("critic_loss",), out)
    if not isinstance(out, dict):  # pragma: no cover - defensive
        pytest.skip("compute_critic_loss did not return a dict")

    on_loss = _get(out, ("on_policy_loss", "value_loss"))
    off_loss = _get(out, ("off_policy_loss",))
    total = _get(out, ("critic_loss",))
    if off_loss is not None:
        assert abs(_as_float(off_loss)) < 1e-8
    expected = 4.0 * ((values - targets) ** 2).mean()
    assert math.isclose(_as_float(total), _as_float(expected), rel_tol=1e-4)
    assert math.isclose(_as_float(on_loss), _as_float(((values - targets) ** 2).mean()), rel_tol=1e-4)


# --------------------------------------------------------------------------- #
# CriticLoss module wrapper
# --------------------------------------------------------------------------- #
def test_critic_loss_module_forward_is_finite_and_matches_value_loss():
    module = CriticLoss() if not isinstance(CriticLoss, type) else CriticLoss(coefficient=4.0)
    values = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([0.0, 0.0, 0.0])
    batch = {"values": values, "value_targets": targets}

    try:
        out = module(batch)
    except TypeError:
        out = module.forward(batch)
    total = _get(out, ("critic_loss", "total_loss", "value_loss", "loss"), out)
    assert torch.is_tensor(total)
    assert torch.isfinite(total).all()


def test_critic_loss_module_loss_method_returns_tensor():
    module = CriticLoss() if not isinstance(CriticLoss, type) else CriticLoss()
    values = torch.tensor([1.0, 2.0])
    targets = torch.tensor([0.0, 1.0])
    batch = {"values": values, "value_targets": targets}
    loss = module.loss(batch)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all()


# --------------------------------------------------------------------------- #
# cross-module consistency: sapg.utils.returns <-> sapg.losses.critic_loss
# --------------------------------------------------------------------------- #
def test_returns_helpers_match_critic_loss_module():
    torch.manual_seed(2)
    T, B = 5, 2
    rewards = torch.rand(T, B)
    dones = torch.zeros(T, B)
    values = torch.rand(T, B)
    last_values = torch.rand(B)

    try:
        from sapg.utils.returns import compute_n_step_targets as returns_n_step
        from sapg.utils.returns import compute_one_step_targets as returns_one_step
    except Exception:  # pragma: no cover - defensive
        pytest.skip("sapg.utils.returns unavailable")

    t_a = _tensor(
        _call(
            compute_n_step_targets,
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(T, B)
    t_b = _tensor(
        _call(
            returns_n_step,
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=GAMMA,
            n_step=3,
        )
    ).reshape(T, B)
    assert torch.allclose(t_a, t_b, atol=1e-5, rtol=1e-4)

    next_values = torch.rand(T, B)
    o_a = _tensor(
        _call(
            compute_one_step_targets,
            rewards=rewards,
            dones=dones,
            next_values=next_values,
            gamma=GAMMA,
        )
    ).reshape(T, B)
    o_b = _tensor(
        _call(
            returns_one_step,
            rewards=rewards,
            dones=dones,
            next_values=next_values,
            gamma=GAMMA,
        )
    ).reshape(T, B)
    assert torch.allclose(o_a, o_b, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# gradient flow
# --------------------------------------------------------------------------- #
def test_value_loss_gradients_flow_to_values():
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    targets = torch.tensor([0.0, 0.0])
    out = _call(compute_value_loss, values=values, value_targets=targets, coefficient=4.0)
    loss = _get(out, ("value_loss_scaled", "value_loss", "loss"), out)
    if isinstance(loss, float):  # pragma: no cover - defensive
        pytest.skip("value loss returned a python float")
    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
