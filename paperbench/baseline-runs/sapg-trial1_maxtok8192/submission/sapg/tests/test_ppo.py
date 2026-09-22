"""Unit tests for the PPO core (sapg/sapg/ppo.py).

These tests validate the clipped surrogate objective, GAE advantage
estimation, the adaptive KL learning-rate scheduler, advantage
normalization, and the assembled PPO loss against hand-computed
reference values.

Run with::

    python -m pytest sapg/tests/test_ppo.py -v
    # or, without pytest:
    python sapg/tests/test_ppo.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Make the package importable when running this file directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../sapg (project root)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg.sapg.ppo import (  # noqa: E402
    KLScheduler,
    PPOHyperParams,
    clip_grad_norm_,
    clipped_surrogate_loss,
    compute_gae,
    compute_ppo_loss,
    explained_variance,
    normalize_advantages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def _assert_close(a, b, tol: float = 1e-6, msg: str = ""):
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)
    if not torch.allclose(a_t, b_t, atol=tol, rtol=0.0):
        raise AssertionError(
            f"{msg} values differ: {a_t.tolist()} vs {b_t.tolist()} (tol={tol})"
        )


# ---------------------------------------------------------------------------
# 1. Clipped surrogate objective
# ---------------------------------------------------------------------------
def test_clipped_surrogate_positive_advantage():
    """With A > 0, the objective should increase the action probability,
    but the ratio is clipped at 1 + eps."""
    old_log_probs = torch.zeros(1)
    advantages = torch.ones(1) * 2.0
    clip_eps = 0.1

    # ratio = exp(0) = 1 -> unclipped term = 1 * 2 = 2
    new_log_probs = torch.zeros(1)
    loss, ratio, clip_frac = clipped_surrogate_loss(
        new_log_probs, old_log_probs, advantages, clip_eps
    )
    # loss is the negative surrogate: -min(2, clip(1)*2) = -2
    assert _approx(loss.item(), -2.0), loss.item()
    assert _approx(ratio.item(), 1.0)
    assert _approx(clip_frac.item(), 0.0)

    # ratio = exp(0.5) ~ 1.6487 > 1.1 -> clipped term = 1.1 * 2 = 2.2
    new_log_probs = torch.tensor([0.5])
    loss, ratio, clip_frac = clipped_surrogate_loss(
        new_log_probs, old_log_probs, advantages, clip_eps
    )
    expected_ratio = math.exp(0.5)
    assert _approx(ratio.item(), expected_ratio, tol=1e-5)
    # min(1.6487*2, 1.1*2) = 2.2 -> loss = -2.2
    assert _approx(loss.item(), -2.2, tol=1e-5), loss.item()
    assert _approx(clip_frac.item(), 1.0)


def test_clipped_surrogate_negative_advantage():
    """With A < 0, the objective should decrease the action probability,
    but the ratio is clipped at 1 - eps."""
    old_log_probs = torch.zeros(1)
    advantages = torch.ones(1) * -2.0
    clip_eps = 0.1

    # ratio = exp(-0.5) ~ 0.6065 < 0.9 -> clipped term = 0.9 * -2 = -1.8
    new_log_probs = torch.tensor([-0.5])
    loss, ratio, clip_frac = clipped_surrogate_loss(
        new_log_probs, old_log_probs, advantages, clip_eps
    )
    expected_ratio = math.exp(-0.5)
    assert _approx(ratio.item(), expected_ratio, tol=1e-5)
    # min(0.6065 * -2, 0.9 * -2) = min(-1.213, -1.8) = -1.8 -> loss = 1.8
    assert _approx(loss.item(), 1.8, tol=1e-5), loss.item()
    assert _approx(clip_frac.item(), 1.0)


def test_clipped_surrogate_no_clip_region():
    """Inside the trust region the objective equals -ratio * A."""
    old_log_probs = torch.zeros(3)
    new_log_probs = torch.tensor([0.05, -0.05, 0.0])
    advantages = torch.tensor([1.0, -1.0, 0.5])
    loss, ratio, clip_frac = clipped_surrogate_loss(
        new_log_probs, old_log_probs, advantages, clip_eps=0.1
    )
    expected = -torch.mean(
        torch.exp(new_log_probs) * advantages
    )
    assert _approx(loss.item(), expected.item(), tol=1e-6), (
        loss.item(),
        expected.item(),
    )
    assert _approx(clip_frac.item(), 0.0)


# ---------------------------------------------------------------------------
# 2. GAE
# ---------------------------------------------------------------------------
def test_gae_single_step_no_done():
    """One step, no termination: adv = r + gamma * V_next - V."""
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])
    dones = torch.tensor([[0.0]])
    next_value = torch.tensor([2.0])
    gamma, tau = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, dones, next_value, gamma, tau)
    expected_adv = 1.0 + gamma * 2.0 - 0.5
    expected_ret = expected_adv + 0.5
    assert _approx(adv.item(), expected_adv, tol=1e-5), adv.item()
    assert _approx(ret.item(), expected_ret, tol=1e-5), ret.item()


def test_gae_two_steps_hand_computed():
    """Two-step trajectory, hand-computed GAE-lambda."""
    gamma, tau = 0.99, 0.95
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0]])
    dones = torch.tensor([[0.0], [0.0]])
    next_value = torch.tensor([0.0])

    adv, ret = compute_gae(rewards, values, dones, next_value, gamma, tau)

    # t=1: delta_1 = 1 + gamma*0 - 0 = 1 ; adv_1 = 1
    # t=0: delta_0 = 1 + gamma*0 - 0 = 1 ; adv_0 = 1 + gamma*tau*adv_1
    adv_1 = 1.0
    adv_0 = 1.0 + gamma * tau * adv_1
    assert _approx(adv[1].item(), adv_1, tol=1e-5), adv.tolist()
    assert _approx(adv[0].item(), adv_0, tol=1e-5), adv.tolist()
    # returns = advantages + values
    assert _approx(ret[0].item(), adv_0, tol=1e-5)
    assert _approx(ret[1].item(), adv_1, tol=1e-5)


def test_gae_done_masks_bootstrap():
    """A done flag at step t must zero the bootstrap from t+1."""
    gamma, tau = 0.99, 0.95
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0]])
    dones = torch.tensor([[0.0], [1.0]])  # episode ends at t=1
    next_value = torch.tensor([100.0])  # should be masked out

    adv, ret = compute_gae(rewards, values, dones, next_value, gamma, tau)
    # t=1: delta_1 = 1 + gamma*(1-1)*100 - 0 = 1
    assert _approx(adv[1].item(), 1.0, tol=1e-5), adv.tolist()
    # t=0: delta_0 = 1 + gamma*(1-0)*0 - 0 = 1 ; adv_0 = 1 + gamma*tau*1
    adv_0 = 1.0 + gamma * tau * 1.0
    assert _approx(adv[0].item(), adv_0, tol=1e-5), adv.tolist()


def test_gae_shapes():
    T, N = 5, 3
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    next_value = torch.randn(N)
    adv, ret = compute_gae(rewards, values, dones, next_value)
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)


# ---------------------------------------------------------------------------
# 3. Advantage normalization
# ---------------------------------------------------------------------------
def test_normalize_advantages():
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
    norm = normalize_advantages(adv)
    assert _approx(norm.mean().item(), 0.0, tol=1e-5)
    assert _approx(norm.std(unbiased=False).item(), 1.0, tol=1e-4)


# ---------------------------------------------------------------------------
# 4. KL scheduler
# ---------------------------------------------------------------------------
def test_kl_scheduler_decreases_on_high_kl():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-3)
    sched = KLScheduler(opt, kl_threshold=0.016, min_lr=1e-6, max_lr=1e-2, scale=1.5)

    # KL far above 2 * threshold -> lr should shrink by 1/scale
    new_lr = sched.step(0.1)
    assert _approx(new_lr, 1e-3 / 1.5, tol=1e-9), new_lr
    assert _approx(opt.param_groups[0]["lr"], new_lr, tol=1e-12)


def test_kl_scheduler_increases_on_low_kl():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-3)
    sched = KLScheduler(opt, kl_threshold=0.016, min_lr=1e-6, max_lr=1e-2, scale=1.5)

    # KL below threshold / 2 -> lr should grow by scale
    new_lr = sched.step(0.001)
    assert _approx(new_lr, 1e-3 * 1.5, tol=1e-9), new_lr


def test_kl_scheduler_unchanged_in_band():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-3)
    sched = KLScheduler(opt, kl_threshold=0.016, min_lr=1e-6, max_lr=1e-2, scale=1.5)

    new_lr = sched.step(0.016)  # exactly at threshold -> unchanged
    assert _approx(new_lr, 1e-3, tol=1e-9), new_lr


def test_kl_scheduler_clamps_to_bounds():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-2)
    sched = KLScheduler(opt, kl_threshold=0.016, min_lr=1e-6, max_lr=1e-2, scale=1.5)
    # Repeated low-KL steps must not exceed max_lr
    for _ in range(20):
        lr = sched.step(0.0)
    assert lr <= 1e-2 + 1e-12, lr

    opt2 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-6)
    sched2 = KLScheduler(opt2, kl_threshold=0.016, min_lr=1e-6, max_lr=1e-2, scale=1.5)
    for _ in range(20):
        lr2 = sched2.step(1.0)
    assert lr2 >= 1e-6 - 1e-12, lr2


# ---------------------------------------------------------------------------
# 5. Full PPO loss assembly
# ---------------------------------------------------------------------------
def test_compute_ppo_loss_components():
    torch.manual_seed(0)
    n = 8
    new_log_probs = torch.zeros(n)
    old_log_probs = torch.zeros(n)
    values = torch.zeros(n)
    returns = torch.ones(n)
    advantages = torch.ones(n)
    entropy = torch.zeros(n)
    actions = torch.zeros(n)

    total, info = compute_ppo_loss(
        new_log_probs,
        old_log_probs,
        values,
        returns,
        advantages,
        entropy,
        actions,
        clip_eps=0.1,
        entropy_coeff=0.0,
        critic_coeff=4.0,
        bounds_loss_coeff=1e-4,
        action_bound=1.0,
    )
    # policy loss = -1 (ratio=1, A=1); value loss = 1 (0 vs 1)
    assert _approx(info["policy_loss"], -1.0, tol=1e-5), info
    assert _approx(info["value_loss"], 1.0, tol=1e-5), info
    assert _approx(info["entropy"], 0.0, tol=1e-5), info
    assert _approx(info["bounds_loss"], 0.0, tol=1e-5), info
    # total = policy + 4 * value + 0 * entropy + 1e-4 * bounds
    expected_total = -1.0 + 4.0 * 1.0
    assert _approx(info["total_loss"], expected_total, tol=1e-4), info
    assert _approx(total.item(), expected_total, tol=1e-4), total.item()


def test_compute_ppo_loss_bounds_penalty():
    """Actions outside [-bound, bound] incur a penalty."""
    n = 4
    new_log_probs = torch.zeros(n)
    old_log_probs = torch.zeros(n)
    values = torch.zeros(n)
    returns = torch.zeros(n)
    advantages = torch.zeros(n)
    entropy = torch.zeros(n)
    actions = torch.tensor([0.5, 1.5, -2.0, 0.0])  # two out-of-bounds

    _, info = compute_ppo_loss(
        new_log_probs,
        old_log_probs,
        values,
        returns,
        advantages,
        entropy,
        actions,
        bounds_loss_coeff=1.0,
        action_bound=1.0,
    )
    # excess: 0, 0.5, 1.0, 0 -> mean = 0.375
    assert _approx(info["bounds_loss"], 0.375, tol=1e-5), info


def test_compute_ppo_loss_entropy_coeff():
    n = 4
    new_log_probs = torch.zeros(n)
    old_log_probs = torch.zeros(n)
    values = torch.zeros(n)
    returns = torch.zeros(n)
    advantages = torch.zeros(n)
    entropy = torch.ones(n)
    actions = torch.zeros(n)

    _, info = compute_ppo_loss(
        new_log_probs,
        old_log_probs,
        values,
        returns,
        advantages,
        entropy,
        actions,
        entropy_coeff=0.01,
        critic_coeff=0.0,
        bounds_loss_coeff=0.0,
    )
    # total = 0 (policy) + 0 (value) - 0.01 * 1 (entropy bonus)
    assert _approx(info["total_loss"], -0.01, tol=1e-6), info


# ---------------------------------------------------------------------------
# 6. Utilities
# ---------------------------------------------------------------------------
def test_explained_variance():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # perfect prediction -> EV = 1
    assert _approx(explained_variance(y_true, y_true), 1.0, tol=1e-5)
    # constant prediction -> EV = 0
    pred = torch.full_like(y_true, y_true.mean())
    assert _approx(explained_variance(pred, y_true), 0.0, tol=1e-5)


def test_clip_grad_norm():
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
    norm = clip_grad_norm_([p], max_norm=1.0)
    assert _approx(norm, 5.0, tol=1e-5), norm
    new_norm = p.grad.norm().item()
    assert _approx(new_norm, 1.0, tol=1e-5), new_norm


def test_ppo_hyperparams_defaults():
    hp = PPOHyperParams()
    assert _approx(hp.gamma, 0.99)
    assert _approx(hp.tau, 0.95)
    assert _approx(hp.clip_eps, 0.1)
    assert _approx(hp.critic_coeff, 4.0)
    assert _approx(hp.kl_threshold, 0.016)
    assert _approx(hp.grad_norm_clip, 1.0)
    assert _approx(hp.bounds_loss_coeff, 1e-4)
    assert _approx(hp.entropy_coeff, 0.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
