"""Tests for CORE COMPONENT #3 -- the Random Network Distillation (RND) exploration bonus.

Paper reference (Sec. 3.3, "Exploration with Random Network Distillation"):

    R'(s_t, a_t) = R(s_t, a_t) + lambda * | f(s_{t+1}) - f_hat(s_{t+1}) |^2

and Algorithm 2:

    Calculate RND bonus R_t^RND = || f(s_{t+1}) - f_hat(s_{t+1}) ||^2 with normalization
    Add (s_t, s_{t+1}, a_t, R_t + lambda * R_t^RND) to D
    Optimize f_hat_theta w.r.t. MSE loss on D using Adam

and the paper's qualitative claim:

    "as the state coverage increases, RND bonuses decay to zero and a performed policy
    is recovered."

This module checks the behaviours implied by those lines:

* ``f`` is frozen/random and never updated; ``f_hat`` *is* trained with MSE + Adam.
* ``R_t^RND`` is computed on the *next* state ``s_{t+1}`` and is non-negative.
* ``R' = R + lambda * R^RND`` exactly (with ``lambda`` == 0 the task reward is unchanged).
* Normalization (Burda et al. 2018 style) keeps bonuses bounded / scale-invariant.
* The bonus *decays* on states the predictor has been trained on (coverage growth).
* Serialization round-trips (state dicts, save/load, RunningMeanStd stats).

All tests are dependency-tolerant: they skip when ``rice.algorithms.rnd`` or ``torch``
is unavailable, and they never require MuJoCo/gym.
"""

from __future__ import annotations

import numpy as np
import pytest

try:  # package-relative import when collected as ``rice.tests.test_rnd``
    from . import _helpers
except Exception:  # pragma: no cover - direct/sys.path execution
    import _helpers  # type: ignore

# --------------------------------------------------------------------------------------
# Tolerant imports / module level skips
# --------------------------------------------------------------------------------------
rnd_mod = _helpers.import_optional("algorithms.rnd")
if rnd_mod is None:  # pragma: no cover - guarded import
    pytest.skip(
        "rice.algorithms.rnd is not importable; skipping RND tests",
        allow_module_level=True,
    )

RND = getattr(rnd_mod, "RND", None)
RNDConfig = getattr(rnd_mod, "RNDConfig", None)
RNDNetwork = getattr(rnd_mod, "RNDNetwork", None)
RunningMeanStd = getattr(rnd_mod, "RunningMeanStd", None)
compute_rnd_bonus = getattr(rnd_mod, "compute_rnd_bonus", None)
make_rnd = getattr(rnd_mod, "make_rnd", None)

if RND is None or RNDConfig is None:  # pragma: no cover - guarded import
    pytest.skip(
        "rice.algorithms.rnd is missing RND/RNDConfig; skipping RND tests",
        allow_module_level=True,
    )

pytestmark = pytest.mark.skipif(
    not _helpers.have_torch(), reason="torch is required for RND tests"
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _key(mapping, *names, default=None):
    """First present key/attribute among ``names`` (tolerates naming drift)."""
    if mapping is None:
        return default
    for name in names:
        if isinstance(mapping, dict):
            if name in mapping:
                return mapping[name]
        elif hasattr(mapping, name):
            return getattr(mapping, name)
    return default


def _flat(value):
    """Flatten any per-state error/bonus container to a 1-D float array."""
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _make_rnd(obs_dim: int = 4, seed: int = 0, net_arch=(32, 32), output_dim: int = 16, **overrides):
    """Build a small, deterministic RND module with clean (unnormalized) defaults.

    Normalization is disabled unless overridden so that raw-error identity checks
    (``bonus == ||f - f_hat||^2``) can be verified exactly.
    """
    kwargs = dict(
        net_arch=net_arch,
        output_dim=output_dim,
        activation="relu",
        normalize_obs=False,
        normalize_reward=False,
        predictor_learning_rate=1e-3,
        coef=0.01,
        reduction="mean",
        seed=seed,
        device="cpu",
        verbose=0,
    )
    kwargs.update(overrides)
    config = RNDConfig(**kwargs)
    return RND(obs_dim=obs_dim, config=config, device="cpu", seed=seed)


def _batch(n=64, obs_dim=4, seed=0, loc=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=(n, obs_dim)).astype(np.float32)


# ======================================================================================
# 1. Frozen target ``f`` and trainable predictor ``f_hat`` (Algorithm 2, last lines)
# ======================================================================================
def test_target_network_is_frozen_and_predictor_is_trainable():
    torch = _helpers.torch_or_skip()
    rnd = _make_rnd()
    obs = _batch(seed=1)

    assert hasattr(rnd, "target") and hasattr(rnd, "predictor")

    # The target ``f`` must not require gradients (frozen, random init).
    for p in rnd.target.parameters():
        assert not p.requires_grad, "target network f must be frozen"

    # The predictor ``f_hat`` must be trainable.
    assert any(p.requires_grad for p in rnd.predictor.parameters()), (
        "predictor network f_hat must be trainable"
    )

    before_target = [p.detach().clone() for p in rnd.target.parameters()]
    before_pred = [p.detach().clone() for p in rnd.predictor.parameters()]

    rnd.update(obs, normalize_obs=False, update_stats=False, n_updates=3)

    for ref, p in zip(before_target, rnd.target.parameters()):
        assert torch.allclose(ref, p.detach()), "target f changed during f_hat training"

    changed = any(
        not torch.allclose(ref, p.detach()) for ref, p in zip(before_pred, rnd.predictor.parameters())
    )
    assert changed, "predictor f_hat did not move after an update step"


def test_target_outputs_are_deterministic_across_calls():
    rnd = _make_rnd()
    obs = _batch(n=8, seed=2)
    a = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    b = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    assert np.allclose(a, b), "raw squared error must be deterministic for a fixed state"


# ======================================================================================
# 2. Bonus definition: ||f(s_{t+1}) - f_hat(s_{t+1})||^2 >= 0, computed on s_{t+1}
# ======================================================================================
def test_squared_error_is_non_negative_and_per_state():
    rnd = _make_rnd()
    obs = _batch(n=32, seed=3)
    err = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    assert err.size == obs.shape[0]
    assert np.all(err >= 0.0), "squared error (RND bonus before scaling) must be >= 0"
    assert np.all(np.isfinite(err))


def test_intrinsic_reward_equals_squared_error_without_normalization():
    """With reward normalization disabled, R^RND must be exactly ||f - f_hat||^2."""
    rnd = _make_rnd()
    obs = _batch(n=16, seed=4)
    err = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    bonus = _flat(rnd.intrinsic_reward(obs, normalize=False, update_stats=False))
    assert bonus.size == err.size
    assert np.allclose(bonus, err, rtol=1e-5, atol=1e-6)


def test_bonus_uses_next_state():
    """Algorithm 2 computes R^RND at s_{t+1}: different next states -> different bonuses."""
    rnd = _make_rnd()
    next_a = _batch(n=4, seed=5, loc=0.0)
    next_b = _batch(n=4, seed=5, loc=2.0)
    bonus_a = _flat(rnd.intrinsic_reward(next_a, normalize=False, update_stats=False))
    bonus_b = _flat(rnd.intrinsic_reward(next_b, normalize=False, update_stats=False))
    assert not np.allclose(bonus_a, bonus_b), "bonus must depend on the (next) state input"


# ======================================================================================
# 3. Augmented reward: R' = R + lambda * R^RND  (Sec. 3.3 / Algorithm 2)
# ======================================================================================
def test_augmented_reward_matches_paper_formula():
    lam = 0.5
    rnd = _make_rnd()
    rewards = np.array([1.0, -2.0, 0.25, 3.0], dtype=np.float64)
    next_obs = _batch(n=rewards.size, seed=6)

    bonus = _flat(rnd.intrinsic_reward(next_obs, normalize=False, update_stats=False))
    augmented = rnd.augmented_reward(
        rewards, next_obs, coef=lam, normalize=False, update_stats=False
    )
    augmented = _flat(augmented)
    assert augmented.size == rewards.size
    assert np.allclose(augmented, rewards + lam * bonus, rtol=1e-5, atol=1e-6)


def test_zero_coefficient_recovers_task_reward():
    """lambda == 0 must leave the task reward untouched (RND disabled)."""
    rnd = _make_rnd()
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    next_obs = _batch(n=3, seed=7)
    augmented = _flat(
        rnd.augmented_reward(rewards, next_obs, coef=0.0, normalize=False, update_stats=False)
    )
    assert np.allclose(augmented, rewards, rtol=1e-6, atol=1e-6)


def test_scaled_bonus_respects_coefficient():
    rnd = _make_rnd()
    obs = _batch(n=8, seed=8)
    raw = _flat(rnd.intrinsic_reward(obs, normalize=False, update_stats=False))
    scaled = _flat(rnd.scaled_bonus(obs, coef=0.1, normalize=False, update_stats=False))
    assert np.allclose(scaled, 0.1 * raw, rtol=1e-5, atol=1e-6)


def test_sum_reduction_is_larger_than_mean_reduction():
    """``reduction`` scales the same error: sum >= mean for output_dim > 1."""
    mean_rnd = _make_rnd(seed=11, reduction="mean")
    sum_rnd = _make_rnd(seed=11, reduction="sum")
    obs = _batch(n=8, seed=9)
    err_mean = _flat(mean_rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    err_sum = _flat(sum_rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    assert np.all(err_sum >= err_mean - 1e-6)
    assert np.allclose(err_sum, err_mean * mean_rnd.config.output_dim, rtol=1e-3, atol=1e-3)


# ======================================================================================
# 4. Predictor training (MSE + Adam) reduces the bonus on covered states
# ======================================================================================
def test_update_reduces_error_on_trained_states():
    rnd = _make_rnd(predictor_learning_rate=5e-3)
    obs = _batch(n=64, seed=10)
    before = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False)).mean()

    rnd.update(obs, normalize_obs=False, update_stats=False, n_updates=200, batch_size=64)

    after = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False)).mean()
    assert np.isfinite(after)
    assert after < before, "predictor f_hat must reduce the MSE on states it is trained on"
    assert after < 0.5 * before, "expected a substantial reduction after 200 Adam steps"


def test_bonus_decays_as_coverage_increases():
    """Paper claim: 'as the state coverage increases, RND bonuses decay to zero'."""
    rnd = _make_rnd(predictor_learning_rate=5e-3)
    states = _batch(n=128, seed=12)

    first = _flat(rnd.intrinsic_reward(states, normalize=False, update_stats=False)).mean()
    for _ in range(6):
        rnd.update(states, normalize_obs=False, update_stats=False, n_updates=50, batch_size=64)
    last = _flat(rnd.intrinsic_reward(states, normalize=False, update_stats=False)).mean()

    assert last < first, "bonus must decay on repeatedly visited states"
    assert last < 0.3 * first, "bonus should decay substantially once coverage increases"


def test_novel_states_keep_higher_bonus_than_covered_states():
    """Novelty signal: error on unseen states stays above error on trained states."""
    rnd = _make_rnd(predictor_learning_rate=5e-3)
    covered = _batch(n=64, seed=13, loc=0.0)
    novel = _batch(n=64, seed=13, loc=6.0)

    for _ in range(4):
        rnd.update(covered, normalize_obs=False, update_stats=False, n_updates=100, batch_size=64)

    err_covered = _flat(rnd.squared_error(covered, normalize_obs=False, update_stats=False)).mean()
    err_novel = _flat(rnd.squared_error(novel, normalize_obs=False, update_stats=False)).mean()
    assert err_covered < err_novel, (
        "RND bonus should be lower on covered states than on novel states "
        f"(covered={err_covered:.4f}, novel={err_novel:.4f})"
    )


def test_update_accepts_arrays_and_returns_finite_loss():
    rnd = _make_rnd()
    obs = _batch(n=32, seed=14)
    out = rnd.update(obs, normalize_obs=False, update_stats=False, n_updates=2, return_losses=True)
    if isinstance(out, dict):
        loss = _key(out, "loss", "total", "rnd_loss", "mse", default=None)
        if loss is not None:
            assert np.isfinite(float(loss))
    elif out is not None:
        assert np.isfinite(float(out))


def test_update_from_rollout_buffer():
    """Algorithm 2 optimizes f_hat on D (the rollout buffer), using next states."""
    ppo_mod = _helpers.import_optional("algorithms.ppo")
    if ppo_mod is None or not hasattr(ppo_mod, "RolloutBuffer"):
        pytest.skip("rice.algorithms.ppo.RolloutBuffer unavailable")

    rnd = _make_rnd()
    buffer = ppo_mod.RolloutBuffer()
    rng = np.random.default_rng(15)
    for _ in range(16):
        obs = rng.normal(size=4).astype(np.float32)
        next_obs = obs + 0.1
        buffer.add(obs, np.zeros(2, dtype=np.float32), 1.0, next_obs, False)

    out = rnd.update_from_buffer(buffer, key="next_obs", n_updates=2, batch_size=8)
    assert out is None or isinstance(out, (float, dict, np.floating, np.ndarray))


# ======================================================================================
# 5. Normalization (Burda et al. 2018 recipe)
# ======================================================================================
def test_running_mean_std_normalizes_to_unit_statistics():
    if RunningMeanStd is None:
        pytest.skip("RunningMeanStd unavailable")
    rng = np.random.default_rng(16)
    rms = RunningMeanStd(shape=(3,), epsilon=1e-4, clip=None)

    data = rng.normal(loc=2.0, scale=3.0, size=(20000, 3))
    rms.update(data)
    normed = np.asarray(rms.normalize(data, update=False), dtype=np.float64)

    assert abs(float(normed.mean())) < 1e-2
    assert abs(float(normed.std()) - 1.0) < 0.05


def test_running_mean_std_state_roundtrip():
    if RunningMeanStd is None:
        pytest.skip("RunningMeanStd unavailable")
    rng = np.random.default_rng(17)
    data = rng.normal(loc=-1.0, scale=0.5, size=(500, 2))

    rms = RunningMeanStd(shape=(2,))
    rms.update(data)
    restored = RunningMeanStd(shape=(2,))
    restored.load_state_dict(rms.state_dict())

    a = np.asarray(rms.normalize(data, update=False), dtype=np.float64)
    b = np.asarray(restored.normalize(data, update=False), dtype=np.float64)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_normalized_bonus_is_finite_and_non_negative():
    rnd = _make_rnd(normalize_obs=True, normalize_reward=True, clip_obs=5.0, clip_reward=10.0)
    obs = _batch(n=64, seed=18, loc=3.0, scale=4.0)
    bonus = _flat(rnd.intrinsic_reward(obs, normalize=None, update_stats=True))
    assert np.all(np.isfinite(bonus)), "normalized bonus must stay finite"
    assert np.all(bonus >= -1e-8), "normalized bonus must be non-negative"


# ======================================================================================
# 6. Configuration / construction surface
# ======================================================================================
def test_rnd_config_defaults_follow_burda_recipe():
    cfg = RNDConfig()
    assert getattr(cfg, "activation", "relu") in ("relu", "tanh")
    assert getattr(cfg, "coef", 0.01) == pytest.approx(0.01), "default lambda (coef) documented as 0.01"
    assert getattr(cfg, "predictor_learning_rate", 1e-3) == pytest.approx(1e-3)
    assert getattr(cfg, "normalize_obs", True) is True
    assert getattr(cfg, "normalize_reward", True) is True


def test_rnd_config_clone_override():
    cfg = RNDConfig()
    clone = cfg.clone(coef=0.25)
    assert clone.coef == pytest.approx(0.25)
    assert cfg.coef == pytest.approx(0.01), "clone must not mutate the source config"


def test_make_rnd_accepts_coefficient_aliases():
    if make_rnd is None:
        pytest.skip("make_rnd unavailable")
    rnd = make_rnd(obs_dim=3, net_arch=(16, 16), device="cpu", seed=0)
    assert isinstance(rnd, RND)

    # ``lambda`` / ``coef`` aliases accepted via a mapping (``lambda`` is a keyword).
    rnd2 = make_rnd(obs_dim=3, net_arch=(16, 16), device="cpu", seed=0, **{"lambda": 0.05})
    assert isinstance(rnd2, RND)

    obs = _batch(n=4, obs_dim=3, seed=19)
    bonus = _flat(rnd2.intrinsic_reward(obs, normalize=False, update_stats=False))
    assert bonus.size == 4 and np.all(bonus >= 0.0)


def test_rnd_network_forward_shapes():
    if RNDNetwork is None:
        pytest.skip("RNDNetwork unavailable")
    net = RNDNetwork(input_dim=5, net_arch=(16, 16), output_dim=8, activation="relu")
    out = np.asarray(net(_torch_zeros(4, 5)).detach().cpu().numpy())
    assert out.shape == (4, 8)


def _torch_zeros(rows, cols):
    torch = _helpers.torch_or_skip()
    return torch.zeros(rows, cols, dtype=torch.float32)


# ======================================================================================
# 7. Serialization + bookkeeping
# ======================================================================================
def test_state_dict_roundtrip_preserves_errors():
    rnd = _make_rnd()
    obs = _batch(n=16, seed=20)
    rnd.update(obs, normalize_obs=False, update_stats=False, n_updates=5)

    state = rnd.state_dict()
    clone = _make_rnd()
    clone.load_state_dict(state)

    a = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    b = _flat(clone.squared_error(obs, normalize_obs=False, update_stats=False))
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_reset_reward_statistics_is_safe():
    rnd = _make_rnd(normalize_reward=True)
    obs = _batch(n=8, seed=21)
    rnd.intrinsic_reward(obs, update_stats=True)
    rnd.reset_reward_statistics()  # must not raise
    assert hasattr(rnd, "reward_rms")


def test_mean_error_and_bonus_properties_are_finite():
    rnd = _make_rnd()
    obs = _batch(n=16, seed=22)
    rnd.intrinsic_reward(obs, normalize=False, update_stats=True)
    for value in (rnd.mean_error, rnd.mean_bonus):
        if value is not None:
            assert np.isfinite(float(value))


def test_decay_report_is_a_mapping():
    rnd = _make_rnd()
    for seed in range(5):
        rnd.intrinsic_reward(_batch(n=8, seed=100 + seed), normalize=False, update_stats=True)
    report = rnd.decay_report(first=2, last=2)
    assert report is not None
    assert isinstance(report, dict)


def test_compute_rnd_bonus_free_function():
    if compute_rnd_bonus is None:
        pytest.skip("compute_rnd_bonus unavailable")
    rnd = _make_rnd()
    obs = _batch(n=8, seed=23)
    bonus = _flat(compute_rnd_bonus(rnd, obs, normalize=False, update_stats=False))
    err = _flat(rnd.squared_error(obs, normalize_obs=False, update_stats=False))
    assert bonus.size == err.size
    assert np.all(bonus >= 0.0)


# ======================================================================================
# 8. Reduction to the RND-only / no-exploration regimes (used by Experiments II/IV)
# ======================================================================================
def test_lambda_zero_is_equivalent_to_plain_ppo_rewards():
    """lambda=0 regime == No-Refine/PPO fine-tuning objective (task reward only)."""
    rnd = _make_rnd()
    rewards = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    next_obs = _batch(n=3, seed=24)
    aug = _flat(rnd.augmented_reward(rewards, next_obs, coef=0.0, normalize=False, update_stats=False))
    assert np.allclose(aug, rewards, rtol=1e-6, atol=1e-6)


def test_small_lambda_keeps_bonus_subordinate():
    """Table 3 lambdas (0.001 - 0.01) must not swamp the task reward for unit rewards."""
    rnd = _make_rnd(normalize_reward=True, clip_reward=10.0)
    rewards = np.ones(16, dtype=np.float64)
    next_obs = _batch(n=16, seed=25)
    lam = 0.01
    aug = _flat(rnd.augmented_reward(rewards, next_obs, coef=lam, normalize=True, update_stats=True))
    assert np.all(aug > 0.0)
    assert np.abs(aug - rewards).max() <= lam * 10.0 + 1e-6
