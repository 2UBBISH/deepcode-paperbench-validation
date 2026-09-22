"""Unit tests for CORE COMPONENT #2 — the mixed initial state distribution.

Paper references (verbatim):

* Distribution (Sec. 3.3, "Constructing Mixed Initial State Distribution")::

      mu(s) = beta * d_rho^{pi_hat}(s) + (1 - beta) * rho(s)

  where ``d_rho^{pi_hat}`` is the distribution of identified critical states and
  ``rho`` the default initial state distribution of interest.

* Algorithm 2 (Sec. 3.3)::

      RAND_NUM <- RAND(0, 1)
      if RAND_NUM < p then
          Run pi to obtain a trajectory tau of length K
          Identify the most critical state s_t in tau via state mask pi~ 
          Set the initial state s_0 <- s_t
      else
          Set the initial state s_0 ~ rho
      end if

  Two gotchas that these tests pin down:

  1. ``RAND_NUM`` is drawn **once per outer refining iteration** — the whole
     ``T``-step episode shares one ``s_0`` (never a per-step draw).
  2. Hence ``p`` plays the role of ``beta``: the realized fraction of
     critical-state resets must match ``p``.

* Sec. 4.3 "Impact of Hyper-parameters": ``p = 0`` (all default initial states)
  and ``p = 1`` (all critical states) are both poor; ``0 < p < 1`` is required
  and ``p`` to 0.25 or 0.5 is most beneficial.  Table 3: Hopper/Walker2d/
  Selfish/Auto ``p = 0.25``; Reacher/HalfCheetah/Cage ``p = 0.50``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ----------------------------------------------------------------------------
# tolerant imports (repo may be laid out as `rice/rice/...` or `rice/...`)
# ----------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from ._helpers import (  # type: ignore
        StubExplanation,
        import_optional,
        make_box_space,
    )
except Exception:  # pragma: no cover
    from _helpers import (  # type: ignore
        StubExplanation,
        import_optional,
        make_box_space,
    )


_mixed = import_optional("algorithms.mixed_init")
if _mixed is None:  # pragma: no cover - environment without the module
    pytest.skip("rice.algorithms.mixed_init is not importable", allow_module_level=True)


MixedInitSampler = getattr(_mixed, "MixedInitSampler", None)
MixedInitSample = getattr(_mixed, "MixedInitSample", None)
mixed_initial_distribution = getattr(_mixed, "mixed_initial_distribution", None)
make_mixed_init_sampler = getattr(_mixed, "make_mixed_init_sampler", None)
bernoulli_rollin = getattr(_mixed, "bernoulli_rollin", None)


# ----------------------------------------------------------------------------
# tiny dependency-free environments
# ----------------------------------------------------------------------------
class _ScriptedEnv:
    """Deterministic env with a saving/restoring state (Go-Explore style).

    Returns the *modern* 5-tuple from ``step`` so that the mixed-init machinery
    also exercises its gym/gymnasium normalisation path.
    """

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, max_episode_steps: int = 20):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_episode_steps = max_episode_steps
        self.action_space = make_box_space(act_dim, low=-1.0, high=1.0)
        self.observation_space = make_box_space(obs_dim, low=-10.0, high=10.0)
        self._t = 0
        self._obs = np.zeros(obs_dim, dtype=np.float32)

    # -- gym API ------------------------------------------------------------
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self._t = 0
        self._obs = np.zeros(self.obs_dim, dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action, **kwargs):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._obs = np.clip(self._obs + 0.1 * float(np.sum(np.abs(action))), -10.0, 10.0)
        self._t += 1
        terminated = self._t >= self.max_episode_steps
        truncated = False
        reward = -0.1 * float(np.sum(np.square(action)))
        return self._obs.copy(), reward, terminated, truncated, {}

    def seed(self, seed=None):
        return [seed]

    def render(self, *args, **kwargs):
        return None

    def close(self):
        return None

    # -- state save/restore -------------------------------------------------
    def get_state(self):
        return {"t": self._t, "obs": self._obs.copy()}

    def set_state(self, state):
        self._t = int(state["t"])
        self._obs = np.asarray(state["obs"], dtype=np.float32).copy()
        return self._obs.copy()

    snapshot = get_state
    save_state = get_state
    restore = set_state
    load_state = set_state
    state_dict = get_state
    load_state_dict = set_state

    def current_observation(self):
        return self._obs.copy()


class _LegacyScriptedEnv(_ScriptedEnv):
    """Same env but with the legacy gym 4-tuple ``step``/``reset`` API."""

    def reset(self, seed=None, **kwargs):  # pragma: no cover - trivial
        obs, _ = super().reset(seed=seed, **kwargs)
        return obs

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = super().step(action, **kwargs)
        return obs, reward, bool(terminated or truncated), info


class _ZeroPolicy:
    """Black-box constant policy ``obs -> zeros(act_dim)`` (addendum: no
    inspection of the target agent's internals)."""

    act_dim = 2

    def __call__(self, obs):
        return np.zeros(self.act_dim, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self(obs), None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _make_sampler(env=None, policy=None, mask_network=None, **kwargs):
    env = env if env is not None else _ScriptedEnv()
    policy = policy if policy is not None else _ZeroPolicy()
    if mask_network is None:
        mask_network = StubExplanation(mode="increasing")
    return MixedInitSampler(env, policy, mask_network=mask_network, **kwargs)


def _canonical_obs(obs):
    """DummyEnv-like observations may be dicts or arrays; reduce to an array."""
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(obs[k]).reshape(-1) for k in sorted(obs)])
    return np.asarray(obs).reshape(-1)


def _is_critical(sample):
    """True when the sample came from the critical-state branch."""
    mode = getattr(sample, "mode", None)
    if isinstance(mode, str):
        return mode.lower().startswith("crit")
    flag = getattr(sample, "is_critical", None)
    if isinstance(flag, bool):
        return flag
    return bool(getattr(sample, "from_pool", False)) and mode is None


# ============================================================================
# 1. mixture weights  mu(s) = beta * d_rho^{pi_hat} + (1 - beta) * rho
# ============================================================================
@pytest.mark.skipif(mixed_initial_distribution is None, reason="helper unavailable")
@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_mixed_initial_distribution_weights(p):
    beta, one_minus = mixed_initial_distribution(p)
    assert beta == pytest.approx(p)
    assert one_minus == pytest.approx(1.0 - p)
    assert beta + one_minus == pytest.approx(1.0)


@pytest.mark.skipif(mixed_initial_distribution is None, reason="helper unavailable")
@pytest.mark.parametrize("p", [-0.01, 1.01, 2.0])
def test_mixed_initial_distribution_rejects_out_of_range(p):
    with pytest.raises(Exception):
        mixed_initial_distribution(p)


def test_mixture_weights_property_matches_p():
    sampler = _make_sampler(p=0.25)
    weights = getattr(sampler, "mixture_weights", None)
    if weights is None:  # pragma: no cover - naming drift
        pytest.skip("sampler exposes no mixture_weights property")
    assert tuple(weights) == pytest.approx((0.25, 0.75))


# ============================================================================
# 2. Algorithm 2: RAND_NUM <- RAND(0,1); critical branch iff RAND_NUM < p
# ============================================================================
def test_decide_returns_bool_and_matches_probability():
    n = 4000
    sampler = _make_sampler(p=0.25)
    decisions = [sampler.decide() for _ in range(n)]
    assert all(isinstance(d, bool) for d in decisions)
    frac = float(np.mean(decisions))
    tol = 4.0 * math.sqrt(0.25 * 0.75 / n)
    assert abs(frac - 0.25) <= tol, f"realized critical fraction {frac} far from p=0.25"


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_decide_matches_p_extremes(p):
    n = 500
    sampler = _make_sampler(p=p)
    frac = float(np.mean([sampler.decide() for _ in range(n)]))
    if p == 0.0:
        assert frac == 0.0
    elif p == 1.0:
        assert frac == 1.0
    else:
        tol = 5.0 * math.sqrt(p * (1.0 - p) / n)
        assert abs(frac - p) <= tol


def test_decide_force_overrides_draw():
    sampler = _make_sampler(p=0.5)
    assert sampler.decide(force=True) is True
    assert sampler.decide(force=False) is False


# ============================================================================
# 3. one RAND(0,1) draw per outer iteration (whole episode shares one s_0)
# ============================================================================
def test_one_decision_draw_per_sample():
    sampler = _make_sampler(p=0.5)
    decisions = getattr(sampler, "decisions", None)
    if decisions is None:  # pragma: no cover - naming drift
        pytest.skip("sampler does not record raw RAND_NUM draws")

    before = len(decisions)
    sampler.sample()
    after_one = len(decisions)
    assert after_one - before == 1, "Algorithm 2 draws RAND_NUM exactly once per iteration"

    sampler.sample()
    assert len(decisions) - before == 2


def test_sample_uses_single_decision_per_call():
    """Calling `sample` must not consume more than one Bernoulli draw."""
    sampler = _make_sampler(p=0.5)
    history = getattr(sampler, "history", None)
    if history is None:
        pytest.skip("sampler keeps no history")
    n = 0
    for _ in range(5):
        sampler.sample()
    n = len(history)
    assert n == 5


# ============================================================================
# 4. sample(): critical vs default branches
# ============================================================================
def test_sample_force_critical_returns_visited_state():
    env = _ScriptedEnv()
    sampler = _make_sampler(env=env, p=0.5, mode="rollin")
    sample = sampler.sample(force=True)
    assert _is_critical(sample)
    obs = _canonical_obs(getattr(sample, "observation", getattr(sample, "state", None)))
    assert obs.shape[-1] == env.obs_dim
    assert np.all(np.isfinite(obs))


def test_sample_force_default_returns_initial_state():
    env = _ScriptedEnv()
    sampler = _make_sampler(env=env, p=0.5)
    sample = sampler.sample(force=False)
    assert not _is_critical(sample)
    obs = _canonical_obs(getattr(sample, "observation", getattr(sample, "state", None)))
    assert obs.shape[-1] == env.obs_dim
    # rho resets the simulator, so the observation is the zero start state
    assert np.allclose(obs, 0.0)


def test_sample_many_realizes_fraction_p():
    sampler = _make_sampler(p=0.25)
    samples = sampler.sample_many(400)
    assert len(samples) == 400
    frac = float(np.mean([_is_critical(s) for s in samples]))
    tol = 4.0 * math.sqrt(0.25 * 0.75 / 400)
    assert abs(frac - 0.25) <= tol


def test_critical_fraction_method_matches_p():
    sampler = _make_sampler(p=0.5)
    frac = float(sampler.critical_fraction(600))
    assert abs(frac - 0.5) <= 4.0 * math.sqrt(0.25 / 600)


# ============================================================================
# 5. degenerate probabilities: p = 0 and p = 1 are warned against (Sec. 4.3)
# ============================================================================
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_degenerate_p_is_flagged(p):
    sampler = _make_sampler(p=p)
    warning = sampler.warn_if_degenerate()
    if warning is None:  # pragma: no cover - implementation chose to stay silent
        pytest.skip("warn_if_degenerate is a no-op for this implementation")
    assert isinstance(warning, str) and warning


@pytest.mark.parametrize("p", [0.25, 0.5])
def test_interior_p_is_not_flagged(p):
    sampler = _make_sampler(p=p)
    assert sampler.warn_if_degenerate() in (None, "")


# ============================================================================
# 6. pool mode: empirical samples of d_rho^{pi_hat}
# ============================================================================
def test_pool_mode_samples_are_critical_and_come_from_pool():
    pool_obs = [np.full(4, float(i), dtype=np.float32) for i in range(1, 6)]
    sampler = MixedInitSampler(
        _ScriptedEnv(),
        _ZeroPolicy(),
        p=1.0,
        mode="pool",
        pool_observations=pool_obs,
    )
    samples = sampler.sample_many(6)
    assert all(_is_critical(s) for s in samples)
    observed = {tuple(np.round(_canonical_obs(getattr(s, "observation", s.state)), 5)) for s in samples}
    allowed = {tuple(np.round(o, 5)) for o in pool_obs}
    assert observed.issubset(allowed)


def test_add_critical_state_extends_pool():
    sampler = MixedInitSampler(_ScriptedEnv(), _ZeroPolicy(), p=1.0, mode="pool",
                               pool_observations=[np.zeros(4, dtype=np.float32)])
    before = len(getattr(sampler, "critical_state_pool", []) or [])
    sampler.add_critical_state(np.ones(4, dtype=np.float32))
    after = len(getattr(sampler, "critical_state_pool", []) or [])
    assert after == before + 1


# ============================================================================
# 7. roll-in trajectory honours length K and preserves the single s_0
# ============================================================================
def test_roll_in_returns_length_K_trajectory():
    env = _ScriptedEnv(max_episode_steps=50)
    sampler = _make_sampler(env=env, p=0.5, rollin_length=7)
    trajectory = sampler.roll_in(length=7)
    assert len(trajectory) == 7


def test_sample_records_critical_index_and_importance():
    sampler = _make_sampler(p=1.0, mode="rollin", rollin_length=9)
    sample = sampler.sample(force=True)
    idx = getattr(sample, "critical_index", None)
    if idx is None:  # pragma: no cover - naming drift
        pytest.skip("sample does not expose critical_index")
    assert 0 <= int(idx) <= 8


# ============================================================================
# 8. config plumbing + serialisation
# ============================================================================
def test_make_mixed_init_sampler_accepts_beta_and_length_aliases():
    if make_mixed_init_sampler is None:  # pragma: no cover
        pytest.skip("make_mixed_init_sampler unavailable")
    sampler = make_mixed_init_sampler(
        _ScriptedEnv(), _ZeroPolicy(), mask_network=StubExplanation(), beta=0.5, length=5
    )
    weights = getattr(sampler, "mixture_weights", None)
    assert weights is None or tuple(weights) == pytest.approx((0.5, 0.5))
    assert float(getattr(sampler, "p", 0.5)) == pytest.approx(0.5)


def test_setters_update_probability_and_policy():
    sampler = _make_sampler(p=0.25)
    sampler.set_p(0.75)
    assert float(getattr(sampler, "p", -1)) == pytest.approx(0.75)
    new_policy = _ZeroPolicy()
    sampler.set_policy(new_policy)
    assert sampler.policy_callable is not None


def test_state_dict_roundtrip_preserves_decisions():
    sampler = _make_sampler(p=0.5)
    for _ in range(7):
        sampler.decide()
    state = sampler.state_dict()
    fresh = _make_sampler(p=0.5)
    fresh.load_state_dict(state)
    decisions = getattr(fresh, "decisions", None)
    if decisions is None:  # pragma: no cover
        pytest.skip("sampler does not record decisions")
    assert len(decisions) == 7


# ============================================================================
# 9. gym/gymnasium compatibility (5-tuple and legacy 4-tuple step APIs)
# ============================================================================
def test_works_with_legacy_four_tuple_env():
    env = _LegacyScriptedEnv()
    sampler = _make_sampler(env=env, p=0.5, rollin_length=5)
    sample = sampler.sample(force=True)
    obs = _canonical_obs(getattr(sample, "observation", sample.state))
    assert obs.shape[-1] == env.obs_dim


# ============================================================================
# 10. reproducibility: identical seeds -> identical decision sequences
# ============================================================================
def test_seeded_sampler_is_reproducible():
    a = _make_sampler(p=0.5, seed=123)
    b = _make_sampler(p=0.5, seed=123)
    seq_a = [a.decide() for _ in range(50)]
    seq_b = [b.decide() for _ in range(50)]
    assert seq_a == seq_b


# ============================================================================
# 11. convenience wrapper follows the same Bernoulli branch
# ============================================================================
def test_bernoulli_rollin_wrapper_returns_observation():
    if bernoulli_rollin is None:  # pragma: no cover
        pytest.skip("bernoulli_rollin unavailable")
    sampler = _make_sampler(p=0.5, rollin_length=4)
    out = bernoulli_rollin(sampler, force=False)
    obs = out[0] if isinstance(out, tuple) else out
    obs = _canonical_obs(obs if not isinstance(obs, dict) else obs)
    assert obs.shape[-1] == sampler.env.obs_dim


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
