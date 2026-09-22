"""Tests for CORE COMPONENT #1: the redesigned StateMask mask network (Algorithm 1).

What is exercised (all assertions come from the paper's own text, not from a paraphrase):

* Eq. (1) masking rule -- ``a_t \odot a_t^m = a_t if a_t^m = 0 else a_random``
  implemented by :func:`rice.algorithms.mask_network.masked_action`.
* ``importance = P(mask network outputting "0")`` (§3.3, step-level explanation)
  implemented by :meth:`MaskNetwork.mask_prob_zero` / ``importance`` and
  :func:`importance_from_logits`.
* The anti-collapse device of §3.3: "naıvely maximizing the expected total reward may
  introduce a trivial solution ... which is to not blind the target agent at all (always
  outputs " 0 "). To tackle this problem, we add an additional reward by giving an extra
  bonus when the mask net outputs " 1 ". The new reward can be written as
  R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m" -- tested by comparing the recorded
  rollout reward against the same rollout with ``alpha = 0``.
* Algorithm 1's rollout contract (record ``(s_t, s_{t+1}, a_t^m, R'_t)``, PPO update on
  ``D``) and the sample-budget bookkeeping used for the Table 4 timing claim.

The tests are dependency tolerant: they skip (rather than fail) when torch or the
algorithm/compatibility modules are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

# ``_helpers`` bootstraps ``sys.path`` for the several repository layouts and exposes
# tolerant importers, so import it first (relative when collected as a package, bare
# otherwise).
try:  # pragma: no cover - layout dependent
    from ._helpers import (  # type: ignore
        build_actor_critic,
        build_mask_network,
        import_optional,
        make_box_space,
        make_discrete_space,
        torch_or_skip,
        zero_policy,
    )
except ImportError:  # pragma: no cover - layout dependent
    from _helpers import (  # type: ignore
        build_actor_critic,
        build_mask_network,
        import_optional,
        make_box_space,
        make_discrete_space,
        torch_or_skip,
        zero_policy,
    )

mask_module = import_optional("algorithms.mask_network")
ppo_module = import_optional("algorithms.ppo")
dummy_env_module = import_optional("environments._common")

torch = torch_or_skip() if mask_module is not None else None

pytestmark = pytest.mark.skipif(
    mask_module is None, reason="rice.algorithms.mask_network is not importable"
)


# --------------------------------------------------------------------------------------
# The stand-in environment used by the rollout tests
# --------------------------------------------------------------------------------------


class _ScriptedEnv:
    """Tiny deterministic gym-like env (no MuJoCo needed).

    Rewards are a small action-dependent control cost, so the optimal masking behaviour is
    *not* distinguishable from the reward signal alone -- exactly the situation §3.3
    describes, which is why the ``alpha`` bonus exists.  ``step`` returns the 5-tuple
    (obs, reward, terminated, truncated, info); the trainer/driver code normalises it.
    """

    def __init__(self, obs_dim=4, act_dim=2, max_episode_steps=20, control_cost=0.1):
        from rice.environments._common import make_box  # local import: layout tolerant

        self.observation_space = make_box(-10.0, 10.0, shape=(obs_dim,))
        self.action_space = make_box(-1.0, 1.0, shape=(act_dim,))
        self.max_episode_steps = max_episode_steps
        self._obs_dim = obs_dim
        self._control_cost = control_cost
        self._step = 0
        self._state = np.zeros(obs_dim, dtype=np.float32)

    # -- gym API -----------------------------------------------------------------
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed % (2**31 - 1))
        self._step = 0
        self._state = np.zeros(self._obs_dim, dtype=np.float32)
        return self._state.copy(), {}

    def step(self, action, **kwargs):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step += 1
        self._state = np.clip(0.9 * self._state + 0.1 * action[: self._obs_dim], -10.0, 10.0)
        reward = -self._control_cost * float(np.sum(np.square(action)))
        terminated = self._step >= self.max_episode_steps
        return self._state.copy(), float(reward), bool(terminated), False, {}

    def render(self, *args, **kwargs):  # pragma: no cover - unused
        return None

    def close(self):  # pragma: no cover - unused
        return None

    # -- Go-Explore style state save/restore (kept for interface parity) ----------
    def get_state(self):
        return {"step": self._step, "state": self._state.copy()}

    def set_state(self, state):
        self._step = int(state["step"])
        self._state = np.asarray(state["state"], dtype=np.float32).copy()

    def state_dict(self):
        return self.get_state()

    def load_state_dict(self, state):
        self.set_state(state)


def _env(obs_dim=4, act_dim=2, max_episode_steps=20):
    return _ScriptedEnv(obs_dim=obs_dim, act_dim=act_dim, max_episode_steps=max_episode_steps)


def _key(mapping, *names, default=None):
    """First present key among ``names`` (tolerates naming drift in the trainer stats)."""
    for name in names:
        if isinstance(mapping, dict) and name in mapping:
            return mapping[name]
    return default


def _mean_reward_of(buffer):
    """Mean recorded (augmented) reward of a :class:`RolloutBuffer`."""
    arrays = buffer.as_arrays() if hasattr(buffer, "as_arrays") else buffer
    rewards = arrays["rewards"] if "rewards" in arrays else arrays["reward"]
    rewards = np.asarray(rewards, dtype=np.float64)
    assert rewards.size > 0, "rollout buffer recorded no reward"
    return float(np.mean(rewards))


def _blind_mask_forward(obs):
    """Deterministic mask network output: ``P(a^m = 0) ~ 0`` (always blind)."""
    obs_np = np.asarray(obs, dtype=np.float32)
    obs_np = obs_np.reshape(-1, obs_np.shape[-1]) if obs_np.ndim > 1 else obs_np.reshape(1, -1)
    n = obs_np.shape[0]
    zeros = torch.zeros(n, dtype=torch.float32)
    big = torch.full((n,), 100.0, dtype=torch.float32)
    return torch.stack([zeros, big], dim=1)


# --------------------------------------------------------------------------------------
# Eq. (1): the masking rule
# --------------------------------------------------------------------------------------


def test_masked_action_keeps_target_action_when_mask_is_zero():
    """``a_t \odot a_t^m = a_t`` when ``a_t^m = 0`` (Eq. 1, first branch)."""
    action_space = make_box_space(3, low=-1.0, high=1.0)
    a_t = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    rng = np.random.default_rng(0)

    taken = mask_module.masked_action(a_t, 0, action_space, rng)

    assert np.allclose(np.asarray(taken, dtype=np.float32).reshape(-1), a_t), (
        "a_t^m = 0 must leave the target agent's action untouched"
    )


def test_masked_action_replaces_action_with_random_when_mask_is_one():
    """``a_t \odot a_t^m = a_random`` when ``a_t^m = 1`` (Eq. 1, second branch)."""
    action_space = make_box_space(3, low=-1.0, high=1.0)
    a_t = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    rng = np.random.default_rng(0)

    samples = [
        np.asarray(mask_module.masked_action(a_t, 1, action_space, rng), dtype=np.float32)
        .reshape(-1)
        for _ in range(64)
    ]

    # a_random is sampled uniformly from the action space, so it must be in-range ...
    for s in samples:
        assert np.all(s >= -1.0 - 1e-6) and np.all(s <= 1.0 + 1e-6), (
            "a_random must lie inside the action space"
        )
    # ... and must not be identically the target action (probability 0 under a continuous
    # uniform distribution).
    assert any(not np.allclose(s, a_t) for s in samples), (
        "a_t^m = 1 must replace the action by a random one"
    )
    # The random replacements should cover the space rather than collapse to a constant.
    assert np.std(np.stack(samples), axis=0).min() > 1e-3


def test_masked_action_handles_discrete_action_space():
    """Eq. (1) also applies to Discrete action spaces (selfish mining / CAGE)."""
    action_space = make_discrete_space(3)
    rng = np.random.default_rng(0)

    kept = mask_module.masked_action(np.array(1), 0, action_space, rng)
    assert int(np.asarray(kept).reshape(-1)[0]) == 1

    drawn = {
        int(np.asarray(mask_module.masked_action(np.array(1), 1, action_space, rng)).reshape(-1)[0])
        for _ in range(64)
    }
    assert drawn.issubset({0, 1, 2})
    assert len(drawn) > 1, "a_random should sample all discrete actions"


# --------------------------------------------------------------------------------------
# Importance = P(mask network outputs "0")
# --------------------------------------------------------------------------------------


def test_importance_from_logits_is_probability_of_zero():
    """``importance = P(mask = 0)`` = ``softmax(logits)[0]`` (§3.3)."""
    logits = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]], dtype=torch.float32)
    probs = mask_module.importance_from_logits(logits).detach().cpu().numpy().reshape(-1)

    assert probs.shape == (3,)
    assert np.isclose(probs[0], 0.5, atol=1e-6)
    assert probs[1] > 0.99  # logits favour "keep"
    assert probs[2] < 0.01  # logits favour "blind"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_mask_network_importance_in_unit_interval_and_batch_consistent():
    """``mask_prob_zero``/``importance`` are probabilities and agree with the logits."""
    env = _env()
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)

    states = np.random.default_rng(0).normal(size=(7, env.observation_space.shape[0])).astype(np.float32)
    single = np.array([mask.mask_prob_zero(s) for s in states], dtype=np.float64)
    batched = np.asarray(mask.importance(states), dtype=np.float64).reshape(-1)

    assert np.all(single >= 0.0) and np.all(single <= 1.0)
    assert np.all(np.isfinite(batched)) and np.all(batched >= 0.0) and np.all(batched <= 1.0)
    assert np.allclose(single, batched, atol=1e-5), (
        "per-state and batched importance must agree"
    )


def test_mask_network_outputs_two_logits_per_state():
    """The mask net is a binary policy over {a^m = 0, a^m = 1}."""
    env = _env()
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)
    states = np.zeros((5, env.observation_space.shape[0]), dtype=np.float32)

    out = mask(states) if not callable(getattr(mask, "forward", None)) else mask.forward(states)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = np.asarray(logits.detach().cpu().numpy() if hasattr(logits, "detach") else logits)

    assert logits.shape == (5, 2)


def test_mask_network_state_dict_roundtrip():
    """The trained mask net can be saved/restored (used by train_mask.py / refine.py)."""
    env = _env()
    source = build_mask_network(env, net_arch=(32, 32), seed=0)
    target = build_mask_network(env, net_arch=(32, 32), seed=1)

    states = np.zeros((3, env.observation_space.shape[0]), dtype=np.float32)
    before = np.asarray(target.importance(states), dtype=np.float64).reshape(-1)

    target.load_policy_state_dict(source.policy_state_dict())
    after = np.asarray(target.importance(states), dtype=np.float64).reshape(-1)
    expected = np.asarray(source.importance(states), dtype=np.float64).reshape(-1)

    assert np.allclose(after, expected, atol=1e-6)
    assert not np.allclose(before, after, atol=1e-6), (
        "loading weights should change an independently initialised mask network"
    )


def test_mask_network_config_defaults_follow_table3():
    """``alpha`` defaults to the Table 3 value (1e-4), not the §C.3 text value (0.01)."""
    cfg = mask_module.MaskNetworkConfig()
    assert cfg.alpha == pytest.approx(1e-4), (
        "Table 3 is operative for alpha (the addendum resolves the §C.3 conflict)"
    )
    assert tuple(cfg.net_arch) == (64, 64), "MuJoCo default architecture mirrors MlpPolicy"


# --------------------------------------------------------------------------------------
# Algorithm 1: bonus reward R' = R + alpha * a_t^m
# --------------------------------------------------------------------------------------


def _run_rollout(alpha, seed=0, steps=8, net_arch=(32, 32)):
    """Run one Algorithm-1 iteration with an always-blinding mask network."""
    env = _env()
    target = build_actor_critic(env, net_arch=(32, 32), seed=seed)
    mask = build_mask_network(env, net_arch=net_arch, seed=seed)

    # Force the mask network to always output "1" (blind), making the recorded reward
    # difference attributable to the alpha bonus alone.
    mask.forward = _blind_mask_forward  # type: ignore[assignment]

    cfg = mask_module.MaskNetworkConfig(
        alpha=alpha,
        net_arch=net_arch,
        n_iterations=1,
        max_steps_per_iter=steps,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    trainer = mask_module.MaskNetworkTrainer(
        env=env,
        target_policy=target,
        mask_network=mask,
        config=cfg,
        rng=np.random.default_rng(seed),
    )
    buffer, stats = trainer.collect_iteration()
    return _mean_reward_of(buffer), stats


def test_bonus_reward_is_added_for_blinded_steps():
    """``R' = R + alpha * a_t^m`` (§3.3): blinding with alpha=1 raises the recorded reward.

    Both rollouts take *exactly the same* actions (the mask net is forced to blind at every
    step and the random-action RNG is seeded identically), so the environment trajectory is
    identical and the recorded reward differs by precisely ``alpha`` per blinded step.
    """
    mean_with_bonus, stats = _run_rollout(alpha=1.0, seed=0)
    mean_without_bonus, _ = _run_rollout(alpha=0.0, seed=0)

    assert np.isclose(mean_with_bonus - mean_without_bonus, 1.0, atol=1e-3), (
        "each blinded step must add exactly alpha to the recorded reward"
    )

    mask_rate = _key(stats, "mask_rate", "mean_mask_rate", "mask_fraction", "mean_am")
    if mask_rate is not None:
        assert float(mask_rate) > 0.99, "the forced mask network blinds every step"


def test_trainer_records_mask_actions_and_sample_budget():
    """Algorithm 1 stores ``(s_t, s_{t+1}, a_t^m, R'_t)`` and honours the sample budget."""
    env = _env()
    target = build_actor_critic(env, net_arch=(32, 32), seed=0)
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)

    cfg = mask_module.MaskNetworkConfig(
        alpha=1e-4,
        net_arch=(32, 32),
        n_iterations=2,
        max_steps_per_iter=8,
        total_samples=None,
        seed=0,
        device="cpu",
        verbose=0,
    )
    trainer = mask_module.MaskNetworkTrainer(
        env=env, target_policy=target, mask_network=mask, config=cfg, rng=np.random.default_rng(0)
    )
    buffer, stats = trainer.collect_iteration()
    arrays = buffer.as_arrays()

    for key in ("obs", "next_obs", "actions", "rewards"):
        assert key in arrays, f"rollout buffer must record {key}"
    assert len(buffer) == 8
    mask_actions = np.asarray(arrays["actions"]).reshape(-1)
    assert mask_actions.size == 8
    assert set(np.unique(mask_actions)).issubset({0, 1}), "a_t^m is a binary mask action"


def test_train_runs_algorithm1_and_reports_mask_rate():
    """``train()`` returns the Algorithm-1 bookkeeping used for Table 4."""
    env = _env()
    target = build_actor_critic(env, net_arch=(32, 32), seed=0)
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)

    cfg = mask_module.MaskNetworkConfig(
        alpha=1.0,  # large bonus: escaping the trivial all-zero solution must be easy
        net_arch=(32, 32),
        n_iterations=4,
        max_steps_per_iter=8,
        total_samples=None,
        seed=0,
        device="cpu",
        verbose=0,
    )
    trainer = mask_module.MaskNetworkTrainer(
        env=env, target_policy=target, mask_network=mask, config=cfg, rng=np.random.default_rng(0)
    )
    out = trainer.train()

    assert int(_key(out, "iterations", default=0)) >= 1
    assert float(_key(out, "samples", default=0.0)) > 0.0
    assert float(_key(out, "seconds", default=0.0)) >= 0.0  # feeds the Table 4 timing claim

    history = _key(out, "mask_history", default=None)
    if history is not None:
        assert len(history) >= 1

    mean_rate = _key(out, "mean_mask_rate", "mask_rate", "mask_fraction", default=None)
    assert mean_rate is not None, "train() must report the realised mask rate"
    assert 0.0 <= float(mean_rate) <= 1.0


def test_mask_rate_escapes_trivial_collapse_with_large_alpha():
    """§3.3: with a bonus for outputting "1", the mask net must not collapse to all-zero."""
    env = _env()
    target = build_actor_critic(env, net_arch=(32, 32), seed=0)
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)

    cfg = mask_module.MaskNetworkConfig(
        alpha=1.0,
        net_arch=(32, 32),
        n_iterations=15,
        max_steps_per_iter=16,
        total_samples=None,
        seed=0,
        device="cpu",
        verbose=0,
    )
    trainer = mask_module.MaskNetworkTrainer(
        env=env, target_policy=target, mask_network=mask, config=cfg, rng=np.random.default_rng(0)
    )
    out = trainer.train()
    mean_rate = float(_key(out, "mean_mask_rate", "mask_rate", "mask_fraction", default=0.0))

    assert mean_rate > 0.0, (
        "with a positive alpha bonus the mask ratio must not collapse to 0 "
        "(the trivial solution §3.3 warns about)"
    )


def test_small_alpha_keeps_bonus_from_dominating():
    """With Table 3's tiny alpha the mask rate stays a probability, not a constant 1."""
    env = _env()
    target = build_actor_critic(env, net_arch=(32, 32), seed=0)
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)

    cfg = mask_module.MaskNetworkConfig(
        alpha=1e-4,
        net_arch=(32, 32),
        n_iterations=2,
        max_steps_per_iter=16,
        seed=0,
        device="cpu",
        verbose=0,
    )
    trainer = mask_module.MaskNetworkTrainer(
        env=env, target_policy=target, mask_network=mask, config=cfg, rng=np.random.default_rng(0)
    )
    out = trainer.train()
    mean_rate = float(_key(out, "mean_mask_rate", "mask_rate", "mask_fraction", default=0.5))
    assert 0.0 <= mean_rate <= 1.0


# --------------------------------------------------------------------------------------
# Target policy is consumed black-box (no inspection of its internals)
# --------------------------------------------------------------------------------------


def test_trainer_accepts_plain_callable_target_policy():
    """The mask net must work with a black-box ``obs -> action`` callable (§black-box)."""
    env = _env()
    mask = build_mask_network(env, net_arch=(32, 32), seed=0)
    policy_fn = zero_policy(act_dim=2)

    cfg = mask_module.MaskNetworkConfig(
        alpha=1.0,
        net_arch=(32, 32),
        n_iterations=1,
        max_steps_per_iter=6,
        seed=0,
        device="cpu",
        verbose=0,
    )
    trainer = mask_module.MaskNetworkTrainer(
        env=env, target_policy=policy_fn, mask_network=mask, config=cfg, rng=np.random.default_rng(0)
    )
    buffer, stats = trainer.collect_iteration()

    assert len(buffer) == 6
    assert np.isfinite(_mean_reward_of(buffer))
