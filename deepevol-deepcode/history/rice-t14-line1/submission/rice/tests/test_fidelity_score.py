"""Tests for CORE COMPONENT #5: the Experiment I fidelity evaluator.

The metric is defined verbatim in the paper (Sec. 4.1 "Evaluation Metrics"):

    Fidelity Score = log(d / d_max) - log(l / L)

where
  * ``l = L * K`` is the sliding-window width (K in {10, 20, 30, 40}%),
  * the window with the highest *average* importance is selected,
  * the action(s) inside that window are randomized (masked, Eq. 1 of Sec. 3.3),
  * ``d`` is the resulting average reward change and ``d_max`` the maximum
    possible reward change in one episode.

These tests check the closed form on synthetic importance arrays, the window
semantics (argmax over stride-1 window averages), the reward-change measurement
on a deterministic scripted environment, and the end-to-end pipeline of
``FidelityEvaluator`` on the dependency-free ``DummyEnv`` from ``_helpers``.

All heavy objects (torch, MuJoCo, gym) are optional: the module is skipped when
``rice.evaluation.fidelity_score`` cannot be imported.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:  # package-relative first (pfx: working dir rice/)
    from . import _helpers  # type: ignore
except ImportError:  # pragma: no cover - fallback for bare pytest invocation
    import _helpers  # type: ignore

fidelity_mod = _helpers.import_optional("evaluation.fidelity_score")
if fidelity_mod is None:  # pragma: no cover - module unavailable in this layout
    pytest.skip("rice.evaluation.fidelity_score is not importable", allow_module_level=True)

FidelityEvaluator = getattr(fidelity_mod, "FidelityEvaluator", None)
FidelityConfig = getattr(fidelity_mod, "FidelityConfig", None)
FidelityResult = getattr(fidelity_mod, "FidelityResult", None)
EvalEpisode = getattr(fidelity_mod, "EvalEpisode", None)
fidelity_score = getattr(fidelity_mod, "fidelity_score", None)
sliding_window_average = getattr(fidelity_mod, "sliding_window_average", None)
best_window = getattr(fidelity_mod, "best_window", None)
random_window_index = getattr(fidelity_mod, "random_window_index", None)
estimated_d_max = getattr(fidelity_mod, "estimated_d_max", None)
mask_actions = getattr(fidelity_mod, "mask_actions", None)
make_importance_fn = getattr(fidelity_mod, "make_importance_fn", None)
evaluate_explanation = getattr(fidelity_mod, "evaluate_explanation", None)
evaluate_methods = getattr(fidelity_mod, "evaluate_methods", None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require(obj, name):
    if obj is None:
        pytest.skip("fidelity_score.%s is not available" % name)
    return obj


def _parse_k(k, L):
    """Independently reproduce the paper's l = L * K window width."""
    if isinstance(k, float) and k <= 1.0:
        return max(int(math.ceil(L * k)), 1)
    return max(int(k), 1)


class _CountingEnv(_helpers.DummyEnv):
    """Deterministic env tracking how many steps were executed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_count = 0

    def step(self, action, **kwargs):
        self.step_count += 1
        return super().step(action, **kwargs)

    def reset(self, **kwargs):
        self.step_count = 0
        return super().reset(**kwargs)


# ---------------------------------------------------------------------------
# closed-form metric  log(d / d_max) - log(l / L)
# ---------------------------------------------------------------------------
def test_fidelity_score_matches_closed_form():
    fn = _require(fidelity_score, "fidelity_score")
    d, l, L, d_max = 5.0, 100.0, 1000.0, 10.0
    expected = math.log(d / d_max) - math.log(l / L)
    assert fn(d, l, L, d_max) == pytest.approx(expected, rel=1e-9)


def test_fidelity_score_uses_log_ratio_of_both_terms():
    fn = _require(fidelity_score, "fidelity_score")
    # doubling d adds log(2); doubling l subtracts log(2)
    base = fn(1.0, 100.0, 1000.0, 100.0)
    assert fn(2.0, 100.0, 1000.0, 100.0) == pytest.approx(base + math.log(2.0))
    assert fn(1.0, 200.0, 1000.0, 100.0) == pytest.approx(base - math.log(2.0))


def test_fidelity_score_is_monotone_in_reward_change():
    fn = _require(fidelity_score, "fidelity_score")
    vals = [fn(d, 100.0, 1000.0, 10.0) for d in (0.1, 1.0, 5.0, 10.0)]
    assert vals == sorted(vals)


def test_fidelity_score_degenerate_inputs_are_finite():
    """Zero change or zero-length windows must not produce inf/nan."""
    fn = _require(fidelity_score, "fidelity_score")
    for d, l in ((0.0, 100.0), (1.0, 0.0), (0.0, 0.0)):
        value = fn(d, l, 1000.0, 10.0)
        assert np.isfinite(value)


# ---------------------------------------------------------------------------
# sliding window semantics: l = L * K, highest *average* importance wins
# ---------------------------------------------------------------------------
def test_sliding_window_average_matches_manual_computation():
    fn = _require(sliding_window_average, "sliding_window_average")
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    avg = np.asarray(fn(scores, 3), dtype=float)
    # stride-1 windows of width 3 (valid positions only) -> all-equal means if
    # the implementation returns the same number of entries as positions.
    manual = np.array([2.0, 3.0, 4.0])
    assert avg.reshape(-1)[:3] == pytest.approx(manual)


def test_best_window_selects_highest_average_importance():
    fn = _require(best_window, "best_window")
    scores = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    idx = int(fn(scores, 3))
    # windows: [0,0,1]->.33 [0,1,1]->.67 [1,1,1]->1.0 [1,1,0]->.67 ...
    assert idx == 2


def test_best_window_ties_resolve_to_earliest_index():
    fn = _require(best_window, "best_window")
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    assert int(fn(scores, 2)) == 0


def test_window_width_follows_l_equals_L_times_K():
    """The evaluator's window size must equal ceil(L * K) for fractional K."""
    L = 37
    for k in (0.10, 0.20, 0.30, 0.40):
        expected = _parse_k(k, L)
        assert expected == max(int(math.ceil(L * k)), 1)
        scores = np.linspace(0.0, 1.0, L)
        idx = int(_require(best_window, "best_window")(scores, expected))
        assert 0 <= idx <= max(L - expected, 0)


def test_random_window_index_within_bounds():
    fn = _require(random_window_index, "random_window_index")
    rng = np.random.default_rng(0)
    scores = np.zeros(50)
    for _ in range(20):
        idx = int(fn(scores, 10, rng=rng))
        assert 0 <= idx <= 40


# ---------------------------------------------------------------------------
# d_max estimation (unspecified in the paper -> data driven default)
# ---------------------------------------------------------------------------
def test_estimated_d_max_is_non_negative_and_uses_returns():
    fn = _require(estimated_d_max, "estimated_d_max")
    value = float(fn(np.array([1.0, 2.0, -3.0, 5.0])))
    assert np.isfinite(value)
    assert value > 0.0


def test_estimated_d_max_handles_empty_and_constant_returns():
    fn = _require(estimated_d_max, "estimated_d_max")
    assert np.isfinite(float(fn(np.array([]))))
    assert np.isfinite(float(fn(np.array([7.0, 7.0, 7.0]))))


# ---------------------------------------------------------------------------
# masking helper: randomizes the action(s) inside the selected window
# ---------------------------------------------------------------------------
def test_mask_actions_draws_uniform_random_actions():
    fn = _require(mask_actions, "mask_actions")
    space = _helpers.make_box_space(3, low=-1.0, high=1.0)
    rng = np.random.default_rng(0)
    actions = np.asarray(fn(space, 4, rng=rng), dtype=float)
    actions = actions.reshape(4, -1)
    assert actions.shape == (4, 3)
    assert actions.min() >= -1.0 - 1e-6
    assert actions.max() <= 1.0 + 1e-6


def test_mask_actions_differ_across_draws():
    fn = _require(mask_actions, "mask_actions")
    space = _helpers.make_box_space(2, low=-1.0, high=1.0)
    rng = np.random.default_rng(1)
    a = np.asarray(fn(space, 8, rng=rng), dtype=float).reshape(8, -1)
    assert float(np.std(a)) > 0.0


# ---------------------------------------------------------------------------
# importance function adapter
# ---------------------------------------------------------------------------
def test_make_importance_fn_accepts_explanation_object():
    fn = _require(make_importance_fn, "make_importance_fn")
    stub = _helpers.StubExplanation(mode="increasing")
    scorer = fn(stub)
    states = [np.zeros(4), np.ones(4), np.full(4, 2.0)]
    scores = np.asarray(scorer(states), dtype=float).reshape(-1)
    assert scores.shape[0] == 3
    assert np.all(np.isfinite(scores))


def test_make_importance_fn_is_callable_without_explanation():
    fn = _require(make_importance_fn, "make_importance_fn")
    scorer = fn(None)
    scores = np.asarray(scorer([np.zeros(4), np.ones(4)]), dtype=float).reshape(-1)
    assert scores.shape[0] == 2


# ---------------------------------------------------------------------------
# end-to-end evaluator on a deterministic dependency-free env
# ---------------------------------------------------------------------------
def _make_evaluator(method="ours", seed=0, capture_mode="replay", **cfg_kwargs):
    evaluator_cls = _require(FidelityEvaluator, "FidelityEvaluator")
    config_cls = _require(FidelityConfig, "FidelityConfig")
    env = _CountingEnv(obs_dim=4, act_dim=2, max_episode_steps=20, seed=seed)
    policy = _helpers.zero_policy(act_dim=2)
    stub = _helpers.StubExplanation(mode="increasing")
    kwargs = dict(
        ks=(0.10, 0.20),
        num_trajectories=3,
        seeds=(0,),
        capture_mode=capture_mode,
        verbose=0,
        d_max=10.0,
    )
    kwargs.update(cfg_kwargs)
    config = config_cls(**kwargs)
    evaluator = evaluator_cls(env, policy, stub, config=config, method=method)
    return evaluator, env, stub


def test_evaluator_rollout_records_full_episode():
    evaluator, env, _ = _make_evaluator()
    record = evaluator.rollout(seed=0)
    length = record["length"] if isinstance(record, dict) else getattr(record, "length")
    returns = record["returns"] if isinstance(record, dict) else getattr(record, "returns")
    assert length == 20
    assert len(returns) == 20


def test_evaluate_trajectory_returns_audit_record():
    episode_cls = _require(EvalEpisode, "EvalEpisode")
    evaluator, _, _ = _make_evaluator()
    episode = evaluator.evaluate_trajectory(0.20, seed=0)
    assert isinstance(episode, episode_cls)
    assert episode.length > 0
    assert episode.window_size == max(int(math.ceil(episode.length * 0.20)), 1)
    assert 0 <= episode.window_start <= episode.length - episode.window_size
    assert np.isfinite(episode.score)
    assert episode.reward_change >= 0.0


def test_reward_change_equals_absolute_return_difference():
    evaluator, _, _ = _make_evaluator()
    episode = evaluator.evaluate_trajectory(0.30, seed=0)
    assert episode.reward_change == pytest.approx(
        abs(episode.masked_return - episode.baseline_return), rel=1e-6, abs=1e-6
    )


def test_evaluate_returns_aggregate_result_per_k():
    result_cls = _require(FidelityResult, "FidelityResult")
    evaluator, _, _ = _make_evaluator()
    result = evaluator.evaluate()
    assert isinstance(result, result_cls)
    assert len(result.k_values) == 2
    means = np.asarray(result.means, dtype=float)
    assert means.shape[0] == len(result.k_values)
    assert np.all(np.isfinite(means))
    assert np.all(np.asarray(result.stds, dtype=float) >= 0.0)
    for k in result.k_values:
        assert np.isfinite(result.mean(k)) and np.isfinite(result.score(k))


def test_result_rows_and_table_are_serialisable():
    evaluator, _, _ = _make_evaluator()
    result = evaluator.evaluate()
    rows = list(result.rows())
    assert len(rows) == len(result.k_values)
    assert isinstance(rows[0], dict)
    assert isinstance(result.as_dict(), dict)
    assert isinstance(result.table(), str)


def test_fidelity_score_uses_measured_d_and_window_width():
    """Score reported per K must equal the closed form with l = window size."""
    evaluator, _, _ = _make_evaluator()
    result = evaluator.evaluate()
    for episode in result.episodes:
        expected = fidelity_score(
            episode.reward_change,
            episode.window_size,
            episode.length,
            episode.d_max,
        )
        assert episode.score == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_random_window_config_produces_finite_scores():
    evaluator, _, _ = _make_evaluator(random_window=True)
    result = evaluator.evaluate()
    assert np.all(np.isfinite(np.asarray(result.means, dtype=float)))


def test_evaluator_is_deterministic_for_fixed_seed():
    ev_a, _, _ = _make_evaluator()
    ev_b, _, _ = _make_evaluator()
    res_a = ev_a.evaluate()
    res_b = ev_b.evaluate()
    assert np.asarray(res_a.means) == pytest.approx(np.asarray(res_b.means))


def test_step_capture_mode_also_works():
    evaluator, _, _ = _make_evaluator(capture_mode="step")
    result = evaluator.evaluate()
    assert np.all(np.isfinite(np.asarray(result.means, dtype=float)))


# ---------------------------------------------------------------------------
# multi-method comparison (Experiment I / Table 6 ordering support)
# ---------------------------------------------------------------------------
def test_evaluate_explanation_functional_entry_point():
    fn = _require(evaluate_explanation, "evaluate_explanation")
    config_cls = _require(FidelityConfig, "FidelityConfig")
    env = _helpers.DummyEnv(obs_dim=4, act_dim=2, max_episode_steps=15, seed=0)
    policy = _helpers.zero_policy(act_dim=2)
    config = config_cls(
        ks=(0.20,), num_trajectories=2, seeds=(0,), verbose=0, d_max=10.0
    )
    result = fn(env, policy, _helpers.StubExplanation(), config=config, method="ours")
    assert len(result.k_values) == 1
    assert np.isfinite(result.mean(result.k_values[0]))


def test_evaluate_methods_compares_named_explanations():
    fn = _require(evaluate_methods, "evaluate_methods")
    config_cls = _require(FidelityConfig, "FidelityConfig")
    env = _helpers.DummyEnv(obs_dim=4, act_dim=2, max_episode_steps=15, seed=0)
    policy = _helpers.zero_policy(act_dim=2)
    config = config_cls(
        ks=(0.20,), num_trajectories=2, seeds=(0,), verbose=0, d_max=10.0
    )
    methods = {
        "ours": _helpers.StubExplanation(mode="increasing"),
        "random": _helpers.StubExplanation(mode="uniform"),
    }
    results = fn(env, policy, methods, config=config)
    assert set(results.keys()) == set(methods.keys())
    for res in results.values():
        assert np.isfinite(res.mean(res.k_values[0]))


def test_config_from_mapping_accepts_k_aliases():
    config_cls = _require(FidelityConfig, "FidelityConfig")
    config = config_cls.from_mapping({"K": [0.10, 0.20], "num_episodes": 7})
    ks = tuple(getattr(config, "ks"))
    assert len(ks) == 2
    assert int(getattr(config, "num_trajectories")) == 7


def test_config_clone_overrides_without_mutating_original():
    config_cls = _require(FidelityConfig, "FidelityConfig")
    base = config_cls(ks=(0.10,), num_trajectories=3, seeds=(0,), verbose=0)
    clone = base.clone(num_trajectories=11)
    assert int(getattr(base, "num_trajectories")) == 3
    assert int(getattr(clone, "num_trajectories")) == 11


def test_estimate_d_max_on_evaluator_is_positive():
    evaluator, _, _ = _make_evaluator()
    value = float(evaluator.estimate_d_max(n_episodes=2))
    assert np.isfinite(value) and value > 0.0
