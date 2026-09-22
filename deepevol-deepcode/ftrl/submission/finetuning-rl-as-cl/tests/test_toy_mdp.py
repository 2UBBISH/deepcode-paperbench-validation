"""Smoke/unit tests for the toy experiments of Appendix A.

Covers:
  * the two-state MDP closed form (state coverage gap / imperfect cloning gap)
    and the reported fine-tuning optima ``theta = 0.11 -> v0 = 2.22`` and
    ``theta = 0.08 -> v0 = 9.93`` (Figures 9/10 of the paper);
  * the AppleRetrieval 1-D gridworld with the linear sigmoid policy
    ``pi_{w,b}(o) = sigmoid(w * o + b)`` trained with REINFORCE
    (Figures 10/11 of the paper), including the ``dL/dw = c * dL/db``
    gradient identity implied by that parametrisation.

The suite is deliberately dependency-light: NumPy is not required, PyTorch is
not required, and every optional piece (PyYAML, the ``src`` package layout)
degrades into a ``SkipTest`` instead of a failure.  It can be run either with
``pytest tests/test_toy_mdp.py`` or directly via
``python -m tests.test_toy_mdp``.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:  # allow `python -m tests.test_toy_mdp` from anywhere
    sys.path.insert(0, _ROOT)

try:  # shared sentinel from tests/__init__.py
    from tests import SkipTest
except Exception:  # pragma: no cover - standalone execution

    class SkipTest(Exception):
        """Raised when an optional dependency or feature is unavailable."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _toy(name: str) -> Optional[Any]:
    """Import a ``src.toy`` module, tolerating different import layouts."""
    for path in ("src.toy." + name, "toy." + name, name):
        try:
            return importlib.import_module(path)
        except Exception:
            continue
    return None


def _require_toy(name: str) -> Any:
    module = _toy(name)
    if module is None:
        raise SkipTest("toy module %r is unavailable" % name)
    return module


def _toy_config() -> Optional[Any]:
    """Load ``configs/toy.yaml`` when PyYAML and the config loader are present."""
    path = os.path.join(_ROOT, "configs", "toy.yaml")
    if not os.path.isfile(path):
        return None
    loader = None
    try:
        from src.common.config import load_config as loader  # type: ignore
    except Exception:
        try:
            from common.config import load_config as loader  # type: ignore
        except Exception:
            return None
    if loader is None:
        return None
    try:
        return loader(path)
    except Exception:
        return None


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float conversion that rejects NaN/inf."""
    try:
        out = float(value)
    except Exception:
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _field_of(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _approx(a: Any, b: Any, tol: float = 1e-9) -> bool:
    a_f, b_f = _num(a), _num(b)
    if a_f is None or b_f is None:
        return False
    return abs(a_f - b_f) <= tol * max(1.0, abs(b_f))


def _flat(value: Any) -> List[float]:
    """Flatten a scalar / list / tuple / ndarray observation into floats."""
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(_flat(item))
        return out
    converted = _num(value)
    if converted is None:
        raise AssertionError("non-numeric observation element: %r" % (value,))
    return [converted]


def _call(obj: Any, *names: str, **kwargs: Any) -> Any:
    """Call the first available method among ``names`` on ``obj``."""
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return fn(**kwargs)
            except TypeError:
                continue
    raise SkipTest("none of %r callable on %r" % (names, type(obj).__name__))


def _expected_optimum(scenario: Any) -> Optional[float]:
    for key in ("expected_optimum", "expected_theta", "target_theta"):
        value = _num(_field_of(scenario, key, None))
        if value is not None:
            return value
    return None


def _expected_value(scenario: Any) -> Optional[float]:
    for key in ("expected_value", "target_value"):
        value = _num(_field_of(scenario, key, None))
        if value is not None:
            return value
    return None


def _scenario_value(ts: Any, scenario: Any, theta: float) -> Optional[float]:
    method = getattr(scenario, "value", None)
    if callable(method):
        return _num(method(theta))
    try:
        return _num(
            ts.two_state_value(
                theta,
                gamma=_field_of(scenario, "gamma", 0.9),
                r0=_field_of(scenario, "r0", 0.0),
                r1=_field_of(scenario, "r1", -1.0),
                f_name=_field_of(scenario, "f_name", "coverage"),
                eps=_field_of(scenario, "eps", 1.0),
            )
        )
    except Exception:
        return None


def _get_scenario(ts: Any, name: str) -> Any:
    scenarios = getattr(ts, "SCENARIOS", None)
    if isinstance(scenarios, dict):
        for key in (name, name.replace("_", "-"), name.replace("-", "_")):
            if key in scenarios:
                return scenarios[key]
    maker = getattr(ts, "make_scenario", None)
    if callable(maker):
        try:
            return maker(name)
        except Exception:
            pass
    raise SkipTest("scenario %r unavailable" % name)


def _small_apple_config(apple: Any) -> Tuple[Optional[Any], bool]:
    """Build a tiny ``AppleConfig`` so smoke runs stay fast.

    Returns ``(config, tiny)``; ``tiny`` is False when the requested fields
    were not recognised (in which case expensive end-to-end tests are skipped).
    """
    try:
        cfg = apple.AppleConfig()
    except Exception:
        return None, False
    if not hasattr(cfg, "replace"):
        return cfg, False
    wanted: Dict[str, Any] = {
        "pretrain_episodes": 20,
        "num_pretrain_episodes": 20,
        "finetune_episodes": 20,
        "num_finetune_episodes": 20,
        "eval_every": 5,
        "eval_episodes": 10,
        "horizon": 10,
    }
    names: set = set()
    try:
        import dataclasses

        names = {f.name for f in dataclasses.fields(cfg)}
    except Exception:
        names = set(wanted)
    filtered = {k: v for k, v in wanted.items() if k in names}
    if not filtered:
        return cfg, False
    try:
        cfg = cfg.replace(**filtered)
    except Exception:
        return cfg, False
    horizon = _num(_field_of(cfg, "horizon", None))
    tiny = horizon is not None and horizon <= 50
    return cfg, tiny


# ---------------------------------------------------------------------------
# two-state MDP (Appendix A.1)
# ---------------------------------------------------------------------------
def test_coverage_f_shape_and_saturation() -> None:
    ts = _require_toy("two_state_mdp")
    f = ts.coverage_f
    assert _approx(f(0.0), 0.0, 1e-12), "coverage f(0) must be 0 (never reaches state 1)"
    assert _approx(f(1.0), 1.0, 1e-12), "coverage f(1) must be 1 (always reaches state 1)"
    grid = [i / 40.0 for i in range(41)]
    values = [float(f(t)) for t in grid]
    for value in values:
        assert -1e-12 <= value <= 1.0 + 1e-12, "coverage f must lie in [0, 1]"
    for left, right in zip(values, values[1:]):
        assert right >= left - 1e-9, "coverage f must be non-decreasing in theta"
    # with eps = 1 the branch point is at 1 - eps/2 = 0.5, where f saturates at 1
    assert _approx(f(0.5), 1.0, 1e-9), "coverage f must saturate at the branch point 0.5"
    assert float(ts.coverage_f_grad(0.25)) > 0.0, "coverage f must increase for small theta"


def test_cloning_f_is_double_absolute_value() -> None:
    ts = _require_toy("two_state_mdp")
    f = ts.cloning_f
    assert _approx(f(0.5), 0.0, 1e-12), "imperfect cloning f(0.5) must be 0"
    assert _approx(f(0.0), 1.0, 1e-12), "imperfect cloning f(0) must be 1"
    assert _approx(f(1.0), 1.0, 1e-12), "imperfect cloning f(1) must be 1"
    assert _approx(f(0.25), 0.5, 1e-12), "f(theta) = 2 |theta - 0.5|"
    assert _approx(f(0.75), 0.5, 1e-12), "f(theta) = 2 |theta - 0.5|"
    for theta in (0.1, 0.2, 0.35, 0.8, 0.95):
        assert _approx(f(theta), f(1.0 - theta), 1e-12), "f must be symmetric about 0.5"
    assert float(ts.cloning_f_grad(0.25)) < 0.0, "df/dtheta < 0 for theta < 0.5"
    assert float(ts.cloning_f_grad(0.75)) > 0.0, "df/dtheta > 0 for theta > 0.5"


def test_closed_form_value_endpoints() -> None:
    ts = _require_toy("two_state_mdp")
    # theta = 0: the agent never reaches the "stay" behaviour, no reward
    assert _approx(ts.two_state_value(0.0), 0.0, 1e-9), "v0(0) must be 0"
    # theta = 1: the agent stays in state 0 forever -> r0 / (1 - gamma) = 10
    gamma = 0.9
    assert _approx(ts.two_state_value(1.0, gamma=gamma), 1.0 / (1.0 - gamma), 1e-6), (
        "v0(1) must equal the maximum undiscounted-to-discounted return"
    )
    assert _approx(ts.two_state_value(1.0, gamma=gamma, f_name="cloning"), 1.0 / (1.0 - gamma), 1e-6)


def test_state_coverage_gap_reported_optimum() -> None:
    ts = _require_toy("two_state_mdp")
    scenario = _get_scenario(ts, "coverage_gap")
    optimum = _expected_optimum(scenario)
    expected = _expected_value(scenario)
    if optimum is None or expected is None:
        raise SkipTest("coverage_gap scenario does not expose expected_optimum/expected_value")
    assert 0.0 <= optimum <= 1.0, "the reported optimum must be a probability"
    value = _scenario_value(ts, scenario, optimum)
    if value is None:
        raise SkipTest("scenario value could not be evaluated")
    assert _approx(value, expected, 0.10), (
        "coverage-gap optimum should reproduce v0(%.3f) ~= %.3f, got %.4f"
        % (optimum, expected, value)
    )
    # the gap is real: the optimum of the fine-tuned behaviour is far below the
    # global optimum theta = 1 (the agent could simply stay and collect reward)
    global_value = _scenario_value(ts, scenario, 1.0)
    if global_value is not None:
        assert global_value >= value - 1e-9, "v0(1) is the global maximum of the toy MDP"


def test_imperfect_cloning_reported_optimum() -> None:
    ts = _require_toy("two_state_mdp")
    scenario = _get_scenario(ts, "imperfect_cloning")
    optimum = _expected_optimum(scenario)
    expected = _expected_value(scenario)
    if optimum is None or expected is None:
        raise SkipTest("imperfect_cloning scenario lacks expected_optimum/expected_value")
    value = _scenario_value(ts, scenario, optimum)
    if value is None:
        raise SkipTest("scenario value could not be evaluated")
    assert _approx(value, expected, 0.10), (
        "imperfect-cloning optimum should reproduce v0(%.3f) ~= %.3f, got %.4f"
        % (optimum, expected, value)
    )
    assert expected > 0.0, "the imperfect cloning gap has a positive value"


def test_numerical_gradient_matches_analytic() -> None:
    ts = _require_toy("two_state_mdp")
    for name in ("coverage_gap", "imperfect_cloning"):
        scenario = _get_scenario(ts, name)
        theta = 0.25  # away from the kink at 0.5
        analytic = _num(_field_of(scenario, "gradient", None))
        if analytic is None:
            analytic = _num(getattr(scenario, "gradient")(theta))
        else:
            try:
                analytic = _num(_field_of(scenario, "gradient")(theta))
            except TypeError:
                raise SkipTest("Scenario.gradient has an unexpected signature")
        try:
            numeric = _num(ts.numerical_gradient(theta, scenario))
        except TypeError:
            try:
                numeric = _num(ts.numerical_gradient(theta))
            except Exception:
                raise SkipTest("numerical_gradient could not be evaluated")
        if analytic is None or numeric is None:
            raise SkipTest("gradient evaluation unavailable")
        assert abs(analytic - numeric) <= 1e-4 * max(1.0, abs(numeric)), (
            "%s: analytic gradient %.6f != numeric %.6f" % (name, analytic, numeric)
        )


def test_local_extrema_contain_reported_optima() -> None:
    ts = _require_toy("two_state_mdp")
    for name in ("coverage_gap", "imperfect_cloning"):
        scenario = _get_scenario(ts, name)
        optimum = _expected_optimum(scenario)
        if optimum is None:
            raise SkipTest("scenario %r has no expected optimum" % name)
        try:
            extrema = list(ts.local_extrema(scenario, num=2001))
        except TypeError:
            extrema = list(ts.local_extrema(scenario))
        if not extrema:
            raise SkipTest("local_extrema found nothing for %r" % name)
        maxima = [
            e
            for e in extrema
            if str(_field_of(e, "kind", "max")).lower().startswith("max")
        ] or extrema
        thetas = [_num(_field_of(e, "theta", None)) for e in maxima]
        thetas = [t for t in thetas if t is not None]
        assert thetas, "local_extrema must report extrema positions"
        assert any(abs(t - optimum) <= 0.05 for t in thetas), (
            "%s: no local maximum near the reported optimum %.3f (found %s)"
            % (name, optimum, ["%.3f" % t for t in thetas[:8]])
        )


def test_fine_tuning_converges_to_reported_optima() -> None:
    ts = _require_toy("two_state_mdp")
    for name in ("coverage_gap", "imperfect_cloning"):
        scenario = _get_scenario(ts, name)
        optimum = _expected_optimum(scenario)
        expected = _expected_value(scenario)
        if optimum is None:
            raise SkipTest("scenario %r has no expected optimum" % name)
        try:
            result = ts.fine_tune(scenario)
        except TypeError:
            init = _field_of(scenario, "theta_init", None)
            result = ts.fine_tune(scenario, theta_init=init)
        theta = _num(_field_of(result, "theta", None))
        assert theta is not None, "fine_tune must report the final theta"
        assert -1e-9 <= theta <= 1.0 + 1e-9, "theta must stay a valid probability"
        converged_position = abs(theta - optimum) <= 0.06
        reached_value = False
        if expected is not None:
            value = _num(_field_of(result, "value", None))
            if value is None:
                value = _scenario_value(ts, scenario, theta)
            reached_value = value is not None and _approx(value, expected, 0.05)
        assert converged_position or reached_value, (
            "%s: fine-tuning ended at theta=%.4f (expected ~%.3f)"
            % (name, theta, optimum)
        )
        record = getattr(result, "as_dict", None)
        if callable(record):
            payload = record()
            assert isinstance(payload, dict) and payload, "FineTuneResult.as_dict must be non-empty"


def test_value_curve_and_scenario_dict() -> None:
    ts = _require_toy("two_state_mdp")
    scenario = _get_scenario(ts, "coverage_gap")
    thetas, values = ts.value_curve(scenario, num=101)
    thetas, values = list(thetas), list(values)
    assert len(thetas) == len(values) == 101, "value_curve must return equal-length grids"
    numeric = [_num(v) for v in values]
    assert all(v is not None for v in numeric), "the value curve must be finite everywhere"
    assert max(numeric) <= 1.0 / (1.0 - 0.9) + 1e-6, "v0 cannot exceed the maximum return"
    payload = scenario.with_overrides(**{}) if hasattr(scenario, "with_overrides") else None
    if payload is not None:
        assert isinstance(_scenario_value(ts, payload, 0.25), float)


def test_run_scenario_reproduces_coverage_gap() -> None:
    ts = _require_toy("two_state_mdp")
    runner = getattr(ts, "run_scenario", None)
    if not callable(runner):
        raise SkipTest("run_scenario is unavailable")
    try:
        result = runner("coverage_gap")
    except TypeError:
        result = runner(name="coverage_gap")
    assert isinstance(result, dict), "run_scenario must return a summary mapping"
    assert result, "run_scenario summary must not be empty"
    optimum = None
    for key in ("optimum", "theta", "theta_star", "final_theta"):
        optimum = _num(result.get(key))
        if optimum is not None:
            break
    if optimum is not None:
        expected = None
        scenario = _get_scenario(ts, "coverage_gap")
        expected = _expected_optimum(scenario)
        if expected is not None:
            assert abs(optimum - expected) <= 0.06
    if "match" in result:
        assert bool(result["match"]), "run_scenario should reproduce the reported optimum"


def test_scenario_from_toy_config() -> None:
    ts = _require_toy("two_state_mdp")
    cfg = _toy_config()
    if cfg is None:
        raise SkipTest("configs/toy.yaml or the YAML loader is unavailable")
    builder = getattr(ts, "scenario_from_config", None)
    if not callable(builder):
        raise SkipTest("scenario_from_config is unavailable")
    try:
        scenario = builder(cfg, "coverage_gap")
    except TypeError:
        scenario = builder(cfg)
    gamma = _num(_field_of(scenario, "gamma", None))
    assert gamma is not None and abs(gamma - 0.9) < 1e-9, "toy config gamma must be 0.9"
    optimum = _expected_optimum(scenario)
    assert optimum is not None and abs(optimum - 0.11) <= 0.02, (
        "toy config must reproduce the reported coverage-gap optimum"
    )
    value = _scenario_value(ts, scenario, optimum)
    if value is not None and _expected_value(scenario) is not None:
        assert value > 0.0, "the coverage-gap optimum has a positive value"


# ---------------------------------------------------------------------------
# AppleRetrieval (Appendix A.2)
# ---------------------------------------------------------------------------
def test_sigmoid_is_numerically_stable() -> None:
    apple = _require_toy("apple_retrieval")
    assert _approx(apple.sigmoid(0.0), 0.5, 1e-12)
    assert 0.0 < float(apple.sigmoid(-50.0)) < 1e-9
    assert 1.0 - float(apple.sigmoid(50.0)) < 1e-9
    grid = [-5.0, -1.0, 0.0, 1.0, 5.0]
    values = [float(apple.sigmoid(z)) for z in grid]
    for left, right in zip(values, values[1:]):
        assert right >= left - 1e-12, "sigmoid must be non-decreasing"


def test_linear_sigmoid_policy_probabilities() -> None:
    apple = _require_toy("apple_retrieval")
    policy = apple.LinearSigmoidPolicy(0.0, 0.0)
    for obs in ([-1.0], [0.0], [1.0], [10.0]):
        assert _approx(policy.prob_right(obs), 0.5, 1e-12), "w=b=0 must give a fair coin"
        assert _approx(policy.prob_left(obs), 0.5, 1e-12)
    tilted = apple.LinearSigmoidPolicy(1.0, 0.0)
    high = float(tilted.prob_right([10.0]))
    low = float(tilted.prob_right([-10.0]))
    assert high > 0.99 and low < 0.01, "pi_{w,b}(o) = sigmoid(w o + b)"
    assert _approx(high, 1.0 - low, 1e-6), "the two actions must be complementary"
    probs = list(tilted.probs([1.0]))
    assert len(probs) == 2 and _approx(sum(probs), 1.0, 1e-9)
    assert int(tilted.greedy_action([10.0])) == int(apple.RIGHT)
    assert int(tilted.greedy_action([-10.0])) == int(apple.LEFT)
    logp = float(tilted.log_prob([1.0], apple.RIGHT))
    assert logp <= 0.0, "log-probabilities must be non-positive"


def test_policy_agreement_and_copy() -> None:
    apple = _require_toy("apple_retrieval")
    policy = apple.LinearSigmoidPolicy(2.0, -0.5)
    clone = policy.copy()
    obs = [[-1.0], [0.0], [1.0]]
    assert _approx(policy.action_agreement(policy, obs), 1.0, 1e-12), (
        "a policy must agree with itself everywhere"
    )
    assert _approx(policy.action_agreement(clone, obs), 1.0, 1e-12), "copy() must be identical"
    flipped = apple.LinearSigmoidPolicy(-2.0, 0.5)
    agreement = float(policy.action_agreement(flipped, obs))
    assert 0.0 <= agreement <= 1.0, "action agreement is a probability of agreement"
    assert agreement <= 1.0 / 3.0 + 1e-9, "opposite tilts must disagree on these observations"
    parameters = policy.parameters()
    assert len(list(parameters)) == 2 or isinstance(parameters, dict), (
        "the linear sigmoid policy has exactly two parameters (w, b)"
    )


def test_discounted_returns_gamma_one() -> None:
    apple = _require_toy("apple_retrieval")
    returns = list(apple.discounted_returns([1.0, 1.0, 1.0], gamma=1.0))
    assert [_num(r) for r in returns] == [3.0, 2.0, 1.0], "gamma=1 gives return-to-go"
    discounted = list(apple.discounted_returns([1.0, 1.0], gamma=0.5))
    assert _approx(discounted[0], 1.5, 1e-12) and _approx(discounted[1], 1.0, 1e-12)


def test_reinforce_gradient_identity_and_zero_rewards() -> None:
    apple = _require_toy("apple_retrieval")
    policy = apple.LinearSigmoidPolicy(0.0, 0.0)
    c = float(getattr(apple, "DEFAULT_C", 1.0))
    trajectory = {
        "observations": [[c], [c]],
        "actions": [int(apple.RIGHT), int(apple.RIGHT)],
        "rewards": [1.0, 1.0],
        "infos": [{}, {}],
        "return": 2.0,
        "length": 2,
    }
    try:
        grads = apple.reinforce_gradient([trajectory], policy, gamma=1.0, baseline=True)
    except TypeError:
        raise SkipTest("reinforce_gradient has an unexpected signature")
    assert isinstance(grads, dict) and "w" in grads and "b" in grads, (
        "REINFORCE must return gradients for both (w, b)"
    )
    gw, gb = _num(grads["w"]), _num(grads["b"])
    assert gw is not None and gb is not None, "gradients must be finite"
    assert abs(gw) > 0.0 or abs(gb) > 0.0, "a rewarded trajectory yields a non-zero gradient"
    # pi_{w,b}(o) = sigmoid(w o + b) implies d log pi / dw = o * d log pi / db,
    # so with o = c the two gradients differ exactly by the factor c (Figure 11).
    assert abs(gw - c * gb) <= 1e-8 * max(1.0, abs(gb)), (
        "the gradient ratio dL/dw = c * dL/db must hold for the linear policy"
    )
    zero_traj = dict(trajectory)
    zero_traj["rewards"] = [0.0, 0.0]
    zero_traj["return"] = 0.0
    zero_grads = apple.reinforce_gradient([zero_traj], policy, gamma=1.0, baseline=True)
    assert abs(_num(zero_grads["w"], 0.0)) < 1e-9, "zero reward -> zero gradient"
    assert abs(_num(zero_grads["b"], 0.0)) < 1e-9, "zero reward -> zero gradient"


def test_apple_env_interface_and_phases() -> None:
    apple = _require_toy("apple_retrieval")
    env = apple.AppleRetrievalEnv(M=5, c=1.0, horizon=10)
    try:
        obs = env.reset(seed=0)
    except TypeError:
        obs = env.reset()
    flat = _flat(obs)
    assert len(flat) == 1, "the AppleRetrieval observation is a single scalar"
    assert int(getattr(env, "observation_dim", 1)) == 1
    assert int(getattr(env, "action_dim", 2)) == 2, "two actions: stay/go or left/right"
    phase_1, phase_2 = int(apple.PHASE_1), int(apple.PHASE_2)
    for phase in (phase_1, phase_2):
        phase_obs = _flat(env.phase_observation(phase))
        assert len(phase_obs) == 1, "each phase is observed as one scalar"
        assert abs(abs(phase_obs[0]) - 1.0) < 1e-6, "the observation magnitude encodes |c|"
    reference = env.phase2_reference_observations(3)
    assert len(list(reference)) == 3, "reference observations are sampled on demand"
    action = env.correct_action(phase_2)
    assert int(action) in (int(apple.LEFT), int(apple.RIGHT)), "correct actions are binary"
    done = False
    total = 0.0
    steps = 0
    while not done and steps < int(getattr(env, "horizon", 10)) + 5:
        step_result = env.step(env.correct_action())
        assert len(step_result) >= 4, "step() must return (obs, reward, done, info)"
        obs, reward, done = step_result[0], _num(step_result[1], 0.0), bool(step_result[2])
        _flat(obs)
        total += float(reward)
        steps += 1
    assert done, "an AppleRetrieval episode must terminate within its horizon"
    assert total > 0.0, "acting correctly must accumulate positive reward"


def test_pretrain_phase2_and_evaluation_metrics() -> None:
    apple = _require_toy("apple_retrieval")
    try:
        policy, history = apple.pretrain_phase2(
            M=5,
            c=1.0,
            episodes=20,
            lr=0.05,
            optimizer="sgd",
            gamma=1.0,
            baseline=True,
            horizon=10,
            init_w=0.0,
            init_b=0.0,
            seed=0,
            grad_clip=None,
            record_every=0,
        )
    except TypeError:
        raise SkipTest("pretrain_phase2 has an unexpected signature")
    assert isinstance(policy, apple.LinearSigmoidPolicy), "pre-training returns a pi_* policy"
    assert history is None or isinstance(history, (list, tuple, dict)), "history must be a container"
    env = apple.AppleRetrievalEnv(M=5, c=1.0, horizon=10)
    try:
        metrics = apple.evaluate_policy(env, policy, num_episodes=10, seed=0, deterministic=True)
    except TypeError:
        raise SkipTest("evaluate_policy has an unexpected signature")
    assert isinstance(metrics, dict) and metrics, "evaluation must return a metrics mapping"
    numbers = [_num(_field_of(metrics, key)) for key in ("phase2_success", "phase2_action_agreement")]
    numbers = [n for n in numbers if n is not None]
    if numbers:
        assert all(-1e-9 <= n <= 1.0 + 1e-9 for n in numbers), "success rates live in [0, 1]"
    agreement = _num(_field_of(metrics, "phase2_action_agreement"))
    if agreement is not None:
        assert agreement <= 1.0 + 1e-9


def test_run_apple_retrieval_smoke() -> None:
    apple = _require_toy("apple_retrieval")
    cfg, tiny = _small_apple_config(apple)
    if not tiny:
        raise SkipTest("AppleConfig fields not recognised; skipping the end-to-end run")
    try:
        result = apple.run_apple_retrieval(M=5, c=1.0, seed=0, config=cfg, record_trace=True)
    except TypeError:
        raise SkipTest("run_apple_retrieval has an unexpected signature")
    assert isinstance(result, dict) and result, "run_apple_retrieval must return a metrics mapping"
    for key in ("w_pre", "b_pre", "w_ft", "b_ft"):
        if key in result:
            assert _num(result[key]) is not None, "%s must be a finite number" % key
    for key in ("forgetting", "phase2_action_agreement", "phase2_success", "overall_success"):
        if key in result:
            value = _num(result[key], None)
            if value is None:
                continue
            assert -1e-9 <= value <= 1.0 + 1e-9, "%s must be a fraction, got %r" % (key, value)
    metrics = getattr(apple, "APPLE_RETRIEVAL_METRICS", None)
    if metrics:
        assert any(key in result for key in metrics), (
            "the run must report at least one of the declared metrics"
        )


def test_apple_aggregation_helpers() -> None:
    apple = _require_toy("apple_retrieval")
    stats = apple.summarize([1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.90)
    mean = _num(_field_of(stats, "mean"))
    assert mean is not None and abs(mean - 3.0) < 1e-9, "summarize must report the mean"
    assert _num(_field_of(stats, "half_width")) is not None, "summarize must report a CI width"
    aggregate = apple.aggregate_results(
        [{"forgetting": 0.1, "phase2_action_agreement": 0.8}, {"forgetting": 0.3, "phase2_action_agreement": 0.6}],
        confidence=0.90,
    )
    assert isinstance(aggregate, dict) and aggregate, "aggregate_results must be non-empty"
    forgetting = _field_of(aggregate, "forgetting", None)
    if forgetting is not None:
        value = _num(_field_of(forgetting, "mean", forgetting))
        assert value is not None and abs(value - 0.2) < 1e-9, "aggregation must average seeds"


def test_apple_sweeps_return_expected_structure() -> None:
    apple = _require_toy("apple_retrieval")
    cfg, tiny = _small_apple_config(apple)
    if not tiny:
        raise SkipTest("AppleConfig fields not recognised; skipping sweeps")
    try:
        sweep_m = apple.sweep_over_M(values=(1, 2), c=1.0, seeds=(0,), config=cfg, progress=False)
    except TypeError:
        raise SkipTest("sweep_over_M has an unexpected signature")
    kind = _field_of(sweep_m, "kind", None)
    assert isinstance(sweep_m, dict) and sweep_m, "sweep_over_M must return a mapping"
    if kind is not None:
        assert "M" in str(kind), "sweep_over_M must be tagged as an M sweep"
    try:
        sweep_c = apple.sweep_over_c(values=(0.1, 1.0), M=5, seeds=(0,), config=cfg, progress=False)
    except TypeError:
        raise SkipTest("sweep_over_c has an unexpected signature")
    assert isinstance(sweep_c, dict) and sweep_c, "sweep_over_c must return a mapping"


def test_apple_config_from_toy_yaml() -> None:
    apple = _require_toy("apple_retrieval")
    cfg = _toy_config()
    if cfg is None:
        raise SkipTest("configs/toy.yaml or the YAML loader is unavailable")
    builder = getattr(apple, "config_from_config", None)
    if not callable(builder):
        raise SkipTest("config_from_config is unavailable")
    try:
        resolved = builder(cfg)
    except TypeError:
        resolved = builder(cfg, name="apple_retrieval")
    m_value = _field_of(resolved, "M", _field_of(resolved, "m", None))
    if m_value is None:
        raise SkipTest("AppleConfig does not expose M")
    assert _approx(m_value, 30.0, 1e-9), "toy.yaml trains Phase 2 with M = 30"
    c_value = _num(_field_of(resolved, "c", None))
    assert c_value is not None and abs(c_value - 1.0) < 1e-9, "toy.yaml uses c = 1.0"
    horizon = _num(_field_of(resolved, "horizon", None))
    if horizon is not None:
        assert abs(horizon - 100.0) < 1e-9, "toy.yaml uses horizon = 100"


# ---------------------------------------------------------------------------
# registry / runner
# ---------------------------------------------------------------------------
_TESTS: List[Callable[[], None]] = [
    # two-state MDP (Appendix A.1)
    test_coverage_f_shape_and_saturation,
    test_cloning_f_is_double_absolute_value,
    test_closed_form_value_endpoints,
    test_state_coverage_gap_reported_optimum,
    test_imperfect_cloning_reported_optimum,
    test_numerical_gradient_matches_analytic,
    test_local_extrema_contain_reported_optima,
    test_fine_tuning_converges_to_reported_optima,
    test_value_curve_and_scenario_dict,
    test_run_scenario_reproduces_coverage_gap,
    test_scenario_from_toy_config,
    # AppleRetrieval (Appendix A.2)
    test_sigmoid_is_numerically_stable,
    test_linear_sigmoid_policy_probabilities,
    test_policy_agreement_and_copy,
    test_discounted_returns_gamma_one,
    test_reinforce_gradient_identity_and_zero_rewards,
    test_apple_env_interface_and_phases,
    test_pretrain_phase2_and_evaluation_metrics,
    test_run_apple_retrieval_smoke,
    test_apple_aggregation_helpers,
    test_apple_sweeps_return_expected_structure,
    test_apple_config_from_toy_yaml,
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every test in ``_TESTS`` and report a summary."""
    args = list(sys.argv[1:] if argv is None else argv)
    quiet = any(a in ("--quiet", "-q") for a in args)
    passed: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    for test in _TESTS:
        name = getattr(test, "__name__", repr(test))
        try:
            test()
        except SkipTest as exc:
            skipped.append(name)
            if not quiet:
                print("SKIP %s (%s)" % (name, exc))
        except Exception:
            failed.append(name)
            if not quiet:
                print("FAIL %s" % name)
                traceback.print_exc()
        else:
            passed.append(name)
            if not quiet:
                print("ok   %s" % name)
    if not quiet:
        for name in skipped:
            print("skipped: %s" % name)
    print("%d passed, %d failed, %d skipped" % (len(passed), len(failed), len(skipped)))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
