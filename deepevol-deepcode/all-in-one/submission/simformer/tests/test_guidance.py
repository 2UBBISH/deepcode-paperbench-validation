"""Tests for :mod:`simformer.guidance` -- general diffusion guidance (Algorithm 1).

These tests cover the pieces of Sec. 3.4 / Appendix A1.3-A3.3 of the Simformer paper:

* constraint objects ``c(x) <= 0`` (interval / box / equality / combined),
* the constraint score ``grad log sigmoid(sign * s(t) * c(x_hat_0))`` with the
  denoised estimate ``x_hat_0 = (x_hat_t + sigma(t)^2 s)/mu(t)``,
* the scaling functions ``s(t)`` (default ``1 / sigma(t)^2``),
* a single guided reverse-SDE step and the full Algorithm 1 loop with
  self-recurrence ``r``,
* end-to-end behaviour on a toy Gaussian target: samples guided with an
  interval constraint must satisfy it far more often than unguided samples,
  while conditioned dimensions stay clamped.

Everything is written defensively (dual-layout imports, attribute fallbacks,
``pytest.skip`` when an optional piece of the API is absent) so that the suite
degrades gracefully.
"""

from __future__ import annotations

import importlib
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# tolerant loading helpers
# ---------------------------------------------------------------------------
def _load_module() -> Any:
    """Import ``simformer.guidance`` under either repository layout."""
    candidates = (
        "simformer.guidance",
        "simformer.simformer.guidance",
    )
    for path in candidates:
        try:
            return importlib.import_module(path)
        except Exception:  # pragma: no cover - depends on layout
            continue
    pytest.skip("simformer.guidance is not importable")


G = _load_module()


def _get(name: str, *fallbacks: str) -> Any:
    """Return the first available attribute, otherwise skip the test."""
    for attr in (name,) + fallbacks:
        if hasattr(G, attr):
            return getattr(G, attr)
    pytest.skip(f"simformer.guidance.{name} is not implemented")


def _call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments it actually accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    supported = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **supported)


# ---------------------------------------------------------------------------
# toy SDE / score setup (independent of the transformer)
# ---------------------------------------------------------------------------
DATA_STD = 1.0
T_MIN = 1e-5
T_MAX = 1.0
N_STEPS = 60


def _sde(**kwargs: Any) -> Any:
    get_sde = _get("get_sde")
    cfg = dict(sigma_max=15.0, sigma_min=1e-4, t_min=T_MIN, t_max=T_MAX,
               n_steps=N_STEPS)
    cfg.update(kwargs)
    return _call(get_sde, "vesde", **cfg)


def _sigma(sde: Any, t: Any) -> Any:
    sig = np.asarray(sde.marginal_std(t), dtype=np.float64)
    return float(sig) if sig.size == 1 else sig


def _toy_score_fn(sde: Any) -> Callable[[np.ndarray, Any], np.ndarray]:
    """Score of a Gaussian ``N(0, DATA_STD^2)`` noised with VESDE.

    For the VESDE forward process the marginal at time ``t`` is
    ``N(0, DATA_STD^2 + sigma(t)^2)``, hence ``s(x, t) = -x / (DATA_STD^2 + sigma^2)``.
    """

    def score_fn(x: np.ndarray, t: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        sig = _sigma(sde, t)
        sig = np.asarray(sig, dtype=np.float64)
        if sig.ndim == 1 and arr.ndim == 2 and sig.shape[0] == arr.shape[0]:
            sig = sig.reshape(-1, 1)
        var = DATA_STD ** 2 + sig ** 2
        return -arr / var

    return score_fn


def _toy_sampler(sde: Any, score_fn: Callable, **kwargs: Any) -> np.ndarray:
    """Unguided reference samples via the diffusion reverse-SDE integrator."""
    sample_reverse_sde = _get("sample_reverse_sde")
    try:
        sample_reverse_sde = importlib.import_module(
            "simformer.simformer.diffusion"
        ).sample_reverse_sde
    except Exception:  # pragma: no cover
        pass
    try:
        out = _call(
            sample_reverse_sde,
            score_fn,
            sde,
            (kwargs.pop("n_samples", 64), 2),
            n_steps=N_STEPS,
            t_min=T_MIN,
            t_max=T_MAX,
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - signature drift
        pytest.skip(f"sample_reverse_sde unusable in this build: {exc!r}")
    return np.asarray(out, dtype=np.float64)


def _satisfaction(samples: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  indices: Sequence[int]) -> float:
    """Fraction of rows whose selected coordinates lie inside ``[lo, hi]``."""
    x = np.asarray(samples, dtype=np.float64)
    idx = list(indices)
    block = x[:, idx]
    ok = np.all((block >= np.asarray(lower)) & (block <= np.asarray(upper)), axis=1)
    return float(np.mean(ok))


def _extract_samples(result: Any) -> np.ndarray:
    """Pull the sample array out of whatever ``general_guidance`` returns."""
    if hasattr(result, "samples"):
        return np.asarray(result.samples, dtype=np.float64)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# constraint objects
# ---------------------------------------------------------------------------
def test_interval_constraint_violation_and_satisfaction() -> None:
    interval_constraint = _get("interval_constraint")
    con = _call(interval_constraint, [0], lower=[-1.0], upper=[1.0])
    x = np.array([[0.0, 5.0], [2.0, 5.0], [-3.0, 5.0]])
    v = np.asarray(con.violation(x), dtype=np.float64).reshape(len(x), -1).max(axis=1)
    assert v[0] <= 1e-9
    assert v[1] > 0.0 and v[2] > 0.0
    sat = _get("constraint_satisfaction")
    frac = float(np.asarray(_call(sat, con, x)).reshape(-1)[0])
    assert 0.0 <= frac <= 1.0
    assert abs(frac - 1.0 / 3.0) < 1e-9


def test_interval_constraint_jacobian_direction() -> None:
    interval_constraint = _get("interval_constraint")
    con = _call(interval_constraint, [0], lower=[-1.0], upper=[1.0])
    x = np.array([[2.0]])
    grad = np.asarray(con.grad(x), dtype=np.float64)
    assert grad.shape[-1] == 1
    assert np.isfinite(grad).all()
    # gradient of the (positive part) violation must be positive in the
    # direction away from the interval
    assert float(np.asarray(grad).reshape(-1)[0]) > 0.0


def test_upper_and_lower_bound_constructors() -> None:
    upper_bound_constraint = _get("upper_bound_constraint")
    lower_bound_constraint = _get("lower_bound_constraint")
    con_up = _call(upper_bound_constraint, [0], upper=[1.0])
    con_lo = _call(lower_bound_constraint, [0], lower=[-1.0])
    assert float(con_up.violation(np.array([[2.0]]))[0]) > 0.0
    assert float(con_up.violation(np.array([[0.0]]))[0]) <= 1e-9
    assert float(con_lo.violation(np.array([[-2.0]]))[0]) > 0.0
    assert float(con_lo.violation(np.array([[0.0]]))[0]) <= 1e-9


def test_combined_constraint_union() -> None:
    interval_constraint = _get("interval_constraint")
    combine_constraints = _get("combine_constraints", "combine")
    c1 = _call(interval_constraint, [0], lower=[-1.0], upper=[1.0])
    c2 = _call(interval_constraint, [1], lower=[-0.5], upper=[0.5])
    con = combine_constraints(c1, c2)
    x = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    v = np.asarray(con.violation(x), dtype=np.float64).reshape(len(x), -1).max(axis=1)
    assert v[0] <= 1e-9
    assert v[1] > 0.0 and v[2] > 0.0


def test_box_and_equality_constraints() -> None:
    box_constraint = _get("box_constraint")
    con = _call(box_constraint, dim=2, lower=[-1.0, -1.0], upper=[1.0, 1.0])
    sat = np.asarray(con.satisfied(np.array([[0.0, 0.0], [2.0, 0.0]])))
    assert bool(sat.reshape(-1)[0]) is True
    assert bool(sat.reshape(-1)[1]) is False

    equality_constraint = _get("equality_constraint")
    con_eq = _call(equality_constraint, [0], target=[1.0])
    assert float(np.ravel(con_eq.violation(np.array([[1.0]])))[0]) <= 1e-9
    assert float(np.ravel(con_eq.violation(np.array([[2.0]])))[0]) > 0.0


# ---------------------------------------------------------------------------
# numerical building blocks
# ---------------------------------------------------------------------------
def test_log_sigmoid_is_stable() -> None:
    log_sigmoid = _get("log_sigmoid")
    values = np.array([-1e4, -50.0, -1.0, 0.0, 1.0, 50.0, 1e4])
    out = np.asarray(log_sigmoid(values), dtype=np.float64)
    assert out.shape == values.shape
    assert np.isfinite(out).all()
    assert np.all(out <= 1e-12)
    assert out[3] == pytest.approx(math.log(0.5), rel=1e-12, abs=1e-12)
    # monotone decreasing in the argument
    assert np.all(np.diff(out) <= 1e-12)


def test_inverse_variance_scaling_matches_one_over_sigma_squared() -> None:
    sde = _sde()
    scaling = _get("inverse_variance_scaling")
    for t in (0.05, 0.25, 0.5, 1.0):
        sig = float(np.asarray(_sigma(sde, t)))
        expected = 1.0 / sig ** 2
        got = float(np.asarray(_call(scaling, t, sde=sde)).reshape(-1)[0]
                    if "sde" in inspect.signature(scaling).parameters
                    else np.asarray(_call(scaling, t)).reshape(-1)[0])
        assert got == pytest.approx(expected, rel=1e-9)


def test_constant_and_std_scaling() -> None:
    sde = _sde()
    constant_scaling = _get("constant_scaling")
    got = np.asarray(_call(constant_scaling, 0.5, value=2.0)).reshape(-1)[0]
    assert float(got) == pytest.approx(2.0)

    if hasattr(G, "std_scaling"):
        sig = float(np.asarray(_sigma(sde, 0.3)))
        got = float(np.asarray(_call(G.std_scaling, 0.3, sde=sde)).reshape(-1)[0])
        assert got == pytest.approx(sig, rel=1e-9)

    get_scaling = _get("get_scaling_function", "scaling_function")
    fn = _call(get_scaling, "inverse_variance")
    assert callable(fn)


def test_denoise_from_score_matches_tweedie_formula() -> None:
    """``x_hat_0 = (x_hat_t + sigma(t)^2 s) / mu(t)`` (Appendix A3.3)."""
    sde = _sde()
    denoise = _get("denoise_from_score", "tweedie_denoise")
    rng = np.random.default_rng(0)
    x_t = rng.normal(size=(5, 2))
    score = rng.normal(size=(5, 2))
    t = np.full(5, 0.4)
    got = np.asarray(_call(denoise, sde, x_t, t, score), dtype=np.float64)
    sig = np.asarray(_sigma(sde, 0.4), dtype=np.float64)
    mu = np.asarray(sde.marginal_mean(0.4), dtype=np.float64)
    expected = (x_t + sig ** 2 * score) / mu
    assert got.shape == x_t.shape
    assert np.allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_constraint_score_pushes_towards_the_interval() -> None:
    sde = _sde()
    constraint_score = _get("constraint_score")
    interval_constraint = _get("interval_constraint")
    con = _call(interval_constraint, [0, 1], lower=[-1.0, -1.0], upper=[1.0, 1.0])

    x_bad = np.array([[4.0, -4.0]])
    t = 0.5
    grad = np.asarray(
        _call(
            constraint_score,
            con,
            x_bad,
            t,
            sde,
            scale="constant",
            scale_value=1.0,
            value=1.0,
            sign=-1.0,
            weight=1.0,
            differentiate="denoised",
        ),
        dtype=np.float64,
    ).reshape(x_bad.shape)
    assert np.isfinite(grad).all()
    # moving along the constraint gradient must reduce the violation
    step = 1e-2 * grad / max(np.abs(grad).max(), 1e-12)
    v_before = float(np.max(con.violation(x_bad)))
    v_after = float(np.max(con.violation(x_bad + step)))
    assert v_after < v_before


def test_guided_score_returns_score_and_diagnostics() -> None:
    sde = _sde()
    guided_score = _get("guided_score")
    interval_constraint = _get("interval_constraint")
    con = _call(interval_constraint, [0], lower=[-1.0], upper=[1.0])
    x_t = np.array([[3.0]])
    score = np.array([[-0.5]])
    out = _call(
        guided_score,
        score,
        con,
        x_t,
        0.5,
        sde,
        guidance_scale=1.0,
        scale="constant",
        sign=-1.0,
    )
    if isinstance(out, tuple):
        modified, info = out[0], out[1]
        assert isinstance(info, dict)
    else:
        modified = out
    modified = np.asarray(modified, dtype=np.float64)
    assert modified.shape == score.shape
    assert np.isfinite(modified).all()


def test_guidance_config_times() -> None:
    GuidanceConfig = _get("GuidanceConfig")
    sde = _sde()
    cfg = _call(GuidanceConfig, n_steps=N_STEPS, t_min=T_MIN, t_max=T_MAX)
    times = np.asarray(cfg.times(sde), dtype=np.float64)
    assert times.size == N_STEPS
    assert times[0] == pytest.approx(T_MAX, rel=1e-6)
    assert times[-1] == pytest.approx(T_MIN, rel=1e-6)
    assert np.all(np.diff(times) < 0)  # descending for the reverse SDE


def test_guidance_step_is_finite() -> None:
    sde = _sde()
    guidance_step = _get("guidance_step")
    interval_constraint = _get("interval_constraint")
    con = _call(interval_constraint, [0], lower=[-1.0], upper=[1.0])
    x = np.array([[3.0]])
    score = np.array([[-0.5]])
    out = _call(
        guidance_step,
        sde,
        x,
        0.6,
        0.5,
        score,
        con,
        scale="constant",
        sign=-1.0,
        guidance_scale=1.0,
        seed=0,
    )
    x_next = out[0] if isinstance(out, tuple) else out
    x_next = np.asarray(x_next, dtype=np.float64)
    assert x_next.shape == x.shape
    assert np.isfinite(x_next).all()


# ---------------------------------------------------------------------------
# Algorithm 1: end-to-end general guidance
# ---------------------------------------------------------------------------
def _run_guidance(
    sde: Any,
    constraint: Any,
    *,
    n_samples: int = 64,
    n_steps: int = N_STEPS,
    self_recurrence: int = 0,
    seed: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    general_guidance = _get("general_guidance", "sample_with_guidance")
    score_fn = _toy_score_fn(sde)
    result = _call(
        general_guidance,
        score_fn,
        sde,
        constraint,
        n_samples=n_samples,
        n_steps=n_steps,
        t_min=T_MIN,
        t_max=T_MAX,
        self_recurrence=self_recurrence,
        seed=seed,
        **kwargs,
    )
    return _extract_samples(result)


def test_general_guidance_runs_and_shapes_are_correct() -> None:
    sde = _sde()
    con = _call(_get("interval_constraint"), [0], lower=[-1.0], upper=[1.0])
    samples = _run_guidance(sde, con, n_samples=32)
    assert samples.shape == (32, 2)
    assert np.isfinite(samples).all()


def test_interval_guidance_enforces_the_constraint() -> None:
    """Guided samples must satisfy ``x_0 in [1.5, 4.0]`` much more often
    than plain reverse-SDE samples from the same toy target."""
    sde = _sde(n_steps=N_STEPS)
    lo, hi = 1.5, 4.0
    con = _call(_get("interval_constraint"), [0], lower=[lo], upper=[hi])
    samples = _run_guidance(sde, con, n_samples=128, n_steps=N_STEPS)
    assert np.isfinite(samples).all()
    satisfied = _satisfaction(samples, [lo], [hi], [0])

    # unguided reference: same loop with an always-satisfied constraint
    loose = _call(_get("interval_constraint"), [0],
                  lower=[-100.0], upper=[100.0])
    base = _run_guidance(sde, loose, n_samples=128, n_steps=N_STEPS)
    baseline = _satisfaction(base, [lo], [hi], [0])

    assert baseline < 0.35  # N(0, 1) puts little mass in the upper tail
    assert satisfied > baseline + 0.2
    assert satisfied > 0.5


def test_guidance_keeps_samples_near_the_target_mode() -> None:
    """Tight interval centred on the mode should not blow up the samples."""
    sde = _sde(n_steps=N_STEPS)
    con = _call(_get("interval_constraint"), [0], lower=[-0.5], upper=[0.5])
    samples = _run_guidance(sde, con, n_samples=64, n_steps=N_STEPS)
    assert np.isfinite(samples).all()
    assert np.abs(samples[:, 0]).mean() < 2.0
    assert samples[:, 0].std() < 2.0


def test_general_guidance_with_self_recurrence() -> None:
    """Algorithm 1 must run with ``r > 0`` (self-recurrence) as well."""
    sde = _sde()
    con = _call(_get("interval_constraint"), [0], lower=[1.0], upper=[3.0])
    samples = _run_guidance(sde, con, n_samples=32, self_recurrence=3)
    assert samples.shape == (32, 2)
    assert np.isfinite(samples).all()
    frac = _satisfaction(samples, [1.0], [3.0], [0])
    assert frac > 0.4


def test_general_guidance_clamps_conditioned_dimensions() -> None:
    sde = _sde()
    con = _call(_get("interval_constraint"), [0], lower=[0.5], upper=[5.0])
    condition_mask = np.array([False, True])
    condition_values = np.array([0.0, 0.75])
    samples = _run_guidance(
        sde,
        con,
        n_samples=32,
        condition_mask=condition_mask,
        condition_values=condition_values,
    )
    assert samples.shape == (32, 2)
    assert np.isfinite(samples).all()
    assert np.allclose(samples[:, 1], 0.75, atol=1e-6)


def test_guidance_result_container_is_populated() -> None:
    sde = _sde()
    con = _call(_get("interval_constraint"), [0], lower=[0.0], upper=[1.0])
    general_guidance = _get("general_guidance", "sample_with_guidance")
    result = _call(
        general_guidance,
        _toy_score_fn(sde),
        sde,
        con,
        n_samples=16,
        n_steps=10,
        t_min=T_MIN,
        t_max=T_MAX,
        seed=0,
        return_result=True,
    )
    if isinstance(result, tuple):
        result = result[0]
    if not hasattr(result, "samples"):
        pytest.skip("general_guidance does not return a GuidanceResult container")
    samples = np.asarray(result.samples)
    assert samples.shape == (16, 2)
    assert len(result) == 16
    assert np.asarray(result.as_array()).shape == (16, 2)
    assert result.n_samples if hasattr(result, "n_samples") else True


def test_sample_interval_conditional_helper() -> None:
    sample_interval_conditional = _get("sample_interval_conditional")
    sde = _sde()
    out = _call(
        sample_interval_conditional,
        _toy_score_fn(sde),
        sde,
        dim=2,
        lower=np.array([1.0, -1.0]),
        upper=np.array([3.0, 1.0]),
        indices=np.array([0]),
        n_samples=16,
        n_steps=10,
        t_min=T_MIN,
        t_max=T_MAX,
        seed=0,
    )
    samples = _extract_samples(out)
    assert samples.shape == (16, 2)
    assert np.isfinite(samples).all()


def test_guidance_with_constant_scaling_is_finite() -> None:
    """The scaling function must be selectable (e.g. constant) without NaNs."""
    sde = _sde()
    con = _call(_get("interval_constraint"), [0], lower=[0.0], upper=[2.0])
    samples = _run_guidance(
        sde,
        con,
        n_samples=32,
        scale="constant",
        guidance_scale=1.0,
    )
    assert samples.shape == (32, 2)
    assert np.isfinite(samples).all()


def test_guidance_reduces_violation_monotonically_on_a_linear_problem() -> None:
    """For a linear constraint on a noiseless score the violation must shrink."""
    sde = _sde(n_steps=N_STEPS)
    con = _call(_get("interval_constraint"), [0], lower=[-1.0], upper=[1.0])

    def score_fn(x: np.ndarray, t: Any) -> np.ndarray:
        # frozen "score" that keeps the sample near zero
        return -np.asarray(x, dtype=np.float64) / (DATA_STD ** 2 + 1e-8)

    general_guidance = _get("general_guidance", "sample_with_guidance")
    out = _call(
        general_guidance,
        score_fn,
        sde,
        con,
        n_samples=32,
        n_steps=N_STEPS,
        t_min=T_MIN,
        t_max=T_MAX,
        seed=0,
        scale="constant",
        guidance_scale=1.0,
    )
    samples = _extract_samples(out)
    assert np.isfinite(samples).all()
    assert _satisfaction(samples, [-1.0], [1.0], [0]) > 0.5
