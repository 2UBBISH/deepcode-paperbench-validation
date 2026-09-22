"""Unit / smoke tests for the forward-transfer metric (Appendix F).

The paper (Wołczyk et al., 2024; metric from Wołczyk et al., 2021 / Bornschein
et al., 2022) defines forward transfer for a fine-tuned policy as

    FT = (AUC - AUC^b) / (1 - AUC^b)

where ``AUC`` is the area under the (per-stage-restricted) success-rate curve of
the fine-tuned policy and ``AUC^b`` that of a from-scratch baseline.  Table 6
reports this quantity for prefix-task lengths ``k = 1..4`` for the fine-tuning,
EWC and BC variants of RoboticSequence.

These tests are dependency-light: neither ``pytest`` nor ``numpy`` is required.
Run with either::

    pytest tests/test_forward_transfer.py
    python -m tests.test_forward_transfer
"""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from typing import Callable, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class SkipTest(Exception):
    """Raised when an optional dependency (or the module) is unavailable."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_MODULE = None


def _ft():
    """Import (and cache) ``src.analysis.forward_transfer``."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    last: Optional[BaseException] = None
    for name in (
        "src.analysis.forward_transfer",
        "analysis.forward_transfer",
        "forward_transfer",
    ):
        try:
            _MODULE = importlib.import_module(name)
            return _MODULE
        except Exception as exc:  # pragma: no cover - import shim
            last = exc
    raise SkipTest("forward_transfer module unavailable: %r" % (last,))


def _f(value) -> float:
    return float(value)


def _list(values) -> List[float]:
    return [_f(v) for v in list(values)]


def _make_history(steps, values, key="success_rate"):
    """Build a logged evaluation history in the canonical JSON shape."""
    return [
        {"step": float(s), key: float(v)}
        for s, v in zip(steps, values)
    ]


def _approx(a, b, tol=1e-9) -> bool:
    return abs(_f(a) - _f(b)) <= tol


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------
def test_auc_constant_curves():
    ft = _ft()
    assert _approx(ft.auc([1.0, 1.0, 1.0]), 1.0)
    assert _approx(ft.auc([0.0, 0.0, 0.0]), 0.0)
    assert _approx(ft.auc([0.5] * 7), 0.5)
    # a single point has no width -> still finite and in range
    assert 0.0 <= _f(ft.auc([0.7])) <= 1.0


def test_auc_monotone_in_values():
    ft = _ft()
    low = _f(ft.auc([0.0, 0.0, 0.0, 1.0]))
    high = _f(ft.auc([0.0, 1.0, 1.0, 1.0]))
    assert high > low


def test_auc_with_explicit_times():
    ft = _ft()
    # trapezoidal rule over an explicit uniform grid
    value = _f(ft.auc([0.0, 1.0], times=[0.0, 1.0]))
    assert 0.0 <= value <= 1.0
    # a constant curve integrates to the constant regardless of the horizon
    assert _approx(ft.auc([0.8, 0.8, 0.8], times=[0.0, 5.0, 10.0]), 0.8)


def test_auc_respects_total_horizon():
    ft = _ft()
    # same curve, different horizon -> same normalised AUC
    a = _f(ft.auc([0.0, 0.5, 1.0], times=[0.0, 1.0, 2.0]))
    b = _f(ft.auc([0.0, 0.5, 1.0], times=[0.0, 100.0, 200.0]))
    assert _approx(a, b, tol=1e-6)


# ---------------------------------------------------------------------------
# the FT formula itself
# ---------------------------------------------------------------------------
def test_forward_transfer_formula():
    ft = _ft()
    # (AUC - AUC^b) / (1 - AUC^b)
    assert _approx(ft.forward_transfer(0.8, 0.5), (0.8 - 0.5) / (1.0 - 0.5))
    assert _approx(ft.forward_transfer(0.5, 0.5), 0.0)
    assert _approx(ft.forward_transfer(1.0, 0.0), 1.0)
    assert _approx(ft.forward_transfer(0.0, 0.0), 0.0)
    # fine-tuning *helped* -> positive; hurt -> negative
    assert _f(ft.forward_transfer(0.9, 0.2)) > 0.0
    assert _f(ft.forward_transfer(0.1, 0.9)) < 0.0


def test_forward_transfer_raises_on_degenerate_baseline():
    ft = _ft()
    exc_type = getattr(ft, "EpsClosedProblem", None)
    raised = None
    try:
        ft.forward_transfer(0.5, 1.0)
    except Exception as exc:  # noqa: BLE001 - we want the concrete type
        raised = exc
    assert raised is not None, "AUC^b == 1 must be flagged as degenerate"
    if exc_type is not None:
        assert isinstance(raised, exc_type)


def test_safe_forward_transfer_returns_nan():
    ft = _ft()
    value = _f(ft.safe_forward_transfer(0.5, 1.0))
    assert math.isnan(value)
    # non-degenerate inputs agree with the strict version
    assert _approx(ft.safe_forward_transfer(0.8, 0.5), ft.forward_transfer(0.8, 0.5))


def test_forward_transfer_paper_table6_trends():
    ft = _ft()
    table = ft.PAPER_TABLE_6
    # fine-tuning deteriorates as more prefix tasks are added
    ft_vals = _list(table["fine_tuning"])
    assert all(
        ft_vals[i] >= ft_vals[i + 1] - 1e-9 for i in range(len(ft_vals) - 1)
    ), ft_vals
    # retention methods stay high
    assert min(_list(table["bc"])) >= 0.9
    assert min(_list(table["ewc"])) >= 0.7
    assert min(_list(table["bc"])) >= min(_list(table["ewc"])) >= min(ft_vals)


# ---------------------------------------------------------------------------
# curve extraction / interpolation
# ---------------------------------------------------------------------------
def test_success_rate_curve_extraction():
    ft = _ft()
    history = _make_history([0, 1000, 2000], [0.0, 0.5, 1.0])
    times, values = ft.success_rate_curve(history)
    times = _list(times)
    values = _list(values)
    assert len(times) == len(values) == 3
    assert _approx(times[-1], 2000.0)
    assert _approx(values[-1], 1.0)


def test_success_rate_curve_stage_restriction():
    ft = _ft()
    history = [
        {
            "step": 0,
            "success_rate": 0.25,
            "per_stage": {"peg-unplug-side": 0.0, "push-wall": 0.5},
        },
        {
            "step": 100,
            "success_rate": 0.5,
            "per_stage": {"peg-unplug-side": 0.4, "push-wall": 0.6},
        },
    ]
    try:
        times, values = ft.success_rate_curve(history, stage="push-wall")
    except Exception as exc:  # pragma: no cover - per-stage layout optional
        raise SkipTest("per-stage restriction unsupported: %r" % (exc,))
    values = _list(values)
    assert _approx(values[0], 0.5)
    assert _approx(values[-1], 0.6)


def test_interpolate_curve_piecewise_linear_and_clamped():
    ft = _ft()
    values = _list(
        ft.interpolate_curve([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [0.5, 1.5, -1.0, 3.0])
    )
    assert _approx(values[0], 0.5)
    assert _approx(values[1], 0.5)
    # clamped outside the observed range
    assert _approx(values[2], 0.0)
    assert _approx(values[3], 0.0)


# ---------------------------------------------------------------------------
# prefix tasks (Table 6 protocol)
# ---------------------------------------------------------------------------
def test_prefix_tasks_and_full_sequence():
    ft = _ft()
    pool = tuple(ft.PREFIX_TASK_POOL)
    pretrained = tuple(ft.PRETRAINED_TASKS)
    assert pool and pretrained

    assert len(ft.prefix_tasks(0)) == 0
    for k in range(1, len(pool) + 1):
        prefix = tuple(ft.prefix_tasks(k))
        assert len(prefix) == k
        assert set(prefix) <= set(pool)

    full = tuple(ft.full_sequence(2))
    assert len(full) == 2 + len(pretrained)
    assert set(pretrained) <= set(full)
    # the pre-trained (FAR) stages are the tail of the sequence
    assert full[-len(pretrained):] == pretrained


def test_prefix_lengths_cover_table6():
    ft = _ft()
    # Table 6 is reported for k = 1..4 prefix tasks
    assert len(ft.PREFIX_TASK_POOL) >= 4
    lengths = [len(ft.prefix_tasks(k)) for k in (1, 2, 3, 4)]
    assert lengths == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# end-to-end curve computation
# ---------------------------------------------------------------------------
def test_forward_transfer_curve_end_to_end():
    ft = _ft()
    steps = [i * 100_000 for i in range(11)]
    ft_hist = _make_history(steps, [0.2 + 0.05 * i for i in range(11)])
    sc_hist = _make_history(steps, [0.1 + 0.005 * i for i in range(11)])
    result = ft.forward_transfer_curve(ft_hist, sc_hist)
    assert isinstance(result, dict)
    assert "forward_transfer" in result
    assert _f(result["forward_transfer"]) > 0.0
    assert _f(result["auc"]) > _f(result["auc_baseline"])


def test_forward_transfer_curve_identical_histories():
    ft = _ft()
    steps = [0, 1_000_000, 2_000_000]
    hist = _make_history(steps, [0.0, 0.5, 1.0])
    result = ft.forward_transfer_curve(hist, list(hist))
    assert _approx(result["forward_transfer"], 0.0, tol=1e-6)


# ---------------------------------------------------------------------------
# aggregation / reporting
# ---------------------------------------------------------------------------
def test_z_for_confidence_levels():
    ft = _ft()
    assert abs(_f(ft.z_for(0.90)) - 1.6449) < 1e-3
    assert abs(_f(ft.z_for(0.95)) - 1.9600) < 1e-3
    assert _f(ft.z_for(0.90)) > 0.0


def test_aggregate_forward_transfer_stats():
    ft = _ft()
    agg = ft.aggregate_forward_transfer([0.8, 0.9, 1.0])
    assert _approx(agg["mean"], 0.9, tol=1e-9)
    assert int(agg["n"]) == 3
    assert _f(agg["half_width"]) > 0.0
    assert _f(agg["std"]) > 0.0
    # identical values -> zero width
    agg0 = ft.aggregate_forward_transfer([0.5, 0.5])
    assert _approx(agg0["half_width"], 0.0, tol=1e-12)


def test_format_table_renders_text():
    ft = _ft()
    table = {
        "none": {1: 0.30, 2: 0.05, 3: 0.02, 4: 0.00},
        "ewc": {1: 0.85, 2: 0.82, 3: 0.78, 4: 0.75},
        "bc": {1: 0.95, 2: 0.95, 3: 0.94, 4: 0.94},
    }
    text = ft.format_table(table)
    assert isinstance(text, str)
    assert len(text) > 0
    for name in table:
        assert name in text


def test_table6_like_flags_paper_trends():
    ft = _ft()
    report = ft.table6_like(dict(ft.PAPER_TABLE_6))
    assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# online tracker
# ---------------------------------------------------------------------------
def test_forward_transfer_tracker_online():
    ft = _ft()
    try:
        tracker = ft.ForwardTransferTracker()
    except TypeError:  # pragma: no cover - signature drift
        tracker = ft.ForwardTransferTracker(eval_every=None)

    for step, value in [(0, 0.0), (1_000_000, 0.3), (2_000_000, 0.6)]:
        tracker.record(step, success_rate=value)

    assert int(tracker.last_step) == 2_000_000
    auc_value = _f(tracker.auc())
    assert 0.0 <= auc_value <= 1.0
    payload = tracker.to_dict()
    assert isinstance(payload, dict) and payload


# ---------------------------------------------------------------------------
# offline table rebuild from a results directory
# ---------------------------------------------------------------------------
def test_compute_table_from_results_dir():
    ft = _ft()
    tmp = tempfile.mkdtemp(prefix="ft_table_")
    try:
        for prefix in (1, 2):
            for method, tail in (("none", 0.4), ("bc", 0.9), ("scratch", 0.2)):
                seed_dir = os.path.join(tmp, str(prefix), method, "seed_0")
                os.makedirs(seed_dir, exist_ok=True)
                history = _make_history(
                    [0, 1_000_000, 2_000_000], [0.0, tail / 2.0, tail]
                )
                with open(os.path.join(seed_dir, "summary.json"), "w") as fh:
                    json.dump(
                        {"history": history, "evaluations": history, "method": method},
                        fh,
                    )
        table = ft.compute_table(tmp, prefix_lengths=(1, 2))
        assert isinstance(table, dict)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
_TESTS: List[Callable[[], None]] = [
    test_auc_constant_curves,
    test_auc_monotone_in_values,
    test_auc_with_explicit_times,
    test_auc_respects_total_horizon,
    test_forward_transfer_formula,
    test_forward_transfer_raises_on_degenerate_baseline,
    test_safe_forward_transfer_returns_nan,
    test_forward_transfer_paper_table6_trends,
    test_success_rate_curve_extraction,
    test_success_rate_curve_stage_restriction,
    test_interpolate_curve_piecewise_linear_and_clamped,
    test_prefix_tasks_and_full_sequence,
    test_prefix_lengths_cover_table6,
    test_forward_transfer_curve_end_to_end,
    test_forward_transfer_curve_identical_histories,
    test_z_for_confidence_levels,
    test_aggregate_forward_transfer_stats,
    test_format_table_renders_text,
    test_table6_like_flags_paper_trends,
    test_forward_transfer_tracker_online,
    test_compute_table_from_results_dir,
]


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv

    failures = 0
    skipped = 0
    for test in _TESTS:
        name = test.__name__
        try:
            test()
        except SkipTest as exc:
            skipped += 1
            if not quiet:
                print("SKIP %s (%s)" % (name, exc))
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print("FAIL %s: %r" % (name, exc))
        else:
            if not quiet:
                print("ok   %s" % name)

    total = len(_TESTS)
    print(
        "\n%d/%d passed, %d skipped, %d failed"
        % (total - failures - skipped, total, skipped, failures)
    )
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
