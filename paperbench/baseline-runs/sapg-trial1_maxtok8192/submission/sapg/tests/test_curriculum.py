"""Unit tests for the SAPG success-tolerance curriculum.

Validates the paper's curriculum behavior:
  * delta starts at 7.5cm (0.075) and decreases by 10% (x0.9) whenever the
    average number of successes per episode exceeds 3.
  * delta is clamped at the 1cm (0.01) floor.
  * state_dict / load_state_dict round-trips correctly.
  * update_batch handles per-environment success vectors (SAPG block structure).

Run directly:  python tests/test_curriculum.py
Or with pytest: pytest tests/test_curriculum.py
"""

from __future__ import annotations

import math
import os
import sys

# Make the project root importable when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.curriculum import CurriculumConfig, SuccessToleranceCurriculum  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


def _assert_close(a, b, tol=1e-9, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b} (tol={tol})")


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = CurriculumConfig()
    _assert_close(cfg.initial_delta, 0.075, msg="initial_delta should be 7.5cm")
    _assert_close(cfg.min_delta, 0.01, msg="min_delta should be 1cm")
    _assert_close(cfg.decrease_factor, 0.9, msg="decrease_factor should be 0.9")
    _assert_close(cfg.success_threshold, 3.0, msg="success_threshold should be 3")
    _assert(cfg.enabled is True, "curriculum should be enabled by default")


def test_config_kwargs_override():
    cfg = CurriculumConfig(initial_delta=0.5, min_delta=0.1, success_threshold=5.0)
    _assert_close(cfg.initial_delta, 0.5)
    _assert_close(cfg.min_delta, 0.1)
    _assert_close(cfg.success_threshold, 5.0)


# ---------------------------------------------------------------------------
# Basic delta behavior
# ---------------------------------------------------------------------------
def test_initial_delta():
    cur = SuccessToleranceCurriculum()
    _assert_close(cur.delta, 0.075, msg="initial delta")
    _assert(not cur.at_minimum, "should not be at minimum initially")


def test_no_decrease_below_threshold():
    """Avg successes <= 3 must NOT trigger a decrease."""
    cur = SuccessToleranceCurriculum()
    for _ in range(10):
        changed = cur.update(successes=3.0, num_episodes=1)
        _assert(not changed, "delta should not change at threshold boundary")
    _assert_close(cur.delta, 0.075, msg="delta unchanged below/at threshold")


def test_decrease_above_threshold():
    """Avg successes > 3 triggers a 10% decrease."""
    cur = SuccessToleranceCurriculum()
    changed = cur.update(successes=4.0, num_episodes=1)
    _assert(changed, "delta should change when avg successes > 3")
    _assert_close(cur.delta, 0.075 * 0.9, tol=1e-12, msg="delta decreased by 10%")


def test_repeated_decreases():
    cur = SuccessToleranceCurriculum()
    expected = 0.075
    for _ in range(5):
        cur.update(successes=10.0, num_episodes=1)
        expected *= 0.9
    _assert_close(cur.delta, expected, tol=1e-12, msg="repeated 10% decreases")


def test_min_delta_floor():
    """delta must never go below min_delta (1cm)."""
    cur = SuccessToleranceCurriculum()
    for _ in range(200):
        cur.update(successes=100.0, num_episodes=1)
    _assert_close(cur.delta, 0.01, tol=1e-12, msg="delta clamped at floor")
    _assert(cur.at_minimum, "should report at_minimum once floored")


def test_normalized_delta():
    cur = SuccessToleranceCurriculum()
    _assert_close(cur.normalized_delta, 1.0, tol=1e-9, msg="normalized at start")
    # Drive to the floor.
    for _ in range(200):
        cur.update(successes=100.0, num_episodes=1)
    _assert_close(cur.normalized_delta, 0.0, tol=1e-9, msg="normalized at floor")


# ---------------------------------------------------------------------------
# Batch / per-env updates (SAPG block structure)
# ---------------------------------------------------------------------------
def test_update_batch_mean():
    """update_batch should use the mean of per-env successes."""
    cur = SuccessToleranceCurriculum()
    # mean = (4+4+4+4)/4 = 4 > 3 -> decrease
    changed = cur.update_batch([4.0, 4.0, 4.0, 4.0])
    _assert(changed, "batch mean > 3 should decrease delta")
    _assert_close(cur.delta, 0.075 * 0.9, tol=1e-12)


def test_update_batch_no_decrease():
    cur = SuccessToleranceCurriculum()
    # mean = (1+2+3+2)/4 = 2 <= 3 -> no decrease
    changed = cur.update_batch([1.0, 2.0, 3.0, 2.0])
    _assert(not changed, "batch mean <= 3 should not decrease delta")
    _assert_close(cur.delta, 0.075, tol=1e-12)


def test_update_batch_empty():
    cur = SuccessToleranceCurriculum()
    changed = cur.update_batch([])
    _assert(not changed, "empty batch should be a no-op")
    _assert_close(cur.delta, 0.075)


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------
def test_ema_smoothing():
    """With ema_alpha < 1, a single high value should not immediately trigger."""
    cfg = CurriculumConfig(ema_alpha=0.1)
    cur = SuccessToleranceCurriculum(cfg)
    # First update: ema = 0.1 * 100 = 10 > 3 -> triggers on first step.
    cur.update(successes=100.0, num_episodes=1)
    _assert_close(cur.delta, 0.075 * 0.9, tol=1e-12)

    # Now a low value should pull the EMA down and stop triggering.
    cur2 = SuccessToleranceCurriculum(CurriculumConfig(ema_alpha=0.1))
    cur2.update(successes=0.0, num_episodes=1)  # ema = 0
    changed = cur2.update(successes=0.0, num_episodes=1)
    _assert(not changed, "low EMA should not trigger decrease")


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def test_warmup_blocks_decrease():
    cfg = CurriculumConfig(warmup_episodes=5)
    cur = SuccessToleranceCurriculum(cfg)
    for _ in range(4):
        changed = cur.update(successes=100.0, num_episodes=1)
        _assert(not changed, "warmup should block decreases")
    _assert_close(cur.delta, 0.075, msg="delta unchanged during warmup")
    # After warmup, decrease should occur.
    changed = cur.update(successes=100.0, num_episodes=1)
    _assert(changed, "decrease should occur after warmup")


# ---------------------------------------------------------------------------
# Disabled curriculum
# ---------------------------------------------------------------------------
def test_disabled_curriculum():
    cfg = CurriculumConfig(enabled=False)
    cur = SuccessToleranceCurriculum(cfg)
    for _ in range(10):
        changed = cur.update(successes=100.0, num_episodes=1)
        _assert(not changed, "disabled curriculum must never change delta")
    _assert_close(cur.delta, 0.075)


# ---------------------------------------------------------------------------
# State dict round-trip
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip():
    cur = SuccessToleranceCurriculum()
    for _ in range(3):
        cur.update(successes=10.0, num_episodes=1)
    state = cur.state_dict()
    _assert("delta" in state, "state_dict must contain delta")

    cur2 = SuccessToleranceCurriculum()
    cur2.load_state_dict(state)
    _assert_close(cur2.delta, cur.delta, tol=1e-12, msg="delta restored")
    _assert_close(cur2.normalized_delta, cur.normalized_delta, tol=1e-12)


def test_reset():
    cur = SuccessToleranceCurriculum()
    for _ in range(5):
        cur.update(successes=10.0, num_episodes=1)
    cur.reset()
    _assert_close(cur.delta, 0.075, msg="reset restores initial delta")


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------
def test_history_recorded():
    cur = SuccessToleranceCurriculum()
    cur.update(successes=10.0, num_episodes=1)
    cur.update(successes=0.0, num_episodes=1)
    hist = cur.history
    _assert(len(hist) >= 2, "history should record updates")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
