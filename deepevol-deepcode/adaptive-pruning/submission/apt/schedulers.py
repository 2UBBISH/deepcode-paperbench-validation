"""Training schedules used by APT (Adaptive Pruning and Tuning).

This module implements every schedule that the paper specifies:

* **Cubic sparsity schedule** (Appendix A / Eq. in Sec. 4.2, Table 6):
  ``gamma_t = gamma_T + (1 - gamma_T) * (1 - t / T) ** 3`` controlling the LM
  parameter size so that the constraint of Eq. (1) of §3

      ``1 - C(Theta_t, M_t) / C(Theta_0, M_0) >= gamma_t``

  i.e. ``C(Theta_t, M_t) <= (1 - gamma_t) * C(Theta_0, M_0)`` is satisfied.

* **Mask decay** (Algorithm 1, Appendix C): pruned masks are *not* set to zero
  instantly, they are decreased by ``alpha = 0.01`` per adjustment step
  (``M_1 <- min(1, M_1 + alpha)``, ``M_0 <- max(0, M_0 - alpha)``).

* **``mu`` schedule** (§4.4, Appendix A): the self-knowledge distillation weight
  in ``L = mu * L_distill + (1 - mu) * L_ft`` is ``0`` before pruning starts and
  linearly increases to ``1`` at the end of pruning.

* **Tuning budget ``Delta_t``** (§3, §4.3): the number of tuning parameters is
  restricted to a limit ``Delta_t`` at each step.  The adaptive rank controller
  grows ranks with ``r_apt' = floor(r_apt * Delta_t' / Delta_t)``.

* **Two-stage training window** (Appendix A, Table 6): first prune + train with
  self-distillation for ``distill_epochs`` epochs, then fine-tune the pruned LM
  for the remaining epochs to recover end-task performance.

* **LR schedule** (reasonable default, documented in the plan): linear warmup
  followed by linear decay to zero, using the learning rates from Table 6.

Everything is implemented with plain Python numbers so that it can be used both
as a pure-python schedule and as a ``torch.optim.lr_scheduler.LambdaLR`` factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # constants
    "DEFAULT_ALPHA",
    "DEFAULT_CUBIC_EXPONENT",
    "DEFAULT_EMA_BETA",
    "DEFAULT_TARGET_SPARSITY",
    "DEFAULT_INITIAL_RANK",
    "DEFAULT_SCALING",
    # sparsity
    "SparsitySchedule",
    "CubicSparsitySchedule",
    "LinearSparsitySchedule",
    "ConstantSparsitySchedule",
    "make_sparsity_schedule",
    # mu / distillation
    "MuSchedule",
    "make_mu_schedule",
    # tuning budget / rank
    "TuningBudgetSchedule",
    "CubicTuningBudgetSchedule",
    "ConstantBudgetSchedule",
    "rank_update",
    "make_tuning_budget_schedule",
    # mask decay
    "MaskDecaySchedule",
    "anneal_mask",
    "anneal_masks",
    # adjustment steps / phases
    "AdjustmentStepSchedule",
    "compute_pruning_window",
    "epochs_to_steps",
    # lr
    "lr_factor",
    "build_lr_scheduler",
    # aggregation
    "ScheduleBank",
    "build_schedules",
]


# --------------------------------------------------------------------------- #
# Defaults taken from the paper / addendum
# --------------------------------------------------------------------------- #

#: Mask decay rate ``alpha`` (Algorithm 1, Appendix A/C).
DEFAULT_ALPHA = 0.01
#: Exponent of the cubic sparsity schedule (Appendix A).
DEFAULT_CUBIC_EXPONENT = 3.0
#: EMA coefficient of the moving-averaged salience ``S_bar = beta*S_bar + (1-beta)*S``.
DEFAULT_EMA_BETA = 0.85
#: Target sparsity used for RoBERTa / T5 in Table 2.
DEFAULT_TARGET_SPARSITY = 0.60
#: Initial APT adapter rank (§4.3 / Appendix A).
DEFAULT_INITIAL_RANK = 8
#: Static APT adapter scaling factor ``s`` (§4.1 / Appendix A).
DEFAULT_SCALING = 2.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or denominator == 0:
        return default
    return numerator / denominator


def epochs_to_steps(
    epochs: float,
    steps_per_epoch: int,
    *,
    start: int = 0,
    max_steps: Optional[int] = None,
) -> int:
    """Convert an epoch count to a step index.

    ``Table 6`` reports epochs (``Epochs`` and ``Distill epochs``); the training
    loop needs absolute step counts for the cubic schedule and the ``mu`` ramp.
    """
    steps = int(round(float(epochs) * max(int(steps_per_epoch), 0)))
    steps += int(start)
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    return max(steps, 0)


def compute_pruning_window(
    total_epochs: int,
    distill_epochs: int,
    steps_per_epoch: int,
    *,
    warmup_steps: int = 0,
) -> Dict[str, int]:
    """Step window of the two-stage APT schedule (Appendix A, Table 6).

    Stage 1 (prune + self-distill) lasts ``distill_epochs``; stage 2 (recover)
    lasts ``total_epochs - distill_epochs``.  ``T`` in the cubic schedule is the
    number of pruning/distillation steps; ``pruning_start_step`` accounts for an
    optional warmup during which Adam/LoRA stabilise before masks move.
    """
    total_steps = epochs_to_steps(total_epochs, steps_per_epoch)
    pruning_steps = epochs_to_steps(
        min(distill_epochs, total_epochs), steps_per_epoch, max_steps=total_steps
    )
    pruning_start = int(min(warmup_steps, pruning_steps))
    return {
        "total_steps": int(total_steps),
        "pruning_steps": int(pruning_steps),
        "pruning_start_step": int(pruning_start),
        "pruning_end_step": int(pruning_steps),
        "recover_start_step": int(pruning_steps),
        "recover_end_step": int(total_steps),
        "steps_per_epoch": int(steps_per_epoch),
    }


# --------------------------------------------------------------------------- #
# Sparsity schedules
# --------------------------------------------------------------------------- #


class SparsitySchedule:
    """Base class mapping a training step to a sparsity target ``gamma_t``.

    ``gamma_t`` is the ratio of *pruned* parameters to the original parameter
    count, so the parameter budget of the LM at step ``t`` is
    ``(1 - gamma_t) * C(Theta_0, M_0)`` (Eq. (1) of §3).
    """

    #: sparsity at the very beginning of pruning (0 => fully dense, 1 => empty)
    initial_sparsity: float = 0.0
    #: sparsity at the end of pruning, ``gamma_T``
    target_sparsity: float = DEFAULT_TARGET_SPARSITY
    #: total number of pruning steps ``T``
    total_steps: int = 0

    def __call__(self, step: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    # -- convenience ------------------------------------------------------ #
    def at(self, step: int) -> float:
        return float(self(step))

    def sparsities(self, steps: Optional[Iterable[int]] = None) -> List[float]:
        if steps is None:
            steps = range(int(self.total_steps) + 1)
        return [float(self(s)) for s in steps]

    def parameter_budget(
        self,
        step: int,
        original_param_count: float,
        *,
        sparsity: Optional[float] = None,
    ) -> float:
        """``C_top-i`` budget allowed by Eq. (1) at ``step``."""
        gamma = self.at(step) if sparsity is None else float(sparsity)
        return float(original_param_count) * (1.0 - gamma)

    def budget_bounds(
        self, step: int, original_param_count: float
    ) -> Tuple[float, float]:
        """Return ``(min_param_count, max_param_count)`` allowed at ``step``."""
        budget = self.parameter_budget(step, original_param_count)
        return 0.0, budget

    def target_param_count(self, original_param_count: float) -> float:
        """Parameter count reached at the end of pruning."""
        return float(original_param_count) * (1.0 - self.target_sparsity)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"{self.__class__.__name__}(initial_sparsity={self.initial_sparsity}, "
            f"target_sparsity={self.target_sparsity}, total_steps={self.total_steps})"
        )


class CubicSparsitySchedule(SparsitySchedule):
    """Cubic sparsity schedule from Appendix A.

    .. math::

        \\gamma_t = \\gamma_T + (\\gamma_0 - \\gamma_T)\\left(1 - \\frac{t}{T}\\right)^3

    With ``gamma_0 = 1`` this is exactly the paper's
    ``gamma_T + (1 - gamma_T)(1 - t / T)^3``.  Steps beyond ``T`` are clamped to
    ``gamma_T`` so the schedule is safe if training is extended.
    """

    def __init__(
        self,
        target_sparsity: float = DEFAULT_TARGET_SPARSITY,
        total_steps: int = 1000,
        initial_sparsity: float = 1.0,
        warmup_steps: int = 0,
        exponent: float = DEFAULT_CUBIC_EXPONENT,
    ) -> None:
        if total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        self.target_sparsity = float(_clamp(target_sparsity, 0.0, 1.0))
        self.initial_sparsity = float(_clamp(initial_sparsity, 0.0, 1.0))
        self.total_steps = int(total_steps)
        self.warmup_steps = int(max(0, warmup_steps))
        self.exponent = float(exponent)

    # -- core ------------------------------------------------------------- #
    def _progress(self, step: int) -> float:
        """``t / T`` of the cubic curve, in ``[0, 1]``."""
        start = min(self.warmup_steps, self.total_steps)
        span = max(self.total_steps - start, 0)
        if span <= 0:
            return 1.0
        return _clamp(_safe_div(int(step) - start, span), 0.0, 1.0)

    def __call__(self, step: int) -> float:
        step = int(step)
        if step <= self.warmup_steps or self.total_steps <= 0:
            progress = 0.0 if self.total_steps > 0 else 1.0
        else:
            progress = self._progress(step)
        gamma = self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * (
            1.0 - progress
        ) ** self.exponent
        return float(_clamp(gamma, 0.0, 1.0))


class LinearSparsitySchedule(SparsitySchedule):
    """Linear sparsity ramp (kept for ablation / sanity comparison)."""

    def __init__(
        self,
        target_sparsity: float = DEFAULT_TARGET_SPARSITY,
        total_steps: int = 1000,
        initial_sparsity: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        self.target_sparsity = float(_clamp(target_sparsity, 0.0, 1.0))
        self.initial_sparsity = float(_clamp(initial_sparsity, 0.0, 1.0))
        self.total_steps = int(total_steps)
        self.warmup_steps = int(max(0, warmup_steps))

    def __call__(self, step: int) -> float:
        start = min(self.warmup_steps, self.total_steps)
        span = max(self.total_steps - start, 0)
        progress = 1.0 if span <= 0 else _clamp(_safe_div(int(step) - start, span), 0.0, 1.0)
        gamma = self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * progress
        return float(_clamp(gamma, 0.0, 1.0))


class ConstantSparsitySchedule(SparsitySchedule):
    """No-op schedule (e.g. the recovery stage keeps ``gamma_T`` fixed)."""

    def __init__(self, sparsity: float = DEFAULT_TARGET_SPARSITY, total_steps: int = 0) -> None:
        self.target_sparsity = float(_clamp(sparsity, 0.0, 1.0))
        self.initial_sparsity = self.target_sparsity
        self.total_steps = int(total_steps)

    def __call__(self, step: int) -> float:
        return float(self.target_sparsity)


def make_sparsity_schedule(
    kind: str = "cubic",
    *,
    target_sparsity: float = DEFAULT_TARGET_SPARSITY,
    total_steps: int = 1000,
    initial_sparsity: float = 1.0,
    warmup_steps: int = 0,
    exponent: float = DEFAULT_CUBIC_EXPONENT,
) -> SparsitySchedule:
    """Factory used by the configs / training loop."""
    key = str(kind).strip().lower()
    if key in ("cubic", "cubic_sparsity", "cubic-schedule"):
        return CubicSparsitySchedule(
            target_sparsity=target_sparsity,
            total_steps=total_steps,
            initial_sparsity=initial_sparsity,
            warmup_steps=warmup_steps,
            exponent=exponent,
        )
    if key in ("linear", "linear_sparsity"):
        return LinearSparsitySchedule(
            target_sparsity=target_sparsity,
            total_steps=total_steps,
            initial_sparsity=initial_sparsity,
            warmup_steps=warmup_steps,
        )
    if key in ("constant", "none"):
        return ConstantSparsitySchedule(sparsity=target_sparsity, total_steps=total_steps)
    raise ValueError(f"unknown sparsity schedule: {kind!r}")


# --------------------------------------------------------------------------- #
# mu (distillation) schedule
# --------------------------------------------------------------------------- #


class MuSchedule:
    """Linear ramp for the self-distillation weight ``mu`` (§4.4).

    ``mu = 0`` before pruning starts (pure task loss) and ``mu = 1`` at the end
    of the pruning stage (pure distillation + recovery).  ``start_value`` /
    ``end_value`` are exposed because Ablations (e.g. "w/o D_S") disable the
    distillation term entirely.
    """

    def __init__(
        self,
        pruning_start_step: int = 0,
        pruning_end_step: int = 1000,
        start_value: float = 0.0,
        end_value: float = 1.0,
    ) -> None:
        self.pruning_start_step = int(pruning_start_step)
        self.pruning_end_step = int(max(pruning_end_step, pruning_start_step))
        self.start_value = float(start_value)
        self.end_value = float(end_value)

    def __call__(self, step: int) -> float:
        step = int(step)
        if step <= self.pruning_start_step:
            return self.start_value
        if step >= self.pruning_end_step:
            return self.end_value
        span = max(self.pruning_end_step - self.pruning_start_step, 1)
        progress = _clamp(_safe_div(step - self.pruning_start_step, span), 0.0, 1.0)
        return float(
            self.start_value + (self.end_value - self.start_value) * progress
        )

    def at(self, step: int) -> float:
        return float(self(step))

    def is_active(self, step: int) -> bool:
        return float(self(step)) > 0.0

    def values(self, steps: Optional[Iterable[int]] = None) -> List[float]:
        if steps is None:
            steps = range(self.pruning_end_step + 1)
        return [float(self(s)) for s in steps]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"MuSchedule(start={self.pruning_start_step}, end={self.pruning_end_step}, "
            f"from={self.start_value}, to={self.end_value})"
        )


def make_mu_schedule(
    *,
    pruning_start_step: int = 0,
    pruning_end_step: int = 1000,
    start_value: float = 0.0,
    end_value: float = 1.0,
    enabled: bool = True,
) -> MuSchedule:
    if not enabled:
        return MuSchedule(
            pruning_start_step=pruning_start_step,
            pruning_end_step=pruning_end_step,
            start_value=0.0,
            end_value=0.0,
        )
    return MuSchedule(
        pruning_start_step=pruning_start_step,
        pruning_end_step=pruning_end_step,
        start_value=start_value,
        end_value=end_value,
    )


# --------------------------------------------------------------------------- #
# Tuning budget (Delta_t) and rank updates
# --------------------------------------------------------------------------- #


def rank_update(
    rank: int,
    prev_budget: float,
    new_budget: float,
    *,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> int:
    """Rank growth rule of §4.3.

    The paper increases the ranks of the top-half salient layers *linearly*
    according to the tuning budget::

        r_apt' = floor(r_apt * Delta_t' / Delta_t)

    Ranks never shrink (APT only grows the adapter ranks) and are clamped to
    ``[min_rank, max_rank]``.
    """
    prev = float(prev_budget)
    new = float(new_budget)
    if prev <= 0.0 or new <= 0.0:
        return int(max(int(rank), int(min_rank)))
    grown = int(math.floor(int(rank) * (new / prev)))
    grown = max(grown, int(rank), int(min_rank))
    if max_rank is not None:
        grown = min(grown, int(max_rank))
    return int(grown)


class TuningBudgetSchedule:
    """Limit ``Delta_t`` on the number of tuning parameters (Eq. (1) of §3).

    The budget ramps linearly from ``initial`` to ``final`` across the pruning
    stage.  ``final`` defaults to ``initial * max_growth`` which corresponds to
    "salient layers' ranks linearly increased" (Appendix A).
    """

    def __init__(
        self,
        initial: float,
        final: Optional[float] = None,
        *,
        pruning_start_step: int = 0,
        pruning_end_step: int = 1000,
        total_steps: Optional[int] = None,
        max_growth: float = 1.0,
        exponent: float = 1.0,
    ) -> None:
        self.initial = float(initial)
        self.max_growth = float(max_growth)
        if final is None:
            final = float(initial) * max(self.max_growth, 1.0)
        self.final = float(final)
        self.pruning_start_step = int(pruning_start_step)
        self.pruning_end_step = int(max(pruning_end_step, pruning_start_step))
        self.total_steps = int(total_steps if total_steps is not None else self.pruning_end_step)
        self.exponent = float(exponent)

    def __call__(self, step: int) -> float:
        step = int(step)
        if step <= self.pruning_start_step:
            return self.initial
        if step >= self.pruning_end_step:
            return self.final
        span = max(self.pruning_end_step - self.pruning_start_step, 1)
        progress = _clamp(_safe_div(step - self.pruning_start_step, span), 0.0, 1.0)
        progress = progress ** self.exponent
        return float(self.initial + (self.final - self.initial) * progress)

    def at(self, step: int) -> float:
        return float(self(step))

    def ratio(self, step: int, prev_step: int) -> float:
        """``Delta_t' / Delta_t`` used by :func:`rank_update`."""
        prev = self.at(prev_step)
        cur = self.at(step)
        return _safe_div(cur, prev, default=1.0)

    def rank_for_step(self, rank: int, step: int, prev_step: int, **kwargs) -> int:
        return rank_update(rank, self.at(prev_step), self.at(step), **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"TuningBudgetSchedule(initial={self.initial}, final={self.final}, "
            f"start={self.pruning_start_step}, end={self.pruning_end_step})"
        )


class CubicTuningBudgetSchedule(TuningBudgetSchedule):
    """Variant where ``Delta_t`` follows the cubic (mirror) shape."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("exponent", 3.0)
        super().__init__(*args, **kwargs)


class ConstantBudgetSchedule(TuningBudgetSchedule):
    """Fixed tuning budget (used when the adapters are frozen by design)."""

    def __init__(self, value: float, **kwargs) -> None:
        super().__init__(value, value, **kwargs)

    def __call__(self, step: int) -> float:  # noqa: D102
        return self.initial


def make_tuning_budget_schedule(
    *,
    initial: Optional[float] = None,
    final: Optional[float] = None,
    kind: str = "linear",
    max_growth: float = 1.0,
    pruning_start_step: int = 0,
    pruning_end_step: int = 1000,
    total_steps: Optional[int] = None,
) -> Optional[TuningBudgetSchedule]:
    """Factory; returns ``None`` when no budget is provided (rank growth off)."""
    if initial is None or initial <= 0:
        return None
    key = str(kind).strip().lower()
    cls = CubicTuningBudgetSchedule if key == "cubic" else TuningBudgetSchedule
    return cls(
        initial=float(initial),
        final=None if final is None else float(final),
        max_growth=max_growth,
        pruning_start_step=pruning_start_step,
        pruning_end_step=pruning_end_step,
        total_steps=total_steps,
    )


# --------------------------------------------------------------------------- #
# Mask decay
# --------------------------------------------------------------------------- #


def anneal_mask(
    current,
    target: float,
    alpha: float = DEFAULT_ALPHA,
):
    """One gradual mask update of Algorithm 1.

    ``M_1 <- min(1, M_1 + alpha)`` for retained blocks (``target == 1``) and
    ``M_0 <- max(0, M_0 - alpha)`` for pruned blocks (``target == 0``).
    Works for floats and for tensor-like objects supporting ``+``/``min``.
    """
    alpha = float(alpha)
    if target >= 0.5:
        try:
            return min(1.0, float(current) + alpha)
        except (TypeError, ValueError):
            raise
    return max(0.0, float(current) - alpha)


def anneal_masks(masks, targets, alpha: float = DEFAULT_ALPHA):
    """Vectorised gradual update over an iterable of float masks."""
    return [anneal_mask(m, t, alpha) for m, t in zip(masks, targets)]


class MaskDecaySchedule:
    """Per-step mask decay rate ``alpha``.

    The paper uses a constant ``alpha = 0.01``.  A decreasing ``alpha``
    (``decay_kind="linear"``) is provided for stability experiments: the mask
    approaches its target more slowly the longer pruning runs.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        *,
        total_steps: int = 0,
        decay_kind: str = "constant",
        final_alpha: Optional[float] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.total_steps = int(total_steps)
        self.decay_kind = str(decay_kind).strip().lower()
        self.final_alpha = float(final_alpha if final_alpha is not None else alpha)

    def __call__(self, step: int) -> float:
        if self.decay_kind != "linear" or self.total_steps <= 0:
            return self.alpha
        progress = _clamp(_safe_div(int(step), self.total_steps), 0.0, 1.0)
        return float(self.alpha + (self.final_alpha - self.alpha) * progress)

    def at(self, step: int) -> float:
        return float(self(step))

    # -- convenience wrappers -------------------------------------------- #
    def update(self, current, target: float, step: Optional[int] = None, alpha: Optional[float] = None):
        a = self.at(step) if (alpha is None and step is not None) else (
            self.alpha if alpha is None else float(alpha)
        )
        return anneal_mask(current, target, a)

    def harden(self, current: float, threshold: float = 0.5) -> float:
        return 1.0 if float(current) >= float(threshold) else 0.0

    def steps_to_target(self, current: float, target: float, alpha: Optional[float] = None) -> int:
        """Number of gradual updates needed to move ``current`` to ``target``."""
        a = float(self.alpha if alpha is None else alpha)
        if a <= 0.0:
            return 0
        delta = abs(float(current) - float(target))
        return int(math.ceil(delta / a))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"MaskDecaySchedule(alpha={self.alpha}, kind={self.decay_kind})"


# --------------------------------------------------------------------------- #
# Adjustment steps (the set T of Algorithm 1)
# --------------------------------------------------------------------------- #


class AdjustmentStepSchedule:
    """Decides on which steps blocks are re-selected / masks are updated.

    Algorithm 1 lists an "Adjustment step set" ``T``; in practice the block
    selection and mask update run every ``interval`` steps (default: every step).
    """

    def __init__(
        self,
        interval: int = 1,
        *,
        pruning_start_step: int = 0,
        pruning_end_step: int = 1000,
        total_steps: Optional[int] = None,
    ) -> None:
        self.interval = int(max(1, interval))
        self.pruning_start_step = int(pruning_start_step)
        self.pruning_end_step = int(max(pruning_end_step, pruning_start_step))
        self.total_steps = int(total_steps if total_steps is not None else self.pruning_end_step)

    def __call__(self, step: int) -> bool:
        step = int(step)
        if step < self.pruning_start_step or step > self.pruning_end_step:
            return False
        return (step - self.pruning_start_step) % self.interval == 0

    def should_adjust(self, step: int) -> bool:
        return bool(self(step))

    def adjust_steps(self) -> List[int]:
        return [
            s
            for s in range(self.pruning_start_step, self.pruning_end_step + 1)
            if self(s)
        ]

    def is_pruning_step(self, step: int) -> bool:
        return self.pruning_start_step <= int(step) <= self.pruning_end_step

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"AdjustmentStepSchedule(interval={self.interval}, "
            f"start={self.pruning_start_step}, end={self.pruning_end_step})"
        )


# --------------------------------------------------------------------------- #
# Learning-rate schedule (linear warmup + linear decay)
# --------------------------------------------------------------------------- #


def lr_factor(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    min_factor: float = 0.0,
    kind: str = "linear",
) -> float:
    """Multiplier applied to the base learning rate.

    Defaults follow the reproduction plan: linear warmup then linear decay
    ("linear decay after warmup"), with ``min_factor`` as the floor.
    """
    step = int(step)
    total_steps = int(max(total_steps, 1))
    warmup_steps = int(max(warmup_steps, 0))
    if warmup_steps > 0 and step < warmup_steps:
        return float(max(min_factor, _safe_div(step, warmup_steps, default=1.0)))
    span = max(total_steps - warmup_steps, 1)
    progress = _clamp(_safe_div(step - warmup_steps, span), 0.0, 1.0)
    if str(kind).strip().lower() == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        factor = 1.0 - progress
    return float(max(min_factor, factor))


def build_lr_scheduler(
    optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    kind: str = "linear",
    min_factor: float = 0.0,
):
    """Wrap :func:`lr_factor` into a torch ``LambdaLR`` (lazy torch import)."""
    import torch  # local import keeps the module usable without torch

    def _fn(step: int) -> float:
        return lr_factor(
            step, total_steps, warmup_steps=warmup_steps, min_factor=min_factor, kind=kind
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_fn)


# --------------------------------------------------------------------------- #
# Aggregated schedule bank
# --------------------------------------------------------------------------- #


@dataclass
class ScheduleBank:
    """Bundle of all schedules a training run needs.

    ``step(global_step)`` returns the scalar values consumed by the APT training
    loop (Appendix C, Algorithm 1): the sparsity constraint, the ``mu`` weight,
    the mask decay rate, the tuning budget and whether this step adjusts blocks.
    """

    sparsity: SparsitySchedule
    mu: MuSchedule
    mask_decay: MaskDecaySchedule
    adjustment: AdjustmentStepSchedule
    tuning_budget: Optional[TuningBudgetSchedule] = None
    lr_kind: str = "linear"
    lr_warmup_steps: int = 0
    lr_min_factor: float = 0.0
    meta: Dict[str, float] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------- #
    def at(self, step: int) -> Dict[str, float]:
        step = int(step)
        values = {
            "step": float(step),
            "sparsity": float(self.sparsity(step)),
            "mu": float(self.mu(step)),
            "alpha": float(self.mask_decay(step)),
            "adjust": 1.0 if self.adjustment(step) else 0.0,
        }
        budget = self.tuning_budget(step) if self.tuning_budget is not None else 0.0
        values["tuning_budget"] = float(budget)
        values["lr_factor"] = lr_factor(
            step,
            self.meta.get("total_steps", self.adjustment.total_steps),
            warmup_steps=self.lr_warmup_steps,
            min_factor=self.lr_min_factor,
            kind=self.lr_kind,
        )
        return values

    def step(self, step: int) -> Dict[str, float]:
        return self.at(step)

    def parameter_budget(self, step: int, original_param_count: float) -> float:
        """Constraint of Eq. (1) for the block selector's binary search."""
        return self.sparsity.parameter_budget(step, original_param_count)

    def rank_for_step(
        self, rank: int, step: int, prev_step: Optional[int] = None, **kwargs
    ) -> int:
        if self.tuning_budget is None:
            return int(rank)
        if prev_step is None:
            prev_step = max(int(step) - 1, 0)
        return self.tuning_budget.rank_for_step(rank, step, prev_step, **kwargs)

    def is_pruning_phase(self, step: int) -> bool:
        return self.adjustment.is_pruning_step(step)

    def is_recovery_phase(self, step: int) -> bool:
        return not self.is_pruning_phase(step)

    def curve(self, n_steps: Optional[int] = None) -> List[Dict[str, float]]:
        total = int(n_steps if n_steps is not None else self.meta.get("total_steps", 0))
        return [self.at(s) for s in range(max(total, 0) + 1)]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"ScheduleBank(sparsity={self.sparsity!r}, mu={self.mu!r}, "
            f"mask_decay={self.mask_decay!r}, adjustment={self.adjustment!r})"
        )


def build_schedules(
    config: Optional[Dict] = None,
    *,
    target_sparsity: float = DEFAULT_TARGET_SPARSITY,
    total_steps: int = 1000,
    pruning_start_step: int = 0,
    pruning_end_step: Optional[int] = None,
    alpha: float = DEFAULT_ALPHA,
    initial_rank: int = DEFAULT_INITIAL_RANK,
    tuning_budget_initial: Optional[float] = None,
    tuning_budget_final: Optional[float] = None,
    adjustment_interval: int = 1,
    lr_warmup_steps: int = 0,
    lr_kind: str = "linear",
    **overrides,
) -> ScheduleBank:
    """Create a :class:`ScheduleBank` from a config dict or explicit arguments.

    Values found in ``config`` (e.g. ``apt/configs/default.yaml``) take
    precedence over the equivalent keyword arguments, so the experiment scripts
    can simply pass the loaded YAML.
    """
    cfg = dict(config or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    total_steps = int(cfg.get("total_steps", total_steps))
    target_sparsity = float(cfg.get("target_sparsity", target_sparsity))
    alpha = float(cfg.get("alpha", alpha))
    pruning_start_step = int(cfg.get("pruning_start_step", pruning_start_step))
    if pruning_end_step is None:
        pruning_end_step = cfg.get("pruning_end_step", cfg.get("prune_steps", None))
    pruning_end_step = int(
        pruning_end_step if pruning_end_step is not None else max(total_steps, 1)
    )
    adjustment_interval = int(cfg.get("adjustment_interval", adjustment_interval))

    sparsity = make_sparsity_schedule(
        cfg.get("sparsity_schedule", cfg.get("schedule", "cubic")),
        target_sparsity=target_sparsity,
        total_steps=total_steps,
        initial_sparsity=float(cfg.get("initial_sparsity", 1.0)),
        warmup_steps=int(cfg.get("sparsity_warmup_steps", 0)),
        exponent=float(cfg.get("cubic_exponent", DEFAULT_CUBIC_EXPONENT)),
    )

    mu = make_mu_schedule(
        pruning_start_step=pruning_start_step,
        pruning_end_step=pruning_end_step,
        start_value=float(cfg.get("mu_start", 0.0)),
        end_value=float(cfg.get("mu_end", 1.0)),
        enabled=bool(cfg.get("use_distillation",
                             cfg.get("distillation", cfg.get("use_self_distillation", True)))),
    )

    mask_decay = MaskDecaySchedule(
        alpha=alpha,
        total_steps=total_steps,
        decay_kind=str(cfg.get("alpha_schedule", cfg.get("mask_decay_kind", "constant"))),
        final_alpha=cfg.get("alpha_final", None),
    )

    adjustment = AdjustmentStepSchedule(
        interval=adjustment_interval,
        pruning_start_step=pruning_start_step,
        pruning_end_step=pruning_end_step,
        total_steps=total_steps,
    )

    budget_initial = cfg.get("tuning_budget_initial", tuning_budget_initial)
    budget_final = cfg.get("tuning_budget_final", tuning_budget_final)
    tuning_budget = make_tuning_budget_schedule(
        initial=budget_initial,
        final=budget_final,
        kind=str(cfg.get("tuning_budget_schedule", "linear")),
        max_growth=float(cfg.get("max_rank_growth", cfg.get("max_growth", 1.0))),
        pruning_start_step=pruning_start_step,
        pruning_end_step=pruning_end_step,
        total_steps=total_steps,
    )

    meta = {
        "total_steps": float(total_steps),
        "pruning_start_step": float(pruning_start_step),
        "pruning_end_step": float(pruning_end_step),
        "target_sparsity": float(target_sparsity),
        "initial_rank": float(cfg.get("initial_rank", initial_rank)),
        "scaling": float(cfg.get("scaling", cfg.get("scaling_factor", DEFAULT_SCALING))),
        "alpha": float(alpha),
        "ema_beta": float(cfg.get("ema_beta", cfg.get("beta", DEFAULT_EMA_BETA))),
        "tau": float(cfg.get("tau", 4.0)),
    }

    return ScheduleBank(
        sparsity=sparsity,
        mu=mu,
        mask_decay=mask_decay,
        adjustment=adjustment,
        tuning_budget=tuning_budget,
        lr_kind=str(cfg.get("lr_schedule", lr_kind)),
        lr_warmup_steps=int(cfg.get("lr_warmup_steps", lr_warmup_steps)),
        lr_min_factor=float(cfg.get("lr_min_factor", 0.0)),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _self_test() -> None:  # pragma: no cover - manual sanity check
    # 1) cubic schedule reproduces the paper's formula gamma_T + (1-gamma_T)(1-t/T)^3
    gamma_T, T = 0.6, 100
    sched = CubicSparsitySchedule(target_sparsity=gamma_T, total_steps=T, initial_sparsity=1.0)
    for t in (0, 25, 50, 75, 100, 120):
        expected = gamma_T + (1.0 - gamma_T) * (1.0 - min(t, T) / T) ** 3
        got = sched(t)
        assert abs(got - expected) < 1e-6, (t, got, expected)
    assert abs(sched(0) - 1.0) < 1e-9
    assert abs(sched(T) - gamma_T) < 1e-9
    # monotone decreasing
    prev = sched(0)
    for t in range(1, T + 1):
        cur = sched(t)
        assert cur <= prev + 1e-12
        prev = cur
    # parameter budget of Eq. (1)
    assert abs(sched.parameter_budget(T, 1000.0) - 400.0) < 1e-6

    # 2) mu ramp: 0 before pruning, 1 at the end of pruning
    mu = MuSchedule(pruning_start_step=10, pruning_end_step=110)
    assert mu(0) == 0.0 and mu(10) == 0.0
    assert abs(mu(60) - 0.5) < 1e-9
    assert mu(110) == 1.0 and mu(200) == 1.0

    # 3) rank update r' = floor(r * Delta_t' / Delta_t)
    assert rank_update(8, 100.0, 150.0) == 12
    assert rank_update(8, 100.0, 100.0) == 8
    assert rank_update(8, 100.0, 50.0) == 8  # never shrinks
    budget = TuningBudgetSchedule(initial=100.0, final=200.0,
                                  pruning_start_step=0, pruning_end_step=100)
    assert abs(budget(50) - 150.0) < 1e-9
    assert rank_update(8, budget(0), budget(50)) == 12
    assert rank_update(8, budget(50), budget(100)) == 10  # floor(8*200/150)

    # 4) mask decay: gradual by alpha=0.01 (Algorithm 1)
    decay = MaskDecaySchedule(alpha=DEFAULT_ALPHA)
    assert abs(decay.update(0.5, 0.0) - 0.49) < 1e-12
    assert abs(decay.update(0.5, 1.0) - 0.51) < 1e-12
    assert abs(decay.update(0.005, 0.0) - 0.0) < 1e-12  # clamped by max(0, .)
    assert abs(decay.update(0.999, 1.0) - 1.0) < 1e-12  # clamped by min(1, .)
    assert decay.steps_to_target(1.0, 0.0) == 100

    # 5) LR factor: linear warmup + linear decay
    assert abs(lr_factor(0, 100, warmup_steps=10) - 0.0) < 1e-9
    assert abs(lr_factor(5, 100, warmup_steps=10) - 0.5) < 1e-9
    assert abs(lr_factor(10, 100, warmup_steps=10) - 1.0) < 1e-9
    assert abs(lr_factor(100, 100, warmup_steps=10) - 0.0) < 1e-9

    # 6) full bank on a Table 6 style configuration (SST2 / GLUE-big)
    steps_per_epoch = 1000
    window = compute_pruning_window(total_epochs=40, distill_epochs=20,
                                    steps_per_epoch=steps_per_epoch)
    assert window["pruning_steps"] == window["total_steps"] // 2
    bank = build_schedules(
        None,
        target_sparsity=0.6,
        total_steps=window["total_steps"],
        pruning_start_step=window["pruning_start_step"],
        pruning_end_step=window["pruning_end_step"],
        tuning_budget_initial=1e6,
        tuning_budget_final=4e6,
        lr_warmup_steps=0.06 * window["total_steps"],
    )
    v0, vmid, vend = bank.at(0), bank.at(window["pruning_steps"]), bank.at(window["total_steps"])
    assert abs(v0["sparsity"] - 1.0) < 1e-9
    assert abs(vend["sparsity"] - 0.6) < 1e-9
    assert v0["mu"] == 0.0 and vend["mu"] == 1.0
    assert v0["tuning_budget"] < vend["tuning_budget"]
    assert bank.parameter_budget(0, 1e6) == 0.0
    assert abs(bank.parameter_budget(window["total_steps"], 1e6) - 4e5) < 1e-6
    assert vmid["alpha"] == DEFAULT_ALPHA
    assert bank.is_pruning_phase(0) and not bank.is_recovery_phase(window["total_steps"])
    print("[schedulers] self-test passed:", {k: round(v, 4) for k, v in vend.items()})


if __name__ == "__main__":  # pragma: no cover
    _self_test()
