"""Adaptive and Efficient LM Tuning (A_T) -- adapter rank controller.

Implements the tuning half of APT (Section 4.3, Appendix A, Section 6):

Salience scoring of APT adapters
    ``I(H_apt) = sum_{i,j} S(W_B i,j)`` -- the summation of the tuning-parameter
    salience scores in ``W_B`` (Equation (3)), computed from the reduction over
    batch/sequence of ``|activation * gradient|`` products.

Dynamically adding APT adapter parameters
    * sort all tuning layers by their importance ``I(H_apt)``,
    * linearly increase the ranks of the **top-half** salient adapters following
      the tuning budget ``Delta_t``,
    * ``r_apt' = floor(r_apt * Delta_t' / Delta_t)``,
    * concatenate Gaussian ``N(0, sigma^2)`` rows to ``W_A`` and zeros to
      ``W_B`` (same as LoRA initialization) so the layer output is unchanged
      before and after the new parameters are added.

Training stability (Section 6)
    "We reset the optimizer every time after each parameter size changes", so the
    controller tracks a pending ``needs_optimizer_reset`` flag and can either
    clear the optimizer state in place or recreate a fresh optimizer object.

The controller is intentionally duck-typed: it only requires objects exposing an
``adapter`` attribute with ``rank`` / ``lora_b`` / ``increase_rank`` (as provided
by :class:`apt.adapters.MaskedLinear` / :class:`apt.adapters.APTAdapter`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------------------
# Optional intra-package imports (defensive: module must import in a bare env).
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .adapters import APTAdapter, MaskedLinear, iter_masked_linears
except Exception:  # pragma: no cover
    APTAdapter = None  # type: ignore
    MaskedLinear = None  # type: ignore

    def iter_masked_linears(module, names=None):  # type: ignore
        """Fallback discovery of APT-wrapped linear layers."""
        for name, mod in module.named_modules():
            if hasattr(mod, "adapter") and hasattr(mod, "base_weight"):
                if names is None or name in names:
                    yield name, mod


try:  # pragma: no cover - import guard
    from .schedulers import (
        DEFAULT_INITIAL_RANK,
        DEFAULT_SCALING,
        TuningBudgetSchedule,
        rank_update as _scheduler_rank_update,
    )
except Exception:  # pragma: no cover
    DEFAULT_INITIAL_RANK = 8
    DEFAULT_SCALING = 2.0
    TuningBudgetSchedule = None  # type: ignore
    _scheduler_rank_update = None  # type: ignore


__all__ = [
    "RankController",
    "AdapterRankController",
    "TuningController",
    "RankUpdateResult",
    "RankControllerConfig",
    "adapter_importance",
    "importance_from_salience",
    "top_half_names",
    "top_fraction_names",
    "rank_after_budget",
    "local_rank_update",
    "reset_optimizer_state",
    "recreate_optimizer",
    "build_optimizer",
    "trainable_parameters",
    "DEFAULT_TOP_FRACTION",
    "DEFAULT_MIN_RANK",
    "IMPORTANCE_MODES",
]

DEFAULT_TOP_FRACTION = 0.5  # "top-half salient ones" (Section 4.3)
DEFAULT_MIN_RANK = 1
IMPORTANCE_MODES = ("salience", "weight", "uniform", "random")


# ======================================================================================
# Small numeric helpers
# ======================================================================================
def _to_float(value: Any) -> float:
    """Best-effort conversion of tensor / numpy / python scalars to ``float``."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().sum().item())
    try:  # numpy scalars / arrays
        return float(value)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        return float(sum(float(v) for v in value))
    except Exception:
        return 0.0


def _param_norm(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().float().norm().item())


def local_rank_update(
    rank: int,
    prev_budget: float,
    new_budget: float,
    *,
    min_rank: int = DEFAULT_MIN_RANK,
    max_rank: Optional[int] = None,
) -> int:
    """``r_apt' = floor(r_apt * Delta_t' / Delta_t)`` (Section 4.3).

    Ranks never shrink (APT only grows its tuning parameters) and are clamped to
    ``[min_rank, max_rank]``.
    """
    rank = max(int(rank), int(min_rank))
    if prev_budget is None or new_budget is None:
        return rank
    prev_budget = float(prev_budget)
    new_budget = float(new_budget)
    if prev_budget <= 0.0 or new_budget <= 0.0:
        return rank
    ratio = new_budget / prev_budget
    if ratio <= 1.0:
        # Budget did not grow: keep the current rank (never shrink).
        return rank
    new_rank = int(math.floor(rank * ratio))
    new_rank = max(new_rank, rank)  # monotone growth
    if max_rank is not None:
        new_rank = min(new_rank, int(max_rank))
    return int(new_rank)


def rank_after_budget(
    rank: int,
    prev_budget: float,
    new_budget: float,
    *,
    min_rank: int = DEFAULT_MIN_RANK,
    max_rank: Optional[int] = None,
) -> int:
    """Alias of :func:`local_rank_update`, preferring the scheduler implementation."""
    if _scheduler_rank_update is not None:
        try:
            return int(
                _scheduler_rank_update(
                    rank, prev_budget, new_budget, min_rank=min_rank, max_rank=max_rank
                )
            )
        except TypeError:
            return int(_scheduler_rank_update(rank, prev_budget, new_budget))
        except Exception:
            pass
    return local_rank_update(
        rank, prev_budget, new_budget, min_rank=min_rank, max_rank=max_rank
    )


def top_fraction_names(
    importance: Dict[str, float],
    fraction: float = DEFAULT_TOP_FRACTION,
    *,
    min_keep: int = 1,
) -> List[str]:
    """Names of the most salient adapters ("top-half" when ``fraction=0.5``).

    Sorted by descending importance with a deterministic name tie-break.
    """
    names = sorted(importance.keys(), key=lambda n: (-float(importance[n]), str(n)))
    if not names:
        return []
    keep = int(math.ceil(len(names) * float(fraction)))
    keep = max(keep, int(min_keep)) if names else 0
    keep = min(keep, len(names))
    return names[:keep]


def top_half_names(
    importance: Dict[str, float], *, min_keep: int = 1
) -> List[str]:
    """Convenience wrapper for the paper's "top-half salient" adapters."""
    return top_fraction_names(importance, DEFAULT_TOP_FRACTION, min_keep=min_keep)


def importance_from_salience(salience: Any, name: str) -> Optional[float]:
    """Extract ``I(H_apt)`` for ``name`` from a salience container.

    Accepts the return value of ``apt.salience.OutlierAwareSalience.collect()``
    (a dict with an inner ``"bsal"`` mapping) or a flat ``name -> score`` dict.
    For the inner ``bsal`` tensors the **total** entry (``[-1]``) is used, since
    that holds ``I(H_apt)`` for the whole adapter.
    """
    if salience is None:
        return None
    if isinstance(salience, torch.Tensor):  # not a mapping: cannot resolve a name
        return None
    if isinstance(salience, dict):
        inner = None
        for key in ("bsal", "adapter", "tuning", "adapters"):
            if key in salience and isinstance(salience[key], dict):
                inner = salience[key]
                break
        if inner is not None and name in inner:
            value = inner[name]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return 0.0
                return float(value.detach().float().reshape(-1)[-1].item())
            if isinstance(value, (list, tuple)) and value:
                return _to_float(value[-1])
            return _to_float(value)
        for key in (name, str(name)):
            if key in salience:
                value = salience[key]
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return 0.0
                    return float(value.detach().float().reshape(-1)[-1].item())
                if isinstance(value, (list, tuple)) and value:
                    return _to_float(value[-1])
                return _to_float(value)
        # last resort: nested dicts keyed by name
        for value in salience.values():
            if isinstance(value, dict) and name in value:
                return importance_from_salience(value, name)
    return None


def _adapter_of(module: Any) -> Any:
    return getattr(module, "adapter", None)


def adapter_importance(
    module: Any,
    salience: Any = None,
    *,
    name: str = "",
    mode: str = "salience",
    fallback: float = 0.0,
) -> float:
    """``I(H_apt) = sum_{i,j} S(W_B i,j)`` for one wrapped linear layer.

    ``mode``:
      * ``"salience"`` -- prefer the externally computed ``bsal`` (or the value
        cached on the adapter via ``set_b_salience``), then fall back to the
        adapter's own ``importance()``, then to ``||W_B||_F``;
      * ``"weight"`` -- ``||W_B||_F`` proxy (used by the importance ablation);
      * ``"uniform"`` -- identical importance for every adapter;
      * ``"random"`` -- deterministic pseudo-random importance (sanity checks).
    """
    mode = (mode or "salience").lower()
    if mode not in IMPORTANCE_MODES:
        mode = "salience"

    if mode == "uniform":
        return 1.0
    if mode == "random":
        rng = random.Random(hash(str(name)) & 0xFFFFFFFF)
        return rng.random()

    adapter = _adapter_of(module)
    if adapter is None:
        return float(fallback)

    if mode == "weight":
        return _param_norm(getattr(adapter, "lora_b", None))

    # mode == "salience"
    external = importance_from_salience(salience, name) if salience is not None else None
    if external is not None:
        return float(external)

    for attr in ("_bsal", "bsal", "_salience", "b_salience"):
        cached = getattr(adapter, attr, None)
        if cached is not None:
            value = _to_float(cached)
            if value:
                return value

    importance_fn = getattr(adapter, "importance", None)
    if callable(importance_fn):
        try:
            value = _to_float(importance_fn())
            if value:
                return value
        except Exception:
            pass

    return _param_norm(getattr(adapter, "lora_b", None))


# ======================================================================================
# Optimizer helpers (Section 6: reset after every parameter-size change)
# ======================================================================================
def trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """All parameters of ``model`` requiring gradients, de-duplicated."""
    seen: set = set()
    params: List[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        params.append(param)
    return params


def reset_optimizer_state(optimizer: Optional[torch.optim.Optimizer]) -> bool:
    """Clear an optimizer's momentum/variance state in place.

    Returns ``True`` when an optimizer was actually reset.
    """
    if optimizer is None:
        return False
    try:
        optimizer.state.clear()
    except Exception:
        return False
    return True


def recreate_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    **overrides: Any,
) -> torch.optim.Optimizer:
    """Build a *fresh* optimizer of the same class/hyperparameters.

    Because APT changes parameter objects (rank growth re-creates ``nn.Parameter``
    tensors and pruning removes entries), the safest interpretation of "reset the
    optimizer every time after each parameter size changes" (Section 6) is to
    create a new optimizer instance over the current parameters, which also drops
    all stale momentum/variance state.
    """
    cls = type(optimizer)
    defaults = dict(getattr(optimizer, "defaults", {}) or {})
    params = list(parameters) if parameters is not None else []
    if not params:
        for group in optimizer.param_groups:
            params.extend([p for p in group.get("params", []) if p.requires_grad])
    if not params:
        return optimizer

    # Group-level overrides (e.g. per-group lr / weight decay) are preserved.
    kwargs: Dict[str, Any] = {}
    first_group = optimizer.param_groups[0] if optimizer.param_groups else {}
    for key, value in first_group.items():
        if key == "params":
            continue
        kwargs[key] = value
    # ``defaults`` holds the canonical hyper-parameters; group values win.
    merged = dict(defaults)
    merged.update(kwargs)
    merged.update(overrides)
    merged = {k: v for k, v in merged.items() if not isinstance(v, (list, tuple)) or k in ("betas",)}
    try:
        return cls(params, **merged)
    except TypeError:
        # Fall back to only the truly common keyword arguments.
        safe = {
            k: v
            for k, v in merged.items()
            if k in ("lr", "weight_decay", "betas", "eps", "momentum", "dampening", "nesterov", "amsgrad")
        }
        return cls(params, **safe)


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    name: str = "adamw",
    parameters: Optional[Iterable[torch.nn.Parameter]] = None,
) -> torch.optim.Optimizer:
    """AdamW (default) optimizer over the model's trainable parameters.

    The plan's missing-detail default is AdamW with weight decay 0.01 and the
    Table 6 learning rates (2e-4 for GLUE/SQuAD, 1e-4 for CNN/DM).
    """
    params = list(parameters) if parameters is not None else trainable_parameters(model)
    name = (name or "adamw").lower()
    if name in ("adamw", "adam_w"):
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
    if name in ("adam",):
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)
    if name in ("sgd",):
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer name: {name!r}")


# ======================================================================================
# Result containers
# ======================================================================================
@dataclass
class RankControllerConfig:
    """Hyper-parameters of the adaptive tuning rank controller."""

    initial_rank: int = DEFAULT_INITIAL_RANK  # Appendix A: ranks start at 8
    scaling: float = DEFAULT_SCALING  # Appendix A: scaling factor 2, static
    top_fraction: float = DEFAULT_TOP_FRACTION  # "top-half salient ones"
    min_rank: int = DEFAULT_MIN_RANK
    max_rank: Optional[int] = None
    max_step_ratio: Optional[float] = None
    importance_mode: str = "salience"
    enabled: bool = True  # ``False`` reproduces the "w/o A_T" ablation
    reset_optimizer: bool = True
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "initial_rank": self.initial_rank,
            "scaling": self.scaling,
            "top_fraction": self.top_fraction,
            "min_rank": self.min_rank,
            "max_rank": self.max_rank,
            "max_step_ratio": self.max_step_ratio,
            "importance_mode": self.importance_mode,
            "enabled": self.enabled,
            "reset_optimizer": self.reset_optimizer,
            "seed": self.seed,
        }


@dataclass
class RankUpdateResult:
    """Outcome of one adaptive-tuning (rank growth) step."""

    step: Optional[int] = None
    budget: float = 0.0
    prev_budget: float = 0.0
    ratio: float = 1.0
    changed: List[str] = field(default_factory=list)
    selected: List[str] = field(default_factory=list)
    old_ranks: Dict[str, int] = field(default_factory=dict)
    new_ranks: Dict[str, int] = field(default_factory=dict)
    importance: Dict[str, float] = field(default_factory=dict)
    added_parameters: int = 0
    optimizer_reset: bool = False
    optimizer: Optional[torch.optim.Optimizer] = None
    skipped: bool = False

    @property
    def grew(self) -> bool:
        return bool(self.changed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "budget": self.budget,
            "prev_budget": self.prev_budget,
            "ratio": self.ratio,
            "changed": list(self.changed),
            "selected": list(self.selected),
            "old_ranks": dict(self.old_ranks),
            "new_ranks": dict(self.new_ranks),
            "added_parameters": self.added_parameters,
            "optimizer_reset": self.optimizer_reset,
            "skipped": self.skipped,
        }

    def summary(self) -> str:
        return (
            f"RankUpdate(step={self.step}, budget={self.budget:.4g}, "
            f"ratio={self.ratio:.4g}, grown={len(self.changed)}, "
            f"added_params={self.added_parameters}, "
            f"optimizer_reset={self.optimizer_reset})"
        )


# ======================================================================================
# Rank controller
# ======================================================================================
class RankController:
    """Adaptive tuning controller for APT adapters (Section 4.3).

    Typical use inside the Algorithm 1 training loop::

        result = controller.step(model=model, salience=salience, step=step,
                                 optimizer=optimizer)
        if result.optimizer is not None:
            optimizer = result.optimizer

    The controller never shrinks ranks and preserves layer outputs exactly when
    new parameters are added (zero ``W_B`` columns, Gaussian ``W_A`` rows).
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        *,
        initial_rank: int = DEFAULT_INITIAL_RANK,
        scaling: float = DEFAULT_SCALING,
        top_fraction: float = DEFAULT_TOP_FRACTION,
        min_rank: int = DEFAULT_MIN_RANK,
        max_rank: Optional[int] = None,
        max_step_ratio: Optional[float] = None,
        importance_mode: str = "salience",
        enabled: bool = True,
        reset_optimizer: bool = True,
        budget_schedule: Any = None,
        budget_initial: float = 1.0,
        budget_final: float = 2.0,
        pruning_start_step: int = 0,
        pruning_end_step: int = 1,
        total_steps: Optional[int] = None,
        growth_kind: str = "linear",
        max_growth: float = 1.0,
        schedule_bank: Any = None,
        seed: Optional[int] = None,
        names: Optional[Sequence[str]] = None,
        config: Optional[RankControllerConfig] = None,
    ) -> None:
        if config is None:
            config = RankControllerConfig(
                initial_rank=int(initial_rank),
                scaling=float(scaling),
                top_fraction=float(top_fraction),
                min_rank=int(min_rank),
                max_rank=max_rank,
                max_step_ratio=max_step_ratio,
                importance_mode=importance_mode,
                enabled=bool(enabled),
                reset_optimizer=bool(reset_optimizer),
                seed=seed,
            )
        self.config = config
        self.model = model
        self.names = list(names) if names is not None else None

        # ----- tuning budget Delta_t -------------------------------------------------
        self.budget_schedule = budget_schedule
        if self.budget_schedule is None and schedule_bank is not None:
            self.budget_schedule = getattr(schedule_bank, "tuning_budget", None)
        if self.budget_schedule is None and budget_schedule is None:
            self.budget_schedule = self._make_default_budget(
                annual_initial=budget_initial,
                final=budget_final,
                pruning_start_step=pruning_start_step,
                pruning_end_step=(
                    pruning_end_step if pruning_end_step is not None else (total_steps or 1)
                ),
                kind=growth_kind,
                max_growth=max_growth,
            )

        self._prev_budget: Optional[float] = None
        self._last_ranks: Dict[str, int] = {}
        self._history: List[RankUpdateResult] = []
        self._optimizer_reset_pending = False
        self._total_growth_events = 0

        if model is not None:
            self.set_initial_ranks(self.config.initial_rank, model=model)

    # ----------------------------------------------------------------------------------
    # Budget construction
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _make_default_budget(
        *,
        annual_initial: float = 1.0,
        final: float = 2.0,
        pruning_start_step: int = 0,
        pruning_end_step: int = 1,
        kind: str = "linear",
        max_growth: float = 1.0,
    ) -> Any:
        """Default relative tuning-budget schedule ``Delta_t``.

        The absolute scale of ``Delta_t`` cancels out in ``r' = floor(r * Delta_t'/Delta_t)``,
        so the default is a *relative* budget ramping from 1.0 to ``final``
        (or ``1 + max_growth``) across the pruning window.
        """
        budget_final = float(final if final is not None else 1.0 + float(max_growth))
        budget_final = max(budget_final, 1.0 + 1e-9)
        if TuningBudgetSchedule is not None:
            try:
                if str(kind).lower() == "cubic":
                    from .schedulers import CubicTuningBudgetSchedule  # type: ignore

                    return CubicTuningBudgetSchedule(
                        initial=annual_initial,
                        final=budget_final,
                        pruning_start_step=pruning_start_step,
                        pruning_end_step=pruning_end_step,
                        max_growth=max_growth,
                    )
                return TuningBudgetSchedule(
                    initial=annual_initial,
                    final=budget_final,
                    pruning_start_step=pruning_start_step,
                    pruning_end_step=pruning_end_step,
                    max_growth=max_growth,
                )
            except Exception:
                pass

        start = float(annual_initial)
        end = float(budget_final)

        def _budget(step: int) -> float:  # pragma: no cover - fallback only
            if pruning_end_step <= pruning_start_step:
                return end
            frac = (float(step) - pruning_start_step) / float(
                pruning_end_step - pruning_start_step
            )
            frac = min(max(frac, 0.0), 1.0)
            if str(kind).lower() == "cubic":
                frac = 1.0 - (1.0 - frac) ** 3
            return start + (end - start) * frac

        return _budget

    # ----------------------------------------------------------------------------------
    # Adapter discovery
    # ----------------------------------------------------------------------------------
    def adapters(self, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
        """Mapping ``name -> wrapped linear`` for tunable APT adapters."""
        model = model if model is not None else self.model
        if model is None:
            return {}
        result: Dict[str, Any] = {}
        for name, module in iter_masked_linears(model, self.names):
            adapter = _adapter_of(module)
            if adapter is None:
                continue
            rank = getattr(adapter, "rank", None)
            lora_a = getattr(adapter, "lora_a", getattr(adapter, "W_A", None))
            if rank is None and lora_a is None:
                continue
            result[str(name)] = module
        return result

    def adapter_names(self, model: Optional[torch.nn.Module] = None) -> List[str]:
        return sorted(self.adapters(model).keys())

    @staticmethod
    def _rank_of(module: Any) -> int:
        adapter = _adapter_of(module)
        rank = getattr(adapter, "rank", None)
        if rank is not None:
            try:
                return int(rank)
            except Exception:
                pass
        for attr in ("lora_a", "W_A"):
            tensor = getattr(adapter, attr, None)
            if isinstance(tensor, torch.Tensor):
                # lora_a is (rank, d_in)
                return int(tensor.shape[-2]) if tensor.dim() == 2 else 0
        return 0

    def ranks(self, model: Optional[torch.nn.Module] = None) -> Dict[str, int]:
        return {name: self._rank_of(mod) for name, mod in self.adapters(model).items()}

    def total_rank(self, model: Optional[torch.nn.Module] = None) -> int:
        return int(sum(self.ranks(model).values()))

    def num_tuning_parameters(self, model: Optional[torch.nn.Module] = None) -> int:
        total = 0
        for _name, module in self.adapters(model).items():
            adapter = _adapter_of(module)
            counter = getattr(adapter, "num_tuning_parameters", None)
            if callable(counter):
                try:
                    total += int(counter())
                    continue
                except Exception:
                    pass
            for param in getattr(adapter, "parameters", lambda: [])():
                total += int(param.numel())
        return total

    # ----------------------------------------------------------------------------------
    # Initial ranks
    # ----------------------------------------------------------------------------------
    def set_initial_ranks(
        self, rank: Optional[int] = None, *, model: Optional[torch.nn.Module] = None
    ) -> Dict[str, int]:
        """Force every adapter to the initial rank (Appendix A: ``r_apt = 8``).

        Only *growth* is allowed: adapters already holding a larger rank keep it.
        """
        target = int(rank if rank is not None else self.config.initial_rank)
        applied: Dict[str, int] = {}
        for name, module in self.adapters(model).items():
            adapter = _adapter_of(module)
            current = self._rank_of(module)
            if current >= target:
                applied[name] = current
                continue
            setter = getattr(adapter, "set_rank", None)
            increaser = getattr(adapter, "increase_rank", None)
            ok = False
            if callable(increaser):
                try:
                    ok = bool(increaser(target))
                except Exception:
                    ok = False
            elif callable(setter):
                try:
                    ok = bool(setter(target))
                except Exception:
                    ok = False
            if ok:
                applied[name] = target
                self._optimizer_reset_pending = True
            else:
                applied[name] = current
        self._last_ranks = applied
        return applied

    # ----------------------------------------------------------------------------------
    # Importance scoring
    # ----------------------------------------------------------------------------------
    def importance(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        salience: Any = None,
        mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """``I(H_apt)`` for every APT adapter (Section 4.3).

        Uses the activation*gradient salience of the tuning parameters summed over
        ``W_B``; ``mode="weight"`` reproduces the "w/o salience" ablation where the
        importance signal is replaced by the parameter magnitude.
        """
        mode = mode or self.config.importance_mode
        scores: Dict[str, float] = {}
        for name, module in self.adapters(model).items():
            scores[name] = adapter_importance(
                module, salience, name=name, mode=mode
            )
        return scores

    # ----------------------------------------------------------------------------------
    # Planning and applying rank growth
    # ----------------------------------------------------------------------------------
    def budget_at(self, step: Optional[int], *, budget: Any = None) -> float:
        """Evaluate the tuning budget ``Delta_t`` for ``step``."""
        if budget is not None:
            if callable(budget):
                return float(budget(step))
            return float(budget)
        schedule = self.budget_schedule
        if schedule is None:
            return 1.0
        if callable(schedule):
            try:
                return float(schedule(step))
            except TypeError:
                return float(schedule())  # type: ignore[misc]
        at = getattr(schedule, "at", None)
        if callable(at):
            return float(at(step))
        raise TypeError(f"Unsupported tuning budget schedule type: {type(schedule)!r}")

    def plan(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        salience: Any = None,
        step: Optional[int] = None,
        budget: Any = None,
        prev_budget: Optional[float] = None,
    ) -> Tuple[Dict[str, int], RankUpdateResult]:
        """Compute target ranks for the top-half adapters without touching parameters."""
        modules = self.adapters(model)
        importance = self.importance(model=model, salience=salience)
        new_budget = self.budget_at(step, budget=budget)
        old_budget = self._prev_budget if prev_budget is None else float(prev_budget)

        result = RankUpdateResult(
            step=step,
            budget=float(new_budget),
            prev_budget=float(old_budget) if old_budget is not None else float(new_budget),
            importance=dict(importance),
        )
        result.old_ranks = {name: self._rank_of(mod) for name, mod in modules.items()}

        if not modules:
            result.skipped = True
            return {}, result

        if not self.config.enabled:
            # "w/o A_T" ablation: keep the initial ranks, add no parameters.
            result.skipped = True
            result.new_ranks = dict(result.old_ranks)
            return dict(result.old_ranks), result

        if old_budget is None or float(old_budget) <= 0.0:
            # First adaptive-tuning step establishes the budget baseline.
            result.ratio = 1.0
            result.new_ranks = dict(result.old_ranks)
            result.selected = top_half_names(importance, min_keep=1)
            return dict(result.old_ranks), result

        ratio = float(new_budget) / float(old_budget)
        if self.config.max_step_ratio is not None:
            ratio = min(ratio, float(self.config.max_step_ratio))
        result.ratio = max(ratio, 1.0)

        selected = top_half_names(importance, min_keep=1)
        result.selected = list(selected)
        selected_set = set(selected)

        targets: Dict[str, int] = {}
        for name, module in modules.items():
            current = int(result.old_ranks.get(name, self._rank_of(module)))
            if name in selected_set:
                targets[name] = rank_after_budget(
                    current,
                    old_budget,
                    new_budget,
                    min_rank=self.config.min_rank,
                    max_rank=self.config.max_rank,
                )
            else:
                targets[name] = current
        result.new_ranks = dict(targets)
        return targets, result

    def apply_ranks(
        self, targets: Dict[str, int], *, model: Optional[torch.nn.Module] = None
    ) -> Tuple[List[str], int]:
        """Grow the adapters named in ``targets`` to their target ranks.

        New ``W_A`` rows are Gaussian and new ``W_B`` columns are zeros, so layer
        outputs are unchanged; returns ``(changed_names, added_parameters)``.
        """
        modules = self.adapters(model)
        changed: List[str] = []
        added = 0
        for name, target in targets.items():
            module = modules.get(name)
            if module is None:
                continue
            adapter = _adapter_of(module)
            current = self._rank_of(module)
            target = int(target)
            if target <= current:
                continue
            before = self.num_tuning_parameters_for(module)
            increaser = getattr(adapter, "increase_rank", None)
            setter = getattr(adapter, "set_rank", None)
            ok = False
            if callable(increaser):
                try:
                    ok = bool(increaser(target))
                except Exception:
                    ok = False
            elif callable(setter):
                try:
                    ok = bool(setter(target))
                except Exception:
                    ok = False
            if not ok:
                continue
            after = self.num_tuning_parameters_for(module)
            added += max(int(after - before), 0)
            changed.append(name)
        if changed:
            self._optimizer_reset_pending = True
            self._total_growth_events += 1
        return changed, added

    @staticmethod
    def num_tuning_parameters_for(module: Any) -> int:
        adapter = _adapter_of(module)
        counter = getattr(adapter, "num_tuning_parameters", None)
        if callable(counter):
            try:
                return int(counter())
            except Exception:
                pass
        total = 0
        for param in getattr(adapter, "parameters", lambda: [])():
            total += int(param.numel())
        return total

    def step(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        salience: Any = None,
        step: Optional[int] = None,
        budget: Any = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_factory: Optional[Callable[[torch.nn.Module], torch.optim.Optimizer]] = None,
        apply: bool = True,
    ) -> RankUpdateResult:
        """One adaptive-tuning step: score -> sort -> grow top-half ranks.

        Resets the optimizer afterwards when the parameter size changed
        (Section 6 / Appendix C Algorithm 1).
        """
        model_ref = model if model is not None else self.model
        targets, result = self.plan(
            model=model_ref, salience=salience, step=step, budget=budget
        )

        if not result.skipped and apply:
            changed, added = self.apply_ranks(targets, model=model_ref)
            result.changed = changed
            result.added_parameters = added

        # Track the budget baseline for the next call.
        if not result.skipped:
            self._prev_budget = float(result.budget)
        elif self._prev_budget is None and result.budget:
            self._prev_budget = float(result.budget)
        self._last_ranks = dict(result.new_ranks)

        # ----- optimizer reset after parameter-size changes ---------------------
        if result.changed and self.config.reset_optimizer:
            if optimizer_factory is not None and model_ref is not None:
                try:
                    result.optimizer = optimizer_factory(model_ref)
                    result.optimizer_reset = True
                except Exception:
                    result.optimizer = optimizer
            elif optimizer is not None:
                try:
                    result.optimizer = recreate_optimizer(
                        optimizer, trainable_parameters(model_ref) if model_ref is not None else None
                    )
                    result.optimizer_reset = True
                except Exception:
                    result.optimizer_reset = reset_optimizer_state(optimizer)
                    result.optimizer = optimizer
            if result.optimizer_reset:
                self._optimizer_reset_pending = False

        self._history.append(result)
        return result

    # Convenience alias mirroring the loop's phrasing in Appendix C.
    def maybe_grow(self, **kwargs: Any) -> RankUpdateResult:
        return self.step(**kwargs)

    # ----------------------------------------------------------------------------------
    # Bookkeeping
    # ----------------------------------------------------------------------------------
    def needs_optimizer_reset(self) -> bool:
        """Whether a parameter-size change is still pending an optimizer reset."""
        return bool(self._optimizer_reset_pending)

    def consume_optimizer_reset(self) -> bool:
        flag = bool(self._optimizer_reset_pending)
        self._optimizer_reset_pending = False
        return flag

    def mark_optimizer_reset(self, flag: bool = True) -> None:
        self._optimizer_reset_pending = bool(flag)

    @property
    def history(self) -> List[RankUpdateResult]:
        return list(self._history)

    @property
    def last_result(self) -> Optional[RankUpdateResult]:
        return self._history[-1] if self._history else None

    @property
    def growth_events(self) -> int:
        return int(self._total_growth_events)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "prev_budget": self._prev_budget,
            "last_ranks": dict(self._last_ranks),
            "optimizer_reset_pending": bool(self._optimizer_reset_pending),
            "growth_events": int(self._total_growth_events),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        state = state or {}
        self._prev_budget = state.get("prev_budget", self._prev_budget)
        self._last_ranks = dict(state.get("last_ranks", self._last_ranks) or {})
        self._optimizer_reset_pending = bool(
            state.get("optimizer_reset_pending", self._optimizer_reset_pending)
        )
        self._total_growth_events = int(state.get("growth_events", self._total_growth_events))

    def summary(self, model: Optional[torch.nn.Module] = None) -> str:
        modules = self.adapters(model)
        ranks = self.ranks(model)
        values = sorted(ranks.values())
        mean_rank = sum(values) / len(values) if values else 0.0
        return (
            f"RankController(adapters={len(modules)}, total_rank={sum(values)}, "
            f"mean_rank={mean_rank:.2f}, min_rank={min(values) if values else 0}, "
            f"max_rank={max(values) if values else 0}, "
            f"tuning_params={self.num_tuning_parameters(model)}, "
            f"growth_events={self._total_growth_events}, "
            f"enabled={self.config.enabled}, mode={self.config.importance_mode})"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RankController(initial_rank={self.config.initial_rank}, "
            f"top_fraction={self.config.top_fraction}, "
            f"min_rank={self.config.min_rank}, max_rank={self.config.max_rank}, "
            f"enabled={self.config.enabled})"
        )


# Readability aliases used across the code base / paper wording.
AdapterRankController = RankController
TuningController = RankController


# ======================================================================================
# Self test
# ======================================================================================
def _self_test() -> bool:  # pragma: no cover - executed manually
    """Sanity checks for importance scoring, top-half selection and safe growth."""
    ok = True

    # ---- toy model with two masked linears ------------------------------------------
    class Tiny(torch.nn.Module):
        def __init__(self, d_in=16, d_out=16, rank=4):
            super().__init__()
            self.proj = torch.nn.ModuleDict()
            torch.manual_seed(0)
            base = torch.nn.Linear(d_in, d_out, bias=False)
            linear = None
            if MaskedLinear is not None:
                linear = MaskedLinear(base, kind=0, out_group_size=1)
                linear.adapter.set_rank(rank)
            else:
                linear = base
            self.linear1 = linear
            base2 = torch.nn.Linear(d_in, d_out, bias=False)
            self.linear2 = MaskedLinear(base2, kind=0, out_group_size=1) if MaskedLinear else base2

    model = Tiny()
    controller = RankController(
        model,
        budget_initial=1.0,
        budget_final=1.6,
        pruning_start_step=0,
        pruning_end_step=100,
        initial_rank=4,
        seed=0,
    )

    # 1) importance dict covers all adapters
    imp = controller.importance(
        salience={"bsal": {"linear1": torch.tensor([1.0, 5.0])}}
    )
    assert set(imp.keys()) == {"linear1", "linear2"}, imp
    assert abs(imp["linear1"] - 5.0) < 1e-6, imp

    # 2) top-half selection keeps the most salient adapter
    halves = top_half_names(imp)
    assert "linear1" in halves and len(halves) == 1, (halves, imp)

    # 3) rank update formula floor(r * Delta'/Delta)
    assert rank_after_budget(8, 1.0, 1.5) == 12
    assert rank_after_budget(8, 1.0, 1.1) == 8  # floor(8.8) == 8
    assert rank_after_budget(8, 2.0, 1.0) == 8  # never shrink

    # 4) growth preserves the layer output exactly
    x = torch.randn(2, 16)
    if MaskedLinear is not None and hasattr(model.linear1, "adapter"):
        before = model.linear1(x).detach().clone()
        controller.set_initial_ranks(4)
        result = controller.step(model=model, salience=imp, step=50)
        after = model.linear1(x).detach()
        assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()
        assert controller.ranks()["linear1"] >= 4
        ok = ok and bool(result.changed or result.skipped)
    print(f"[rank_controller] self-test passed (ranks={controller.ranks()})")
    return ok


if __name__ == "__main__":  # pragma: no cover
    _self_test()
