"""Training / inference efficiency measurement for APT (Sec. 5.3, Appendix I).

The paper reports four efficiency metrics for every (model, method) pair:

* **Train. Time** -- relative training speed measured by *time to accuracy* (TTA),
  i.e. the wall-clock time spent until the model reaches 97% of the fully
  fine-tuned baseline's performance (footnote 5).  For knowledge-distillation
  methods the training time of the teacher model plus the student is counted
  ("For fair comparisons, we consider the training time of the teacher model
  plus the student for methods using knowledge distillation.").
* **Train. Mem.** -- relative training peak memory.
* **Inf. Time** -- relative inference latency; the paper measures inference
  efficiency "based on throughput (data processed per second)".
* **Inf. Mem.** -- relative inference peak memory.

All figures in Table 2 are *normalized to full fine-tuning (FT)* while the raw
numbers live in Table 11 (reproduced in :data:`TABLE11_RAW`).  Peak GPU memory
is obtained with ``torch.cuda.max_memory_allocated()`` as stated in the
Addendum ("APT Implementation").  The inference test batch size is 128 for
small models, 32 for LLaMA-7B and 4 for LLaMA-13B (Sec. 5.3).

This module is deliberately dependency-light: ``torch`` is imported lazily so
that the pure-Python bookkeeping helpers (TTA, normalization, table formatting)
can be used -- and unit-tested -- in a bare environment.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # constants
    "TTA_FRACTION",
    "SMALL_MODEL_INFERENCE_BATCH_SIZE",
    "LLAMA_7B_INFERENCE_BATCH_SIZE",
    "LLAMA_13B_INFERENCE_BATCH_SIZE",
    "DEFAULT_SEQUENCE_LENGTH",
    "TABLE11_RAW",
    "TABLE2_RELATIVE",
    "INFERENCE_BATCH_SIZES",
    "METRIC_KEYS",
    "METHOD_ORDER",
    # low level measurement
    "cuda_available",
    "synchronize",
    "reset_peak_memory",
    "peak_memory_mb",
    "peak_memory_bytes",
    "current_memory_mb",
    "free_memory_mb",
    "PeakMemoryTracker",
    "Timer",
    "measure_peak_memory",
    "measure_runtime",
    # TTA
    "TimeToAccuracy",
    "TimeToAccuracyTracker",
    "time_to_accuracy",
    "combine_distill_time",
    # training efficiency
    "TrainingEfficiencyTracker",
    "measure_training_efficiency",
    # inference efficiency
    "inference_batch_size_for",
    "make_dummy_batch",
    "measure_inference",
    "measure_inference_throughput",
    "measure_inference_latency",
    "measure_inference_memory",
    "benchmark_inference",
    # model size
    "parameter_count",
    "model_size_mb",
    "model_size_breakdown",
    # normalization / reporting
    "EfficiencyResult",
    "relative_metric",
    "normalize_efficiency",
    "relative_from_table11",
    "efficiency_row",
    "format_efficiency",
    "speedup",
    "efficiency_summary",
]

# --------------------------------------------------------------------------- #
# Constants (Sec. 5.3 + Appendix I)
# --------------------------------------------------------------------------- #

#: Fraction of the reference (FT) score that defines the TTA target.
TTA_FRACTION: float = 0.97

#: Inference test batch sizes from Sec. 5.3.
SMALL_MODEL_INFERENCE_BATCH_SIZE: int = 128
LLAMA_7B_INFERENCE_BATCH_SIZE: int = 32
LLAMA_13B_INFERENCE_BATCH_SIZE: int = 4

#: Default input sequence length used for throughput measurement.
DEFAULT_SEQUENCE_LENGTH: int = 128

INFERENCE_BATCH_SIZES: Dict[str, int] = {
    "small": SMALL_MODEL_INFERENCE_BATCH_SIZE,
    "llama-7b": LLAMA_7B_INFERENCE_BATCH_SIZE,
    "llama-13b": LLAMA_13B_INFERENCE_BATCH_SIZE,
}

METRIC_KEYS: Tuple[str, ...] = ("train_time", "train_mem", "inf_time", "inf_mem")

#: Ordering used in Table 2 / Table 11.
METHOD_ORDER: Tuple[str, ...] = (
    "FT",
    "LoRA",
    "LoRA+Prune",
    "Prune+Distill",
    "LoRA+Prune+Distill",
    "APT",
)

#: Table 11 -- raw efficiency metrics (Appendix I).  Training times in seconds,
#: inference times in milliseconds, all memory footprints in MB.
TABLE11_RAW: Dict[str, Dict[str, Dict[str, float]]] = {
    "roberta-base": {
        "FT": {
            "sparsity": 0.0,
            "train_time_s": 127.0,
            "train_mem_mb": 2696.0,
            "inf_time_ms": 220.8,
            "inf_mem_mb": 1157.0,
        },
        "LoRA": {
            "sparsity": 0.0,
            "train_time_s": 2714.0,
            "train_mem_mb": 1630.0,
            "inf_time_ms": 181.8,
            "inf_mem_mb": 1157.0,
        },
        "LoRA+Prune": {
            "sparsity": 0.60,
            "train_time_s": 6513.0,
            "train_mem_mb": 1630.0,
            "inf_time_ms": 84.0,
            "inf_mem_mb": 869.0,
        },
        "Prune+Distill": {
            "sparsity": 0.60,
            "train_time_s": 1899.0,
            "train_mem_mb": 4544.0,
            "inf_time_ms": 85.2,
            "inf_mem_mb": 917.0,
        },
        "LoRA+Prune+Distill": {
            "sparsity": 0.60,
            "train_time_s": 8299.0,
            "train_mem_mb": 3813.0,
            "inf_time_ms": 87.0,
            "inf_mem_mb": 952.0,
        },
        "APT": {
            "sparsity": 0.60,
            "train_time_s": 752.0,
            "train_mem_mb": 1890.0,
            "inf_time_ms": 91.3,
            "inf_mem_mb": 904.0,
        },
    },
    "t5-base": {
        "FT": {
            "sparsity": 0.0,
            "train_time_s": 366.0,
            "train_mem_mb": 7217.0,
            "inf_time_ms": 248.1,
            "inf_mem_mb": 2347.0,
        },
        "LoRA": {
            "sparsity": 0.0,
            "train_time_s": 935.0,
            "train_mem_mb": 4476.0,
            "inf_time_ms": 254.2,
            "inf_mem_mb": 2347.0,
        },
        "LoRA+Prune": {
            "sparsity": 0.60,
            "train_time_s": 14417.0,
            "train_mem_mb": 4476.0,
            "inf_time_ms": 116.8,
            "inf_mem_mb": 1724.0,
        },
        "APT": {
            "sparsity": 0.60,
            "train_time_s": 1774.0,
            "train_mem_mb": 5332.0,
            "inf_time_ms": 185.0,
            "inf_mem_mb": 1913.0,
        },
    },
}

#: Table 2 -- relative efficiency numbers as reported in the paper.  Kept for
#: sanity checking the normalization pipeline against the published values.
TABLE2_RELATIVE: Dict[str, Dict[str, Dict[str, float]]] = {
    "roberta-base": {
        "FT": {"train_time": 100.0, "train_mem": 100.0, "inf_time": 100.0, "inf_mem": 100.0},
        "LoRA": {"train_time": 2137.0, "train_mem": 60.5, "inf_time": 100.0, "inf_mem": 100.0},
        "LoRA+Prune": {"train_time": 5128.3, "train_mem": 60.5, "inf_time": 38.0, "inf_mem": 75.1},
        "Prune+Distill": {"train_time": 1495.3, "train_mem": 168.5, "inf_time": 38.6, "inf_mem": 79.2},
        "LoRA+Prune+Distill": {"train_time": 6534.6, "train_mem": 141.4, "inf_time": 39.4, "inf_mem": 82.3},
        "APT": {"train_time": 592.1, "train_mem": 70.1, "inf_time": 41.3, "inf_mem": 78.1},
    },
    "t5-base": {
        "FT": {"train_time": 100.0, "train_mem": 100.0, "inf_time": 100.0, "inf_mem": 100.0},
        "LoRA": {"train_time": 255.5, "train_mem": 62.0, "inf_time": 100.0, "inf_mem": 100.0},
        "LoRA+Prune": {"train_time": 4523.5, "train_mem": 62.0, "inf_time": 47.1, "inf_mem": 73.4},
        "APT": {"train_time": 484.7, "train_mem": 73.9, "inf_time": 74.6, "inf_mem": 81.5},
    },
}

_METRIC_ALIASES: Dict[str, str] = {
    "train_time": "train_time",
    "train_time_s": "train_time",
    "train_seconds": "train_time",
    "tta": "train_time",
    "tta_seconds": "train_time",
    "time": "train_time",
    "train_mem": "train_mem",
    "train_mem_mb": "train_mem",
    "train_peak_mem": "train_mem",
    "train_peak_mem_mb": "train_mem",
    "train_peak_memory_mb": "train_mem",
    "inf_time": "inf_time",
    "inf_time_ms": "inf_time",
    "inf_latency": "inf_time",
    "latency": "inf_time",
    "latency_ms": "inf_time",
    "inf_mem": "inf_mem",
    "inf_mem_mb": "inf_mem",
    "inf_peak_mem": "inf_mem",
    "inf_peak_mem_mb": "inf_mem",
    "peak_memory_mb": "inf_mem",
}


# --------------------------------------------------------------------------- #
# Low-level measurement primitives
# --------------------------------------------------------------------------- #

def _torch():
    """Import torch lazily; return ``None`` when unavailable."""
    try:
        import torch  # noqa: WPS433

        return torch
    except Exception:  # pragma: no cover
        return None


def cuda_available() -> bool:
    """True when a CUDA device with a working PyTorch backend is present."""
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def synchronize(device: Any = None) -> None:
    """Block until all kernels on ``device`` have finished (no-op on CPU)."""
    torch = _torch()
    if torch is None or not cuda_available():
        return
    try:
        if device is not None and getattr(device, "type", None) == "cpu":
            return
        torch.cuda.synchronize(device)
    except Exception:  # pragma: no cover
        pass


def reset_peak_memory(device: Any = None) -> None:
    """Reset the CUDA peak-memory statistics (``max_memory_allocated``)."""
    torch = _torch()
    if torch is None or not cuda_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # pragma: no cover
        pass


def peak_memory_bytes(device: Any = None) -> int:
    """``torch.cuda.max_memory_allocated()`` for ``device`` (0 on CPU)."""
    torch = _torch()
    if torch is None or not cuda_available():
        return 0
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:  # pragma: no cover
        return 0


def peak_memory_mb(device: Any = None) -> float:
    """Peak allocated GPU memory in MB (Appendix I unit)."""
    return peak_memory_bytes(device) / (1024.0 ** 2)


def current_memory_mb(device: Any = None) -> float:
    """Currently allocated GPU memory in MB."""
    torch = _torch()
    if torch is None or not cuda_available():
        return 0.0
    try:
        return float(torch.cuda.memory_allocated(device)) / (1024.0 ** 2)
    except Exception:  # pragma: no cover
        return 0.0


def free_memory_mb(device: Any = None) -> float:
    """Free GPU memory in MB."""
    torch = _torch()
    if torch is None or not cuda_available():
        return 0.0
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return float(free) / (1024.0 ** 2)
    except Exception:  # pragma: no cover
        return 0.0


class PeakMemoryTracker(contextlib.AbstractContextManager):
    """Context manager returning the CUDA peak memory (MB) consumed inside.

    Usage::

        with PeakMemoryTracker() as mem:
            train_one_step()
        print(mem.peak_mb)
    """

    def __init__(self, device: Any = None, reset: bool = True):
        self.device = device
        self.reset = reset
        self.start_mb = 0.0
        self.peak_mb = 0.0
        self.peak_bytes = 0

    def __enter__(self) -> "PeakMemoryTracker":
        if self.reset:
            reset_peak_memory(self.device)
        self.start_mb = current_memory_mb(self.device)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        synchronize(self.device)
        self.peak_bytes = peak_memory_bytes(self.device)
        self.peak_mb = self.peak_bytes / (1024.0 ** 2)
        return False

    def as_dict(self) -> Dict[str, float]:
        return {
            "peak_memory_mb": self.peak_mb,
            "peak_memory_bytes": float(self.peak_bytes),
            "start_memory_mb": self.start_mb,
        }


class Timer:
    """Wall-clock timer with CUDA synchronization."""

    def __init__(self, device: Any = None, sync: bool = True):
        self.device = device
        self.sync = sync
        self._start = 0.0
        self.elapsed = 0.0

    def start(self) -> "Timer":
        if self.sync:
            synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self.sync:
            synchronize(self.device)
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False


def measure_peak_memory(fn: Callable[[], Any], device: Any = None,
                        reset: bool = True) -> Tuple[Any, float]:
    """Run ``fn()`` and return ``(result, peak_memory_mb)``."""
    with PeakMemoryTracker(device=device, reset=reset) as tracker:
        result = fn()
    return result, tracker.peak_mb


def measure_runtime(fn: Callable[[], Any], device: Any = None,
                    sync: bool = True) -> Tuple[Any, float]:
    """Run ``fn()`` and return ``(result, elapsed_seconds)``."""
    with Timer(device=device, sync=sync) as timer:
        result = fn()
    return result, timer.elapsed


# --------------------------------------------------------------------------- #
# Time to accuracy (TTA)
# --------------------------------------------------------------------------- #

class TimeToAccuracy:
    """Track time-to-accuracy to a fraction of a reference (FT) score.

    A history of ``(elapsed_seconds, metric_value)`` observations is kept; the
    reported TTA is the first time at which the metric reaches
    ``fraction * reference`` (linearly interpolated between the two bracketing
    observations).  If the target is never reached, :attr:`reached` is False and
    :attr:`value` holds the interpolated/extrapolated time at the last update.
    """

    def __init__(self, reference: float, fraction: float = TTA_FRACTION,
                 higher_is_better: bool = True):
        self.reference = float(reference)
        self.fraction = float(fraction)
        self.higher_is_better = bool(higher_is_better)
        self.history: List[Tuple[float, float]] = []
        self.value: Optional[float] = None
        self.reached: bool = False

    @property
    def target(self) -> float:
        return self.fraction * self.reference

    def _satisfies(self, metric: float) -> bool:
        if self.higher_is_better:
            return metric >= self.target
        return metric <= self.target

    def update(self, elapsed_seconds: float, metric_value: float) -> bool:
        """Record an observation; return True once the target was reached."""
        self.history.append((float(elapsed_seconds), float(metric_value)))
        if self.reached:
            return True
        if not self._satisfies(metric_value):
            return False
        # Interpolate between the previous and the current observation.
        if len(self.history) == 1:
            self.value = float(elapsed_seconds)
        else:
            t0, m0 = self.history[-2]
            t1, m1 = self.history[-1]
            denom = (m1 - m0)
            if abs(denom) < 1e-12:
                self.value = float(elapsed_seconds)
            else:
                frac = (self.target - m0) / denom
                frac = min(max(frac, 0.0), 1.0)
                self.value = t0 + frac * (t1 - t0)
        self.reached = True
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "fraction": self.fraction,
            "target": self.target,
            "tta_seconds": self.value,
            "reached": self.reached,
            "n_observations": len(self.history),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        val = "n/a" if self.value is None else "{:.1f}s".format(self.value)
        return "TimeToAccuracy(target={:.2f}, tta={}, reached={})".format(
            self.target, val, self.reached
        )


#: Alias kept for parity with :mod:`apt.training`.
TimeToAccuracyTracker = TimeToAccuracy


def time_to_accuracy(history: Sequence[Tuple[float, float]], reference: float,
                     fraction: float = TTA_FRACTION,
                     higher_is_better: bool = True) -> Optional[float]:
    """Compute the TTA (seconds) from a ``[(time, metric), ...]`` history."""
    tracker = TimeToAccuracy(reference, fraction, higher_is_better)
    for t, m in history:
        tracker.update(t, m)
    return tracker.value


def combine_distill_time(teacher_seconds: float, student_seconds: float) -> float:
    """Training time for KD methods counts teacher **plus** student (Sec. 5.3)."""
    return float(teacher_seconds) + float(student_seconds)


# --------------------------------------------------------------------------- #
# Training efficiency
# --------------------------------------------------------------------------- #

class TrainingEfficiencyTracker:
    """Accumulate training wall-clock time and peak memory across a run.

    Example::

        tracker = TrainingEfficiencyTracker(reference=94.8)
        tracker.start()
        for step, batch in enumerate(loader):
            train_step(batch)
            if step % eval_every == 0:
                tracker.record_eval(accuracy)
        tracker.stop()
        print(tracker.as_dict())
    """

    def __init__(self, reference: Optional[float] = None,
                 fraction: float = TTA_FRACTION,
                 higher_is_better: bool = True,
                 device: Any = None,
                 include_teacher: bool = False):
        self.reference = None if reference is None else float(reference)
        self.fraction = float(fraction)
        self.higher_is_better = bool(higher_is_better)
        self.device = device
        self.include_teacher = bool(include_teacher)
        self.tta = (None if self.reference is None
                    else TimeToAccuracy(self.reference, self.fraction,
                                        self.higher_is_better))
        self._timer = Timer(device=device)
        self._tracker = PeakMemoryTracker(device=device)
        self.total_seconds: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.teacher_seconds: float = 0.0
        self.steps: int = 0
        self.eval_history: List[Tuple[float, float]] = []

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> "TrainingEfficiencyTracker":
        self._tracker.__enter__()
        self._timer.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self.total_seconds = self._timer.stop()
        self._tracker.__exit__(None, None, None)
        self.peak_memory_mb = self._tracker.peak_mb
        return self.as_dict()

    def add_teacher_time(self, seconds: float) -> None:
        """Account for teacher training time (KD methods, Sec. 5.3)."""
        self.teacher_seconds += float(seconds)

    #: Alias, kept for readability at call sites.
    add_time = add_teacher_time

    def step(self) -> None:
        self.steps += 1

    # -- accuracy tracking ------------------------------------------------ #
    def record_eval(self, metric: float,
                    elapsed: Optional[float] = None) -> bool:
        """Record an evaluation result; returns True when TTA was reached."""
        if elapsed is None:
            synchronize(self.device)
            elapsed = time.perf_counter() - self._timer._start
        self.eval_history.append((float(elapsed), float(metric)))
        if self.tta is None:
            return False
        return self.tta.update(elapsed, metric)

    # -- reporting -------------------------------------------------------- #
    @property
    def train_time_s(self) -> float:
        """Total training seconds, counting the teacher when requested."""
        return self.total_seconds + self.teacher_seconds

    @property
    def tta_seconds(self) -> Optional[float]:
        if self.tta is None or self.tta.value is None:
            return None
        # The teacher runs first for KD methods, so it is added to the TTA.
        return self.tta.value + self.teacher_seconds

    def as_dict(self) -> Dict[str, Any]:
        return {
            "train_time_s": self.train_time_s,
            "tta_seconds": self.tta_seconds,
            "train_peak_mem_mb": self.peak_memory_mb,
            "steps": self.steps,
            "teacher_seconds": self.teacher_seconds,
            "tta_reached": bool(self.tta.reached) if self.tta else None,
            "eval_history": list(self.eval_history),
        }


def measure_training_efficiency(
    train_fn: Callable[..., Any],
    reference: Optional[float] = None,
    *,
    eval_fn: Optional[Callable[[], float]] = None,
    eval_every: int = 0,
    device: Any = None,
    teacher_seconds: float = 0.0,
    fraction: float = TTA_FRACTION,
    protocol: str = "tta",
) -> Dict[str, Any]:
    """Measure training time / peak memory of a training routine.

    Parameters
    ----------
    train_fn:
        Callable performing the training run.  When ``eval_fn``/``eval_every``
        are given it is first tried with a keyword ``on_step`` callback; if it
        does not accept one it is called without arguments instead (TTA then
        falls back to the wall-clock time).
    reference:
        FT reference score, needed for the TTA protocol.
    eval_fn:
        Callable returning the current dev metric.
    protocol:
        ``"tta"`` (time to 97% accuracy, default) or ``"wall"`` (total time).
    """
    tracker: Optional[TimeToAccuracy] = None

    with PeakMemoryTracker(device=device) as mem:
        with Timer(device=device) as timer:
            if eval_fn is not None and eval_every and reference is not None:
                tracker = TimeToAccuracy(reference, fraction)
                counters = {"n": 0}

                def on_step(*_a, **_kw):
                    counters["n"] += 1
                    if counters["n"] % max(1, int(eval_every)) == 0:
                        synchronize(device)
                        tracker.update(time.perf_counter() - timer._start,
                                       eval_fn())

                try:
                    train_fn(on_step=on_step)
                except TypeError:
                    train_fn()
            else:
                train_fn()

    out: Dict[str, Any] = {
        "train_time_s": timer.elapsed + float(teacher_seconds),
        "train_peak_mem_mb": mem.peak_mb,
    }
    if tracker is not None:
        tta = tracker.value
        out["tta_seconds"] = None if tta is None else tta + float(teacher_seconds)
        out["tta_reached"] = tracker.reached
    if protocol == "wall":
        out["tta_seconds"] = out["train_time_s"]
    return out


# --------------------------------------------------------------------------- #
# Inference efficiency
# --------------------------------------------------------------------------- #

def inference_batch_size_for(model: Any = None, *,
                             n_parameters: Optional[int] = None,
                             model_name: Optional[str] = None) -> int:
    """Inference batch size per Sec. 5.3 (128 small, 32 for 7B, 4 for 13B)."""
    name = (model_name or "").lower()
    if not name and model is not None:
        name = str(getattr(getattr(model, "config", None),
                           "_name_or_path", "") or "").lower()
    if "13b" in name:
        return LLAMA_13B_INFERENCE_BATCH_SIZE
    if "7b" in name:
        return LLAMA_7B_INFERENCE_BATCH_SIZE
    if n_parameters is None and model is not None:
        n_parameters = parameter_count(model)
    if n_parameters is None:
        return SMALL_MODEL_INFERENCE_BATCH_SIZE
    if n_parameters >= 11_000_000_000:  # ~13B
        return LLAMA_13B_INFERENCE_BATCH_SIZE
    if n_parameters >= 6_000_000_000:  # ~7B
        return LLAMA_7B_INFERENCE_BATCH_SIZE
    return SMALL_MODEL_INFERENCE_BATCH_SIZE


def make_dummy_batch(batch_size: int = SMALL_MODEL_INFERENCE_BATCH_SIZE,
                     seq_length: int = DEFAULT_SEQUENCE_LENGTH,
                     vocab_size: int = 50265,
                     device: Any = None,
                     kind: str = "encoder") -> Dict[str, Any]:
    """Build a synthetic batch for throughput measurement.

    ``kind`` selects a plain encoder batch or a seq2seq batch (T5-style) with
    decoder inputs.
    """
    torch = _torch()
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required to build dummy batches")
    if device is None:
        device = "cuda" if cuda_available() else "cpu"
    ids = torch.randint(0, max(int(vocab_size), 8),
                        (int(batch_size), int(seq_length)),
                        device=device, dtype=torch.long)
    attention = torch.ones_like(ids)
    batch: Dict[str, Any] = {"input_ids": ids, "attention_mask": attention}
    if str(kind).lower() in ("seq2seq", "t5", "cnndm"):
        batch["decoder_input_ids"] = torch.randint(
            0, max(int(vocab_size), 8), (int(batch_size), 8),
            device=device, dtype=torch.long,
        )
        batch["decoder_attention_mask"] = torch.ones_like(
            batch["decoder_input_ids"]
        )
    return batch


def _model_device(model: Any) -> Any:
    try:
        for p in model.parameters():
            return p.device
    except Exception:  # pragma: no cover
        pass
    return "cuda" if cuda_available() else "cpu"


def _infer_batch_kind(model: Any) -> str:
    cfg = getattr(model, "config", None)
    name = (str(getattr(cfg, "_name_or_path", "") or "")
            + " " + str(getattr(cfg, "model_type", "") or "")).lower()
    return "seq2seq" if ("t5" in name or "bart" in name) else "encoder"


def _forward_once(model: Any, batch: Dict[str, Any]) -> None:
    torch = _torch()
    with torch.no_grad():
        out = model(**batch)
    # Touch the logits so nothing gets optimized away.
    try:
        logits = out.logits if hasattr(out, "logits") else out[0]
        _ = float(logits.float().sum().item())
    except Exception:  # pragma: no cover
        pass


def measure_inference(model: Any,
                      batch_size: int = SMALL_MODEL_INFERENCE_BATCH_SIZE,
                      seq_length: int = DEFAULT_SEQUENCE_LENGTH,
                      vocab_size: Optional[int] = None,
                      repeats: int = 5,
                      warmup: int = 1,
                      device: Any = None,
                      batch: Optional[Dict[str, Any]] = None,
                      measure_memory: bool = True,
                      auto_batch_size: bool = False) -> Dict[str, float]:
    """Benchmark inference latency / throughput / peak memory (Sec. 5.3).

    Returns ``latency_ms`` (mean per forward call), ``throughput_samples_per_sec``
    (data processed per second -- the paper's inference-efficiency definition)
    and ``peak_memory_mb`` (from ``torch.cuda.max_memory_allocated()``).
    """
    torch = _torch()
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for inference benchmarking")

    if device is None:
        device = _model_device(model)
    if auto_batch_size:
        batch_size = inference_batch_size_for(model)

    if vocab_size is None:
        vocab_size = int(getattr(getattr(model, "config", None),
                                 "vocab_size", 50265) or 50265)

    if batch is None:
        batch = make_dummy_batch(batch_size=batch_size, seq_length=seq_length,
                                 vocab_size=vocab_size, device=device,
                                 kind=_infer_batch_kind(model))

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    # Warm-up allocates caches / loads kernels.
    for _ in range(max(int(warmup), 0)):
        _forward_once(model, batch)
    synchronize(device)

    if measure_memory:
        reset_peak_memory(device)

    total_time = 0.0
    n = max(int(repeats), 1)
    use_events = cuda_available() and hasattr(torch.cuda, "Event")
    for _ in range(n):
        if use_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            _forward_once(model, batch)
            end_ev.record()
            torch.cuda.synchronize()
            total_time += start_ev.elapsed_time(end_ev) / 1000.0
        else:
            t0 = time.perf_counter()
            _forward_once(model, batch)
            total_time += time.perf_counter() - t0

    synchronize(device)
    latency_s = total_time / n
    peak = peak_memory_mb(device) if measure_memory else 0.0

    if was_training and hasattr(model, "train"):
        model.train()

    return {
        "batch_size": float(batch_size),
        "seq_length": float(seq_length),
        "latency_ms": latency_s * 1000.0,
        "latency_s": latency_s,
        "throughput_samples_per_sec": float(batch_size) / max(latency_s, 1e-12),
        "peak_memory_mb": peak,
    }


#: Backwards-compatible name (also used by ``apt.training``).
measure_inference_throughput = measure_inference


def measure_inference_latency(model: Any, **kwargs: Any) -> float:
    """Latency in milliseconds for a single forward pass."""
    return measure_inference(model, **kwargs)["latency_ms"]


def measure_inference_memory(model: Any, **kwargs: Any) -> float:
    """Peak inference memory in MB."""
    return measure_inference(model, **kwargs)["peak_memory_mb"]


def benchmark_inference(model: Any, *args: Any, **kwargs: Any) -> Dict[str, float]:
    """Alias of :func:`measure_inference`."""
    return measure_inference(model, *args, **kwargs)


# --------------------------------------------------------------------------- #
# Model size helpers
# --------------------------------------------------------------------------- #

def parameter_count(model: Any, trainable_only: bool = False) -> int:
    """Number of parameters (optionally trainable only)."""
    try:
        if trainable_only:
            return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # pragma: no cover
        return 0


def model_size_mb(model: Any, dtype_bytes: Optional[int] = None) -> float:
    """In-memory parameter footprint in MB."""
    total_bytes = 0
    for p in model.parameters():
        try:
            total_bytes += int(p.numel()) * int(p.element_size())
        except Exception:  # pragma: no cover
            total_bytes += int(p.numel()) * int(dtype_bytes or 4)
    return total_bytes / (1024.0 ** 2)


def model_size_breakdown(model: Any) -> Dict[str, float]:
    """Parameter count / size for the whole model and its tunable subset."""
    total = parameter_count(model)
    trainable = parameter_count(model, trainable_only=True)
    return {
        "parameters": float(total),
        "trainable_parameters": float(trainable),
        "tunable_fraction": (float(trainable) / total) if total else 0.0,
        "size_mb": model_size_mb(model),
    }


# --------------------------------------------------------------------------- #
# Normalization / reporting
# --------------------------------------------------------------------------- #

def relative_metric(value: Optional[float],
                    reference: Optional[float]) -> Optional[float]:
    """``value / reference * 100`` (the paper's normalization to FT)."""
    if value is None or reference in (None, 0):
        return None
    return float(value) / float(reference) * 100.0


def speedup(reference_time: float, method_time: float) -> float:
    """Speedup of ``method_time`` over ``reference_time``."""
    if method_time in (None, 0):
        return float("nan")
    return float(reference_time) / float(method_time)


def _canonical_metric(key: str) -> str:
    return _METRIC_ALIASES.get(str(key).lower(), str(key).lower())


def _first(data: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class EfficiencyResult:
    """Raw + FT-normalized efficiency numbers for one (model, method) pair."""

    model: str = ""
    method: str = ""
    sparsity: float = 0.0
    train_time_s: Optional[float] = None
    train_peak_mem_mb: Optional[float] = None
    inf_time_ms: Optional[float] = None
    inf_mem_mb: Optional[float] = None
    inf_throughput: Optional[float] = None
    reference: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def relative(self,
                 reference: Optional[Dict[str, float]] = None
                 ) -> Dict[str, Optional[float]]:
        """Relative-to-FT values (percent) for the four Table 2 metrics."""
        ref = dict(self.reference)
        if reference:
            ref.update({_canonical_metric(k): v for k, v in reference.items()})
        return {
            "train_time": relative_metric(self.train_time_s, ref.get("train_time")),
            "train_mem": relative_metric(self.train_peak_mem_mb, ref.get("train_mem")),
            "inf_time": relative_metric(self.inf_time_ms, ref.get("inf_time")),
            "inf_mem": relative_metric(self.inf_mem_mb, ref.get("inf_mem")),
        }

    def as_dict(self, reference: Optional[Dict[str, float]] = None,
                relative: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model": self.model,
            "method": self.method,
            "sparsity": self.sparsity,
            "train_time_s": self.train_time_s,
            "train_peak_mem_mb": self.train_peak_mem_mb,
            "inf_time_ms": self.inf_time_ms,
            "inf_mem_mb": self.inf_mem_mb,
            "inf_throughput": self.inf_throughput,
        }
        if relative:
            out["relative"] = self.relative(reference)
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, model: str = "",
                  method: str = "", sparsity: float = 0.0) -> "EfficiencyResult":
        """Build from a (possibly aliased) measurement dictionary."""
        ref = data.get("reference", {}) or {}
        return cls(
            model=data.get("model", model),
            method=data.get("method", method),
            sparsity=float(data.get("sparsity", sparsity) or 0.0),
            train_time_s=_first(data, "train_time_s", "tta_seconds", "train_time"),
            train_peak_mem_mb=_first(data, "train_peak_mem_mb", "train_mem_mb",
                                     "train_mem"),
            inf_time_ms=_first(data, "inf_time_ms", "inf_time", "latency_ms"),
            inf_mem_mb=_first(data, "inf_mem_mb", "inf_mem", "peak_memory_mb"),
            inf_throughput=_first(data, "inf_throughput",
                                  "throughput_samples_per_sec"),
            reference={_canonical_metric(k): float(v) for k, v in ref.items()},
        )


def normalize_efficiency(raw: Dict[str, Any],
                         reference: Dict[str, Any]
                         ) -> Dict[str, Optional[float]]:
    """Normalize raw efficiency values to an FT reference (Table 2 protocol).

    ``raw`` / ``reference`` accept any alias in :data:`_METRIC_ALIASES`
    (e.g. ``train_time_s``, ``train_peak_mem_mb``, ``inf_time_ms``,
    ``inf_mem_mb``).
    """
    raw_c = {_canonical_metric(k): v for k, v in raw.items()}
    ref_c = {_canonical_metric(k): v for k, v in reference.items()}
    return {key: relative_metric(raw_c.get(key), ref_c.get(key))
            for key in METRIC_KEYS}


def relative_from_table11(model: str, method: str) -> Dict[str, Optional[float]]:
    """Table 2 relative values derived from the Table 11 raw numbers."""
    if model not in TABLE11_RAW:
        raise KeyError("unknown model {!r}; expected one of {}".format(
            model, sorted(TABLE11_RAW)))
    table = TABLE11_RAW[model]
    if method not in table:
        raise KeyError("unknown method {!r} for {!r}".format(method, model))
    return normalize_efficiency(table[method], table["FT"])


def efficiency_row(model: str, method: str) -> Dict[str, Optional[float]]:
    """Relative metrics for one row of Table 2 (from the raw table)."""
    return relative_from_table11(model, method)


def format_efficiency(metrics: Dict[str, Optional[float]], digits: int = 1,
                      as_percent: bool = True) -> str:
    """Format a relative efficiency dict as ``"592.1% 70.1% 41.3% 78.1%"``."""
    parts: List[str] = []
    for key in METRIC_KEYS:
        val = metrics.get(key)
        if val is None:
            parts.append("-")
        elif as_percent:
            parts.append("{:.{d}f}%".format(float(val), d=digits))
        else:
            parts.append("{:.{d}f}".format(float(val), d=digits))
    return " ".join(parts)


def efficiency_summary(results: Iterable[EfficiencyResult],
                       reference: Optional[Dict[str, float]] = None,
                       digits: int = 1) -> str:
    """Render a small markdown table of efficiency results."""
    lines = [
        "| model | method | sparsity | train_time | train_mem | inf_time | inf_mem |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        rel = r.relative(reference)
        cells = " | ".join(
            "-" if rel[k] is None else "{:.{d}f}%".format(float(rel[k]), d=digits)
            for k in METRIC_KEYS
        )
        lines.append("| {} | {} | {:.0%} | {} |".format(
            r.model or "-", r.method or "-", r.sparsity or 0.0, cells))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> bool:  # pragma: no cover - run manually
    ok = True

    # --- TTA target & interpolation ------------------------------------- #
    tta = TimeToAccuracy(reference=94.8, fraction=0.97)
    assert abs(tta.target - 91.956) < 1e-6, tta.target
    tta.update(10.0, 80.0)
    assert not tta.reached
    tta.update(20.0, 95.0)
    assert tta.reached
    expected = 10.0 + (91.956 - 80.0) / (95.0 - 80.0) * 10.0
    assert abs(tta.value - expected) < 1e-6, (tta.value, expected)

    # --- normalization matches Table 11 -> Table 2 ---------------------- #
    checks = [
        ("roberta-base", "APT", 752.0, 127.0),
        ("roberta-base", "LoRA+Prune", 6513.0, 127.0),
        ("t5-base", "APT", 1774.0, 366.0),
    ]
    for model, method, raw_t, ft_t in checks:
        rel = relative_from_table11(model, method)
        assert abs(rel["train_time"] - raw_t / ft_t * 100.0) < 1e-6, \
            (model, method, rel)

    # Spot-check against the published Table 2 percentages.
    apt = relative_from_table11("roberta-base", "APT")
    assert abs(apt["train_time"] - 592.1) < 0.2, apt
    assert abs(apt["train_mem"] - 70.1) < 0.2, apt
    assert abs(apt["inf_time"] - 41.3) < 0.2, apt
    assert abs(apt["inf_mem"] - 78.1) < 0.2, apt
    lora = relative_from_table11("roberta-base", "LoRA")
    assert abs(lora["train_time"] - 2137.0) < 0.2, lora
    assert abs(lora["train_mem"] - 60.5) < 0.2, lora
    t5_apt = relative_from_table11("t5-base", "APT")
    assert abs(t5_apt["inf_mem"] - 81.5) < 0.2, t5_apt

    # --- aliases --------------------------------------------------------- #
    norm = normalize_efficiency(
        {"train_time_s": 752.0, "train_peak_mem_mb": 1890.0,
         "inf_time_ms": 91.3, "inf_mem_mb": 904.0},
        {"train_time_s": 127.0, "train_peak_mem_mb": 2696.0,
         "inf_time_ms": 220.8, "inf_mem_mb": 1157.0},
    )
    assert abs(norm["train_time"] - 592.1) < 0.2, norm

    # --- KD training time counts teacher + student ---------------------- #
    assert abs(combine_distill_time(100.0, 250.0) - 350.0) < 1e-9

    # --- inference batch size rule -------------------------------------- #
    assert inference_batch_size_for(model_name="roberta-base") == 128
    assert inference_batch_size_for(model_name="meta-llama/Llama-2-7b-hf") == 32
    assert inference_batch_size_for(model_name="meta-llama/Llama-2-13b-hf") == 4
    assert inference_batch_size_for(n_parameters=125_000_000) == 128

    # --- formatting ------------------------------------------------------ #
    row = format_efficiency(relative_from_table11("roberta-base", "APT"))
    assert row == "592.1% 70.1% 41.3% 78.1%", row

    # --- EfficiencyResult round trip ------------------------------------ #
    raw = dict(TABLE11_RAW["t5-base"]["APT"])
    raw["reference"] = dict(TABLE11_RAW["t5-base"]["FT"])
    er = EfficiencyResult.from_dict(raw, model="t5-base", method="APT")
    rel = er.relative()
    assert abs(rel["inf_mem"] - 81.5) < 0.3, rel

    # --- torch-dependent paths (skipped when torch is unavailable) ------ #
    torch = _torch()
    if torch is not None:
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        res = measure_inference(model, batch_size=8, seq_length=4, repeats=1,
                                warmup=0, measure_memory=False)
        assert res["throughput_samples_per_sec"] > 0, res
        assert res["latency_ms"] > 0, res
        assert parameter_count(model) == 16 * 16 + 16 + 16 * 4 + 4
        assert model_size_mb(model) > 0

        tr = TrainingEfficiencyTracker(reference=100.0)
        tr.start()
        tr.record_eval(50.0)
        tr.record_eval(99.0)
        d = tr.stop()
        assert d["tta_seconds"] is not None and d["tta_seconds"] >= 0, d
        assert d["train_peak_mem_mb"] >= 0

    assert ok
    print("[apt.eval.efficiency] self-test passed"
          + ("" if cuda_available() else " (CPU mode: GPU memory checks skipped)"))
    return ok


if __name__ == "__main__":  # pragma: no cover
    _self_test()
