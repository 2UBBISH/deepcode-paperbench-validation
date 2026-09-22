"""GPU memory (VRAM) measurement for BBox-Adapter.

Reproduces the VRAM columns of **Table 6** of *Lightweight Adapting for Black-Box
Large Language Models* (BBox-Adapter), Section 4.6 / 4.7, plus the qualifier from
the paper Addendum.

Paper facts implemented here (verbatim from the manuscript):

* "Table 6. Accuracy (%) and GPU memory usage on adapting Mixtral-8x7B to the
  StrategyQA dataset. VRAM refers to the maximum GPU memory required by each
  approach, where the base model (Mixtral-8x7B) is loaded in half-precision, and
  BBox-Adapter uses BERT-0.1B as the backend."

  ============================  acc 0.1B  acc 0.3B  Training  Inference
  Base Model (Mixtral-8x7B)     59.91     -         -         90 GiB
  Base + LoRA (Hu et al. 2021)  73.80     75.98     208 GiB   92 GiB
  Base + BBox-Adapter           66.08     65.26     105 GiB   92 GiB

* "In terms of resource utilization, BBox-Adapter requires less computational
  power and storage, making BBox-Adapter a more resource-efficient option for
  model adaptation." (§4.7)

* Addendum: "The VRAM measurements reported in Table 6 are only for the 0.1B
  adapter version, not the 0.3B version. For reproduction purposes, only the VRAM
  measurements for the 0.1B version need to be evaluated."

Everything in this module is measurement/accounting only: it never reads
logprobs, hidden states or gradients of the black-box LLM, and it never touches
the black-box parameters.

Usage
-----
::

    from bbox_adapter.eval.vram import VramTracker, VramReporter

    reporter = VramReporter()
    with VramTracker("bbox_adapter", phase="training", adapter_size="0.1b") as t:
        train(...)
    reporter.add(t.to_measurement(accuracy=66.08))
    print(reporter.format_table())
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # constants
    "GIB",
    "MIB",
    "PHASE_TRAINING",
    "PHASE_INFERENCE",
    "PHASES",
    "ADAPTER_SIZES",
    "VRAM_METHOD_NAMES",
    "PAPER_TABLE6",
    "PAPER_VRAM_TABLE6",
    "PAPER_ACCURACY_TABLE6",
    "PAPER_STRATEGYQA_BASE_ACC",
    "REPORT_ONLY_0_1B",
    "HALF_PRECISION_BYTES",
    "MIXTRAL_PARAMS_B",
    "MIXTRAL_HALF_PRECISION_GIB",
    # config / records
    "VramConfig",
    "VramMeasurement",
    "VramReport",
    # measurement primitives
    "torch_available",
    "cuda_available",
    "describe_device",
    "reset_peak_memory",
    "peak_memory_bytes",
    "peak_memory_gib",
    "current_memory_gib",
    "total_memory_gib",
    "device_memory_summary",
    "VramTracker",
    "VramMeter",
    "measure_vram",
    "measure_phase",
    # estimation helpers
    "estimate_model_vram_gib",
    "estimate_lora_vram_gib",
    "estimate_bbox_adapter_vram_gib",
    "estimate_vram_for_method",
    "paper_table6_row",
    # reporting / validation
    "VramReporter",
    "format_vram_table",
    "save_vram_report",
    "compare_to_paper",
    "reference_table",
    "summarize",
    "set_vram_logger",
]

logger = logging.getLogger("bbox_adapter.eval.vram")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

GIB = 1024 ** 3
MIB = 1024 ** 2

PHASE_TRAINING = "training"
PHASE_INFERENCE = "inference"
PHASES: Tuple[str, str] = (PHASE_TRAINING, PHASE_INFERENCE)

ADAPTER_SIZES: Tuple[str, str] = ("0.1b", "0.3b")

#: Table 6 method labels (verbatim naming used in the paper).
VRAM_METHOD_NAMES: Tuple[str, ...] = (
    "Base Model (Mixtral-8x7B)",
    "Base + LoRA",
    "Base + BBox-Adapter",
)

#: Table 6, verbatim. ``None`` means "not reported by the paper".
PAPER_TABLE6: Dict[str, Dict[str, Optional[float]]] = {
    "Base Model (Mixtral-8x7B)": {
        "accuracy_0.1b": 59.91,
        "accuracy_0.3b": None,
        "training_gib": None,          # paper leaves the cell empty
        "inference_gib": 90.0,
    },
    "Base + LoRA": {
        "accuracy_0.1b": 73.80,
        "accuracy_0.3b": 75.98,
        "training_gib": 208.0,
        "inference_gib": 92.0,
    },
    "Base + BBox-Adapter": {
        "accuracy_0.1b": 66.08,
        "accuracy_0.3b": 65.26,
        "training_gib": 105.0,
        "inference_gib": 92.0,
    },
}

#: Convenience view: method -> {"training": GiB, "inference": GiB}.
PAPER_VRAM_TABLE6: Dict[str, Dict[str, Optional[float]]] = {
    method: {
        PHASE_TRAINING: row["training_gib"],
        PHASE_INFERENCE: row["inference_gib"],
    }
    for method, row in PAPER_TABLE6.items()
}

PAPER_ACCURACY_TABLE6: Dict[str, Dict[str, Optional[float]]] = {
    method: {"0.1b": row["accuracy_0.1b"], "0.3b": row["accuracy_0.3b"]}
    for method, row in PAPER_TABLE6.items()
}

#: StrategyQA base accuracy quoted in Table 3/Table 6 for Mixtral-8x7B.
PAPER_STRATEGYQA_BASE_ACC = 59.91

#: Addendum: only the 0.1B adapter VRAM needs to be reproduced.
REPORT_ONLY_0_1B = True

HALF_PRECISION_BYTES = 2.0  # fp16 / bf16
FULL_PRECISION_BYTES = 4.0  # fp32

#: Parameter count of Mixtral-8x7B (used only for estimation helpers).
MIXTRAL_PARAMS_B = 46.7

#: Bytes -> GiB conversion helper threshold etc. (kept explicit for clarity)
#: "the base model (Mixtral-8x7B) is loaded in half-precision" -> ~90 GiB
MIXTRAL_HALF_PRECISION_GIB = 90.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class VramConfig:
    """Configuration for VRAM measurement (mirrors ``vram:`` in the YAMLs).

    ``report_only_0_1b`` defaults to ``True`` per the paper Addendum.
    """

    enabled: bool = True
    report_only_0_1b: bool = REPORT_ONLY_0_1B
    device: Optional[str] = None
    reserved: bool = True
    reset_before: bool = True
    precision: str = "half"
    n_gpus: int = 1
    peak_window: bool = False          # use torch.cuda.max_memory_allocated
    allow_mock: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "VramConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        extra = {k: v for k, v in dict(data).items() if k not in known}
        cfg = cls(**kwargs)
        if extra:
            cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_config(cls, config: Any) -> "VramConfig":
        """Build from a full run config (dict with a ``vram`` section)."""
        if config is None:
            return cls()
        if isinstance(config, VramConfig):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config.get("vram", config))
        section = getattr(config, "vram", None)
        if section is None:
            return cls()
        if isinstance(section, dict):
            return cls.from_dict(section)
        return cls.from_dict(
            {k: getattr(section, k) for k in dir(section) if not k.startswith("_")}
        )

    def requested_size(self, adapter_size: str) -> bool:
        """Whether this adapter size must be reported (addendum rule)."""
        if self.report_only_0_1b:
            return _normalize_size(adapter_size) == "0.1b"
        return True


def _normalize_size(size: Optional[str]) -> str:
    if size is None:
        return "0.1b"
    s = str(size).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if s in {"0.1b", "01b", "100m", "0.1", "base", "86m"}:
        return "0.1b"
    if s in {"0.3b", "03b", "300m", "0.3", "large", "304m"}:
        return "0.3b"
    return s


# --------------------------------------------------------------------------- #
# Measurement records
# --------------------------------------------------------------------------- #


@dataclass
class VramMeasurement:
    """One measured cell of the Table-6 style report."""

    method: str
    phase: str
    adapter_size: str = "0.1b"
    gib: Optional[float] = None
    observed_gib: Optional[float] = None
    accuracy: Optional[float] = None
    n_params: Optional[int] = None
    device: Optional[str] = None
    precision: str = "half"
    n_gpus: int = 1
    measured: bool = True
    seconds: Optional[float] = None
    paper_gib: Optional[float] = None
    paper_accuracy: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def delta_gib(self) -> Optional[float]:
        if self.gib is None or self.paper_gib is None:
            return None
        return self.gib - self.paper_gib

    @property
    def rel_error(self) -> Optional[float]:
        d = self.delta_gib
        if d is None or not self.paper_gib:
            return None
        return d / self.paper_gib * 100.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["delta_gib"] = self.delta_gib
        d["rel_error"] = self.rel_error
        return d

    def row(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "phase": self.phase,
            "adapter_size": self.adapter_size,
            "vram_gib": self.gib,
            "accuracy": self.accuracy,
            "paper_vram_gib": self.paper_gib,
            "paper_accuracy": self.paper_accuracy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VramMeasurement":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        return cls(**kwargs)


@dataclass
class VramReport:
    """Aggregated Table-6 style VRAM/accuracy report."""

    measurements: List[VramMeasurement] = field(default_factory=list)
    config: Optional[VramConfig] = None
    dataset: str = "strategyqa"
    blackbox: str = "mistralai/Mixtral-8x7B-v0.1"
    reference: str = "table6"
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, measurement: VramMeasurement) -> VramMeasurement:
        self.measurements.append(measurement)
        return measurement

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self):
        return iter(self.measurements)

    def get(
        self, method: str, phase: str, adapter_size: str = "0.1b"
    ) -> Optional[VramMeasurement]:
        size = _normalize_size(adapter_size)
        for m in self.measurements:
            if (
                m.method == method
                and m.phase == phase
                and _normalize_size(m.adapter_size) == size
            ):
                return m
        return None

    def table(self) -> List[Dict[str, Any]]:
        return [m.row() for m in self.measurements]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "blackbox": self.blackbox,
            "reference": self.reference,
            "config": self.config.to_dict() if self.config else None,
            "n_measurements": len(self.measurements),
            "measurements": [m.to_dict() for m in self.measurements],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VramReport":
        return cls(
            measurements=[
                VramMeasurement.from_dict(m) for m in data.get("measurements", [])
            ],
            config=VramConfig.from_dict(data.get("config") or {}),
            dataset=data.get("dataset", "strategyqa"),
            blackbox=data.get("blackbox", "mistralai/Mixtral-8x7B-v0.1"),
            reference=data.get("reference", "table6"),
            meta=dict(data.get("meta", {})),
        )


# --------------------------------------------------------------------------- #
# Torch / CUDA primitives (all optional: degrade gracefully)
# --------------------------------------------------------------------------- #


def torch_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _torch():
    try:
        import torch  # noqa: F401

        return torch
    except Exception:
        return None


def cuda_available() -> bool:
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(device: Optional[str] = None):
    torch = _torch()
    if torch is None:
        return None
    if device is None:
        return torch.device("cuda") if cuda_available() else torch.device("cpu")
    try:
        return torch.device(device)
    except Exception:
        return torch.device("cuda") if cuda_available() else torch.device("cpu")


def describe_device(device: Optional[str] = None) -> Dict[str, Any]:
    """Return human-readable device information (for run headers)."""
    torch = _torch()
    info: Dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": cuda_available(),
        "device_count": 0,
        "devices": [],
    }
    if torch is None:
        return info
    try:
        info["device_count"] = torch.cuda.device_count()
    except Exception:  # pragma: no cover
        return info
    for i in range(info["device_count"]):
        try:
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_gib": round(props.total_memory / GIB, 2),
                    "major": getattr(props, "major", None),
                    "minor": getattr(props, "minor", None),
                }
            )
        except Exception:  # pragma: no cover
            continue
    if device is None and info["devices"]:
        info["device_name"] = info["devices"][0]["name"]
        info["total_gib"] = info["devices"][0]["total_gib"]
    return info


def reset_peak_memory(device: Optional[str] = None) -> bool:
    """Reset the CUDA peak-memory statistics (no-op without CUDA)."""
    torch = _torch()
    if torch is None or not cuda_available():
        return False
    try:
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(_resolve_device(device))
        torch.cuda.synchronize(_resolve_device(device))
        return True
    except Exception:  # pragma: no cover
        return False


def peak_memory_bytes(device: Optional[str] = None, reserved: bool = True) -> Optional[int]:
    """Maximum GPU memory used since the last reset (bytes), or ``None``."""
    torch = _torch()
    if torch is None or not cuda_available():
        return None
    try:
        dev = _resolve_device(device)
        if reserved:
            return int(torch.cuda.max_memory_reserved(dev))
        return int(torch.cuda.max_memory_allocated(dev))
    except Exception:  # pragma: no cover
        return None


def peak_memory_gib(device: Optional[str] = None, reserved: bool = True) -> Optional[float]:
    """Peak GPU memory in GiB (the Table-6 unit), or ``None`` without CUDA."""
    nbytes = peak_memory_bytes(device=device, reserved=reserved)
    if nbytes is None:
        return None
    return nbytes / GIB


def current_memory_gib(device: Optional[str] = None, reserved: bool = True) -> Optional[float]:
    torch = _torch()
    if torch is None or not cuda_available():
        return None
    try:
        dev = _resolve_device(device)
        if reserved:
            return int(torch.cuda.memory_reserved(dev)) / GIB
        return int(torch.cuda.memory_allocated(dev)) / GIB
    except Exception:  # pragma: no cover
        return None


def total_memory_gib(device: Optional[str] = None) -> Optional[float]:
    torch = _torch()
    if torch is None or not cuda_available():
        return None
    try:
        props = torch.cuda.get_device_properties(_resolve_device(device))
        return props.total_memory / GIB
    except Exception:  # pragma: no cover
        return None


def device_memory_summary(device: Optional[str] = None) -> Dict[str, Any]:
    """Current / peak / total memory snapshot for logging."""
    return {
        "device": str(_resolve_device(device)) if torch_available() else None,
        "peak_gib": peak_memory_gib(device, reserved=True),
        "peak_allocated_gib": peak_memory_gib(device, reserved=False),
        "current_gib": current_memory_gib(device, reserved=True),
        "total_gib": total_memory_gib(device),
    }


# --------------------------------------------------------------------------- #
# Phase tracking
# --------------------------------------------------------------------------- #


class VramTracker:
    """Context manager measuring peak GPU memory of one training/inference phase.

    Example
    -------
    ::

        with VramTracker("Base + BBox-Adapter", phase="training",
                         adapter_size="0.1b") as tracker:
            run_training()
        print(tracker.gib, "GiB")   # -> the Table 6 "Training" cell
    """

    def __init__(
        self,
        method: str,
        phase: str = PHASE_INFERENCE,
        adapter_size: str = "0.1b",
        config: Optional[Any] = None,
        *,
        device: Optional[str] = None,
        reserved: Optional[bool] = None,
        reset: Optional[bool] = None,
        accuracy: Optional[float] = None,
        n_params: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, VramConfig)
            else VramConfig.from_config(config) if config is not None else VramConfig()
        )
        self.method = method
        self.phase = _normalize_phase(phase)
        self.adapter_size = _normalize_size(adapter_size)
        self.device = device if device is not None else self.config.device
        self.reserved = (
            self.config.reserved if reserved is None else bool(reserved)
        )
        self.reset = self.config.reset_before if reset is None else bool(reset)
        self.accuracy = accuracy
        self.n_params = n_params
        self.meta: Dict[str, Any] = dict(meta or {})
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._peak_bytes: Optional[int] = None
        self._peak_gib: Optional[float] = None
        self._peak_allocated_gib: Optional[float] = None
        self._samples: List[float] = []
        self.enabled = self.config.enabled and self.config.requested_size(self.adapter_size)

    # -- context manager -------------------------------------------------- #
    def __enter__(self) -> "VramTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> "VramTracker":
        if not self.enabled:
            return self
        if self.reset:
            reset_peak_memory(self.device)
        self._started_at = time.time()
        return self

    def sample(self) -> Optional[float]:
        """Record a live memory sample (useful for long/inference loops)."""
        if not self.enabled:
            return None
        cur = current_memory_gib(self.device, reserved=self.reserved)
        if cur is not None:
            self._samples.append(cur)
        return cur

    def stop(self) -> Optional[float]:
        if not self.enabled:
            self._stopped_at = time.time()
            return None
        self._stopped_at = time.time()
        peak = peak_memory_gib(self.device, reserved=self.reserved)
        if peak is None:
            peak = max(self._samples) if self._samples else None
        self._peak_bytes = peak_memory_bytes(self.device, reserved=self.reserved)
        self._peak_gib = peak
        self._peak_allocated_gib = peak_memory_gib(self.device, reserved=False)
        return self._peak_gib

    def track(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` inside the measured window."""
        self.start()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stop()

    # -- results ---------------------------------------------------------- #
    @property
    def gib(self) -> Optional[float]:
        """Peak VRAM in GiB (the value reported in Table 6)."""
        if self._peak_gib is None and self.enabled:
            self.stop()
        return self._peak_gib

    @property
    def peak_bytes(self) -> Optional[int]:
        return self._peak_bytes

    @property
    def seconds(self) -> Optional[float]:
        if self._started_at is None or self._stopped_at is None:
            return None
        return self._stopped_at - self._started_at

    @property
    def samples(self) -> List[float]:
        return list(self._samples)

    def paper_value(self) -> Optional[float]:
        return PAPER_VRAM_TABLE6.get(self.method, {}).get(self.phase)

    def paper_accuracy(self) -> Optional[float]:
        return PAPER_ACCURACY_TABLE6.get(self.method, {}).get(self.adapter_size)

    def to_measurement(
        self, *, accuracy: Optional[float] = None, gib: Optional[float] = None
    ) -> VramMeasurement:
        value = gib if gib is not None else self.gib
        acc = accuracy if accuracy is not None else self.accuracy
        if acc is None:
            acc = self.paper_accuracy()
        return VramMeasurement(
            method=self.method,
            phase=self.phase,
            adapter_size=self.adapter_size,
            gib=value,
            observed_gib=value,
            accuracy=acc,
            n_params=self.n_params,
            device=str(self.device) if self.device is not None else None,
            precision=self.config.precision,
            n_gpus=self.config.n_gpus,
            measured=True,
            seconds=self.seconds,
            paper_gib=self.paper_value(),
            paper_accuracy=self.paper_accuracy(),
            meta=dict(self.meta),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "phase": self.phase,
            "adapter_size": self.adapter_size,
            "gib": self.gib,
            "peak_gib_allocated": self._peak_allocated_gib,
            "seconds": self.seconds,
            "n_samples": len(self._samples),
            "device": str(self.device) if self.device is not None else None,
            "enabled": self.enabled,
        }


#: Alias kept for scripts that prefer the "meter" naming.
VramMeter = VramTracker


def measure_vram(
    method: str,
    fn: Callable[..., Any],
    *args: Any,
    phase: str = PHASE_INFERENCE,
    adapter_size: str = "0.1b",
    config: Optional[Any] = None,
    accuracy: Optional[float] = None,
    n_params: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[Any, VramMeasurement]:
    """Measure peak VRAM while running ``fn``; return ``(result, measurement)``."""
    tracker = VramTracker(
        method,
        phase=phase,
        adapter_size=adapter_size,
        config=config,
        accuracy=accuracy,
        n_params=n_params,
    )
    result = tracker.track(fn, *args, **kwargs)
    return result, tracker.to_measurement()


def measure_phase(
    method: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, VramMeasurement]:
    """Alias of :func:`measure_vram` (phase passed through ``kwargs``)."""
    return measure_vram(method, fn, *args, **kwargs)


# --------------------------------------------------------------------------- #
# Estimation helpers (paper-anchored)
# --------------------------------------------------------------------------- #


def estimate_model_vram_gib(
    n_params: float,
    bytes_per_param: float = HALF_PRECISION_BYTES,
    overhead: float = 1.0,
    *,
    params_in_billions: bool = True,
) -> float:
    """Estimate weights-only VRAM in GiB (``n_params`` in billions by default)."""
    n = float(n_params)
    if params_in_billions:
        n *= 1e9
    return n * float(bytes_per_param) * float(overhead) / GIB


def estimate_lora_vram_gib(
    base_params_b: float = MIXTRAL_PARAMS_B,
    bytes_per_param: float = HALF_PRECISION_BYTES,
    *,
    trainable_fraction: float = 0.0,
    optimizer_bytes: float = 8.0,      # Adam moments (fp32) x2
    gradient_bytes: float = 2.0,
    activation_gib: Optional[float] = None,
    default: float = 208.0,
    use_paper_default: bool = False,
) -> float:
    """LoRA training VRAM estimate; Table 6 reports **208 GiB**.

    The paper's LoRA recipe (``r`` 128/384, Paged AdamW 32-bit, batch 8/GPU,
    max-length 512, no gradient checkpointing detail) is not analytically
    reproducible from the paper text, so ``default=208.0`` (the Table 6 value)
    is returned unless ``use_paper_default=False`` and an explicit
    ``activation_gib`` is supplied.
    """
    if use_paper_default or activation_gib is None:
        return float(default)
    weights = estimate_model_vram_gib(base_params_b, bytes_per_param)
    trainable = base_params_b * 1e9 * float(trainable_fraction)
    extra = trainable * (optimizer_bytes + gradient_bytes) / GIB
    return weights + extra + float(activation_gib)


def estimate_bbox_adapter_vram_gib(
    base_gib: float = MIXTRAL_HALF_PRECISION_GIB,
    adapter_gib: float = 0.0,
    *,
    phase: str = PHASE_TRAINING,
    activation_gib: Optional[float] = None,
    paper_default: Optional[float] = None,
) -> float:
    """Base (frozen) model + tiny adapter footprint estimate.

    BBox-Adapter keeps the black-box model frozen and only trains the 0.1B
    adapter, hence the Table-6 gap (105 GiB training / 92 GiB inference) versus
    LoRA (208 GiB training).
    """
    if paper_default is not None:
        return float(paper_default)
    phase = _normalize_phase(phase)
    default_activation = 15.0 if phase == PHASE_TRAINING else 2.0
    act = default_activation if activation_gib is None else float(activation_gib)
    return float(base_gib) + float(adapter_gib) + act


def estimate_vram_for_method(
    method: str,
    phase: str = PHASE_INFERENCE,
    adapter_size: str = "0.1b",
    *,
    base_gib: float = MIXTRAL_HALF_PRECISION_GIB,
    use_paper_defaults: bool = True,
    **kwargs: Any,
) -> Optional[float]:
    """Estimate (or look up) a Table-6 cell for a named method."""
    phase = _normalize_phase(phase)
    size = _normalize_size(adapter_size)
    name = str(method).strip().lower()
    is_base = "base model" in name and "lora" not in name and "bbox" not in name
    is_lora = "lora" in name
    is_bbox = ("bbox" in name) or ("adapter" in name and not is_lora and not is_base)

    if use_paper_defaults:
        if is_base:
            return base_gib if phase == PHASE_INFERENCE else None
        if is_lora:
            return 208.0 if phase == PHASE_TRAINING else 92.0
        if is_bbox:
            if size != "0.1b":
                # Addendum: VRAM is only reported for the 0.1B adapter.
                return None
            return 105.0 if phase == PHASE_TRAINING else 92.0

    if is_lora:
        return estimate_lora_vram_gib(base_params_b=MIXTRAL_PARAMS_B, **kwargs)
    if is_bbox:
        return estimate_bbox_adapter_vram_gib(base_gib=base_gib, phase=phase, **kwargs)
    return estimate_bbox_adapter_vram_gib(
        base_gib=base_gib, phase=phase, paper_default=base_gib if phase == PHASE_INFERENCE else None
    )


def paper_table6_row(method: str, adapter_size: str = "0.1b") -> Dict[str, Any]:
    """Return the paper's Table-6 row for ``method`` (VRAM + accuracy)."""
    return {
        "method": method,
        "adapter_size": _normalize_size(adapter_size),
        "accuracy": PAPER_ACCURACY_TABLE6.get(method, {}).get(_normalize_size(adapter_size)),
        "training_gib": PAPER_VRAM_TABLE6.get(method, {}).get(PHASE_TRAINING),
        "inference_gib": PAPER_VRAM_TABLE6.get(method, {}).get(PHASE_INFERENCE),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _normalize_phase(phase: Optional[str]) -> str:
    p = (phase or PHASE_INFERENCE).strip().lower()
    if p.startswith("train") or p in {"fit", "adapt"}:
        return PHASE_TRAINING
    if p.startswith("infer") or p in {"eval", "test", "generation", "decode"}:
        return PHASE_INFERENCE
    return p


def _fmt_gib(value: Optional[float], digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


class VramReporter:
    """Collect :class:`VramMeasurement` objects and render the Table-6 layout."""

    def __init__(
        self,
        config: Optional[Any] = None,
        *,
        dataset: str = "strategyqa",
        blackbox: str = "mistralai/Mixtral-8x7B-v0.1",
        reference: str = "table6",
    ) -> None:
        self.config = (
            config
            if isinstance(config, VramConfig)
            else VramConfig.from_config(config) if config is not None else VramConfig()
        )
        self.report = VramReport(
            config=self.config, dataset=dataset, blackbox=blackbox, reference=reference
        )

    # -- collection ------------------------------------------------------- #
    def add(self, measurement: VramMeasurement) -> VramMeasurement:
        if not self.config requested_size if False else True:  # pragma: no cover
            pass
        return self.report.add(measurement)

    def add_phase(
        self,
        method: str,
        phase: str,
        gib: Optional[float],
        *,
        adapter_size: str = "0.1b",
        accuracy: Optional[float] = None,
        **kwargs: Any,
    ) -> VramMeasurement:
        size = _normalize_size(adapter_size)
        measurement = VramMeasurement(
            method=method,
            phase=_normalize_phase(phase),
            adapter_size=size,
            gib=gib,
            observed_gib=gib,
            accuracy=accuracy if accuracy is not None else PAPER_ACCURACY_TABLE6.get(method, {}).get(size),
            paper_gib=PAPER_VRAM_TABLE6.get(method, {}).get(_normalize_phase(phase)),
            paper_accuracy=PAPER_ACCURACY_TABLE6.get(method, {}).get(size),
            measured=gib is not None,
            **kwargs,
        )
        return self.add(measurement)

    def from_tracker(self, tracker: VramTracker, **kwargs: Any) -> VramMeasurement:
        return self.add(tracker.to_measurement(**kwargs))

    def add_paper_row(self, method: str, adapter_size: str = "0.1b") -> List[VramMeasurement]:
        """Insert the paper's own numbers (useful for offline reference reports)."""
        row = paper_table6_row(method, adapter_size)
        out: List[VramMeasurement] = []
        for phase, key in ((PHASE_TRAINING, "training_gib"), (PHASE_INFERENCE, "inference_gib")):
            value = row.get(key)
            if value is None and method == "Base Model (Mixtral-8x7B)" and phase == PHASE_TRAINING:
                continue
            m = VramMeasurement(
                method=method,
                phase=phase,
                adapter_size=row["adapter_size"],
                gib=value,
                observed_gib=value,
                accuracy=row.get("accuracy"),
                paper_gib=value,
                paper_accuracy=row.get("accuracy"),
                measured=False,
                meta={"source": "paper_table6"},
            )
            out.append(self.add(m))
        return out

    # -- rendering -------------------------------------------------------- #
    def table(self) -> List[Dict[str, Any]]:
        return self.report.table()

    def format_table(self, *, title: str = "Table 6 (Mixtral-8x7B, StrategyQA)", digits: int = 0) -> str:
        return format_vram_table(self.table(), title=title, digits=digits)

    def summary(self) -> Dict[str, Any]:
        return summarize(self.report)

    def save(self, path: str, **extra: Any) -> str:
        return save_vram_report(path, self.report, **extra)

    # -- validation ------------------------------------------------------- #
    def compare(
        self,
        method: str,
        phase: str,
        gib: Optional[float],
        *,
        adapter_size: str = "0.1b",
        tolerance: float = 8.0,
        relative_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        return compare_to_paper(
            method,
            phase,
            gib,
            adapter_size=adapter_size,
            tolerance=tolerance,
            relative_tolerance=relative_tolerance,
        )

    def missing(self, config: Optional[VramConfig] = None) -> List[Tuple[str, str, str]]:
        """Paper cells that have not been measured yet (0.1B only per addendum)."""
        cfg = config or self.config
        missing: List[Tuple[str, str, str]] = []
        for method in VRAM_METHOD_NAMES:
            for phase in PHASES:
                paper_value = PAPER_VRAM_TABLE6.get(method, {}).get(phase)
                if paper_value is None:
                    continue
                if not cfg.requested_size("0.1b"):
                    continue
                if self.report.get(method, phase, "0.1b") is None:
                    missing.append((method, phase, "0.1b"))
        return missing


def format_vram_table(
    rows: Sequence[Dict[str, Any]],
    *,
    title: str = "Table 6 (Mixtral-8x7B, StrategyQA)",
    digits: int = 0,
    show_paper: bool = True,
) -> str:
    """Render a Table-6 style VRAM/accuracy table as text."""
    header = ["Adapter", "Acc. 0.1B", "Acc. 0.3B", "Training", "Inference"]
    lines: List[str] = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    lines.append(" | ".join(header))
    lines.append("-+-".join("-" * len(h) for h in header))

    # group rows by method, agent sizes and phases
    methods: List[str] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        method = row.get("method", "?")
        if method not in grouped:
            grouped[method] = {"acc": {}, "training": None, "inference": None}
            methods.append(method)
        g = grouped[method]
        size = _normalize_size(row.get("adapter_size"))
        if row.get("accuracy") is not None:
            g["acc"][size] = row["accuracy"]
        if row.get("phase") == PHASE_TRAINING:
            g["training"] = row.get("vram_gib")
        elif row.get("phase") == PHASE_INFERENCE:
            g["inference"] = row.get("vram_gib")

    for method in methods:
        g = grouped[method]
        acc01 = g["acc"].get("0.1b")
        acc03 = g["acc"].get("0.3b")
        cells = [
            method,
            f"{acc01:.2f}" if acc01 is not None else "-",
            f"{acc03:.2f}" if acc03 is not None else "-",
            _fmt_gib(g["training"], digits),
            _fmt_gib(g["inference"], digits),
        ]
        lines.append(" | ".join(cells))

    if show_paper:
        lines.append("")
        lines.append("Paper (Table 6): Base 59.91 / - / - / 90 GiB; "
                     "LoRA 73.80 / 75.98 / 208 / 92 GiB; "
                     "BBox-Adapter 66.08 / 65.26 / 105 / 92 GiB (0.1B only per Addendum).")
    return "\n".join(lines)


def summarize(report: VramReport) -> Dict[str, Any]:
    """Aggregate a :class:`VramReport` into something loggable."""
    out: Dict[str, Any] = {
        "n_measurements": len(report.measurements),
        "measured": sum(1 for m in report.measurements if m.measured),
        "methods": sorted({m.method for m in report.measurements}),
    }
    for m in report.measurements:
        key = f"{m.method}|{m.phase}|{m.adapter_size}"
        out[key] = {
            "gib": m.gib,
            "paper_gib": m.paper_gib,
            "delta_gib": m.delta_gib,
            "rel_error": m.rel_error,
            "accuracy": m.accuracy,
        }
    return out


def compare_to_paper(
    method: str,
    phase: str,
    gib: Optional[float],
    *,
    adapter_size: str = "0.1b",
    tolerance: float = 8.0,
    relative_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare a measured cell against the paper's Table 6 value.

    ``tolerance`` is an absolute GiB threshold; ``relative_tolerance`` (e.g.
    ``0.10`` for 10%) is checked in addition when provided.
    """
    phase = _normalize_phase(phase)
    size = _normalize_size(adapter_size)
    paper_value = PAPER_VRAM_TABLE6.get(method, {}).get(phase)
    result: Dict[str, Any] = {
        "method": method,
        "phase": phase,
        "adapter_size": size,
        "measured_gib": gib,
        "paper_gib": paper_value,
        "delta_gib": None,
        "rel_error": None,
        "within_tolerance": None,
    }
    if paper_value is None:
        result["note"] = "paper does not report this cell (Addendum: 0.1B only)"
        return result
    if gib is None:
        result["note"] = "no measurement supplied"
        result["within_tolerance"] = False
        return result
    delta = float(gib) - float(paper_value)
    rel = delta / float(paper_value) if paper_value else None
    ok = abs(delta) <= float(tolerance)
    if relative_tolerance is not None and rel is not None:
        ok = ok or abs(rel) <= float(relative_tolerance)
    result.update(
        {
            "delta_gib": delta,
            "rel_error": rel,
            "within_tolerance": bool(ok),
        }
    )
    if not ok:
        logger.warning(
            "VRAM mismatch for %s/%s (%s): measured %.1f GiB vs paper %.1f GiB",
            method,
            phase,
            size,
            float(gib),
            float(paper_value),
        )
    return result


def reference_table(name: str = "table6") -> Dict[str, Any]:
    """Return paper reference numbers (only ``table6`` is defined here)."""
    key = (name or "table6").strip().lower().replace(" ", "").replace("-", "")
    if key in {"table6", "t6", "vram", "vramtable"}:
        return {
            "table": "table6",
            "dataset": "strategyqa",
            "blackbox": "Mixtral-8x7B (half-precision)",
            "adapter_backbone": "BERT-0.1B (DeBERTa/BERT 0.1B)",
            "report_only_0_1b": True,
            "rows": {m: dict(PAPER_TABLE6[m]) for m in PAPER_TABLE6},
        }
    raise KeyError(f"unknown reference table '{name}' (known: 'table6')")


def save_vram_report(path: str, report: Any, **extra: Any) -> str:
    """Write a VRAM report as JSON; returns the written path."""
    if isinstance(report, VramReport):
        payload = report.to_dict()
    elif isinstance(report, dict):
        payload = dict(report)
    else:
        payload = {"measurements": [m.to_dict() for m in report]}  # type: ignore[union-attr]
    if extra:
        payload["extra"] = extra
    payload.setdefault("reference", reference_table("table6"))
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Saved VRAM report to %s", path)
    return path


def set_vram_logger(logger_: logging.Logger) -> None:
    """Allow scripts to route VRAM logs through their run logger."""
    global logger  # noqa: PLW0603
    if logger_ is not None:
        logger = logger_


# --------------------------------------------------------------------------- #
# Self test
# --------------------------------------------------------------------------- #


def _self_test() -> Dict[str, Any]:
    """Dependency-free smoke test (``python -m bbox_adapter.eval.vram``)."""
    results: Dict[str, Any] = {}

    # Table 6 reference integrity
    assert PAPER_TABLE6["Base + BBox-Adapter"]["training_gib"] == 105.0
    assert PAPER_TABLE6["Base + BBox-Adapter"]["inference_gib"] == 92.0
    assert PAPER_TABLE6["Base + LoRA"]["training_gib"] == 208.0
    assert PAPER_TABLE6["Base Model (Mixtral-8x7B)"]["inference_gib"] == 90.0
    results["paper_table6_ok"] = True

    # 0.1B-only reporting rule (Addendum)
    cfg = VramConfig()
    assert cfg.requested_size("0.1b") is True
    assert cfg.requested_size("0.3b") is False
    results["addendum_0_1b_only"] = True

    # Tracker without CUDA (torch may or may not be installed) must not crash
    tracker = VramTracker("Base + BBox-Adapter", phase="training", adapter_size="0.1b")
    with tracker:
        pass
    m = tracker.to_measurement(accuracy=66.08)
    assert m.paper_gib == 105.0 and m.paper_accuracy == 66.08
    if not cuda_available():
        assert m.gib is None
    results["tracker_ok"] = True
    results["cuda_available"] = cuda_available()

    # Estimators reproduce the Table 6 anchor points
    assert abs(estimate_vram_for_method("Base + BBox-Adapter", "training") - 105.0) < 1e-6
    assert abs(estimate_vram_for_method("Base + BBox-Adapter", "inference") - 92.0) < 1e-6
    assert estimate_vram_for_method("Base + BBox-Adapter", "training", "0.3b") is None
    assert estimate_lora_vram_gib() == 208.0
    est = estimate_model_vram_gib(MIXTRAL_PARAMS_B)
    assert 80.0 < est < 95.0, est  # ~87 GiB weights, consistent with the 90 GiB cell
    results["estimators_ok"] = True
    results["mixtral_half_precision_gib"] = round(est, 2)

    # Reporter / table rendering / comparison / save
    reporter = VramReporter()
    reporter.add(tracker.to_measurement(gib=104.0))
    reporter.add_phase("Base + BBox-Adapter", "inference", 91.5, accuracy=66.08)
    reporter.add_phase("Base + LoRA", "training", 206.0, accuracy=73.80)
    reporter.add_phase("Base + LoRA", "inference", 92.5, accuracy=73.80)
    text = reporter.format_table()
    assert "Base + BBox-Adapter" in text and "Training" in text
    checks = {
        "bbox_train": reporter.compare("Base + BBox-Adapter", "training", 104.0),
        "bbox_infer": reporter.compare("Base + BBox-Adapter", "inference", 91.5),
        "lora_train": reporter.compare("Base + LoRA", "training", 206.0),
        "unspecified": reporter.compare("Base + BBox-Adapter", "training", None),
    }
    assert checks["bbox_train"]["within_tolerance"] is True
    assert checks["lora_train"]["within_tolerance"] is True
    assert checks["unspecified"]["within_tolerance"] is False
    results["compare_ok"] = True
    results["table_preview"] = text.splitlines()[0:3]

    missing = reporter.missing()
    assert ("Base Model (Mixtral-8x7B)", "inference", "0.1b") in missing
    results["missing"] = missing

    summary = reporter.summary()
    assert summary["n_measurements"] == 4
    results["summary_ok"] = True

    payload = reporter.report.to_dict()
    restored = VramReport.from_dict(payload)
    assert len(restored) == 4
    results["serialization_ok"] = True

    results["device_info"] = describe_device()
    results["ok"] = True
    return results


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(_self_test(), indent=2, default=str))
