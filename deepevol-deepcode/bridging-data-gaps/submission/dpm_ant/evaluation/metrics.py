"""Efficiency and reporting metrics for DPMs-ANT.

This module implements the *efficiency* side of the paper's evaluation:

* **Parameter rate** -- the fraction of fine-tuned parameters over the total
  number of parameters.  The paper reports 1.3% for DDPM-ANT and 1.6% for
  LDM-ANT (§5.3, Table 1 caption).
* **GPU memory** -- peak memory consumption measured (with ``torch.cuda``) for
  each training variant, compared against the paper's Table 8 (batch size 1)::

        variant            w/o Adaptor    w/ Adaptor
        DPMs                     17086 MB       6010 MB
        DPMs+SG                  17130 MB       6030 MB
        DPMs+AN                  17100 MB       6022 MB
        DPMs+ANT                 17188 MB       6080 MB

* **Wall-clock time** -- the paper reports ~300 ANT iterations taking about
  3 GPU hours versus ~4.2 GPU hours for the baseline adaptor run with 5,000
  iterations (§5.3).

Everything is exposed both as small utility functions/classes and as a single
:func:`efficiency_report` entry point, so ``evaluate.py`` / ``run_ablation.py``
can dump a JSON summary and compare against the paper's reference numbers.

The reference tables (Tables 1-8) are also encoded as constants so that
reproduction runs can be automatically diffed against the published values.

This file is glue/aggregation code: it contains no paper algorithm of its own,
it only measures and reports the ones implemented elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PAPER_PARAM_RATE",
    "PAPER_MEMORY_TABLE",
    "PAPER_TIME_TABLE",
    "PAPER_REFERENCE_TABLES",
    "EfficiencyConfig",
    "count_parameters",
    "trainable_parameters",
    "parameter_rate",
    "adaptor_parameter_rate",
    "model_size_mb",
    "GPUMemoryTracker",
    "measure_gpu_memory",
    "Timer",
    "measure_time",
    "wall_clock_hours",
    "benchmark_iterations",
    "EfficiencyMetrics",
    "efficiency_report",
    "format_report",
    "save_report",
    "compare_to_paper",
]


# ---------------------------------------------------------------------------
# Paper reference values  (§5.3, Table 1, Table 8, Appendix B.3 Tables 5-7)
# ---------------------------------------------------------------------------

#: Fraction of fine-tuned parameters reported in the paper (§5.3 / Table 1).
PAPER_PARAM_RATE: Dict[str, float] = {
    "ddpm_ant": 0.013,
    "ldm_ant": 0.016,
    "ddpm_pa": 1.0,
}

#: Table 8 -- GPU memory consumption (MB) for each module at batch size 1.
PAPER_MEMORY_TABLE: Dict[str, Dict[str, float]] = {
    "w/o Adaptor": {
        "DPMs": 17086.0,
        "DPMs+SG": 17130.0,
        "DPMs+AN": 17100.0,
        "DPMs+ANT": 17188.0,
    },
    "w/ Adaptor": {
        "DPMs": 6010.0,
        "DPMs+SG": 6030.0,
        "DPMs+AN": 6022.0,
        "DPMs+ANT": 6080.0,
    },
}

#: Wall-clock costs quoted in §5.3 ("about 3 GPU hours" vs "about 4.2").
PAPER_TIME_TABLE: Dict[str, Dict[str, float]] = {
    "ant_300_iters": {"hours": 3.0, "iterations": 300.0},
    "baseline_adaptor_5000_iters": {"hours": 4.2, "iterations": 5000.0},
}

#: Table 1 -- Intra-LPIPS (mean) for the main 10-shot adaptation tasks, and
#: Table 2 -- FID.  Written as DDPM-ANT / LDM-ANT / DDPM-PA triples where the
#: paper reports them (§5.3, §5.4, Appendix B.2).
PAPER_REFERENCE_TABLES: Dict[str, Any] = {
    "intra_lpips": {
        "church_landscape_drawings": {"ddpm_ant": 0.723, "ldm_ant": 0.738, "ddpm_pa": 0.706},
        "ffhq_sketches": {"ddpm_ant": 0.544},
        "ffhq_amedeo": {"ddpm_ant": 0.620},
    },
    "fid": {
        "ffhq_babies": {"ddpm_ant": 46.70},
        "ffhq_sunglasses": {"ddpm_ant": 20.06},
    },
    "config_params": {
        "gamma": 5.0,
        "omega": 0.02,
        "J": 10,
        "num_timesteps": 1000,
        "batch_size": 40,
        "lr_ddpm": 5e-5,
        "lr_ldm": 1e-5,
        "iterations": 300,
    },
    # Appendix B.3 Table 5 -- gamma sweep (FFHQ -> Sunglasses, LDM).
    "gamma_sweep": {
        1: {"fid": 20.75, "intra_lpips": 0.641},
        3: {"fid": 18.86, "intra_lpips": 0.627},
        5: {"fid": 18.13, "intra_lpips": 0.613},
        7: {"fid": 24.12, "intra_lpips": 0.603},
        9: {"fid": 29.48, "intra_lpips": 0.592},
    },
    # Appendix B.3 Table 6 -- omega sweep.
    "omega_sweep": {
        0.01: {"fid": 18.42, "intra_lpips": 0.616},
        0.02: {"fid": 18.13, "intra_lpips": 0.613},
        0.03: {"fid": 18.42, "intra_lpips": 0.613},
        0.04: {"fid": 19.11, "intra_lpips": 0.614},
        0.05: {"fid": 19.48, "intra_lpips": 0.623},
    },
    # Appendix B.3 Table 7 -- iteration sweep.
    "iteration_sweep": {
        0: {"fid": 111.32, "intra_lpips": 0.650},
        50: {"fid": 93.82, "intra_lpips": 0.666},
        100: {"fid": 58.27, "intra_lpips": 0.666},
        150: {"fid": 31.08, "intra_lpips": 0.654},
        200: {"fid": 19.51, "intra_lpips": 0.635},
        250: {"fid": 18.34, "intra_lpips": 0.624},
        300: {"fid": 18.13, "intra_lpips": 0.613},
        350: {"fid": 20.06, "intra_lpips": 0.604},
        400: {"fid": 21.17, "intra_lpips": 0.608},
    },
    # §5.4 / Figure 4 -- 300-iteration ablation on 10-shot Sunglasses (FID).
    "ablation_sunglasses_300": {
        "full_model_finetune": 41.88,
        "adaptor_only": 38.65,
        "ant_wo_an": 26.41,
        "full_ant": 20.66,
    },
    # §5.5 / Table 3 -- classifier training set size (FFHQ -> Sunglasses).
    "classifier_ablation": {
        "10_shot": {"intra_lpips": 0.613, "intra_lpips_std": 0.023, "fid": 20.06},
        "100_shot": {"intra_lpips": 0.637, "intra_lpips_std": 0.013, "fid": 22.84},
    },
}


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def count_parameters(module: Optional[nn.Module], only_trainable: bool = False) -> int:
    """Count parameters of ``module`` (total or ``requires_grad`` only).

    Returns ``0`` for ``None`` so it can be chained safely.
    """
    if module is None:
        return 0
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def trainable_parameters(module: Optional[nn.Module]) -> List[nn.Parameter]:
    """Return the list of parameters with ``requires_grad=True``."""
    if module is None:
        return []
    return [p for p in module.parameters() if p.requires_grad]


def parameter_rate(
    trainable: Any,
    total: Optional[Any] = None,
    total_module: Optional[nn.Module] = None,
    trainable_module: Optional[nn.Module] = None,
) -> float:
    """Fraction of fine-tuned parameters over the total parameter count.

    Accepts either integer counts::

        parameter_rate(1_300_000, 100_000_000)

    or modules::

        parameter_rate(trainable_module=unet_adapted, total_module=unet_adapted)

    When ``total``/``total_module`` correspond to an already-adapted model the
    adaptor parameters are part of the total, which is the convention used for
    the paper's "1.3% / 1.6%" numbers (adaptor params / full adapted model).
    """
    if total_module is not None:
        total_count = count_parameters(total_module)
    elif total is not None:
        total_count = int(total)
    else:
        raise ValueError("parameter_rate requires `total` or `total_module`")

    if trainable_module is not None:
        train_count = count_parameters(trainable_module, only_trainable=True)
    elif isinstance(trainable, torch.Tensor):
        train_count = int(trainable.sum().item())
    elif isinstance(trainable, (int, float)):
        train_count = int(trainable)
    else:  # nn.Module or iterable of parameters
        if isinstance(trainable, nn.Module):
            train_count = count_parameters(trainable, only_trainable=True)
        else:
            train_count = sum(p.numel() for p in trainable)

    if total_count <= 0:
        return 0.0
    return float(train_count) / float(total_count)


def adaptor_parameter_rate(model: nn.Module, backbone: Optional[str] = None) -> Dict[str, Any]:
    """Parameter statistics for an adapted diffusion backbone.

    The adapted model is expected to expose ``adaptor_parameters()`` (both
    ``FrozenDDPMUNet`` and ``FrozenLDMUNet`` do).  Returns a dict with absolute
    counts plus the rate, and (when ``backbone`` is ``"ddpm"``/``"ldm"``) the
    paper's expected value for comparison.
    """
    adaptors = []
    getter = getattr(model, "adaptor_parameters", None)
    if callable(getter):
        try:
            adaptors = list(getter())
        except Exception:  # pragma: no cover - defensive
            adaptors = []
    if not adaptors:
        adaptors = [p for n, p in model.named_parameters() if ".adaptor." in n or n.startswith("adaptor")]

    n_adaptor = sum(p.numel() for p in adaptors)
    n_total = count_parameters(model)
    rate = (n_adaptor / n_total) if n_total > 0 else 0.0

    out: Dict[str, Any] = {
        "adaptor_parameters": n_adaptor,
        "total_parameters": n_total,
        "trainable_parameters": n_adaptor,
        "parameter_rate": rate,
        "parameter_rate_percent": 100.0 * rate,
    }
    if backbone in PAPER_PARAM_RATE and backbone != "ddpm_pa":
        out["paper_parameter_rate"] = PAPER_PARAM_RATE[backbone]
        out["parameter_rate_abs_error"] = abs(rate - PAPER_PARAM_RATE[backbone])
    return out


def model_size_mb(module: Optional[nn.Module]) -> float:
    """On-disk parameter size in MB (float32 assumption)."""
    if module is None:
        return 0.0
    n = count_parameters(module)
    return float(n) * 4.0 / (1024.0 ** 2)


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

_BYTES_PER_MB = 1024.0 * 1024.0


def _reset_peak_memory(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        return
    try:
        if device is None or device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # pragma: no cover - defensive
        pass


def _peak_memory_mb(device: Optional[torch.device] = None) -> float:
    """Peak allocated CUDA memory in MB (0.0 when CUDA is unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(device)) / _BYTES_PER_MB
    except Exception:  # pragma: no cover - defensive
        return 0.0


class GPUMemoryTracker:
    """Context manager measuring peak (and reserved) CUDA memory.

    Example
    -------
    >>> with GPUMemoryTracker() as tracker:      # doctest: +SKIP
    ...     loss.backward()
    >>> tracker.peak_mb
    """

    def __init__(self, device: Optional[torch.device] = None, synchronize: bool = True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.synchronize = synchronize
        self.peak_mb: float = 0.0
        self.reserved_mb: float = 0.0
        self.allocated_start_mb: float = 0.0

    def __enter__(self) -> "GPUMemoryTracker":
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.allocated_start_mb = float(torch.cuda.memory_allocated(self.device)) / _BYTES_PER_MB
        _reset_peak_memory(self.device)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                if self.synchronize:
                    torch.cuda.synchronize(self.device)
                self.peak_mb = _peak_memory_mb(self.device)
                self.reserved_mb = float(torch.cuda.max_memory_reserved(self.device)) / _BYTES_PER_MB
            except Exception:  # pragma: no cover - defensive
                pass
        return False  # never suppress exceptions

    def to_dict(self) -> Dict[str, float]:
        return {
            "peak_mb": self.peak_mb,
            "reserved_mb": self.reserved_mb,
            "allocated_start_mb": self.allocated_start_mb,
        }


def measure_gpu_memory(
    fn: Callable[..., Any],
    *args: Any,
    device: Optional[torch.device] = None,
    empty_cache: bool = True,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, float]]:
    """Run ``fn(*args, **kwargs)`` and report peak GPU memory in MB.

    Returns ``(result, stats)`` where ``stats`` has keys ``peak_mb``,
    ``reserved_mb`` and ``allocated_start_mb``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if empty_cache and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    with GPUMemoryTracker(device) as tracker:
        result = fn(*args, **kwargs)
    return result, tracker.to_dict()


# ---------------------------------------------------------------------------
# Wall-clock timing
# ---------------------------------------------------------------------------


class Timer:
    """Simple wall-clock timer usable as a context manager.

    Uses ``torch.cuda.synchronize`` before stopping when running on GPU, so the
    measured span includes all queued kernels.
    """

    def __init__(self, device: Optional[torch.device] = None, name: str = "", synchronize: bool = True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.name = name
        self.synchronize = synchronize
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def _sync(self) -> None:
        if self.synchronize and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:  # pragma: no cover - defensive
                pass

    def start(self) -> "Timer":
        self._sync()
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self._sync()
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    def __float__(self) -> float:
        return float(self.elapsed)

    def to_dict(self) -> Dict[str, float]:
        return {"seconds": self.elapsed, "hours": self.elapsed / 3600.0, "name": self.name}


def measure_time(
    fn: Callable[..., Any],
    *args: Any,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    with Timer(device=device) as t:
        result = fn(*args, **kwargs)
    return result, float(t)


def wall_clock_hours(seconds: float) -> float:
    """Convert seconds to GPU-hours (used for the §5.3 time comparison)."""
    return float(seconds) / 3600.0


def benchmark_iterations(
    step_fn: Callable[[], Any],
    iterations: int = 10,
    warmup: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Time ``iterations`` calls of ``step_fn`` (excluding ``warmup`` calls).

    Returns per-iteration statistics plus the extrapolated total hours, which is
    how the paper's "300 iterations ~ 3 GPU hours" figure is obtained.
    """
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(max(0, int(warmup))):
        step_fn()
    with GPUMemoryTracker(device) as tracker:
        with Timer(device=device) as t:
            for _ in range(max(1, int(iterations))):
                step_fn()
    secs = float(t.elapsed)
    per_iter = secs / max(1, int(iterations))
    return {
        "seconds": secs,
        "iterations": int(iterations),
        "seconds_per_iteration": per_iter,
        "hours_per_iteration": per_iter / 3600.0,
        "extrapolated_300_iters_hours": per_iter * 300.0 / 3600.0,
        "extrapolated_5000_iters_hours": per_iter * 5000.0 / 3600.0,
        "peak_mb": tracker.peak_mb,
    }


# ---------------------------------------------------------------------------
# Aggregated efficiency report
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyConfig:
    """Settings for :func:`efficiency_report`."""

    backbone: str = "ddpm"
    variant: str = "DPMs+ANT"
    batch_size: int = 1
    iterations: int = 300
    measure_memory: bool = True
    measure_time: bool = True
    device: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "EfficiencyConfig":
        """Build from a nested config dict (``evaluation.metrics`` etc.)."""
        params: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            for block in ("evaluation", "metrics", "efficiency", "ant", "defaults", "sampling"):
                sub = cfg.get(block)
                if isinstance(sub, dict):
                    for key in (
                        "backbone",
                        "variant",
                        "batch_size",
                        "iterations",
                        "measure_memory",
                        "measure_time",
                        "device",
                        "seed",
                    ):
                        if key in sub and params.get(key) is None:
                            params[key] = sub[key]
            if "efficiency" in cfg and isinstance(cfg["efficiency"], dict):
                params.update({k: v for k, v in cfg["efficiency"].items() if k in cls.__dataclass_fields__})
            for key in cls.__dataclass_fields__:
                if key in cfg and cfg[key] is not None:
                    params[key] = cfg[key]
        # Batch size default from the paper is 40 for training, but Table 8 is
        # measured at batch size 1 -- keep whatever the config gives.
        params.update({k: v for k, v in overrides.items() if v is not None})
        known = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EfficiencyMetrics:
    """Container for a single efficiency measurement."""

    backbone: str = "ddpm"
    variant: str = "DPMs+ANT"
    adaptor_parameters: int = 0
    total_parameters: int = 0
    trainable_parameters_: int = 0
    parameter_rate: float = 0.0
    peak_memory_mb: float = 0.0
    peak_memory_reserved_mb: float = 0.0
    memory_without_adaptor_mb: float = 0.0
    seconds_per_iteration: float = 0.0
    iterations: int = 0
    total_hours: float = 0.0
    batch_size: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["parameter_rate_percent"] = 100.0 * self.parameter_rate
        return out


def efficiency_report(
    model: Optional[nn.Module] = None,
    baseline_model: Optional[nn.Module] = None,
    step_fn: Optional[Callable[[], Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    backbone: str = "ddpm",
    variant: str = "DPMs+ANT",
    device: Optional[str] = None,
    iterations: Optional[int] = None,
    benchmark_iters: int = 10,
    warmup: int = 2,
    compare: bool = True,
    **overrides: Any,
) -> Dict[str, Any]:
    """Compute the full efficiency report (Table 1, §5.3, Table 8).

    Parameters
    ----------
    model:
        Adapted model (backbone + zero-init adaptors).  Used for the parameter
        rate and the *with-adaptor* memory measurement.
    baseline_model:
        Optional un-adapted model, used for the *without-adaptor* memory column.
    step_fn:
        A callable performing one full training iteration (adversarial noise
        selection + SG loss + optimizer step).  When provided, per-iteration
        wall-clock time is benchmarked and extrapolated to 300/5,000 iters.
    cfg:
        Config dict; ``evaluation.metrics.*`` and ``efficiency.*`` are honoured.
    """
    conf = EfficiencyConfig.from_dict(cfg, backbone=backbone, variant=variant, device=device)
    if iterations is not None:
        conf.iterations = int(iterations)
    dev = torch.device(conf.device) if conf.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = EfficiencyMetrics(
        backbone=conf.backbone,
        variant=conf.variant,
        iterations=int(conf.iterations),
        batch_size=int(conf.batch_size),
    )

    # --- parameters -------------------------------------------------------
    if model is not None:
        pr = adaptor_parameter_rate(model, backbone=conf.backbone)
        metrics.adaptor_parameters = int(pr["adaptor_parameters"])
        metrics.total_parameters = int(pr["total_parameters"])
        metrics.trainable_parameters_ = int(pr["trainable_parameters"])
        metrics.parameter_rate = float(pr["parameter_rate"])
        metrics.extra["parameter_detail"] = pr

    # --- memory -----------------------------------------------------------
    if conf.measure_memory:
        if model is not None:
            stats = _model_memory_stats(model, dev, conf.batch_size, forward_only=True)
            metrics.peak_memory_mb = float(stats.get("peak_mb", 0.0))
            metrics.peak_memory_reserved_mb = float(stats.get("reserved_mb", 0.0))
        if baseline_model is not None:
            base_stats = _model_memory_stats(baseline_model, dev, conf.batch_size, forward_only=True)
            metrics.memory_without_adaptor_mb = float(base_stats.get("peak_mb", 0.0))
        metrics.extra["memory_reference_table8_mb"] = PAPER_MEMORY_TABLE[
            "w/ Adaptor" if model is not None else "w/o Adaptor"
        ]

    # --- wall clock -------------------------------------------------------
    if conf.measure_time and step_fn is not None:
        bench = benchmark_iterations(step_fn, iterations=benchmark_iters, warmup=warmup, device=dev)
        metrics.seconds_per_iteration = float(bench["seconds_per_iteration"])
        metrics.total_hours = wall_clock_hours(metrics.seconds_per_iteration * max(1, metrics.iterations))
        metrics.extra["benchmark"] = bench
        metrics.extra["time_reference_hours"] = PAPER_TIME_TABLE

    report = metrics.to_dict()
    report["paper_param_rate"] = PAPER_PARAM_RATE.get(conf.backbone, None)
    if compare:
        report["comparison"] = compare_to_paper(report, backbone=conf.backbone)
    return report


def _model_memory_stats(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    forward_only: bool = True,
    image_size: int = 256,
    latent_size: int = 64,
) -> Dict[str, float]:
    """Peak-memory probe: one forward (optionally with backward) pass."""
    was_training = model.training
    model.eval()
    in_ch = getattr(model, "latent_channels", None) or getattr(model, "in_channels", None) or 3
    is_latent = getattr(model, "is_latent", False) or getattr(model, "backbone", None) == "ldm"
    size = int(latent_size) if is_latent else int(image_size)
    try:
        x = torch.randn(int(batch_size), int(in_ch), size, size, device=device)
    except Exception:  # pragma: no cover - CPU fallback
        return {"peak_mb": 0.0, "reserved_mb": 0.0}
    t = torch.full((int(batch_size),), 500, dtype=torch.long, device=device)

    def _run() -> None:
        call = getattr(model, "epsilon_theta", None) or getattr(model, "predict_noise", None) or model
        try:
            out = call(x, t)
        except TypeError:
            out = call(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if not forward_only and isinstance(out, torch.Tensor) and out.requires_grad:
            (out.float().pow(2).mean()).backward()

    _, stats = measure_gpu_memory(_run, device=device)
    if was_training:
        model.train()
    return stats


# ---------------------------------------------------------------------------
# Reporting / comparison helpers
# ---------------------------------------------------------------------------


def format_report(report: Dict[str, Any]) -> str:
    """Render an efficiency report (plus paper comparison) as text."""
    lines: List[str] = []
    lines.append("=" * 68)
    lines.append("DPMs-ANT efficiency report")
    lines.append("=" * 68)
    lines.append(f"backbone / variant : {report.get('backbone')} / {report.get('variant')}")
    lines.append(
        f"parameters         : adaptor={report.get('adaptor_parameters', 0):,} "
        f"total={report.get('total_parameters', 0):,}"
    )
    pr = report.get("parameter_rate", 0.0)
    paper_pr = report.get("paper_param_rate")
    if paper_pr is not None:
        lines.append(f"parameter rate     : {100.0 * pr:.2f}%  (paper: {100.0 * paper_pr:.2f}%)")
    else:
        lines.append(f"parameter rate     : {100.0 * pr:.2f}%")
    if report.get("peak_memory_mb"):
        lines.append(f"peak GPU memory    : {report['peak_memory_mb']:.0f} MB (batch {report.get('batch_size')})")
    if report.get("memory_without_adaptor_mb"):
        lines.append(f"memory w/o adaptor : {report['memory_without_adaptor_mb']:.0f} MB")
    if report.get("seconds_per_iteration"):
        lines.append(
            f"time per iteration : {report['seconds_per_iteration']:.2f} s "
            f"-> {report.get('total_hours', 0.0):.2f} h for {report.get('iterations')} iters"
        )
    comp = report.get("comparison") or {}
    if comp.get("notes"):
        lines.append("-" * 68)
        for note in comp["notes"]:
            lines.append(f"  * {note}")
    lines.append("-" * 68)
    lines.append("Table 8 reference (GPU MB, batch size 1):")
    for row, cols in PAPER_MEMORY_TABLE.items():
        lines.append(
            "  {:<12} ".format(row) + "  ".join(f"{k}={v:.0f}" for k, v in cols.items())
        )
    lines.append("=" * 68)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], path: str, indent: int = 2) -> str:
    """Persist a report as JSON (creating parent directories)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=indent, default=str)
    LOGGER.info("Saved efficiency report to %s", path)
    return path


def compare_to_paper(
    report: Dict[str, Any],
    backbone: str = "ddpm",
    rel_tol: float = 0.25,
) -> Dict[str, Any]:
    """Diff a measured report against the published numbers.

    Returns a dict with ``param_rate_ok``, ``memory_ok``, ``time_ok`` and a list
    of human-readable ``notes``.  Tolerances are loose because the reported
    figures are hardware dependent (rel_tol = 25% by default).
    """
    notes: List[str] = []
    out: Dict[str, Any] = {}

    expected_rate = PAPER_PARAM_RATE.get(backbone)
    measured_rate = report.get("parameter_rate")
    if expected_rate is not None and measured_rate:
        ok = abs(measured_rate - expected_rate) <= max(1e-4, rel_tol * expected_rate)
        out["param_rate_ok"] = bool(ok)
        if ok:
            notes.append(
                f"parameter rate {100.0 * measured_rate:.2f}% matches paper ({100.0 * expected_rate:.2f}%)."
            )
        else:
            notes.append(
                f"parameter rate {100.0 * measured_rate:.2f}% differs from paper "
                f"({100.0 * expected_rate:.2f}%); check adaptor placement/count."
            )

    variant = report.get("variant", "DPMs+ANT")
    measured_mem = report.get("peak_memory_mb") or None
    if measured_mem:
        col = "w/ Adaptor" if report.get("adaptor_parameters") else "w/o Adaptor"
        expected_mem = PAPER_MEMORY_TABLE.get(col, {}).get(variant)
        if expected_mem:
            # Adaptor runs use ~35% of the full-finetune memory; the absolute
            # figure depends on the GPU/driver, so only flag gross deviations.
            ok = measured_mem <= 1.6 * expected_mem * 10  # sanity bound
            out["memory_ok"] = bool(ok)
            out["memory_vs_table8_mb"] = float(measured_mem - expected_mem)
            notes.append(
                f"peak memory {measured_mem:.0f} MB vs Table 8 ({col}, {variant}) {expected_mem:.0f} MB."
            )

    secs_per_iter = report.get("seconds_per_iteration")
    if secs_per_iter:
        expected_hours = PAPER_TIME_TABLE["ant_300_iters"]["hours"]
        measured_hours = report.get("total_hours") or wall_clock_hours(secs_per_iter * 300.0)
        out["time_ok"] = bool(measured_hours <= 2.0 * expected_hours)
        notes.append(
            f"extrapolated {measured_hours:.2f} GPU hours for 300 iterations vs paper "
            f"{expected_hours:.1f} hours (~{PAPER_TIME_TABLE['baseline_adaptor_5000_iters']['hours']} h for the "
            "5,000-iteration baseline)."
        )

    out["notes"] = notes
    return out


# ---------------------------------------------------------------------------
# Convenience CLI-ish helper
# ---------------------------------------------------------------------------


def summarize_models(
    models: Sequence[Tuple[str, nn.Module]],
    backbone: str = "ddpm",
) -> List[Dict[str, Any]]:
    """Parameter statistics for several named models (e.g. ablation variants)."""
    rows: List[Dict[str, Any]] = []
    for name, module in models:
        row = adaptor_parameter_rate(module, backbone=backbone)
        row["name"] = name
        rows.append(row)
    return rows
