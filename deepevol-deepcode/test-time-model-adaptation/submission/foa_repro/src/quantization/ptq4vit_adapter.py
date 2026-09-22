"""PTQ4ViT adapter for the FOA reproduction (paper Section 4.2 / Table 4 and Appendix B.2 / Table 17).

The paper quantizes the frozen ViT-Base backbone with **PTQ4ViT** (Yuan et al., 2022,
https://github.com/hahnyuan/PTQ4ViT) to 8-bit and 6-bit using 32 *randomly selected samples
from the training set* for calibration, and then applies FOA unchanged:

    "We adopt PTQ4ViT (Yuan et al., 2022) for 8-bit and 6-bit model quantization with 32
     randomly selected samples from the training set."                                   (B.2)

    "On the contrary, our FOA is adaptable to these quantized models. We demonstrate this by
     applying FOA to quantized ViT models and benchmarking it against T3A. ... Notably, FOA
     with an 8-bit ViT surpasses the performance of the gradient-based TENT method using a
     full precision 32-bit ViT on ImageNet-C, achieving 63.5% accuracy (our FOA, 8-bit) vs.
     59.6% (TENT, 32-bit)."                                                               (4.2)

    "The implementation details of PTQ4ViT for model quantization are partially missing from
     the main text. We refer to the original PTQ4ViT paper for the complete quantization
     process details."                                                              (Addendum)

This module therefore provides:

1. `try_load_official_ptq4vit` / `quantize_with_official_ptq4vit` - use the official PTQ4ViT
   repository when it is installed/vendored (preferred path, per the addendum), which is the
   place where the complete quantization process lives.
2. A self-contained, dependency-light emulation of PTQ4ViT's **twin-uniform** post-training
   quantization for the forward-only inference that FOA needs:
   * weights: asymmetric uniform, per-output-channel scales chosen by the twin-uniform scale
     search (minimise the quantization MSE over candidate clip points);
   * activations: asymmetric uniform, static per-tensor scales, also twin-uniform searched
     over cached calibration statistics;
   * attention: optional softmax-aware quantization of attention logits / softmax weights /
     value matmul through a re-implemented attention forward that degrades gracefully to the
     original timm forward.

Everything is inference-only: the quantized model keeps `requires_grad=False` and the FOA
loop never calls `backward()`.  Straight-through estimators are used so the same quantized
model *can* also be driven by gradient baselines for comparison, but no code path here
performs a backward pass.

Reported reference numbers (Section 4.2 / Table 4 and Table 17):
    * FOA 8-bit: 63.5% accuracy, 3.8% average ECE
    * FOA 6-bit: 55.8% accuracy, 5.5% average ECE
    * 8-bit FOA (63.5%) > 32-bit TENT (59.6%)

Public interface
----------------
* `QuantConfig`                      - dataclass of quantization hyper-parameters.
* `FakeQuantizer`                    - weight/activation fake-quantization module.
* `QuantizedLinear` / `QuantizedConv2d`
* `twin_uniform_search`              - PTQ4ViT-style clipping search (min-MSE scale).
* `select_qparams_minmax`            - baseline min-max scale rule.
* `quantize_tensor` / `fake_quantize_ste`
* `replace_linear_layers`            - in-place module surgery.
* `quantize_attention_modules`       - softmax-aware attention quantization.
* `calibrate_activations`            - static activation range estimation (32 samples).
* `quantize_model`                   - full pipeline: replace + patch + calibrate + freeze.
* `build_calibration_stream`         - random ImageNet-1K *training* images for calibration.
* `build_quantized_model`            - FOA ViT-Base wrapper -> quantized backbone.
* `build_quantized_foa`              - ready-to-run FOA runner on a quantized model.
* `run_quantized` / `parse_args` / `main` - CLI reproducing the Table 4 / Table 17 rows.
* `TABLE4_REFERENCE`, `SUPPORTED_BITS`, `DEFAULT_CALIBRATION_SAMPLES`, `PTQ4VIT_REPO`
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants (paper: Appendix B.2 / Section 4.2 / Table 4 / Table 17)
# --------------------------------------------------------------------------------------

#: PTQ4ViT bit-widths evaluated in the paper (Table 4, Table 17).
SUPPORTED_BITS: Tuple[int, ...] = (8, 6)

#: "32 randomly selected samples from the training set" (Appendix B.2).
DEFAULT_CALIBRATION_SAMPLES: int = 32

#: Official implementation referenced by the paper / addendum.
PTQ4VIT_REPO: str = "https://github.com/hahnyuan/PTQ4ViT"

#: Reference numbers reported by the paper (Table 4 accuracy; Table 17 average ECE).
#: FOA: 63.5% / 3.8% at 8-bit and 55.8% / 5.5% at 6-bit (Section 4.2 + Table 17);
#: NoAdapt: 10.8% / 9.9% average ECE (Table 17); T3A: 25.9% / 30.1% average ECE.
TABLE4_REFERENCE: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "8bit": {
        "foa": {"accuracy": 63.5, "ece": 3.8},
        "noadapt": {"accuracy": None, "ece": 10.8},
        "t3a": {"accuracy": None, "ece": 25.9},
    },
    "6bit": {
        "foa": {"accuracy": 55.8, "ece": 5.5},
        "noadapt": {"accuracy": None, "ece": 9.9},
        "t3a": {"accuracy": None, "ece": 30.1},
    },
}

#: Number of candidate clipping points in the twin-uniform scale search (PTQ4ViT/LSQ-style).
DEFAULT_NUM_CANDIDATES: int = 100

#: Number of scalar samples cached per activation layer for the scale search.
DEFAULT_CACHE_PER_LAYER: int = 1 << 15  # 32768 scalars

#: Default ImageNet normalization (matches src/data/datasets.py and the timm ViT config).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE: int = 224

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Hyper-parameters of the post-training quantizer.

    Defaults follow paper Appendix B.2: PTQ4ViT with 32 calibration samples from the
    ImageNet-1K *training* set, 8-bit (or 6-bit) for weights and activations.

    Fields
    ------
    bits                 : default bit-width (8 or 6).
    weight_bits          : override for weights (defaults to `bits`).
    activation_bits      : override for activations (defaults to `bits`).
    calibration_samples  : number of unlabeled samples used for calibration (paper: 32).
    calibration_split    : dataset split used for calibration (`"train"` per the paper).
    seed                 : RNG seed for selecting the random calibration samples.
    quantize_weights     : enable weight quantization.
    quantize_activations : enable static activation quantization.
    per_channel_weights  : per-output-channel weight scales (PTQ4ViT does this).
    symmetric_weights    : symmetric (zero_point = 0) weight grid.
    symmetric_activations: symmetric activation grid.
    scale_search         : `"twin_uniform"` (PTQ4ViT, default) or `"minmax"`.
    num_candidates       : candidate clip points for the twin-uniform search.
    dynamic_activations  : recompute activation scales per batch instead of statically
                           calibrated scales (fallback when no calibration data exists).
    quantize_attention   : softmax-aware quantization of attention logits/weights/output.
    quantize_patch_embed : quantize the patch-embedding convolution (off by default).
    quantize_head        : quantize the classification head (off by default: the paper
                           keeps the head in full precision for the FOA comparison).
    exclude_prefixes     : module-name prefixes that are never quantized.
    cache_per_layer      : scalar samples cached per layer during calibration.
    """

    bits: int = 8
    weight_bits: Optional[int] = None
    activation_bits: Optional[int] = None
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES
    calibration_split: str = "train"
    seed: int = 0
    quantize_weights: bool = True
    quantize_activations: bool = True
    per_channel_weights: bool = True
    symmetric_weights: bool = False
    symmetric_activations: bool = False
    scale_search: str = "twin_uniform"
    num_candidates: int = DEFAULT_NUM_CANDIDATES
    dynamic_activations: bool = False
    quantize_attention: bool = True
    quantize_patch_embed: bool = False
    quantize_head: bool = False
    exclude_prefixes: Tuple[str, ...] = ("head",)
    cache_per_layer: int = DEFAULT_CACHE_PER_LAYER
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- resolution helpers -------------------------------------------------------------
    @property
    def resolved_weight_bits(self) -> int:
        return int(self.weight_bits if self.weight_bits is not None else self.bits)

    @property
    def resolved_activation_bits(self) -> int:
        return int(self.activation_bits if self.activation_bits is not None else self.bits)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolved_weight_bits"] = self.resolved_weight_bits
        d["resolved_activation_bits"] = self.resolved_activation_bits
        return d

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "QuantConfig":
        """Build from a FOA YAML config (`model.precision` and/or a `quantization` block)."""
        values: Dict[str, Any] = {}
        quant_block = _cfg_get(cfg, "quantization") if cfg is not None else None
        if isinstance(quant_block, dict):
            values.update({k: v for k, v in quant_block.items() if k in cls.__dataclass_fields__})
            if "bits" in quant_block and quant_block.get("bits") is not None:
                values["bits"] = int(quant_block["bits"])
        elif quant_block is not None:
            for key in cls.__dataclass_fields__:  # pragma: no cover - attribute-style configs
                val = getattr(quant_block, key, None)
                if val is not None:
                    values[key] = val

        if cfg is not None and "bits" not in values:
            precision = _cfg_get(cfg, "model", "precision")
            if precision in (6, 8):
                values["bits"] = int(precision)
            elif precision in ("8bit", "int8"):
                values["bits"] = 8
            elif precision in ("6bit", "int6"):
                values["bits"] = 6
        if cfg is not None and "seed" not in values:
            seed = _cfg_get(cfg, "seed")
            if seed is not None:
                values["seed"] = int(seed)
        if cfg is not None and "calibration_samples" not in values:
            n_src = _cfg_get(cfg, "source_stats", "num_samples")
            if n_src is not None:
                # The paper uses 32 random *training* samples for calibration; the source
                # statistics bank uses Q=32 validation samples. Reuse only as a fallback.
                values["calibration_samples"] = int(n_src)

        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def validate(self) -> None:
        if int(self.bits) not in SUPPORTED_BITS:
            logger.warning(
                "QuantConfig.bits=%s is outside the paper's evaluated set %s (continuing).",
                self.bits, SUPPORTED_BITS,
            )
        if not (2 <= int(self.resolved_weight_bits) <= 16):
            raise ValueError(f"unsupported weight bit-width: {self.resolved_weight_bits}")
        if not (2 <= int(self.resolved_activation_bits) <= 16):
            raise ValueError(f"unsupported activation bit-width: {self.resolved_activation_bits}")
        if self.scale_search not in ("twin_uniform", "minmax", "mse"):
            raise ValueError(f"unknown scale_search: {self.scale_search!r}")


# --------------------------------------------------------------------------------------
# Low-level quantizers (twin-uniform, PTQ4ViT style)
# --------------------------------------------------------------------------------------


def _qmax(bits: int, symmetric: bool) -> Tuple[float, float]:
    """Return (qmin, qmax) of the integer grid.

    Asymmetric unsigned grid ``[0, 2**b - 1]`` by default (PTQ4ViT uses unsigned uniform
    quantization with a searched clipping threshold); symmetric
    ``[-(2**(b-1)-1), 2**(b-1)-1]`` when requested.
    """
    if symmetric:
        q = 2 ** (bits - 1) - 1
        return -float(q), float(q)
    return 0.0, float(2 ** bits - 1)


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: float,
    qmax: float,
) -> torch.Tensor:
    """Round `x` onto the integer grid and de-quantize back to float (fake quant)."""
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    zero_point = torch.as_tensor(zero_point, dtype=x.dtype, device=x.device)
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return (q - zero_point) * scale


def fake_quantize_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: float,
    qmax: float,
) -> torch.Tensor:
    """Fake quantization with a straight-through estimator (identity gradient)."""
    dq = quantize_tensor(x, scale, zero_point, qmin, qmax)
    return x + (dq - x).detach()


def select_qparams_minmax(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = False,
    dim: Optional[int] = None,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain min-max scale (baseline rule, used for comparison with twin-uniform)."""
    qmin, qmax = _qmax(bits, symmetric)
    xf = x.detach()
    if dim is None:
        lo = torch.min(xf)
        hi = torch.max(xf)
        if symmetric:
            m = torch.maximum(lo.abs(), hi.abs()).clamp(min=eps)
            scale = m / qmax
            zero = torch.zeros_like(scale)
        else:
            lo = torch.minimum(lo, xf.new_zeros(()))
            hi = torch.maximum(hi, xf.new_zeros(()))
            scale = ((hi - lo) / (qmax - qmin)).clamp(min=eps)
            zero = torch.round(qmin - lo / scale)
        return scale.to(x.dtype), zero.to(x.dtype)

    # per-channel along `dim` (keepdim for broadcasting)
    if symmetric:
        m = torch.amax(xf.abs(), dim=dim, keepdim=True).clamp(min=eps)
        scale = m / qmax
        zero = torch.zeros_like(scale)
    else:
        lo = torch.amin(xf, dim=dim, keepdim=True)
        hi = torch.amax(xf, dim=dim, keepdim=True)
        lo = torch.minimum(lo, xf.new_zeros(()))
        hi = torch.maximum(hi, xf.new_zeros(()))
        scale = ((hi - lo) / (qmax - qmin)).clamp(min=eps)
        zero = torch.round(qmin - lo / scale)
    return scale.to(x.dtype), zero.to(x.dtype)


@torch.no_grad()
def twin_uniform_search(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = False,
    dim: Optional[int] = None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PTQ4ViT-style **twin-uniform** quantization-parameter search.

    Candidate step sizes ``delta_j = delta_max * j / M`` (``j = 1..M``) are evaluated with the
    same rounding/de-quantization rule the model will use at inference time, and the
    candidate minimising the element-wise squared reconstruction error
    ``||Q(x) - x||_2^2`` is selected.  This mirrors PTQ4ViT's twin-uniform quantizer, which
    searches the uniform step (clipping threshold) by minimising the quantization MSE.
    """
    qmin, qmax = _qmax(bits, symmetric)
    levels = (qmax - qmin)

    if dim is None:
        xf = x.detach().float().reshape(-1)
        if xf.numel() == 0:
            one = torch.ones((), dtype=x.dtype, device=x.device)
            return one, torch.zeros((), dtype=x.dtype, device=x.device)
        if symmetric:
            a = xf.abs().max().clamp(min=eps)
            lo, hi = -a, a
        else:
            lo = torch.minimum(xf.min(), xf.new_zeros(()))
            hi = torch.maximum(xf.max(), xf.new_zeros(()))
            hi = torch.maximum(hi, lo + eps)
        span = (hi - lo).clamp(min=eps)

        ratios = torch.linspace(1.0 / num_candidates, 1.0, num_candidates,
                                device=x.device, dtype=xf.dtype)
        best_err: Optional[torch.Tensor] = None
        best_scale = (span / levels).clamp(min=eps)
        best_zero = torch.zeros((), device=x.device, dtype=xf.dtype)
        for r in ratios:
            scale = (span * r / levels).clamp(min=eps)
            if symmetric:
                zero = torch.zeros((), device=x.device, dtype=xf.dtype)
            else:
                zero = torch.round(qmin - lo / scale)
            dq = quantize_tensor(xf, scale, zero, qmin, qmax)
            err = (dq - xf).pow(2).mean()
            if best_err is None or float(err) < float(best_err):
                best_err = err
                best_scale, best_zero = scale, zero
        return best_scale.to(x.dtype), best_zero.to(x.dtype)

    # per-channel search along `dim` (weights): vectorised over the channel dim.
    xf = x.detach().float()
    reduce_dims = [d for d in range(xf.dim()) if d != dim]
    if symmetric:
        a = xf.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=eps)
        lo, hi = -a, a
    else:
        lo = torch.minimum(xf.amin(dim=reduce_dims, keepdim=True), xf.new_zeros(()))
        hi = torch.maximum(xf.amax(dim=reduce_dims, keepdim=True), xf.new_zeros(()))
        hi = torch.maximum(hi, lo + eps)
    span = (hi - lo).clamp(min=eps)

    best_err = None
    best_scale = (span / levels).clamp(min=eps)
    best_zero = torch.zeros_like(best_scale)
    for j in range(1, num_candidates + 1):
        r = j / num_candidates
        scale = (span * r / levels).clamp(min=eps)
        if symmetric:
            zero = torch.zeros_like(scale)
        else:
            zero = torch.round(qmin - lo / scale)
        dq = quantize_tensor(xf, scale, zero, qmin, qmax)
        err = (dq - xf).pow(2)
        for d in sorted(reduce_dims, reverse=True):
            err = err.mean(dim=d, keepdim=True)
        if best_err is None:
            best_err, best_scale, best_zero = err, scale, zero
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale, best_scale)
            best_zero = torch.where(better, zero, best_zero)
    return best_scale.to(x.dtype), best_zero.to(x.dtype)


def select_qparams_twin_uniform(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = False,
    dim: Optional[int] = None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit alias for readability in the calibration code."""
    return twin_uniform_search(x, bits, symmetric=symmetric, dim=dim,
                               num_candidates=num_candidates)


# --------------------------------------------------------------------------------------
# Quantized modules
# --------------------------------------------------------------------------------------


class FakeQuantizer(nn.Module):
    """Static (or dynamic) asymmetric uniform fake-quantizer for activations.

    Parameters are estimated from calibration statistics (`observe` + `finalize`) with the
    selected scale-search rule; `forward` applies the frozen parameters (or re-estimates them
    per batch when `dynamic=True`).
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        dynamic: bool = False,
        scale_search: str = "twin_uniform",
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        cache_per_layer: int = DEFAULT_CACHE_PER_LAYER,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.dynamic = bool(dynamic)
        self.scale_search = scale_search
        self.num_candidates = int(num_candidates)
        self.cache_per_layer = int(cache_per_layer)
        self.enabled = bool(enabled)
        self.qmin, self.qmax = _qmax(self.bits, self.symmetric)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.calibrated = False
        self.num_observations = 0
        self._cache: Optional[torch.Tensor] = None

    # -- calibration -------------------------------------------------------------------
    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Cache a subsample of activation values for the later scale search."""
        if not self.enabled or self.dynamic:
            return
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return
        if flat.numel() > self.cache_per_layer:
            stride = max(1, flat.numel() // self.cache_per_layer)
            flat = flat[::stride][: self.cache_per_layer]
        flat = flat.float().cpu()
        if self._cache is None:
            self._cache = flat
        else:
            self._cache = torch.cat([self._cache, flat])[: 2 * self.cache_per_layer]
        self.num_observations += 1

    @torch.no_grad()
    def finalize(self) -> None:
        """Estimate and freeze the quantization parameters."""
        if not self.enabled or self.dynamic:
            self.calibrated = True
            return
        if self._cache is None or self._cache.numel() == 0:
            logger.debug("FakeQuantizer.finalize: no observations; keeping default parameters")
            self.calibrated = True
            return
        x = self._cache
        if self.scale_search == "minmax":
            scale, zero = select_qparams_minmax(x, self.bits, symmetric=self.symmetric)
        else:
            scale, zero = select_qparams_twin_uniform(
                x, self.bits, symmetric=self.symmetric, num_candidates=self.num_candidates
            )
        self.scale = scale.detach().to(torch.float32).reshape(())
        self.zero_point = zero.detach().to(torch.float32).reshape(())
        self.calibrated = True
        self._cache = None

    # -- inference ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.dynamic:
            scale, zero = select_qparams_minmax(
                x.detach(), self.bits, symmetric=self.symmetric
            )
            return fake_quantize_ste(x, scale, zero, self.qmin, self.qmax)
        return fake_quantize_ste(x, self.scale, self.zero_point, self.qmin, self.qmax)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"bits={self.bits}, symmetric={self.symmetric}, dynamic={self.dynamic}, "
                f"search={self.scale_search}, calibrated={self.calibrated}")


class QuantizedLinear(nn.Module):
    """`nn.Linear` with PTQ4ViT-style post-training weight (and activation) quantization.

    * Weights are quantized once (per-output-channel asymmetric uniform, twin-uniform scale
      search) and the de-quantized weights are cached, so inference is a plain `F.linear`.
    * The input activation passes through a `FakeQuantizer` when activation quantization is
      enabled; the output stays in full precision (the next layer's quantizer handles it),
      matching the standard post-training-quantization convention.
    """

    def __init__(
        self,
        linear: nn.Linear,
        config: QuantConfig,
        name: str = "",
        quantize_activation: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.name = name
        self.weight_bits = config.resolved_weight_bits
        self.activation_bits = config.resolved_activation_bits
        self.per_channel = bool(config.per_channel_weights)
        self.symmetric_w = bool(config.symmetric_weights)
        self.num_candidates = int(config.num_candidates)
        self.scale_search = config.scale_search
        self.qmin, self.qmax = _qmax(self.weight_bits, self.symmetric_w)

        weight = linear.weight.detach().clone()
        bias = None if linear.bias is None else linear.bias.detach().clone()

        # -- weight quantization (per-output-channel by default) ------------------------
        if config.quantize_weights:
            dim = 0 if self.per_channel else None
            if self.scale_search == "minmax":
                w_scale, w_zero = select_qparams_minmax(
                    weight, self.weight_bits, symmetric=self.symmetric_w, dim=dim
                )
            else:
                w_scale, w_zero = select_qparams_twin_uniform(
                    weight, self.weight_bits, symmetric=self.symmetric_w, dim=dim,
                    num_candidates=self.num_candidates,
                )
            dq_weight = quantize_tensor(weight, w_scale, w_zero, self.qmin, self.qmax)
            if dim is not None:
                self.register_buffer("w_scale", w_scale.detach().reshape(-1).float())
                self.register_buffer("w_zero", w_zero.detach().reshape(-1).float())
            else:
                self.register_buffer("w_scale", w_scale.detach().reshape(()).float())
                self.register_buffer("w_zero", w_zero.detach().reshape(()).float())
            self.weight_quantized = True
        else:
            dq_weight = weight
            self.register_buffer("w_scale", torch.tensor(1.0))
            self.register_buffer("w_zero", torch.tensor(0.0))
            self.weight_quantized = False
        self.register_buffer("weight_q", dq_weight.detach())

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        do_act = config.quantize_activations if quantize_activation is None else bool(quantize_activation)
        self.act_quant = FakeQuantizer(
            bits=self.activation_bits,
            symmetric=config.symmetric_activations,
            dynamic=config.dynamic_activations,
            scale_search=config.scale_search,
            num_candidates=config.num_candidates,
            cache_per_layer=config.cache_per_layer,
            enabled=do_act,
        )

    # ---------------------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """De-quantized weights (API compatibility with `nn.Linear`)."""
        return self.weight_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quant(x)
        return F.linear(x_q, self.weight_q, self.bias)

    def observe_input(self, x: torch.Tensor) -> None:
        self.act_quant.observe(x)

    def finalize(self) -> None:
        self.act_quant.finalize()

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w{self.weight_bits}/a{self.activation_bits}, name={self.name!r}")


class QuantizedConv2d(nn.Module):
    """Per-output-channel post-training quantized 2-D convolution (patch embedding)."""

    def __init__(self, conv: nn.Conv2d, config: QuantConfig, name: str = "") -> None:
        super().__init__()
        self.name = name
        self.weight_bits = config.resolved_weight_bits
        self.symmetric_w = bool(config.symmetric_weights)
        weight = conv.weight.detach().clone()
        if config.quantize_weights:
            dim = 0  # per output channel
            if config.scale_search == "minmax":
                w_scale, w_zero = select_qparams_minmax(
                    weight, self.weight_bits, symmetric=self.symmetric_w, dim=dim
                )
            else:
                w_scale, w_zero = select_qparams_twin_uniform(
                    weight, self.weight_bits, symmetric=self.symmetric_w, dim=dim,
                    num_candidates=config.num_candidates,
                )
            qmin, qmax = _qmax(self.weight_bits, self.symmetric_w)
            weight = quantize_tensor(weight, w_scale, w_zero, qmin, qmax)
            self.register_buffer("w_scale", w_scale.detach().reshape(-1).float())
            self.register_buffer("w_zero", w_zero.detach().reshape(-1).float())
        else:
            self.register_buffer("w_scale", torch.tensor(1.0))
            self.register_buffer("w_zero", torch.tensor(0.0))
        self.register_buffer("weight_q", weight.detach())
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().clone())
        else:
            self.bias = None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.act_quant = FakeQuantizer(
            bits=config.resolved_activation_bits,
            symmetric=config.symmetric_activations,
            dynamic=config.dynamic_activations,
            scale_search=config.scale_search,
            num_candidates=config.num_candidates,
            cache_per_layer=config.cache_per_layer,
            enabled=config.quantize_activations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            self.act_quant(x), self.weight_q, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
        )

    def finalize(self) -> None:
        self.act_quant.finalize()


# --------------------------------------------------------------------------------------
# Module surgery
# --------------------------------------------------------------------------------------


def _iter_named_modules(root: nn.Module, prefix: str = "") -> Iterator[Tuple[str, nn.Module]]:
    for name, child in root.named_children():
        full = f"{prefix}.{name}" if prefix else name
        yield full, child
        yield from _iter_named_modules(child, full)


def _is_excluded(name: str, config: QuantConfig) -> bool:
    for pref in config.exclude_prefixes:
        if not pref:
            continue
        if name == pref or name.startswith(pref + "."):
            return True
    if not config.quantize_head and name.split(".")[-1] in ("head", "fc", "classifier"):
        return True
    if not config.quantize_patch_embed and "patch_embed" in name:
        return True
    return False


def replace_linear_layers(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
    parent_prefix: str = "",
) -> List[Tuple[str, QuantizedLinear]]:
    """Recursively replace `nn.Linear` (and optionally the patch-embed conv) in-place.

    Returns the list of ``(qualified_name, quantized_module)`` pairs for calibration.
    """
    config = config or QuantConfig()
    replaced: List[Tuple[str, QuantizedLinear]] = []
    for name, child in list(model.named_children()):
        full = f"{parent_prefix}.{name}" if parent_prefix else name
        if isinstance(child, nn.Linear):
            if _is_excluded(full, config):
                continue
            q = QuantizedLinear(child, config, name=full)
            setattr(model, name, q)
            replaced.append((full, q))
        elif isinstance(child, nn.Conv2d) and config.quantize_patch_embed and "patch_embed" in full:
            setattr(model, name, QuantizedConv2d(child, config, name=full))
        else:
            replaced.extend(replace_linear_layers(child, config, full))
    return replaced


# --------------------------------------------------------------------------------------
# Attention quantization (PTQ4ViT softmax-aware treatment, optional and defensive)
# --------------------------------------------------------------------------------------


def _patch_attention_forward(module: nn.Module, config: QuantConfig) -> bool:
    """Patch a timm attention module so attention logits/weights/output are fake-quantized.

    The patch re-implements timm's standard attention forward (qkv -> reshape -> matmul ->
    softmax -> matmul -> proj) using the module's own attributes, inserting a per-tensor
    quantizer on the attention logits (before softmax), on the softmax output and on the
    value-matmul result - the parts PTQ4ViT quantizes in a softmax-aware way.  Any contract
    mismatch falls back to the original forward (returns ``False``).
    """
    if not all(hasattr(module, attr) for attr in ("qkv", "proj", "num_heads")):
        return False

    bits = config.resolved_activation_bits
    search = config.scale_search
    cand = config.num_candidates
    cache = config.cache_per_layer
    sym = config.symmetric_activations
    dyn = config.dynamic_activations

    logit_q = FakeQuantizer(bits=bits, symmetric=sym, dynamic=dyn, scale_search=search,
                            num_candidates=cand, cache_per_layer=cache)
    prob_q = FakeQuantizer(bits=bits, symmetric=sym, dynamic=dyn, scale_search=search,
                           num_candidates=cand, cache_per_layer=cache)
    ctx_q = FakeQuantizer(bits=bits, symmetric=sym, dynamic=dyn, scale_search=search,
                          num_candidates=cand, cache_per_layer=cache)
    module.add_module("_foa_quant_logits", logit_q)
    module.add_module("_foa_quant_probs", prob_q)
    module.add_module("_foa_quant_ctx", ctx_q)

    import types

    original_forward = module.forward
    observations: Dict[str, List[torch.Tensor]] = {"logits": [], "probs": [], "ctx": []}

    def quantized_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        try:
            qkv = self.qkv(x)
        except Exception:  # pragma: no cover - defensive
            return original_forward(x)
        if isinstance(qkv, (tuple, list)):
            qkv = qkv[0]
        B, N, C = qkv.shape
        head_dim = C // int(self.num_heads)
        qkv = qkv.reshape(B, N, 3, int(self.num_heads), head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = getattr(self, "scale", head_dim ** -0.5)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, dtype=q.dtype, device=q.device)
        logits = (q @ k.transpose(-2, -1)) * scale
        observations["logits"].append(logits.detach())
        logits = logit_q(logits)
        probs = logits.softmax(dim=-1)
        observations["probs"].append(probs.detach())
        probs = prob_q(probs)
        attn_drop = getattr(self, "attn_drop", None)
        probs = attn_drop(probs) if attn_drop is not None else probs
        ctx = (probs @ v).transpose(1, 2).reshape(B, N, C)
        observations["ctx"].append(ctx.detach())
        ctx = ctx_q(ctx)
        out = self.proj(ctx)
        proj_drop = getattr(self, "proj_drop", None)
        return proj_drop(out) if proj_drop is not None else out

    def finalize_attention(*_args: Any, **_kwargs: Any) -> None:
        for key, quant in (("logits", logit_q), ("probs", prob_q), ("ctx", ctx_q)):
            cached = observations.pop(key, [])
            for tensor in cached[:2]:
                quant.observe(tensor)
            quant.finalize()

    module.forward = types.MethodType(quantized_forward, module)
    module._foa_original_forward = original_forward  # type: ignore[attr-defined]
    module._foa_finalize_attention = finalize_attention  # type: ignore[attr-defined]
    return True


def quantize_attention_modules(model: nn.Module, config: QuantConfig) -> int:
    """Patch all attention modules of a ViT with softmax-aware quantized forwards."""
    if not config.quantize_attention:
        return 0
    patched = 0
    for name, child in _iter_named_modules(model):
        cls_name = type(child).__name__.lower()
        is_attention = "attention" in cls_name or (
            hasattr(child, "qkv") and hasattr(child, "proj") and hasattr(child, "num_heads")
        )
        if not is_attention:
            continue
        if getattr(child, "_foa_finalize_attention", None) is not None:
            continue  # already patched
        try:
            if _patch_attention_forward(child, config):
                patched += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not patch attention %s: %s", name, exc)
    if patched:
        logger.info("Patched %d attention module(s) with quantized attention forward.", patched)
    return patched


def finalize_attention_modules(model: nn.Module) -> None:
    for _name, child in _iter_named_modules(model):
        fin = getattr(child, "_foa_finalize_attention", None)
        if callable(fin):
            try:
                fin()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Attention finalize failed: %s", exc)


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def _to_device(batch: Any, device: torch.device) -> torch.Tensor:
    if isinstance(batch, dict):
        for key in ("image", "images", "x", "input", "pixel_values"):
            if key in batch:
                batch = batch[key]
                break
        else:
            batch = next(iter(batch.values()))
    elif isinstance(batch, (tuple, list)):
        batch = batch[0]
    return batch.to(device)


def _safe_forward(model: nn.Module, images: torch.Tensor) -> Any:
    """Forward helper tolerant of the FOA wrapper, timm, and plain module contracts."""
    if hasattr(model, "forward_with_features"):
        try:
            return model.forward_with_features(images)
        except TypeError:
            return model.forward_with_features(images, prompt=None)
    if hasattr(model, "forward_features"):
        try:
            return model.forward_features(images)
        except TypeError:
            return model.forward_features(images, prompt=None)
    return model(images)


@torch.no_grad()
def calibrate_activations(
    model: nn.Module,
    calibration_batches: Iterable[Any],
    config: Optional[QuantConfig] = None,
    device: Optional[torch.device] = None,
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], Any]] = None,
    verbose: bool = False,
) -> int:
    """Estimate static activation ranges from the calibration samples (paper: 32 images).

    Runs inference over `calibration_batches`, lets every `FakeQuantizer` observe its input
    distribution and then freezes the quantization parameters.  Returns the number of
    calibration images consumed.
    """
    config = config or QuantConfig()
    device = device or _default_device()
    model.eval()
    n_seen = 0
    for batch in calibration_batches:
        images = _to_device(batch, device)
        if forward_fn is not None:
            forward_fn(model, images)
        else:
            _safe_forward(model, images)
        n_seen += int(images.shape[0])
        if verbose:
            logger.info("calibration: %d images seen", n_seen)
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            module.finalize()
    finalize_attention_modules(model)
    logger.info("Activation quantization calibrated on %d image(s).", n_seen)
    return n_seen


def quantize_model(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
    calibration_batches: Optional[Iterable[Any]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> nn.Module:
    """Full post-training quantization pipeline: replace, patch, calibrate, freeze.

    `model` is modified **in place** (FOA is forward-only, so the quantized weights are simply
    frozen buffers) and returned for convenience.
    """
    config = config or QuantConfig()
    config.validate()
    device = device or _default_device()

    if config.quantize_attention:
        quantize_attention_modules(model, config)
    replaced = replace_linear_layers(model, config)
    if verbose:
        logger.info("Quantized %d linear layer(s) to %d-bit (scale search: %s).",
                    len(replaced), config.resolved_weight_bits, config.scale_search)
    for _name, module in replaced:
        module.to(device)

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    if calibration_batches is not None:
        calibrate_activations(model, calibration_batches, config, device=device, verbose=verbose)
    else:
        # No calibration data: fall back to dynamic (per-batch) activation scales.
        logger.warning("No calibration data provided; switching activations to dynamic scaling.")
        for module in model.modules():
            if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                module.act_quant.dynamic = True
    return model


# --------------------------------------------------------------------------------------
# FOA integration
# --------------------------------------------------------------------------------------


def build_calibration_stream(
    cfg: Any = None,
    num_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    seed: int = 0,
    device: Optional[torch.device] = None,
    images_dir: Optional[str] = None,
    split: str = "train",
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Iterator[torch.Tensor]:
    """Iterate over `num_samples` **training** images for PTQ calibration (Appendix B.2).

    Preference order: explicit `images_dir` (ImageFolder) -> `src.data.datasets.build_source_stream`
    (HuggingFace ImageNet-1K) -> error.  Images use the standard ViT 224x224 preprocessing.
    Because Appendix B.2 specifies *randomly selected* training samples, the ImageFolder path
    draws a seeded random subset (the HuggingFace path is driven by `build_source_stream`).
    """
    if images_dir:
        from torchvision import datasets, transforms

        tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        ds = datasets.ImageFolder(images_dir, transform=tf)
        n = min(int(num_samples), len(ds))
        rng = np.random.RandomState(int(seed))
        indices = rng.choice(len(ds), size=n, replace=False) if n < len(ds) else list(range(len(ds)))
        subset = torch.utils.data.Subset(ds, sorted(int(i) for i in indices))
        loader = torch.utils.data.DataLoader(subset, batch_size=min(16, max(1, n)), shuffle=False)
        for images, _labels in loader:
            yield images
        return

    try:
        from ..data.datasets import build_source_stream  # type: ignore
    except Exception:  # pragma: no cover - script-style import fallback
        try:
            from data.datasets import build_source_stream  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Could not locate src/data/datasets.build_source_stream for calibration. "
                "Pass `images_dir` pointing at an ImageFolder of ImageNet-1K training images."
            ) from exc

    try:
        yield from build_source_stream(
            cfg=cfg, num_samples=num_samples, seed=seed, device=device,
            image_size=image_size, split=split,
        )
    except TypeError:
        # `build_source_stream` in this reproduction may not accept a `split` kwarg.
        yield from build_source_stream(
            cfg=cfg, num_samples=num_samples, seed=seed, device=device, image_size=image_size,
        )


def build_quantized_model(
    cfg: Any = None,
    bits: Optional[int] = None,
    model: Optional[nn.Module] = None,
    calibration_batches: Optional[Iterable[Any]] = None,
    config: Optional[QuantConfig] = None,
    device: Optional[torch.device] = None,
    num_calibration: Optional[int] = None,
    calibration_images_dir: Optional[str] = None,
    verbose: bool = True,
    **overrides: Any,
) -> nn.Module:
    """Build (or accept) the frozen ViT-Base wrapper and quantize it with PTQ4ViT-style PTQ.

    The returned object keeps the exact `ViTWithCLSFeatures` interface
    (`forward_with_features`, `forward_features`, `head`, `embed_dim`, `num_layers`), so the
    FOA loop is unchanged - Section 4.2: FOA is applied to the quantized model as-is and never
    calls `backward()`.
    """
    qcfg = config or QuantConfig.from_config(cfg, bits=bits, **overrides)
    device = device or _default_device()

    if model is None:
        try:
            from ..models.vit_loader import build_vit  # type: ignore
        except Exception:  # pragma: no cover
            from models.vit_loader import build_vit  # type: ignore

        model_name = _cfg_get(cfg, "model", "name") or "vit_base_patch16_224"
        checkpoint = _cfg_get(cfg, "model", "checkpoint")
        pretrained = _cfg_get(cfg, "model", "pretrained")
        num_classes = _cfg_get(cfg, "model", "num_classes") or 1000
        model = build_vit(
            model_name=model_name,
            checkpoint=checkpoint,
            pretrained=True if pretrained is None else bool(pretrained),
            num_classes=int(num_classes),
            device=str(device),
        )

    # The official PTQ4ViT path is preferred when available (addendum).
    if calibration_batches is not None:
        official = quantize_with_official_ptq4vit(
            model, qcfg, calibration_batches, device=device
        )
        if official is not None:
            return official
        calibration_batches = list(calibration_batches)

    inner = getattr(model, "model", model)  # quantize the timm backbone, not the wrapper

    if calibration_batches is None:
        n_samples = int(num_calibration or qcfg.calibration_samples)
        calibration_batches = build_calibration_stream(
            cfg=cfg,
            num_samples=n_samples,
            seed=qcfg.seed,
            device=device,
            images_dir=calibration_images_dir,
            split=qcfg.calibration_split,
        )

    quantize_model(inner, qcfg, calibration_batches=calibration_batches,
                   device=device, verbose=verbose)
    if hasattr(model, "eval"):
        model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    logger.info("Quantized FOA backbone ready: %d-bit weights / %d-bit activations.",
                qcfg.resolved_weight_bits, qcfg.resolved_activation_bits)
    return model


def build_quantized_foa(
    cfg: Any = None,
    bits: Optional[int] = None,
    source_stats: Any = None,
    device: Optional[torch.device] = None,
    num_calibration: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build a ready-to-run FOA runner on top of an 8-bit / 6-bit quantized ViT-Base.

    This is the integration point with `src.method.foa.build_foa`: the quantized (frozen)
    model is handed to the ordinary Algorithm 1 loop, which performs forward passes only.
    """
    qcfg = QuantConfig.from_config(cfg, bits=bits)
    device = device or _default_device()
    model = build_quantized_model(
        cfg=cfg, config=qcfg, device=device, num_calibration=num_calibration, **kwargs
    )
    try:
        from ..method.foa import build_foa  # type: ignore
    except Exception:  # pragma: no cover
        from method.foa import build_foa  # type: ignore
    return build_foa(cfg=cfg, source_stats=source_stats, model=model, device=device)


# --------------------------------------------------------------------------------------
# Official PTQ4ViT hook (preferred when the repository is available)
# --------------------------------------------------------------------------------------


def try_load_official_ptq4vit() -> Optional[Dict[str, Any]]:
    """Attempt to import the official PTQ4ViT modules.

    Returns a dict with the imported object when `PTQ4ViT` (or its modules) are on the path,
    otherwise ``None``.  The official repo is used as-is when present (the addendum notes the
    complete quantization process lives there), while the self-contained emulation above is
    the fallback for a self-contained reproduction.
    """
    import importlib

    candidates = [
        ("quant.quant_model", "quant_model"),
        ("ptq.ptq", "ptq"),
        ("ptq4vit.quant_model", "quant_model"),
        ("quant_model", "quant_model"),
    ]
    for mod_name, attr in candidates:
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        obj = getattr(module, attr, module)
        logger.info("Found official PTQ4ViT module: %s", mod_name)
        return {"module": obj, "name": mod_name}
    logger.info(
        "Official PTQ4ViT not importable; using the self-contained twin-uniform PTQ emulation "
        "(%s).", PTQ4VIT_REPO,
    )
    return None


def quantize_with_official_ptq4vit(
    model: nn.Module,
    config: QuantConfig,
    calibration_batches: Iterable[Any],
    device: Optional[torch.device] = None,
) -> Optional[nn.Module]:
    """Use the official PTQ4ViT quantizer if available; otherwise return ``None``."""
    loaded = try_load_official_ptq4vit()
    if loaded is None:
        return None
    try:  # pragma: no cover - depends on the external repository API
        inner = getattr(model, "model", model)
        obj = loaded["module"]
        qmodel = obj.quant_model(inner, w_bits=config.resolved_weight_bits,
                                 a_bits=config.resolved_activation_bits)
        calib = getattr(obj, "calib", None)
        if calib is not None:
            calib(qmodel, calibration_batches, device=device)
        return model
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Official PTQ4ViT path failed (%s); falling back to the emulation.", exc)
        return None


# --------------------------------------------------------------------------------------
# Runner helpers / CLI (Table 4 / Table 17)
# --------------------------------------------------------------------------------------


def _import_run_foa():
    """Import `scripts/run_foa` regardless of the invocation style."""
    import importlib

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    last_exc: Optional[Exception] = None
    for name in ("scripts.run_foa", "run_foa"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on cwd
            last_exc = exc
    raise ImportError(f"Could not import scripts/run_foa.py ({last_exc})")


def _metrics_class_subset(cfg: Any) -> Optional[List[int]]:
    n_eval = _cfg_get(cfg, "data", "num_classes_eval")
    n_cls = _cfg_get(cfg, "model", "num_classes")
    if n_eval is not None and n_cls is not None and int(n_eval) != int(n_cls):
        return list(range(int(n_eval)))
    return None


def run_quantized(
    cfg: Any,
    bits: int = 8,
    method: str = "foa",
    corruption: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_calibration: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_images_dir: Optional[str] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run FOA (or the NoAdapt reference) on the quantized backbone for one stream."""
    device = device or _default_device()
    run_foa_mod = _import_run_foa()

    if method == "noadapt":
        # NoAdapt needs no source statistics on a quantized model; build and predict.
        model = build_quantized_model(
            cfg=cfg, bits=bits, device=device, num_calibration=num_calibration,
            calibration_images_dir=calibration_images_dir, verbose=verbose,
        )
        loader = run_foa_mod.build_test_loader(cfg, dataset=None, corruption=corruption,
                                               limit_batches=limit_batches)
        accumulator = run_foa_mod.ResultAccumulator(
            ece_bins=int(_cfg_get(cfg, "eval", "ece_bins") or 15)
        )
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images, targets = run_foa_mod.unpack_batch(batch) if hasattr(run_foa_mod, "unpack_batch") else (None, None)
                if images is None:
                    images = _to_device(batch, device)
                    targets = None
                images = images.to(device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                out = _safe_forward(model, images)
                logits = out["logits"] if isinstance(out, dict) else (
                    out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
                )
                if targets is not None:
                    accumulator.update(logits, targets)
        summary = accumulator.compute()
        return {
            "bits": bits, "method": "noadapt", "corruption": corruption,
            "accuracy": summary.get("accuracy"), "ece": summary.get("ece"),
            "num_samples": summary.get("num_samples"),
        }

    qcfg = QuantConfig.from_config(cfg, bits=bits)
    model = build_quantized_model(
        cfg=cfg, config=qcfg, device=device, num_calibration=num_calibration,
        calibration_images_dir=calibration_images_dir, verbose=verbose,
    )
    loader = run_foa_mod.build_test_loader(cfg, dataset=None, corruption=corruption,
                                           limit_batches=limit_batches)

    try:
        from ..method.foa import build_foa  # type: ignore
    except Exception:  # pragma: no cover
        from method.foa import build_foa  # type: ignore

    runner = build_foa(cfg=cfg, model=model, device=device)
    out = runner.run(loader, verbose=verbose)
    return {
        "bits": bits, "method": "foa", "corruption": corruption,
        "accuracy": out.get("accuracy"), "ece": out.get("ece"),
        "num_samples": out.get("num_samples"), "per_batch": out.get("per_batch"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOA on quantized ViT-Base (PTQ4ViT 8-bit / 6-bit) - Table 4 / Table 17."
    )
    parser.add_argument("--config", default="configs/foa_quantized.yaml")
    parser.add_argument("--extra-config", default=None)
    parser.add_argument("--bits", type=int, choices=list(SUPPORTED_BITS), default=None,
                        help="bit-width; default: run both 8-bit and 6-bit as in Table 4.")
    parser.add_argument("--num-calibration", type=int, default=DEFAULT_CALIBRATION_SAMPLES,
                        help="random training samples for PTQ4ViT calibration (paper: 32).")
    parser.add_argument("--calibration-images-dir", default=None,
                        help="ImageFolder with ImageNet-1K training images (calibration).")
    parser.add_argument("--method", default="foa", choices=["foa", "noadapt"])
    parser.add_argument("--corruptions", nargs="*", default=None)
    parser.add_argument("--all-corruptions", action="store_true",
                        help="run all 15 ImageNet-C corruptions and average.")
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source-stats", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.utils.config import load_config, save_config, config_to_dict
    except Exception:  # pragma: no cover
        print("Could not import src.utils.config; run this script from the repository root.")
        return 2

    paths = [p for p in (args.config, args.extra_config) if p]
    cfg = load_config(*paths)
    if args.severity is not None:
        cfg.setdefault("data", {})["severity"] = args.severity
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.source_stats:
        cfg.setdefault("source_stats", {})["path"] = args.source_stats
    set_seed(int(_cfg_get(cfg, "seed") or 0))

    bits_list = [args.bits] if args.bits else list(SUPPORTED_BITS)
    if args.all_corruptions:
        run_foa_mod = _import_run_foa()
        corruptions: List[Optional[str]] = list(run_foa_mod.IMAGENET_C_CORRUPTIONS)
    else:
        corruptions = list(args.corruptions) if args.corruptions else [None]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results: Dict[str, Any] = {}
    for bits in bits_list:
        per_corruption: Dict[str, Dict[str, Any]] = {}
        for corruption in corruptions:
            t0 = time.time()
            res = run_quantized(
                cfg, bits=bits, method=args.method, corruption=corruption, device=device,
                num_calibration=args.num_calibration,
                calibration_images_dir=args.calibration_images_dir,
                limit_batches=args.limit_batches, verbose=not args.quiet,
            )
            res["wall_clock_s"] = time.time() - t0
            per_corruption[corruption or "single"] = res
            logger.info("%d-bit %s %s -> acc=%s ECE=%s", bits, args.method,
                        corruption or "-", res.get("accuracy"), res.get("ece"))
        key = f"{bits}bit"
        results[key] = {
            "bits": bits,
            "method": args.method,
            "per_corruption": per_corruption,
            **(_average_metrics(per_corruption) if len(per_corruption) > 1 else {}),
        }

    reference = TABLE4_REFERENCE
    payload = {
        "config": config_to_dict(cfg),
        "args": vars(args),
        "reference": reference,
        "results": results,
    }
    out_path = args.output or os.path.join(
        _cfg_get(cfg, "output_dir") or "./outputs", "quantized_results.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        save_config(_jsonable(payload), out_path)
    except Exception:
        import json

        with open(out_path, "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2)
    print(f"Results written to {out_path}")
    return 0


def _average_metrics(per_corruption: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    accs = [v.get("accuracy") for v in per_corruption.values() if v.get("accuracy") is not None]
    eces = [v.get("ece") for v in per_corruption.values() if v.get("ece") is not None]
    return {
        "accuracy": float(np.mean(accs)) if accs else None,
        "ece": float(np.mean(eces)) if eces else None,
        "num_corruptions": len(per_corruption),
    }


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup that works for dict-like and attribute-like configs."""
    if cfg is None:
        return default
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return default if cur is None else cur


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def set_seed(seed: int = 0) -> None:
    """Seed python/numpy/torch (calibration samples must be reproducible per Appendix B.2)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "QuantConfig",
    "FakeQuantizer",
    "QuantizedLinear",
    "QuantizedConv2d",
    "quantize_tensor",
    "fake_quantize_ste",
    "select_qparams_minmax",
    "select_qparams_twin_uniform",
    "twin_uniform_search",
    "replace_linear_layers",
    "quantize_attention_modules",
    "finalize_attention_modules",
    "calibrate_activations",
    "quantize_model",
    "build_calibration_stream",
    "build_quantized_model",
    "build_quantized_foa",
    "try_load_official_ptq4vit",
    "quantize_with_official_ptq4vit",
    "run_quantized",
    "parse_args",
    "main",
    "set_seed",
    "SUPPORTED_BITS",
    "DEFAULT_CALIBRATION_SAMPLES",
    "PTQ4VIT_REPO",
    "TABLE4_REFERENCE",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
