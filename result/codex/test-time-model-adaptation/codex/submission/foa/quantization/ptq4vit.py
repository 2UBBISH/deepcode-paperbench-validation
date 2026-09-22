"""PTQ4ViT-style post-training quantisation of a ViT (Section 4.2 of the paper).

The paper adopts PTQ4ViT (Yuan et al., ECCV 2022) to obtain the 8-bit and 6-bit ViT-Base
models and calibrates it with 32 randomly selected samples from the training set.  The
main text of the FOA paper does not repeat the details of PTQ4ViT, so we follow the
original paper:

1. every MatMul input (Q, K, V, the attention logits, the attention output, the MLP
   inputs) and every weight is quantised with a *uniform* quantiser;
2. the activations whose distribution is extremely unbalanced - the post-Softmax and the
   post-GELU activations - use the *twin uniform quantisation* of PTQ4ViT (separate
   scaling factor for the negative and the positive part);
3. the clipping thresholds are determined by a greedy *layer-wise proxy quantisation*
   search: each quantiser searches a grid of clipping ratios around the observed range
   and keeps the value that minimises the reconstruction error of its layer;
4. LayerNorm / residual additions stay in full precision.

Quantisation is implemented as *fake quantisation* (round to the integer grid and
de-quantise), which is numerically equivalent to the quantised model while keeping the
forward-only FOA loop unchanged - FOA never needs integer kernels nor gradients, so the
exact same ``forward_tokens`` interface works for the 32-bit and the 8-/6-bit model.
"""
from __future__ import annotations

from types import MethodType
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import (
    TwinUniformQuantizer,
    UniformQuantizer,
    iter_quantizers,
    make_quantizer,
)


# --------------------------------------------------------------------------------------
# quantised building blocks
# --------------------------------------------------------------------------------------
class QuantLinear(nn.Module):
    """``nn.Linear`` with a quantised weight and a quantised input activation."""

    def __init__(
        self,
        linear: nn.Linear,
        bits: int = 8,
        per_channel_weight: bool = False,
        twin_input: bool = False,
        percentile: float = 0.999,
        search_ratios=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    ) -> None:
        super().__init__()
        self.linear = linear
        self.weight_quant = make_quantizer(
            bits,
            per_channel=per_channel_weight,
            channel_axis=0 if per_channel_weight else None,
            percentile=percentile,
            search_ratios=search_ratios,
        )
        self.act_quant = make_quantizer(
            bits, twin=twin_input, percentile=percentile, search_ratios=search_ratios
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calling the quantiser modules (instead of their ``quantize`` method) keeps the
        # calibration hooks alive
        w = self.weight_quant(self.linear.weight)
        xq = self.act_quant(x)
        return F.linear(xq, w, self.linear.bias)


class QuantMatMul(nn.Module):
    """MatMul with quantised operands (``@`` keeps the batch / head dimensions)."""

    def __init__(self, bits: int = 8, twin_left: bool = False, twin_right: bool = False):
        super().__init__()
        self.left_quant = make_quantizer(bits, twin=twin_left)
        self.right_quant = make_quantizer(bits, twin=twin_right)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.left_quant(a) @ self.right_quant(b)


class QuantizedAttention(nn.Module):
    """Drop-in replacement of timm's ``Attention`` with PTQ4ViT quantisers inserted."""

    def __init__(self, attn: nn.Module, bits: int = 8, per_channel_weight: bool = False):
        super().__init__()
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.attn_dim = attn.attn_dim
        self.scale = attn.scale
        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.gate = getattr(attn, "gate", None)
        self.norm = attn.norm
        # the original projections are *shared*, so the source weights are preserved
        self.qkv = QuantLinear(attn.qkv, bits=bits, per_channel_weight=per_channel_weight)
        self.proj = QuantLinear(attn.proj, bits=bits, per_channel_weight=per_channel_weight)
        self.qk_matmul = QuantMatMul(bits)
        self.softmax_quant = make_quantizer(bits, twin=True)   # post-Softmax
        self.pv_matmul = QuantMatMul(bits)
        self._attn_impl = attn

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        q = q * self.scale
        attn = self.qk_matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.softmax_quant(attn)
        x = self.pv_matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuantizedMlp(nn.Module):
    """Drop-in replacement of timm's ``Mlp`` with PTQ4ViT quantisers inserted."""

    def __init__(self, mlp: nn.Module, bits: int = 8, per_channel_weight: bool = False):
        super().__init__()
        self.fc1 = QuantLinear(mlp.fc1, bits=bits, per_channel_weight=per_channel_weight)
        self.act = mlp.act
        self.drop1 = mlp.drop1
        self.norm = mlp.norm
        self.gelu_quant = make_quantizer(bits, twin=True)      # post-GELU
        self.fc2 = QuantLinear(mlp.fc2, bits=bits, per_channel_weight=per_channel_weight)
        self.drop2 = mlp.drop2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.gelu_quant(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --------------------------------------------------------------------------------------
# model surgery
# --------------------------------------------------------------------------------------
def quantize_vit(vit: nn.Module, bits: int = 8, per_channel_weight: bool = False) -> nn.Module:
    """Replace every ``Attention``/``Mlp`` submodule of a timm ViT in place.

    The original parameters are shared (not copied), so the quantised model only adds the
    quantiser buffers.
    """
    for blk in vit.blocks:
        blk.attn = QuantizedAttention(blk.attn, bits=bits, per_channel_weight=per_channel_weight)
        blk.mlp = QuantizedMlp(blk.mlp, bits=bits, per_channel_weight=per_channel_weight)
    vit.is_quantized = True
    vit.quant_bits = bits
    return vit


def _rebuild_mlp(mlp: QuantizedMlp) -> nn.Module:
    container = nn.Module()
    container.fc1 = mlp.fc1.linear
    container.act = mlp.act
    container.drop1 = mlp.drop1
    container.norm = mlp.norm
    container.fc2 = mlp.fc2.linear
    container.drop2 = mlp.drop2

    def _forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)

    container.forward = MethodType(_forward, container)
    return container


def dequantize_vit(vit: nn.Module) -> nn.Module:
    """Undo :func:`quantize_vit` so the full-precision model can be evaluated again."""
    for blk in vit.blocks:
        attn = getattr(blk.attn, "_attn_impl", None)
        if attn is not None:
            blk.attn = attn
        if isinstance(blk.mlp, QuantizedMlp):
            blk.mlp = _rebuild_mlp(blk.mlp)
    vit.is_quantized = False
    return vit


def set_quantization_enabled(vit: nn.Module, enabled: bool) -> None:
    for q in iter_quantizers(vit):
        q.enabled = enabled


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_observations(model: nn.Module, batches: Iterable[torch.Tensor]) -> None:
    """Run the model and record the range seen by every quantiser."""
    handles = []
    for q in iter_quantizers(model):
        q.observed_min = None
        q.observed_max = None
        handles.append(q.register_forward_hook(lambda m, inp, out: m.observe(inp[0])))
    was_training = model.training
    model.eval()
    for images in batches:
        model(images)
    for h in handles:
        h.remove()
    model.train(was_training)


@torch.no_grad()
def init_scales(model: nn.Module) -> None:
    for q in iter_quantizers(model):
        q.init_from_observations()


@torch.no_grad()
def search_clipping_thresholds(
    model: nn.Module,
    batches: Sequence[torch.Tensor],
) -> Dict[str, float]:
    """Greedy search of the clipping ratio of every quantiser.

    For each quantiser the ratio of ``cfg.search_ratios`` that minimises the MSE between
    the quantised and the reference activation is selected.  This is the layer-wise proxy
    quantisation metric of PTQ4ViT (the original paper additionally weights the
    reconstruction error by a Hessian estimate; the FOA paper does not restate PTQ4ViT's
    details, see the addendum).
    """
    reference: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook():
        def _hook(module, inp, out):
            if id(module) not in reference:
                reference[id(module)] = inp[0].detach().float().clone()
        return _hook

    quantizers = list(iter_quantizers(model))
    for q in quantizers:
        handles.append(q.register_forward_hook(make_hook()))
    was_training = model.training
    model.eval()
    for images in batches:
        model(images)
    for h in handles:
        h.remove()
    model.train(was_training)

    chosen: Dict[str, float] = {}
    for idx, q in enumerate(quantizers):
        x = reference.get(id(q))
        if x is None:
            continue
        best_ratio, best_err = 1.0, float("inf")
        for ratio in q.cfg.search_ratios:
            q.set_ratio(ratio)
            err = float(F.mse_loss(q.quantize(x), x).item())
            if err < best_err:
                best_err, best_ratio = err, ratio
        q.set_ratio(best_ratio)
        chosen[f"{type(q).__name__}_{idx}"] = float(best_ratio)
    return chosen


@torch.no_grad()
def calibrate(
    model: nn.Module,
    calibration_batches: Sequence[torch.Tensor],
    search: bool = True,
) -> Dict[str, float]:
    """Full PTQ4ViT calibration: observe ranges -> initialise scales -> grid search."""
    set_quantization_enabled(model, False)
    collect_observations(model, calibration_batches)
    set_quantization_enabled(model, True)
    init_scales(model)
    ratios: Dict[str, float] = {}
    if search and calibration_batches:
        ratios = search_clipping_thresholds(model, calibration_batches[:1])
    return ratios


def quantized_parameter_bytes(model: nn.Module) -> int:
    """Size of the quantised weights in bytes (used for the memory estimates of the paper)."""
    bits = int(getattr(model, "quant_bits", 8))
    total = 0
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * bits // 8
    return total


def ideal_memory_ratio(bits: int) -> float:
    """The paper estimates the memory of a quantised model as ``(bits / 32) x`` that of the
    32-bit model (Liu et al., 2021b)."""
    return bits / 32.0
