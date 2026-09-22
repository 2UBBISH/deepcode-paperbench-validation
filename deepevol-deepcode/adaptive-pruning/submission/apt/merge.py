"""Merge and physical pruning for APT (Sec. 4.1 footnote 2, Sec. 6).

After APT training both the tuning parameters and the pruning masks must vanish
from the deployed model:

* the LoRA term ``s * W_B W_A`` is folded into the frozen weight ``W`` so that
  *"we do not count tuning parameters since they can be fully merged after
  training"* (Sec. 4.1, footnote 2), i.e. APT adds **no inference overhead**;
* the pruned MHA heads, FFN neurons and hidden dimensions are physically
  removed, so the returned model is a plain HuggingFace model whose attention /
  FFN shapes are the retained ones (Sec. 6: *"we reset the optimizer every time
  after each parameter size changes"* -- the size change is materialised here).

Paper references
----------------
* Eq. (1) ``H_apt(X) = m_o o (W + s W_B W_A) X o m_i`` -- what has to be folded.
* Eq. (4) ``C(Theta_t; M_t) ~ d_m sum_i (4 n_h^i d_h + 2 n_f^i)`` -- the parameter
  count of the retained model, used to verify the realised sparsity.
* Appendix C: ``C_head``, ``C_neuron``, ``C_dimension`` reference costs for
  RoBERTa-base (196608 / 1536 / 110592) and the note that gated FFNs (T5) count
  three linear layers and that the T5 decoder cross-attention layers count as
  well.

Index-map scheme (documented default)
-------------------------------------
The residual stream position ``p`` of a BERT-like/T5 model is canonically
written ``p = h * d_h + r`` (head-major, ``d_h = d_model / n_heads``).  Given a
retained head set ``H`` and a retained hidden-dimension set ``D`` we build a
single consistent index map:

1. ``slots(h) = sorted(r for p in D if p // d_h == h for r = p % d_h)`` -- the
   residual positions retained *inside* head ``h``.
2. ``k = min_h |slots(h)|`` (uniform, required to instantiate a HF model with a
   scalar ``num_attention_heads``); every head keeps the first ``k`` slots.
3. new width ``d_m' = k * |H|`` and the new-to-old hidden index map is
   ``[h * d_h + slots(h)[s] for h in H for s in range(k)]``.

With no dimension pruning (``D`` = all ones) this degenerates to the trivial
map and only heads shrink; with no head pruning only the width shrinks.  Owing
to the uniform requirement the realised sparsity is always **at least** the
requested one, never less.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:  # pragma: no cover - constants are re-defined below if the import fails
    from .adapters import BLOCK_TYPES, DIMENSION, HEAD, NEURON, iter_masked_linears
except Exception:  # pragma: no cover
    HEAD, NEURON, DIMENSION = 0, 1, 2
    BLOCK_TYPES = {0: "head", 1: "neuron", 2: "dimension"}

    def iter_masked_linears(module, names=None):  # type: ignore
        for name, child in module.named_modules():
            if hasattr(child, "merged_weight") and hasattr(child, "adapter"):
                if names is None or name in names:
                    yield name, child


__all__ = [
    "PrunePlan",
    "build_prune_plan",
    "plan_from_mask_state",
    "plan_to_dict",
    "plan_from_dict",
    "merge_adapters",
    "merge_masks_into_weights",
    "harden_masks",
    "physical_prune",
    "merge_and_prune",
    "restore_dense_model",
    "verify_merge_equivalence",
    "count_parameters",
    "pruned_param_count",
    "model_size_mb",
    "parameter_breakdown",
    "removed_blocks",
    "save_pruned_model",
    "save_plan",
    "load_plan",
    "HEAD",
    "NEURON",
    "DIMENSION",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _to_bool_list(values, threshold: float) -> List[bool]:
    """Coerce a mask tensor / list into a list of booleans."""
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        values = values.detach().to(torch.float32).reshape(-1).tolist()
    return [float(v) > float(threshold) for v in values]


def _to_float_list(values) -> List[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [float(v) for v in values.detach().to(torch.float32).reshape(-1).tolist()]
    return [float(v) for v in values]


def _sites_of(module) -> str:
    """Return the attention site of a masked linear ('self' or 'cross')."""
    name = str(getattr(module, "module_name", "") or "").lower()
    if "cross" in name:
        return "cross"
    return "self"


def _layer_of(module, default: int = 0) -> int:
    idx = getattr(module, "layer_idx", None)
    if idx is None or int(idx) < 0:
        return default
    return int(idx)


def _group_mask(module) -> Optional[torch.Tensor]:
    """Best-effort retrieval of the per-group output mask of a masked linear."""
    getter = getattr(module, "get_output_group_mask", None)
    if callable(getter):
        try:
            value = getter()
            if value is not None:
                return value
        except Exception:
            pass
    for attr in ("mask_out", "output_mask"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _input_mask(module) -> Optional[torch.Tensor]:
    for attr in ("mask_in", "input_mask"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _base_weight(module) -> Optional[torch.Tensor]:
    for attr in ("base_weight", "weight"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _unwrap(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and getattr(model, "_apt_wrapper", False):
        return inner
    if isinstance(inner, nn.Module) and not isinstance(inner, (nn.Linear, nn.Embedding)):
        # APTModelWrapper stores the HF model as ``.model``
        return inner
    return model


def _layer_stack(name: str) -> Tuple[str, int]:
    """Extract ``(stack, layer_index)`` from a parameter name."""
    layer = -1
    match = re.search(r"\.layers?\.(\d+)\.", name)
    if match:
        layer = int(match.group(1))
    else:
        match = re.search(r"\.layer\.(\d+)\.", name)  # T5 uses ``layer.0/1``
        if match is None:
            match = re.search(r"\.blocks?\.(\d+)\.", name)
        if match is not None:
            # T5: block index lives in ``.block.N.``
            block = re.search(r"\.block\.(\d+)\.", name)
            layer = int(block.group(1)) if block else int(match.group(1))
    stack = "dec" if ("decoder" in name.split(".")[:3]) else "enc"
    if "cross" in name.lower() or "encdecattention" in name.lower():
        stack = "cross"
    return stack, layer


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of (optionally trainable) parameters of ``model``."""
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())
    return total


def pruned_param_count(model: nn.Module) -> int:
    """Alias of :func:`count_parameters` (Eq. (4) numerator of the sparsity)."""
    return count_parameters(model)


def model_size_mb(model: nn.Module, dtype_bytes: Optional[int] = None) -> float:
    """Approximate parameter memory of ``model`` in MB."""
    bytes_total = 0
    for param in model.parameters():
        if dtype_bytes is None:
            bytes_total += int(param.numel()) * int(param.element_size())
        else:
            bytes_total += int(param.numel()) * int(dtype_bytes)
    return bytes_total / (1024.0 ** 2)


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------
@dataclass
class PrunePlan:
    """Retained structure of an APT-pruned model.

    Fields mirror the three block families of Sec. 4.2 / Appendix C:
    MHA heads (``m_o`` grouped per head), FFN neurons (``m_o`` per neuron) and
    hidden dimensions (``m_i``).  ``head_keep`` / ``neuron_keep`` are keyed by
    ``"<site>:<layer>"`` and ``"<layer>"`` respectively.
    """

    d_model: int
    n_heads: int = 0
    d_h: int = 0
    n_layers: int = 0
    dim_keep: List[bool] = field(default_factory=list)
    head_keep: Dict[str, List[bool]] = field(default_factory=dict)
    neuron_keep: Dict[str, List[bool]] = field(default_factory=dict)
    ffn_linear_count: int = 2
    attn_linear_count: int = 4
    threshold: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dim_keep:
            self.dim_keep = [True] * int(self.d_model)
        if not self.d_h and self.n_heads:
            self.d_h = int(self.d_model) // int(self.n_heads)

    # ---- derived ---------------------------------------------------------------
    def head_keys(self) -> List[str]:
        return sorted(self.head_keep.keys(), key=_key_sort)

    def neuron_keys(self) -> List[str]:
        return sorted(self.neuron_keep.keys(), key=_key_sort)

    def kept_dims(self) -> List[int]:
        return [i for i, keep in enumerate(self.dim_keep) if keep]

    def kept_heads(self, key: str) -> List[int]:
        return [i for i, keep in enumerate(self.head_keep.get(key, [])) if keep]

    def kept_neurons(self, key: str) -> List[int]:
        return [i for i, keep in enumerate(self.neuron_keep.get(key, [])) if keep]

    @property
    def n_heads_kept(self) -> int:
        counts = [len(self.kept_heads(k)) for k in self.head_keep]
        return min(counts) if counts else int(self.n_heads)

    @property
    def n_ffn_kept(self) -> int:
        counts = [len(self.kept_neurons(k)) for k in self.neuron_keep]
        return min(counts) if counts else 0

    def cell_counts(self) -> Dict[str, int]:
        return {
            "d_model": int(self.d_model),
            "d_model_kept": len(self.kept_dims()),
            "n_heads": int(self.n_heads),
            "n_heads_kept": int(self.n_heads_kept),
            "n_ffn_kept": int(self.n_ffn_kept),
            "n_layers": int(self.n_layers),
        }

    # ---- index maps ------------------------------------------------------------
    def index_maps(self, require_grid: bool = True) -> Dict[str, Any]:
        """Build the new-to-old index maps described in the module docstring.

        Returns a dict with ``dim_map`` (new hidden index -> old hidden index),
        ``head_rows[key]`` (old q/k/v rows of the retained heads),
        ``neuron_idx[key]``, ``head_idx[key]``, ``d_h`` and the realised sizes.
        """
        d_model = int(self.d_model)
        n_heads = int(self.n_heads) if self.n_heads else 0
        d_h = int(self.d_h) if self.d_h else (d_model // n_heads if n_heads else d_model)

        dim_keep = list(self.dim_keep) or [True] * d_model
        if len(dim_keep) != d_model:  # tolerate off-by-one masks
            dim_keep = (dim_keep + [True] * d_model)[:d_model]
        kept_dims = [i for i, keep in enumerate(dim_keep) if keep]

        if not n_heads or d_h <= 0:
            # dimension-only pruning: no attention grid to keep consistent
            maps = {
                "dim_map": kept_dims,
                "head_rows": {},
                "head_idx": {},
                "neuron_idx": {k: self.kept_neurons(k) for k in self.neuron_keep},
                "d_h": d_h,
                "d_model": len(kept_dims),
                "n_heads": max(n_heads, 1) if n_heads else 1,
                "n_ffn": self.n_ffn_kept,
                "kept_dims": kept_dims,
                "dropped_heads": {},
            }
            return maps

        dim_set = set(kept_dims)
        n_heads_new = self.n_heads_kept if self.head_keep else n_heads
        maps_head_rows: Dict[str, List[int]] = {}
        maps_head_idx: Dict[str, List[int]] = {}
        dropped: Dict[str, List[int]] = {}
        slots_per_head: Dict[str, Dict[int, List[int]]] = {}

        keys = self.head_keys() or ["self:0"]
        for key in keys:
            kept = self.kept_heads(key)
            if not kept and self.n_heads:
                kept = list(range(n_heads))
            kept = kept[:n_heads_new]
            slots: Dict[int, List[int]] = {}
            for h in kept:
                rows = [h * d_h + r for r in range(d_h) if (h * d_h + r) in dim_set]
                slots[h] = rows
            slots_per_head[key] = slots

        # uniform within-head width across all heads / keys
        widths = [len(rows) for key in slots_per_head for rows in slots_per_head[key].values()]
        k = min(widths) if widths else d_h
        if require_grid:
            k = max(k, 1)
        if not require_grid and k <= 0:
            k = d_h

        dim_map: List[int] = []
        for key in keys:
            kept = [h for h in slots_per_head[key] if len(slots_per_head[key][h]) >= k]
            dropped[key] = [h for h in slots_per_head[key] if len(slots_per_head[key][h]) < k]
            rows: List[int] = []
            idx: List[int] = []
            for position, h in enumerate(kept):
                slots = slots_per_head[key][h][:k]
                rows.extend(slots)
                idx.append(h)
            maps_head_rows[key] = rows
            maps_head_idx[key] = idx
            if not dim_map:  # the residual map is shared across all layers
                for h in kept:
                    slots = slots_per_head[key][h][:k]
                    dim_map.extend(slots)

        if not dim_map:
            dim_map = kept_dims

        n_heads_final = min((len(v) for v in maps_head_idx.values()), default=n_heads_new)
        if n_heads_final <= 0:
            n_heads_final = 1
        # truncate keys that kept more heads than the uniform target
        for key in list(maps_head_rows):
            idx = maps_head_idx[key][:n_heads_final]
            maps_head_idx[key] = idx
            rows: List[int] = []
            for h in idx:
                rows.extend(slots_per_head[key][h][:k])
            maps_head_rows[key] = rows

        n_layers = max(1, int(self.n_layers) or 1)
        dim_map = dim_map[: k * n_heads_final] if n_heads_final else dim_map
        if len(dim_map) != k * n_heads_final:
            # keep the map self-consistent even with heterogeneous masks
            dim_map = (dim_map + list(range(k * n_heads_final)))[: k * n_heads_final]

        return {
            "dim_map": dim_map,
            "head_rows": maps_head_rows,
            "head_idx": maps_head_idx,
            "neuron_idx": {k2: self.kept_neurons(k2) for k2 in self.neuron_keep},
            "d_h": int(k),
            "d_model": int(len(dim_map)),
            "n_heads": int(n_heads_final),
            "n_ffn": int(self.n_ffn_kept),
            "kept_dims": kept_dims,
            "dropped_heads": dropped,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "d_h": int(self.d_h),
            "n_layers": int(self.n_layers),
            "dim_keep": [bool(v) for v in self.dim_keep],
            "head_keep": {k: [bool(v) for v in v2] for k, v2 in self.head_keep.items()},
            "neuron_keep": {k: [bool(v) for v in v2] for k, v2 in self.neuron_keep.items()},
            "ffn_linear_count": int(self.ffn_linear_count),
            "attn_linear_count": int(self.attn_linear_count),
            "threshold": float(self.threshold),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrunePlan":
        return cls(
            d_model=int(data["d_model"]),
            n_heads=int(data.get("n_heads", 0)),
            d_h=int(data.get("d_h", 0)),
            n_layers=int(data.get("n_layers", 0)),
            dim_keep=[bool(v) for v in data.get("dim_keep", [])],
            head_keep={k: [bool(v) for v in v2] for k, v2 in data.get("head_keep", {}).items()},
            neuron_keep={
                k: [bool(v) for v in v2] for k, v2 in data.get("neuron_keep", {}).items()
            },
            ffn_linear_count=int(data.get("ffn_linear_count", 2)),
            attn_linear_count=int(data.get("attn_linear_count", 4)),
            threshold=float(data.get("threshold", 0.5)),
            meta=dict(data.get("meta", {})),
        )

    def summary(self) -> str:
        counts = self.cell_counts()
        return (
            "PrunePlan(d_model {d_model}->{d_model_kept}, n_heads {n_heads}->{n_heads_kept}, "
            "n_ffn->{n_ffn_kept}, layers {n_layers})".format(**counts)
        )


def _key_sort(key: str) -> Tuple[int, int, str]:
    try:
        layer = int(re.findall(r"\d+", str(key))[-1])
    except Exception:
        layer = 0
    return (layer, 0, str(key))


def plan_to_dict(plan: PrunePlan) -> Dict[str, Any]:
    return plan.as_dict()


def plan_from_dict(data: Dict[str, Any]) -> PrunePlan:
    return PrunePlan.from_dict(data)


def save_plan(plan: PrunePlan, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan_to_dict(plan), handle, indent=2)
    return path


def load_plan(path: str) -> PrunePlan:
    with open(path, "r", encoding="utf-8") as handle:
        return plan_from_dict(json.load(handle))


# --------------------------------------------------------------------------------------
# plan construction from masks
# --------------------------------------------------------------------------------------
def _model_shape_from(model: nn.Module) -> Dict[str, int]:
    config = getattr(model, "config", None)
    if config is None:
        inner = _unwrap(model)
        config = getattr(inner, "config", None)
    d_model = int(getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0) or 0)
    n_heads = int(getattr(config, "num_attention_heads", 0) or getattr(config, "num_heads", 0) or 0)
    n_layers = int(
        getattr(config, "num_hidden_layers", 0)
        or getattr(config, "num_layers", 0)
        or getattr(config, "n_layers", 0)
        or 0
    )
    n_ffn = int(
        getattr(config, "intermediate_size", 0) or getattr(config, "d_ff", 0) or 0
    )
    ffn_layers = 3 if getattr(config, "is_gated_act", False) or _is_gated_config(config) else 2
    attn_layers = 4
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_ffn": n_ffn,
        "ffn_linear_count": ffn_layers,
        "attn_linear_count": attn_layers,
    }


def _is_gated_config(config) -> bool:
    for attr in ("feed_forward_proj", "dense_act_fn", "hidden_act"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and ("gated" in value.lower() or "gelu_new" == value.lower()):
            if "gated" in value.lower():
                return True
    return bool(getattr(config, "use_gated_ffn", False))


def build_prune_plan(
    model: nn.Module,
    threshold: float = 0.5,
    dim_threshold: Optional[float] = None,
    require_grid: bool = True,
    n_layers: Optional[int] = None,
) -> PrunePlan:
    """Derive the retained structure from the masks of a wrapped APT model.

    Head masks (``m_o`` grouped per head) and neuron masks (``m_o`` per neuron)
    are read from the ``MaskedLinear`` modules; the hidden-dimension mask
    (``m_i``) is shared across the model and therefore read once.
    """
    dim_threshold = threshold if dim_threshold is None else float(dim_threshold)
    inner = _unwrap(model)
    shape = _model_shape_from(inner)

    head_values: Dict[str, torch.Tensor] = {}
    neuron_values: Dict[str, torch.Tensor] = {}
    dim_values: Optional[torch.Tensor] = None

    for name, module in iter_masked_linears(model):
        kind = getattr(module, "kind", None)
        if kind is None:
            continue
        site = _sites_of(module)
        layer = _layer_of(module)
        key = f"{site}:{layer}"
        if int(kind) == int(DIMENSION):
            mask = _input_mask(module)
            if mask is not None and dim_values is None:
                dim_values = mask.detach().to(torch.float32).clone()
            continue
        grouped = _group_mask(module)
        if grouped is None:
            continue
        grouped = grouped.detach().to(torch.float32).reshape(-1)
        if int(kind) == int(HEAD):
            prev = head_values.get(key)
            if prev is None or prev.numel() != grouped.numel():
                head_values[key] = grouped.clone()
            else:
                head_values[key] = torch.minimum(prev, grouped)
        elif int(kind) == int(NEURON):
            prev = neuron_values.get(key)
            if prev is None:
                neuron_values[key] = grouped.clone()
    # if no dedicated DIMENSION module exists the shared m_i lives on the head modules
    if dim_values is None:
        for name, module in iter_masked_linears(model):
            if int(getattr(module, "kind", -1)) != int(HEAD):
                continue
            mask = _input_mask(module)
            if mask is not None:
                dim_values = mask.detach().to(torch.float32).clone()
                break

    d_model = int(dim_values.numel()) if dim_values is not None else shape["d_model"]
    n_heads = shape["n_heads"]
    if head_values:
        widths = {v.numel() for v in head_values.values()}
        if widths:
            n_heads = max(widths)
    d_h = max(1, d_model // n_heads) if n_heads else d_model

    plan = PrunePlan(
        d_model=d_model,
        n_heads=n_heads,
        d_h=d_h,
        n_layers=int(n_layers or shape["n_layers"] or 0),
        dim_keep=_to_bool_list(dim_values, dim_threshold) or [True] * d_model,
        head_keep={k: _to_bool_list(v, threshold) for k, v in head_values.items()},
        neuron_keep={k: _to_bool_list(v, threshold) for k, v in neuron_values.items()},
        ffn_linear_count=int(shape["ffn_linear_count"]),
        attn_linear_count=int(shape["attn_linear_count"]),
        threshold=float(threshold),
    )
    plan.meta["require_grid"] = bool(require_grid)
    return plan


def plan_from_mask_state(mask_state, threshold: Optional[float] = None, n_layers: int = 0) -> PrunePlan:
    """Build a :class:`PrunePlan` from an :class:`apt.masks.MaskState`."""
    thr = float(getattr(mask_state, "threshold", 0.5)) if threshold is None else float(threshold)
    shape = getattr(mask_state, "shape", None)

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    d_model = int(_get(shape, "d_model", 0) or 0)
    n_heads = int(_get(shape, "n_heads", 0) or 0)
    if n_layers <= 0:
        n_layers = int(_get(shape, "n_layers", 0) or 0)

    dim_mask = mask_state.dim_mask_for() if hasattr(mask_state, "dim_mask_for") else None
    dim_keep = _to_bool_list(dim_mask, thr)
    if not dim_keep:
        dim_keep = [True] * d_model

    head_keep: Dict[str, List[bool]] = {}
    neuron_keep: Dict[str, List[bool]] = {}
    for layer in range(int(n_layers)):
        for site in ("self", "cross"):
            getter = getattr(mask_state, "head_mask_for", None)
            value = None
            if callable(getter):
                try:
                    value = getter(layer, site)
                except TypeError:
                    try:
                        value = getter(layer)
                    except Exception:
                        value = None
                except Exception:
                    value = None
            if value is None:
                continue
            key = f"{site}:{layer}"
            head_keep[key] = _to_bool_list(value, thr)
        getter = getattr(mask_state, "neuron_mask_for", None)
        if callable(getter):
            try:
                value = getter(layer)
            except Exception:
                value = None
            if value is not None:
                neuron_keep[str(layer)] = _to_bool_list(value, thr)

    return PrunePlan(
        d_model=d_model,
        n_heads=n_heads,
        d_h=max(1, d_model // n_heads) if n_heads else d_model,
        n_layers=int(n_layers),
        dim_keep=dim_keep,
        head_keep=head_keep,
        neuron_keep=neuron_keep,
        ffn_linear_count=int(_get(shape, "ffn_linear_count", 2) or 2),
        attn_linear_count=int(_get(shape, "attn_linear_count", 4) or 4),
        threshold=thr,
    )


# --------------------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------------------
def harden_masks(model: nn.Module, threshold: float = 0.5) -> nn.Module:
    """Snap all annealing masks of a wrapped model to 0/1 (Appendix C)."""
    inner = _unwrap(model)
    for target in (model, inner):
        if target is None:
            continue
        harden = getattr(target, "harden_masks", None)
        if callable(harden):
            try:
                harden(threshold)
            except TypeError:
                harden()
    for name, module in iter_masked_linears(model):
        out = _group_mask(module)
        if out is not None:
            binary = (out > threshold).to(out.dtype)
            for attr in ("mask_out", "output_mask"):
                if isinstance(getattr(module, attr, None), torch.Tensor):
                    setattr(module, attr, binary)
                    break
        inp = _input_mask(module)
        if inp is not None:
            binary_in = (inp > threshold).to(inp.dtype)
            for attr in ("mask_in", "input_mask"):
                if isinstance(getattr(module, attr, None), torch.Tensor):
                    setattr(module, attr, binary_in)
                    break
    return model


def merge_adapters(
    model: nn.Module,
    threshold: float = 0.0,
    restore_linears: bool = True,
    trainable_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Fold ``s * W_B W_A`` into the frozen weights and (optionally) drop the wrappers.

    Implements the footnote-2 guarantee of Sec. 4.1: after this call the APT
    tuning parameters are fully merged, so inference has zero adapter overhead.
    """
    harden_masks(model, threshold if threshold > 0 else 0.5) if restore_linears else None
    inner = _unwrap(model)
    if restore_linears:
        restore = getattr(model, "restore_base_linears", None) or getattr(
            inner, "restore_base_linears", None
        )
        if callable(restore):
            restore(merge=True, threshold=threshold)
            return model
        try:  # module-level helper of apt.model_wrapper
            from .model_wrapper import restore_base_linears  # type: ignore

            return restore_base_linears(model, merge=True, threshold=threshold)
        except Exception:
            pass

    for name, module in iter_masked_linears(model):
        merged = getattr(module, "merged_weight", None)
        base = _base_weight(module)
        if not callable(merged) or base is None:
            continue
        with torch.no_grad():
            value = merged(threshold)
            if trainable_dtype is not None:
                value = value.to(trainable_dtype)
            base.copy_(value.to(base.dtype))
    return model


def merge_masks_into_weights(model: nn.Module, threshold: float = 0.0) -> nn.Module:
    """Alias of :func:`merge_adapters` kept for readability in scripts."""
    return merge_adapters(model, threshold=threshold, restore_linears=False)


def restore_dense_model(model: nn.Module, threshold: float = 0.5) -> nn.Module:
    """Harden masks + merge adapters + replace masked linears by ``nn.Linear``."""
    return merge_adapters(model, threshold=threshold, restore_linears=True)


# --------------------------------------------------------------------------------------
# physical pruning
# --------------------------------------------------------------------------------------
_ATTN_Q = (".attention.self.query.", ".attention.q_lin.", "selfattention.q.", "selfattention.q_proj.")
_ATTN_K = (".attention.self.key.", ".attention.k_lin.", "selfattention.k.", "selfattention.k_proj.")
_ATTN_V = (".attention.self.value.", ".attention.v_lin.", "selfattention.v.", "selfattention.v_proj.")
_ATTN_O = (".attention.output.dense.", ".attention.out_lin.", "selfattention.o.", "selfattention.out_proj.")
_CROSS_Q = ("encdecattention.q.", "crossattention.q.")
_CROSS_K = ("encdecattention.k.", "crossattention.k.")
_CROSS_V = ("encdecattention.v.", "crossattention.v.")
_CROSS_O = ("encdecattention.o.", "crossattention.o.")
_FFN_IN = (".intermediate.dense.", "densereludense.wi.", "densereludense.wi_0.", "densereludense.wi_1.",
           "fc1.", "ffn.linear_in.")
_FFN_OUT = (".output.dense.", "densereludense.wo.", "fc2.", "ffn.linear_out.")
_NORM_HINTS = ("layernorm", ".ln_", "layer_norm", "norm.weight", "norm.bias", "rmsnorm")
_EMB_HINTS = ("word_embeddings", "shared.weight", "position_embeddings", "token_type_embeddings",
              "wte.", "wpe.", "embed_tokens", "word_embeddings_layernorm")
_OUT_HEADS = ("classifier", "pooler.dense", "lm_head", "qa_outputs", "score", "answer.", "qa_classifier",
              "sequence_summary", "prediction_head")


def _slice_rows(tensor: torch.Tensor, rows: Sequence[int]) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.clone()
    index = torch.as_tensor(list(rows), dtype=torch.long, device=tensor.device)
    try:
        return tensor.index_select(0, index).clone()
    except (IndexError, RuntimeError):
        index = index.clamp(max=tensor.shape[0] - 1)
        return tensor.index_select(0, index).clone()


def _slice_cols(tensor: torch.Tensor, cols: Sequence[int]) -> torch.Tensor:
    if tensor.dim() != 2:
        return _slice_rows(tensor, cols) if tensor.dim() == 1 else tensor.clone()
    index = torch.as_tensor(list(cols), dtype=torch.long, device=tensor.device)
    try:
        return tensor.index_select(1, index).clone()
    except (IndexError, RuntimeError):
        index = index.clamp(max=tensor.shape[1] - 1)
        return tensor.index_select(1, index).clone()


def _matches(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(p in lowered for p in patterns)


def _slice_parameter(
    name: str,
    tensor: torch.Tensor,
    maps: Dict[str, Any],
    d_model_old: int,
) -> torch.Tensor:
    """Slice one parameter of the dense (already merged) model."""
    lowered = name.lower()
    dim_map = list(maps["dim_map"])
    old_heads = int(maps.get("n_heads_old") or 0) or None
    site, layer = _layer_stack(name)
    head_key = f"{site}:{layer}"
    head_rows = maps["head_rows"].get(head_key)
    if head_rows is None:
        # fall back to the matching self/cross site of that layer, else any key
        for key in maps["head_rows"]:
            if key.endswith(f":{layer}"):
                head_rows = maps["head_rows"][key]
                break
    if head_rows is None and maps["head_rows"]:
        head_rows = next(iter(maps["head_rows"].values()))
    neuron_rows = maps["neuron_idx"].get(str(layer))
    if neuron_rows is None:
        neuron_rows = maps["neuron_idx"].get(f"{site}:{layer}")
    if neuron_rows is None and maps["neuron_idx"]:
        neuron_rows = next(iter(maps["neuron_idx"].values()))
    is_norm = _matches(name, _NORM_HINTS)
    is_emb = _matches(name, _EMB_HINTS)

    # ---- attention projections ------------------------------------------------
    if _matches(name, _CROSS_Q + _CROSS_K + _CROSS_V + _ATTN_Q + _ATTN_K + _ATTN_V):
        if tensor.dim() == 1:  # bias
            return _slice_rows(tensor, head_rows or range(tensor.shape[0]))
        if _matches(name, _CROSS_K + _CROSS_V):
            # key/value of the decoder cross-attention read the *encoder* width
            cols = maps.get("dim_map_enc") or dim_map
            return _slice_cols(_slice_rows(tensor, head_rows or range(tensor.shape[0])), cols)
        return _slice_cols(_slice_rows(tensor, head_rows or range(tensor.shape[0])), dim_map)
    if _matches(name, _CROSS_O + _ATTN_O):
        if tensor.dim() == 1:
            return _slice_rows(tensor, dim_map)
        cols = maps.get("dim_map_enc") or head_rows or range(tensor.shape[1])
        return _slice_cols(_slice_rows(tensor, dim_map), head_rows or range(tensor.shape[1]))
    if "relative_attention_bias" in lowered:
        return _slice_cols(tensor, head_rows or range(tensor.shape[-1]))

    # ---- FFN ------------------------------------------------------------------
    if _matches(name, _FFN_IN):
        if tensor.dim() == 1:
            return _slice_rows(tensor, neuron_rows or range(tensor.shape[0]))
        return _slice_cols(_slice_rows(tensor, neuron_rows or range(tensor.shape[0])), dim_map)
    if _matches(name, _FFN_OUT):
        if tensor.dim() == 1:
            return _slice_rows(tensor, dim_map)
        return _slice_cols(_slice_rows(tensor, dim_map), neuron_rows or range(tensor.shape[1]))

    # ---- norms / embeddings / task heads --------------------------------------
    if is_emb:
        if tensor.dim() == 2 and tensor.shape[1] == d_model_old:
            return _slice_cols(tensor, dim_map)
        if tensor.dim() == 1 and tensor.shape[0] == d_model_old:
            return _slice_rows(tensor, dim_map)
        return tensor.clone()
    if is_norm:
        if tensor.dim() == 1 and tensor.shape[0] == d_model_old:
            return _slice_rows(tensor, dim_map)
        return tensor.clone()
    if _matches(name, _OUT_HEADS):
        if tensor.dim() == 2 and tensor.shape[1] == d_model_old:
            return _slice_cols(tensor, dim_map)
        if tensor.dim() == 2 and tensor.shape[0] == d_model_old:
            return _slice_rows(tensor, dim_map)
        if tensor.dim() == 1 and tensor.shape[0] == d_model_old:
            return _slice_rows(tensor, dim_map)
        return tensor.clone()

    # ---- heuristic fallback ---------------------------------------------------
    if tensor.dim() == 2:
        rows_old, cols_old = tensor.shape
        if rows_old == d_model_old and cols_old == d_model_old:
            return _slice_cols(_slice_rows(tensor, dim_map), dim_map)
        if cols_old == d_model_old:
            return _slice_cols(tensor, dim_map)
        if rows_old == d_model_old:
            return _slice_rows(tensor, dim_map)
    if tensor.dim() == 1 and tensor.shape[0] == d_model_old:
        return _slice_rows(tensor, dim_map)
    return tensor.clone()


def _new_config(model: nn.Module, d_model: int, n_heads: int, n_ffn: int, d_h: int):
    config = model.config
    try:
        new_config = copy.deepcopy(config)
    except Exception:  # pragma: no cover
        new_config = config.__class__.from_dict(config.to_dict())
    for attr, value in (
        ("hidden_size", d_model),
        ("d_model", d_model),
        ("num_attention_heads", n_heads),
        ("num_heads", n_heads),
        ("intermediate_size", n_ffn),
        ("d_ff", n_ffn),
        ("head_dim", d_h),
        ("d_kv", d_h),
    ):
        if hasattr(new_config, attr):
            setattr(new_config, attr, int(value))
    return new_config


def physical_prune(
    model: nn.Module,
    plan: Optional[PrunePlan] = None,
    threshold: float = 0.5,
    restore_linears: bool = True,
    require_grid: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Return a plain HF model with the pruned heads / neurons / dims removed.

    The input may be an APT-wrapped model (adapters are merged first) or an
    already merged dense model.
    """
    wrapped = restore_linears and hasattr(model, "restore_base_linears")
    inner = _unwrap(model)
    if restore_linears:
        model = merge_adapters(model, threshold=threshold, restore_linears=True)
        inner = _unwrap(model)
    dense = inner

    if plan is None:
        raise ValueError("physical_prune requires a PrunePlan (use build_prune_plan)")
    d_model_old = int(plan.d_model)
    maps = plan.index_maps(require_grid=require_grid)
    shape = _model_shape_from(dense)
    maps["n_heads_old"] = int(plan.n_heads or shape["n_heads"])
    # encoder width for the T5 cross-attention key/value projections
    maps["dim_map_enc"] = list(maps["dim_map"])

    d_model_new = int(maps["d_model"])
    n_heads_new = int(max(1, maps["n_heads"]))
    d_h_new = int(max(1, maps["d_h"]))
    n_ffn_new = int(maps["n_ffn"] or shape["n_ffn"])
    if d_h_new * n_heads_new != d_model_new:
        d_h_new = max(1, d_model_new // max(1, n_heads_new))
        d_model_new = d_h_new * n_heads_new
        maps["d_h"] = d_h_new
        maps["d_model"] = d_model_new

    new_config = _new_config(dense, d_model_new, n_heads_new, n_ffn_new, d_h_new)
    try:
        new_model = dense.__class__(new_config)
    except Exception:
        new_model = copy.deepcopy(dense)
        new_model.config = new_config

    old_state = dense.state_dict()
    new_state = new_model.state_dict()
    copied: Dict[str, torch.Tensor] = {}
    for name, tensor in old_state.items():
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        sliced = _slice_parameter(name, tensor.to(torch.float32) if tensor.is_floating_point() else tensor,
                                  maps, d_model_old)
        if dtype is not None and sliced.is_floating_point():
            sliced = sliced.to(dtype)
        if name in new_state and tuple(sliced.shape) == tuple(new_state[name].shape):
            copied[name] = sliced
    missing = [k for k in new_state if k not in copied]
    for key in missing:
        # keep randomly initialised parameters of the new head / classifier
        copied[key] = new_state[key]
    new_model.load_state_dict(copied, strict=True)
    setattr(new_model, "apt_prune_plan", plan)
    setattr(new_model, "apt_wrapped_source", bool(wrapped))
    if hasattr(new_model, "config"):
        new_model.config.apt_pruned = True
    return new_model


def merge_and_prune(
    model: nn.Module,
    threshold: float = 0.5,
    plan: Optional[PrunePlan] = None,
    require_grid: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """One-shot inference export: merge LoRA + physically remove pruned blocks.

    This is the function used to verify Sec. 4.1's claim that tuning parameters
    add no inference overhead (``plan`` defaults to the model's current masks).
    """
    if plan is None:
        plan = build_prune_plan(model, threshold=threshold, require_grid=require_grid)
    return physical_prune(
        model, plan=plan, threshold=threshold, restore_linears=True,
        require_grid=require_grid, dtype=dtype,
    )


def removed_blocks(plan: PrunePlan) -> Dict[str, Any]:
    """Counts of the physical blocks removed by ``plan``."""
    dim_removed = int(sum(1 for keep in plan.dim_keep if not keep))
    heads_removed = sum(
        len(plan.head_keep[k]) - len(plan.kept_heads(k)) for k in plan.head_keep
    )
    neurons_removed = sum(
        len(plan.neuron_keep[k]) - len(plan.kept_neurons(k)) for k in plan.neuron_keep
    )
    return {
        "heads_removed": int(heads_removed),
        "neurons_removed": int(neurons_removed),
        "dimensions_removed": dim_removed,
        "heads_kept": int(plan.n_heads_kept),
        "neurons_kept": int(plan.n_ffn_kept),
        "dimensions_kept": int(len(plan.kept_dims())),
    }


def parameter_breakdown(model: nn.Module) -> Dict[str, float]:
    """Parameter counts / memory of a (pruned) model, in millions and MB."""
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return {
        "total_params": float(total),
        "trainable_params": float(trainable),
        "total_params_m": total / 1e6,
        "size_mb": model_size_mb(model),
    }


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------
def verify_merge_equivalence(
    wrapped_model: nn.Module,
    inputs: Optional[Dict[str, torch.Tensor]] = None,
    threshold: float = 1e-4,
    atol: float = 1e-5,
) -> Dict[str, float]:
    """Check that merging the adapters leaves the forward pass unchanged.

    Runs the wrapped model before and after :func:`merge_adapters` on a small
    random batch and returns the maximum absolute difference.  ``threshold``
    masks are first hardened in both passes so that only the merge is compared.
    """
    model = _unwrap(wrapped_model)
    device = next(model.parameters()).device
    if inputs is None:
        config = getattr(model, "config", None)
        vocab = int(getattr(config, "vocab_size", 1000) or 1000)
        seq = 8
        inputs = {
            "input_ids": torch.randint(0, min(vocab, 1000), (2, seq), device=device),
            "attention_mask": torch.ones(2, seq, dtype=torch.long, device=device),
        }
    harden_masks(wrapped_model, 0.5)
    with torch.no_grad():
        before = wrapped_model(**inputs)
        before_tensor = (before.logits if hasattr(before, "logits") else before[0]).detach().clone()
    merged = merge_adapters(wrapped_model, threshold=0.0, restore_linears=False)
    with torch.no_grad():
        after = merged(**inputs)
        after_tensor = (after.logits if hasattr(after, "logits") else after[0]).detach().clone()
    diff = float((before_tensor - after_tensor).abs().max().item())
    return {
        "max_abs_diff": diff,
        "mean_abs_diff": float((before_tensor - after_tensor).abs().mean().item()),
        "equivalent": float(diff <= max(atol, threshold)),
    }


def save_pruned_model(
    model: nn.Module,
    output_dir: str,
    tokenizer=None,
    plan: Optional[PrunePlan] = None,
) -> str:
    """Persist a pruned HF model (plus tokenizer and plan) to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    save_fn = getattr(model, "save_pretrained", None)
    if callable(save_fn):
        model.save_pretrained(output_dir)
    else:  # pragma: no cover
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass
    if plan is not None:
        save_plan(plan, os.path.join(output_dir, "apt_prune_plan.json"))
    return output_dir


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------
def _self_test() -> bool:
    """Dependency-light sanity checks for the plan/index-map/slice logic."""
    torch.manual_seed(0)
    # 1) plan with no pruning must be a no-op
    plan = PrunePlan(d_model=12, n_heads=3, d_h=4, n_layers=2,
                     dim_keep=[True] * 12,
                     head_keep={"self:0": [True] * 3, "self:1": [True] * 3},
                     neuron_keep={"0": [True] * 8, "1": [True] * 8})
    maps = plan.index_maps()
    assert maps["dim_map"] == list(range(12)), maps["dim_map"]
    assert maps["n_heads"] == 3 and maps["d_h"] == 4
    assert maps["head_rows"]["self:0"] == list(range(12))

    # 2) prune one head and the tail dims of the remaining heads
    dim_keep = [True] * 12
    for h in (1, 2):
        for r in (3,):
            dim_keep[h * 4 + r] = False
    plan = PrunePlan(d_model=12, n_heads=3, d_h=4, n_layers=2, dim_keep=dim_keep,
                     head_keep={"self:0": [True, False, True]},
                     neuron_keep={"0": [True] * 4 + [False] * 4})
    maps = plan.index_maps()
    assert maps["n_heads"] == 2, maps["n_heads"]
    assert maps["d_h"] == 3, maps["d_h"]
    assert maps["d_model"] == 6, maps["d_model"]
    assert maps["head_rows"]["self:0"] == [0, 1, 2, 8, 9, 10], maps["head_rows"]
    assert maps["neuron_idx"]["0"] == [0, 1, 2, 3]
    assert removed_blocks(plan)["heads_removed"] == 1

    # 3) slicing helpers round-trip shapes
    w = torch.randn(12, 12)
    sliced = _slice_parameter("roberta.encoder.layer.0.attention.self.query.weight", w, maps, 12)
    assert tuple(sliced.shape) == (6, 6), sliced.shape
    o = torch.randn(12, 12)
    sliced_o = _slice_parameter("roberta.encoder.layer.0.attention.output.dense.weight", o, maps, 12)
    assert tuple(sliced_o.shape) == (6, 6), sliced_o.shape
    ffn_in = torch.randn(8, 12)
    sliced_ffn = _slice_parameter("roberta.encoder.layer.0.intermediate.dense.weight", ffn_in, maps, 12)
    assert tuple(sliced_ffn.shape) == (4, 6), sliced_ffn.shape
    emb = torch.randn(50, 12)
    sliced_emb = _slice_parameter("roberta.embeddings.word_embeddings.weight", emb, maps, 12)
    assert tuple(sliced_emb.shape) == (50, 6), sliced_emb.shape
    ln = torch.randn(12)
    assert tuple(_slice_parameter("roberta.encoder.layer.0.attention.output.LayerNorm.weight",
                                  ln, maps, 12).shape) == (6,)

    # 4) JSON round-trip
    assert PrunePlan.from_dict(plan.as_dict()).cell_counts() == plan.cell_counts()
    return True


if __name__ == "__main__":  # pragma: no cover
    print("merge self-test:", _self_test())
