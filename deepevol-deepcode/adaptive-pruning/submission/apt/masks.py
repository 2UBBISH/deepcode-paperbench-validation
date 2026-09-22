"""Mask state, gradual mask decay and sparsity bookkeeping for APT.

This module implements the pruning-mask machinery of APT:

* **Mask state** -- binary/annealed masks for MHA heads, FFN neurons and the
  model hidden dimension, initialised to ones (Appendix C, Algorithm 1:
  "Initial parameters and masks :math:`\\Theta_0, M_0`").
* **Gradual (annealed) decay** -- Appendix C / Section 4.1 ("In our
  implementation, for training stability, we do not set the pruned blocks'
  corresponding masks to 0 directly but gradually decrease their values by
  :math:`\\alpha=0.01`") and Appendix A ("we gradually decrease the pruning
  masks of pruned blocks by :math:`\\alpha<1` instead of instantly setting
  them from ones to zeros").  Following Algorithm 1 this is

      M_1^{(t)} <- min(1, M_1^{(t-1)} + alpha)     (retained blocks)
      M_0^{(t)} <- max(0, M_0^{(t-1)} - alpha)     (pruned blocks)

* **Sparsity bookkeeping** -- tracks the approximated LM parameter count
  :math:`\\mathcal{C}(\\Theta_t; \\mathcal{M}_t)` of Equation (6)
  (:math:`d_m \\sum_i (4 n_h^i d_h + 2 n_f^i)`) and the realised sparsity
  :math:`\\gamma_t` (definition from Appendix A: ratio of pruned parameter
  size to the total parameter size).  Densities are re-computed after every
  parameter-size change and a *optimizer reset request* is raised whenever
  the number of retained blocks (i.e. the parameter shapes / effective
  parameter count) changes (Section 6: "We reset the optimizer every time
  after each parameter size changes to avoid stability issues").

The masks are consumed by :mod:`apt.block_selection` (which decides *which*
blocks to retain) and are pushed onto the :class:`apt.adapters.MaskedLinear`
modules of a wrapped model by :meth:`MaskState.apply_to_model`.

Contract with :class:`apt.adapters.MaskedLinear`
------------------------------------------------
Every masked linear layer should expose

* ``kind``       : ``0`` (HEAD), ``1`` (NEURON) or ``2`` (DIMENSION) --
  describes which *block type its output mask prunes*,
* ``layer_idx``  : index of the owning transformer layer,
* ``module_name``: dotted name of the module (used to resolve the attention
  ``site``, e.g. ``"query"`` / ``"value"``),
* ``num_out_groups`` : number of output groups (``n_heads`` for a head-type
  layer, ``n_ffn`` for a neuron-type layer, ``1``/``d_model`` otherwise),
* ``set_input_mask``, ``set_output_mask``, ``set_output_group_mask``.

The hidden-dimension (``m_i``) mask is a single shared tensor per model so
that every module reading the residual stream is masked consistently with the
same object (``apt.adapters.make_masked_linear(share_input_mask=...)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# imports from sibling modules (tolerate partial checkouts)
# ---------------------------------------------------------------------------
try:  # block type constants live in apt.adapters
    from .adapters import BLOCK_TYPES, DIMENSION, HEAD, NEURON  # type: ignore
except Exception:  # pragma: no cover - defensive
    HEAD, NEURON, DIMENSION = 0, 1, 2
    BLOCK_TYPES = {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}

try:
    from .adapters import iter_masked_linears as _iter_masked_linears  # type: ignore
except Exception:  # pragma: no cover - defensive
    _iter_masked_linears = None

try:
    from .block_selection import (  # type: ignore
        BlockSelector,
        ModelShape,
        approx_param_count as _approx_param_count,
    )
except Exception:  # pragma: no cover - defensive
    BlockSelector = None  # type: ignore
    ModelShape = None  # type: ignore
    _approx_param_count = None  # type: ignore


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MASK_THRESHOLD",
    "HEAD",
    "NEURON",
    "DIMENSION",
    "KIND_NAMES",
    "NAME_TO_KIND",
    "BlockDescriptor",
    "MaskState",
    "MaskManager",
    "MaskController",
    "coerce_block",
    "coerce_blocks",
    "expand_group_mask",
    "model_sites",
    "find_masked_linears",
    "approx_lm_param_count",
    "masked_parameter_summary",
]

#: Gradual decay step used by APT (Appendix A/C, Algorithm 1).
DEFAULT_ALPHA = 0.01

#: Threshold used to read a (soft) mask value as binary.
DEFAULT_MASK_THRESHOLD = 0.5

KIND_NAMES = {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}
NAME_TO_KIND = {
    "head": HEAD,
    "heads": HEAD,
    "mha": HEAD,
    "attn_head": HEAD,
    "neuron": NEURON,
    "neurons": NEURON,
    "ffn": NEURON,
    "intermediate": NEURON,
    "dimension": DIMENSION,
    "dimensions": DIMENSION,
    "dim": DIMENSION,
    "dims": DIMENSION,
    "hidden": DIMENSION,
    "hidden_dim": DIMENSION,
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _shape_attr(shape: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that also accepts ``dict``-like shape descriptions."""
    if shape is None:
        return default
    if isinstance(shape, Mapping):
        return shape.get(name, default)
    return getattr(shape, name, default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:  # pragma: no cover - defensive
        return default


def _to_tensor(values: Any, size: Optional[int], dtype, device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        out = values.detach().to(dtype=dtype, device=device).reshape(-1)
    else:
        out = torch.as_tensor(values, dtype=dtype, device=device).reshape(-1)
    if size is not None and out.numel() != size:
        raise ValueError(f"expected mask of size {size}, got {out.numel()}")
    return out


def model_sites(shape: Any) -> Tuple[str, ...]:
    """Names of the attention sites (adapters) of one MHA layer.

    APT places adapters in the query and value projections of every MHA layer
    (Section 4.1); model wrappers may report their own site names through
    ``shape.sites`` / ``shape.attn_sites``.
    """
    if shape is None:
        return ("attn",)
    for attr in ("sites", "attn_sites", "adapter_sites"):
        value = _shape_attr(shape, attr, None)
        if value:
            if isinstance(value, str):
                return (value,)
            return tuple(str(v) for v in value)
    n = _as_int(_shape_attr(shape, "n_attn_sites", 1), 1)
    if n <= 1:
        return ("attn",)
    if n == 2:
        return ("query", "value")
    if n == 3:
        return ("query", "key", "value")
    return ("query", "key", "value", "output")[:n]


def expand_group_mask(values: torch.Tensor, out_features: int, group_size: int) -> torch.Tensor:
    """Expand one mask value per output group to a per-output-neuron mask."""
    if group_size <= 1:
        if values.numel() == out_features:
            return values
        reps = int(math.ceil(out_features / max(values.numel(), 1)))
        return values.repeat(reps)[:out_features]
    expanded = values.repeat_interleave(group_size)
    if expanded.numel() < out_features:  # pad for safety (non-divisible shapes)
        pad = out_features - expanded.numel()
        one = torch.ones(pad, dtype=expanded.dtype, device=expanded.device)
        expanded = torch.cat([expanded, one])
    return expanded[:out_features]


def find_masked_linears(module: torch.nn.Module, names: Optional[Sequence[str]] = None):
    """Yield ``(name, MaskedLinear)`` pairs (delegates to :mod:`apt.adapters`)."""
    if _iter_masked_linears is not None:
        yield from _iter_masked_linears(module, names)
        return
    for name, sub in module.named_modules():  # pragma: no cover - fallback path
        if hasattr(sub, "set_output_group_mask") and hasattr(sub, "num_out_groups"):
            if names is None or name in names:
                yield name, sub


def approx_lm_param_count(
    n_heads: float,
    n_ffn: float,
    d_model: float,
    head_dim: float,
    ffn_linear_count: int = 2,
    attn_linear_count: int = 4,
    n_layers: int = 1,
) -> float:
    """Approximated LM parameter count, Equation (6).

    :math:`\\mathcal{C}(\\Theta_t;\\mathcal{M}_t) \\approx
    d_m \\sum_i (4 n_h^i d_h + 2 n_f^i)`.  ``ffn_linear_count`` is 3 for gated
    FFNs (T5/LLaMA-like, Appendix C) and 2 otherwise; bias / layer-norm terms
    and tuning parameters are ignored (footnote 2 of Section 4.1).
    """
    if _approx_param_count is not None:
        try:
            return float(
                _approx_param_count(
                    n_heads=n_heads,
                    n_ffn=n_ffn,
                    d_model=d_model,
                    head_dim=head_dim,
                    ffn_linear_count=ffn_linear_count,
                    attn_linear_count=attn_linear_count,
                )
            ) * float(n_layers if n_layers else 1) / max(float(n_layers if n_layers else 1), 1.0)
        except TypeError:
            try:
                return float(
                    _approx_param_count(
                        n_heads, n_ffn, d_model, head_dim, ffn_linear_count, attn_linear_count
                    )
                )
            except Exception:  # pragma: no cover - defensive
                pass
        except Exception:  # pragma: no cover - defensive
            pass
    per_layer = attn_linear_count * float(head_dim) * float(n_heads) + ffn_linear_count * float(n_ffn)
    return per_layer * float(d_model) * float(max(n_layers, 1))


# ---------------------------------------------------------------------------
# block descriptors
# ---------------------------------------------------------------------------
@dataclass
class BlockDescriptor:
    """A single prunable block: one head, one FFN neuron or one hidden dim."""

    kind: int
    layer: int
    index: int
    site: Optional[str] = None

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(int(self.kind), str(self.kind))

    @property
    def name(self) -> str:
        if self.site:
            return f"{self.kind_name}.{self.layer}.{self.index}.{self.site}"
        return f"{self.kind_name}.{self.layer}.{self.index}"

    def as_tuple(self) -> Tuple[int, int, int, Optional[str]]:
        return (int(self.kind), int(self.layer), int(self.index), self.site)

    def __iter__(self):
        yield from self.as_tuple()

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


def coerce_block(item: Any) -> BlockDescriptor:
    """Convert a block into a :class:`BlockDescriptor`.

    Accepts ``BlockDescriptor``/``Block``-like objects (any object exposing
    ``kind``, ``layer`` and ``index``), ``(kind, layer, index[, site])``
    tuples and strings such as ``"head.0.3"`` / ``"neuron.5.12.query"``.
    """
    if isinstance(item, BlockDescriptor):
        return item
    if isinstance(item, str):
        parts = item.split(".")
        if parts and parts[0].lower() in NAME_TO_KIND:
            kind = NAME_TO_KIND[parts[0].lower()]
            parts = parts[1:]
        else:
            kind = HEAD
        layer = _as_int(parts[0], 0) if parts else 0
        index = _as_int(parts[1], 0) if len(parts) > 1 else 0
        site = parts[2] if len(parts) > 2 else None
        return BlockDescriptor(kind=kind, layer=layer, index=index, site=site)
    if isinstance(item, Mapping):
        return BlockDescriptor(
            kind=int(item.get("kind", HEAD)),
            layer=int(item.get("layer", 0)),
            index=int(item.get("index", 0)),
            site=item.get("site", None),
        )
    if isinstance(item, (tuple, list)):
        kind = int(item[0])
        layer = int(item[1]) if len(item) > 1 else 0
        index = int(item[2]) if len(item) > 2 else 0
        site = item[3] if len(item) > 3 else None
        return BlockDescriptor(kind=kind, layer=layer, index=index, site=site)
    # Block-like objects produced by apt.block_selection.Block
    kind = getattr(item, "kind", HEAD)
    if isinstance(kind, str):
        kind = NAME_TO_KIND.get(kind.lower(), HEAD)
    layer = getattr(item, "layer", 0)
    index = getattr(item, "index", 0)
    site = getattr(item, "site", None)
    return BlockDescriptor(kind=int(kind), layer=_as_int(layer), index=_as_int(index), site=site)


def coerce_blocks(items: Optional[Iterable[Any]]) -> List[BlockDescriptor]:
    if items is None:
        return []
    return [coerce_block(item) for item in items]


# ---------------------------------------------------------------------------
# mask state
# ---------------------------------------------------------------------------
class MaskState:
    """Container of the annealed APT pruning masks ``M_1`` (retained) / ``M_0``.

    Masks are stored as float tensors in ``[0, 1]``.  Retained blocks move
    towards 1 and pruned blocks towards 0 with step ``alpha`` (Algorithm 1).
    """

    def __init__(
        self,
        shape: Any,
        alpha: float = DEFAULT_ALPHA,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        hard: bool = False,
        threshold: float = DEFAULT_MASK_THRESHOLD,
        share_head_masks_across_sites: bool = True,
    ) -> None:
        self.shape = shape
        self.alpha = float(alpha)
        self.dtype = dtype
        self.device = device
        self.hard = bool(hard)
        self.threshold = float(threshold)
        self.share_head_masks_across_sites = bool(share_head_masks_across_sites)

        self.n_layers = _as_int(_shape_attr(shape, "n_layers", 1), 1)
        self.n_heads = _as_int(_shape_attr(shape, "n_heads", 1), 1)
        self.n_ffn = _as_int(_shape_attr(shape, "n_ffn", 1), 1)
        self.d_model = _as_int(_shape_attr(shape, "d_model", 1), 1)
        self.head_dim = _as_int(_shape_attr(shape, "head_dim", None), 0) or max(self.d_model // max(self.n_heads, 1), 1)
        self.ffn_linear_count = _as_int(_shape_attr(shape, "ffn_linear_count", 2), 2)
        self.attn_linear_count = _as_int(_shape_attr(shape, "attn_linear_count", 4), 4)
        self.n_cross_attn_layers = _as_int(_shape_attr(shape, "n_cross_attn_layers", 0), 0)
        self.sites: Tuple[str, ...] = model_sites(shape)

        # (layer, site) -> tensor(n_heads) ; layer -> tensor(n_ffn) ; tensor(d_model)
        self.head_values: Dict[Tuple[int, str], torch.Tensor] = {}
        self.neuron_values: Dict[int, torch.Tensor] = {}
        self.dim_values: torch.Tensor = torch.empty(0, dtype=dtype)

        self.version = 0
        self.n_updates = 0
        self.history: List[Dict[str, float]] = []

        self._optimizer_reset_pending = False
        self.optimizer_reset_counter = 0
        self.last_counts: Tuple[float, float, float] = (-1.0, -1.0, -1.0)

        self.reset()

    # -- construction -------------------------------------------------------
    def _new_ones(self, size: int) -> torch.Tensor:
        return torch.ones(int(size), dtype=self.dtype, device=self.device)

    def reset(self) -> None:
        """Set all masks to 1 (Algorithm 1, ``M_0`` initialisation)."""
        self.head_values = {}
        self.neuron_values = {}
        first: Dict[int, torch.Tensor] = {}
        for layer in range(self.n_layers):
            base = self._new_ones(self.n_heads)
            first[layer] = base
            for site in self.sites:
                if self.share_head_masks_across_sites:
                    self.head_values[(layer, site)] = base
                else:
                    self.head_values[(layer, site)] = self._new_ones(self.n_heads)
        self.neuron_values = {layer: self._new_ones(self.n_ffn) for layer in range(self.n_layers)}
        self.dim_values = self._new_ones(self.d_model)
        self.version += 1
        self.n_updates = 0

    def all_ones(self) -> bool:
        return all(
            bool(torch.all(v >= 1.0 - 1e-6)) for v in self.head_values.values()
        ) and all(bool(torch.all(v >= 1.0 - 1e-6)) for v in self.neuron_values.values()) and bool(
            torch.all(self.dim_values >= 1.0 - 1e-6)
        )

    # -- accessors ----------------------------------------------------------
    def sites_for_layer(self, layer: int) -> Tuple[str, ...]:
        found = tuple(site for (l, site) in self.head_values.keys() if l == int(layer))
        return found or self.sites

    def head_mask_for(self, layer: int, site: Optional[str] = None) -> torch.Tensor:
        layer = int(layer)
        site = site or self.sites[0]
        key = (layer, site)
        if key not in self.head_values:
            if self.share_head_masks_across_sites and (layer, self.sites[0]) in self.head_values:
                return self.head_values[(layer, self.sites[0])]
            self.head_values[key] = self._new_ones(self.n_heads)
        return self.head_values[key]

    def set_head_mask(self, layer: int, values: Any, site: Optional[str] = None) -> None:
        tensor = _to_tensor(values, self.n_heads, self.dtype, self.device)
        sites = (site,) if site else self.sites_for_layer(layer)
        for s in sites:
            if self.hard:
                tensor = (tensor >= self.threshold).to(tensor.dtype)
            self.head_values[(int(layer), s)] = tensor

    def neuron_mask_for(self, layer: int) -> torch.Tensor:
        layer = int(layer)
        if layer not in self.neuron_values:
            self.neuron_values[layer] = self._new_ones(self.n_ffn)
        return self.neuron_values[layer]

    def set_neuron_mask(self, layer: int, values: Any) -> None:
        tensor = _to_tensor(values, self.n_ffn, self.dtype, self.device)
        if self.hard:
            tensor = (tensor >= self.threshold).to(tensor.dtype)
        self.neuron_values[int(layer)] = tensor

    def dim_mask_for(self, size: Optional[int] = None) -> Optional[torch.Tensor]:
        if size is not None and int(size) != self.dim_values.numel():
            return None
        return self.dim_values

    def set_dim_mask(self, values: Any) -> None:
        tensor = _to_tensor(values, self.d_model, self.dtype, self.device)
        if self.hard:
            tensor = (tensor >= self.threshold).to(tensor.dtype)
        self.dim_values = tensor

    # -- decay --------------------------------------------------------------
    def _apply_to_block(self, block: BlockDescriptor, delta: float) -> None:
        with torch.no_grad():
            if block.kind == HEAD:
                tensor = self.head_mask_for(block.layer, block.site)
                if 0 <= block.index < tensor.numel():
                    tensor[block.index] = min(1.0, max(0.0, float(tensor[block.index]) + delta))
            elif block.kind == NEURON:
                tensor = self.neuron_mask_for(block.layer)
                if 0 <= block.index < tensor.numel():
                    tensor[block.index] = min(1.0, max(0.0, float(tensor[block.index]) + delta))
            elif block.kind == DIMENSION:
                tensor = self.dim_values
                if 0 <= block.index < tensor.numel():
                    tensor[block.index] = min(1.0, max(0.0, float(tensor[block.index]) + delta))
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown block kind {block.kind}")

    def decay(
        self,
        retained: Optional[Iterable[Any]] = None,
        pruned: Optional[Iterable[Any]] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, int]:
        """Gradually move retained masks towards 1 and pruned masks towards 0.

        Implements ``M_1 <- min(1, M_1 + alpha)`` and
        ``M_0 <- max(0, M_0 - alpha)`` of Algorithm 1 (``alpha = 0.01``).
        """
        step = float(self.alpha if alpha is None else alpha)
        retained_blocks = coerce_blocks(retained)
        pruned_blocks = coerce_blocks(pruned)
        for block in retained_blocks:
            self._apply_to_block(block, +step)
        for block in pruned_blocks:
            self._apply_to_block(block, -step)
        self.n_updates += 1
        self.version += 1
        return {"retained": len(retained_blocks), "pruned": len(pruned_blocks)}

    def apply_selection(self, selection: Any, alpha: Optional[float] = None) -> Dict[str, int]:
        """Apply one selection (``apt.block_selection.SelectionResult``) step."""
        retained = getattr(selection, "retained", None)
        pruned = getattr(selection, "pruned", None)
        if retained is None and isinstance(selection, Mapping):
            retained = selection.get("retained")
            pruned = selection.get("pruned")
        return self.decay(retained=retained, pruned=pruned, alpha=alpha)

    # -- hardening ----------------------------------------------------------
    def harden(self, threshold: Optional[float] = None, inplace: bool = True) -> "MaskState":
        """Snap all mask values to binary (used before merge / inference)."""
        thr = float(self.threshold if threshold is None else threshold)
        target = self if inplace else self.clone()

        def _snap(t: torch.Tensor) -> torch.Tensor:
            return (t >= thr).to(t.dtype)

        with torch.no_grad():
            new_heads: Dict[Tuple[int, str], torch.Tensor] = {}
            for (layer, site), tensor in target.head_values.items():
                snapped = _snap(tensor)
                if not inplace:
                    new_heads[(layer, site)] = snapped
                else:
                    tensor.copy_(snapped)
            if not inplace:
                target.head_values = new_heads
            for layer, tensor in list(target.neuron_values.items()):
                snapped = _snap(tensor)
                if inplace:
                    tensor.copy_(snapped)
                else:
                    target.neuron_values[layer] = snapped
            snapped_dim = _snap(target.dim_values)
            if inplace:
                target.dim_values.copy_(snapped_dim)
            else:
                target.dim_values = snapped_dim
        target.hard = True
        target.version += 1
        return target

    def clone(self) -> "MaskState":
        new = MaskState(
            self.shape,
            alpha=self.alpha,
            dtype=self.dtype,
            device=self.device,
            hard=self.hard,
            threshold=self.threshold,
            share_head_masks_across_sites=self.share_head_masks_across_sites,
        )
        new.head_values = {k: v.clone() for k, v in self.head_values.items()}
        new.neuron_values = {k: v.clone() for k, v in self.neuron_values.items()}
        new.dim_values = self.dim_values.clone()
        new.version = self.version
        new.n_updates = self.n_updates
        return new

    # -- retained counts / parameter bookkeeping ----------------------------
    def _count(tensor: torch.Tensor, threshold: Optional[float] = None, strict: bool = True) -> float:
        if tensor.numel() == 0:
            return 0.0
        if threshold is None:
            thr = 0.5 if not strict else 0.5
        else:
            thr = float(threshold)
        return float((tensor >= thr).sum().item())

    def layer_head_mass(self, layer: int, threshold: Optional[float] = None) -> float:
        """Number of retained heads of a layer (heads are shared across sites)."""
        tensor = self.head_mask_for(layer, self.sites_for_layer(layer)[0])
        return self._count(tensor, threshold)

    def layer_neuron_mass(self, layer: int, threshold: Optional[float] = None) -> float:
        return self._count(self.neuron_mask_for(layer), threshold)

    def dim_mass(self, threshold: Optional[float] = None) -> float:
        return self._count(self.dim_values, threshold)

    def counts(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """Retained block counts: total heads, total neurons and hidden size."""
        heads = sum(self.layer_head_mass(l, threshold) for l in range(self.n_layers))
        neurons = sum(self.layer_neuron_mass(l, threshold) for l in range(self.n_layers))
        return {
            "n_heads": heads,
            "n_neurons": neurons,
            "d_model": self.dim_mass(threshold),
            "n_layers": self.n_layers,
        }

    def effective_heads_per_layer(self, threshold: Optional[float] = None) -> List[float]:
        return [self.layer_head_mass(l, threshold) for l in range(self.n_layers)]

    def effective_neurons_per_layer(self, threshold: Optional[float] = None) -> List[float]:
        return [self.layer_neuron_mass(l, threshold) for l in range(self.n_layers)]

    def param_count(self, threshold: Optional[float] = None, soft: bool = False) -> float:
        """Approximated LM parameter count under the current masks, Eq. (6)."""
        d_m = float(self.dim_values.sum().item()) if soft else self.dim_mass(threshold)
        heads = self.effective_heads_per_layer(threshold)
        neurons = self.effective_neurons_per_layer(threshold)
        total = 0.0
        for layer in range(self.n_layers):
            total += approx_lm_param_count(
                n_heads=heads[layer],
                n_ffn=neurons[layer],
                d_model=d_m,
                head_dim=self.head_dim,
                ffn_linear_count=self.ffn_linear_count,
                attn_linear_count=self.attn_linear_count,
                n_layers=1,
            )
        if self.n_cross_attn_layers:
            # cross-attention layers of encoder-decoder LMs are counted as well
            # (Appendix C): they add 4 attn matrices per layer but no FFN.
            cross_d_m = d_m
            cross_heads = sum(heads) / max(len(heads), 1)
            for _ in range(self.n_cross_attn_layers):
                total += approx_lm_param_count(
                    n_heads=cross_heads,
                    n_ffn=0.0,
                    d_model=cross_d_m,
                    head_dim=self.head_dim,
                    ffn_linear_count=self.ffn_linear_count,
                    attn_linear_count=self.attn_linear_count,
                    n_layers=1,
                )
        return float(total)

    def original_param_count(self) -> float:
        """``C_0``: parameter count with every mask set to one."""
        clone = self.clone()
        clone.reset()
        return clone.param_count()

    def sparsity(self, threshold: Optional[float] = None) -> float:
        """Realised sparsity ``gamma`` = pruned parameters / total parameters."""
        c0 = self.original_param_count()
        if c0 <= 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - self.param_count(threshold) / c0)))

    # -- optimizer reset bookkeeping ---------------------------------------
    def needs_optimizer_reset(self) -> bool:
        return bool(self._optimizer_reset_pending)

    def mark_optimizer_reset(self, flag: bool = True) -> None:
        self._optimizer_reset_pending = bool(flag)

    def consume_optimizer_reset(self) -> bool:
        flag = self._optimizer_reset_pending
        if flag:
            self.optimizer_reset_counter += 1
            self._optimizer_reset_pending = False
        return flag

    def _detect_parameter_size_change(self) -> bool:
        """Recompute density bookkeeping; flag shape changes."""
        counts = self.counts()
        current = (counts["n_heads"], counts["n_neurons"], counts["d_model"])
        changed = current != self.last_counts and self.last_counts != (-1.0, -1.0, -1.0)
        self.last_counts = current
        if changed:
            self.mark_optimizer_reset(True)
        return changed

    # -- serialisation ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "hard": self.hard,
            "head_values": {f"{layer}::{site}": v.detach().cpu().clone() for (layer, site), v in self.head_values.items()},
            "neuron_values": {int(layer): v.detach().cpu().clone() for layer, v in self.neuron_values.items()},
            "dim_values": self.dim_values.detach().cpu().clone(),
            "n_updates": self.n_updates,
            "version": self.version,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "MaskState":
        self.alpha = float(state.get("alpha", self.alpha))
        self.threshold = float(state.get("threshold", self.threshold))
        self.hard = bool(state.get("hard", self.hard))
        heads: Dict[Tuple[int, str], torch.Tensor] = {}
        for key, value in dict(state.get("head_values", {})).items():
            layer_s, _, site = str(key).partition("::")
            heads[(int(layer_s), site)] = value.to(dtype=self.dtype, device=self.device)
        if heads:
            self.head_values = heads
        neurons = {int(k): v.to(dtype=self.dtype, device=self.device) for k, v in dict(state.get("neuron_values", {})).items()}
        if neurons:
            self.neuron_values = neurons
        dim = state.get("dim_values")
        if dim is not None:
            self.dim_values = dim.to(dtype=self.dtype, device=self.device)
        self.n_updates = _as_int(state.get("n_updates", self.n_updates), self.n_updates)
        self.version = _as_int(state.get("version", self.version), self.version)
        return self

    # -- model plumbing -----------------------------------------------------
    def _site_for_module(self, module: torch.nn.Module) -> str:
        name = str(getattr(module, "module_name", "") or "").lower()
        layer = _as_int(getattr(module, "layer_idx", 0), 0)
        sites = self.sites_for_layer(layer)
        for site in sites:
            if site and site.lower() in name:
                return site
        return sites[0]

    def apply_to_model(self, model: torch.nn.Module, names: Optional[Sequence[str]] = None) -> int:
        """Push current mask values onto every masked linear of ``model``.

        Returns the number of modules that were updated.
        """
        updated = 0
        for name, module in find_masked_linears(model, names):
            layer = _as_int(getattr(module, "layer_idx", -1), -1)
            kind = getattr(module, "kind", HEAD)
            if isinstance(kind, str):
                kind = NAME_TO_KIND.get(kind.lower(), HEAD)
            kind = int(kind)

            base_weight = getattr(module, "base_weight", None)
            in_features = _as_int(getattr(module, "in_features", None), 0)
            out_features = _as_int(getattr(module, "out_features", None), 0)
            if base_weight is not None and hasattr(base_weight, "shape"):
                if not in_features and base_weight.dim() == 2:
                    in_features = int(base_weight.shape[1])
                if not out_features and base_weight.dim() == 2:
                    out_features = int(base_weight.shape[0])

            # ---- input mask (m_i) : hidden dimension of the residual stream
            if hasattr(module, "set_input_mask") and in_features:
                if in_features == self.dim_values.numel():
                    module.set_input_mask(self.dim_values)
                    updated += 1
                elif kind == NEURON and layer >= 0 and in_features == self.n_ffn:
                    module.set_input_mask(self.neuron_mask_for(layer))

            # ---- output mask (m_o) : heads / neurons / hidden dimension
            if not hasattr(module, "set_output_group_mask"):
                continue
            num_groups = getattr(module, "num_out_groups", None)
            if callable(num_groups):  # pragma: no cover - defensive
                num_groups = num_groups()
            num_groups = _as_int(num_groups, 0)

            values: Optional[torch.Tensor] = None
            if kind == HEAD and layer >= 0:
                values = self.head_mask_for(layer, self._site_for_module(module))
            elif kind == NEURON and layer >= 0:
                values = self.neuron_mask_for(layer)
            elif kind == DIMENSION:
                values = self.dim_values

            if values is None:
                continue
            if num_groups and values.numel() == num_groups and out_features != values.numel():
                module.set_output_group_mask(values)
                updated += 1
            else:
                group_size = max(int(round(out_features / max(values.numel(), 1))), 1) if out_features else 1
                expanded = expand_group_mask(values, out_features, group_size) if out_features else values
                if hasattr(module, "set_output_mask"):
                    module.set_output_mask(expanded)
                    updated += 1
                else:  # pragma: no cover - defensive
                    module.set_output_group_mask(values)
                    updated += 1
        return updated

    # -- logging ------------------------------------------------------------
    def summary(self, threshold: Optional[float] = None) -> Dict[str, float]:
        counts = self.counts(threshold)
        origin = {
            "n_heads": float(self.n_heads * self.n_layers),
            "n_neurons": float(self.n_ffn * self.n_layers),
            "d_model": float(self.d_model),
        }
        return {
            "sparsity": self.sparsity(threshold),
            "param_count": self.param_count(threshold),
            "param_count_original": self.original_param_count(),
            "retained_heads": counts["n_heads"],
            "retained_neurons": counts["n_neurons"],
            "retained_dims": counts["d_model"],
            "head_ratio": counts["n_heads"] / max(origin["n_heads"], 1.0),
            "neuron_ratio": counts["n_neurons"] / max(origin["n_neurons"], 1.0),
            "dim_ratio": counts["d_model"] / max(origin["d_model"], 1.0),
            "n_updates": float(self.n_updates),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.summary()
        return (
            f"MaskState(sparsity={s['sparsity']:.3f}, heads={s['retained_heads']:.0f}/{self.n_heads * self.n_layers}, "
            f"neurons={s['retained_neurons']:.0f}/{self.n_ffn * self.n_layers}, dims={s['retained_dims']:.0f}/{self.d_model})"
        )


# ---------------------------------------------------------------------------
# mask manager (state + selection + bookkeeping)
# ---------------------------------------------------------------------------
class MaskManager:
    """Drives one APT pruning step (Algorithm 1, lines "Select blocks" /
    "Update masks").

    Responsibilities:

    1. run the latency-saliency knapsack (:mod:`apt.block_selection`) for the
       current sparsity target,
    2. anneal the masks with ``alpha`` towards the selected pattern,
    3. push the masks into the model,
    4. keep the sparsity bookkeeping up to date and request an optimizer reset
       whenever the parameter shapes (retained block counts) change.
    """

    def __init__(
        self,
        shape: Any,
        selector: Optional[Any] = None,
        alpha: float = DEFAULT_ALPHA,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        hard: bool = False,
        threshold: float = DEFAULT_MASK_THRESHOLD,
        share_head_masks_across_sites: bool = True,
    ) -> None:
        self.shape = shape
        self.selector = selector
        if self.selector is None and BlockSelector is not None:
            try:
                self.selector = BlockSelector(shape)
            except Exception:  # pragma: no cover - defensive
                self.selector = None
        self.masks = MaskState(
            shape,
            alpha=alpha,
            dtype=dtype,
            device=device,
            hard=hard,
            threshold=threshold,
            share_head_masks_across_sites=share_head_masks_across_sites,
        )
        self.blocks: List[Any] = []
        self.selection: Optional[Any] = None
        self.sparsity_target: float = 0.0
        self.density_dirty: bool = False
        self.step_count: int = 0
        self.history: List[Dict[str, float]] = []
        self.masks._detect_parameter_size_change()

    # -- selection ----------------------------------------------------------
    def enumerate_blocks(self, salience: Optional[Mapping[str, Any]] = None) -> List[Any]:
        if self.selector is None:
            raise RuntimeError("MaskManager requires an apt.block_selection.BlockSelector")
        self.blocks = self.selector.enumerate_blocks(salience)
        if self.density_dirty and hasattr(self.selector, "recompute_densities"):
            # "Since the number of heads, neurons, and hidden dimensions is
            # ever-changing during pruning, we re-calculate the density after
            # executing each parameter size change." (Appendix C)
            self.selector.recompute_densities(self.blocks, counts=self.masks.counts())
            self.density_dirty = False
        return self.blocks

    def select(
        self,
        salience: Optional[Mapping[str, Any]] = None,
        sparsity: Optional[float] = None,
    ) -> Any:
        blocks = self.enumerate_blocks(salience)
        target = self.sparsity_target if sparsity is None else float(sparsity)
        self.sparsity_target = target
        if hasattr(self.selector, "select_to_sparsity"):
            selection = self.selector.select_to_sparsity(
                blocks,
                target,
                original_param_count=self.masks.original_param_count(),
            )
        else:  # pragma: no cover - defensive
            selection = self.selector.select(
                blocks,
                target_param_count=(1.0 - target) * self.masks.original_param_count(),
            )
        self.selection = selection
        return selection

    # -- one step -----------------------------------------------------------
    def step(
        self,
        model: Optional[torch.nn.Module] = None,
        salience: Optional[Mapping[str, Any]] = None,
        sparsity: Optional[float] = None,
        selection: Optional[Any] = None,
        alpha: Optional[float] = None,
        apply: bool = True,
    ) -> Dict[str, Any]:
        """Perform one mask update (Algorithm 1 body, after the salience EMA)."""
        if selection is None:
            selection = self.select(salience=salience, sparsity=sparsity)
        before = self.masks.counts()
        decay_info = self.masks.apply_selection(selection, alpha=alpha)
        changed = self.masks._detect_parameter_size_change()
        if changed:
            self.density_dirty = True
        if apply and model is not None:
            self.masks.apply_to_model(model)
        self.step_count += 1
        info = {
            "step": self.step_count,
            "sparsity_target": self.sparsity_target,
            "sparsity": self.masks.sparsity(),
            "param_count": self.masks.param_count(),
            "retained_before": before,
            "retained": self.masks.counts(),
            "decayed_retained": decay_info["retained"],
            "decayed_pruned": decay_info["pruned"],
            "parameter_size_changed": bool(changed),
            "needs_optimizer_reset": self.masks.needs_optimizer_reset(),
        }
        self.history.append(
            {
                "step": info["step"],
                "sparsity": info["sparsity"],
                "param_count": info["param_count"],
                "n_heads": info["retained"]["n_heads"],
                "n_neurons": info["retained"]["n_neurons"],
                "d_model": info["retained"]["d_model"],
            }
        )
        return info

    # -- passthroughs -------------------------------------------------------
    def reset(self) -> None:
        self.masks.reset()
        self.masks._detect_parameter_size_change()
        self.selection = None
        self.blocks = []
        self.density_dirty = False

    def needs_optimizer_reset(self) -> bool:
        return self.masks.needs_optimizer_reset()

    def consume_optimizer_reset(self) -> bool:
        return self.masks.consume_optimizer_reset()

    def harden(self, threshold: Optional[float] = None) -> MaskState:
        return self.masks.harden(threshold=threshold)

    def apply_to_model(self, model: torch.nn.Module) -> int:
        return self.masks.apply_to_model(model)

    def sparsity(self) -> float:
        return self.masks.sparsity()

    def param_count(self) -> float:
        return self.masks.param_count()

    @property
    def alpha(self) -> float:
        return self.masks.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.masks.alpha = float(value)

    def summary(self) -> Dict[str, float]:
        return self.masks.summary()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MaskManager({self.masks!r}, target={self.sparsity_target:.3f})"


#: Convenience alias used by the training loop / scripts.
MaskController = MaskManager


def masked_parameter_summary(model: torch.nn.Module) -> Dict[str, float]:
    """Count pruned units / tuning parameters of a wrapped model."""
    pruned = 0
    tuning = 0
    n_modules = 0
    for _name, module in find_masked_linears(model):
        n_modules += 1
        if hasattr(module, "prune_count"):
            pruned += int(module.prune_count())
        if hasattr(module, "num_tuning_parameters"):
            tuning += int(module.num_tuning_parameters())
    return {"masked_linears": float(n_modules), "pruned_units": float(pruned), "tuning_parameters": float(tuning)}


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual check
    try:
        from .block_selection import BlockSelector, roberta_base_shape  # type: ignore

        shape = roberta_base_shape()
        selector = BlockSelector(shape)
    except Exception as exc:
        print(f"[masks] block_selection unavailable ({exc}); running shape-less smoke test")
        selector = None

        class _S:
            n_layers = 12
            n_heads = 12
            n_ffn = 3072
            d_model = 768

        shape = _S()

    manager = MaskManager(shape, selector=selector, alpha=0.01)
    print("[masks] initial:", manager.summary())

    if selector is not None:
        selection = manager.select(sparsity=0.6)
        for _ in range(5):
            info = manager.step(selection=selection)
        print("[masks] after 5 decay steps:", info)
        assert manager.masks.sparsity() < 0.6 + 1e-6
        assert info["needs_optimizer_reset"] in (True, False)
    else:
        selection = {
            "retained": [(HEAD, 0, 0), (NEURON, 0, 0)],
            "pruned": [(HEAD, 1, 3), (NEURON, 1, 5), (DIMENSION, 0, 7)],
        }
        for _ in range(3):
            info = manager.step(selection=selection)
        head0 = manager.masks.head_mask_for(0)[0].item()
        head1 = manager.masks.head_mask_for(1)[3].item()
        dim7 = manager.masks.dim_values[7].item()
        assert abs(head0 - 1.0) < 1e-9, head0
        assert abs(head1 - 0.97) < 1e-6, head1
        assert abs(dim7 - 0.97) < 1e-6, dim7
        print("[masks] decay ok:", info)

    manager.masks.harden()
    print("[masks] hardened summary:", manager.summary())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
