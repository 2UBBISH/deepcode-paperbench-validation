"""APT model wrapper: inject APT adapters into HuggingFace transformer blocks.

Paper references
----------------
* Section 4.1 (APT adapter): ``H_apt(X) = m_o ∘ (W + s · W_B W_A) X ∘ m_i``.
  "In transformer-based LM fine-tuning, we add APT adapters in queries and values of
  multi-head attention (MHA) layers.  We also add APT adapter in feed-forward network
  (FFN) layers when fine-tuning smaller models like RoBERTa and T5 for fast training
  convergence.  In these cases, ``m_i`` prunes transformers' hidden dimension and
  ``m_o`` prunes attention heads in MHA and internal neurons in FFN layers."
* Section 4.3: ranks ``r_apt`` are grown dynamically; new ``W_B`` columns are
  zero-initialised so the layer output stays unchanged.
* Appendix A: adapter ranks start at 8, scaling factor 2, masks gradually decreased
  by ``alpha < 1`` instead of instantly zeroed; pruning then recovery stages.
* Appendix C: block types ``f(b)`` = 0 head / 1 neuron / 2 dimension; gated FFN layers
  (T5, LLaMA-like) contain *three* linear layers, and T5's decoder cross-attention
  layers must also be counted.

This module is the "glue" between the raw HuggingFace model and every other APT
component:

* it locates attention (self / cross) and FFN projections per transformer layer,
* replaces them with :class:`apt.adapters.MaskedLinear` wrappers (with an
  :class:`apt.adapters.APTAdapter` on the tunable ones: q/v/FFN, as in §4.1),
* owns the authoritative pruning masks -- the shared hidden-dimension mask ``m_i``
  and the per-layer head / neuron group masks ``m_o`` -- and pushes them into the
  wrappers (sharing a single tensor object, so an in-place update propagates),
* exposes block metadata (head / neuron / dimension) and parameter counts,
* provides hooks for the activation / gradient caching used by :mod:`apt.salience`.

Mask roles per wrapped projection (``in`` = ``m_i``, ``out`` = ``m_o``)::

    query / key / value   : in=hidden-dim   out=head-groups   (adapter on q, v)
    attn output dense     : in=head-expan.  out=hidden-dim   (mask only)
    ffn in-projection     : in=hidden-dim   out=neuron-groups(adapter)
    ffn out-projection    : in=neurons      out=hidden-dim   (adapter)

The mask-only wrappers (key projections, attention output) are numerically
transparent (all-ones masks, zero adapter) but are required so that physically
removing pruned heads / neurons / hidden dimensions at merge time is exact.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# adapter imports (defensive: the wrapper degrades gracefully in bare installs)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from .adapters import (  # noqa: F401
        BLOCK_TYPES,
        DIMENSION,
        HEAD,
        NEURON,
        APTAdapter,
        MaskedLinear,
        iter_masked_linears,
        make_masked_linear,
        masked_linears_by_layer,
    )
except Exception:  # pragma: no cover
    HEAD, NEURON, DIMENSION = 0, 1, 2
    BLOCK_TYPES = {0: "head", 1: "neuron", 2: "dimension"}
    APTAdapter = None  # type: ignore
    MaskedLinear = None  # type: ignore

    def _fallback_iter_masked_linears(module, names=None):  # type: ignore
        for name, mod in module.named_modules():
            if hasattr(mod, "base_weight") and hasattr(mod, "mask_in"):
                if names is None or name in set(names):
                    yield name, mod

    iter_masked_linears = _fallback_iter_masked_linears  # type: ignore

    def masked_linears_by_layer(module):  # type: ignore
        out: Dict[int, List[Tuple[str, Any]]] = {}
        for name, mod in iter_masked_linears(module):
            out.setdefault(int(getattr(mod, "layer_idx", -1)), []).append((name, mod))
        return out

    def make_masked_linear(*args, **kwargs):  # type: ignore
        raise RuntimeError("apt.adapters is unavailable: cannot build MaskedLinear")


__all__ = [
    "APTModelWrapper",
    "ModelWrapper",
    "wrap_model",
    "unwrapped",
    "is_wrapped",
    "detect_model_type",
    "config_hparams",
    "discover_block_lists",
    "block_layout_for",
    "BlockLayout",
    "LayerInfo",
    "WrapEntry",
    "apt_shape",
    "masked_modules",
    "tunable_modules",
    "wrapped_layer_indices",
    "count_tuning_parameters",
    "count_lm_parameters",
    "enable_salience_cache",
    "enable_grad_capture",
    "clear_salience_caches",
    "restore_base_linears",
    "model_from_config",
    "SITE_SELF",
    "SITE_CROSS",
    "SITE_FFN",
]

SITE_SELF = "self_attn"
SITE_CROSS = "cross_attn"
SITE_FFN = "ffn"

# head-mask site keys (mirrors ``MaskState.head_mask_for(layer, site)``)
HEAD_SITE_SELF = "self"
HEAD_SITE_CROSS = "cross"

# ---------------------------------------------------------------------------
# architecture knowledge
# ---------------------------------------------------------------------------
_MODEL_TYPE_ALIASES = {
    "xlm-roberta": "roberta",
    "xlm_roberta": "roberta",
    "xlmroberta": "roberta",
    "mt5": "t5",
    "t5v1.1": "t5",
    "gptj": "gpt2",
    "mistral": "llama",
    "falcon": "llama",
    "bloom": "opt",
    "deberta-v2": "deberta",
    "deberta_v2": "deberta",
    "mbart": "bart",
    "pegasus": "bart",
    "marian": "bart",
}

# candidate relative paths for the generic (non-T5) layout probe
_Q_CANDIDATES = (
    "attention.self.query",
    "self_attn.q_proj",
    "self_attention.query",
    "attention.q_lin",
    "attn.q_proj",
    "attention.query",
    "self_attn.q",
    "query",
    "q_proj",
    "q_lin",
)
_K_CANDIDATES = (
    "attention.self.key",
    "self_attn.k_proj",
    "self_attention.key",
    "attention.k_lin",
    "attn.k_proj",
    "attention.key",
    "self_attn.k",
    "key",
    "k_proj",
    "k_lin",
)
_V_CANDIDATES = (
    "attention.self.value",
    "self_attn.v_proj",
    "self_attention.value",
    "attention.v_lin",
    "attn.v_proj",
    "attention.value",
    "self_attn.v",
    "value",
    "v_proj",
    "v_lin",
)
_O_CANDIDATES = (
    "attention.output.dense",
    "self_attn.out_proj",
    "self_attn.o_proj",
    "attention.out_lin",
    "attn.c_proj",
    "attention.output",
    "o_proj",
    "out_proj",
)
# groups of FFN input projections (gated FFNs list two)
_FFN_IN_GROUPS = (
    ("intermediate.dense",),
    ("fc1",),
    ("mlp.fc1",),
    ("wi_0", "wi_1"),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("dense_h_to_4h",),
    ("ffn.lin1",),
    ("mlp.c_fc",),
)
_FFN_OUT_CANDIDATES = (
    "output.dense",
    "fc2",
    "mlp.fc2",
    "wo",
    "mlp.down_proj",
    "dense_4h_to_h",
    "ffn.lin2",
    "mlp.c_proj",
)


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------
def _resolve(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    """Resolve a dotted path (numeric parts index ``nn.ModuleList``)."""
    if not dotted:
        return root
    obj: Any = root
    for part in dotted.split("."):
        if obj is None:
            return None
        if isinstance(part, str) and part.isdigit():
            try:
                obj = obj[int(part)]
            except Exception:
                return None
        else:
            obj = getattr(obj, part, None)
    return obj if isinstance(obj, nn.Module) else None


def _set_module(root: nn.Module, dotted: str, new: nn.Module) -> bool:
    """Replace ``root.<dotted>`` by ``new`` (in place)."""
    parts = dotted.split(".")
    parent = _resolve(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if parent is None:
        return False
    attr = parts[-1]
    if attr.isdigit():
        try:
            parent[int(attr)] = new  # type: ignore[index]
            return True
        except Exception:
            return False
    if not hasattr(parent, attr):
        return False
    setattr(parent, attr, new)
    return True


def _first_path(block: nn.Module, candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        mod = _resolve(block, cand)
        if isinstance(mod, nn.Linear):
            return cand
    return None


def _to_mask(values: Any, size: int, like: torch.Tensor, name: str = "") -> Optional[torch.Tensor]:
    """Coerce ``values`` into a flat float mask of ``size`` elements."""
    if values is None:
        return None
    t = torch.as_tensor(values, dtype=like.dtype, device=like.device).reshape(-1)
    if t.numel() != size:
        warnings.warn(
            f"[apt.model_wrapper] mask {name or '<unnamed>'}: expected {size} values, "
            f"got {t.numel()} -- skipped",
            stacklevel=2,
        )
        return None
    return t


def _call(mod: nn.Module, method: str, *args) -> bool:
    fn = getattr(mod, method, None)
    if fn is None:
        return False
    try:
        fn(*args)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"[apt.model_wrapper] {method} failed on {type(mod).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# model / config introspection
# ---------------------------------------------------------------------------
def detect_model_type(model: nn.Module) -> str:
    """Best-effort model family detection (``roberta``/``bert``/``t5``/...)."""
    cfg = getattr(model, "config", None)
    raw = ""
    if cfg is not None:
        for key in ("model_type", "model_type_name", "architectures"):
            val = getattr(cfg, key, None)
            if isinstance(val, (list, tuple)) and val:
                val = val[0]
            if isinstance(val, str) and val:
                raw = val
                break
    if not raw:
        for cls in type(model).__mro__:
            name = cls.__name__.lower()
            for fam in (
                "roberta",
                "distilbert",
                "bert",
                "t5",
                "opt",
                "llama",
                "gpt2",
                "bart",
                "deberta",
                "electra",
            ):
                if fam in name:
                    raw = fam
                    break
            if raw:
                break
    raw = raw.lower()
    if "distil" in raw and "roberta" in raw:
        return "distilbert"
    if "roberta" in raw:
        return "roberta"
    return _MODEL_TYPE_ALIASES.get(raw, raw or "generic")


def _cfg(config, *names, default=None):
    for n in names:
        v = getattr(config, n, None)
        if v is not None:
            return v
    return default


def config_hparams(model: nn.Module) -> Dict[str, Any]:
    """Extract LM shape hyper-parameters from (possibly nested) HF configs."""
    cfg = getattr(model, "config", None)
    if cfg is not None and not hasattr(cfg, "hidden_size") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    d_model = int(_cfg(cfg, "hidden_size", "d_model", "n_embd", "hidden_dim", "dim", default=768))
    n_heads = int(
        _cfg(
            cfg,
            "num_attention_heads",
            "num_heads",
            "n_head",
            "decoder_attention_heads",
            "encoder_attention_heads",
            default=max(1, d_model // 64),
        )
    )
    head_dim = int(_cfg(cfg, "head_dim", "d_kv", default=max(1, d_model // max(1, n_heads))))
    n_ffn = int(
        _cfg(
            cfg,
            "intermediate_size",
            "ffn_dim",
            "d_ff",
            "encoder_ffn_dim",
            "decoder_ffn_dim",
            "n_inner",
            default=4 * d_model,
        )
    )
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_ffn": n_ffn,
        "layers": int(
            _cfg(
                cfg,
                "num_hidden_layers",
                "num_layers",
                "n_layer",
                "encoder_layers",
                "num_decoder_layers",
                default=12,
            )
        ),
        "model_type": detect_model_type(model),
    }


def discover_block_lists(model: nn.Module, model_type: Optional[str] = None) -> Dict[str, List[nn.Module]]:
    """Return ``{"encoder": [...], "decoder": [...]}`` lists of transformer blocks."""
    mt = (model_type or detect_model_type(model)).lower()

    def _mlist(root_path: str) -> List[nn.Module]:
        mod = _resolve(model, root_path)
        if isinstance(mod, (nn.ModuleList, nn.Sequential, list, tuple)):
            return list(mod)
        return []

    lists: Dict[str, List[nn.Module]] = {}
    if mt in ("bert", "roberta", "deberta", "distilbert", "electra"):
        for prefix in ("bert", "roberta", "electra", "distilbert"):
            layers = _mlist(f"{prefix}.encoder.layer")
            if layers:
                lists["encoder"] = layers
                break
        if not lists:
            for path in ("model.encoder.layer", "transformer.layer", "encoder.layer"):
                layers = _mlist(path)
                if layers:
                    lists["encoder"] = layers
                    break
    elif mt == "t5":
        for path in ("encoder.block", "model.encoder.block", "shared.encoder.block"):
            enc = _mlist(path)
            if enc:
                lists["encoder"] = enc
                break
        for path in ("decoder.block", "model.decoder.block"):
            dec = _mlist(path)
            if dec:
                lists["decoder"] = dec
                break
    elif mt in ("opt", "bloom"):
        for path in ("model.decoder.layers", "decoder.layers", "model.layers"):
            layers = _mlist(path)
            if layers:
                lists["encoder"] = layers
                break
    elif mt in ("llama", "mistral"):
        for path in ("model.layers", "model.decoder.layers"):
            layers = _mlist(path)
            if layers:
                lists["encoder"] = layers
                break
    elif mt == "gpt2":
        for path in ("transformer.h", "h"):
            layers = _mlist(path)
            if layers:
                lists["encoder"] = layers
                break
    elif mt == "bart":
        for path in ("model.encoder.layers", "encoder.layers"):
            enc = _mlist(path)
            if enc:
                lists["encoder"] = enc
                break
        for path in ("model.decoder.layers", "decoder.layers"):
            dec = _mlist(path)
            if dec:
                lists["decoder"] = dec
                break

    if not lists:  # generic fallback: first plausible block list
        for _, mod in model.named_modules():
            if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) >= 1:
                first = list(mod)[0]
                if _first_path(first, _Q_CANDIDATES) is not None:
                    lists["encoder"] = list(mod)
                    break
    return lists


# ---------------------------------------------------------------------------
# per-block layout
# ---------------------------------------------------------------------------
@dataclass
class BlockLayout:
    """Relative paths of the wrap targets inside one transformer block."""

    self_attn: Optional[Dict[str, str]] = None
    cross_attn: Optional[Dict[str, str]] = None
    ffn_in: List[str] = field(default_factory=list)
    ffn_out: List[str] = field(default_factory=list)
    gated: bool = False
    ffn_linear_count: int = 2
    attn_linear_count: int = 4

    @property
    def has_ffn(self) -> bool:
        return bool(self.ffn_in or self.ffn_out)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "self_attn": dict(self.self_attn) if self.self_attn else None,
            "cross_attn": dict(self.cross_attn) if self.cross_attn else None,
            "ffn_in": list(self.ffn_in),
            "ffn_out": list(self.ffn_out),
            "gated": self.gated,
            "ffn_linear_count": self.ffn_linear_count,
            "attn_linear_count": self.attn_linear_count,
        }


def _t5_layout(block: nn.Module) -> BlockLayout:
    """T5 block: ``block.layer`` holds SelfAttention / (EncDecAttention) / DenseReluDense."""
    layout = BlockLayout()
    sub = list(getattr(block, "layer", []))
    for i, mod in enumerate(sub):
        has_attn = all(hasattr(mod, a) for a in ("q", "k", "v", "o"))
        has_ffn = hasattr(mod, "wo") and (hasattr(mod, "wi") or hasattr(mod, "wi_0"))
        if has_attn:
            paths = {a: f"layer.{i}.{a}" for a in ("q", "k", "v", "o")}
            if any(_resolve(block, p) is None for p in paths.values()):
                continue
            if layout.self_attn is None:
                layout.self_attn = paths
            elif layout.cross_attn is None:
                layout.cross_attn = paths
        elif has_ffn:
            in_paths: List[str]
            if _resolve(block, f"layer.{i}.wi_0") is not None:
                in_paths = [f"layer.{i}.wi_0", f"layer.{i}.wi_1"]
                layout.gated = True
            else:
                in_paths = [f"layer.{i}.wi"]
            out_path = f"layer.{i}.wo"
            if all(_resolve(block, p) is not None for p in in_paths + [out_path]):
                layout.ffn_in, layout.ffn_out = in_paths, [out_path]
                layout.ffn_linear_count = len(in_paths) + 1
    return layout


def _generic_layout(block: nn.Module) -> BlockLayout:
    layout = BlockLayout()
    q = _first_path(block, _Q_CANDIDATES)
    k = _first_path(block, _K_CANDIDATES)
    v = _first_path(block, _V_CANDIDATES)
    o = _first_path(block, _O_CANDIDATES)
    if q and k and v and o:
        layout.self_attn = {"q": q, "k": k, "v": v, "o": o}
    for group in _FFN_IN_GROUPS:
        if all(_resolve(block, p) is not None for p in group):
            layout.ffn_in = list(group)
            layout.gated = len(group) > 1
            break
    for cand in _FFN_OUT_CANDIDATES:
        mod = _resolve(block, cand)
        if isinstance(mod, nn.Linear) and cand not in (layout.self_attn or {}).values():
            layout.ffn_out = [cand]
            break
    layout.ffn_linear_count = len(layout.ffn_in) + len(layout.ffn_out)
    return layout


def block_layout_for(
    block: nn.Module, model_type: Optional[str] = None, stack: str = "encoder"
) -> Optional[BlockLayout]:
    """Detect the wrap targets of a single transformer block."""
    sub = getattr(block, "layer", None)
    if isinstance(sub, (nn.ModuleList, list, tuple)) and any(hasattr(m, "q") for m in sub):
        return _t5_layout(block)
    layout = _generic_layout(block)
    if not layout.self_attn and not layout.has_ffn:
        return None
    return layout


# ---------------------------------------------------------------------------
# metadata containers
# ---------------------------------------------------------------------------
@dataclass
class LayerInfo:
    """Per-layer block metadata used by salience / selection / distillation."""

    layer: int                      # global layer index
    stack: str                      # "encoder" | "decoder"
    block_index: int
    layout: BlockLayout
    n_heads: int
    head_dim: int
    d_model: int
    n_ffn: int
    ffn_linear_count: int
    gated: bool
    has_cross_attn: bool = False

    @property
    def key(self) -> str:
        return f"{self.stack[0]}{self.block_index}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "stack": self.stack,
            "block_index": self.block_index,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "d_model": self.d_model,
            "n_ffn": self.n_ffn,
            "ffn_linear_count": self.ffn_linear_count,
            "gated": self.gated,
            "has_cross_attn": self.has_cross_attn,
            "layout": self.layout.as_dict(),
        }


@dataclass
class WrapEntry:
    """One injected :class:`MaskedLinear`."""

    name: str
    layer: int
    site: str                    # SITE_SELF | SITE_CROSS | SITE_FFN
    role_in: str                 # "hidden" | "head" | "neuron" | ""
    role_out: str                # "hidden" | "head" | "neuron" | ""
    kind: int                    # HEAD / NEURON (group-mask semantics)
    out_group_size: int
    head_site: str               # "self" | "cross" | ""
    tunable: bool
    module: nn.Module
    filled_features: int = 0     # out_features for neuron blocks, d_model otherwise
    gated: bool = False
    block_ref: Any = None        # the owning transformer block (injection only)

    @property
    def num_out_units(self) -> int:
        return int(self.module.out_features) if isinstance(self.module, nn.Linear) else 0


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------
class APTModelWrapper(nn.Module):
    """Wrap an HF LM with APT adapters + structured pruning masks (§4.1).

    Parameters
    ----------
    model:
        The HuggingFace model to wrap (mutated in place: its linear layers are
        replaced by :class:`MaskedLinear` instances).
    rank, scaling:
        Initial tuning rank ``r_apt`` (8 in Table 6) and LoRA scaling constant
        ``s`` (2, Appendix A).
    model_type:
        Optional override of the detected model family.
    wrap_attn / wrap_ffn:
        Place APT adapters in MHA query/value projections and (for smaller models
        such as RoBERTa / T5) in the FFN layers.
    wrap_cross_attn:
        Also wrap the T5 decoder cross-attention projections.
    wrap_mask_only:
        Additionally wrap key projections and attention output projections with
        *mask-only* adapters so that physically removing pruned heads / neurons /
        hidden dimensions is exact.  These layers are numerically transparent.
    cache_for_salience:
        Enable the activation caches consumed by :mod:`apt.salience`.
    capture_grad:
        Also capture the output gradient of each wrapped projection.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int = 8,
        scaling: float = 2.0,
        model_type: Optional[str] = None,
        wrap_attn: bool = True,
        wrap_ffn: bool = True,
        wrap_cross_attn: bool = True,
        wrap_mask_only: bool = True,
        cache_for_salience: bool = True,
        capture_grad: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_type = (model_type or detect_model_type(model)).lower()
        self.rank = int(rank)
        self.scaling = float(scaling)
        self.wrap_mask_only = bool(wrap_mask_only)
        self.cache_for_salience = bool(cache_for_salience)
        self.capture_grad = bool(capture_grad)
        self.verbose = bool(verbose)

        self.entries: List[WrapEntry] = []
        self.layer_infos: Dict[int, LayerInfo] = {}
        self._head_mask_names: Dict[Tuple[int, str], str] = {}
        self._neuron_mask_names: Dict[int, str] = {}
        self._expanded_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        cfg = config_hparams(model)
        self.d_model = int(cfg["d_model"])
        self.n_heads = int(cfg["n_heads"])
        self.head_dim = int(cfg["head_dim"])
        self.n_ffn = int(cfg["n_ffn"])

        self.register_buffer("dim_mask", torch.ones(self.d_model), persistent=False)
        self.n_layers = 0

        self._build_plan(wrap_attn=wrap_attn, wrap_ffn=wrap_ffn, wrap_cross_attn=wrap_cross_attn)
        self._inject(cache_for_salience=cache_for_salience)
        self.sync_masks()

    # ------------------------------------------------------------------ plan
    def _build_plan(self, *, wrap_attn: bool, wrap_ffn: bool, wrap_cross_attn: bool) -> None:
        blocks = discover_block_lists(self.model, self.model_type)
        if not blocks:
            raise ValueError(
                "[apt.model_wrapper] could not locate transformer blocks; "
                f"pass model_type explicitly (detected '{self.model_type}')"
            )
        cfg = config_hparams(self.model)
        layer_idx = 0
        for stack in ("encoder", "decoder"):
            for i, block in enumerate(blocks.get(stack, [])):
                layout = block_layout_for(block, self.model_type, stack)
                if layout is None:
                    if self.verbose:
                        warnings.warn(f"[apt.model_wrapper] no wrap targets in {stack} block {i}")
                    layer_idx += 1
                    continue
                n_heads = max(1, int(cfg["n_heads"]))
                head_dim = max(1, int(cfg["head_dim"]))
                d_model = max(1, int(cfg["d_model"]))
                n_ffn = max(1, int(cfg["n_ffn"]))
                if layout.self_attn:
                    q_mod = _resolve(block, layout.self_attn["q"])
                    if isinstance(q_mod, nn.Linear) and q_mod.in_features > 0:
                        d_model = int(q_mod.in_features)
                    o_mod = _resolve(block, layout.self_attn["o"])
                    if isinstance(o_mod, nn.Linear) and o_mod.in_features > 0:
                        attn_out = int(o_mod.in_features)
                        if attn_out % n_heads == 0:
                            head_dim = attn_out // n_heads
                if layout.ffn_in:
                    f_in = _resolve(block, layout.ffn_in[0])
                    if isinstance(f_in, nn.Linear) and f_in.out_features > 0:
                        n_ffn = int(f_in.out_features)
                info = LayerInfo(
                    layer=layer_idx,
                    stack=stack,
                    block_index=i,
                    layout=layout,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    d_model=d_model,
                    n_ffn=n_ffn,
                    ffn_linear_count=int(layout.ffn_linear_count),
                    gated=bool(layout.gated),
                    has_cross_attn=bool(wrap_cross_attn and layout.cross_attn),
                )
                self.layer_infos[layer_idx] = info
                self._register_layer_masks(info)
                self._plan_layer(
                    block, info, wrap_attn=wrap_attn, wrap_ffn=wrap_ffn, wrap_cross_attn=wrap_cross_attn
                )
                layer_idx += 1
        self.n_layers = layer_idx

    def _register_layer_masks(self, info: LayerInfo) -> None:
        for head_site, attn in (
            (HEAD_SITE_SELF, info.layout.self_attn),
            (HEAD_SITE_CROSS, info.layout.cross_attn),
        ):
            if not attn:
                continue
            name = f"head_mask_{info.layer}_{head_site}"
            self.register_buffer(name, torch.ones(info.n_heads), persistent=False)
            self._head_mask_names[(info.layer, head_site)] = name
        if info.layout.ffn_in or info.layout.ffn_out:
            name = f"neuron_mask_{info.layer}"
            self.register_buffer(name, torch.ones(info.n_ffn), persistent=False)
            self._neuron_mask_names[info.layer] = name

    def _plan_layer(
        self,
        block: nn.Module,
        info: LayerInfo,
        *,
        wrap_attn: bool,
        wrap_ffn: bool,
        wrap_cross_attn: bool,
    ) -> None:
        def add(
            rel: str,
            site: str,
            role_in: str,
            role_out: str,
            kind: int,
            group_size: int,
            head_site: str,
            tunable: bool,
            filled_features: int = 0,
        ) -> None:
            mod = _resolve(block, rel)
            if not isinstance(mod, nn.Linear):
                return
            self.entries.append(
                WrapEntry(
                    name=rel,
                    layer=info.layer,
                    site=site,
                    role_in=role_in,
                    role_out=role_out,
                    kind=kind,
                    out_group_size=int(group_size),
                    head_site=head_site,
                    tunable=bool(tunable),
                    module=mod,
                    filled_features=int(filled_features or mod.out_features),
                    gated=bool(info.gated),
                    block_ref=block,
                )
            )

        if wrap_attn and info.layout.self_attn:
            attn = info.layout.self_attn
            for key in ("q", "k", "v"):
                add(attn[key], SITE_SELF, "hidden", "head", HEAD, info.head_dim, HEAD_SITE_SELF, key in ("q", "v"))
            if self.wrap_mask_only:
                add(attn["o"], SITE_SELF, "head", "hidden", HEAD, 1, HEAD_SITE_SELF, False, info.d_model)

        if wrap_cross_attn and info.layout.cross_attn:
            cross = info.layout.cross_attn
            for key in ("q", "k", "v"):
                add(
                    cross[key],
                    SITE_CROSS,
                    "hidden",
                    "head",
                    HEAD,
                    info.head_dim,
                    HEAD_SITE_CROSS,
                    key in ("q", "v"),
                )
            if self.wrap_mask_only:
                add(cross["o"], SITE_CROSS, "head", "hidden", HEAD, 1, HEAD_SITE_CROSS, False, info.d_model)

        if wrap_ffn and info.layout.has_ffn:
            for rel in info.layout.ffn_in:
                add(rel, SITE_FFN, "hidden", "neuron", NEURON, 1, "", True, info.n_ffn)
            for rel in info.layout.ffn_out:
                add(rel, SITE_FFN, "neuron", "hidden", NEURON, 1, "", True, info.d_model)

    # --------------------------------------------------------------- inject
    def _inject(self, *, cache_for_salience: bool) -> None:
        """Replace the planned ``nn.Linear`` layers by :class:`MaskedLinear`."""
        named = {id(mod): name for name, mod in self.model.named_modules()}
        new_entries: List[WrapEntry] = []
        for entry in self.entries:
            block: nn.Module = entry.block_ref
            prefix = named.get(id(block), "")
            full = f"{prefix}.{entry.name}" if prefix else entry.name
            wrapped = self._make_wrapper(entry, full_name=full, cache_for_salience=cache_for_salience)
            if wrapped is None:
                continue
            if not _set_module(block, entry.name, wrapped):
                warnings.warn(f"[apt.model_wrapper] failed to replace {full}")
                continue
            if self.verbose:
                print(
                    f"[apt.model_wrapper] wrapped {full} "
                    f"(role_in={entry.role_in}, role_out={entry.role_out}, tunable={entry.tunable})"
                )
            entry.module = wrapped
            entry.name = full
            for attr, value in (
                ("apt_tunable", entry.tunable),
                ("apt_site", entry.site),
                ("apt_role_in", entry.role_in),
                ("apt_role_out", entry.role_out),
            ):
                try:
                    setattr(wrapped, attr, value)
                except Exception:
                    pass
            new_entries.append(entry)
        self.entries = new_entries

    def _make_wrapper(self, entry: WrapEntry, *, full_name: str, cache_for_salience: bool):
        """Build one ``MaskedLinear`` (adapter-bearing when tunable)."""
        linear: nn.Linear = entry.module  # type: ignore[assignment]
        kwargs = dict(
            kind=entry.kind,
            out_group_size=entry.out_group_size,
            layer_idx=entry.layer,
            module_name=full_name,
            cache_for_salience=cache_for_salience,
        )
        if entry.tunable and make_masked_linear is not None:
            try:
                return make_masked_linear(linear, rank=self.rank, scaling=self.scaling, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"[apt.model_wrapper] adapter injection failed for {full_name}: {exc}")
        if MaskedLinear is not None:
            try:
                return MaskedLinear(base_layer=linear, adapter=None, **kwargs)  # type: ignore[misc]
            except Exception:
                pass
            try:
                return mask_only_fallback(linear, self.rank, self.scaling, cache_for_salience, **kwargs)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"[apt.model_wrapper] mask-only wrap failed for {full_name}: {exc}")
        return None

    # ----------------------------------------------------------- mask state
    # ---- accessors -------------------------------------------------------
    def head_mask(self, layer: int, site: str = HEAD_SITE_SELF) -> Optional[torch.Tensor]:
        name = self._head_mask_names.get((layer, site))
        if name is None:
            return None
        return getattr(self, name)

    def neuron_mask(self, layer: int) -> Optional[torch.Tensor]:
        name = self._neuron_mask_names.get(layer)
        if name is None:
            return None
        return getattr(self, name)

    def expanded_head_mask(self, layer: int, site: str = HEAD_SITE_SELF) -> Optional[torch.Tensor]:
        """Head-group mask expanded to the ``d_model`` (concatenated-head) layout."""
        info = self.layer_infos.get(layer)
        hm = self.head_mask(layer, site)
        if info is None or hm is None:
            return None
        if info.n_heads * info.head_dim != info.d_model:
            return None
        return hm.repeat_interleave(info.head_dim)

    def head_mask_names(self) -> Dict[Tuple[int, str], str]:
        return dict(self._head_mask_names)

    def neuron_mask_names(self) -> Dict[int, str]:
        return dict(self._neuron_mask_names)

    # ---- writers ---------------------------------------------------------
    def set_dim_mask(self, values: Any) -> bool:
        t = _to_mask(values, self.d_model, self.dim_mask, "dim_mask")
        if t is None:
            return False
        with torch.no_grad():
            self.dim_mask.copy_(t)
        return True

    def set_head_mask(self, layer: int, values: Any, site: str = HEAD_SITE_SELF) -> bool:
        name = self._head_mask_names.get((layer, site))
        if name is None:
            return False
        buf = getattr(self, name).reshape(-1)
        t = _to_mask(values, buf.numel(), buf, f"head_mask[{layer},{site}]")
        if t is None:
            return False
        with torch.no_grad():
            buf.copy_(t)
        return True

    def set_head_mask_by_name(self, buffer_name: str, values: Any) -> bool:
        """Set a (possibly multi-site) head mask buffer by its own name."""
        buf = getattr(self, buffer_name, None)
        if buf is None:
            return False
        buf = buf.reshape(-1)
        t = _to_mask(values, buf.numel(), buf, buffer_name)
        if t is None:
            return False
        with torch.no_grad():
            buf.copy_(t)
        return True

    def set_neuron_mask(self, layer: int, values: Any) -> bool:
        name = self._neuron_mask_names.get(layer)
        if name is None:
            return False
        buf = getattr(self, name).reshape(-1)
        t = _to_mask(values, buf.numel(), buf, f"neuron_mask[{layer}]")
        if t is None:
            return False
        with torch.no_grad():
            buf.copy_(t)
        return True

    def set_head_masks(self, layer: int, self_mask: Any = None, cross_mask: Any = None) -> None:
        if self_mask is not None:
            self.set_head_mask(layer, self_mask, HEAD_SITE_SELF)
        if cross_mask is not None:
            self.set_head_mask(layer, cross_mask, HEAD_SITE_CROSS)

    def set_neuron_masks(self, values: Dict[int, Any]) -> None:
        for layer, v in values.items():
            self.set_neuron_mask(layer, v)

    def set_mask_by_name(self, name: str, values: Any) -> bool:
        """Generic mask writer (accepts dim / head / neuron buffer names)."""
        if name in ("dim", "dim_mask", "hidden"):
            return self.set_dim_mask(values)
        if name.startswith("head_mask_"):
            return self.set_head_mask_by_name(name, values)
        if name.startswith("neuron_mask_"):
            return self.set_head_mask_by_name(name, values)
        if name.isdigit():
            return self.set_neuron_mask(int(name), values)
        return False

    # ---- synchronisation -------------------------------------------------
    def sync_masks(self, *, hard: bool = False, threshold: float = 0.5) -> None:
        """Push the owned masks into every wrapped module (sharing tensors)."""
        if hard:
            with torch.no_grad():
                self.dim_mask.copy_((self.dim_mask >= threshold).to(self.dim_mask.dtype))
                for name in self._head_mask_names.values():
                    buf = getattr(self, name).reshape(-1)
                    buf.copy_((buf >= threshold).to(buf.dtype))
                for name in self._neuron_mask_names.values():
                    buf = getattr(self, name).reshape(-1)
                    buf.copy_((buf >= threshold).to(buf.dtype))
        self._expanded_cache = {}
        for entry in self.entries:
            mod = entry.module
            if entry.role_in == "hidden":
                _call(mod, "set_input_mask", self.dim_mask)
            elif entry.role_in == "neuron":
                nm = self.neuron_mask(entry.layer)
                if nm is not None:
                    _call(mod, "set_input_mask", nm)
            elif entry.role_in == "head":
                exp = self.expanded_head_mask(entry.layer, entry.head_site or HEAD_SITE_SELF)
                if exp is not None:
                    self._expanded_cache[(entry.layer, entry.head_site)] = exp
                else:
                    # head_dim * n_heads != d_model (e.g. QKVO with d_head != d_model/n_heads):
                    # fall back to broadcasting the head mask over the projection output.
                    exp = self.head_mask(entry.layer, entry.head_site or HEAD_SITE_SELF)
                if exp is not None:
                    _call(mod, "set_input_mask", exp)
            if entry.role_out == "head":
                hm = self.head_mask(entry.layer, entry.head_site or HEAD_SITE_SELF)
                if hm is not None:
                    _call(mod, "set_output_group_mask", hm)
            elif entry.role_out == "neuron":
                nm = self.neuron_mask(entry.layer)
                if nm is not None:
                    _call(mod, "set_output_group_mask", nm)
            elif entry.role_out == "hidden":
                _call(mod, "set_output_mask", self.dim_mask)

    def apply_masks(
        self,
        dim_mask: Any = None,
        head_masks: Optional[Dict[Any, Any]] = None,
        neuron_masks: Optional[Dict[Any, Any]] = None,
        *,
        hard: bool = False,
        threshold: float = 0.5,
    ) -> None:
        """Set masks by value and propagate them.

        ``head_masks`` accepts keys ``(layer, site)``, ``layer`` (self-attention)
        or strings such as ``"self@0"``.
        """
        if dim_mask is not None:
            self.set_dim_mask(dim_mask)
        for key, values in (head_masks or {}).items():
            if isinstance(key, tuple):
                self.set_head_mask(int(key[0]), values, str(key[1]))
            elif isinstance(key, int):
                self.set_head_mask(int(key), values, HEAD_SITE_SELF)
            elif isinstance(key, str) and "@" in key:
                site, layer = key.split("@", 1)
                self.set_head_mask(int(layer), values, site)
            elif isinstance(key, str):
                try:
                    self.set_head_mask(int(key), values, HEAD_SITE_SELF)
                except (TypeError, ValueError):
                    continue
        for key, values in (neuron_masks or {}).items():
            self.set_neuron_mask(int(key), values)
        self.sync_masks(hard=hard, threshold=threshold)

    def decay_masks(self, alpha: float = 0.01, *, threshold: float = 0.5) -> None:
        """Gradual mask update (Appendix C: ``alpha = 0.01``).

        Values above ``threshold`` are pushed towards 1, values below towards 0,
        instead of being instantly set to 0.
        """
        with torch.no_grad():
            for buf in self._all_mask_buffers():
                upper = (buf >= threshold).to(buf.dtype)
                buf.copy_(torch.clamp(buf + alpha * (2.0 * upper - 1.0), 0.0, 1.0))
        self.sync_masks()

    def harden_masks(self, threshold: float = 0.5) -> None:
        """Snap every mask to 0/1 (used before merge / inference)."""
        self.sync_masks(hard=True, threshold=threshold)

    def _all_mask_buffers(self) -> List[torch.Tensor]:
        bufs = [self.dim_mask.reshape(-1)]
        bufs += [getattr(self, n).reshape(-1) for n in self._head_mask_names.values()]
        bufs += [getattr(self, n).reshape(-1) for n in self._neuron_mask_names.values()]
        return bufs

    # ---- mask state exchange --------------------------------------------
    def apply_mask_state(self, mask_state: Any, *, hard: bool = False, threshold: float = 0.5) -> None:
        """Push a :class:`apt.masks.MaskState` into the wrapped modules.

        Handles both the flat global layout (``n_layers * n_heads`` head values,
        one hidden-dim mask shared by all layers) and a per-layer stacked layout.
        """
        get_dim = getattr(mask_state, "dim_mask_for", None)
        if callable(get_dim):
            values = None
            try:
                values = get_dim(self.d_model)
            except TypeError:
                try:
                    values = get_dim()
                except Exception:
                    values = None
            except Exception:
                values = None
            if values is not None:
                t = torch.as_tensor(values).reshape(-1)
                if t.numel() == self.d_model:
                    self.set_dim_mask(t)
                elif t.numel() > self.d_model:
                    self.set_dim_mask(t[: self.d_model])

        multi_site = self.n_layers > 0 and self._is_multi_site()
        for layer in sorted(self.layer_infos):
            get_heads = getattr(mask_state, "head_mask_for", None)
            if callable(get_heads) and (layer, HEAD_SITE_SELF) in self._head_mask_names:
                self._read_head_values(layer, get_heads, multi_site)
            get_neurons = getattr(mask_state, "neuron_mask_for", None)
            if callable(get_neurons) and layer in self._neuron_mask_names:
                try:
                    values = get_neurons(layer)
                except Exception:
                    values = None
                if values is not None:
                    self.set_neuron_mask(layer, values)
        self.sync_masks(hard=hard, threshold=threshold)

    def _is_multi_site(self) -> bool:
        """True if every head mask buffer holds *all* attention sites at once."""
        sites = set(self._head_mask_names)
        n_self = sum(1 for (_, s) in sites if s == HEAD_SITE_SELF)
        if n_self == 0:
            return False
        # a multi-site layout registers one buffer per (layer, site) pair and
        # MaskState hands out stacked values for every site at once
        return False

    def _read_head_values(self, layer: int, get_heads, multi_site: bool) -> None:
        values = None
        for probe in (HEAD_SITE_SELF, "self", "query", None):
            try:
                values = get_heads(layer, probe) if probe is not None else get_heads(layer)
            except TypeError:
                try:
                    values = get_heads(layer)
                except Exception:
                    values = None
            except Exception:
                values = None
            if values is not None:
                break
        if values is None:
            return
        t = torch.as_tensor(values).reshape(-1)
        n_self = int(self.head_mask(layer, HEAD_SITE_SELF).numel())
        if multi_site:
            per_site = n_self
            n_sites = t.numel() // per_site if per_site else 0
            for site_idx, site in enumerate((HEAD_SITE_SELF, HEAD_SITE_CROSS)):
                if site_idx >= n_sites or (layer, site) not in self._head_mask_names:
                    continue
                chunk = t[site_idx * per_site : (site_idx + 1) * per_site]
                self.set_head_mask(layer, chunk, site)
            return
        if t.numel() == n_self:
            self.set_head_mask(layer, t, HEAD_SITE_SELF)
        elif t.numel() % n_self == 0:
            n_sites = t.numel() // n_self
            self.set_head_mask(layer, t[:n_self], HEAD_SITE_SELF)
            if n_sites > 1 and (layer, HEAD_SITE_CROSS) in self._head_mask_names:
                self.set_head_mask(layer, t[n_self : 2 * n_self], HEAD_SITE_CROSS)
        elif t.numel() % max(1, self.n_layers) == 0:
            # flat global layout: slice out this layer's value range
            per_layer = t.numel() // max(1, self.n_layers)
            if per_layer == n_self:
                chunk = t[layer * per_layer : (layer + 1) * per_layer]
                self.set_head_mask(layer, chunk, HEAD_SITE_SELF)

    def apply_selection(self, selection: Any, *, global_head_masks: Optional[torch.Tensor] = None,
                        neuron_masks: Optional[Dict[int, Any]] = None) -> None:
        """Convenience bridge from a block-selection result to this wrapper."""
        if global_head_masks is not None:
            g = torch.as_tensor(global_head_masks).reshape(-1)
            for layer in sorted(self.layer_infos):
                n = int(self.head_mask(layer, HEAD_SITE_SELF).numel())
                if g.numel() >= (layer + 1) * n:
                    self.set_head_mask(layer, g[layer * n : (layer + 1) * n], HEAD_SITE_SELF)
        if neuron_masks:
            for layer, values in neuron_masks.items():
                self.set_neuron_mask(int(layer), values)
        self.sync_masks()

    def mask_snapshot(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Serialisable copy of all masks (optionally hardened)."""

        def _b(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().clone().reshape(-1)
            return (t >= threshold).to(t.dtype) if threshold is not None else t

        return {
            "dim": _b(self.dim_mask),
            "heads": {
                f"{layer}@{site}": _b(getattr(self, name))
                for (layer, site), name in self._head_mask_names.items()
            },
            "neurons": {str(layer): _b(getattr(self, name)) for layer, name in self._neuron_mask_names.items()},
        }

    def load_mask_snapshot(self, snap: Dict[str, Any]) -> None:
        if "dim" in snap:
            self.set_dim_mask(snap["dim"])
        for key, values in (snap.get("heads") or {}).items():
            site, layer = key.split("@", 1)
            self.set_head_mask(int(layer), values, site)
        for layer, values in (snap.get("neurons") or {}).items():
            self.set_neuron_mask(int(layer), values)
        self.sync_masks()

    # ---- bookkeeping -----------------------------------------------------
    def retained_counts(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Counts of retained heads / neurons / hidden dims (hard threshold)."""
        return {
            "dim": int((self.dim_mask >= threshold).sum().item()),
            "heads": {
                f"{layer}@{site}": int((getattr(self, name) >= threshold).sum().item())
                for (layer, site), name in self._head_mask_names.items()
            },
            "neurons": {
                layer: int((getattr(self, name) >= threshold).sum().item())
                for layer, name in self._neuron_mask_names.items()
            },
        }

    def layer_keep_flags(self, threshold: float = 0.5) -> List[int]:
        """1 if a transformer layer still has retained heads / neurons (for phi)."""
        flags: List[int] = []
        for layer, info in sorted(self.layer_infos.items()):
            alive = True
            hm = self.head_mask(layer, HEAD_SITE_SELF)
            if hm is not None:
                alive = alive and bool((hm >= threshold).any().item())
            nm = self.neuron_mask(layer)
            if nm is not None and nm.numel() == info.n_ffn:
                alive = alive and bool((nm >= threshold).any().item())
            flags.append(1 if alive else 0)
        return flags

    def sparsity(self, threshold: float = 0.5, block_size: float = 64.0) -> Dict[str, float]:
        """Fraction of pruned heads / neurons / dims and the total parameter sparsity.

        The ``total`` entry follows the paper's definition (ratio of pruned
        parameter size to total parameter size, rounded to hardware-friendly
        multiples of ``block_size`` columns as in §5.3).
        """
        total_heads = sum(info.n_heads for info in self.layer_infos.values())
        total_neurons = sum(info.n_ffn for info in self.layer_infos.values())
        kept_heads = 0
        for layer in self.layer_infos:
            hm = self.head_mask(layer, HEAD_SITE_SELF)
            if hm is not None:
                kept_heads += int((hm >= threshold).sum().item())
        kept_neurons = 0
        for layer, info in self.layer_infos.items():
            nm = self.neuron_mask(layer)
            if nm is not None and nm.numel() == info.n_ffn:
                kept_neurons += int((nm >= threshold).sum().item())

        def _round(x: float) -> float:
            return float(math.ceil(x / block_size) * block_size)

        kept_dim = _round(float((self.dim_mask >= threshold).sum().item()))
        kept_dim = min(kept_dim, float(self.d_model))
        out = {
            "head": 1.0 - (kept_heads / total_heads) if total_heads else 0.0,
            "neuron": 1.0 - (kept_neurons / total_neurons) if total_neurons else 0.0,
            "dim": 1.0 - (kept_dim / self.d_model) if self.d_model else 0.0,
        }
        # total LM-parameter sparsity using the paper's estimate (eq. 4 / App. C)
        kept_param = 0.0
        full_param = 0.0
        for _, info in sorted(self.layer_infos.items()):
            hm = self.head_mask(info.layer, HEAD_SITE_SELF)
            nm = self.neuron_mask(info.layer)
            h_kept = float((hm >= threshold).sum().item()) if hm is not None else info.n_heads
            n_kept = float((nm >= threshold).sum().item()) if nm is not None else info.n_ffn
            attn_sites = info.n_heads + (info.n_heads if info.has_cross_attn else 0)
            attn_sites_kept = h_kept + (h_kept if info.has_cross_attn else 0)
            kept_param += kept_dim * (
                4.0 * attn_sites_kept * info.head_dim + info.ffn_linear_count * n_kept
            )
            full_param += info.d_model * (
                4.0 * attn_sites * info.head_dim + info.ffn_linear_count * info.n_ffn
            )
        out["total"] = max(0.0, 1.0 - kept_param / full_param) if full_param else 0.0
        return out

    def block_metadata(self) -> Dict[str, Any]:
        """Per-layer block metadata (heads / neurons / dims) for selection."""
        return {
            "model_type": self.model_type,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "ffn_linear_count": self.ffn_linear_count(),
            "attn_linear_count": 4,
            "n_cross_attn_layers": self.n_cross_attn_layers(),
            "sites": ("query", "value"),
            "layers": {layer: info.as_dict() for layer, info in self.layer_infos.items()},
        }

    def ffn_linear_count(self) -> int:
        return max([info.ffn_linear_count for info in self.layer_infos.values()] or [2])

    def n_cross_attn_layers(self) -> int:
        return sum(1 for i in self.layer_infos.values() if i.has_cross_attn)

    # ---- parameter accounting -------------------------------------------
    def tuned_parameters(self) -> int:
        return count_tuning_parameters(self.model)

    def lm_parameters(self) -> int:
        return count_lm_parameters(self.model)

    def num_adapters(self) -> int:
        return len(tunable_modules(self.model))

    def num_masked_linears(self) -> int:
        return len(list(iter_masked_linears(self.model)))

    def name_parameters(self) -> Dict[str, int]:
        return {
            name: int(p.numel())
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

    # ---- caching / hooks -------------------------------------------------
    def set_salience_caching(self, flag: bool = True, *, capture_grad: Optional[bool] = None) -> None:
        self.cache_for_salience = bool(flag)
        enable_salience_cache(self.model, flag)
        if capture_grad is not None:
            enable_grad_capture(self.model, capture_grad)
            self.capture_grad = bool(capture_grad)

    def clear_caches(self) -> None:
        clear_salience_caches(self.model)

    # ---- plumbing --------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _apply(self, fn, recurse: bool = True):  # noqa: D401
        super()._apply(fn, recurse=recurse)
        try:  # buffers were recreated -> re-share the mask tensor objects
            self.sync_masks()
        except Exception:  # pragma: no cover
            pass
        return self

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            inner = modules.get("model") if modules else None
            if inner is not None:
                try:
                    return getattr(inner, name)
                except AttributeError:
                    pass
            raise

    def train(self, mode: bool = True):  # noqa: D401
        super().train(mode)
        return self

    def extra_repr(self) -> str:
        return (
            f"model_type={self.model_type}, n_layers={getattr(self, 'n_layers', 0)}, "
            f"d_model={self.d_model}, n_heads={self.n_heads}, n_ffn={self.n_ffn}, "
            f"adapters={self.num_adapters()}, rank={self.rank}, scaling={self.scaling}"
        )


class ModelWrapper(APTModelWrapper):
    """Alias kept for readability in downstream code."""


# ---------------------------------------------------------------------------
# helpers used during injection
# ---------------------------------------------------------------------------
def mask_only_fallback(linear: nn.Linear, rank: int, scaling: float, cache: bool, **kwargs):
    """Build a frozen-adapter ``MaskedLinear`` (adapter contributes nothing).

    Used when ``MaskedLinear(adapter=None)`` is unsupported: the adapter is
    created normally and then frozen; ``W_B`` is zero-initialised, so the
    contribution ``s·W_B W_A`` stays exactly zero.
    """
    mod = make_masked_linear(linear, rank=rank, scaling=scaling, **kwargs)
    adapter = getattr(mod, "adapter", None)
    if adapter is not None:
        for p in adapter.parameters():
            p.requires_grad_(False)
    try:
        mod.apt_frozen_adapter = True  # type: ignore[attr-defined]
    except Exception:
        pass
    return mod


# ---------------------------------------------------------------------------
# module-level API
# ---------------------------------------------------------------------------
def wrap_model(model: nn.Module, **kwargs) -> APTModelWrapper:
    """Wrap ``model`` in place (idempotent)."""
    if isinstance(model, APTModelWrapper):
        return model
    return APTModelWrapper(model, **kwargs)


def unwrapped(model: nn.Module) -> nn.Module:
    """Return the inner HF model whether or not it is wrapped."""
    return model.model if isinstance(model, APTModelWrapper) else model


def is_wrapped(model: nn.Module) -> bool:
    if isinstance(model, APTModelWrapper):
        return True
    return bool(list(iter_masked_linears(model)))


def masked_modules(model: nn.Module) -> Dict[str, nn.Module]:
    return {name: mod for name, mod in iter_masked_linears(model)}


def tunable_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Wrapped modules carrying a *trainable* APT adapter."""
    out: List[Tuple[str, nn.Module]] = []
    for name, mod in iter_masked_linears(model):
        if getattr(mod, "apt_frozen_adapter", False):
            continue
        if getattr(mod, "apt_tunable", None) is False:
            continue
        adapter = getattr(mod, "adapter", None)
        if adapter is None:
            continue
        params = [p for p in adapter.parameters()]
        if params and not any(p.requires_grad for p in params):
            continue
        if getattr(adapter, "rank", 0) == 0:
            continue
        out.append((name, mod))
    return out


def wrapped_layer_indices(model: nn.Module) -> List[int]:
    if isinstance(model, APTModelWrapper):
        return sorted(model.layer_infos)
    return sorted(masked_linears_by_layer(model).keys())


def count_tuning_parameters(model: nn.Module) -> int:
    """Total number of *tuning* (adapter) parameters."""
    total = 0
    seen = set()
    for _, mod in iter_masked_linears(model):
        if getattr(mod, "apt_frozen_adapter", False):
            continue
        adapter = getattr(mod, "adapter", None)
        if adapter is None or id(adapter) in seen:
            continue
        seen.add(id(adapter))
        fn = getattr(adapter, "num_tuning_parameters", None)
        if callable(fn):
            try:
                total += int(fn())
                continue
            except Exception:
                pass
        total += sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    return total


def count_lm_parameters(model: nn.Module) -> int:
    """Frozen LM parameters held by the wrapped projections (pre-pruning size)."""
    total = 0
    for _, mod in iter_masked_linears(model):
        for attr in ("base_weight", "base_bias"):
            p = getattr(mod, attr, None)
            if isinstance(p, torch.Tensor):
                total += int(p.numel())
    return total


def enable_salience_cache(model: nn.Module, flag: bool = True) -> None:
    """Toggle activation caching on every wrapped module."""
    for _, mod in iter_masked_linears(model):
        try:
            setattr(mod, "cache_for_salience", bool(flag))
            if not flag:
                clear_salience_caches(mod)
        except Exception:
            continue


def enable_grad_capture(model: nn.Module, flag: bool = True) -> None:
    """Toggle output-gradient capture (used by the salience scorer)."""
    for _, mod in iter_masked_linears(model):
        try:
            setattr(mod, "_capture_grad", bool(flag))
        except Exception:
            continue


def clear_salience_caches(model: nn.Module) -> None:
    for _, mod in iter_masked_linears(model):
        for cache_name in ("_cached_input", "_cached_output", "_cached_output_grad"):
            if hasattr(mod, cache_name):
                try:
                    setattr(mod, cache_name, None)
                except Exception:
                    continue


def restore_base_linears(model: nn.Module, *, merge: bool = True, threshold: float = 0.0) -> nn.Module:
    """Replace every ``MaskedLinear`` by a plain ``nn.Linear``.

    With ``merge=True`` the (masked) adapter product is folded into the weight,
    reproducing the paper's "tuning parameters can be fully merged after
    training" property.  The resulting linear layers keep the *un-pruned* shapes;
    physically removing pruned indices is :mod:`apt.merge`'s job.
    """
    for name, mod in list(iter_masked_linears(model)):
        weight = None
        if merge:
            fn = getattr(mod, "merged_weight", None)
            if callable(fn):
                for arg in (threshold,):
                    try:
                        weight = fn(arg)
                        break
                    except Exception:
                        weight = None
                if weight is None:
                    try:
                        weight = fn()
                    except Exception:
                        weight = None
        if weight is None:
            weight = getattr(mod, "base_weight", None)
        if weight is None:
            base = getattr(mod, "base_layer", None)
            weight = getattr(base, "weight", None)
        if weight is None:
            continue
        bias = getattr(mod, "base_bias", None)
        if bias is None:
            base = getattr(mod, "base_layer", None)
            bias = getattr(base, "bias", None)
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        with torch.no_grad():
            linear.weight.copy_(weight.detach())
            if bias is not None and linear.bias is not None:
                linear.bias.copy_(bias.detach().reshape(-1))
        linear.to(weight.device)
        if not _set_module(model, name, linear):
            warnings.warn(f"[apt.model_wrapper] could not restore {name}")
    return model


def model_from_config(config, model_type: Optional[str] = None):
    """Instantiate the matching HF model class for a config (best effort)."""
    try:
        from transformers import AutoModel, AutoModelForSeq2SeqLM  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to build models from configs") from exc
    mt = (model_type or getattr(config, "model_type", "") or "").lower()
    if mt in ("t5", "mt5"):
        return AutoModelForSeq2SeqLM.from_config(config)
    return AutoModel.from_config(config)


def apt_shape(model_or_wrapper: nn.Module, model_type: Optional[str] = None):
    """Build a :class:`apt.block_selection.ModelShape` for a wrapped model."""
    from .block_selection import ModelShape  # local import (avoids cycles)

    if isinstance(model_or_wrapper, APTModelWrapper):
        wrapper = model_or_wrapper
        info0 = next(iter(wrapper.layer_infos.values()))
        return ModelShape(
            d_model=wrapper.d_model,
            n_layers=wrapper.n_layers,
            n_heads=max(1, wrapper.n_heads),
            n_ffn=int(info0.n_ffn),
            ffn_linear_count=int(wrapper.ffn_linear_count()),
            attn_linear_count=4,
            n_cross_attn_layers=int(wrapper.n_cross_attn_layers()),
            model_type=wrapper.model_type,
        )
    md = masked_linears_by_layer(model_or_wrapper)
    if not md:
        raise ValueError("[apt.model_wrapper] apt_shape needs a wrapped model")
    cfg = config_hparams(model_or_wrapper)
    return ModelShape(
        d_model=int(cfg["d_model"]),
        n_layers=max(md.keys()) + 1,
        n_heads=int(cfg["n_heads"]),
        n_ffn=int(cfg["n_ffn"]),
        ffn_linear_count=2,
        attn_linear_count=4,
        n_cross_attn_layers=0,
        model_type=(model_type or cfg["model_type"]),
    )


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _fake_bert(n_layers: int = 2, d_model: int = 32, n_heads: int = 4, n_ffn: int = 64):
    """Build a minimal BERT-shaped module (attribute names follow HF)."""

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = nn.Module()
            self.self.query = nn.Linear(d_model, d_model)
            self.self.key = nn.Linear(d_model, d_model)
            self.self.value = nn.Linear(d_model, d_model)
            self.output = nn.Module()
            self.output.dense = nn.Linear(d_model, d_model)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.intermediate = nn.Module()
            self.intermediate.dense = nn.Linear(d_model, n_ffn)
            self.output = nn.Module()
            self.output.dense = nn.Linear(n_ffn, d_model)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([Layer() for _ in range(n_layers)])

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace

            self.bert = Inner()
            self.config = SimpleNamespace(
                model_type="bert",
                hidden_size=d_model,
                num_attention_heads=n_heads,
                intermediate_size=n_ffn,
                num_hidden_layers=n_layers,
            )

        def forward(self, x):
            h = x
            for layer in self.bert.encoder.layer:
                q = layer.attention.self.query(h)
                k = layer.attention.self.key(h)
                v = layer.attention.self.value(h)
                scores = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
                h = h + layer.attention.output.dense(scores @ v)
                h = h + layer.output.dense(torch.relu(layer.intermediate.dense(h)))
            return h

    return Model()


def _fake_t5(n_layers: int = 2, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, gated: bool = True):
    """Minimal T5-shaped encoder-decoder module (self+cross attention, gated FFN)."""

    class SelfAttn(nn.Module):
        def __init__(self, cross=False):
            super().__init__()
            self.is_decoder = cross
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            self.o = nn.Linear(d_model, d_model)

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            if gated:
                self.wi_0 = nn.Linear(d_model, d_ff)
                self.wi_1 = nn.Linear(d_model, d_ff)
            else:
                self.wi = nn.Linear(d_model, d_ff)
            self.wo = nn.Linear(d_ff, d_model)

    class Block(nn.Module):
        def __init__(self, decoder=False):
            super().__init__()
            layers = [SelfAttn()] if not decoder else [SelfAttn(cross=True), SelfAttn(cross=True)]
            layers.append(FFN())
            self.layer = nn.ModuleList(layers)

    class Stack(nn.Module):
        def __init__(self, decoder=False):
            super().__init__()
            self.block = nn.ModuleList([Block(decoder) for _ in range(n_layers)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace

            self.encoder = Stack()
            self.decoder = Stack(decoder=True)
            self.config = SimpleNamespace(
                model_type="t5",
                d_model=d_model,
                hidden_size=d_model,
                num_heads=n_heads,
                num_attention_heads=n_heads,
                d_kv=d_model // n_heads,
                d_ff=d_ff,
                intermediate_size=d_ff,
                num_layers=n_layers,
                num_hidden_layers=n_layers,
            )

        def forward(self, x):
            h = x
            for block in self.encoder.block:
                sa = block.layer[0]
                h = h + sa.o(sa.v(h))
                ff = block.layer[1]
                if gated:
                    h = h + ff.wo(ff.wi_1(h) * torch.relu(ff.wi_0(h)))
                else:
                    h = h + ff.wo(torch.relu(ff.wi(h)))
            for block in self.decoder.block:
                sa = block.layer[0]
                h = h + sa.o(sa.v(h))
                ca = block.layer[1]
                h = h + ca.o(ca.v(h))
                ff = block.layer[2]
                if gated:
                    h = h + ff.wo(ff.wi_1(h) * torch.relu(ff.wi_0(h)))
                else:
                    h = h + ff.wo(torch.relu(ff.wi(h)))
            return h

    return Model()


def _self_test() -> bool:
    ok = True
    torch.manual_seed(0)

    # ---------------- generic encoder with transformer blocks ----------------
    model = _fake_bert()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        ref = model(x)

    wrapper = APTModelWrapper(model, rank=4, scaling=2.0)
    wrapped = wrapper.model

    # 1) wrapping must be numerically transparent (all-ones masks, zero W_B)
    with torch.no_grad():
        out = wrapped(x)
    diff = (out - ref).abs().max().item()
    print(f"[test] transparent wrap max|diff| = {diff:.3e}")
    ok = ok and diff < 1e-6

    # 2) adapter / mask bookkeeping (q, v, ffn_in, ffn_out per layer)
    n_adapters = wrapper.num_adapters()
    n_masked = wrapper.num_masked_linears()
    print(f"[test] adapters = {n_adapters}, masked linears = {n_masked}, tuning params = {wrapper.tuned_parameters()}")
    ok = ok and n_adapters == 4 * 2
    ok = ok and n_masked == 6 * 2
    ok = ok and wrapper.tuned_parameters() == 4 * 2 * (4 * 32 + 32 * 4)

    # 3) shared hidden-dimension mask propagates in place
    wrapper.dim_mask[:4] = 0.0
    q0 = wrapper.entries[0].module
    ok = ok and float(q0.mask_in[:4].abs().sum()) == 0.0

    # 4) head pruning changes the output and is reported
    wrapper.set_head_mask(0, torch.tensor([0.0, 1.0, 1.0, 1.0]))
    with torch.no_grad():
        out2 = wrapped(x)
    ok = ok and (out2 - ref).abs().max().item() > 1e-6
    sp = wrapper.sparsity()
    print(f"[test] sparsity after pruning head 0 and 4 dims: {sp}")
    ok = ok and abs(sp["head"] - 1.0 / (2 * 4)) < 1e-6
    ok = ok and abs(sp["dim"] - 4.0 / 32) < 1e-6
    ok = ok and sp["total"] > 0.0

    # 5) gradual decay pushes masks towards 0 / 1 (alpha = 0.01)
    before = float(wrapper.head_mask(0)[0].item())
    wrapper.decay_masks(alpha=0.01)
    after = float(wrapper.head_mask(0)[0].item())
    ok = ok and after < before

    # 6) harden + metadata
    wrapper.harden_masks()
    ok = ok and float(wrapper.head_mask(0)[0].item()) == 0.0
    meta = wrapper.block_metadata()
    ok = ok and meta["layers"][0]["n_heads"] == 4 and meta["layers"][0]["n_ffn"] == 64
    ok = ok and wrapper.ffn_linear_count() == 2

    # 7) rank growth keeps the adapter output unchanged (§4.3)
    q_mod = wrapper.entries[0].module
    probe = torch.randn(2, 32)
    with torch.no_grad():
        before_out = q_mod.adapter(probe)
    q_mod.increase_rank(q_mod.rank + 4)
    with torch.no_grad():
        after_out = q_mod.adapter(probe)
    print(f"[test] rank 4 -> {q_mod.rank}, max|delta| = {(after_out - before_out).abs().max().item():.3e}")
    ok = ok and q_mod.rank == 8

    # 8) shape object for block selection
    shape = apt_shape(wrapper)
    print(f"[test] shape d_model={shape.d_model}, n_layers={shape.n_layers}, C_head={shape.head_block_cost():.0f}")
    ok = ok and shape.d_model == 32 and shape.n_layers == 2

    # 9) mask snapshot round-trip
    snap = wrapper.mask_snapshot(threshold=0.5)
    wrapper.apply_masks(dim_mask=torch.ones(32), head_masks={(0, "self"): torch.ones(4)})
    wrapper.load_mask_snapshot(snap)
    ok = ok and float(wrapper.head_mask(0)[0].item()) == 0.0

    # 10) parameter accounting helper
    lm_params = count_lm_parameters(wrapper)
    print(f"[test] wrapped LM params = {lm_params}")
    ok = ok and lm_params > 0

    # 11) reference RoBERTa-base block costs (Appendix C)
    from .block_selection import ModelShape

    rs = ModelShape(d_model=768, n_layers=12, n_heads=12, n_ffn=3072, ffn_linear_count=2)
    ok = ok and abs(rs.head_block_cost() - 196608) < 1e-6
    ok = ok and abs(rs.neuron_block_cost() - 1536) < 1e-6
    ok = ok and abs(rs.dim_block_cost() - 110592) < 1e-6

    # ---------------- T5: cross attention + gated FFN ----------------
    t5 = _fake_t5()
    xt = torch.randn(2, 5, 32)
    with torch.no_grad():
        ref_t5 = t5(xt)
    w5 = APTModelWrapper(t5, rank=4)
    ok = ok and w5.n_cross_attn_layers() == 2
    ok = ok and w5.ffn_linear_count() == 3
    ok = ok and w5.n_layers == 4
    with torch.no_grad():
        out_t5 = w5.model(xt)
    d5 = (out_t5 - ref_t5).abs().max().item()
    print(f"[test] T5 wrap max|diff| = {d5:.3e}, cross-enc layers = {w5.n_cross_attn_layers()}, "
          f"ffn_linear_count = {w5.ffn_linear_count()}")
    ok = ok and d5 < 1e-6
    ok = ok and w5.num_adapters() == (2 + 2) * 2  # encoder&decoder: attn q,v + ffn in/out
    ok = ok and len(w5.head_mask_names()) == 6     # 2 blocks x (self, cross)

    # ---------------- optional real HF models ----------------
    try:
        from transformers import RobertaConfig, RobertaModel  # type: ignore

        cfg = RobertaConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=32,
        )
        hf = RobertaModel(cfg)
        hf.eval()
        ids = torch.randint(0, 100, (2, 6))
        with torch.no_grad():
            ref_hf = hf(input_ids=ids).last_hidden_state
        hw = APTModelWrapper(hf, rank=4)
        with torch.no_grad():
            out_hf = hw.model(input_ids=ids).last_hidden_state
        d = (out_hf - ref_hf).abs().max().item()
        print(f"[test] HF RoBERTa transparent wrap max|diff| = {d:.3e}, adapters={hw.num_adapters()}")
        ok = ok and d < 1e-4
        hshape = apt_shape(hw)
        ok = ok and hshape.d_model == 64 and hshape.n_layers == 2
    except ImportError:
        print("[test] transformers not installed -- skipped HF RoBERTa test")

    try:
        from transformers import T5Config, T5ForConditionalGeneration  # type: ignore

        t5cfg = T5Config(
            vocab_size=100,
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            decoder_start_token_id=0,
        )
        hf5 = T5ForConditionalGeneration(t5cfg)
        hf5.eval()
        ids = torch.randint(0, 100, (2, 6))
        with torch.no_grad():
            ref5 = hf5(input_ids=ids, decoder_input_ids=ids, output_hidden_states=True).last_hidden_state
        hw5 = APTModelWrapper(hf5, rank=4)
        with torch.no_grad():
            out5 = hw5.model(input_ids=ids, decoder_input_ids=ids, output_hidden_states=True).last_hidden_state
        d5b = (out5 - ref5).abs().max().item()
        print(f"[test] HF T5 transparent wrap max|diff| = {d5b:.3e}, adapters={hw5.num_adapters()}, "
              f"ffn_linear_count={hw5.ffn_linear_count()}")
        ok = ok and d5b < 1e-4
    except ImportError:
        print("[test] transformers not installed -- skipped HF T5 test")

    # ---------------- restore_base_linears (merge path) ----------------
    model2 = _fake_bert()
    w2 = APTModelWrapper(model2, rank=4)
    with torch.no_grad():
        pre = w2.model(x)
    restore_base_linears(w2.model, merge=True)
    ok = ok and not is_wrapped(w2.model)
    with torch.no_grad():
        post = w2.model(x)
    d_back = (post - pre).abs().max().item()
    print(f"[test] after restore_base_linears max|diff| = {d_back:.3e}")
    ok = ok and d_back < 1e-6

    print("[test] model_wrapper self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":  # pragma: no cover
    _self_test()
