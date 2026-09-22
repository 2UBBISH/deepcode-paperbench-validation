"""Materialise the pruned architecture so that inference really is faster.

During APT training the LM keeps its original tensor shapes and is "pruned" by
multiplying activations with the (soft, annealed) masks.  To obtain the
inference speedups reported in Tables 2/3 the masks must be *baked in*: the
attention heads, FFN neurons and hidden dimensions that were dropped are
physically removed, producing a genuinely smaller model.

``materialize_*`` therefore

1. reads the retained blocks from a :class:`~apt.blocks.PruningState`,
2. merges the trained APT adapter into the frozen weight
   (``W_eff = W + s W_B W_A``, "tuning parameters can be fully merged after
   training"),
3. slices the merged weights/frozen weights down to the retained heads,
   neurons and dimensions and rebuilds the model with a smaller config.

Implementation notes
--------------------
* RoBERTa / BERT: the head count is stored per layer
  (``attention_head_size`` / ``all_head_size`` / ``num_attention_heads`` are
  updated accordingly), so heads can be pruned non-uniformly.
* T5: the relative position bias is computed by the first block and *shared*
  with the rest of the stack, which forces a uniform number of heads inside a
  stack.  We therefore keep a head in a stack when it is retained in at least
  half of that stack's attention modules.  Encoder self-attention, decoder
  self-attention and decoder cross-attention form three independent stacks and
  may keep different head counts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .blocks import DIM, HEAD, NEURON


def _fresh_model(model: nn.Module, conf) -> nn.Module:
    """Instantiate an empty model of the same class with a modified config."""
    if hasattr(type(model), "_from_config"):
        return type(model)._from_config(conf)
    return type(model).from_config(conf)  # pragma: no cover - legacy API


def _idx_select(t: torch.Tensor, idx: Sequence[int], dim: int) -> torch.Tensor:
    if t is None:
        return None
    index = torch.tensor(list(idx), dtype=torch.long, device=t.device)
    return t.index_select(dim, index).contiguous()


def _merged_weight(lin) -> torch.Tensor:
    """``W + s W_B W_A`` of an APT-wrapped linear (masks applied separately)."""
    w = lin.base.weight.detach().clone()
    if lin.use_lora:
        w = w + lin.scaling * (lin.lora_B.detach() @ lin.lora_A.detach())
    return w


@dataclass
class PrunedPlan:
    kept_dims: List[int] = field(default_factory=list)
    kept_heads: Dict[str, List[int]] = field(default_factory=dict)
    kept_neurons: Dict[str, List[int]] = field(default_factory=dict)

    def prune_slice(self, lin: nn.Linear, rows: Sequence[int], cols: Sequence[int]) -> nn.Linear:
        new = nn.Linear(len(cols), len(rows), bias=lin.bias is not None)
        with torch.no_grad():
            w = _idx_select(lin.weight, rows, 0)
            w = _idx_select(w, cols, 1)
            new.weight.copy_(w)
            if lin.bias is not None:
                new.bias.copy_(_idx_select(lin.bias, rows, 0))
        return new


def plan_from_state(topo, state, uniform_heads: bool = False) -> PrunedPlan:
    """Derive the retained head / neuron / dimension index lists."""
    plan = PrunedPlan()
    retain = state.retain

    plan.kept_dims = [
        topo.block_meta[b.bid][2]
        for bi, b in enumerate(topo.blocks)
        if b.kind == DIM and bool(retain[bi])
    ]

    for kind, store in ((HEAD, plan.kept_heads), (NEURON, plan.kept_neurons)):
        for bi, b in enumerate(topo.blocks):
            if b.kind != kind or not retain[bi]:
                continue
            _, owner, index = topo.block_meta[b.bid]
            store.setdefault(owner, []).append(index)
        for k in store:
            store[k] = sorted(store[k])
    if uniform_heads:
        _make_uniform_heads(plan)
    return plan


def _make_uniform_heads(plan: PrunedPlan, thresholds: float = 0.5) -> None:
    """Force a common head set per stack (see the module docstring)."""
    stacks = {
        "enc": [k for k in plan.kept_heads if k.startswith("e")],
        "dec": [k for k in plan.kept_heads if k.startswith("d") and "x" not in k],
        "cross": [k for k in plan.kept_heads if k.startswith("dx") or "x" in k],
    }
    n_heads = max((len(v) for v in plan.kept_heads.values()), default=0)
    for name, keys in stacks.items():
        if not keys:
            continue
        counts: Dict[int, int] = {}
        for k in keys:
            for h in plan.kept_heads.get(k, []):
                counts[h] = counts.get(h, 0) + 1
        keep = sorted(h for h, c in counts.items() if c >= thresholds * max(1, len(keys)))
        if not keep:
            keep = sorted(plan.kept_heads[keys[0]])[:1]
        for k in keys:
            plan.kept_heads[k] = list(keep)
    return None


# --------------------------------------------------------------------------- #
# RoBERTa / BERT
# --------------------------------------------------------------------------- #
def materialize_roberta(topo, state) -> nn.Module:
    model = topo.model
    plan = plan_from_state(topo, state)
    d_h_old = topo.d_h
    kept = plan.kept_dims
    if not kept:
        raise ValueError("the plan retains no hidden dimension")

    new_conf = copy.deepcopy(model.config)
    new_conf.hidden_size = len(kept)
    # The per-layer head count is restored below; using 1 here keeps the
    # ``hidden_size % num_attention_heads`` assertion of the model definitions
    # happy while the real shapes are written in explicitly.
    new_conf.num_attention_heads = 1
    new_conf.max_position_embeddings = getattr(new_conf, "max_position_embeddings", 512)
    new_conf.intermediate_size = max(
        1, max((len(v) for v in plan.kept_neurons.values()), default=1)
    )

    new_model = _fresh_model(model, new_conf)

    backbone = getattr(model, "roberta", None) or getattr(model, "bert", None)
    new_backbone = getattr(new_model, "roberta", None) or getattr(new_model, "bert", None)
    old_d_m = backbone.config.hidden_size

    # ---- embeddings ---------------------------------------------------- #
    with torch.no_grad():
        for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
            old_e = getattr(backbone.embeddings, name, None)
            new_e = getattr(new_backbone.embeddings, name, None)
            if old_e is None or new_e is None:
                continue
            new_e.weight = nn.Parameter(_idx_select(old_e.weight, kept, 1))
        ln_old = backbone.embeddings.LayerNorm
        ln_new = new_backbone.embeddings.LayerNorm
        ln_new.weight = nn.Parameter(_idx_select(ln_old.weight, kept, 0))
        ln_new.bias = nn.Parameter(_idx_select(ln_old.bias, kept, 0))

    # ---- transformer layers -------------------------------------------- #
    for l, (old_layer, new_layer) in enumerate(zip(backbone.encoder.layer, new_backbone.encoder.layer)):
        heads_l = plan.kept_heads.get(f"l{l}") or [0]
        row_q = _head_rows(heads_l, d_h_old)
        n_head_l = len(heads_l)
        neu = plan.kept_neurons.get(f"l{l}") or [0]

        for attr, rows, cols in (
            ("q", row_q, kept),
            ("k", row_q, kept),
            ("v", row_q, kept),
        ):
            old = getattr(topo.linears[f"l{l}.{attr}"], "base")
            w = _merged_weight(topo.linears[f"l{l}.{attr}"])
            setattr(
                new_layer.attention.self,
                {"q": "query", "k": "key", "v": "value"}[attr],
                _copy_linear(w, old.bias, rows, cols),
            )
        o_lin = topo.linears[f"l{l}.o"]
        new_layer.attention.output.dense = _copy_linear(
            _merged_weight(o_lin), o_lin.base.bias, kept, row_q
        )
        fc1 = topo.linears[f"l{l}.fc1"]
        new_layer.intermediate.dense = _copy_linear(
            _merged_weight(fc1), fc1.base.bias, neu, kept
        )
        fc2 = topo.linears[f"l{l}.fc2"]
        new_layer.output.dense = _copy_linear(
            _merged_weight(fc2), fc2.base.bias, kept, neu
        )

        attn = new_layer.attention.self
        attn.num_attention_heads = n_head_l
        attn.attention_head_size = d_h_old
        attn.all_head_size = n_head_l * d_h_old
        for ln_old, ln_new in (
            (old_layer.attention.output.LayerNorm, new_layer.attention.output.LayerNorm),
            (old_layer.output.LayerNorm, new_layer.output.LayerNorm),
        ):
            ln_new.weight = nn.Parameter(_idx_select(ln_old.weight, kept, 0))
            ln_new.bias = nn.Parameter(_idx_select(ln_old.bias, kept, 0))

    _remap_dense_linears(model, new_model, old_d_m, kept)
    return new_model


def _head_rows(heads: Sequence[int], d_h: int) -> List[int]:
    rows: List[int] = []
    for h in heads:
        rows.extend(range(h * d_h, (h + 1) * d_h))
    return rows


def _copy_linear(w: torch.Tensor, b: Optional[torch.Tensor], rows: Sequence[int], cols: Sequence[int]) -> nn.Linear:
    new = nn.Linear(len(cols), len(rows), bias=b is not None)
    with torch.no_grad():
        new.weight.copy_(_idx_select(_idx_select(w, rows, 0), cols, 1))
        if b is not None:
            new.bias.copy_(_idx_select(b, rows, 0))
    return new


def _remap_dense_linears(old_model, new_model, old_d_m: int, kept: Sequence[int]) -> None:
    """Slice the task-head / pooler / embedding-adjacent linears."""
    old_named = dict(old_model.named_modules())
    new_named = dict(new_model.named_modules())
    skip = ("encoder.layer", "decoder.block", "encoder.block")
    with torch.no_grad():
        for name, mod in new_named.items():
            if not isinstance(mod, nn.Linear):
                continue
            if any(s in name for s in skip):
                continue
            old = old_named.get(name)
            if not isinstance(old, nn.Linear):
                continue
            rows = list(range(mod.out_features))
            cols = list(range(mod.in_features))
            if old.in_features == old_d_m:
                cols = list(kept)
            if old.out_features == old_d_m:
                rows = list(kept)
            if len(rows) == old.out_features and len(cols) == old.in_features:
                mod.weight.copy_(old.weight)
                if mod.bias is not None and old.bias is not None:
                    mod.bias.copy_(old.bias)
                continue
            new_mod = _copy_linear(old.weight, old.bias, rows, cols)
            parent = new_model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else new_model
            setattr(parent, name.rsplit(".", 1)[-1], new_mod)
        # non-linear 1-D parameters that live in the residual stream
        for name, mod in new_named.items():
            if isinstance(mod, nn.LayerNorm):
                old = old_named.get(name)
                if isinstance(old, nn.LayerNorm) and old.weight.numel() == old_d_m:
                    mod.weight = nn.Parameter(_idx_select(old.weight, kept, 0))
                    mod.bias = nn.Parameter(_idx_select(old.bias, kept, 0))
    # keep the (optional) masked-LM head tied to the input embeddings
    old_head = getattr(old_model, "lm_head", None)
    new_head = getattr(new_model, "lm_head", None)
    if old_head is not None and new_head is not None and hasattr(new_head, "decoder"):
        old_emb = old_model.get_input_embeddings().weight
        if old_head.decoder.weight is old_emb:
            new_head.decoder.weight = new_model.get_input_embeddings().weight


# --------------------------------------------------------------------------- #
# T5
# --------------------------------------------------------------------------- #
def materialize_t5(topo, state) -> nn.Module:
    model = topo.model
    plan = plan_from_state(topo, state, uniform_heads=True)
    kept = plan.kept_dims
    d_h = topo.d_h
    d_m_old = model.config.d_model

    new_conf = copy.deepcopy(model.config)
    new_conf.d_model = len(kept)
    new_conf.num_heads = 1          # restored per attention module below
    new_conf.d_ff = max(1, max((len(v) for v in plan.kept_neurons.values()), default=1))
    new_model = _fresh_model(model, new_conf)
    _remap_embeddings(model, new_model, kept)

    # ---- encoder ------------------------------------------------------- #
    for l, (old_blk, new_blk) in enumerate(zip(model.encoder.block, new_model.encoder.block)):
        _rewrite_t5_attn(topo, f"e{l}", old_blk.layer[0].SelfAttention,
                         new_blk.layer[0].SelfAttention, plan, kept, d_h)
        _rewrite_t5_ffn(topo, f"e{l}", old_blk.layer[1], new_blk.layer[1], plan, kept, d_m_old)
        _rewrite_layernorms(old_blk, new_blk, kept, d_m_old)

    # ---- decoder ------------------------------------------------------- #
    for l, (old_blk, new_blk) in enumerate(zip(model.decoder.block, new_model.decoder.block)):
        _rewrite_t5_attn(topo, f"d{l}", old_blk.layer[0].SelfAttention,
                         new_blk.layer[0].SelfAttention, plan, kept, d_h)
        _rewrite_t5_attn(topo, f"d{l}x", old_blk.layer[1].EncDecAttention,
                         new_blk.layer[1].EncDecAttention, plan, kept, d_h)
        _rewrite_t5_ffn(topo, f"d{l}", old_blk.layer[2], new_blk.layer[2], plan, kept, d_m_old)
        _rewrite_layernorms(old_blk, new_blk, kept, d_m_old)

    _rewrite_layernorms(model.encoder, new_model.encoder, kept, d_m_old, only_final=True)
    _rewrite_layernorms(model.decoder, new_model.decoder, kept, d_m_old, only_final=True)
    return new_model


def _rewrite_t5_attn(topo, key, old_attn, new_attn, plan, kept, d_h):
    heads = plan.kept_heads.get(key) or [0]
    row_q = _head_rows(heads, d_h)
    for attr in ("q", "k", "v"):
        lin = topo.linears[f"{key}.{attr}"]
        setattr(new_attn, attr, _copy_linear(_merged_weight(lin), lin.base.bias, row_q, kept))
    o_lin = topo.linears[f"{key}.o"]
    new_attn.o = _copy_linear(_merged_weight(o_lin), o_lin.base.bias, kept, row_q)
    new_attn.n_heads = len(heads)
    new_attn.inner_dim = len(heads) * d_h
    if old_attn.has_relative_attention_bias and old_attn.relative_attention_bias is not None:
        rb = old_attn.relative_attention_bias.weight
        new_attn.relative_attention_bias = nn.Embedding(rb.size(0), len(heads))
        with torch.no_grad():
            new_attn.relative_attention_bias.weight.copy_(_idx_select(rb, heads, 1))
        new_attn.has_relative_attention_bias = True
    else:
        new_attn.relative_attention_bias = None
        new_attn.has_relative_attention_bias = False


def _rewrite_t5_ffn(topo, key, old_ffn, new_ffn, plan, kept, d_m_old):
    neurons = plan.kept_neurons.get(key) or [0]
    dense_old = old_ffn.DenseReluDense
    names = []
    for attr in ("wi", "wi_0", "wi_1", "wo"):
        if hasattr(dense_old, attr):
            names.append(attr)
    for attr in names:
        lin = topo.linears.get(f"{key}.{attr}")
        if lin is None:
            continue
        if attr == "wo":
            new = _copy_linear(_merged_weight(lin), lin.base.bias, kept, neurons)
        else:
            new = _copy_linear(_merged_weight(lin), lin.base.bias, neurons, kept)
        setattr(new_ffn.DenseReluDense, attr, new)


def _rewrite_layernorms(old_mod, new_mod, kept, d_m_old, only_final: bool = False):
    old_named = dict(old_mod.named_modules())
    new_named = dict(new_mod.named_modules())
    with torch.no_grad():
        for name, mod in new_named.items():
            if not _is_t5_norm(mod):
                continue
            if only_final and "." in name:
                continue
            old = old_named.get(name)
            if old is None or old.weight.numel() != d_m_old:
                continue
            mod.weight = nn.Parameter(_idx_select(old.weight, kept, 0))


def _is_t5_norm(mod) -> bool:
    return type(mod).__name__ in {"T5LayerNorm", "T5RMSNorm"}


def _remap_embeddings(old_model, new_model, kept) -> None:
    with torch.no_grad():
        new_model.shared.weight = nn.Parameter(_idx_select(old_model.shared.weight, kept, 1))
        if getattr(new_model, "lm_head", None) is not None and old_model.lm_head.weight is old_model.shared.weight:
            new_model.lm_head.weight = new_model.shared.weight
        for stack in ("encoder", "decoder"):
            old_s = getattr(old_model, stack, None)
            new_s = getattr(new_model, stack, None)
            if old_s is None or new_s is None:
                continue
            if getattr(old_s, "embed_tokens", None) is old_model.shared:
                new_s.embed_tokens = new_model.shared


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def materialize(topo, state) -> nn.Module:
    if topo.kind == "t5":
        return materialize_t5(topo, state)
    return materialize_roberta(topo, state)


def n_parameters(model: nn.Module, ignore_shared: bool = False) -> int:
    """Total parameter count, de-duplicating tied weights."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


__all__ = [
    "PrunedPlan",
    "plan_from_state",
    "materialize",
    "materialize_roberta",
    "materialize_t5",
    "n_parameters",
]
