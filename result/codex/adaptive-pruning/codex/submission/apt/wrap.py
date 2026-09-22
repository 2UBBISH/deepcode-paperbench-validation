"""Attach APT adapters, pruning masks and salience hooks to RoBERTa / T5.

The wrapper replaces every prunable :class:`torch.nn.Linear` of the transformer
blocks by an :class:`~apt.adapter.APTLinear` and records, for every linear, which
feature indices belong to which structural block.  Three block families are
built (see :mod:`apt.blocks`):

``head``    one MHA head: out-rows of Q/K/V plus the in-cols of O
``neuron``  one FFN neuron: the out-row of the expanding matrices plus the
            in-col of the contracting matrix
``dim``     one transformer hidden dimension: the in-cols of Q/K/V/O (and the
            expanding FFN matrices) plus the out-rows of O (and the contracting
            FFN matrix)

Following Section 4.1 of the paper the APT adapter (the *trainable* part) is
placed on the query / value projections and, for the smaller RoBERTa and T5
models, also on the feed-forward projections; the key / output projections are
pruned but carry no tuning parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .adapter import APTLinear
from .blocks import Block, DIM, HEAD, NEURON, PruningState


@dataclass
class WrapConfig:
    initial_rank: int = 8
    scaling: float = 2.0
    dropout: float = 0.0
    track: bool = True
    adapt_ffn: bool = True
    """Whether APT adapters (trainable low-rank branches) are also added to the
    FFN projections.  The paper enables this for RoBERTa and T5 (Section 4.1) --
    "We also add APT adapter in feed-forward network (FFN) layers when
    fine-tuning smaller models like RoBERTa and T5 for fast training
    convergence." -- and disables it for LLaMA."""


class Topology:
    """Registry of wrapped linears plus the prunable-block table."""

    def __init__(self, model: nn.Module, kind: str) -> None:
        self.model = model
        self.kind = kind
        self.linears: Dict[str, APTLinear] = {}
        self.blocks: List[Block] = []
        #: ``bid -> (kind, owner, index)`` where ``owner`` is the attention /
        #: FFN module key and ``index`` is the head or neuron index.
        self.block_meta: Dict[str, tuple] = {}
        self.d_h: int = 1
        self.d_m: int = 1

    # ------------------------------------------------------------- registry
    def wrap(
        self,
        parent: nn.Module,
        attr: str,
        key: str,
        cfg: WrapConfig,
        use_lora: bool = True,
    ) -> APTLinear:
        base = getattr(parent, attr)
        assert isinstance(base, nn.Linear), f"{key} is not a Linear layer"
        wrapped = APTLinear(
            base,
            r=cfg.initial_rank,
            scaling=cfg.scaling,
            use_lora=use_lora,
            track=cfg.track,
            dropout=cfg.dropout,
            name=key,
        )
        setattr(parent, attr, wrapped)
        self.linears[key] = wrapped
        wrapped.install_hooks()
        return wrapped

    def add_block(self, block: Block, owner: Optional[str] = None, index: Optional[int] = None) -> None:
        self.blocks.append(block)
        self.block_meta[block.bid] = (block.kind, owner, index)

    def indexes(self, kind: int) -> Dict[str, List[int]]:
        """``owner -> sorted list of block indices`` for a given block kind."""
        out: Dict[str, List[int]] = {}
        for block in self.blocks:
            k, owner, index = self.block_meta[block.bid]
            if k != kind:
                continue
            out.setdefault(owner if owner is not None else "", []).append(int(index))
        for k in out:
            out[k] = sorted(out[k])
        return out

    # ------------------------------------------------------------- helpers
    def build_pruning_state(self, **kwargs) -> PruningState:
        state = PruningState(self.blocks, d_h=self.d_h, **kwargs)
        return state

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for lin in self.linears.values():
            if lin.use_lora:
                params += [lin.lora_A, lin.lora_B]
        return params

    def reset_statistics(self) -> None:
        for lin in self.linears.values():
            lin.reset_statistics()

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        for lin in self.linears.values():
            if lin.use_lora:
                lin.lora_A.requires_grad_(True)
                lin.lora_B.requires_grad_(True)

    def n_tuning_parameters(self) -> int:
        return sum(lin.n_tuning_parameters() for lin in self.linears.values())

    def total_block_parameters(self) -> int:
        return int(sum(b.param_count for b in self.blocks))


# --------------------------------------------------------------------------- #
# Salience aggregation
# --------------------------------------------------------------------------- #
def compute_block_salience(topo: Topology, use_kurtosis: bool = True) -> torch.Tensor:
    """Outlier-aware salience score of every block (Eq. 5 / Appendix B).

    ``use_kurtosis=False`` reproduces the ``w/o kurtosis`` ablation of Table 5.

    The per-feature salience vectors of all wrapped linears are concatenated and
    scattered onto their blocks with ``index_add_``; this keeps the scoring
    function lightweight (it is called at every training step) instead of
    walking the ~38k blocks of a RoBERTa-base model in Python.
    """
    plan = _salience_plan(topo)
    names = plan["names"]

    in_vecs, out_vecs, kin_vecs, kout_vecs = [], [], [], []
    for name in names:
        lin = topo.linears[name]
        sal_in, sal_out = lin.compressed_salience()
        tune_in, tune_out = lin.tuning_salience_in(), lin.tuning_salience()
        zeros_in = torch.zeros(lin.in_features, dtype=torch.float64)
        zeros_out = torch.zeros(lin.out_features, dtype=torch.float64)
        v_in = zeros_in if sal_in is None else sal_in
        v_out = zeros_out if sal_out is None else sal_out
        if tune_in is not None:                # Appendix B: + tuning salience
            v_in = v_in + tune_in
        if tune_out is not None:
            v_out = v_out + tune_out
        in_vecs.append(v_in)
        out_vecs.append(v_out)
        if use_kurtosis:
            kin, kout = lin.activation_kurtosis()
            kin_vecs.append(zeros_in if kin is None else torch.clamp(kin, min=0.0))
            kout_vecs.append(zeros_out if kout is None else torch.clamp(kout, min=0.0))

    flat_in = torch.cat(in_vecs)
    flat_out = torch.cat(out_vecs)
    scores = torch.zeros(len(topo.blocks), dtype=torch.float64)
    scores.index_add_(0, plan["idx_in"], flat_in)
    scores.index_add_(0, plan["idx_out"], flat_out)

    if use_kurtosis:
        k_flat_in = torch.cat(kin_vecs)
        k_flat_out = torch.cat(kout_vecs)
        k_sum = torch.zeros(len(topo.blocks), dtype=torch.float64)
        k_cnt = torch.zeros(len(topo.blocks), dtype=torch.float64)
        k_sum.index_add_(0, plan["idx_in"], k_flat_in)
        k_sum.index_add_(0, plan["idx_out"], k_flat_out)
        k_cnt.index_add_(0, plan["idx_in"], torch.ones_like(k_flat_in))
        k_cnt.index_add_(0, plan["idx_out"], torch.ones_like(k_flat_out))
        mean_kurt = k_sum / k_cnt.clamp_min(1.0)
        scores = scores + torch.sqrt(mean_kurt.clamp_min(0.0))
    return scores


def salience_plan(topo: Topology) -> Dict[str, object]:
    """Static ``feature -> block`` scatter plan, built once per topology.

    ``idx_in`` / ``idx_out`` are the concatenated (over ``names``) block indices
    of every input / output feature, which lets the salience of all blocks be
    computed with a single ``index_add_`` and lets a mask vector be broadcast
    back onto the linear layers with a single gather (used by the Mask Tuning
    baseline so that its mask variables stay differentiable).
    """
    cached = getattr(topo, "_sal_plan", None)
    if cached is not None:
        return cached
    names = list(topo.linears.keys())
    idx_in_parts = {n: torch.zeros(topo.linears[n].in_features, dtype=torch.long) for n in names}
    idx_out_parts = {n: torch.zeros(topo.linears[n].out_features, dtype=torch.long) for n in names}
    for bi, block in enumerate(topo.blocks):
        for sl in block.slices:
            part = idx_in_parts if sl.dim == "in" else idx_out_parts
            part[sl.linear][sl.start : sl.stop] = bi
    plan = {
        "names": names,
        "idx_in": torch.cat([idx_in_parts[n] for n in names]) if names else torch.zeros(0, dtype=torch.long),
        "idx_out": torch.cat([idx_out_parts[n] for n in names]) if names else torch.zeros(0, dtype=torch.long),
    }
    topo._sal_plan = plan  # type: ignore[attr-defined]
    return plan


_salience_plan = salience_plan


# --------------------------------------------------------------------------- #
# RoBERTa
# --------------------------------------------------------------------------- #
def wrap_roberta(model: nn.Module, cfg: Optional[WrapConfig] = None) -> Topology:
    cfg = cfg or WrapConfig()
    topo = Topology(model, "roberta")

    backbone = getattr(model, "roberta", None)
    if backbone is None:
        backbone = getattr(model, "bert", None)
    if backbone is None:
        raise ValueError("expected a RoBERTa/BERT style model with a `.roberta`/`.bert` stack")

    conf = backbone.config
    d_m = conf.hidden_size
    n_h = conf.num_attention_heads
    d_h = d_m // n_h
    d_ff = conf.intermediate_size
    n_L = conf.num_hidden_layers
    topo.d_h, topo.d_m = d_h, d_m

    self_attns = []
    for l, layer in enumerate(backbone.encoder.layer):
        self_attn = layer.attention.self
        out_dense_parent = layer.attention.output
        inter = layer.intermediate
        out = layer.output

        topo.wrap(self_attn, "query", f"l{l}.q", cfg, use_lora=True)
        topo.wrap(self_attn, "key", f"l{l}.k", cfg, use_lora=False)
        topo.wrap(self_attn, "value", f"l{l}.v", cfg, use_lora=True)
        topo.wrap(out_dense_parent, "dense", f"l{l}.o", cfg, use_lora=False)
        topo.wrap(inter, "dense", f"l{l}.fc1", cfg, use_lora=cfg.adapt_ffn)
        topo.wrap(out, "dense", f"l{l}.fc2", cfg, use_lora=cfg.adapt_ffn)
        self_attns.append(l)

    # ----------------------------- blocks -------------------------------- #
    for l in range(n_L):
        for h in range(n_h):
            blk = Block(f"head.l{l}.h{h}", HEAD, "model")
            blk.add(f"l{l}.q", "out", h * d_h, d_h)
            blk.add(f"l{l}.k", "out", h * d_h, d_h)
            blk.add(f"l{l}.v", "out", h * d_h, d_h)
            blk.add(f"l{l}.o", "in", h * d_h, d_h)
            blk.param_count = 4 * d_h * d_m
            topo.add_block(blk, owner=f"l{l}", index=h)
        for n in range(d_ff):
            blk = Block(f"neuron.l{l}.n{n}", NEURON, "model")
            blk.add(f"l{l}.fc1", "out", n, 1)
            blk.add(f"l{l}.fc2", "in", n, 1)
            blk.param_count = 2 * d_m
            topo.add_block(blk, owner=f"l{l}", index=n)
    for j in range(d_m):
        blk = Block(f"dim.{j}", DIM, "model")
        for l in range(n_L):
            blk.add(f"l{l}.q", "in", j, 1)
            blk.add(f"l{l}.k", "in", j, 1)
            blk.add(f"l{l}.v", "in", j, 1)
            blk.add(f"l{l}.o", "out", j, 1)
            blk.add(f"l{l}.fc1", "in", j, 1)
            blk.add(f"l{l}.fc2", "out", j, 1)
        blk.param_count = n_L * (4 * d_m + 2 * d_ff)
        topo.add_block(blk, owner="hidden", index=j)

    topo.extra = {  # type: ignore[attr-defined]
        "n_layers": n_L,
        "n_heads": n_h,
        "n_ffn": d_ff,
        "d_h": d_h,
        "d_m": d_m,
        "ffn_multiplier": 2,
    }
    return topo


# --------------------------------------------------------------------------- #
# T5
# --------------------------------------------------------------------------- #
def _t5_dense_layers(ffn: nn.Module):
    """Return ``(expanding, contracting)`` linear attribute names of a T5 FFN."""
    dense = ffn.DenseReluDense
    if hasattr(dense, "wi_0"):
        return ["wi_0", "wi_1"], "wo"
    return ["wi"], "wo"


def wrap_t5(model: nn.Module, cfg: Optional[WrapConfig] = None) -> Topology:
    cfg = cfg or WrapConfig()
    topo = Topology(model, "t5")

    conf = model.config
    d_m = conf.d_model
    n_h = conf.num_heads
    d_h = conf.d_kv
    inner = n_h * d_h
    d_ff = conf.d_ff
    n_enc = conf.num_layers
    n_dec = conf.num_decoder_layers
    topo.d_h, topo.d_m = d_h, d_m
    ffn_mult = 3 if hasattr(model.encoder.block[0].layer[1].DenseReluDense, "wi_0") else 2

    attn_keys: List[str] = []          # keys of every attention sub-module
    enc_attn: List[str] = []
    dec_attn: List[str] = []
    dec_cross: List[str] = []
    enc_ffn: List[str] = []
    dec_ffn: List[str] = []
    ffn_names: Dict[str, tuple] = {}   # key -> (expanding attrs, contracting attr)

    # ------------------------------- encoder ----------------------------- #
    for l, blk in enumerate(model.encoder.block):
        attn = blk.layer[0].SelfAttention
        key = f"e{l}"
        topo.wrap(attn, "q", f"{key}.q", cfg, use_lora=True)
        topo.wrap(attn, "k", f"{key}.k", cfg, use_lora=False)
        topo.wrap(attn, "v", f"{key}.v", cfg, use_lora=True)
        topo.wrap(attn, "o", f"{key}.o", cfg, use_lora=False)
        enc_attn.append(key)
        attn_keys.append(key)

        ffn = blk.layer[1]
        expand, contract = _t5_dense_layers(ffn)
        ffn_names[key] = (expand, contract)
        for name in expand:
            topo.wrap(ffn.DenseReluDense, name, f"{key}.{name}", cfg, use_lora=cfg.adapt_ffn)
        topo.wrap(ffn.DenseReluDense, contract, f"{key}.{contract}", cfg, use_lora=cfg.adapt_ffn)
        enc_ffn.append(key)

    # ------------------------------- decoder ----------------------------- #
    for l, blk in enumerate(model.decoder.block):
        self_attn = blk.layer[0].SelfAttention
        key = f"d{l}"
        topo.wrap(self_attn, "q", f"{key}.q", cfg, use_lora=True)
        topo.wrap(self_attn, "k", f"{key}.k", cfg, use_lora=False)
        topo.wrap(self_attn, "v", f"{key}.v", cfg, use_lora=True)
        topo.wrap(self_attn, "o", f"{key}.o", cfg, use_lora=False)
        dec_attn.append(key)
        attn_keys.append(key)

        cross = blk.layer[1].EncDecAttention
        ckey = f"{key}x"
        topo.wrap(cross, "q", f"{ckey}.q", cfg, use_lora=True)
        topo.wrap(cross, "k", f"{ckey}.k", cfg, use_lora=False)
        topo.wrap(cross, "v", f"{ckey}.v", cfg, use_lora=True)
        topo.wrap(cross, "o", f"{ckey}.o", cfg, use_lora=False)
        dec_cross.append(ckey)
        attn_keys.append(ckey)

        ffn = blk.layer[2]
        expand, contract = _t5_dense_layers(ffn)
        ffn_names[key] = (expand, contract)
        for name in expand:
            topo.wrap(ffn.DenseReluDense, name, f"{key}.{name}", cfg, use_lora=cfg.adapt_ffn)
        topo.wrap(ffn.DenseReluDense, contract, f"{key}.{contract}", cfg, use_lora=cfg.adapt_ffn)
        dec_ffn.append(key)

    # ------------------------------- blocks ------------------------------ #
    for key in attn_keys:
        for h in range(n_h):
            blk = Block(f"head.{key}.h{h}", HEAD, "model")
            blk.add(f"{key}.q", "out", h * d_h, d_h)
            blk.add(f"{key}.k", "out", h * d_h, d_h)
            blk.add(f"{key}.v", "out", h * d_h, d_h)
            blk.add(f"{key}.o", "in", h * d_h, d_h)
            blk.param_count = 4 * d_h * d_m
            topo.add_block(blk, owner=key, index=h)

    for key in enc_ffn + dec_ffn:
        expand, contract = ffn_names[key]
        for n in range(d_ff):
            blk = Block(f"neuron.{key}.n{n}", NEURON, "model")
            for name in expand:
                blk.add(f"{key}.{name}", "out", n, 1)
            blk.add(f"{key}.{contract}", "in", n, 1)
            blk.param_count = len(expand) * d_m
            topo.add_block(blk, owner=key, index=n)

    n_expand = len(ffn_names[enc_ffn[0]][0])
    for j in range(d_m):
        blk = Block(f"dim.{j}", DIM, "model")
        for key in enc_attn + dec_attn:
            blk.add(f"{key}.q", "in", j, 1)
            blk.add(f"{key}.k", "in", j, 1)
            blk.add(f"{key}.v", "in", j, 1)
            blk.add(f"{key}.o", "out", j, 1)
        for key in dec_cross:
            blk.add(f"{key}.q", "in", j, 1)   # decoder hidden dim
            blk.add(f"{key}.k", "in", j, 1)   # encoder hidden dim
            blk.add(f"{key}.v", "in", j, 1)
            blk.add(f"{key}.o", "out", j, 1)
        for key in enc_ffn + dec_ffn:
            expand, contract = ffn_names[key]
            for name in expand:
                blk.add(f"{key}.{name}", "in", j, 1)
            blk.add(f"{key}.{contract}", "out", j, 1)
        n_attn = len(enc_attn + dec_attn) + len(dec_cross)
        blk.param_count = n_attn * 4 * d_m + (len(enc_ffn) + len(dec_ffn)) * (n_expand + 1) * d_ff
        topo.add_block(blk, owner="hidden", index=j)

    topo.extra = {  # type: ignore[attr-defined]
        "n_layers": n_enc,
        "n_decoder_layers": n_dec,
        "n_heads": n_h,
        "n_ffn": d_ff,
        "d_h": d_h,
        "d_m": d_m,
        "ffn_multiplier": ffn_mult,
        "attn_modules": attn_keys,
        "enc_attn": enc_attn,
        "dec_attn": dec_attn,
        "dec_cross": dec_cross,
        "enc_ffn": enc_ffn,
        "dec_ffn": dec_ffn,
    }
    return topo


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def wrap_model(model: nn.Module, cfg: Optional[WrapConfig] = None) -> Topology:
    """Wrap a RoBERTa / BERT / T5 model, building the prunable-block table."""
    name = type(model).__name__.lower()
    if "t5" in name:
        return wrap_t5(model, cfg)
    if "roberta" in name or "bert" in name:
        return wrap_roberta(model, cfg)
    raise ValueError(f"unsupported model type '{type(model).__name__}'")


__all__ = [
    "Topology",
    "WrapConfig",
    "wrap_model",
    "wrap_roberta",
    "wrap_t5",
    "compute_block_salience",
]
