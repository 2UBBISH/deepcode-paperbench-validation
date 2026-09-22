"""Section 4.1 cost analysis: inference FLOPs for CFG vs. vanilla prompting.

Paper: "Stay on Topic with Classifier-Free Guidance".

Relevant text (Section 4, *Cost Analysis: FLOPs and VRAM*):

    "In terms of computational requirements, CFG requires two passes through the
     network, effectively doubling the amount of FLOPs required for inference.
     ... we calculate the FLOP for each of the benchmark experiments that we ran
     in Section 3.1. We then compare across model sizes, with and without CFG.
     We conclude ... that, across 5 out of 9 tasks, there is a statistically
     insignificant difference between using CFG and using vanilla prompting with
     a model of twice the size at p=.01 ... Of the significantly different tasks,
     2 favor CFG and 2 favor vanilla."

    "While CFG boosts the performance of smaller models, it doubles the demands of
     the kv cache."

The Addendum clarifies how the FLOPs were measured:

    "In the experiments in Section 4.1 FLOPs are measured using this formula:
     https://github.com/google-research/electra/blob/master/flops_computation.py"

This module reimplements the ELECTRA ``flops_computation.py`` estimate -- FLOPs are
accumulated as ``2 * (number of multiply-accumulate parameters touched)`` for every
matmul plus the attention-score/context matmuls, which for a forward pass over a
sequence of length ``L`` reduces to the classic ``~2 * params`` per *token* term plus
the quadratic attention term -- and extends it with:

* an explicit :class:`ModelSpec` for each transformer family used in the paper
  (GPT-2, Pythia, CodeGen-mono, Falcon-7b), including MQA/GQA key-value head counts;
* prefill vs. decode accounting (attention cost grows with the sequence length, so
  per-token FLOPs are not constant during generation);
* the CFG multiplier (``x2`` whenever ``gamma != 1``, since two forward passes through
  the same weights are run per decoding step, see Section 2.2);
* the "model of twice the size" comparison of Section 4.1, i.e. matching a CFG model's
  vanilla-model-equivalent parameter/FLOP budget to the next model in the family;
* the per-task ANCOVA p-values of Appendix C.2 / Table 6 (used by
  :mod:`src.analysis.ancova`) so the ``5 insignificant / 2 CFG / 2 vanilla`` split of
  Section 4 can be checked programmatically.

The module is deliberately dependency-free (pure Python + the standard library) so it
can be unit-tested on CPU without torch/transformers, and it also provides
:func:`spec_from_config` to derive a spec from any HuggingFace config object.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    # constants
    "FLOP_PER_MAC",
    "CFG_INFERENCE_MULTIPLIER",
    "VANILLA_GAMMA",
    "CFG_GAMMA",
    "BENCHMARK_TASKS",
    "SIGNIFICANCE_LEVEL",
    "PAPER_TABLE6",
    "PARAMETER_COUNT_TOLERANCE",
    # specs
    "ModelSpec",
    "make_spec",
    "MODEL_SPECS",
    "MODEL_ALIASES",
    "get_spec",
    "register_spec",
    "spec_from_config",
    "normalize_model_name",
    # per-token flops
    "layer_dense_flops_per_token",
    "attention_flops_per_token",
    "lm_head_flops_per_token",
    "embedding_flops_per_token",
    "nonlinear_flops_per_token",
    "flops_per_token",
    "flops_per_forward",
    "prefill_flops",
    "decode_flops",
    "generation_flops",
    # cfg
    "is_cfg",
    "cfg_multiplier",
    "cfg_overhead_factor",
    "vanilla_flops_per_token",
    "cfg_flops_per_token",
    "inference_flops_per_token",
    "expected_cfg_flops",
    "training_flops_multiplier",
    # model-size comparison
    "doubled_flops_per_token",
    "equivalent_vanilla_spec",
    "flops_table",
    "params_table",
    "format_flops_table",
    "summarize_flops",
    "flops_report",
    "save_flops",
    # table 6 anchors
    "paper_ancova_table",
    "significant_tasks",
    "favor_counts",
    "check_against_paper",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: FLOPs per multiply-accumulate (one multiply + one add).
FLOP_PER_MAC = 2

#: Section 2.2/4: CFG runs two forward passes through the *same* weights per decoding
#: step (conditional on ``c`` and unconditional on ``c_bar``), so inference FLOPs double.
CFG_INFERENCE_MULTIPLIER = 2.0

#: ``gamma == 1`` reproduces vanilla conditional prompting (no guidance).
VANILLA_GAMMA = 1.0

#: Guidance strength used for the Section 5 analyses / Section 4 comparisons.
CFG_GAMMA = 1.5

#: The nine Section 3.1 benchmarks (names mirror ``src.eval.harness_cfg.HARNESS_TASKS``).
BENCHMARK_TASKS: Tuple[str, ...] = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "piqa",
    "sciq",
    "triviaqa",
    "winogrande",
    "lambada_openai",
)

#: ANCOVA significance cutoff used in Section 4.1 (Rutherford 2011).
SIGNIFICANCE_LEVEL = 0.01

#: Per-task ANCOVA p-values from Appendix C.2 / Table 6, mapped to the mode they favour.
#: ``None`` means "not significant at p = .01" (5 of 9 tasks, matching Section 4).
PAPER_TABLE6: Dict[str, Dict[str, Any]] = {
    "arc_challenge": {"p_value": 0.216, "winner": None},
    "arc_easy": {"p_value": 0.355, "winner": None},
    "boolq": {"p_value": 0.345, "winner": None},
    "hellaswag": {"p_value": 0.012, "winner": None},
    "piqa": {"p_value": 0.030, "winner": None},
    "sciq": {"p_value": 0.008, "winner": "cfg"},
    "triviaqa": {"p_value": 0.008, "winner": "vanilla"},
    "winogrande": {"p_value": 0.003, "winner": "vanilla"},
    "lambada_openai": {"p_value": 0.000, "winner": "cfg"},
}

#: Tolerance when checking a reimplemented parameter count against the published size.
PARAMETER_COUNT_TOLERANCE = 0.25


# --------------------------------------------------------------------------------------
# Model specifications
# --------------------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Architecture hyper-parameters needed for the ELECTRA FLOPs formula.

    Only the shapes that enter the matmul budget are stored; everything else
    (activations, dropouts, rotary embeddings) is FLOP-negligible.
    """

    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    vocab_size: int
    embedding_size: Optional[int] = None  # None -> hidden_size
    kv_heads: Optional[int] = None  # None -> num_heads (plain MHA)
    tied_embedding: bool = True  # True -> no separate LM head matmul
    max_seq_len: int = 2048
    rotary: bool = False
    alibi: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------------
    @property
    def emb_dim(self) -> int:
        return int(self.embedding_size or self.hidden_size)

    @property
    def head_dim(self) -> int:
        return int(self.hidden_size // self.num_heads)

    @property
    def n_kv_heads(self) -> int:
        return int(self.kv_heads if self.kv_heads is not None else self.num_heads)

    @property
    def kv_dim(self) -> int:
        """Total width of the key (and value) projection."""
        return int(self.head_dim * self.n_kv_heads)

    @property
    def num_parameters(self) -> int:
        """Parameter count (embeddings + per-layer matrices [+ untied LM head])."""
        d = int(self.hidden_size)
        v = int(self.vocab_size)
        i = int(self.intermediate_size)
        # token embedding
        n = v * self.emb_dim
        # learned absolute positions (GPT-2), when not rotary/alibi
        if not (self.rotary or self.alibi):
            n += self.max_seq_len * self.emb_dim
        # per layer: qkv + out + mlp(up,down) + 2 layer norms
        per_layer = (
            d * (d + 2 * self.kv_dim)  # q, k, v projections
            + d * d  # output projection
            + 2 * d * i  # MLP up + down
            + 2 * d  # layer norms (2 * d each -> small, kept for accuracy)
        )
        n += self.num_layers * per_layer
        # final layer norm
        n += d
        # untied LM head (GPT-2 ties it to the token embedding; Pythia/CodeGen do not)
        if not self.tied_embedding:
            n += self.emb_dim * v
        return int(n)

    # Alias used by scripts / reports.
    @property
    def params(self) -> int:
        return self.num_parameters

    @property
    def params_billions(self) -> float:
        return self.num_parameters / 1e9

    def params_breakdown(self) -> Dict[str, int]:
        """Component-wise parameter counts (useful for sanity checks)."""
        d = int(self.hidden_size)
        v = int(self.vocab_size)
        i = int(self.intermediate_size)
        emb = v * self.emb_dim
        pos = 0 if (self.rotary or self.alibi) else self.max_seq_len * self.emb_dim
        qkv = self.num_layers * d * (d + 2 * self.kv_dim)
        out = self.num_layers * d * d
        mlp = self.num_layers * 2 * d * i
        norms = self.num_layers * 2 * d + d
        head = self.emb_dim * v if not self.tied_embedding else 0
        return {
            "token_embedding": int(emb),
            "position_embedding": int(pos),
            "attention_qkv": int(qkv),
            "attention_out": int(out),
            "mlp": int(mlp),
            "layer_norms": int(norms),
            "lm_head": int(head),
        }

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["num_parameters"] = self.num_parameters
        d["params_billions"] = round(self.params_billions, 4)
        return d

    def replace(self, **kwargs: Any) -> "ModelSpec":
        return _dc_replace(self, **kwargs)


def make_spec(
    name: str,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    intermediate_size: int,
    vocab_size: int,
    embedding_size: Optional[int] = None,
    kv_heads: Optional[int] = None,
    tied_embedding: bool = True,
    max_seq_len: int = 2048,
    rotary: bool = False,
    alibi: bool = False,
    **extra: Any,
) -> ModelSpec:
    """Construct a :class:`ModelSpec` (positional order follows ELECTRA's ``make_spec``)."""
    return ModelSpec(
        name=name,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        kv_heads=kv_heads,
        tied_embedding=tied_embedding,
        max_seq_len=max_seq_len,
        rotary=rotary,
        alibi=alibi,
        extra=dict(extra),
    )


#: Architectures for every model family reported in the paper (Section 3.1 / 3.3).
MODEL_SPECS: Dict[str, ModelSpec] = {
    # ---- GPT-2 (tied input/output embeddings, learned absolute positions) --------
    "gpt2": make_spec("gpt2", 768, 12, 12, 3072, 50257, tied_embedding=True),
    "gpt2-medium": make_spec("gpt2-medium", 1024, 24, 16, 4096, 50257, tied_embedding=True),
    "gpt2-large": make_spec("gpt2-large", 1280, 36, 20, 5120, 50257, tied_embedding=True),
    "gpt2-xl": make_spec("gpt2-xl", 1600, 48, 25, 6400, 50257, tied_embedding=True),
    # ---- Pythia (GPT-NeoX: rotary, untied embeddings, packed vocab 50304) --------
    "pythia-70m": make_spec("pythia-70m", 512, 6, 8, 2048, 50304,
                            tied_embedding=False, rotary=True),
    "pythia-160m": make_spec("pythia-160m", 768, 12, 12, 3072, 50304,
                             tied_embedding=False, rotary=True),
    "pythia-410m": make_spec("pythia-410m", 1024, 24, 16, 4096, 50304,
                             tied_embedding=False, rotary=True),
    "pythia-1b": make_spec("pythia-1b", 2048, 16, 8, 8192, 50304,
                           tied_embedding=False, rotary=True),
    "pythia-1.4b": make_spec("pythia-1.4b", 2048, 24, 16, 8192, 50304,
                             tied_embedding=False, rotary=True),
    "pythia-2.8b": make_spec("pythia-2.8b", 2560, 32, 32, 10240, 50304,
                             tied_embedding=False, rotary=True),
    "pythia-6.9b": make_spec("pythia-6.9b", 4096, 32, 32, 16384, 50304,
                             tied_embedding=False, rotary=True),
    "pythia-12b": make_spec("pythia-12b", 5120, 36, 40, 20480, 50304,
                            tied_embedding=False, rotary=True),
    # ---- CodeGen-mono (Section 3.3.1 Program Synthesis) -------------------------
    "codegen-350m-mono": make_spec("codegen-350m-mono", 1024, 20, 16, 4096, 51200,
                                   tied_embedding=False, rotary=True),
    "codegen-2b-mono": make_spec("codegen-2b-mono", 2560, 32, 32, 10240, 51200,
                                 tied_embedding=False, rotary=True),
    "codegen-6b-mono": make_spec("codegen-6b-mono", 4096, 33, 16, 16384, 51200,
                                 tied_embedding=False, rotary=True),
    # ---- Falcon-7b (Section 5 analysis: multi-query attention + ALiBi) ---------
    "falcon-7b": make_spec("falcon-7b", 4544, 32, 71, 18176, 65024,
                           kv_heads=1, tied_embedding=False, alibi=True),
    # ---- Section 3.2 Chain-of-Thought generators --------------------------------
    "wizardlm-30b": make_spec("wizardlm-30b", 6656, 60, 52, 17920, 32000,
                              tied_embedding=False, rotary=True, max_seq_len=4096),
    "guanaco-65b": make_spec("guanaco-65b", 8192, 80, 64, 22016, 32000,
                             tied_embedding=False, rotary=True, max_seq_len=4096),
}

#: Convenience aliases so HuggingFace ids can be passed straight in.
MODEL_ALIASES: Dict[str, str] = {
    "gpt2-small": "gpt2",
    "openai-community/gpt2": "gpt2",
    "openai-community/gpt2-medium": "gpt2-medium",
    "openai-community/gpt2-large": "gpt2-large",
    "openai-community/gpt2-xl": "gpt2-xl",
    "eleutherai/pythia-70m": "pythia-70m",
    "eleutherai/pythia-160m": "pythia-160m",
    "eleutherai/pythia-410m": "pythia-410m",
    "eleutherai/pythia-1b": "pythia-1b",
    "eleutherai/pythia-1.4b": "pythia-1.4b",
    "eleutherai/pythia-2.8b": "pythia-2.8b",
    "eleutherai/pythia-6.9b": "pythia-6.9b",
    "eleutherai/pythia-12b": "pythia-12b",
    "salesforce/codegen-350m-mono": "codegen-350m-mono",
    "salesforce/codegen-2b-mono": "codegen-2b-mono",
    "salesforce/codegen-6b-mono": "codegen-6b-mono",
    "tiiuae/falcon-7b": "falcon-7b",
    "tiiuae/falcon-7b-instruct": "falcon-7b",
    "wizardlm/wizardlm-30b": "wizardlm-30b",
    "timdettmers/guanaco-65b": "guanaco-65b",
}


def normalize_model_name(name: str) -> str:
    """Lower-case a model name/id and strip a trailing ``-base``/``-instruct`` suffix."""
    key = str(name).strip().lower()
    for suffix in ("-instruct", "-base", "-v1.0", "-v1"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def get_spec(model: Any) -> ModelSpec:
    """Look a :class:`ModelSpec` up by name (aliases and HF ids supported).

    ``model`` may also be an HF config/model, an object exposing ``.config``, a
    :class:`ModelSpec` (returned unchanged) or a mapping of architecture kwargs.
    """
    if isinstance(model, ModelSpec):
        return model
    if not isinstance(model, str):
        if hasattr(model, "config"):
            return spec_from_config(model.config)
        if hasattr(model, "hidden_size") or hasattr(model, "n_embd"):
            return spec_from_config(model)
        if isinstance(model, dict):
            return spec_from_config(model)
        raise TypeError(f"cannot derive a ModelSpec from {type(model)!r}")

    key = normalize_model_name(model)
    if key in MODEL_SPECS:
        return MODEL_SPECS[key]
    if key in MODEL_ALIASES:
        return MODEL_SPECS[MODEL_ALIASES[key]]
    # heuristic: "pythia-1.4b" style ids embedded in a longer path
    for name, spec in MODEL_SPECS.items():
        if key.endswith(name) or name in key:
            return spec
    raise KeyError(
        f"unknown model {model!r}; register it with register_spec() or "
        f"spec_from_config(). Known: {sorted(MODEL_SPECS)}"
    )


def register_spec(spec: ModelSpec, aliases: Iterable[str] = ()) -> ModelSpec:
    """Add (or overwrite) a spec in the global registry."""
    MODEL_SPECS[spec.name] = spec
    for alias in aliases:
        MODEL_ALIASES[normalize_model_name(alias)] = spec.name
    return spec


def _first_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] is not None:
                return obj[n]
        elif hasattr(obj, n) and getattr(obj, n) is not None:
            return getattr(obj, n)
    return default


def spec_from_config(config: Any, name: Optional[str] = None) -> ModelSpec:
    """Build a :class:`ModelSpec` from a HuggingFace config (object or ``dict``).

    Handles GPT-2 (``n_embd``/``n_layer``/``n_head``), GPT-NeoX/Pythia,
    CodeGen (``n_inner``), Falcon (``multi_query``) and LLaMA-style configs
    (``num_key_value_heads`` for GQA).
    """
    d = int(_first_attr(config, ("hidden_size", "n_embd", "d_model"), 0))
    layers = int(_first_attr(config, ("num_hidden_layers", "n_layer"), 0))
    heads = int(_first_attr(config, ("num_attention_heads", "n_head"), 0))
    vocab = int(_first_attr(config, ("vocab_size",), 0))
    inter = _first_attr(config, ("intermediate_size", "n_inner", "ffn_dim"), None)
    if inter is None:
        inter = 4 * d
    inter = int(inter)
    kv_heads = _first_attr(config, ("num_key_value_heads", "num_kv_heads"), None)
    if kv_heads is None:
        multi_query = bool(_first_attr(config, ("multi_query",), False))
        if multi_query:
            kv_heads = 1
    kv_heads = None if kv_heads is None else int(kv_heads)

    tied = bool(_first_attr(config, ("tie_word_embeddings",), False))
    # GPT-2 family ties embeddings by default when the flag is absent.
    if _first_attr(config, ("tie_word_embeddings",), None) is None:
        tied = "gpt" in str(type(config)).lower() or bool(
            _first_attr(config, ("model_type",), "") == "gpt2"
        )

    max_seq = int(
        _first_attr(config, ("max_position_embeddings", "n_positions", "n_ctx"), 2048)
    )
    model_type = str(_first_attr(config, ("model_type",), "")).lower()
    rotary = bool(
        _first_attr(config, ("rotary", "rotary_emb_base", "rope_scaling", "rotary_dim"), None)
        is not None
        or model_type in ("gpt_neox", "llama", "codegen", "falcon")
    )
    alibi = bool(_first_attr(config, ("alibi",), False))

    spec = make_spec(
        name=name or str(_first_attr(config, ("_name_or_path", "_name"), "hf-model")),
        hidden_size=d,
        num_layers=layers,
        num_heads=heads,
        intermediate_size=inter,
        vocab_size=vocab,
        kv_heads=kv_heads,
        tied_embedding=tied,
        max_seq_len=max_seq,
        rotary=rotary,
        alibi=alibi,
    )
    return spec


# --------------------------------------------------------------------------------------
# Per-token FLOPs (ELECTRA formula)
# --------------------------------------------------------------------------------------


def layer_dense_flops_per_token(spec: ModelSpec) -> int:
    """Non-attention matmul FLOPs of one transformer layer, per token.

    Includes the query/key/value and output projections and both MLP matrices
    (multiply-accumulate counted as :data:`FLOP_PER_MAC` ops).
    """
    d = int(spec.hidden_size)
    i = int(spec.intermediate_size)
    kv = int(spec.kv_dim)
    qkv = d * (d + 2 * kv)
    out = d * d
    mlp = 2 * d * i
    return FLOP_PER_MAC * (qkv + out + mlp)


def attention_flops_per_token(
    spec: ModelSpec,
    seq_len: int = 1,
    avg_attn_len: Optional[float] = None,
) -> int:
    """Attention-score + weighted-sum FLOPs per token, amortised over the layer.

    ``QK^T`` and ``AV`` each cost ``2 * n_heads * L * head_dim`` FLOPs for one token,
    so the per-token term grows linearly with the (average) attended length. When
    ``avg_attn_len`` is given (e.g. the mean context length actually attended to), it
    is used instead of ``seq_len``; this matters for decoding with a KV cache.
    """
    length = float(avg_attn_len if avg_attn_len is not None else seq_len)
    length = max(length, 1.0)
    return FLOP_PER_MAC * FLOP_PER_MAC * spec.num_heads * length * spec.head_dim


def lm_head_flops_per_token(spec: ModelSpec) -> int:
    """LM-head projection FLOPs per token (zero when the embedding is tied)."""
    if spec.tied_embedding:
        return 0
    return FLOP_PER_MAC * int(spec.hidden_size) * int(spec.vocab_size)


def embedding_flops_per_token(spec: ModelSpec) -> int:
    """Embedding lookups are gathers: no FLOPs. Kept for API symmetry/documentation."""
    return 0


def nonlinear_flops_per_token(spec: ModelSpec) -> int:
    """Optional estimate of FLOPs spent in GELU/LayerNorm/softmax.

    ELECTRA's script counts only matmuls, so this term is **off by default**; it is
    provided so callers can report the (small) activation overhead if they wish.
    """
    d = int(spec.hidden_size)
    i = int(spec.intermediate_size)
    # ~6 flops/element for gelu + 2*d per layernorm + softmax over the sequence
    return int(spec.num_layers * (6 * i + 2 * (2 * d)) + 5 * int(spec.vocab_size))


def flops_per_token(
    spec: ModelSpec,
    seq_len: int = 1,
    avg_attn_len: Optional[float] = None,
    include_lm_head: bool = True,
    include_nonlinear: bool = False,
) -> int:
    """Forward-pass FLOPs per token at (average) context length ``seq_len``.

    This is the ELECTRA ``FLOPs_compute`` formula specialised to a single token:
    ``2 * (dense params) + attention term``. Note the ``~2 * params`` rule of thumb
    emerges because every parameter is used once with a multiply-accumulate.
    """
    total = spec.num_layers * (
        layer_dense_flops_per_token(spec)
        + attention_flops_per_token(spec, seq_len=seq_len, avg_attn_len=avg_attn_len)
    )
    if include_lm_head:
        total += lm_head_flops_per_token(spec)
    total += embedding_flops_per_token(spec)
    if include_nonlinear:
        total += nonlinear_flops_per_token(spec)
    return int(total)


def flops_per_forward(
    spec: ModelSpec,
    seq_len: int,
    avg_attn_len: Optional[float] = None,
    include_lm_head: bool = True,
    include_nonlinear: bool = False,
) -> int:
    """Total forward FLOPs to process ``seq_len`` tokens with causal attention.

    Attention is quadratic, so the layer attention term is evaluated at the *mean*
    attended length ``(seq_len + 1) / 2`` (each query attends to ``1..L`` keys) unless
    an explicit ``avg_attn_len`` is supplied; the dense terms then scale by ``seq_len``.
    """
    L = int(max(seq_len, 1))
    if avg_attn_len is None:
        avg_attn_len = (L + 1) / 2.0
    dense = (
        spec.num_layers * layer_dense_flops_per_token(spec)
        + (lm_head_flops_per_token(spec) if include_lm_head else 0)
        + embedding_flops_per_token(spec)
    )
    attn = spec.num_layers * attention_flops_per_token(spec, avg_attn_len=avg_attn_len)
    total = L * dense + L * attn
    if include_nonlinear:
        total += L * nonlinear_flops_per_token(spec)
    return int(total)


def prefill_flops(
    spec: ModelSpec,
    prompt_len: int,
    include_lm_head: bool = True,
    include_nonlinear: bool = False,
) -> int:
    """FLOPs to process the prompt (a single forward pass over ``prompt_len`` tokens)."""
    return flops_per_forward(
        spec,
        prompt_len,
        include_lm_head=include_lm_head,
        include_nonlinear=include_nonlinear,
    )


def decode_flops(
    spec: ModelSpec,
    prompt_len: int,
    n_new_tokens: int,
    kv_cache: bool = True,
    include_lm_head: bool = True,
    include_nonlinear: bool = False,
) -> int:
    """FLOPs to autoregressively generate ``n_new_tokens`` continuation tokens.

    With a KV cache every decoding step re-runs the dense matmuls for **one** token and
    attends over the growing context, so the attention term is evaluated at the mean
    context length. Without a cache every step re-processes the full sequence.
    """
    n_new = int(max(n_new_tokens, 0))
    if n_new == 0:
        return 0
    total = 0
    for t in range(n_new):
        ctx = int(prompt_len) + t
        if kv_cache:
            total += flops_per_token(
                spec,
                seq_len=1,
                avg_attn_len=max(ctx, 1),
                include_lm_head=include_lm_head,
                include_nonlinear=include_nonlinear,
            )
        else:
            total += flops_per_forward(
                spec,
                max(ctx, 1),
                include_lm_head=include_lm_head,
                include_nonlinear=include_nonlinear,
            )
    return int(total)


def generation_flops(
    spec: ModelSpec,
    prompt_len: int,
    n_new_tokens: int,
    gamma: float = VANILLA_GAMMA,
    kv_cache: bool = True,
    include_nonlinear: bool = False,
) -> Dict[str, Any]:
    """Prefill + decode FLOPs for a CFG/vanilla generation run.

    Returns a dict with ``vanilla_flops``, ``total_flops``, ``cfg_multiplier`` and the
    per-token averages (useful for the Section 4.1 accuracy-vs-log-FLOP plots).
    """
    pre = prefill_flops(spec, prompt_len, include_nonlinear=include_nonlinear)
    dec = decode_flops(
        spec,
        prompt_len,
        n_new_tokens,
        kv_cache=kv_cache,
        include_nonlinear=include_nonlinear,
    )
    mult = cfg_multiplier(gamma)
    total = int((pre + dec) * mult)
    n_tokens = int(max(prompt_len, 1) + max(n_new_tokens, 0))
    return {
        "model": spec.name,
        "prompt_len": int(prompt_len),
        "n_new_tokens": int(n_new_tokens),
        "gamma": float(gamma),
        "cfg_multiplier": mult,
        "prefill_flops": int(pre * mult),
        "decode_flops": int(dec * mult),
        "vanilla_flops": int(pre + dec),
        "total_flops": total,
        "flops_per_token": float(total) / n_tokens,
        "log_flops": math.log10(max(total, 1)),
    }


# --------------------------------------------------------------------------------------
# CFG multiplier / equivalents
# --------------------------------------------------------------------------------------


def is_cfg(gamma: float) -> bool:
    """True when a guidance strength requires the second (unconditional) forward pass."""
    return float(gamma) != float(VANILLA_GAMMA)


def cfg_multiplier(gamma: float = CFG_GAMMA) -> float:
    """Inference-FLOPs multiplier induced by CFG (Section 4).

    ``gamma == 1`` reproduces vanilla conditional prompting (1 pass);
    any other ``gamma`` runs the unconditional pass as well, i.e. **2x** FLOPs.
    """
    return CFG_INFERENCE_MULTIPLIER if is_cfg(gamma) else 1.0


#: Name kept for symmetry with :func:`src.cfg.sampler.cfG_overhead_factor`.
cfg_overhead_factor = cfg_multiplier


def vanilla_flops_per_token(spec: ModelSpec, seq_len: int = 1, **kwargs: Any) -> int:
    """Per-token FLOPs for vanilla (unguided) inference."""
    return flops_per_token(spec, seq_len=seq_len, **kwargs)


def cfg_flops_per_token(
    spec: ModelSpec, gamma: float = CFG_GAMMA, seq_len: int = 1, **kwargs: Any
) -> int:
    """Per-token FLOPs when CFG is applied (``2x`` vanilla for ``gamma != 1``)."""
    return int(cfg_multiplier(gamma) * flops_per_token(spec, seq_len=seq_len, **kwargs))


def inference_flops_per_token(
    spec: ModelSpec, gamma: float = VANILLA_GAMMA, seq_len: int = 1, **kwargs: Any
) -> float:
    """Per-token inference FLOPs for a (possibly CFG-guided) model."""
    return float(cfg_multiplier(gamma) * flops_per_token(spec, seq_len=seq_len, **kwargs))


def expected_cfg_flops(spec: ModelSpec, seq_len: int = 1, **kwargs: Any) -> int:
    """Section 4 statement: CFG doubles the FLOPs of the same model."""
    return int(CFG_INFERENCE_MULTIPLIER * flops_per_token(spec, seq_len=seq_len, **kwargs))


def training_flops_multiplier() -> float:
    """Training-side analogue (documented for completeness, not used in the paper)."""
    return 3.0


def doubled_flops_per_token(spec: ModelSpec, gamma: float = CFG_GAMMA, seq_len: int = 1) -> int:
    """FLOPs per token of ``spec`` *when run with CFG* (the budget to match)."""
    return int(flops_per_token(spec, seq_len=seq_len) * cfg_multiplier(gamma))


def equivalent_vanilla_spec(
    spec: ModelSpec,
    candidates: Optional[Iterable[Any]] = None,
    gamma: float = CFG_GAMMA,
    metric: str = "params",
) -> Dict[str, Any]:
    """Find the vanilla model whose budget best matches this model's *CFG* budget.

    This operationalises the Section 4.1 question -- "should [users] not run a model
    twice as big instead?" -- by locating the family member whose parameter count (or
    per-token FLOPs) is closest to ``cfg_multiplier(gamma) x budget(spec)``.
    """
    pool: List[ModelSpec] = []
    for c in (candidates if candidates is not None else MODEL_SPECS.values()):
        s = get_spec(c)
        if s.name == spec.name:
            continue
        pool.append(s)
    if not pool:
        return {"target": spec.name, "candidate": None, "ratio": float("nan")}

    mult = cfg_multiplier(gamma)
    if metric == "flops":
        target_budget = mult * flops_per_token(spec, seq_len=1)
        budget = lambda s: flops_per_token(s, seq_len=1)  # noqa: E731
    else:
        target_budget = mult * spec.num_parameters
        budget = lambda s: s.num_parameters  # noqa: E731

    best = min(pool, key=lambda s: abs(math.log(max(budget(s), 1)) - math.log(max(target_budget, 1))))
    ratio = budget(best) / float(target_budget)
    return {
        "target": spec.name,
        "target_budget": float(target_budget),
        "target_params": int(spec.num_parameters),
        "candidate": best.name,
        "candidate_budget": float(budget(best)),
        "candidate_params": int(best.num_parameters),
        "ratio": float(ratio),
        "metric": metric,
        "gamma": float(gamma),
    }


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------


def params_table(specs: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
    """Parameter counts vs. their published sizes for a set of models."""
    names = list(specs) if specs is not None else [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b",
        "pythia-2.8b", "pythia-6.9b", "pythia-12b",
        "codegen-350m-mono", "codegen-2b-mono", "codegen-6b-mono", "falcon-7b",
    ]
    rows = []
    for n in names:
        s = get_spec(n)
        rows.append(
            {
                "model": s.name,
                "params": s.num_parameters,
                "params_billions": round(s.params_billions, 3),
                "hidden_size": s.hidden_size,
                "num_layers": s.num_layers,
                "kv_heads": s.n_kv_heads,
            }
        )
    return rows


def flops_table(
    specs: Optional[Iterable[Any]] = None,
    gamma: float = CFG_GAMMA,
    seq_len: int = 1,
) -> List[Dict[str, Any]]:
    """Per-model vanilla vs. CFG FLOPs per token at context length ``seq_len``."""
    names = list(specs) if specs is not None else [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "pythia-1b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
        "codegen-350m-mono", "codegen-2b-mono", "codegen-6b-mono", "falcon-7b",
    ]
    rows = []
    for n in names:
        s = get_spec(n)
        v = flops_per_token(s, seq_len=seq_len)
        c = cfg_flops_per_token(s, gamma=gamma, seq_len=seq_len)
        rows.append(
            {
                "model": s.name,
                "params_billions": round(s.params_billions, 3),
                "seq_len": int(seq_len),
                "vanilla_flops_per_token": int(v),
                "cfg_flops_per_token": int(c),
                "multiplier": float(cfg_multiplier(gamma)),
                "log10_vanilla_flops": math.log10(max(v, 1)),
                "log10_cfg_flops": math.log10(max(c, 1)),
            }
        )
    return rows


def format_flops_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Plain-text rendering of :func:`flops_table` (for logs/README)."""
    header = f"{'model':<20}{'params(B)':>10}{'vanilla/tok':>14}{'cfg/tok':>14}{'x':>5}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['model']:<20}{r['params_billions']:>10.3f}"
            f"{r['vanilla_flops_per_token']:>14,}{r['cfg_flops_per_token']:>14,}"
            f"{r['multiplier']:>5.1f}"
        )
    return "\n".join(lines)


def summarize_flops(spec: ModelSpec, gamma: float = CFG_GAMMA, seq_len: int = 1) -> str:
    """One-line summary of a spec's parameter/FLOP budget."""
    v = flops_per_token(spec, seq_len=seq_len)
    c = cfg_flops_per_token(spec, gamma=gamma, seq_len=seq_len)
    return (
        f"{spec.name}: {spec.params_billions:.3f}B params | vanilla {v:,} FLOPs/token "
        f"| CFG(gamma={gamma}) {c:,} FLOPs/token ({cfg_multiplier(gamma):.1f}x)"
    )


def flops_report(
    specs: Optional[Iterable[Any]] = None,
    gamma: float = CFG_GAMMA,
    seq_len: int = 1,
) -> Dict[str, Any]:
    """Full Section 4.1 report: table, equivalences and the Table 6 anchors."""
    rows = flops_table(specs, gamma=gamma, seq_len=seq_len)
    equivalents = []
    for r in rows:
        try:
            equivalents.append(equivalent_vanilla_spec(get_spec(r["model"]), gamma=gamma))
        except KeyError:  # pragma: no cover - defensive
            continue
    return {
        "gamma": float(gamma),
        "seq_len": int(seq_len),
        "cfg_multiplier": cfg_multiplier(gamma),
        "flops": rows,
        "table": format_flops_table(rows),
        "equivalents": equivalents,
        "params": params_table([r["model"] for r in rows]),
        "table6": paper_ancova_table(),
        "check": check_against_paper(),
    }


def save_flops(path: str, report: Dict[str, Any]) -> str:
    """Persist a report produced by :func:`flops_report` as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path


# --------------------------------------------------------------------------------------
# Table 6 / ANCOVA anchors (Appendix C.2)
# --------------------------------------------------------------------------------------


def paper_ancova_table(
    p_values: Optional[Dict[str, Dict[str, Any]]] = None,
    alpha: float = SIGNIFICANCE_LEVEL,
) -> List[Dict[str, Any]]:
    """Sortable view of Appendix C.2 Table 6 (ascending p-value)."""
    pv = p_values if p_values is not None else PAPER_TABLE6
    tasks = [t for t in BENCHMARK_TASKS if t in pv] or list(pv)
    rows = []
    for t in tasks:
        entry = pv[t]
        p = float(entry["p_value"] if isinstance(entry, dict) else entry)
        winner = entry.get("winner") if isinstance(entry, dict) else None
        rows.append(
            {
                "task": t,
                "p_value": p,
                "significant": bool(p < alpha),
                "winner": winner if p < alpha else None,
                "favors": winner if p < alpha else None,
            }
        )
    rows.sort(key=lambda r: r["p_value"])
    return rows


def significant_tasks(
    alpha: float = SIGNIFICANCE_LEVEL,
    p_values: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Tasks where CFG and the doubled-size vanilla model differ at ``alpha``."""
    return [r for r in paper_ancova_table(p_values, alpha) if r["significant"]]


def favor_counts(
    alpha: float = SIGNIFICANCE_LEVEL,
    p_values: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Section 4 headline counts: inconclusive / favours CFG / favours vanilla.

    Expected on the paper's numbers: ``{'inconclusive': 5, 'cfg': 2, 'vanilla': 2}``.
    """
    rows = paper_ancova_table(p_values, alpha)
    counts = {"inconclusive": 0, "cfg": 0, "vanilla": 0}
    for r in rows:
        if not r["significant"]:
            counts["inconclusive"] += 1
        elif r["winner"] == "cfg":
            counts["cfg"] += 1
        elif r["winner"] == "vanilla":
            counts["vanilla"] += 1
    return counts


def check_against_paper(
    alpha: float = SIGNIFICANCE_LEVEL,
    p_values: Optional[Dict[str, Dict[str, Any]]] = None,
    expected: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Verify the ``5 insignificant / 2 CFG / 2 vanilla`` split of Section 4."""
    expected = expected or {"inconclusive": 5, "cfg": 2, "vanilla": 2}
    counts = favor_counts(alpha, p_values)
    matches = all(counts.get(k, 0) == v for k, v in expected.items())
    return {
        "alpha": float(alpha),
        "counts": counts,
        "expected": dict(expected),
        "matches_paper": bool(matches),
        "significant": [r["task"] for r in significant_tasks(alpha, p_values)],
    }


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------


def _demo() -> None:
    """Assertions validating the estimator against known sizes and identities."""
    # -- parameter counts land on the published model sizes -------------------------
    expectations = {
        "gpt2": 0.124,
        "gpt2-medium": 0.355,
        "gpt2-large": 0.774,
        "pythia-160m": 0.162,
        "pythia-410m": 0.405,
        "pythia-1.4b": 1.41,
        "pythia-6.9b": 6.86,
        "pythia-12b": 11.8,
        "codegen-350m-mono": 0.36,
        "codegen-2b-mono": 2.45,
        "falcon-7b": 7.2,
    }
    for name, expected_b in expectations.items():
        spec = get_spec(name)
        got = spec.params_billions
        assert abs(got - expected_b) / expected_b < PARAMETER_COUNT_TOLERANCE, (
            f"{name}: {got:.3f}B vs expected ~{expected_b}B"
        )

    # -- the ~2 * params rule of thumb for a 1-token forward pass ------------------
    gpt2 = get_spec("gpt2")
    per_tok = flops_per_token(gpt2, seq_len=1, include_lm_head=False)
    # tied embeddings: per-token FLOPs should be close to 2x the non-embedding params
    non_emb = gpt2.num_parameters - gpt2.vocab_size * gpt2.emb_dim
    assert 0.5 * non_emb < per_tok < 1.5 * non_emb

    # -- CFG doubles inference FLOPs; gamma = 1 does not ---------------------------
    assert cfg_multiplier(1.0) == 1.0
    assert cfg_multiplier(1.5) == 2.0
    assert cfg_multiplier(2.0) == 2.0
    assert not is_cfg(1.0) and is_cfg(1.1)
    v = flops_per_token(gpt2, seq_len=32)
    assert expected_cfg_flops(gpt2, seq_len=32) == 2 * v
    assert cfg_flops_per_token(gpt2, gamma=1.0, seq_len=32) == v
    assert cfg_overhead_factor(1.5) == 2.0

    # -- attention grows with the context length ----------------------------------
    a_short = attention_flops_per_token(gpt2, seq_len=16)
    a_long = attention_flops_per_token(gpt2, seq_len=512)
    assert a_long == 32 * a_short
    assert flops_per_token(gpt2, seq_len=512) > flops_per_token(gpt2, seq_len=16)

    # -- decode with a kv cache equals the sum of per-token estimates --------------
    dec = decode_flops(gpt2, prompt_len=8, n_new_tokens=4)
    manual = sum(flops_per_token(gpt2, seq_len=1, avg_attn_len=8 + t) for t in range(4))
    assert dec == manual
    # without a cache each step re-runs the whole prefix
    assert decode_flops(gpt2, 8, 4, kv_cache=False) > dec
    assert decode_flops(gpt2, 8, 0) == 0

    # -- the "twice as big" comparison of Section 4.1 ------------------------------
    eq = equivalent_vanilla_spec(get_spec("pythia-2.8b"), gamma=1.5)
    assert eq["candidate"] == "pythia-6.9b", eq
    assert abs(eq["target_params"] * 2 - eq["candidate_params"]) / eq["candidate_params"] < 0.3
    eq2 = equivalent_vanilla_spec(get_spec("codegen-350m-mono"), gamma=1.5, metric="flops")
    assert eq2["candidate"] is not None

    # -- flops/params tables and the Table 6 anchors -------------------------------
    rows = flops_table(["gpt2", "pythia-12b"], gamma=1.5)
    assert len(rows) == 2 and rows[1]["multiplier"] == 2.0
    assert "gpt2" in format_flops_table(rows)
    assert summarize_flops(gpt2, 1.5)

    check = check_against_paper()
    assert check["matches_paper"], check
    assert check["counts"] == {"inconclusive": 5, "cfg": 2, "vanilla": 2}
    assert set(check["significant"]) == {"sciq", "triviaqa", "winogrande", "lambada_openai"}
    tbl = paper_ancova_table()
    assert tbl[0]["task"] == "lambada_openai" and tbl[0]["winner"] == "cfg"

    # -- generation accounting -----------------------------------------------------
    gen = generation_flops(gpt2, prompt_len=16, n_new_tokens=16, gamma=1.5)
    assert gen["total_flops"] == 2 * gen["vanilla_flops"]
    assert gen["log_flops"] > 0

    print("flops.py self-test passed")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _demo()
    print()
    print(format_flops_table(flops_table()))
    print()
    print(check_against_paper())
