"""Inference FLOP accounting (Section 4.1 / Appendix C.2).

The paper measures FLOPs with the script released alongside ELECTRA
(``google-research/electra/flops_computation.py``), which counts a
multiply-accumulate as two floating point operations.  We follow the same
convention for a decoder-only transformer forward pass.

Per transformer layer, with sequence length ``S``, hidden size ``d``,
feed-forward inner size ``d_ff`` and ``h`` heads:

    QKV projection          : 2 * S * 3 * d * d
    attention scores (QK^T) : 2 * S * S * d
    attention output (AV)   : 2 * S * S * d
    attention out-projection: 2 * S * d * d
    feed-forward (2 matmuls): 2 * 2 * S * d * d_ff

plus, once for the whole model:

    token embedding lookup  : 0            (a lookup, not a matmul)
    LM head                 : 2 * S * d * V

Classifier-free guidance performs *two* forward passes (conditional and
unconditional), which is the "CFG doubles the inference FLOPs" statement of
Section 4.  Two consequences of the convention above matter for the paper's
argument and are reproduced here:

* the attention term grows with ``S`` (``4 * S * d`` per token per layer),
  so the FLOPs-per-token cost grows with the context length, and
* doubling the model size costs roughly twice the parameters but also
  doubles every matmul, whereas CFG doubles the *number of passes* at fixed
  parameter count.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def _cfg_get(config, name: str, default=None):
    value = getattr(config, name, None)
    if value is None:
        value = default
    return value


def model_dims(config) -> dict:
    """Extract ``(n_layer, d_model, d_ff, vocab)`` from a HF model config."""
    n_layer = _cfg_get(config, "n_layer")
    if n_layer is None:
        n_layer = _cfg_get(config, "num_hidden_layers")
    d_model = _cfg_get(config, "n_embd")
    if d_model is None:
        d_model = _cfg_get(config, "hidden_size")
    d_ff = _cfg_get(config, "n_inner")
    if d_ff is None:
        d_ff = _cfg_get(config, "intermediate_size")
    if d_ff is None:
        d_ff = 4 * d_model
    vocab = _cfg_get(config, "vocab_size")
    return {
        "n_layer": int(n_layer),
        "d_model": int(d_model),
        "d_ff": int(d_ff),
        "vocab": int(vocab),
    }


def flops_per_token(
    config,
    seq_len: int,
    n_passes: int = 1,
    include_lm_head: bool = True,
) -> float:
    """FLOPs required to process one token at context length ``seq_len``.

    Args:
        config: HF model config (GPT-2, Pythia/GPT-NeoX, LLaMA, ...).
        seq_len: context length used for the attention term.
        n_passes: ``2`` for CFG (conditional + unconditional), ``1`` for
            vanilla inference.
        include_lm_head: include the ``2 * d * V`` vocabulary projection.
    """
    dims = model_dims(config)
    d = dims["d_model"]
    d_ff = dims["d_ff"]
    V = dims["vocab"]
    n_layer = dims["n_layer"]
    per_layer = 2 * 3 * d * d + 2 * d * d + 2 * 2 * d * d_ff + 4 * seq_len * d
    total = n_layer * per_layer
    if include_lm_head:
        total += 2 * d * V
    return float(total * n_passes)


def transformer_flops(
    config,
    seq_len: int,
    n_passes: int = 1,
    include_lm_head: bool = True,
    batch_size: int = 1,
) -> float:
    """Total forward-pass FLOPs for a batch of sequences of length ``seq_len``."""
    return batch_size * seq_len * flops_per_token(
        config, seq_len, n_passes=n_passes, include_lm_head=include_lm_head
    )


def cfg_flops_per_token(config, seq_len: int, include_lm_head: bool = True) -> float:
    """FLOPs per token for CFG inference (two passes)."""
    return flops_per_token(config, seq_len, n_passes=2, include_lm_head=include_lm_head)


def flops_from_model(model, seq_len: int, n_passes: int = 1) -> float:
    """Convenience wrapper taking an ``nn.Module`` instead of a config."""
    config = model.config if hasattr(model, "config") else model
    return flops_per_token(config, seq_len, n_passes=n_passes)


def parameter_count(model) -> int:
    """Number of parameters (used for the model-scaling comparisons)."""
    return int(sum(p.numel() for p in model.parameters()))


def kv_cache_bytes(config, seq_len: int, batch_size: int = 1, dtype_bytes: int = 2) -> int:
    """Key/value cache size for a decoder-only transformer.

    Used by the memory analysis discussion of Section 4 (Appendix C.3).
    CFG doubles this quantity because both branches keep a cache.
    """
    dims = model_dims(config)
    n_layer = dims["n_layer"]
    d = dims["d_model"]
    n_head = _cfg_get(config, "n_head", _cfg_get(config, "num_attention_heads", 1))
    d_head = d // int(n_head)
    per_token = 2 * n_layer * 2 * d_head * int(n_head)  # (k, v) per layer
    return batch_size * seq_len * per_token * dtype_bytes
