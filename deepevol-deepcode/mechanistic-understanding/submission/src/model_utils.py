"""GPT2(-medium) model utilities: loading, hooks, residual-stream and MLP access.

Notation follows Section 2 of the paper ("Preliminaries", borrowed from Geva et al. 2022):

    x_i^{l+1} = x_i^l + MLP^l( x_i^l + Att^l(x_i^l) )                                   (Eq. 1)
    MLP^l(x^l) = sigma(W_K^l x^l) W_V^l      W_K^l, W_V^l in R^{d_mlp x d}             (Eq. 2)
    MLP^l(x^l) = sum_i sigma(x^l . k_i^l) v_i^l = sum_i m_i^l v_i^l                   (Eq. 3)

* ``k_i^l`` is the i-th ROW of ``W_K^l`` (key vector).
* ``v_i^l`` is the i-th COLUMN of ``W_V^l`` -- equivalently the i-th row of ``W_V^l``
  when ``W_V`` is stored transposed, which is the convention used throughout this
  code base for convenience (rows == vectors).
* ``x^{l-mid}`` is the *intermittent* residual stream at layer ``l``: after the
  attention block and BEFORE the MLP block.

HuggingFace GPT-2 layout (``Conv1D`` layers store the weight transposed relative to
``nn.Linear``):

    model.transformer.h[l].mlp.c_fc.weight    -> [d_model, d_mlp]   (up projection)
    model.transformer.h[l].mlp.c_proj.weight  -> [d_mlp, d_model]   (down projection)

Therefore, for layer ``l`` and index ``i``:

    k_i^l = mlp.c_fc.weight[:, i]      (column, shape [d_model])
    v_i^l = mlp.c_proj.weight[i, :]    (row,    shape [d_model])

Per the author clarification: "Idx" refers to the index of a value vector in the MLP
weights, i.e. the column of the down-projection in the paper's convention.

Gated Linear Units (Eq. 4) are only used by Llama2-family models, which are OUT OF
SCOPE for this reproduction; the GLU helpers below are provided as a documented stub.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

GPT2_MEDIUM = "openai-community/gpt2-medium"
GPT2_SMALL = "openai-community/gpt2"

#: Models whose MLP blocks are Gated Linear Units (Eq. 4). Out of scope.
GLU_MODEL_MARKERS = ("llama", "mistral", "gemma", "qwen")


# --------------------------------------------------------------------------------------
# Model info / loading
# --------------------------------------------------------------------------------------
@dataclass
class ModelInfo:
    """Basic architectural constants, mirroring the paper's notation."""

    name: str
    n_layers: int  # L
    d_model: int  # d
    d_mlp: int  # d_mlp
    n_heads: int
    vocab_size: int
    is_glu: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        kind = "GLU" if self.is_glu else "MLP"
        return (
            f"<ModelInfo {self.name}: L={self.n_layers} d_model={self.d_model} "
            f"d_mlp={self.d_mlp} n_heads={self.n_heads} vocab={self.vocab_size} "
            f"({kind})>"
        )


def is_glu_model(model_name: str) -> bool:
    """Return True for Llama2-style GLU architectures (out of reproduction scope)."""
    low = model_name.lower()
    return any(marker in low for marker in GLU_MODEL_MARKERS)


def load_model(
    model_name: str = GPT2_MEDIUM,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    eval_mode: bool = True,
):
    """Load a (GPT-2) causal LM plus tokenizer.

    Returns ``(model, tokenizer)``. GPT-2 has tied input/output embeddings, so the
    embedding matrix ``E`` and the unembedding matrix ``U`` are the same tensor
    (``model.transformer.wte.weight``), matching Section 2 of the paper.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if device is not None:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model, tokenizer


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick CUDA when available (fall back to CPU for smoke tests)."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_info(model, name: str = "model") -> ModelInfo:
    """Extract architecture constants from a HF GPT2-style model."""
    cfg = model.config
    n_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers")
    d_model = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size")
    n_heads = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads")
    d_mlp = getattr(cfg, "n_inner", None)
    if d_mlp is None:
        d_mlp = 4 * d_model
    glu = bool(getattr(cfg, "model_type", "") in ("llama", "mistral", "gemma", "qwen2"))
    return ModelInfo(
        name=name,
        n_layers=int(n_layers),
        d_model=int(d_model),
        d_mlp=int(d_mlp),
        n_heads=int(n_heads),
        vocab_size=int(cfg.vocab_size),
        is_glu=glu,
        extra={"model_type": getattr(cfg, "model_type", "gpt2")},
    )


def transformer_layers(model) -> Sequence[torch.nn.Module]:
    """Return the list of transformer blocks for GPT2/Llama-style HF models."""
    base = getattr(model, "transformer", None)
    if base is not None and hasattr(base, "h"):
        return base.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Unsupported model class: {type(model).__name__}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# MLP key / value matrices (Eq. 2)
# --------------------------------------------------------------------------------------
def get_mlp_matrices(model, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(W_K, W_V)`` for ``layer`` both shaped ``[d_mlp, d_model]``.

    Row ``i`` of the returned ``W_K`` is ``MLP.k_i^l``; row ``i`` of ``W_V`` is
    ``MLP.v_i^l`` (paper convention: value vectors are columns of ``W_V``).
    Gradients are preserved (the returned tensors are views).
    """
    mlp = transformer_layers(model)[layer].mlp
    if hasattr(mlp, "c_fc"):  # GPT-2
        # c_fc.weight: [d_model, d_mlp] -> transpose to [d_mlp, d_model]
        W_K = mlp.c_fc.weight.t().contiguous().t()  # keep a differentiable view trick
        W_K = mlp.c_fc.weight.transpose(0, 1)
        # c_proj.weight: [d_mlp, d_model] -> already [d_mlp, d_model]
        W_V = mlp.c_proj.weight
        return W_K, W_V
    if hasattr(mlp, "gate_proj"):  # Llama2-style GLU (out of scope)
        # GLU: (sigma(W1 x) * W2 x) W_V   (Eq. 4); value vectors live in down_proj
        W_V = mlp.down_proj.weight  # [d_model, d_mlp]
        W_K = mlp.gate_proj.weight  # [d_mlp, d_model]
        return W_K, W_V.t()
    raise ValueError("Unrecognised MLP block layout")


def get_mlp_bias(model, layer: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Optional bias terms (GPT-2 has them; the paper omits biases for clarity)."""
    mlp = transformer_layers(model)[layer].mlp
    b_k = getattr(getattr(mlp, "c_fc", None), "bias", None)
    b_v = getattr(getattr(mlp, "c_proj", None), "bias", None)
    return b_k, b_v


def get_value_vector(model, layer: int, idx: int) -> torch.Tensor:
    """``MLP.v_idx^layer`` -- a [d_model] vector (row ``idx`` of ``W_V``)."""
    _, W_V = get_mlp_matrices(model, layer)
    return W_V[idx]


def get_key_vector(model, layer: int, idx: int) -> torch.Tensor:
    """``MLP.k_idx^layer`` -- a [d_model] vector (row ``idx`` of ``W_K``)."""
    W_K, _ = get_mlp_matrices(model, layer)
    return W_K[idx]


def iter_value_vectors(model, layers: Optional[Iterable[int]] = None) -> Iterator[Tuple[int, int, torch.Tensor]]:
    """Yield ``(layer, idx, v)`` for every value vector of the model."""
    info = model_info(model)
    layer_ids = range(info.n_layers) if layers is None else layers
    for l in layer_ids:
        _, W_V = get_mlp_matrices(model, l)
        for i in range(W_V.shape[0]):
            yield l, i, W_V[i]


def iter_key_vectors(model, layers: Optional[Iterable[int]] = None) -> Iterator[Tuple[int, int, torch.Tensor]]:
    """Yield ``(layer, idx, k)`` for every key vector of the model."""
    info = model_info(model)
    layer_ids = range(info.n_layers) if layers is None else layers
    for l in layer_ids:
        W_K, _ = get_mlp_matrices(model, l)
        for i in range(W_K.shape[0]):
            yield l, i, W_K[i]


def all_value_vectors(model, layers: Optional[Iterable[int]] = None) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Stack every value vector into ``[n_vectors, d_model]`` plus ``(layer, idx)`` indices."""
    vectors, indices = [], []
    for l, i, v in iter_value_vectors(model, layers):
        vectors.append(v.detach())
        indices.append((l, i))
    return torch.stack(vectors, dim=0), indices


def all_key_vectors(model, layers: Optional[Iterable[int]] = None) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Stack every key vector into ``[n_vectors, d_model]`` plus ``(layer, idx)`` indices."""
    vectors, indices = [], []
    for l, i, k in iter_key_vectors(model, layers):
        vectors.append(k.detach())
        indices.append((l, i))
    return torch.stack(vectors, dim=0), indices


def scale_key_vector(model, layer: int, idx: int, scale: float, inplace: bool = True) -> torch.Tensor:
    """Multiply ``MLP.k_idx^layer`` by ``scale``.

    Used by the un-alignment experiment (Section 6), where the key vectors of the 7
    most toxic value vectors are scaled by 10x in order to enlarge their activation
    regions ``gamma(k)`` and re-activate toxicity.
    """
    mlp = transformer_layers(model)[layer].mlp
    if not inplace:
        raise ValueError("scale_key_vector requires inplace=True (weights are views)")
    with torch.no_grad():
        if hasattr(mlp, "c_fc"):  # GPT-2: k_i is column i of c_fc.weight [d_model, d_mlp]
            mlp.c_fc.weight[:, idx] *= scale
            return mlp.c_fc.weight[:, idx]
        mlp.gate_proj.weight[idx, :] *= scale
        return mlp.gate_proj.weight[idx, :]


# --------------------------------------------------------------------------------------
# Embedding / unembedding
# --------------------------------------------------------------------------------------
def get_embedding(model) -> torch.Tensor:
    """Input embedding matrix ``E`` shaped ``[vocab, d_model]``."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight
    raise ValueError("Cannot locate input embedding matrix")


def get_unembedding(model) -> torch.Tensor:
    """Unembedding matrix ``U`` shaped ``[vocab, d_model]``.

    GPT-2 ties ``U = E`` (``lm_head`` shares ``wte.weight``).
    """
    out = None
    if hasattr(model, "lm_head"):
        out = model.lm_head.weight
    elif hasattr(model, "get_output_embeddings"):
        out = model.get_output_embeddings().weight
    if out is None:
        return get_embedding(model)
    return out


def project_to_vocab(model, vector: torch.Tensor) -> torch.Tensor:
    """Vocabulary-space projection ``r = E v`` (Section 2 / Appendix A).

    ``vector`` may be a single ``[d_model]`` vector or a batch ``[..., d_model]``;
    the projection is ``r_w = e_w . v`` and the returned tensor has shape
    ``[..., vocab]``.
    """
    E = get_embedding(model)
    return torch.matmul(vector, E.t())


def unembed_hidden_state(model, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the final layer norm + unembedding to a residual stream (logit lens)."""
    ln_f = get_final_norm(model)
    if ln_f is not None:
        hidden = ln_f(hidden)
    U = get_unembedding(model)
    return torch.matmul(hidden, U.t())


def get_final_norm(model):
    """Return the final LayerNorm (``ln_f``) if it exists."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


# --------------------------------------------------------------------------------------
# Activation function sigma (GeLU for GPT-2)
# --------------------------------------------------------------------------------------
def activation_sigma(model, tensor: torch.Tensor) -> torch.Tensor:
    """Apply the MLP non-linearity ``sigma`` (GeLU for GPT-2, SiLU for Llama2)."""
    base = getattr(model, "transformer", None)
    act = getattr(getattr(base, "h", [None])[0] if base is not None else None, "mlp", None)
    if act is not None and hasattr(act, "act"):
        return act.act(tensor)
    if hasattr(model, "act_fn"):
        return model.act_fn(tensor)
    return torch.nn.functional.gelu(tensor)


# --------------------------------------------------------------------------------------
# Hooks: residual streams x^l, x^{l-mid} and MLP activations
# --------------------------------------------------------------------------------------
class ResidualStreamCapture:
    """Capture residual streams of a GPT2-style model for every layer.

    Captured tensors (all shaped ``[batch, seq, d_model]``):

    * ``mid[l]``  -- ``x^{l-mid}``: input to the MLP block of layer ``l``
      (i.e. after attention heads, before the MLP).
    * ``post_attn[l]`` -- alias of ``mid[l]``.
    * ``block_out[l]`` -- output of transformer block ``l`` (= ``x^{l+1}``).
    * ``block_in[0]``  -- embedding output (``x^0``) / ``x^{0-mid}``.

    ``mlp_act[l]`` optionally stores the post-GeLU MLP activations
    ``sigma(W_K^l x^l)`` (the ``m^l`` coefficients of Eq. 3).
    """

    def __init__(self, model, capture_mlp_act: bool = False):
        self.model = model
        self.capture_mlp_act = capture_mlp_act
        self.mid: Dict[int, torch.Tensor] = {}
        self.block_out: Dict[int, torch.Tensor] = {}
        self.mlp_act: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- hook bodies ------------------------------------------------------------------
    def _make_mid_hook(self, layer: int):
        def hook(module, args, kwargs, output):
            x = args[0] if len(args) else kwargs.get("hidden_states")
            self.mid[layer] = x.detach()
            if self.capture_mlp_act:
                W_K, b_k = get_mlp_matrices(self.model, layer)[0], get_mlp_bias(self.model, layer)[0]
                pre = torch.matmul(x.detach(), W_K.t())
                if b_k is not None:
                    pre = pre + b_k
                self.mlp_act[layer] = activation_sigma(self.model, pre)

        return hook

    def _make_block_hook(self, layer: int):
        def hook(module, args, kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.block_out[layer] = out.detach()

        return hook

    def _register(self, module, hook) -> None:
        if hasattr(module, "register_forward_hook"):
            try:
                handle = module.register_forward_hook(hook, with_kwargs=True)
                self._handles.append(handle)
                return
            except TypeError:  # older transformers: hooks without kwargs
                pass
        self._handles.append(module.register_forward_hook(lambda m, a, o: hook(m, a, {}, o)))

    def __enter__(self) -> "ResidualStreamCapture":
        layers = transformer_layers(self.model)
        for l, block in enumerate(layers):
            mlp = block.mlp
            self._register(mlp, self._make_mid_hook(l))
            self._register(block, self._make_block_hook(l))
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self) -> None:
        self.mid.clear()
        self.block_out.clear()
        self.mlp_act.clear()

    # -- accessors --------------------------------------------------------------------
    def get_mid(self, layer: int) -> torch.Tensor:
        """``x^{layer-mid}`` for the most recent forward pass."""
        return self.mid[layer]

    def get_block_out(self, layer: int) -> torch.Tensor:
        return self.block_out[layer]

    def mid_tensor(self) -> torch.Tensor:
        """Stack ``x^{l-mid}`` over layers -> ``[L, batch, seq, d_model]``."""
        return torch.stack([self.mid[l] for l in sorted(self.mid)], dim=0)


@contextlib.contextmanager
def capture_residual_streams(model, capture_mlp_act: bool = False):
    """Context manager yielding a :class:`ResidualStreamCapture`."""
    cap = ResidualStreamCapture(model, capture_mlp_act=capture_mlp_act)
    cap.__enter__()
    try:
        yield cap
    finally:
        cap.close()


def forward_with_residuals(model, input_ids, attention_mask=None, capture_mlp_act: bool = False):
    """Forward pass that also returns captured residual streams.

    Returns ``(outputs, mid, block_out, mlp_act)`` where ``mid``/``block_out`` are
    ``[L, batch, seq, d_model]`` tensors (``mlp_act`` is ``None`` unless requested).
    """
    with capture_residual_streams(model, capture_mlp_act=capture_mlp_act) as cap:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    mid = torch.stack([cap.mid[l] for l in range(len(cap.mid))], dim=0)
    block_out = torch.stack([cap.block_out[l] for l in range(len(cap.block_out))], dim=0)
    mlp_act = None
    if capture_mlp_act and cap.mlp_act:
        mlp_act = torch.stack([cap.mlp_act[l] for l in range(len(cap.mlp_act))], dim=0)
    return outputs, mid, block_out, mlp_act


# --------------------------------------------------------------------------------------
# GLU stub (Llama2 -- OUT OF SCOPE)
# --------------------------------------------------------------------------------------
def glu_components(model, layer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(W1, W2, W_V)`` for a GLU MLP block (Eq. 4).

    OUT OF SCOPE: Llama2 results are not reproduced; this helper only documents the
    mapping between Eq. 4 and HF Llama-style weights so that the code paths exist.

    For a GLU block, value vectors are columns of ``W_V`` and are scaled by
    ``sigma(W1 x) * (W2 x)``; the "gates" are ``sigma(W1 x)`` which block their
    counterparts when the non-linearity is not activated.
    """
    mlp = transformer_layers(model)[layer].mlp
    if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")):
        raise ValueError("Model has no GLU MLP blocks (Llama2/GLU path is out of scope)")
    W1 = mlp.gate_proj.weight  # [d_mlp, d_model] -> gates after sigma
    W2 = mlp.up_proj.weight  # [d_mlp, d_model] -> element-wise counterpart
    W_V = mlp.down_proj.weight.t()  # [d_model, d_mlp] -> [d_mlp, d_model], rows = v_i
    return W1, W2, W_V


def glu_value_vectors(model, layer: int) -> torch.Tensor:
    """Value vectors (rows) of a GLU down-projection. OUT OF SCOPE for this repro."""
    _, _, W_V = glu_components(model, layer)
    return W_V


def scale_glu_gate_vector(model, layer: int, idx: int, scale: float) -> torch.Tensor:
    """Scale gate row ``idx`` of a GLU block (un-alignment analogue). OUT OF SCOPE."""
    mlp = transformer_layers(model)[layer].mlp
    with torch.no_grad():
        mlp.gate_proj.weight[idx, :] *= scale
    return mlp.gate_proj.weight[idx, :]


__all__ = [
    "GPT2_MEDIUM",
    "GPT2_SMALL",
    "GLU_MODEL_MARKERS",
    "ModelInfo",
    "activation_sigma",
    "all_key_vectors",
    "all_value_vectors",
    "capture_residual_streams",
    "forward_with_residuals",
    "get_embedding",
    "get_final_norm",
    "get_key_vector",
    "get_mlp_bias",
    "get_mlp_matrices",
    "get_unembedding",
    "get_value_vector",
    "glu_components",
    "glu_value_vectors",
    "is_glu_model",
    "iter_key_vectors",
    "iter_value_vectors",
    "load_model",
    "model_info",
    "project_to_vocab",
    "resolve_device",
    "scale_glu_gate_vector",
    "scale_key_vector",
    "set_seed",
    "transformer_layers",
    "unembed_hidden_state",
    "ResidualStreamCapture",
]
