"""Model-agnostic access to the internals that the paper manipulates.

The paper works with two architecture families:

* plain MLP blocks (GPT2): ``MLP(x) = gelu(W_K x) W_V``
* gated linear units (Llama2): ``GLU(x) = (silu(W_1 x) * W_2 x) W_V``

Everything the reproduction needs (key vectors, value vectors, activations,
residual streams at ``l-mid`` i.e. after attention and before the MLP, and
in-place rescaling of key vectors for the un-alignment experiment) is exposed
through :class:`TransformerInternals`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


def _as_weight(module) -> torch.Tensor:
    """Return the matrix of a linear-ish submodule as ``W`` with ``y = x W``."""
    if hasattr(module, "weight"):
        w = module.weight
        # nn.Linear stores [out, in] and computes x @ W.T
        if module.__class__.__name__ == "Linear":
            return w
        # transformers Conv1D stores [in, out] and computes x @ W
        return w
    raise AttributeError(f"{module} has no weight")


class TransformerInternals:
    """Uniform accessor over GPT2-style and Llama-style causal LMs."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.base = self._find_base(model)
        self.layers = self._find_layers(self.base)
        self.arch = self._detect_arch(self.layers[0])
        self.final_norm = self._find_final_norm(self.base)
        self.embedding = self.model.get_input_embeddings()
        self.unembedding = self._find_unembedding(model)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _find_base(model):
        for attr in ("transformer", "model"):
            if hasattr(model, attr):
                return getattr(model, attr)
        return model

    @staticmethod
    def _find_layers(base) -> torch.nn.ModuleList:
        for attr in ("h", "layers", "blocks", "decoder"):
            mod = getattr(base, attr, None)
            if isinstance(mod, torch.nn.ModuleList):
                return mod
            if isinstance(mod, torch.nn.Module) and hasattr(mod, "layers"):
                return mod.layers
        raise AttributeError("Could not locate transformer layers")

    @staticmethod
    def _find_final_norm(base):
        for attr in ("ln_f", "final_layernorm", "norm"):
            mod = getattr(base, attr, None)
            if mod is not None:
                return mod
        return None

    @staticmethod
    def _find_unembedding(model):
        if hasattr(model, "lm_head"):
            return model.lm_head
        return None

    @staticmethod
    def _detect_arch(layer) -> str:
        mlp = layer.mlp
        if hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
            return "mlp"  # GPT2
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            return "glu"  # Llama / Mistral style
        raise ValueError(f"Unsupported MLP block: {mlp}")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return int(self.embedding.weight.shape[1] if self.embedding.weight.ndim == 2 else self.embedding.embedding_dim)

    @property
    def d_mlp(self) -> int:
        return self.value_weight(0).shape[0]

    # -------------------------------------------------------------- weights
    def key_weight(self, layer: int, branch: str = "k") -> torch.Tensor:
        """Return ``W_K`` with shape ``[d_mlp, d_model]`` (row ``i`` = key vector ``i``)."""
        mlp = self.layers[layer].mlp
        if self.arch == "mlp":
            if branch not in ("k", "up"):
                raise ValueError("GPT2 MLPs only have a single key matrix")
            return mlp.c_fc.weight.T  # Conv1D: [d_model, d_mlp] -> [d_mlp, d_model]
        if branch == "k":  # gate branch, W_1
            return mlp.gate_proj.weight
        if branch == "up":  # linear branch, W_2
            return mlp.up_proj.weight
        raise ValueError(branch)

    def key_bias(self, layer: int, branch: str = "k") -> Optional[torch.Tensor]:
        mlp = self.layers[layer].mlp
        if self.arch == "mlp":
            return mlp.c_fc.bias
        mod = mlp.gate_proj if branch == "k" else mlp.up_proj
        return mod.bias

    def value_weight(self, layer: int) -> torch.Tensor:
        """Return ``W_V`` with shape ``[d_mlp, d_model]`` (row ``i`` = value vector ``i``)."""
        mlp = self.layers[layer].mlp
        if self.arch == "mlp":
            return mlp.c_proj.weight  # Conv1D: [d_mlp, d_model]
        return mlp.down_proj.weight.T  # nn.Linear [d_model, d_mlp] -> [d_mlp, d_model]

    def key_vectors(self, layer: int, branch: str = "k") -> torch.Tensor:
        return self.key_weight(layer, branch)

    def value_vectors(self, layer: int) -> torch.Tensor:
        return self.value_weight(layer)

    def all_value_vectors(self) -> torch.Tensor:
        """Stack the value vectors of every layer: ``[L * d_mlp, d_model]``."""
        return torch.cat([self.value_weight(l) for l in range(self.n_layers)], dim=0)

    def value_vector_locations(self) -> List[Tuple[int, int]]:
        return [(l, i) for l in range(self.n_layers) for i in range(self.d_mlp)]

    # ------------------------------------------------------------- mutations
    @torch.no_grad()
    def scale_key_vectors(self, layer_indices: List[int], indices: List[int], scale: float, branch: str = "k") -> None:
        """Multiply selected key vectors (rows of ``W_K``) by ``scale`` (Sec. 6)."""
        for l, i in zip(layer_indices, indices):
            w = self.key_weight(l, branch)
            w[i] = w[i] * scale
            self._write_key_weight(l, branch, w)

    def _write_key_weight(self, layer: int, branch: str, w: torch.Tensor) -> None:
        mlp = self.layers[layer].mlp
        with torch.no_grad():
            if self.arch == "mlp":
                mlp.c_fc.weight.copy_(w.T)
            elif branch == "k":
                mlp.gate_proj.weight.copy_(w)
            else:
                mlp.up_proj.weight.copy_(w)

    # --------------------------------------------------------------- forward
    def block(self, layer: int) -> torch.nn.Module:
        return self.layers[layer]

    def mlp_input_module(self, layer: int) -> torch.nn.Module:
        """The module whose *input* is ``x^{l-mid}`` (post-attention residual stream)."""
        block = self.layers[layer]
        if hasattr(block, "ln_2"):
            return block.ln_2
        if hasattr(block, "post_attention_layernorm"):
            return block.post_attention_layernorm
        raise AttributeError("Could not locate the pre-MLP layernorm")

    def mlp_module(self, layer: int) -> torch.nn.Module:
        return self.layers[layer].mlp

    def apply_unembedding(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logit-lens: final layernorm (if any) followed by the unembedding matrix."""
        h = hidden
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.unembedding is not None:
            return self.unembedding(h)
        return h @ self.embedding.weight.T


@dataclass
class ActivationRecord:
    """Container for the activation statistics collected by ``ActivationCollector``."""

    # layer -> Tensor[d_mlp] mean activation over (prompts, timesteps)
    mean_activations: Dict[int, torch.Tensor]
    # layer -> Tensor[d_mlp] fraction of timesteps with activation > 0
    act_fraction: Dict[int, torch.Tensor]


class ActivationCollector:
    """Collects per-neuron MLP activations ``m_i^l = sigma(h^{l} . k_i^l)``.

    The hook is attached to the MLP module itself, so its input is the true
    input of the key matrix (i.e. after the pre-MLP layernorm, which the paper
    omits from its notation).  When ``record_residual`` is set, a second hook on
    the pre-MLP layernorm captures ``x^{l-mid}``, the residual stream after the
    attention block.
    """

    def __init__(self, internals: TransformerInternals, layers: Optional[List[int]] = None,
                 record_residual: bool = False):
        self.internals = internals
        self.layers = layers if layers is not None else list(range(internals.n_layers))
        self.record_residual = record_residual
        self._handles = []
        self._sum: Dict[int, torch.Tensor] = {}
        self._pos: Dict[int, torch.Tensor] = {}
        self._count: Dict[int, int] = {}
        self._residuals: Dict[int, List[torch.Tensor]] = {}

    # ------------------------------------------------------------------ hooks
    def _make_pre_hook(self, layer: int):
        def hook(module, inputs):
            h = inputs[0].detach()
            act = self._activate(h, layer)
            self._accumulate(layer, act)
        return hook

    def _make_residual_hook(self, layer: int):
        def hook(module, inputs):
            self._residuals.setdefault(layer, []).append(
                inputs[0].detach().to(torch.float32).cpu())
        return hook

    def _activate(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """``sigma(h . W_K^T + b)`` for the (already normalised) MLP input ``h``."""
        internals = self.internals
        w_k = internals.key_weight(layer, "k").to(h.dtype)
        bias = internals.key_bias(layer, "k")
        pre = h @ w_k.T
        if bias is not None:
            pre = pre + bias.to(h.dtype)
        return self._nonlinearity(pre, h)

    def _nonlinearity(self, pre: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.internals.arch == "mlp":
            return torch.nn.functional.gelu(pre)
        return torch.nn.functional.silu(pre)

    def _accumulate(self, layer: int, act: torch.Tensor) -> None:
        flat = act.reshape(-1, act.shape[-1]).to(torch.float32)
        s = flat.sum(dim=0)
        p = (flat > 0).sum(dim=0)
        n = flat.shape[0]
        if layer in self._sum:
            self._sum[layer] += s
            self._pos[layer] += p
            self._count[layer] += n
        else:
            self._sum[layer] = s.clone()
            self._pos[layer] = p.clone()
            self._count[layer] = n

    # ------------------------------------------------------------------- api
    def __enter__(self):
        for l in self.layers:
            mlp = self.internals.mlp_module(l)
            self._handles.append(mlp.register_forward_pre_hook(self._make_pre_hook(l)))
            if self.record_residual:
                norm = self.internals.mlp_input_module(l)
                self._handles.append(norm.register_forward_pre_hook(self._make_residual_hook(l)))
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def result(self) -> ActivationRecord:
        mean = {l: (self._sum[l] / max(self._count[l], 1)) for l in self._sum}
        frac = {l: (self._pos[l] / max(self._count[l], 1)) for l in self._sum}
        return ActivationRecord(mean_activations=mean, act_fraction=frac)

    def residuals(self) -> Dict[int, torch.Tensor]:
        return {l: torch.cat(v, dim=0) for l, v in self._residuals.items()}


class ResidualShiftHook:
    """Subtract (or add) a fixed vector from ``x^{l-mid}`` during the forward pass.

    Section 3.3 subtracts a toxic vector from the residual stream of the last
    layer. We implement it as a pre-hook on the pre-MLP layernorm of layer ``l``
    so that the intervention is applied to ``x^{l-mid}`` itself (and therefore
    also propagates to the MLP of that layer and beyond), matching the paper's
    ``x^{L-1} = x^{L-1} - alpha * W``.

    A multiplicative variant (``mode="scale_key"``) is used by Section 6, where
    the *key vectors* are scaled by a factor and the residual stream is not
    modified directly.
    """

    def __init__(self, internals: TransformerInternals, layer: int, vector: torch.Tensor,
                 alpha: float = 1.0, mode: str = "subtract"):
        self.internals = internals
        self.layer = layer
        self.vector = vector
        self.alpha = alpha
        self.mode = mode
        self._handle = None

    def __enter__(self):
        vec = self.vector.to(torch.float32)

        def hook(module, inputs):
            x = inputs[0]
            v = vec.to(device=x.device, dtype=x.dtype)
            delta = self.alpha * v
            if self.mode == "subtract":
                return (x - delta,)
            if self.mode == "add":
                return (x + delta,)
            raise ValueError(self.mode)

        self._handle = self.internals.mlp_input_module(self.layer).register_forward_pre_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def mlp_activation(internals: TransformerInternals, layer: int, x_mid: torch.Tensor,
                   indices: Optional[List[int]] = None) -> torch.Tensor:
    """Compute ``m_i^l`` for the given value-vector indices of ``layer``."""
    h = internals.mlp_input_module(layer)(x_mid)
    w_k = internals.key_weight(layer, "k").to(h.dtype)
    bias = internals.key_bias(layer, "k")
    if indices is not None:
        w_k = w_k[indices]
        bias = None if bias is None else bias[indices]
    pre = h @ w_k.T
    if bias is not None:
        pre = pre + bias.to(h.dtype)
    if internals.arch == "mlp":
        return torch.nn.functional.gelu(pre)
    return torch.nn.functional.silu(pre)
