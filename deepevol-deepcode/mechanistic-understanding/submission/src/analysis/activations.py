"""Mean-activation and MLP activation-region analysis (Section 5.2, Figure 2, Eq. 1).

The paper measures how aligned (DPO) models avoid the regions of hidden space that
previously triggered toxic value vectors.  Two quantities are used:

* the mean activation of a value vector (Section 5.2)::

      m_i = mean_{tokens} sigma(x^l . MLP.k_i^l)

  obtained by generating ``n_tokens`` (20 by default) for the 1,199
  RealToxicityPrompts challenge prompts for both GPT2 and GPT2_DPO and averaging
  the MLP non-linearity output over time;  and

* the activation *region* of a key vector (Eq. 1)::

      gamma(k_i^l) := { g in R^d | sigma(k_i^l . g) > 0 }

  which is the set of residual-stream directions for which the corresponding value
  vector is switched on.  The un-alignment experiment (Section 6, Table 4) scales
  key vectors by 10x, i.e. it enlarges ``gamma(MLP.k_Toxic)``.

``sigma`` is the model's own MLP non-linearity (GeLU for GPT2; the module also
exposes the exact and tanh-approximate GeLU used for offline/analytic region
checks).  All functions operate on the key vector convention produced by
:mod:`src.model_utils` (``W_K`` has shape ``[d_mlp, d_model]`` and row ``i`` is
``MLP.k_i^l``).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.model_utils import (
    capture_residual_streams,
    get_mlp_matrices,
    resolve_device,
)

__all__ = [
    "DEFAULT_N_TOKENS",
    "DEFAULT_TOP_K_MEMBERS",
    "DEFAULT_LAYER",
    "DEFAULT_MLP_IDX",
    "TARGET_VECTOR",
    "MeanActivationResult",
    "LayerVectorStats",
    "collect_mean_activations",
    "collect_mean_activations_by_layer",
    "mean_activations_from_result",
    "compare_mean_activations",
    "activation_region",
    "activation_regions",
    "activation_region_fraction",
    "activation_region_batch",
    "activation_strength",
    "gelu_exact",
    "gelu_tanh",
    "gelu",
    "gelu_approx_error",
    "activation_values",
    "save_activation_result",
    "load_activation_result",
    "cache_path_for",
    "plot_mean_activations",
    "plot_activation_histogram",
]


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TOKENS: int = 20            # Section 5.2: "we generate 20 tokens"
DEFAULT_TOP_K_MEMBERS: int = 5        # Figure 2 shows 5 examples of top vectors
DEFAULT_LAYER: int = 19               # Layer 19 is used throughout Sections 5.1-5.2
DEFAULT_MLP_IDX: int = 770            # MLP.v_770^19, one of the most toxic vectors
TARGET_VECTOR: Tuple[int, int] = (DEFAULT_LAYER, DEFAULT_MLP_IDX)

_GELU_C = 0.044715
_GELU_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------------------
# GeLU variants (used for the MLP activation function sigma)
# ---------------------------------------------------------------------------
def gelu_exact(x):
    """Exact GeLU, ``x * 0.5 * (1 + erf(x / sqrt(2)))``."""
    if isinstance(x, torch.Tensor):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_tanh(x):
    """Tanh approximation of GeLU (Hendrycks & Gimpel, 2016)."""
    if isinstance(x, torch.Tensor):
        return 0.5 * x * (1.0 + torch.tanh(_GELU_SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)))
    return 0.5 * x * (1.0 + math.tanh(_GELU_SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)))


def gelu(x, approximate: bool = False):
    """GeLU with an explicit approximation switch (default: exact erf)."""
    return gelu_tanh(x) if approximate else gelu_exact(x)


def gelu_approx_error(x: float) -> float:
    """Absolute difference between the exact and tanh-approximate GeLU."""
    return abs(float(gelu_exact(x)) - float(gelu_tanh(x)))


# ---------------------------------------------------------------------------
# Activation regions: gamma(k_i^l) = { g | sigma(k_i^l . g) > 0 }   (Eq. 1)
# ---------------------------------------------------------------------------
def activation_values(key_vector: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Raw pre-activations ``k_i^l . g`` for a stack of residual states ``g``.

    Shapes: ``key_vector`` ``[d]``, ``states`` ``[..., d]`` -> ``[...]``.
    """
    key_vector = torch.as_tensor(key_vector)
    states = torch.as_tensor(states)
    return states @ key_vector


def activation_strength(key_vector, states, approximate: bool = False):
    """``sigma(k_i^l . g)`` -- how strongly the value vector is scaled."""
    return gelu(activation_values(key_vector, states), approximate=approximate)


def activation_region(key_vector, g, approximate: bool = False) -> bool:
    """Membership test for Eq. 1 on a single residual state ``g``.

    With GeLU, ``sigma(z) > 0`` holds exactly when the pre-activation ``z`` is
    positive (``sigma(0) == 0`` and ``sigma(z) > 0`` for ``z > 0``); the direct
    non-linearity is evaluated and compared against zero so the helper stays valid
    for any monotone-through-zero activation.
    """
    strength = activation_strength(key_vector, g, approximate=approximate)
    return bool(torch.as_tensor(strength).reshape(-1)[0] > 0)


def activation_regions(key_vector, states, approximate: bool = False):
    """Boolean membership mask of ``gamma(k_i^l)`` for a batch of states."""
    strength = activation_strength(key_vector, states, approximate=approximate)
    return torch.as_tensor(strength) > 0


def activation_region_batch(key_vectors, states, approximate: bool = False):
    """Membership masks for a stack of key vectors.

    ``key_vectors`` ``[n, d]``, ``states`` ``[..., d]`` -> bool ``[..., n]``.
    """
    key_vectors = torch.as_tensor(key_vectors)
    states = torch.as_tensor(states)
    pre = states.reshape(-1, states.shape[-1]) @ key_vectors.t()
    return (gelu(pre, approximate=approximate) > 0).reshape(*states.shape[:-1], key_vectors.shape[0])


def activation_region_fraction(key_vector, states, approximate: bool = False) -> float:
    """Fraction of residual states that fall inside ``gamma(k_i^l)``."""
    mask = activation_regions(key_vector, states, approximate=approximate)
    return float(mask.to(torch.float32).mean().item())


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------
@dataclass
class MeanActivationResult:
    """Mean MLP activations ``m_i`` for a set of (toxicity-ranked) vectors.

    Attributes
    ----------
    indices:
        ``(layer, idx)`` identifiers aligned with the arrays below.
    mean / std / positive_fraction:
        Arrays of shape ``[n_vectors]``: mean activation, standard deviation and
        fraction of tokens for which ``sigma(.) > 0``.
    count:
        Total number of (prompt, token) positions averaged over.
    n_prompts / n_tokens:
        Bookkeeping describing the generation settings.
    values:
        Optional nested view ``{(layer, idx): float}``.
    per_layer:
        Optional ``{layer: np.ndarray[d_mlp]}`` of mean activations of *all* value
        vectors at that layer (Figure 5 orange overlay).
    meta:
        Free-form metadata (model name, token count, position policy, ...).
    """

    indices: List[Tuple[int, int]]
    mean: np.ndarray
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    positive_fraction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    count: int = 0
    n_prompts: int = 0
    n_tokens: int = 0
    values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    per_layer: Dict[int, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        if self.std is None or np.asarray(self.std).size == 0:
            self.std = np.zeros_like(self.mean)
        else:
            self.std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        if self.positive_fraction is None or np.asarray(self.positive_fraction).size == 0:
            self.positive_fraction = np.zeros_like(self.mean)
        else:
            self.positive_fraction = np.asarray(self.positive_fraction, dtype=np.float64).reshape(-1)
        if not self.values:
            self.values = {idx: float(v) for idx, v in zip(self.indices, self.mean)}

    # -- accessors ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __contains__(self, idx: Tuple[int, int]) -> bool:
        return tuple(idx) in self.values

    def mean_of(self, layer: int, idx: int) -> float:
        """Mean activation of vector ``(layer, idx)``."""
        if (int(layer), int(idx)) not in self.values:
            raise KeyError(f"vector ({layer}, {idx}) not present in this result")
        return float(self.values[(int(layer), int(idx))])

    def vector(self, layer: int, idx: int) -> MeanActivationResult:
        """Sub-result containing a single vector (useful for reporting)."""
        layer, idx = int(layer), int(idx)
        pos = self.indices.index((layer, idx))
        return MeanActivationResult(
            indices=[(layer, idx)],
            mean=self.mean[pos:pos + 1],
            std=self.std[pos:pos + 1],
            positive_fraction=self.positive_fraction[pos:pos + 1],
            count=self.count,
            n_prompts=self.n_prompts,
            n_tokens=self.n_tokens,
            per_layer=self.per_layer,
            meta=dict(self.meta),
        )

    def by_layer(self) -> Dict[int, List[Tuple[int, int]]]:
        out: Dict[int, List[Tuple[int, int]]] = {}
        for layer, idx in self.indices:
            out.setdefault(int(layer), []).append((int(layer), int(idx)))
        return out

    def top(self, k: int = DEFAULT_TOP_K_MEMBERS, largest: bool = True) -> List[Tuple[Tuple[int, int], float]]:
        """``k`` vectors with the largest (or smallest) mean activation."""
        order = np.argsort(self.mean)
        if largest:
            order = order[::-1]
        order = order[: max(0, int(k))]
        return [(tuple(self.indices[i]), float(self.mean[i])) for i in order]

    def layer_means(self) -> Dict[int, float]:
        """Mean activation aggregated per layer."""
        acc: Dict[int, List[float]] = {}
        for (layer, _), value in self.values.items():
            acc.setdefault(int(layer), []).append(float(value))
        return {layer: float(np.mean(values)) for layer, values in acc.items()}

    def delta(self, other: "MeanActivationResult") -> "MeanActivationResult":
        """``self - other`` for vectors present in both results (DPO minus GPT2)."""
        shared = [idx for idx in self.indices if idx in other.values]
        mean = np.array([self.values[i] for i in shared], dtype=np.float64)
        other_mean = np.array([other.values[i] for i in shared], dtype=np.float64)
        pos = {idx: p for p, idx in enumerate(self.indices)}
        return MeanActivationResult(
            indices=list(shared),
            mean=mean - other_mean,
            std=self.std[[pos[i] for i in shared]] if self.std.size else None,
            positive_fraction=self.positive_fraction[[pos[i] for i in shared]]
            if self.positive_fraction.size
            else None,
            count=self.count,
            n_prompts=self.n_prompts,
            n_tokens=self.n_tokens,
            meta={"kind": "delta", "reference": other.meta.get("model", "gpt2")},
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "indices": [[int(l), int(i)] for l, i in self.indices],
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "positive_fraction": self.positive_fraction.tolist(),
            "values": {f"{int(l)}:{int(i)}": float(v) for (l, i), v in self.values.items()},
            "count": int(self.count),
            "n_prompts": int(self.n_prompts),
            "n_tokens": int(self.n_tokens),
            "per_layer": {str(int(l)): np.asarray(v).tolist() for l, v in self.per_layer.items()},
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "MeanActivationResult":
        indices = [(int(l), int(i)) for l, i in d.get("indices", [])]  # type: ignore[union-attr]
        return cls(
            indices=indices,
            mean=np.asarray(d.get("mean", []), dtype=np.float64),
            std=np.asarray(d.get("std", []), dtype=np.float64),
            positive_fraction=np.asarray(d.get("positive_fraction", []), dtype=np.float64),
            count=int(d.get("count", 0)),  # type: ignore[arg-type]
            n_prompts=int(d.get("n_prompts", 0)),  # type: ignore[arg-type]
            n_tokens=int(d.get("n_tokens", 0)),  # type: ignore[arg-type]
            per_layer={int(l): np.asarray(v, dtype=np.float64)
                       for l, v in dict(d.get("per_layer", {})).items()},  # type: ignore[union-attr]
            meta=dict(d.get("meta", {})),  # type: ignore[arg-type]
        )


@dataclass
class LayerVectorStats:
    """Per-layer statistics over *all* value vectors (Figure 5 overlays)."""

    layer: int
    mean: np.ndarray          # [d_mlp] mean activation of every value vector
    positive_fraction: np.ndarray  # [d_mlp]
    count: int = 0

    def histogram(self, bins: int = 50, value_range: Optional[Tuple[float, float]] = None):
        """Percentage of value vectors per bin (the orange areas of Figure 5)."""
        counts, edges = np.histogram(self.mean, bins=bins, range=value_range)
        total = max(1, counts.sum())
        return counts.astype(np.float64) / total * 100.0, edges

    def top(self, k: int = DEFAULT_TOP_K_MEMBERS, largest: bool = True):
        order = np.argsort(self.mean)
        if largest:
            order = order[::-1]
        order = order[: max(0, int(k))]
        return [(int(i), float(self.mean[i])) for i in order]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_device(device) -> torch.device:
    return resolve_device(device) if device is None else torch.device(device)


def _group_vectors(vectors: Mapping[Tuple[int, int], torch.Tensor]) -> Dict[int, List[Tuple[int, int]]]:
    grouped: Dict[int, List[Tuple[int, int]]] = {}
    for (layer, idx) in vectors:
        grouped.setdefault(int(layer), []).append((int(layer), int(idx)))
    for layer in grouped:
        grouped[layer] = sorted(grouped[layer], key=lambda t: t[1])
    return grouped


def _tokens_for_batch(tokenizer, texts: Sequence[str], device, max_length: int = 64, padding_side: str = "left"):
    try:
        tokenizer.padding_side = padding_side
    except Exception:  # pragma: no cover - tokenizer without padding_side
        pass
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}


def _position_mask(attention_mask: torch.Tensor, n_tokens: int, n_prompts: int) -> torch.Tensor:
    """Select the last ``n_tokens`` positions of an (optionally left-padded) batch.

    The residual stream at the final prompt token is the state whose unembedding
    produces the first generated token (Figure 1), so the last ``n_tokens``
    positions correspond to the first ``n_tokens`` continuation tokens.
    """
    mask = attention_mask.to(torch.bool)
    if n_tokens and n_tokens > 0:
        lengths = mask.sum(dim=1)
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        start = (lengths - int(n_tokens)).clamp(min=0).unsqueeze(1)
        mask = mask & (positions >= start)
    assert mask.shape[0] == n_prompts
    return mask


def cache_path_for(kind: str, name: str, cache_dir: Optional[str] = None) -> str:
    """Default artifact path for cached activation analyses."""
    root = cache_dir or os.path.join("artifacts", "cache")
    return os.path.join(root, f"{kind}_{name}.npz")


# ---------------------------------------------------------------------------
# core: mean activations over a set of value vectors
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_mean_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    vectors: Mapping[Tuple[int, int], torch.Tensor],
    n_tokens: int = DEFAULT_N_TOKENS,
    batch_size: int = 16,
    n_prompts: Optional[int] = None,
    device=None,
    positions: str = "first",
    max_length: int = 64,
    by_layer: bool = False,
    approximate: bool = False,
    model_name: Optional[str] = None,
    verbose: bool = False,
) -> MeanActivationResult:
    """Measure ``m_i = sigma(x^l . MLP.k_i^l)`` for the given key vectors.

    Parameters
    ----------
    model, tokenizer:
        A (frozen) causal LM; the residual stream is read *after attention and
        before the MLP* (``l-mid``) for every requested layer.
    prompts:
        Prompts to run; the paper uses the 1,199 RealToxicityPrompts challenge
        prompts.
    vectors:
        ``{(layer, idx): key_vector}`` where ``key_vector`` has shape ``[d_model]``.
        These are the MLP key vectors of the toxic value vectors (top members of
        ``MLP.v_Toxic``), or any other direction of interest.
    n_tokens:
        Number of positions to average over per prompt (20 in Section 5.2; ``0``
        averages over the whole sequence including the prompt itself).
    positions:
        ``"first"`` averages the first ``n_tokens`` continuation-token positions
        (equivalent to the last ``n_tokens`` context positions), ``"all"``
        averages every unmasked position.
    by_layer:
        Also compute the mean activation of *all* ``d_mlp`` value vectors of each
        involved layer (consumed by the Figure 5 orange overlay).

    Returns
    -------
    MeanActivationResult with per-vector mean/std/positive-fraction, plus optional
    ``per_layer`` arrays.
    """
    if not vectors:
        raise ValueError("`vectors` must contain at least one (layer, idx) -> key vector")
    device = _as_device(device)
    model.to(device)
    model.eval()

    grouped = _group_vectors(vectors)
    layers = sorted(grouped)
    texts = list(prompts)[: n_prompts] if n_prompts else list(prompts)
    n_prompt_total = len(texts)

    ids: List[Tuple[int, int]] = []
    for layer in layers:
        ids.extend(grouped[layer])
    key_stack = {
        layer: torch.stack([vectors[i].to(device) for i in grouped[layer]], dim=0)  # [n_l, d]
        for layer in layers
    }

    d_mlp_all: Dict[int, int] = {}
    full_keys: Dict[int, torch.Tensor] = {}
    if by_layer:
        for layer in layers:
            W_K, W_V = get_mlp_matrices(model, layer)
            full_keys[layer] = W_K.to(device).detach()
            d_mlp_all[layer] = int(W_K.shape[0])

    pos = {idx: p for p, idx in enumerate(ids)}
    sums = np.zeros(len(ids), dtype=np.float64)
    sumsq = np.zeros(len(ids), dtype=np.float64)
    n_positive = np.zeros(len(ids), dtype=np.float64)
    count = 0
    layer_sum = {layer: np.zeros(d_mlp_all.get(layer, 0) or 0, dtype=np.float64) for layer in layers}
    layer_positive = {layer: np.zeros(d_mlp_all.get(layer, 0) or 0, dtype=np.float64) for layer in layers}

    for start in range(0, n_prompt_total, batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = _tokens_for_batch(tokenizer, batch_texts, device, max_length=max_length)
        attn = enc.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(enc["input_ids"])
        if positions == "all":
            mask = attn.to(torch.bool)
        else:
            mask = _position_mask(attn, n_tokens, len(batch_texts))

        with capture_residual_streams(model, capture_mlp_act=False) as cap:
            model(**enc)
            for layer in layers:
                x_mid = cap.get_mid(layer)
                if x_mid is None:
                    raise RuntimeError(f"no residual stream captured for layer {layer}")
                x_mid = x_mid.to(device)
                flat = x_mid.reshape(-1, x_mid.shape[-1])
                flat_mask = mask.reshape(-1)
                states = flat[flat_mask]                     # [n_states, d]
                if states.numel() == 0:
                    continue
                acts = gelu(states @ key_stack[layer].t(), approximate=approximate)  # [n_states, n_l]
                acts_np = acts.to(torch.float64).cpu().numpy()
                block_positions = [pos[i] for i in grouped[layer]]
                sums[block_positions] += acts_np.sum(axis=0)
                sumsq[block_positions] += (acts_np ** 2).sum(axis=0)
                n_positive[block_positions] += (acts_np > 0).sum(axis=0)
                count += int(acts_np.shape[0])

                if by_layer:
                    all_acts = gelu(states @ full_keys[layer].t(), approximate=approximate)
                    all_np = all_acts.to(torch.float64).cpu().numpy()
                    layer_sum[layer] += all_np.sum(axis=0)
                    layer_positive[layer] += (all_np > 0).sum(axis=0)

        if verbose:
            done = min(start + batch_size, n_prompt_total)
            print(f"[activations] {done}/{n_prompt_total} prompts, {count} token positions")

    count = max(1, count)
    mean = sums / count
    var = np.clip(sumsq / count - mean ** 2, a_min=0.0, a_max=None)
    result = MeanActivationResult(
        indices=ids,
        mean=mean,
        std=np.sqrt(var),
        positive_fraction=n_positive / count,
        count=count,
        n_prompts=n_prompt_total,
        n_tokens=int(n_tokens),
        per_layer={
            layer: (layer_sum[layer] / count) for layer in layers if by_layer
        },
        meta={
            "model": model_name or getattr(getattr(model, "config", None), "_name_or_path", "model"),
            "positions": positions,
            "n_tokens": int(n_tokens),
            "approximate_gelu": bool(approximate),
        },
    )
    if by_layer:
        result.meta["layer_positive_fraction"] = {
            layer: (layer_positive[layer] / count).tolist() for layer in layers
        }
    return result


@torch.no_grad()
def collect_mean_activations_by_layer(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    n_tokens: int = DEFAULT_N_TOKENS,
    batch_size: int = 16,
    n_prompts: Optional[int] = None,
    device=None,
    positions: str = "first",
    max_length: int = 64,
    approximate: bool = False,
    model_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict[int, LayerVectorStats]:
    """Mean activation of *every* value vector in each requested layer.

    This produces the orange areas of Figure 5, which show the distribution of the
    mean activation over all value vectors of a layer during a forward pass of the
    1,199 RealToxicityPrompts prompts.
    """
    device = _as_device(device)
    model.to(device)
    model.eval()
    texts = list(prompts)[: n_prompts] if n_prompts else list(prompts)
    layers = [int(l) for l in layers]

    full_keys = {}
    for layer in layers:
        W_K, _ = get_mlp_matrices(model, layer)
        full_keys[layer] = W_K.to(device).detach()

    sums = {layer: np.zeros(full_keys[layer].shape[0], dtype=np.float64) for layer in layers}
    positive = {layer: np.zeros(full_keys[layer].shape[0], dtype=np.float64) for layer in layers}
    count = 0

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = _tokens_for_batch(tokenizer, batch_texts, device, max_length=max_length)
        attn = enc.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(enc["input_ids"])
        mask = attn.to(torch.bool) if positions == "all" else _position_mask(attn, n_tokens, len(batch_texts))

        with capture_residual_streams(model, capture_mlp_act=False) as cap:
            model(**enc)
            for layer in layers:
                x_mid = cap.get_mid(layer)
                flat = x_mid.to(device).reshape(-1, x_mid.shape[-1])
                states = flat[mask.reshape(-1)]
                if states.numel() == 0:
                    continue
                acts = gelu(states @ full_keys[layer].t(), approximate=approximate)
                acts_np = acts.to(torch.float64).cpu().numpy()
                sums[layer] += acts_np.sum(axis=0)
                positive[layer] += (acts_np > 0).sum(axis=0)
        count += int(mask.sum().item())
        if verbose:
            print(f"[activations] layers-by-vector {min(start + batch_size, len(texts))}/{len(texts)}")

    count = max(1, count)
    return {
        layer: LayerVectorStats(
            layer=layer,
            mean=sums[layer] / count,
            positive_fraction=positive[layer] / count,
            count=count,
        )
        for layer in layers
    }


def mean_activations_from_result(result: MeanActivationResult,
                                layer: Optional[int] = None,
                                indices: Optional[Iterable[Tuple[int, int]]] = None) -> Dict[Tuple[int, int], float]:
    """Filter a result down to a layer / explicit ``(layer, idx)`` subset."""
    out: Dict[Tuple[int, int], float] = {}
    for idx, value in result.values.items():
        if layer is not None and idx[0] != int(layer):
            continue
        if indices is not None and idx not in {tuple(i) for i in indices}:
            continue
        out[idx] = float(value)
    return out


def compare_mean_activations(
    result_before: MeanActivationResult,
    result_after: MeanActivationResult,
    layer: Optional[int] = None,
    k: Optional[int] = None,
) -> Dict[str, object]:
    """Side-by-side comparison of GPT2 vs GPT2_DPO mean activations (Figure 2).

    Returns a dict with the shared ``(layer, idx)`` keys and the before/after/delta
    mean activations, optionally restricted to one layer and/or the ``k`` vectors
    with the largest pre-DPO activation.
    """
    shared = [idx for idx in result_before.indices if idx in result_after.values]
    if layer is not None:
        shared = [idx for idx in shared if idx[0] == int(layer)]
    shared.sort(key=lambda i: result_before.values[i], reverse=True)
    if k is not None:
        shared = shared[: int(k)]
    before = np.array([result_before.values[i] for i in shared], dtype=np.float64)
    after = np.array([result_after.values[i] for i in shared], dtype=np.float64)
    return {
        "indices": shared,
        "before": before,
        "after": after,
        "delta": after - before,
        "relative_drop": np.divide(before - after, np.abs(before),
                                   out=np.zeros_like(before), where=np.abs(before) > 1e-12),
        "mean_drop": float(np.mean(before - after)) if len(shared) else float("nan"),
        "before_name": result_before.meta.get("model", "GPT2"),
        "after_name": result_after.meta.get("model", "GPT2_DPO"),
    }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_activation_result(path: str, result: MeanActivationResult) -> str:
    """Save a result as ``.json`` (with a sibling ``.npz`` for large per-layer arrays)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = result.to_dict()
    if result.per_layer:
        payload["per_layer_keys"] = sorted(str(int(l)) for l in result.per_layer)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_activation_result(path: str) -> MeanActivationResult:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return MeanActivationResult.from_dict(payload)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_mean_activations(results: Sequence[MeanActivationResult],
                          labels: Optional[Sequence[str]] = None,
                          indices: Optional[Sequence[Tuple[int, int]]] = None,
                          layer: Optional[int] = None,
                          k: Optional[int] = None,
                          title: str = "Mean MLP activations for MLP.v_Toxic vectors",
                          xlabel: str = "toxic vector",
                          ylabel: str = r"mean activation $m_i$",
                          out_path: Optional[str] = None,
                          figsize: Tuple[float, float] = (7.0, 4.0),
                          annotate: bool = True,
                          dpi: int = 150):
    """Grouped bar chart of ``m_i`` for GPT2 vs GPT2_DPO (Figure 2 style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("need at least one result to plot")
    labels = list(labels) if labels else [r.meta.get("model", f"run{i}") for i, r in enumerate(results)]
    shared = [idx for idx in results[0].indices if all(idx in r.values for r in results)]
    if layer is not None:
        shared = [idx for idx in shared if idx[0] == int(layer)]
    shared.sort(key=lambda i: results[0].values[i], reverse=True)
    if k is not None:
        shared = shared[: int(k)]
    if indices is not None:
        wanted = [tuple(i) for i in indices]
        shared = [idx for idx in wanted if idx in shared] or wanted

    n_vectors, n_runs = len(shared), len(results)
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / max(1, n_runs)
    x = np.arange(n_vectors)
    for r_i, (res, lab) in enumerate(zip(results, labels)):
        vals = [res.values.get(idx, np.nan) for idx in shared]
        ax.bar(x + r_i * width, vals, width=width, label=str(lab))
    ax.set_xticks(x + width * (n_runs - 1) / 2)
    ax.set_xticklabels([f"MLP.v$_{{{i}}}^{{{l}}}$" for l, i in shared], rotation=45, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if annotate:
        for r_i, res in enumerate(results):
            for x_i, idx in enumerate(shared):
                val = res.values.get(idx, np.nan)
                if np.isfinite(val):
                    ax.text(x_i + r_i * width, val, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=6)
    ax.legend()
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    return fig


def plot_activation_histogram(stats: Mapping[int, LayerVectorStats],
                              bins: int = 50,
                              value_range: Tuple[float, float] = (-5.0, 5.0),
                              title: str = "Distribution of mean activations over value vectors",
                              xlabel: str = r"mean activation $m_i$",
                              ylabel: str = "percentage of value vectors (%)",
                              out_path: Optional[str] = None,
                              figsize: Tuple[float, float] = (7.0, 4.0),
                              dpi: int = 150):
    """Line/area plot of the Figure 5 orange overlay (per-layer activation spread)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for layer in sorted(stats):
        pct, edges = stats[layer].histogram(bins=bins, value_range=value_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, pct, linewidth=1.0, label=f"layer {layer}")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    return fig


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print("activation region of a unit key vector along its own direction:",
          activation_region(torch.ones(4), torch.ones(4)))
    print("GeLU approximation error at 1.0:", gelu_approx_error(1.0))
    demo = MeanActivationResult(
        indices=[(19, 770), (19, 771), (12, 771)],
        mean=np.array([0.31, -0.02, 0.11]),
        std=np.array([0.4, 0.3, 0.2]),
        positive_fraction=np.array([0.4, 0.49, 0.45]),
        count=100,
    )
    print("top vectors:", demo.top(2))
    print("delta:", demo.delta(demo).mean)
