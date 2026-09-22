"""Section 5.1 / 5.2 residual-stream shift analysis (Figures 3, 4, 5 and Appendix C/D).

The paper's formalisation (verbatim from Section 5.2):

    delta_x^{l-mid} := x_DPO^{l-mid} - x_GPT2^{l-mid},   delta_x^{l-mid} in R^d

is viewed as *an offset that allows GPT2_DPO to bypass regions that previously
triggered toxic value vectors*.  The offset is decomposed over the parameter
deltas of the model, and for every earlier MLP block ``j < l`` and every value
vector ``i < d_mlp`` the paper measures

    (Eq. 2)   cos(delta_x^{l-mid}, delta_MLP.v_i^j)

Figure 5 (blue) reports, *per layer j*, the percentage of value vectors whose
``delta_MLP.v`` has each cosine similarity against ``delta_x^{19-mid}``; the
orange overlay reports, on the same abscissa, the percentage of value vectors
whose *mean activation* takes that value during a forward pass over the 1,199
RealToxicityPrompts peaks.  The paper's finding: ``delta_MLP.v`` shifts
*opposite* to ``delta_x`` (negative cosine similarity), yet because GeLU
activations are slightly negative for inactive neurons the contribution flips
and ends up *along* ``delta_x``.

Figure 4 projects layer-19 residual streams (GPT2 and GPT2_DPO, same 1,199
RealToxicityPrompts prompts) onto

    1) the mean residual-stream difference ``mean(delta_x^19)`` and
    2) the principal component of the residual streams,

colouring points by whether they activate ``MLP.v_770^19`` and drawing dotted
lines between the two residual streams of the same prompt.

All quantities here operate on ``l-mid`` states, i.e. the residual stream after
attention heads and *before* the MLP of layer ``l`` (Eq. 4 of Section 2).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.model_utils import (
    capture_residual_streams,
    get_mlp_matrices,
    resolve_device,
)
from src.analysis.activations import (
    activation_strength,
    activation_values,
    gelu,
)

# --------------------------------------------------------------------------- #
# Constants (paper defaults)
# --------------------------------------------------------------------------- #

#: Layer and MLP index of the paper's running example, MLP.v_770^19.
DEFAULT_LAYER = 19
DEFAULT_MLP_IDX = 770
TARGET_VECTOR: Tuple[int, int] = (DEFAULT_LAYER, DEFAULT_MLP_IDX)

#: Cosine-similarity support used by Figure 5's x-axis.
COSINE_RANGE: Tuple[float, float] = (-1.0, 1.0)
#: Figure 5 histogram resolution (also used for the activation overlay).
DEFAULT_BINS = 50
#: RealToxicityPrompts challenge subset size used in Section 5.2.
N_CHALLENGE_PROMPTS = 1199

ARTIFACT_DIR = os.path.join("artifacts", "analysis")


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class ResidualStreamCollection:
    """Mid-layer residual streams for one model (shape ``[n_prompts, d]``)."""

    mid: np.ndarray
    layer: int
    model_name: str = "model"
    positions: str = "first"
    n_tokens: int = 20
    texts: List[str] = field(default_factory=list)
    #: Optional per-layer activations of specific key vectors, shape [n_prompts, n_vectors].
    activations: Optional[np.ndarray] = None
    vector_indices: Optional[List[Tuple[int, int]]] = None

    @property
    def n_prompts(self) -> int:
        return int(self.mid.shape[0])

    @property
    def dim(self) -> int:
        return int(self.mid.shape[1])

    def mean(self) -> np.ndarray:
        return self.mid.mean(axis=0)

    def subset(self, n: int, seed: int = 0) -> "ResidualStreamCollection":
        if n is None or n >= self.n_prompts:
            return self
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(self.n_prompts, size=int(n), replace=False))
        return ResidualStreamCollection(
            mid=self.mid[idx],
            layer=self.layer,
            model_name=self.model_name,
            positions=self.positions,
            n_tokens=self.n_tokens,
            texts=[self.texts[i] for i in idx] if self.texts else [],
            activations=None if self.activations is None else self.activations[idx],
            vector_indices=self.vector_indices,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": int(self.layer),
            "model_name": self.model_name,
            "positions": self.positions,
            "n_tokens": int(self.n_tokens),
            "n_prompts": self.n_prompts,
            "dim": self.dim,
            "texts": list(self.texts[:50]),
        }


@dataclass
class ShiftResult:
    """``delta_x`` between two models, plus derived projection/statistics."""

    delta: np.ndarray                       # [n_prompts, d]
    layer: int
    before_name: str = "gpt2"
    after_name: str = "gpt2_dpo"
    before: Optional[ResidualStreamCollection] = None
    after: Optional[ResidualStreamCollection] = None
    #: Per-prompt activation of the target vector for both models.
    active_before: Optional[np.ndarray] = None
    active_after: Optional[np.ndarray] = None
    mean_activation_before: Optional[float] = None
    mean_activation_after: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_prompts(self) -> int:
        return int(self.delta.shape[0])

    @property
    def mean(self) -> np.ndarray:
        """``mean(delta_x^l)`` — the first Figure 4 projection axis."""
        return self.delta.mean(axis=0)

    @property
    def mean_norm(self) -> float:
        return float(np.linalg.norm(self.mean))

    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.delta, axis=1)

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": int(self.layer),
            "before": self.before_name,
            "after": self.after_name,
            "n_prompts": int(self.n_prompts),
            "mean_norm": self.mean_norm,
            "mean_activation_before": self.mean_activation_before,
            "mean_activation_after": self.mean_activation_after,
            **self.meta,
        }


@dataclass
class ProjectionResult:
    """Figure 4-style 2-D projection of residual streams."""

    coords_before: np.ndarray               # [n_prompts, 2]
    coords_after: np.ndarray                # [n_prompts, 2]
    axis_delta: np.ndarray                  # [d] mean delta_x
    axis_pc: np.ndarray                     # [d] principal component
    explained_variance_ratio: float = 0.0
    active_before: Optional[np.ndarray] = None
    active_after: Optional[np.ndarray] = None
    layer: int = DEFAULT_LAYER
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": int(self.layer),
            "n_prompts": int(self.coords_before.shape[0]),
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "mean_delta_norm": float(np.linalg.norm(self.axis_delta)),
            **self.meta,
        }


@dataclass
class ParameterShiftResult:
    """``delta_MLP.v`` cosine similarities against ``delta_x`` (Eq. 2)."""

    layer: int
    #: cos(delta_x^{l-mid}, delta_MLP.v_i^j) for every j < l and every i.
    cosine: Dict[int, np.ndarray] = field(default_factory=dict)
    #: Norm of each delta_MLP.v_i^j (secondary diagnostic).
    delta_norm: Dict[int, np.ndarray] = field(default_factory=dict)
    n_vectors: Dict[int, int] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def layers(self) -> List[int]:
        return sorted(self.cosine.keys())

    def stacked(self) -> np.ndarray:
        """All cosine similarities concatenated over layers."""
        if not self.cosine:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate([self.cosine[j] for j in self.layers])

    def fraction_negative(self) -> float:
        values = self.stacked()
        if values.size == 0:
            return float("nan")
        return float(np.mean(values < 0.0))

    def mean_per_layer(self) -> Dict[int, float]:
        return {j: float(np.mean(self.cosine[j])) for j in self.layers if self.cosine[j].size}

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": int(self.layer),
            "layers": self.layers,
            "n_vectors": {int(k): int(v) for k, v in self.n_vectors.items()},
            "mean_cosine_per_layer": self.mean_per_layer(),
            "fraction_negative": self.fraction_negative(),
            **self.meta,
        }


# --------------------------------------------------------------------------- #
# Residual-stream collection
# --------------------------------------------------------------------------- #

def _token_positions(attention_mask: torch.Tensor, positions: str, n_tokens: int) -> torch.Tensor:
    """Position mask ``[batch, seq]`` used to average residual streams."""
    mask = attention_mask.to(dtype=torch.bool)
    if positions in ("all", "mean"):
        return mask

    seq = mask.shape[1]
    keep = min(int(max(n_tokens, 1)), seq)
    pos = torch.zeros_like(mask)
    # Keep the last `keep` unmasked positions (the generated continuation).
    for b in range(mask.shape[0]):
        valid = torch.nonzero(mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        pos[b, valid[-keep:]] = True
    return pos


@torch.no_grad()
def collect_residual_streams(
    model,
    tokenizer,
    prompts: Sequence[str],
    layer: int = DEFAULT_LAYER,
    n_tokens: int = 20,
    batch_size: int = 8,
    positions: str = "first",
    device=None,
    model_name: str = "model",
    max_length: int = 96,
    vector_indices: Optional[Sequence[Tuple[int, int]]] = None,
    activation_fn=None,
    verbose: bool = False,
) -> ResidualStreamCollection:
    """Collect ``l-mid`` residual streams of ``layer`` for each prompt.

    ``l-mid`` denotes the residual stream after the attention heads of layer
    ``layer`` and *before* its MLP (the point at which ``delta_x`` is defined).
    Residual states are averaged over the first ``n_tokens`` positions of the
    prompt (``n_tokens=20`` matches the paper's greedy 20-token continuations);
    use ``positions="all"`` to average the full sequence instead.

    When ``vector_indices`` is given the corresponding key vectors are read from
    ``model`` and the mean activation of each value vector is recorded per
    prompt (shape ``[n_prompts, n_vectors]``), which is what Figure 4 colours by.
    """
    device = resolve_device(device)
    model.to(device)
    model.eval()
    activation_fn = activation_fn or gelu

    texts = [str(t) for t in prompts]
    states: List[np.ndarray] = []
    acts: List[np.ndarray] = []
    indices = list(vector_indices) if vector_indices is not None else None

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        pos_mask = _token_positions(attention_mask.cpu(), positions, n_tokens).to(device)

        with capture_residual_streams(model, capture_mlp_act=False) as cap:
            model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        mid = cap.get_mid(layer)
        if mid is None:
            raise ValueError(f"no captured residual stream for layer {layer}")

        mid = mid.to(torch.float32)                                   # [B, T, d]
        weights = pos_mask.to(mid.dtype).unsqueeze(-1)                 # [B, T, 1]
        denom = weights.sum(dim=1).clamp(min=1.0)                      # [B, 1]
        pooled = (mid * weights).sum(dim=1) / denom                    # [B, d]
        states.append(pooled.detach().cpu().numpy().astype(np.float64))

        if indices:
            keys = torch.stack(
                [
                    get_mlp_matrices(model, j)[0][i]
                    for (j, i) in indices
                ]
            ).to(device).to(torch.float32)                              # [n_vectors, d]
            pre = torch.einsum("btd,nd->btn", mid.to(keys.dtype), keys)  # [B, T, n_vectors]
            val = activation_values(keys, mid.reshape(-1, mid.shape[-1])).reshape(
                mid.shape[0], mid.shape[1], keys.shape[0]
            )
            acts.append(val.mean(dim=1).detach().cpu().numpy().astype(np.float64))

        if verbose:
            print(f"[residual] {start + len(chunk)}/{len(texts)} prompts pooled", flush=True)

    mid_arr = np.concatenate(states, axis=0) if states else np.zeros((0, 1))
    return ResidualStreamCollection(
        mid=mid_arr,
        layer=int(layer),
        model_name=model_name,
        positions=positions,
        n_tokens=int(n_tokens),
        texts=texts,
        activations=(np.concatenate(acts, axis=0) if acts else None),
        vector_indices=indices,
    )


# --------------------------------------------------------------------------- #
# delta_x
# --------------------------------------------------------------------------- #

def compute_residual_shift(
    before: ResidualStreamCollection,
    after: ResidualStreamCollection,
    layer: Optional[int] = None,
    target_index: Optional[Tuple[int, int]] = TARGET_VECTOR,
) -> ShiftResult:
    """``delta_x^{l-mid} = x_DPO^{l-mid} - x_GPT2^{l-mid}`` (Section 5.2)."""
    if before.mid.shape[0] != after.mid.shape[0]:
        raise ValueError("both collections must be evaluated on the same prompts")
    if before.mid.shape[1] != after.mid.shape[1]:
        raise ValueError("residual streams must share the same hidden dimension")

    delta = after.mid - before.mid

    active_before = active_after = None
    mean_before = mean_after = None
    if before.activations is not None and after.activations is not None and target_index is not None:
        if before.vector_indices and list(target_index) in [tuple(v) for v in before.vector_indices]:
            k = [tuple(v) for v in before.vector_indices].index(tuple(target_index))
            if k < before.activations.shape[1] and k < after.activations.shape[1]:
                active_before = before.activations[:, k]
                active_after = after.activations[:, k]
                mean_before = float(np.mean(active_before))
                mean_after = float(np.mean(active_after))

    return ShiftResult(
        delta=delta,
        layer=int(layer if layer is not None else before.layer),
        before_name=before.model_name,
        after_name=after.model_name,
        before=before,
        after=after,
        active_before=active_before,
        active_after=active_after,
        mean_activation_before=mean_before,
        mean_activation_after=mean_after,
    )


@torch.no_grad()
def compute_residual_shift_models(
    model_before,
    model_after,
    tokenizer,
    prompts: Sequence[str],
    layer: int = DEFAULT_LAYER,
    n_tokens: int = 20,
    batch_size: int = 8,
    positions: str = "first",
    device=None,
    target_index: Optional[Tuple[int, int]] = TARGET_VECTOR,
    before_name: str = "gpt2",
    after_name: str = "gpt2_dpo",
    verbose: bool = False,
) -> ShiftResult:
    """Collect both models' residual streams and return their difference."""
    vector_indices = [tuple(target_index)] if target_index is not None else None
    common = dict(
        tokenizer=tokenizer,
        prompts=prompts,
        layer=layer,
        n_tokens=n_tokens,
        batch_size=batch_size,
        positions=positions,
        device=device,
        vector_indices=vector_indices,
        verbose=verbose,
    )
    before = collect_residual_streams(model_before, model_name=before_name, **common)
    after = collect_residual_streams(model_after, model_name=after_name, **common)
    return compute_residual_shift(before, after, layer=layer, target_index=target_index)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between matching rows of ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, eps)


def shift_consistency(shift: ShiftResult) -> Dict[str, float]:
    """How *consistent* the per-prompt shift direction is (Figure 3/4 claim)."""
    mean = shift.mean
    cos = cosine_similarity_rows(shift.delta, np.broadcast_to(mean, shift.delta.shape))
    norms = shift.norm()
    return {
        "mean_norm": float(np.linalg.norm(mean)),
        "mean_cosine_to_mean": float(np.mean(cos)),
        "median_cosine_to_mean": float(np.median(cos)),
        "mean_norm_of_shift": float(np.mean(norms)),
        "max_norm_of_shift": float(np.max(norms)) if norms.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Projections (Figure 4)
# --------------------------------------------------------------------------- #

def _orthonormalize(basis: np.ndarray, exclude: Optional[np.ndarray] = None, eps: float = 1e-12) -> np.ndarray:
    """Return ``basis`` orthogonalised against ``exclude`` (unit norm)."""
    v = np.asarray(basis, dtype=np.float64).astype(np.float64)
    if exclude is not None and np.linalg.norm(exclude) > eps:
        e = np.asarray(exclude, dtype=np.float64)
        e = e / max(np.linalg.norm(e), eps)
        v = v - float(np.dot(v, e)) * e
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def projection_axes(
    shift: ShiftResult,
    include_before: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Figure 4 axes: (mean delta_x, principal component of the streams).

    The principal component is taken from the concatenated ``x_GPT2`` /
    ``x_DPO`` streams after removing the mean-delta direction (the paper
    projects onto two *dimensions*: the mean residual difference and the main
    principal component of the residual streams).

    Returns ``(axis_delta, axis_pc, explained_variance_ratio)``.
    """
    axis_delta = shift.mean
    if np.linalg.norm(axis_delta) < 1e-12:
        axis_delta = np.ones_like(axis_delta)

    parts = [shift.after.mid - shift.after.mid.mean(axis=0)] if shift.after is not None else [shift.delta]
    if include_before and shift.before is not None:
        parts.append(shift.before.mid - shift.before.mid.mean(axis=0))
    x = np.concatenate(parts, axis=0)

    # Principal component of the residual streams, computed on the residual
    # (i.e. after removing the mean-delta direction).
    x_orth = x - np.outer(x @ axis_delta, axis_delta) / max(float(axis_delta @ axis_delta), 1e-12)
    xc = x_orth - x_orth.mean(axis=0, keepdims=True)
    n_comp = min(2, xc.shape[0], xc.shape[1])
    if n_comp < 1:
        axis_pc = np.zeros_like(axis_delta)
        ratio = 0.0
    else:
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        axis_pc = vt[0]
        total = float(np.sum(s ** 2))
        ratio = float(s[0] ** 2 / total) if total > 0 else 0.0

    return axis_delta, axis_pc, ratio


def project_residual_streams(
    shift: ShiftResult,
    indices: Optional[Sequence[int]] = None,
    include_before: bool = False,
) -> ProjectionResult:
    """Project both models' layer-``l`` residual streams onto 2 axes (Figure 4)."""
    axis_delta, axis_pc, ratio = projection_axes(shift, include_before=include_before)
    unit_delta = _orthonormalize(axis_delta)
    unit_pc = _orthonormalize(axis_pc, exclude=unit_delta)

    after = shift.after.mid if shift.after is not None else None
    before = shift.before.mid if shift.before is not None else (after - shift.delta if after is not None else None)
    if after is None and before is None:  # pragma: no cover - defensive
        raise ValueError("ShiftResult carries neither stream")

    def coords(x: np.ndarray) -> np.ndarray:
        return np.stack([x @ axis_delta, x @ axis_pc], axis=1)

    coords_after = coords(after) if after is not None else np.zeros((0, 2))
    coords_before = coords(before) if before is not None else np.zeros((0, 2))

    if indices is not None:
        idx = np.asarray(list(indices), dtype=int)
        coords_after = coords_after[idx]
        coords_before = coords_before[idx]

    active_before = active_after = None
    if shift.active_before is not None and shift.active_after is not None:
        active_before, active_after = shift.active_before, shift.active_after
        if indices is not None:
            active_before, active_after = active_before[idx], active_after[idx]

    return ProjectionResult(
        coords_before=coords_before,
        coords_after=coords_after,
        axis_delta=axis_delta,
        axis_pc=axis_pc,
        explained_variance_ratio=ratio,
        active_before=active_before,
        active_after=active_after,
        layer=shift.layer,
        meta={"n_prompts": int(shift.n_prompts)},
    )


def mean_shift_vector(*shifts: ShiftResult) -> np.ndarray:
    """Average ``mean(delta_x^l)`` over several shifts (multi-layer Figure 3)."""
    means = [s.mean for s in shifts if s is not None]
    if not means:
        raise ValueError("at least one ShiftResult is required")
    return np.mean(np.stack(means, axis=0), axis=0)


# --------------------------------------------------------------------------- #
# Eq. 2: cosine similarity vs. delta_MLP.v
# --------------------------------------------------------------------------- #

def parameter_value_deltas(
    model_before,
    model_after,
    layers: Optional[Iterable[int]] = None,
) -> Dict[int, np.ndarray]:
    """``delta_MLP.v^j = MLP.v^j_DPO - MLP.v^j_GPT2`` for each layer ``j``.

    Returns a mapping ``layer -> [d_mlp, d_model]`` (rows are value vectors).
    """
    if layers is None:
        from src.model_utils import model_info

        layers = range(model_info(model_before).n_layers)
    out: Dict[int, np.ndarray] = {}
    for j in layers:
        w_before = get_mlp_matrices(model_before, int(j))[1]
        w_after = get_mlp_matrices(model_after, int(j))[1]
        out[int(j)] = (w_after.detach().to(torch.float64) - w_before.detach().to(torch.float64)).cpu().numpy()
    return out


def parameter_delta_cosine_against_shift(
    delta_x: np.ndarray,
    value_deltas: Mapping[int, np.ndarray],
    layer: int = DEFAULT_LAYER,
    eps: float = 1e-12,
    dtype=np.float64,
) -> ParameterShiftResult:
    """Eq. 2: ``cos(delta_x^{l-mid}, delta_MLP.v_i^j)`` for all ``j < l``, all ``i``.

    ``value_deltas[j]`` is ``delta_MLP.v^j`` with shape ``[d_mlp, d_model]``; its
    rows are the value vectors, so the per-vector shift is one row of the matrix.
    """
    dx = np.asarray(delta_x, dtype=np.float64).reshape(-1)
    dx_norm = float(np.linalg.norm(dx))
    result = ParameterShiftResult(layer=int(layer))

    for j in sorted(int(k) for k in value_deltas.keys()):
        if j >= layer:  # the paper only looks at preceding layers: j < l
            continue
        d = np.asarray(value_deltas[j], dtype=np.float64)
        norms = np.linalg.norm(d, axis=1)
        denom = np.maximum(norms * dx_norm, eps)
        cos = (d @ dx) / denom
        result.cosine[j] = cos
        result.delta_norm[j] = norms
        result.n_vectors[j] = int(d.shape[0])

    result.meta = {
        "delta_x_norm": dx_norm,
        "n_layers": len(result.cosine),
        "total_vectors": int(sum(result.n_vectors.values())),
    }
    return result


def per_layer_histograms(
    result: ParameterShiftResult,
    bins: int = DEFAULT_BINS,
    value_range: Tuple[float, float] = COSINE_RANGE,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Blue Figure 5 histograms: percentages of value vectors per layer (j < l)."""
    hist: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for j in result.layers:
        values = result.cosine[j]
        if values.size == 0:
            continue
        edges = np.linspace(value_range[0], value_range[1], int(bins) + 1)
        counts, _ = np.histogram(values, bins=edges)
        total = counts.sum()
        percent = counts.astype(np.float64) / total * 100.0 if total else counts.astype(np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist[j] = (percent, centers)
    return hist


def activation_histogram(
    activations,
    bins: int = DEFAULT_BINS,
    value_range: Tuple[float, float] = COSINE_RANGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orange Figure 5 overlay: percentage of value vectors per activation value.

    Uses the same abscissa as the cosine-similarity histogram (Figure 5 shares
    one x-axis for both distributions).
    """
    values = np.asarray(activations, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    edges = np.linspace(value_range[0], value_range[1], int(bins) + 1)
    if values.size == 0:
        return np.zeros(int(bins)), 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    percent = counts.astype(np.float64) / total * 100.0 if total else counts.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return percent, centers


def contribution_analysis(
    delta_x: np.ndarray,
    value_deltas: Mapping[int, np.ndarray],
    activations: Mapping[int, np.ndarray],
    layer: int = DEFAULT_LAYER,
    scale: float = 1e-3,
) -> Dict[int, Dict[str, float]]:
    """Why the antipodal shift still pushes *along* ``delta_x``.

    For each preceding layer the paper's argument is:

        contribution = sigma(x . k_i) * delta_MLP.v_i

    Since ``delta_MLP.v`` has negative cosine similarity with ``delta_x`` while
    the mean GeLU activation is *small and negative*, the product flips sign and
    contributes towards ``delta_x``.  We report the mean cosine of the shift
    (negative), the mean activation (negative), and the resulting mean
    projection of the accumulated contribution onto ``delta_x``.
    """
    dx = np.asarray(delta_x, dtype=np.float64).reshape(-1)
    dx_unit = dx / max(float(np.linalg.norm(dx)), 1e-12)
    out: Dict[int, Dict[str, float]] = {}

    for j in sorted(int(k) for k in value_deltas.keys()):
        if j >= layer:
            continue
        d = np.asarray(value_deltas[j], dtype=np.float64)
        norms = np.linalg.norm(d, axis=1)
        cos = (d @ dx_unit) / np.maximum(norms, 1e-12)

        act = None
        if j in activations:
            act = np.asarray(activations[j], dtype=np.float64).reshape(-1)
            if act.shape[0] != d.shape[0]:
                act = None
        if act is None:
            act = np.zeros(d.shape[0], dtype=np.float64)

        contribution = (act * norms) * cos  # projection onto unit delta_x per vector
        out[j] = {
            "mean_cosine": float(np.mean(cos)) if cos.size else float("nan"),
            "mean_activation": float(np.mean(act)) if act.size else float("nan"),
            "mean_norm": float(np.mean(norms)) if norms.size else float("nan"),
            "mean_projection": float(np.mean(contribution) * scale) if contribution.size else float("nan"),
            "sum_projection": float(np.sum(contribution) * scale) if contribution.size else float("nan"),
            "fraction_negative_cosine": float(np.mean(cos < 0)) if cos.size else float("nan"),
        }
    return out


def mean_activations_by_layer(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    n_tokens: int = 20,
    batch_size: int = 8,
    device=None,
    positions: str = "first",
    verbose: bool = False,
) -> Dict[int, np.ndarray]:
    """Mean GeLU activation of *every* value vector in each requested layer."""
    device = resolve_device(device)
    model.to(device)
    model.eval()

    layers = [int(l) for l in layers]
    texts = [str(t) for t in prompts]
    sums: Dict[int, np.ndarray] = {}
    counts = 0

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        pos_mask = _token_positions(attention_mask.cpu(), positions, n_tokens).to(device)

        with capture_residual_streams(model, capture_mlp_act=False) as cap:
            model(input_ids=input_ids, attention_mask=attention_mask)

        for l in layers:
            mid = cap.get_mid(l)
            if mid is None:
                continue
            mid = mid.to(torch.float32)
            keys = get_mlp_matrices(model, l)[0].to(device).to(torch.float32)   # [d_mlp, d]
            flat = mid.reshape(-1, mid.shape[-1])
            vals = activation_values(keys, flat)                                # [T*B, d_mlp]
            vals = vals.reshape(mid.shape[0], mid.shape[1], keys.shape[0])
            w = pos_mask.to(vals.dtype).unsqueeze(-1)
            denom = w.sum(dim=1).clamp(min=1.0)
            per_prompt = (vals * w).sum(dim=1) / denom                          # [B, d_mlp]
            total = per_prompt.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            sums[l] = sums.get(l, np.zeros_like(total)) + total
        counts += len(chunk)

        if verbose:
            print(f"[activations] {start + len(chunk)}/{len(texts)} prompts", flush=True)

    denom = max(counts, 1)
    return {l: v / denom for l, v in sums.items()}


def collect_activation_histogram(
    model,
    tokenizer,
    prompts: Sequence[str],
    layer: int = DEFAULT_LAYER,
    n_tokens: int = 20,
    batch_size: int = 8,
    device=None,
    positions: str = "first",
    verbose: bool = False,
) -> np.ndarray:
    """Per-value-vector mean activations in one layer (orange part of Figure 5)."""
    out = mean_activations_by_layer(
        model,
        tokenizer,
        prompts,
        layers=[int(layer)],
        n_tokens=n_tokens,
        batch_size=batch_size,
        device=device,
        positions=positions,
        verbose=verbose,
    )
    return out.get(int(layer), np.zeros(0, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_shift_result(path: str, shift: ShiftResult, save_arrays: bool = True) -> str:
    """Save a :class:`ShiftResult` (numpy ``.npz`` + JSON sidecar)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(shift.to_dict(), fh, indent=2)
    if save_arrays:
        arrays = {"delta": shift.delta, "mean": shift.mean}
        if shift.active_before is not None:
            arrays["active_before"] = shift.active_before
        if shift.active_after is not None:
            arrays["active_after"] = shift.active_after
        np.savez_compressed(path, **arrays)
    return meta_path


def save_parameter_shift(path: str, result: ParameterShiftResult) -> str:
    """Save an Eq. 2 cosine-similarity result."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    if result.cosine:
        np.savez_compressed(
            path,
            **{f"cos_{j}": result.cosine[j] for j in result.layers},
            **{f"norm_{j}": result.delta_norm[j] for j in result.layers if j in result.delta_norm},
        )
    return meta_path


def load_shift_result(path: str) -> Dict[str, object]:
    """Load a saved :class:`ShiftResult` (JSON metadata + optional arrays)."""
    meta_path = path if path.endswith(".json") else os.path.splitext(path)[0] + ".json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    npz_path = os.path.splitext(meta_path)[0] + ".npz"
    out: Dict[str, object] = {"meta": meta}
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            out.update({k: data[k] for k in data.files})
    return out


def default_paths(layer: int = DEFAULT_LAYER, prefix: str = "gpt2") -> Dict[str, str]:
    """Conventional artifact paths for the shift analysis."""
    base = os.path.join(ARTIFACT_DIR, f"{prefix}_layer{int(layer)}")
    return {
        "shift": f"{base}_shift.npz",
        "cosine": f"{base}_delta_cosine.npz",
        "figure_fig3": os.path.join(ARTIFACT_DIR, f"fig3_residual_shift_{prefix}.png"),
        "figure_fig4": os.path.join(ARTIFACT_DIR, f"fig4_projection_layer{int(layer)}_{prefix}.png"),
        "figure_fig5": os.path.join(ARTIFACT_DIR, f"fig5_delta_cosine_layer{int(layer)}_{prefix}.png"),
        "figure_fig7": os.path.join(ARTIFACT_DIR, f"fig7_shift_layers_{prefix}.png"),
    }


# --------------------------------------------------------------------------- #
# Plotting (Figures 3, 4, 5 and Appendix Figure 7)
# --------------------------------------------------------------------------- #

def plot_residual_shift(
    shift: ShiftResult,
    out_path: Optional[str] = None,
    layer: Optional[int] = None,
    max_lines: int = 200,
    seed: int = 0,
    figsize: Tuple[float, float] = (6.5, 4.0),
    title: Optional[str] = None,
):
    """Figure 3: per-prompt ``delta_x`` as an offset (PCA of the shift itself).

    Each point is one prompt's ``delta_x^{l-mid}`` projected onto its first two
    principal directions; the mean offset ``mean(delta_x)`` is drawn as an arrow,
    illustrating that GPT2_DPO applies a consistent offset to the residual
    stream, taking it out of the toxicity-eliciting regions ``gamma(MLP.k_Toxic)``.
    """
    from src.analysis.plots import make_figure, save_figure, BASE_COLOR, ALIGNED_COLOR

    layer = int(layer if layer is not None else shift.layer)
    x = shift.delta
    rng = np.random.RandomState(seed)
    idx = np.arange(x.shape[0])
    if idx.size > max_lines:
        idx = np.sort(rng.choice(idx, size=int(max_lines), replace=False))
    xs = x[idx]

    xc = xs - xs.mean(axis=0, keepdims=True)
    n_comp = min(2, xc.shape[0], xc.shape[1])
    if n_comp >= 2:
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        coords = xc @ vt[:2].T
        ratio = float(s[:2].sum() ** 0 / 1.0)
    elif n_comp == 1:
        vt = np.linalg.svd(xc, full_matrices=False)[2][:1]
        coords = np.concatenate([xc @ vt.T, np.zeros((xc.shape[0], 1))], axis=1)
    else:  # pragma: no cover - degenerate
        coords = np.zeros((xc.shape[0], 2))

    fig, ax = make_figure(figsize=figsize)
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.45, color=BASE_COLOR, label="per-prompt delta_x", zorder=2)

    mean_coord = np.zeros(2)
    mean_delta = shift.mean
    if n_comp >= 2:
        mean_coord = (mean_delta - xs.mean(axis=0)) @ np.linalg.svd(xc, full_matrices=False)[2][:2].T
    ax.annotate(
        "",
        xy=mean_coord,
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color=ALIGNED_COLOR, lw=2.0),
        zorder=4,
    )
    ax.scatter([0.0], [0.0], marker="x", s=45, color=ALIGNED_COLOR, zorder=5, label="origin")

    consistency = shift_consistency(shift)
    ax.set_title(title or f"Figure 3: delta_x^({layer}-mid) as an offset")
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend(loc="best", fontsize=8)
    fig.text(
        0.01,
        0.01,
        f"mean cos(delta_x, mean delta_x) = {consistency['mean_cosine_to_mean']:.3f}",
        fontsize=8,
        va="bottom",
    )
    return save_figure(fig, out_path)


def plot_projection(
    projection: ProjectionResult,
    out_path: Optional[str] = None,
    max_prompts: int = 120,
    seed: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.5, 4.5),
    legend: bool = True,
):
    """Figure 4: layer-``l`` residual streams projected onto (mean delta_x, PC1).

    Colour indicates whether each residual stream activates ``MLP.v_770^19``;
    dotted lines connect the two residual streams produced by the same prompt;
    the shape of each point encodes the model (GPT2 vs GPT2_DPO).
    """
    from src.analysis.plots import (
        make_figure,
        save_figure,
        scatter_groups,
        paired_lines,
        MODEL_COLORS,
        MODEL_MARKERS,
        ACTIVE_COLOR,
        INACTIVE_COLOR,
    )

    n = min(projection.coords_before.shape[0], projection.coords_after.shape[0])
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    if n > max_prompts:
        idx = np.sort(rng.choice(idx, size=int(max_prompts), replace=False))

    cb = projection.coords_before[idx]
    ca = projection.coords_after[idx]

    active_b = None if projection.active_before is None else np.asarray(projection.active_before)[idx]
    active_a = None if projection.active_after is None else np.asarray(projection.active_after)[idx]

    fig, ax = make_figure(figsize=figsize)

    paired_lines(ax, cb[:, 0], cb[:, 1], ca[:, 0], ca[:, 1])

    if active_b is not None and active_a is not None:
        for coords, active, name in ((cb, active_b, "gpt2"), (ca, active_a, "gpt2_dpo")):
            on = active > 0.0
            ax.scatter(
                coords[on, 0], coords[on, 1], s=22, color=ACTIVE_COLOR,
                marker=MODEL_MARKERS[name], alpha=0.8, linewidths=0.0,
                label=f"{name} (v_770^19 active)",
            )
            ax.scatter(
                coords[~on, 0], coords[~on, 1], s=16, color=INACTIVE_COLOR,
                marker=MODEL_MARKERS[name], alpha=0.7, linewidths=0.0,
                label=f"{name} (inactive)",
            )
    else:
        ax.scatter(cb[:, 0], cb[:, 1], s=18, color=MODEL_COLORS["gpt2"], marker=MODEL_MARKERS["gpt2"], label="gpt2")
        ax.scatter(ca[:, 0], ca[:, 1], s=18, color=MODEL_COLORS["gpt2_dpo"], marker=MODEL_MARKERS["gpt2_dpo"], label="gpt2_dpo")

    # mean delta arrow for reference
    ax.annotate(
        "",
        xy=(1.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2),
        zorder=1,
    )
    ax.set_title(title or f"Figure 4: residual streams at layer {projection.layer}")
    ax.set_xlabel("mean difference in residual streams (delta_x-bar)")
    ax.set_ylabel("principal component of residual streams")
    if legend:
        ax.legend(loc="best", fontsize=7)
    fig.text(0.01, 0.01, f"PC variance ratio = {projection.explained_variance_ratio:.3f}", fontsize=8, va="bottom")
    return save_figure(fig, out_path)


def plot_delta_cosine_histograms(
    result: ParameterShiftResult,
    activations: Optional[Mapping[int, np.ndarray]] = None,
    out_path: Optional[str] = None,
    bins: int = DEFAULT_BINS,
    value_range: Tuple[float, float] = COSINE_RANGE,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 4.5),
    show_layers: int = 5,
    active_axis: bool = True,
):
    """Figure 5: blue cosine-similarity histograms with orange activation overlay.

    The blue distribution per layer is the *percentage of value vectors* whose
    ``delta_MLP.v`` has the given cosine similarity against ``delta_x^{l-mid}``;
    the orange distribution is the *percentage of value vectors* whose mean
    activation (over the 1,199 RealToxicityPrompts) takes the given value.
    Layers are faded by depth so that the trend towards layer ``l`` is visible.
    """
    from src.analysis.plots import (
        make_figure,
        save_figure,
        percentage_histogram,
        HIST_COSINE_COLOR,
        HIST_ACTIVATION_COLOR,
    )
    import matplotlib.pyplot as plt

    fig, ax = make_figure(figsize=figsize)
    layers = result.layers
    if not layers:
        ax.text(0.5, 0.5, "no layers", ha="center", va="center")
        return save_figure(fig, out_path)

    # Keep the `show_layers` layers closest to the target layer (the paper notes
    # most value vectors flip direction as layers approach layer l).
    chosen = layers[-int(show_layers):] if show_layers else layers
    cmap = plt.get_cmap("Blues")
    denom = max(len(chosen) - 1, 1)

    for rank, j in enumerate(chosen):
        color = cmap(0.35 + 0.55 * rank / denom)
        percent, centers = percentage_histogram(result.cosine[j], bins=bins, value_range=value_range)
        ax.fill_between(centers, 0.0, percent, color=color, alpha=0.35, step="mid")
        ax.plot(centers, percent, color=color, lw=1.4, label=f"layer {j}" if rank in (0, len(chosen) - 1) else None)

    ax.set_xlabel("cosine similarity: delta_MLP.v vs. delta_x")
    ax.set_ylabel("percentage of value vectors")
    ax.set_xlim(*value_range)
    ax.set_title(title or f"Figure 5: cos(delta_x^{result.layer}-mid, delta_MLP.v)")

    if activations:
        act = activations.get(max(chosen)) if isinstance(activations, Mapping) else None
        if act is None:
            for j in reversed(chosen):
                if j in activations:
                    act = activations[j]
                    break
        if act is not None:
            percent, centers = percentage_histogram(act, bins=bins, value_range=value_range)
            if active_axis:
                ax2 = ax.twinx()
                ax2.fill_between(centers, 0.0, percent, color=HIST_ACTIVATION_COLOR, alpha=0.30, step="mid")
                ax2.plot(centers, percent, color=HIST_ACTIVATION_COLOR, lw=1.4, label="mean activation")
                ax2.set_ylabel("percentage of value vectors (mean activation)", color=HIST_ACTIVATION_COLOR)
                ax2.tick_params(axis="y", colors=HIST_ACTIVATION_COLOR)
                ax2.spines["right"].set_color(HIST_ACTIVATION_COLOR)
                ax2.legend(loc="upper right", fontsize=7)
            else:
                ax.fill_between(centers, 0.0, percent, color=HIST_ACTIVATION_COLOR, alpha=0.30, step="mid")
                ax.plot(centers, percent, color=HIST_ACTIVATION_COLOR, lw=1.4, label="mean activation")

    ax.legend(loc="upper left", fontsize=7)
    return save_figure(fig, out_path)


def plot_layer_shifts(
    shifts: Mapping[int, ShiftResult],
    out_path: Optional[str] = None,
    activations: Optional[Mapping[int, np.ndarray]] = None,
    bins: int = DEFAULT_BINS,
    value_range: Tuple[float, float] = COSINE_RANGE,
    figsize: Tuple[float, float] = (10.0, 8.0),
):
    """Appendix Figure 7: ``delta_x`` vs ``delta_MLP.v`` for several layers."""
    from src.analysis.plots import make_figure, save_figure

    layers = sorted(int(k) for k in shifts.keys())
    n = len(layers)
    if n == 0:  # pragma: no cover - defensive
        fig, _ = make_figure(figsize=figsize)
        return save_figure(fig, out_path)

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = make_figure(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for ax, l in zip(axes.reshape(-1), layers):
        shift = shifts[l]
        result = shift.meta.get("cosine_result") if isinstance(shift.meta, dict) else None
        if result is None:
            ax.text(0.5, 0.5, f"layer {l}\n(no cosine result)", ha="center", va="center")
            continue
        act = None
        if activations is not None and l in activations:
            act = {l: activations[l]}
        _plot_into(ax, result, act, bins=bins, value_range=value_range, title=f"layer {l}")

    for ax in axes.reshape(-1)[n:]:
        ax.axis("off")

    fig.suptitle("Appendix Figure 7: shift in residual streams vs. shift in MLP value vectors")
    return save_figure(fig, out_path)


def _plot_into(ax, result: ParameterShiftResult, activations, bins: int, value_range, title: str):
    """Draw one Figure 5-style panel into an existing axis."""
    from src.analysis.plots import percentage_histogram, HIST_COSINE_COLOR, HIST_ACTIVATION_COLOR
    import matplotlib.pyplot as plt

    layers = result.layers
    chosen = layers[-5:] if len(layers) > 5 else layers
    cmap = plt.get_cmap("Blues")
    denom = max(len(chosen) - 1, 1)
    for rank, j in enumerate(chosen):
        color = cmap(0.35 + 0.55 * rank / denom)
        percent, centers = percentage_histogram(result.cosine[j], bins=bins, value_range=value_range)
        ax.fill_between(centers, 0.0, percent, color=color, alpha=0.35, step="mid")

    if activations:
        for j in reversed(chosen):
            if j in activations:
                percent, centers = percentage_histogram(activations[j], bins=bins, value_range=value_range)
                ax.fill_between(centers, 0.0, percent, color=HIST_ACTIVATION_COLOR, alpha=0.28, step="mid")
                break

    ax.set_xlim(*value_range)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("cosine similarity", fontsize=8)
    ax.set_ylabel("percentage of value vectors", fontsize=8)


__all__ = [
    "DEFAULT_LAYER",
    "DEFAULT_MLP_IDX",
    "TARGET_VECTOR",
    "COSINE_RANGE",
    "DEFAULT_BINS",
    "N_CHALLENGE_PROMPTS",
    "ResidualStreamCollection",
    "ShiftResult",
    "ProjectionResult",
    "ParameterShiftResult",
    "collect_residual_streams",
    "compute_residual_shift",
    "compute_residual_shift_models",
    "cosine_similarity_rows",
    "shift_consistency",
    "projection_axes",
    "project_residual_streams",
    "mean_shift_vector",
    "parameter_value_deltas",
    "parameter_delta_cosine_against_shift",
    "per_layer_histograms",
    "activation_histogram",
    "contribution_analysis",
    "mean_activations_by_layer",
    "collect_activation_histogram",
    "save_shift_result",
    "save_parameter_shift",
    "load_shift_result",
    "default_paths",
    "plot_residual_shift",
    "plot_projection",
    "plot_delta_cosine_histograms",
    "plot_layer_shifts",
]
