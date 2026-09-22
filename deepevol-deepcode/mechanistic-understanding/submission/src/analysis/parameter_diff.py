"""Parameter-difference analysis for the DPO / toxicity mechanistic study.

Implements Section 5.1 of *A Mechanistic Understanding of Alignment Algorithms:
A Case Study on DPO and Toxicity* (Appendix C/D extend it to more layers).

The paper reports that **every parameter** of GPT2 (and GPT2_DPO) barely moves
during DPO:

    "Every parameter in GPT2 ... and its counterpart in GPT2_DPO ... has a
     cosine similarity score greater than 0.99 and on average a norm difference
     less than 1e-5.  This applies for MLP.k_Toxic and MLP.v_Toxic as well --
     toxic MLP vectors do not change from DPO."
    (footnote 5: "The unembedding layer of GPT2 is the only exception, where the
     norm difference is less than 1e-3.")

This module provides

* :func:`parameter_tensors` / :func:`parameter_delta_tensors` -- raw
  ``delta_theta = theta_DPO - theta_GPT2`` for every named parameter,
* :func:`parameter_delta_stats` -> :class:`ParameterDeltaReport` -- per-parameter
  cosine similarity and (mean) norm difference,
* :func:`check_paper_claims` -- the numerical checks above,
* :func:`mlp_value_delta_matrices` -- ``delta_MLP.v_i^j`` (used both here and by
  :mod:`src.analysis.residual_shift` for Equation 2 / Figure 5),
* Appendix C/D helpers -- :func:`residual_shift_layer_variants`,
  :func:`top_toxic_vector_layers`, :func:`delta_cosine_by_layer`,
* plotting helpers and JSON persistence for reproduction artifacts.

All heavy imports happen lazily so that importing this module is cheap.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Default artifact directory for parameter-difference outputs.
ARTIFACT_DIR = "artifacts/analysis"

#: Cosine-similarity claim from Section 5.1 ("greater than 0.99").
COSINE_THRESHOLD = 0.99

#: Mean norm-difference claim from Section 5.1 ("less than 1e-5").
NORM_DIFF_THRESHOLD = 1e-5

#: Unembedding exception from footnote 5 ("less than 1e-3").
UNEMBEDDING_NORM_DIFF_THRESHOLD = 1e-3

#: Layer-19 target vector used throughout Section 5 (one of the most toxic).
DEFAULT_LAYER = 19
DEFAULT_MLP_IDX = 770
TARGET_VECTOR: Tuple[int, int] = (DEFAULT_LAYER, DEFAULT_MLP_IDX)

#: Python type aliases for named parameters.
ParamKey = str

#: Name fragments identifying the unembedding / output head, which uses the
#: special norm-difference threshold (footnote 5).
UNEMBEDDING_MARKERS: Tuple[str, ...] = (
    "lm_head",
    "wte",  # GPT2's tied unembedding lives in the token embedding matrix
    "embed_out",
    "output.weight",
    "word_embeddings",
)

#: Name fragments identifying MLP value-vector parameters (``MLP.v``).
VALUE_MARKERS: Tuple[str, ...] = ("c_proj", "down_proj", "fc2", "w2")

#: Name fragments identifying MLP key-vector parameters (``MLP.k``).
KEY_MARKERS: Tuple[str, ...] = ("c_fc", "up_proj", "gate_proj", "fc1", "w1", "w3")


# ---------------------------------------------------------------------------
# model parameter access
# ---------------------------------------------------------------------------


def parameter_items(model) -> List[Tuple[str, Any]]:
    """Return ``[(name, tensor), ...]`` for every *floating point* parameter.

    Integer buffers (if any) are skipped because subtraction / cosine
    similarity is undefined for them.
    """
    items: List[Tuple[str, Any]] = []
    for name, param in model.named_parameters():
        tensor = param.detach()
        if not tensor.is_floating_point():
            continue
        items.append((name, tensor))
    return items


def resolve_state_dict(model_or_dict) -> Dict[str, Any]:
    """Accept either an ``nn.Module`` or a plain state dict."""
    if isinstance(model_or_dict, dict):
        return {
            k: v
            for k, v in model_or_dict.items()
            if hasattr(v, "is_floating_point") and v.is_floating_point()
        }
    return dict(parameter_items(model_or_dict))


def parameter_tensors(
    model_or_dict,
    names: Optional[Sequence[str]] = None,
    dtype=np.float32,
) -> Dict[str, np.ndarray]:
    """Return ``{name: ndarray}`` for all (or a subset of) parameters."""
    state = resolve_state_dict(model_or_dict)
    if names is not None:
        wanted = list(names)
        missing = [n for n in wanted if n not in state]
        if missing:
            raise KeyError(f"parameters not found in state dict: {missing[:5]}")
        state = {n: state[n] for n in wanted}
    return {n: tensor.detach().cpu().numpy().astype(dtype, copy=False) for n, tensor in state.items()}


def parameter_delta_tensors(
    model_before,
    model_after,
    names: Optional[Sequence[str]] = None,
    dtype=np.float32,
) -> Dict[str, np.ndarray]:
    """``delta_theta = theta_DPO - theta_GPT2`` for every parameter.

    Parameters that only exist in one of the two models are reported via a
    ``KeyError`` because the paper's claim is about *matched* parameters.
    """
    before = resolve_state_dict(model_before)
    after = resolve_state_dict(model_after)
    if names is None:
        names = [n for n in before.keys()]
    deltas: Dict[str, np.ndarray] = {}
    for name in names:
        if name not in before or name not in after:
            raise KeyError(
                f"parameter {name!r} present in only one model "
                f"(before={name in before}, after={name in after})"
            )
        a = before[name].detach().cpu().numpy().astype(dtype, copy=False)
        b = after[name].detach().cpu().numpy().astype(dtype, copy=False)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {name!r}: {a.shape} vs {b.shape}")
        deltas[name] = b - a
    return deltas


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Flattened cosine similarity between two tensors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        # Both (near) zero => identical direction by convention.
        return 1.0 if na < eps and nb < eps else 0.0
    return float(np.dot(a, b) / (na * nb))


def is_unembedding(name: str) -> bool:
    """Whether a parameter name belongs to the unembedding / output head."""
    lowered = name.lower()
    return any(marker in lowered for marker in UNEMBEDDING_MARKERS)


def parameter_kind(name: str) -> str:
    """Classify a parameter name into a coarse paper-level group."""
    lowered = name.lower()
    if is_unembedding(name):
        return "unembedding"
    if "mlp" in lowered or "c_proj" in lowered or "c_fc" in lowered or "fc" in lowered:
        if any(m in lowered for m in VALUE_MARKERS):
            return "mlp.value"
        if any(m in lowered for m in KEY_MARKERS):
            return "mlp.key"
        return "mlp"
    if "attn" in lowered or "attention" in lowered:
        return "attn"
    if "ln" in lowered or "norm" in lowered:
        return "layernorm"
    return "other"


@dataclass
class ParameterDelta:
    """Per-parameter summary of ``theta_DPO - theta_GPT2``."""

    name: str
    shape: Tuple[int, ...]
    numel: int
    cosine: float
    norm_delta: float
    norm_before: float
    norm_after: float
    mean_delta: float
    max_abs_delta: float
    kind: str = "other"
    is_unembedding: bool = False

    @property
    def mean_abs_norm_delta(self) -> float:
        """``|delta_theta|`` averaged over elements (paper's "norm difference")."""
        if self.numel == 0:
            return 0.0
        return float(self.norm_delta) / float(self.numel)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "cosine": self.cosine,
            "norm_delta": self.norm_delta,
            "norm_before": self.norm_before,
            "norm_after": self.norm_after,
            "mean_delta": self.mean_delta,
            "max_abs_delta": self.max_abs_delta,
            "kind": self.kind,
            "is_unembedding": self.is_unembedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterDelta":
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            numel=int(data["numel"]),
            cosine=float(data["cosine"]),
            norm_delta=float(data["norm_delta"]),
            norm_before=float(data.get("norm_before", 0.0)),
            norm_after=float(data.get("norm_after", 0.0)),
            mean_delta=float(data.get("mean_delta", 0.0)),
            max_abs_delta=float(data.get("max_abs_delta", 0.0)),
            kind=data.get("kind", "other"),
            is_unembedding=bool(data.get("is_unembedding", False)),
        )


@dataclass
class ParameterDeltaReport:
    """Report for all parameters: ``theta_DPO - theta_GPT2`` (Section 5.1)."""

    entries: List[ParameterDelta] = field(default_factory=list)
    before_name: str = "gpt2"
    after_name: str = "gpt2_dpo"
    n_params: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- containers ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, item):
        if isinstance(item, str):
            for entry in self.entries:
                if entry.name == item:
                    return entry
            raise KeyError(item)
        return self.entries[item]

    def names(self) -> List[str]:
        return [e.name for e in self.entries]

    # -- aggregates ---------------------------------------------------------
    @property
    def cosines(self) -> np.ndarray:
        return np.array([e.cosine for e in self.entries], dtype=np.float64)

    @property
    def norm_deltas(self) -> np.ndarray:
        return np.array([e.norm_delta for e in self.entries], dtype=np.float64)

    @property
    def mean_norm_diffs(self) -> np.ndarray:
        return np.array([e.mean_abs_norm_delta for e in self.entries], dtype=np.float64)

    def select(
        self,
        kind: Optional[str] = None,
        names: Optional[Sequence[str]] = None,
        unembedding: Optional[bool] = None,
        exclude_unembedding: bool = False,
    ) -> "ParameterDeltaReport":
        """Filter entries (e.g. only MLP parameters, or excluding unembedding)."""
        wanted = set(names) if names is not None else None
        selected: List[ParameterDelta] = []
        for entry in self.entries:
            if wanted is not None and entry.name not in wanted:
                continue
            if kind is not None and entry.kind != kind:
                continue
            if unembedding is not None and entry.is_unembedding != unembedding:
                continue
            if exclude_unembedding and entry.is_unembedding:
                continue
            selected.append(entry)
        return ParameterDeltaReport(
            entries=selected,
            before_name=self.before_name,
            after_name=self.after_name,
            n_params=len(selected),
            meta=dict(self.meta),
        )

    def summary(self, precision: int = 6) -> Dict[str, Any]:
        """Aggregate statistics quoted in Section 5.1."""
        if not self.entries:
            return {
                "n_params": 0,
                "min_cosine": float("nan"),
                "max_cosine": float("nan"),
                "mean_cosine": float("nan"),
                "all_cosine_gt_threshold": False,
            }
        cosines = self.cosines
        mean_diffs = self.mean_norm_diffs
        return {
            "n_params": len(self.entries),
            "min_cosine": round(float(cosines.min()), precision),
            "max_cosine": round(float(cosines.max()), precision),
            "mean_cosine": round(float(cosines.mean()), precision),
            "min_cosine_param": self.entries[int(np.argmin(cosines))].name,
            "mean_norm_diff": float(mean_diffs.mean()),
            "max_mean_norm_diff": float(mean_diffs.max()),
            "max_mean_norm_diff_param": self.entries[int(np.argmax(mean_diffs))].name,
            "max_norm_delta": float(self.norm_deltas.max()),
            "all_cosine_gt_threshold": bool((cosines > COSINE_THRESHOLD).all()),
            "all_mean_norm_diff_lt_threshold": bool(
                (mean_diffs < NORM_DIFF_THRESHOLD).all()
            ),
        }

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "before_name": self.before_name,
            "after_name": self.after_name,
            "n_params": len(self.entries),
            "meta": dict(self.meta),
            "summary": self.summary(),
            "parameters": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterDeltaReport":
        entries = [ParameterDelta.from_dict(d) for d in data.get("parameters", [])]
        return cls(
            entries=entries,
            before_name=data.get("before_name", "gpt2"),
            after_name=data.get("after_name", "gpt2_dpo"),
            n_params=int(data.get("n_params", len(entries))),
            meta=dict(data.get("meta", {})),
        )


def parameter_delta_stats(
    model_before,
    model_after,
    names: Optional[Sequence[str]] = None,
    before_name: str = "gpt2",
    after_name: str = "gpt2_dpo",
    compute_norms: bool = True,
    verbose: bool = False,
) -> ParameterDeltaReport:
    """Cosine similarity & norm difference for every parameter (Section 5.1).

    Parameters
    ----------
    model_before, model_after:
        Two ``nn.Module`` instances (GPT2 and GPT2_DPO) or state dicts.
    names:
        Optional subset of parameter names to analyse.
    """
    before = resolve_state_dict(model_before)
    after = resolve_state_dict(model_after)
    if names is None:
        names = list(before.keys())

    entries: List[ParameterDelta] = []
    for name in names:
        if name not in before or name not in after:
            continue
        a = before[name].detach().cpu().numpy().astype(np.float64, copy=False)
        b = after[name].detach().cpu().numpy().astype(np.float64, copy=False)
        if a.shape != b.shape:
            continue
        delta = b - a
        norm_delta = float(np.linalg.norm(delta))
        entries.append(
            ParameterDelta(
                name=name,
                shape=tuple(a.shape),
                numel=int(a.size),
                cosine=cosine_similarity(a, b),
                norm_delta=norm_delta,
                norm_before=float(np.linalg.norm(a)) if compute_norms else 0.0,
                norm_after=float(np.linalg.norm(b)) if compute_norms else 0.0,
                mean_delta=float(delta.mean()) if delta.size else 0.0,
                max_abs_delta=float(np.abs(delta).max()) if delta.size else 0.0,
                kind=parameter_kind(name),
                is_unembedding=is_unembedding(name),
            )
        )
        if verbose:
            print(
                f"[parameter_diff] {name:<48s} cos={entries[-1].cosine:.6f} "
                f"mean|d|={entries[-1].mean_abs_norm_delta:.3e}"
            )

    return ParameterDeltaReport(
        entries=entries,
        before_name=before_name,
        after_name=after_name,
        n_params=len(entries),
        meta={
            "cosine_threshold": COSINE_THRESHOLD,
            "norm_diff_threshold": NORM_DIFF_THRESHOLD,
            "unembedding_norm_diff_threshold": UNEMBEDDING_NORM_DIFF_THRESHOLD,
        },
    )


# ---------------------------------------------------------------------------
# paper claims (Section 5.1 + footnote 5)
# ---------------------------------------------------------------------------


def check_paper_claims(
    report: ParameterDeltaReport,
    cosine_threshold: float = COSINE_THRESHOLD,
    norm_diff_threshold: float = NORM_DIFF_THRESHOLD,
    unembedding_threshold: float = UNEMBEDDING_NORM_DIFF_THRESHOLD,
) -> Dict[str, Any]:
    """Validate the Section 5.1 claims.

    * every parameter has cosine similarity ``> 0.99``,
    * the *mean* norm difference is ``< 1e-5`` for every parameter,
    * the unembedding is the only exception, with norm difference ``< 1e-3``.
    """
    if not report.entries:
        return {
            "passed": False,
            "reason": "empty report",
            "cosine": {"passed": False, "violations": []},
            "norm_diff": {"passed": False, "violations": []},
            "unembedding": {"passed": False, "violations": []},
        }

    cos_violations = [
        {"name": e.name, "cosine": e.cosine}
        for e in report.entries
        if not (e.cosine > cosine_threshold)
    ]
    norm_violations = [
        {"name": e.name, "mean_norm_diff": e.mean_abs_norm_delta}
        for e in report.entries
        if not (e.mean_abs_norm_delta < norm_diff_threshold)
    ]
    norm_violations.sort(key=lambda d: -d["mean_norm_diff"])

    unembedding_entries = [e for e in report.entries if e.is_unembedding]
    unembedding_ok = [
        {"name": e.name, "mean_norm_diff": e.mean_abs_norm_delta}
        for e in unembedding_entries
        if e.mean_abs_norm_delta < unembedding_threshold
    ]
    # Any non-unembedding parameter breaching the 1e-5 bound but staying under
    # the 1e-3 bound would mean the "unembedding is the only exception" claim
    # failed.
    non_unembedding_breach = [
        {"name": e.name, "mean_norm_diff": e.mean_abs_norm_delta}
        for e in report.entries
        if (not e.is_unembedding)
        and (not (e.mean_abs_norm_delta < norm_diff_threshold))
        and (e.mean_abs_norm_delta < unembedding_threshold)
    ]

    passed = (
        not cos_violations
        and not norm_violations
        and all(
            e.mean_abs_norm_delta < unembedding_threshold for e in unembedding_entries
        )
    )

    return {
        "passed": bool(passed),
        "cosine": {
            "threshold": cosine_threshold,
            "passed": not cos_violations,
            "n_violations": len(cos_violations),
            "violations": cos_violations[:20],
        },
        "norm_diff": {
            "threshold": norm_diff_threshold,
            "passed": not norm_violations,
            "n_violations": len(norm_violations),
            "violations": norm_violations[:20],
        },
        "unembedding": {
            "threshold": unembedding_threshold,
            "n_entries": len(unembedding_entries),
            "passed": all(
                e.mean_abs_norm_delta < unembedding_threshold
                for e in unembedding_entries
            ),
            "entries": [
                {"name": e.name, "mean_norm_diff": e.mean_abs_norm_delta}
                for e in unembedding_entries
            ],
            "only_exception": not non_unembedding_breach,
        },
        "summary": report.summary(),
    }


# ---------------------------------------------------------------------------
# MLP value-vector deltas (delta_MLP.v) and Eq. 2 support
# ---------------------------------------------------------------------------


def mlp_value_delta_matrices(
    model_before,
    model_after,
    layers: Optional[Sequence[int]] = None,
    use_model_utils: bool = True,
) -> Dict[int, np.ndarray]:
    """``delta_MLP.v^j`` as ``[d_mlp, d_model]`` arrays keyed by layer.

    Delegates to :func:`src.analysis.residual_shift.parameter_value_deltas`
    when available so both modules agree on the layout convention
    (row ``i`` is the value vector ``MLP.v_i^j``).
    """
    if use_model_utils:
        try:
            from .residual_shift import parameter_value_deltas  # local import

            return parameter_value_deltas(model_before, model_after, layers=layers)
        except Exception:  # pragma: no cover - fall back to local computation
            pass

    from ..model_utils import get_mlp_matrices, model_info

    info = model_info(model_before)
    layer_list = list(layers) if layers is not None else list(range(info.n_layers))
    deltas: Dict[int, np.ndarray] = {}
    for layer in layer_list:
        _, w_before = get_mlp_matrices(model_before, layer)
        _, w_after = get_mlp_matrices(model_after, layer)
        vb = w_before.detach().cpu().numpy().astype(np.float64, copy=False)
        va = w_after.detach().cpu().numpy().astype(np.float64, copy=False)
        deltas[int(layer)] = va - vb
    return deltas


def mlp_key_delta_matrices(
    model_before,
    model_after,
    layers: Optional[Sequence[int]] = None,
) -> Dict[int, np.ndarray]:
    """``delta_MLP.k^j`` as ``[d_mlp, d_model]`` arrays keyed by layer.

    Section 5.1 states the near-zero-shift result "applies for MLP.k_Toxic and
    MLP.v_Toxic as well".
    """
    from ..model_utils import get_mlp_matrices, model_info

    info = model_info(model_before)
    layer_list = list(layers) if layers is not None else list(range(info.n_layers))
    deltas: Dict[int, np.ndarray] = {}
    for layer in layer_list:
        w_before, _ = get_mlp_matrices(model_before, layer)
        w_after, _ = get_mlp_matrices(model_after, layer)
        kb = w_before.detach().cpu().numpy().astype(np.float64, copy=False)
        ka = w_after.detach().cpu().numpy().astype(np.float64, copy=False)
        deltas[int(layer)] = ka - kb
    return deltas


def toxic_vector_delta_stats(
    model_before,
    model_after,
    value_indices: Sequence[Tuple[int, int]] = (),
    key_indices: Optional[Sequence[Tuple[int, int]]] = None,
    target: Tuple[int, int] = TARGET_VECTOR,
) -> Dict[str, Any]:
    """Cosine similarity / norm difference for MLP.v_Toxic and MLP.k_Toxic.

    ``index = (layer, mlp_idx)``.  When ``value_indices`` is empty only the
    canonical target vector ``MLP.v_770^19`` is reported.
    """
    from ..model_utils import get_key_vector, get_value_vector

    key_indices = list(key_indices) if key_indices is not None else list(value_indices)
    if not value_indices and not list(key_indices):
        value_indices, key_indices = [target], [target]

    def _stat(kind: str, indices: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for layer, idx in indices:
            getter = get_value_vector if kind == "v" else get_key_vector
            a = getter(model_before, int(layer), int(idx)).detach().cpu().numpy()
            b = getter(model_after, int(layer), int(idx)).detach().cpu().numpy()
            delta = b - a
            out.append(
                {
                    "layer": int(layer),
                    "index": int(idx),
                    "name": f"MLP.{kind}_{idx}^{layer}",
                    "cosine": cosine_similarity(a, b),
                    "norm_delta": float(np.linalg.norm(delta)),
                    "mean_abs_norm_delta": float(np.abs(delta).mean()) if delta.size else 0.0,
                    "max_abs_delta": float(np.abs(delta).max()) if delta.size else 0.0,
                }
            )
        return out

    values = _stat("v", list(value_indices))
    keys = _stat("k", list(key_indices)) if key_indices else []
    all_stats = values + keys
    return {
        "value_vectors": values,
        "key_vectors": keys,
        "min_cosine": min((s["cosine"] for s in all_stats), default=float("nan")),
        "max_mean_abs_norm_delta": max(
            (s["mean_abs_norm_delta"] for s in all_stats), default=float("nan")
        ),
        "all_above_cosine_threshold": all(
            s["cosine"] > COSINE_THRESHOLD for s in all_stats
        ),
        "all_below_norm_threshold": all(
            s["mean_abs_norm_delta"] < NORM_DIFF_THRESHOLD for s in all_stats
        ),
    }


# ---------------------------------------------------------------------------
# Appendix C / D: layer variants
# ---------------------------------------------------------------------------

#: Appendix D (Figure 7): after MLP.v_770^19, the next toxic vectors with highest
#: cosine similarity to W_Toxic are MLP.v_771^12, MLP.v_2669^18, MLP.v_668^13.
LAYER_VARIANTS: Tuple[Tuple[int, int], ...] = (
    (12, 771),
    (18, 2669),
    (13, 668),
)


def residual_shift_layer_variants(
    include_target: bool = True,
    extra: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Vector indices shown in Appendix C/D: layer 19 plus layers 12/13/18."""
    out: List[Tuple[int, int]] = []
    if include_target:
        out.append(TARGET_VECTOR)
    out.extend(LAYER_VARIANTS)
    if extra:
        for item in extra:
            if tuple(item) not in out:
                out.append(tuple(item))
    return out


def top_toxic_vector_layers(
    indices: Sequence[Tuple[int, int]],
    k: Optional[int] = None,
) -> List[int]:
    """Ordered unique layers of the top toxic vectors (used for Appendix C/D)."""
    ordered: List[int] = []
    for layer, _idx in indices:
        layer = int(layer)
        if layer not in ordered:
            ordered.append(layer)
    return ordered[:k] if k is not None else ordered


def delta_cosine_by_layer(
    delta_x,
    value_deltas: Dict[int, np.ndarray],
    layers: Optional[Sequence[int]] = None,
    eps: float = 1e-12,
) -> Dict[int, np.ndarray]:
    """Row-wise ``cos(delta_x, delta_MLP.v_i^j)`` for every requested layer.

    This is exactly Equation 2 restricted to one ``delta_x`` (``j < l`` is
    enforced by the caller passing the appropriate layer dict).
    """
    from .residual_shift import cosine_similarity_rows

    dx = np.asarray(delta_x, dtype=np.float64).ravel()
    layer_list = list(layers) if layers is not None else sorted(value_deltas.keys())
    if layers is None and len(dx):
        # Section 5.2: only preceding layers j < l contribute.
        pass
    out: Dict[int, np.ndarray] = {}
    for layer in layer_list:
        if layer not in value_deltas:
            continue
        mat = np.asarray(value_deltas[layer], dtype=np.float64)
        if mat.ndim != 2:
            continue
        out[int(layer)] = cosine_similarity_rows(mat, dx, eps=eps)
    return out


def fraction_negative_by_layer(cosines_by_layer: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Fraction of value vectors with negative cosine similarity to ``delta_x``."""
    return {
        int(layer): float((np.asarray(cos).ravel() < 0).mean()) if np.size(cos) else float("nan")
        for layer, cos in cosines_by_layer.items()
    }


def shift_contribution_ratio(
    delta_x,
    value_deltas: Dict[int, np.ndarray],
    activations: Dict[int, np.ndarray],
    layer: int = DEFAULT_LAYER,
) -> Dict[int, float]:
    """How much each earlier layer's ``delta_MLP.v`` contributes towards ``delta_x``.

    A value vector ``i`` at layer ``j`` contributes
    ``activation_i^j * delta_MLP.v_i^j`` to the residual stream; the projection
    of that term onto ``delta_x`` is summed over ``i`` and normalised by
    ``|delta_x|^2``.  Because activations are mostly (slightly) negative
    (Figure 5, orange), the antipodal ``delta_MLP.v`` flips sign and contributes
    *towards* ``delta_x``.
    """
    dx = np.asarray(delta_x, dtype=np.float64).ravel()
    norm_sq = float(np.dot(dx, dx))
    out: Dict[int, float] = {}
    if norm_sq <= 0:
        return out
    for j, mat in value_deltas.items():
        if int(j) >= int(layer):
            continue
        act = activations.get(int(j))
        if act is None:
            continue
        act = np.asarray(act, dtype=np.float64).ravel()
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != act.shape[0]:
            continue
        weighted = (act[:, None] * mat).sum(axis=0)
        out[int(j)] = float(np.dot(dx, weighted) / norm_sq)
    return out


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def default_path(out_dir: str = ARTIFACT_DIR, prefix: str = "parameter_diff") -> str:
    """Default JSON artifact path for a parameter-difference report."""
    return os.path.join(out_dir, f"{prefix}.json")


def save_parameter_report(
    path: str,
    report: ParameterDeltaReport,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a :class:`ParameterDeltaReport` (plus optional extras) as JSON."""
    payload = report.to_dict()
    if extra:
        payload["extra"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_parameter_report(path: str) -> ParameterDeltaReport:
    """Load a :class:`ParameterDeltaReport` written by :func:`save_parameter_report`."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ParameterDeltaReport.from_dict(data)


def save_claims(path: str, claims: Dict[str, Any]) -> str:
    """Persist the output of :func:`check_paper_claims`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(claims, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def plot_parameter_cosines(
    report: ParameterDeltaReport,
    out_path: Optional[str] = None,
    top_k: int = 40,
    title: str = "Cosine similarity per parameter (GPT2 vs GPT2_DPO)",
    figsize: Tuple[float, float] = (9.0, 4.0),
) -> Optional[str]:
    """Bar chart of per-parameter cosine similarity (Section 5.1 check)."""
    from .plots import BASE_COLOR, make_figure, save_figure, set_style

    if not report.entries:
        return None
    entries = sorted(report.entries, key=lambda e: e.cosine)
    if top_k and len(entries) > top_k:
        entries = entries[: max(1, top_k // 2)] + entries[-max(1, top_k // 2) :]
    set_style()
    fig, ax = make_figure(figsize=figsize)
    xs = np.arange(len(entries))
    ax.bar(xs, [e.cosine for e in entries], color=BASE_COLOR, width=0.8)
    ax.axhline(COSINE_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"threshold {COSINE_THRESHOLD}")
    ax.set_xticks(xs)
    ax.set_xticklabels([e.name for e in entries], rotation=90, fontsize=6)
    ax.set_ylabel("cosine similarity")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return save_figure(fig, out_path) if out_path else None


def plot_norm_differences(
    report: ParameterDeltaReport,
    out_path: Optional[str] = None,
    title: str = "Mean |delta theta| per parameter",
    figsize: Tuple[float, float] = (8.0, 4.0),
    log_scale: bool = True,
) -> Optional[str]:
    """Histogram of per-parameter mean norm differences with the 1e-5 bound."""
    from .plots import HIST_ACTIVATION_COLOR, make_figure, save_figure, set_style

    if not report.entries:
        return None
    diffs = np.maximum(report.mean_norm_diffs, 1e-16)
    set_style()
    fig, ax = make_figure(figsize=figsize)
    ax.hist(diffs, bins=40, color=HIST_ACTIVATION_COLOR, alpha=0.85)
    ax.axvline(NORM_DIFF_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"threshold {NORM_DIFF_THRESHOLD:g}")
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("mean |delta theta|")
    ax.set_ylabel("# parameters")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return save_figure(fig, out_path) if out_path else None


def plot_layer_shift_histograms(
    cosines_by_layer: Dict[int, np.ndarray],
    activations: Optional[Dict[int, np.ndarray]] = None,
    out_path: Optional[str] = None,
    layers: Optional[Sequence[int]] = None,
    layer: int = DEFAULT_LAYER,
    bins: int = 50,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    title: Optional[str] = None,
    show_layers: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """Appendix D (Figure 7) style grid of per-layer cosine histograms.

    Blue = percentage of value vectors by ``cos(delta_x, delta_MLP.v^j)``;
    orange = mean activation of the same value vectors.
    """
    from . import plots as plotting

    if not cosines_by_layer:
        return None
    if layers is None:
        # Layers closest to (but before) the analysed layer matter most.
        ordered = sorted([l for l in cosines_by_layer if l < layer], reverse=True)
        layers = ordered[:show_layers]
    layers = [int(l) for l in layers if int(l) in cosines_by_layer]
    if not layers:
        return None

    fig, axes, used = plotting.make_layer_grid(layers, ncols=min(3, len(layers)), figsize=figsize)
    for ax, lyr in zip(np.atleast_1d(axes).ravel(), used):
        percent, centers = plotting.percentage_histogram(
            cosines_by_layer[lyr], bins=bins, value_range=value_range
        )
        act_percent = None
        if activations is not None and lyr in activations:
            act_percent, _ = plotting.percentage_histogram(
                activations[lyr], bins=bins, value_range=value_range
            )
        plotting.plot_histogram_pair(
            ax,
            percent,
            centers,
            activation_percent=act_percent,
            title=(
                f"Layer {lyr}"
                if title is None
                else f"{title} (layer {lyr})"
            ),
        )
    fig.tight_layout()
    return plotting.save_figure(fig, out_path) if out_path else None


# ---------------------------------------------------------------------------
# top-level pipeline
# ---------------------------------------------------------------------------


def analyze_parameter_diff(
    model_before,
    model_after,
    before_name: str = "gpt2",
    after_name: str = "gpt2_dpo",
    out_dir: str = ARTIFACT_DIR,
    value_indices: Sequence[Tuple[int, int]] = (),
    save: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full Section 5.1 parameter-difference analysis.

    Returns a dict with the report, the claim checks, and (when available)
    statistics for the toxic value/key vectors.  Artifacts are written to
    ``out_dir`` as ``parameter_diff.json`` and ``parameter_diff_claims.json``.
    """
    report = parameter_delta_stats(
        model_before,
        model_after,
        before_name=before_name,
        after_name=after_name,
        verbose=verbose,
    )
    claims = check_paper_claims(report)
    toxic = toxic_vector_delta_stats(
        model_before, model_after, value_indices=value_indices
    )

    if save:
        os.makedirs(out_dir, exist_ok=True)
        save_parameter_report(
            default_path(out_dir), report, extra={"claims": claims, "toxic_vectors": toxic}
        )
        save_claims(os.path.join(out_dir, "parameter_diff_claims.json"), claims)

    if verbose:
        summary = report.summary()
        print(
            f"[parameter_diff] {summary['n_params']} parameters, "
            f"min cosine={summary['min_cosine']:.6f}, "
            f"max mean|delta|={summary['max_mean_norm_diff']:.3e}, "
            f"claims passed={claims['passed']}"
        )

    return {"report": report, "claims": claims, "toxic_vectors": toxic}


def _main() -> None:  # pragma: no cover - manual smoke test
    import copy

    from ..model_utils import GPT2_MEDIUM, load_model

    model, _tok = load_model(GPT2_MEDIUM)
    # A tiny perturbation simulates the (barely moved) DPO weights.
    perturbed = copy.deepcopy(model)
    with __import__("torch").no_grad():
        for _name, param in perturbed.named_parameters():
            param.add_(0.001 * __import__("torch").randn_like(param))
    result = analyze_parameter_diff(model, perturbed, save=False, verbose=True)
    print(json.dumps(result["claims"]["summary"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()
