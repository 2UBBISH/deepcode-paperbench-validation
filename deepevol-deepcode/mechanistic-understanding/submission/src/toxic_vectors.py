"""Toxic vector extraction, SVD decomposition, and vocabulary projection.

Reproduction of Section 3.1 / 3.2 (and Appendix A) of
"A Mechanistic Understanding of Alignment Algorithms:
A Case Study on DPO and Toxicity".

Paper specification (Section 3.1)
---------------------------------
* Given the probe matrix ``W_Toxic`` (shape ``[d_model, 2]``, column 0 =
  non-toxic, column 1 = toxic -- see the author clarifications), we "search for
  value vectors that promote toxicity, by checking for all value vectors with
  the highest cosine similarity with ``W_Toxic``".  Per the author
  clarification, the direction used is ``W_Toxic[:, 1]``.
* The selected value vectors are ``MLP.v_Toxic`` and their *matching* key
  vectors are ``MLP.k_Toxic``.  Both keep the ``(layer, idx)`` bookkeeping.
* "After extracting a set of ``N (=128)`` ``MLP.v_Toxic`` vectors, we stack them
  into a ``N x d`` matrix.  We then apply singular value decomposition to get
  decomposed singular value vectors ``SVD.U_Toxic``."  The author clarification
  is explicit: the SVD is applied to the *transpose* of the ``N x d`` matrix,
  i.e. to a ``d x N`` matrix, so that the columns of ``U`` (the
  ``SVD.U_Toxic[i]`` vectors) are ``d``-dimensional.

Paper specification (Section 3.2 / Appendix A)
----------------------------------------------
Each value vector writes one sub-update ``m_i^l * v_i^l`` to the residual
stream and, from Appendix A,

    p(w | x + m_i v_i, E) ∝ exp(e_w . x) * exp(e_w . m_i v_i)

so ``e_w . m_i v_i > 0`` promotes token ``w`` while ``< 0`` suppresses it.  The
*static* projection ``r_i^l = E v_i^l`` therefore ranks the tokens promoted by
value vector ``v_i`` (highest dot products ``e_w . v_i`` promoted most).  The
sign of the activation ``m_i`` decides promotion vs suppression.

This module provides:
  * cosine-similarity ranking of all MLP value vectors against ``W_Toxic[:, 1]``
  * extraction of the top-``N`` toxic value/key vectors
  * SVD decomposition (on the transposed ``d x N`` matrix) -> ``SVD.U_Toxic``
  * vocabulary-space projections ``r = E v`` with top-token reporting
  * Table 1 token-group validation helpers
  * JSON/torch persistence used by the intervention, DPO and un-alignment
    scripts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .model_utils import (
    all_key_vectors,
    all_value_vectors,
    get_embedding,
    get_key_vector,
    get_mlp_matrices,
    get_unembedding,
    get_value_vector,
    model_info,
    resolve_device,
)

__all__ = [
    # constants
    "DEFAULT_TOP_N",
    "DEFAULT_TOP_K_TOKENS",
    "TOXIC_INDEX",
    "ARTIFACT_DIR",
    "TOXIC_VECTORS_PT",
    "TOXIC_VECTORS_JSON",
    "TABLE1_VECTORS",
    "TABLE1_EXPECTED",
    # dataclasses
    "ValueVectorRanking",
    "ToxicVectors",
    "VocabProjection",
    "SVDResult",
    # ranking / extraction
    "cosine_similarity_rows",
    "cosine_similarity_to_direction",
    "resolve_toxic_direction",
    "rank_value_vectors_by_cosine",
    "extract_toxic_indices",
    "svd_decompose",
    "extract_toxic_vectors",
    "top_k_indices",
    "group_indices_by_layer",
    # vocabulary space
    "vector_vocab_projection",
    "batch_vocab_projections",
    "top_tokens_for_vectors",
    "bottom_tokens_for_vectors",
    "project_value_vectors",
    "project_svd_vectors",
    "project_direction",
    "table1_projections",
    "check_table1_tokens",
    "promotion_sign",
    # persistence
    "default_paths",
    "save_toxic_vectors",
    "load_toxic_vectors",
    "toxic_vectors_exist",
    "load_toxic_artifact",
    "extract_and_save",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_TOP_N = 128          # Section 3.1: N (=128) toxic value vectors
DEFAULT_TOP_K_TOKENS = 10    # tokens reported per vector (Table 1 shows ~6-7)
TOXIC_INDEX = 1              # column of W_Toxic for the toxic class
NON_TOXIC_INDEX = 0

ARTIFACT_DIR = "artifacts/vectors"
TOXIC_VECTORS_PT = "toxic_vectors.pt"
TOXIC_VECTORS_JSON = "toxic_vectors.json"

# Vectors referenced in Table 1 of the paper, used for validation.
TABLE1_VECTORS: Dict[str, Optional[Union[Tuple[int, int], int]]] = {
    "W_Toxic": None,
    "MLP.v_770^19": (19, 770),
    "MLP.v_771^12": (12, 771),
    "MLP.v_2669^18": (18, 2669),
    "MLP.v_668^13": (13, 668),
    "MLP.v_255^16": (16, 255),
    "MLP.v_882^12": (12, 882),
    "MLP.v_1438^19": (19, 1438),
    "SVD.U_Toxic[0]": 0,
    "SVD.U_Toxic[1]": 1,
    "SVD.U_Toxic[2]": 2,
}

# Expected top tokens (Table 1).  The paper censors offensive tokens with '*';
# matching is therefore substring/censoring tolerant (see `_token_matches`).
TABLE1_EXPECTED: Dict[str, List[str]] = {
    "W_Toxic": ["c*nt", "f*ck", "a**hole", "d*ck", "wh*re", "holes"],
    "MLP.v_770^19": ["sh*t", "a**", "cr*p", "f*ck", "c*nt", "garbage", "trash"],
    "MLP.v_771^12": ["delusional", "hypocritical", "arrogant", "nonsense"],
    "MLP.v_2669^18": ["degener", "whining", "idiots", "stupid", "smug"],
    "MLP.v_668^13": ["losers", "filthy", "disgr", "gad", "feces", "apes", "thous"],
    "MLP.v_255^16": ["disgrace", "shameful", "coward", "unacceptable"],
    "MLP.v_882^12": ["f*ck", "sh*t", "piss", "hilar", "stupidity", "poop"],
    "MLP.v_1438^19": ["c*m", "c*ck", "orgasm", "missionary", "anal"],
    "SVD.U_Toxic[0]": ["a**", "losers", "d*ck", "s*ck", "balls", "jack", "sh*t"],
    "SVD.U_Toxic[1]": ["sexually", "intercourse", "missive", "rogens", "nude"],
    "SVD.U_Toxic[2]": ["sex", "breasts", "girlfriends", "vagina", "boobs"],
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    """Convert torch tensors / lists / arrays to a detached numpy array."""
    if x is None:
        return np.zeros(0, dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    try:  # torch tensor
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    except AttributeError:
        return np.asarray(x, dtype=dtype)


def _to_torch(x: Any, device: Optional[Any] = None):
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if device is not None:
        t = t.to(device)
    return t


def _as_index(idx: Union[Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    layer, i = int(idx[0]), int(idx[1])
    return layer, i


def _index_key(idx: Union[Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    return _as_index(idx)


def _json_default(obj: Any):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    try:  # torch tensor
        return obj.detach().cpu().numpy().tolist()
    except AttributeError:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ValueVectorRanking:
    """Cosine similarity ranking of every MLP value vector vs a direction."""

    indices: List[Tuple[int, int]]
    cosines: np.ndarray
    all_cosines: np.ndarray
    all_indices: List[Tuple[int, int]]
    direction_norm: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)

    def top(self, n: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        n = int(min(max(n, 0), len(self.indices)))
        return self.indices[:n], np.asarray(self.cosines)[:n]

    def layers(self) -> List[int]:
        return [int(l) for l, _ in self.indices]

    def layer_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for layer, _ in self.indices:
            counts[layer] = counts.get(layer, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indices": [[int(l), int(i)] for l, i in self.indices],
            "cosines": [float(c) for c in np.asarray(self.cosines)],
            "direction_norm": float(self.direction_norm),
            "top_n": len(self.indices),
            "n_value_vectors": len(self.all_indices),
            "meta": dict(self.meta),
        }


@dataclass
class SVDResult:
    """SVD of the stacked toxic value vectors.

    ``u`` has shape ``[d_model, n_components]`` (columns are the
    ``SVD.U_Toxic[i]`` vectors, as the paper intends); ``vectors`` exposes the
    same information transposed to ``[n_components, d_model]`` for convenience.
    """

    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray
    n_input_vectors: int
    centered: bool = False
    transposed: bool = True

    @property
    def vectors(self) -> np.ndarray:
        return np.asarray(self.u).T  # [n_components, d_model]

    @property
    def singular_values(self) -> np.ndarray:
        return np.asarray(self.s)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        s = np.asarray(self.s).astype(np.float64) ** 2
        total = float(s.sum())
        if total <= 0:
            return np.zeros_like(s)
        return s / total

    def vector(self, i: int) -> np.ndarray:
        return np.asarray(self.u)[:, int(i)]

    def top(self, k: int = 3) -> np.ndarray:
        return self.vectors[: int(k)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "svd_u": np.asarray(self.u),
            "singular_values": np.asarray(self.s),
            "n_input_vectors": int(self.n_input_vectors),
            "centered": bool(self.centered),
            "transposed": bool(self.transposed),
        }


@dataclass
class ToxicVectors:
    """Container for ``MLP.v_Toxic``, ``MLP.k_Toxic`` and ``SVD.U_Toxic``.

    Attributes
    ----------
    indices : the ``(layer, idx)`` of each selected toxic value vector, ordered
        by descending cosine similarity with ``W_Toxic[:, 1]``.
    value_vectors : ``[N, d_model]`` rows are ``MLP.v_Toxic[i]``.
    key_vectors : ``[N, d_model]`` rows are the matching ``MLP.k_Toxic[i]``.
    svd_u : ``[N, d_model]`` rows are ``SVD.U_Toxic[i]`` (ordered by descending
        singular value).  Internally the SVD was computed on the ``d x N``
        transpose, so ``svd_u = U.T``.
    """

    indices: List[Tuple[int, int]]
    value_vectors: np.ndarray
    key_vectors: np.ndarray
    svd_u: Optional[np.ndarray] = None
    singular_values: Optional[np.ndarray] = None
    cosines: Optional[np.ndarray] = None
    model_name: str = "gpt2"
    d_model: int = 0
    n_layers: int = 0
    top_n: int = DEFAULT_TOP_N
    centered_svd: bool = False
    svd_u_raw: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- basic access ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.indices)

    @property
    def n_vectors(self) -> int:
        return len(self.indices)

    def index(self, i: int) -> Tuple[int, int]:
        return _as_index(self.indices[int(i)])

    def value(self, i: int) -> np.ndarray:
        return np.asarray(self.value_vectors)[int(i)]

    def key(self, i: int) -> np.ndarray:
        return np.asarray(self.key_vectors)[int(i)]

    def svd_vector(self, i: int) -> np.ndarray:
        if self.svd_u is None:
            raise ValueError("ToxicVectors has no SVD basis (compute_svd=False)")
        return np.asarray(self.svd_u)[int(i)]

    def svd_count(self) -> int:
        return 0 if self.svd_u is None else int(np.asarray(self.svd_u).shape[0])

    # -- selection helpers ------------------------------------------------- #
    def top_indices(self, k: int = 7) -> List[Tuple[int, int]]:
        """First ``k`` toxic value-vector indices (highest cosine similarity)."""
        return [self.index(i) for i in range(min(int(k), self.n_vectors))]

    def top_layers(self, k: int = 7) -> List[int]:
        return [int(l) for l, _ in self.top_indices(k)]

    def layer_of(self, i: int) -> int:
        return self.index(i)[0]

    def find(self, layer: int, idx: int) -> Optional[int]:
        """Position of ``(layer, idx)`` in the selected set, if present."""
        for pos, (l, i) in enumerate(self.indices):
            if int(l) == int(layer) and int(i) == int(idx):
                return pos
        return None

    def get_by_index(self, layer: int, idx: int) -> Optional[np.ndarray]:
        pos = self.find(layer, idx)
        return None if pos is None else self.value(pos)

    def layer_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for layer, _ in self.indices:
            counts[int(layer)] = counts.get(int(layer), 0) + 1
        return counts

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "indices": [[int(l), int(i)] for l, i in self.indices],
            "key_indices": [[int(l), int(i)] for l, i in self.indices],
            "model_name": self.model_name,
            "d_model": int(self.d_model),
            "n_layers": int(self.n_layers),
            "top_n": int(self.top_n),
            "centered_svd": bool(self.centered_svd),
            "n_vectors": self.n_vectors,
            "meta": dict(self.meta),
        }
        if self.cosines is not None:
            d["cosines"] = [float(c) for c in np.asarray(self.cosines)]
        if self.singular_values is not None:
            d["singular_values"] = [float(s) for s in np.asarray(self.singular_values)]
        if include_arrays:
            d["value_vectors"] = np.asarray(self.value_vectors)
            d["key_vectors"] = np.asarray(self.key_vectors)
            if self.svd_u is not None:
                d["svd_u"] = np.asarray(self.svd_u)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToxicVectors":
        data = dict(data)
        raw = data.pop("raw", None)
        if raw is not None and not data:
            data = dict(raw)
        indices = [tuple(int(x) for x in pair) for pair in data.get("indices", [])]
        value_vectors = data.get("value_vectors", data.get("value", None))
        key_vectors = data.get("key_vectors", data.get("key", None))
        svd_u = data.get("svd_u", data.get("svd", None))
        meta = data.get("meta", {}) or {}
        if value_vectors is None:
            value_vectors = np.zeros((len(indices), 0), dtype=np.float32)
        return cls(
            indices=indices,
            value_vectors=_to_numpy(value_vectors),
            key_vectors=_to_numpy(key_vectors) if key_vectors is not None else np.zeros_like(
                _to_numpy(value_vectors)
            ),
            svd_u=None if svd_u is None else _to_numpy(svd_u),
            singular_values=(
                None if data.get("singular_values") is None
                else _to_numpy(data["singular_values"]).astype(np.float64)
            ),
            cosines=(
                None if data.get("cosines") is None else _to_numpy(data["cosines"])
            ),
            model_name=data.get("model_name", "gpt2"),
            d_model=int(data.get("d_model", 0) or 0),
            n_layers=int(data.get("n_layers", 0) or 0),
            top_n=int(data.get("top_n", len(indices)) or len(indices)),
            centered_svd=bool(data.get("centered_svd", False)),
            meta=meta,
        )


@dataclass
class VocabProjection:
    """Vocabulary-space projections ``r = E v`` for a set of vectors."""

    labels: List[str]
    token_ids: np.ndarray          # [n_vectors, k]
    token_strings: List[List[str]]
    scores: np.ndarray             # [n_vectors, k]
    top_k: int
    suppressed_token_ids: Optional[np.ndarray] = None
    suppressed_token_strings: Optional[List[List[str]]] = None
    suppressed_scores: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def tokens(self, i: int) -> List[str]:
        return list(self.token_strings[int(i)])

    def scores_for(self, i: int) -> List[float]:
        return [float(s) for s in np.asarray(self.scores)[int(i)]]

    def get(self, label: str) -> Optional[List[str]]:
        if label in self.labels:
            return self.tokens(self.labels.index(label))
        return None

    def as_token_lists(self) -> Dict[str, List[str]]:
        return {lab: self.tokens(i) for i, lab in enumerate(self.labels)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "token_ids": np.asarray(self.token_ids).tolist(),
            "token_strings": [list(t) for t in self.token_strings],
            "scores": np.asarray(self.scores).tolist(),
            "top_k": int(self.top_k),
            "suppressed_token_strings": (
                None if self.suppressed_token_strings is None
                else [list(t) for t in self.suppressed_token_strings]
            ),
            "meta": dict(self.meta),
        }


# --------------------------------------------------------------------------- #
# Cosine similarity ranking
# --------------------------------------------------------------------------- #


def cosine_similarity_rows(
    vectors: np.ndarray, direction: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Row-wise cosine similarity between ``vectors`` and ``direction``."""
    v = np.asarray(_to_numpy(vectors), dtype=np.float64)
    d = np.asarray(_to_numpy(direction), dtype=np.float64).reshape(-1)
    v_norm = np.linalg.norm(v, axis=-1)
    d_norm = float(np.linalg.norm(d))
    denom = np.maximum(v_norm * d_norm, eps)
    return (v @ d) / denom


def cosine_similarity_to_direction(vector: Any, direction: Any, eps: float = 1e-12) -> float:
    """Cosine similarity between a single vector and a direction."""
    a = np.asarray(_to_numpy(vector), dtype=np.float64).reshape(-1)
    b = np.asarray(_to_numpy(direction), dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def resolve_toxic_direction(
    probe: Any = None,
    direction: Any = None,
    probe_path: Optional[str] = None,
    toxic_index: int = TOXIC_INDEX,
) -> np.ndarray:
    """Resolve the ``W_Toxic[:, 1]`` direction from a probe, tensor or path.

    Accepts (in order of priority) an explicit ``direction`` vector, a
    ``src.probe.ToxicityProbe`` (or anything exposing ``toxic_direction``), a
    ``[d_model, 2]`` weight tensor, or ``None`` (load the saved probe artifact,
    training it lazily through ``src.probe.load_or_train_probe`` if needed).
    """
    if direction is not None:
        return np.asarray(_to_numpy(direction), dtype=np.float32).reshape(-1)

    if probe is not None:
        if hasattr(probe, "toxic_direction"):
            return np.asarray(_to_numpy(probe.toxic_direction), dtype=np.float32)
        if hasattr(probe, "W"):
            W = np.asarray(_to_numpy(probe.W), dtype=np.float32)
            if W.ndim == 2 and W.shape[0] == 2 and W.shape[1] != 2:
                W = W.T  # tolerate [2, d] storage
            if W.ndim == 2:
                return W[:, int(toxic_index)]
            return W.reshape(-1)
        arr = np.asarray(_to_numpy(probe), dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] == 2 and arr.shape[1] != 2:
                arr = arr.T
            return arr[:, int(toxic_index)]
        return arr.reshape(-1)

    # Fall back to the saved (or freshly trained) probe artifact.
    try:
        from .probe import load_or_train_probe

        loaded = load_or_train_probe(probe_path) if probe_path else load_or_train_probe()
        return np.asarray(_to_numpy(loaded.toxic_direction), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - depends on artifacts
        raise RuntimeError(
            "Could not resolve W_Toxic[:, 1]; pass `probe=`/`direction=` or train "
            "the probe first (scripts/train_probe.py)."
        ) from exc


def rank_value_vectors_by_cosine(
    model,
    direction: Any,
    layers: Optional[Sequence[int]] = None,
    value_vectors: Optional[Any] = None,
    indices: Optional[Sequence[Tuple[int, int]]] = None,
    top_n: Optional[int] = None,
) -> ValueVectorRanking:
    """Rank every MLP value vector by cosine similarity with ``direction``.

    Section 3.1: "we search for value vectors that promote toxicity, by checking
    for all value vectors with the highest cosine similarity with
    ``W_Toxic``" -- using column 1 of the probe matrix.

    Returns the ranking sorted by descending cosine similarity (the first
    ``top_n`` entries are ``MLP.v_Toxic``).
    """
    if value_vectors is None or indices is None:
        stacked, all_indices = all_value_vectors(model, layers=layers)
        stacked = _to_numpy(stacked)
        all_indices = [tuple(int(x) for x in ix) for ix in all_indices]
    else:
        stacked = _to_numpy(value_vectors)
        all_indices = [tuple(int(x) for x in ix) for ix in indices]

    d = np.asarray(_to_numpy(direction), dtype=np.float64).reshape(-1)
    cosines = cosine_similarity_rows(stacked, d)

    order = np.argsort(-cosines)
    n = len(order) if top_n is None else int(min(max(int(top_n), 0), len(order)))
    # Use a stable sort so equal cosines keep layer/idx order for determinism.
    order = np.argsort(-cosines, kind="stable")

    selected = [all_indices[int(j)] for j in order[:n]]
    return ValueVectorRanking(
        indices=selected,
        cosines=cosines[order[:n]],
        all_cosines=cosines,
        all_indices=all_indices,
        direction_norm=float(np.linalg.norm(d)),
        meta={
            "n_value_vectors": len(all_indices),
            "n_selected": n,
            "d_model": int(stacked.shape[1]) if stacked.ndim == 2 else 0,
            "layers": sorted(set(int(l) for l, _ in all_indices)),
        },
    )


def extract_toxic_indices(
    model,
    probe: Any = None,
    direction: Any = None,
    top_n: int = DEFAULT_TOP_N,
    layers: Optional[Sequence[int]] = None,
    probe_path: Optional[str] = None,
) -> ValueVectorRanking:
    """Convenience wrapper: resolve the toxic direction then rank value vectors."""
    d = resolve_toxic_direction(probe=probe, direction=direction, probe_path=probe_path)
    return rank_value_vectors_by_cosine(model, d, layers=layers, top_n=top_n)


def top_k_indices(indices: Sequence[Tuple[int, int]], k: int = 7) -> List[Tuple[int, int]]:
    """First ``k`` ``(layer, idx)`` pairs of an already-ranked index list."""
    return [tuple(int(x) for x in ix) for ix in list(indices)[: int(k)]]


def group_indices_by_layer(
    indices: Sequence[Tuple[int, int]]
) -> Dict[int, List[int]]:
    """Group ``(layer, idx)`` pairs into ``{layer: [idx, ...]}``."""
    out: Dict[int, List[int]] = {}
    for layer, idx in indices:
        out.setdefault(int(layer), []).append(int(idx))
    return out


def gather_key_vectors(
    model, indices: Sequence[Tuple[int, int]], layers: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Fetch key vectors matching ``indices`` preserving the requested order."""
    wanted = [tuple(int(x) for x in ix) for ix in indices]
    if not wanted:
        return np.zeros((0, 0), dtype=np.float32)

    lookup: Dict[Tuple[int, int], np.ndarray] = {}
    try:
        stacked, k_indices = all_key_vectors(model, layers=layers)
        stacked = _to_numpy(stacked)
        for row, ix in zip(stacked, k_indices):
            lookup[(int(ix[0]), int(ix[1]))] = row
    except Exception:  # pragma: no cover - fallback path
        lookup = {}

    rows: List[np.ndarray] = []
    for layer, idx in wanted:
        row = lookup.get((layer, idx))
        if row is None:
            row = _to_numpy(get_key_vector(model, layer, idx))
        rows.append(np.asarray(row, dtype=np.float32))
    return np.stack(rows, axis=0)


# --------------------------------------------------------------------------- #
# SVD decomposition
# --------------------------------------------------------------------------- #


def svd_decompose(
    value_vectors: Any,
    n_components: Optional[int] = None,
    center: bool = False,
    transpose: bool = True,
) -> SVDResult:
    """SVD of the stacked toxic value vectors.

    Section 3.1 (as corrected by the author clarification): the ``N x d`` matrix
    of stacked value vectors is *transposed* to ``d x N`` before the SVD, so the
    ``U`` matrix is ``d x d`` (or ``d x N`` with ``full_matrices=False``) and its
    columns -- the ``SVD.U_Toxic[i]`` vectors -- are ``d``-dimensional.

    ``center=True`` optionally removes the mean value vector before the SVD
    (not used by the paper; kept for ablations).
    """
    M = np.asarray(_to_numpy(value_vectors), dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"value_vectors must be 2-D, got shape {M.shape}")
    n_input = int(M.shape[0])

    if center:
        M = M - M.mean(axis=0, keepdims=True)
    if transpose:
        M = M.T  # [d_model, N]

    k = None
    if n_components is not None:
        k = int(min(max(int(n_components), 1), min(M.shape)))
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    if k is not None:
        u, s, vt = u[:, :k], s[:k], vt[:k]

    return SVDResult(
        u=np.asarray(u, dtype=np.float32),
        s=np.asarray(s, dtype=np.float64),
        vt=np.asarray(vt, dtype=np.float32),
        n_input_vectors=n_input,
        centered=bool(center),
        transposed=bool(transpose),
    )


# --------------------------------------------------------------------------- #
# Full extraction
# --------------------------------------------------------------------------- #


def extract_toxic_vectors(
    model,
    probe: Any = None,
    direction: Any = None,
    tokenizer: Any = None,
    top_n: int = DEFAULT_TOP_N,
    layers: Optional[Sequence[int]] = None,
    compute_svd: bool = True,
    n_components: Optional[int] = None,
    center: bool = False,
    probe_path: Optional[str] = None,
    model_name: str = "gpt2",
    top_k_tokens: Optional[int] = None,
    verbose: bool = False,
) -> ToxicVectors:
    """Extract ``MLP.v_Toxic``, ``MLP.k_Toxic`` and ``SVD.U_Toxic``.

    Steps (Section 3.1):
      1. cosine-similarity rank all MLP value vectors against ``W_Toxic[:, 1]``;
      2. keep the top ``N`` as ``MLP.v_Toxic`` with matching ``MLP.k_Toxic``;
      3. stack them into an ``N x d`` matrix and SVD the ``d x N`` transpose to
         obtain the ``SVD.U_Toxic`` basis.
    """
    d = resolve_toxic_direction(
        probe=probe, direction=direction, probe_path=probe_path
    )
    ranking = rank_value_vectors_by_cosine(model, d, layers=layers, top_n=top_n)
    ranking.meta["direction_norm"] = float(np.linalg.norm(d))

    # Value vectors: gather rows from the full stack to guarantee consistency.
    stacked, all_indices = all_value_vectors(model, layers=layers)
    stacked = _to_numpy(stacked)
    vlookup = {
        (int(ix[0]), int(ix[1])): row for row, ix in zip(stacked, all_indices)
    }
    value_rows: List[np.ndarray] = []
    for layer, idx in ranking.indices:
        row = vlookup.get((int(layer), int(idx)))
        if row is None:
            row = _to_numpy(get_value_vector(model, layer, idx))
        value_rows.append(np.asarray(row, dtype=np.float32))
    value_vectors = (
        np.stack(value_rows, axis=0) if value_rows else np.zeros((0, 0), np.float32)
    )
    key_vectors = gather_key_vectors(model, ranking.indices, layers=layers)

    svd_u = None
    singular_values = None
    svd_u_raw = None
    centered_svd = bool(center)
    if compute_svd and value_vectors.shape[0] > 0:
        svd = svd_decompose(
            value_vectors, n_components=n_components, center=center, transpose=True
        )
        svd_u = svd.vectors  # [n_components, d_model]
        singular_values = svd.singular_values
        svd_u_raw = svd.u    # [d_model, n_components]
        if verbose:
            print(
                f"[toxic_vectors] SVD on transposed {value_vectors.shape[1]} x "
                f"{value_vectors.shape[0]} matrix -> U {svd.u.shape}"
            )

    try:
        info = model_info(model, name=model_name)
        d_model, n_layers = int(info.d_model), int(info.n_layers)
    except Exception:  # pragma: no cover
        d_model = int(value_vectors.shape[1]) if value_vectors.ndim == 2 else 0
        n_layers = 0

    tv = ToxicVectors(
        indices=list(ranking.indices),
        value_vectors=value_vectors,
        key_vectors=key_vectors,
        svd_u=svd_u,
        singular_values=singular_values,
        cosines=np.asarray(ranking.cosines, dtype=np.float32),
        model_name=model_name,
        d_model=d_model,
        n_layers=n_layers,
        top_n=int(top_n),
        centered_svd=centered_svd,
        svd_u_raw=svd_u_raw,
        meta={
            "n_value_vectors": len(ranking.all_indices),
            "toxic_direction_norm": float(np.linalg.norm(d)),
            "svd_on_transpose": True,
            "svd_explained_variance_ratio": (
                None if singular_values is None
                else [float(x) for x in (np.asarray(singular_values, dtype=np.float64) ** 2)
                      / max(float((np.asarray(singular_values, dtype=np.float64) ** 2).sum()), 1e-12)]
            ),
            "layer_counts": {str(k): v for k, v in ranking.layer_counts().items()},
            "top_k_tokens": top_k_tokens,
        },
    )
    return tv


# --------------------------------------------------------------------------- #
# Vocabulary-space projections (Section 3.2 / Appendix A)
# --------------------------------------------------------------------------- #


def _embedding_matrix(model, use_unembedding: bool = True) -> np.ndarray:
    """Tied GPT2 embedding/unembedding matrix ``E`` of shape ``[vocab, d]``."""
    if use_unembedding:
        try:
            return _to_numpy(get_unembedding(model), dtype=np.float32)
        except Exception:  # pragma: no cover
            pass
    return _to_numpy(get_embedding(model), dtype=np.float32)


def vector_vocab_projection(model, vector: Any, use_unembedding: bool = True) -> np.ndarray:
    """``r = E v``: the static projection of one value vector onto the vocabulary."""
    E = _embedding_matrix(model, use_unembedding=use_unembedding)
    v = np.asarray(_to_numpy(vector), dtype=np.float64).reshape(-1)
    return (E.astype(np.float64) @ v).astype(np.float32)


def batch_vocab_projections(
    model, vectors: Any, use_unembedding: bool = True
) -> np.ndarray:
    """``r = V E^T`` for a stack of vectors ``V`` of shape ``[n, d]``."""
    E = _embedding_matrix(model, use_unembedding=use_unembedding).astype(np.float64)
    V = np.asarray(_to_numpy(vectors), dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    return (V @ E.T).astype(np.float32)


def _decode_tokens(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    if tokenizer is None:
        return [str(int(i)) for i in ids]
    toks: List[str] = []
    for i in ids:
        try:
            toks.append(tokenizer.decode([int(i)], clean_up_tokenization_spaces=False))
        except Exception:  # pragma: no cover
            try:
                toks.append(tokenizer.convert_ids_to_tokens(int(i)))
            except Exception:
                toks.append(str(int(i)))
    return toks


def top_tokens_for_vectors(
    model,
    vectors: Any,
    tokenizer: Any = None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    use_unembedding: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """Top tokens promoted by each vector: highest ``e_w . v`` dot products.

    Returns ``(token_ids [n, k], scores [n, k], token_strings)``.
    """
    proj = batch_vocab_projections(model, vectors, use_unembedding=use_unembedding)
    k = int(min(max(int(top_k), 1), proj.shape[1]))
    ids = np.argpartition(-proj, k - 1, axis=1)[:, :k]
    rows = np.arange(proj.shape[0])[:, None]
    order = np.argsort(-proj[rows, ids], axis=1)
    ids = ids[rows, order]
    scores = proj[rows, ids]
    strings = [_decode_tokens(tokenizer, row) for row in ids]
    return ids, scores, strings


def bottom_tokens_for_vectors(
    model,
    vectors: Any,
    tokenizer: Any = None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    use_unembedding: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """Tokens most suppressed by each vector (lowest ``e_w . v`` dot products)."""
    proj = batch_vocab_projections(model, vectors, use_unembedding=use_unembedding)
    k = int(min(max(int(top_k), 1), proj.shape[1]))
    ids = np.argpartition(proj, k - 1, axis=1)[:, :k]
    rows = np.arange(proj.shape[0])[:, None]
    order = np.argsort(proj[rows, ids], axis=1)
    ids = ids[rows, order]
    scores = proj[rows, ids]
    strings = [_decode_tokens(tokenizer, row) for row in ids]
    return ids, scores, strings


def project_value_vectors(
    model,
    toxic_vectors: Union[ToxicVectors, Any],
    tokenizer: Any = None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    with_suppressed: bool = False,
    labels: Optional[Sequence[str]] = None,
) -> VocabProjection:
    """Project ``MLP.v_Toxic`` onto the vocabulary space (Table 1, rows 2-8)."""
    if isinstance(toxic_vectors, ToxicVectors):
        vectors = toxic_vectors.value_vectors
        default_labels = [_format_mlp_label(*ix) for ix in toxic_vectors.indices]
    else:
        vectors = _to_numpy(toxic_vectors)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        default_labels = [f"MLP.v[{i}]" for i in range(vectors.shape[0])]

    ids, scores, strings = top_tokens_for_vectors(
        model, vectors, tokenizer=tokenizer, top_k=top_k
    )
    sup_ids = sup_scores = None
    sup_strings = None
    if with_suppressed:
        sup_ids, sup_scores, sup_strings = bottom_tokens_for_vectors(
            model, vectors, tokenizer=tokenizer, top_k=top_k
        )
    return VocabProjection(
        labels=list(labels) if labels is not None else default_labels,
        token_ids=ids,
        token_strings=strings,
        scores=scores,
        top_k=int(top_k),
        suppressed_token_ids=sup_ids,
        suppressed_token_strings=sup_strings,
        suppressed_scores=sup_scores,
        meta={"kind": "MLP.v_Toxic"},
    )


def project_svd_vectors(
    model,
    toxic_vectors: ToxicVectors,
    tokenizer: Any = None,
    n: int = 3,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    with_suppressed: bool = False,
) -> VocabProjection:
    """Project the first ``n`` ``SVD.U_Toxic`` vectors (Table 1, rows 9-11)."""
    if toxic_vectors.svd_u is None:
        raise ValueError("ToxicVectors has no SVD basis; run with compute_svd=True")
    vectors = np.asarray(toxic_vectors.svd_u)[: int(n)]
    ids, scores, strings = top_tokens_for_vectors(
        model, vectors, tokenizer=tokenizer, top_k=top_k
    )
    sup_ids = sup_scores = None
    sup_strings = None
    if with_suppressed:
        sup_ids, sup_scores, sup_strings = bottom_tokens_for_vectors(
            model, vectors, tokenizer=tokenizer, top_k=top_k
        )
    return VocabProjection(
        labels=[f"SVD.U_Toxic[{i}]" for i in range(vectors.shape[0])],
        token_ids=ids,
        token_strings=strings,
        scores=scores,
        top_k=int(top_k),
        suppressed_token_ids=sup_ids,
        suppressed_token_strings=sup_strings,
        suppressed_scores=sup_scores,
        meta={"kind": "SVD.U_Toxic"},
    )


def project_direction(
    model,
    direction: Any,
    tokenizer: Any = None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    label: str = "W_Toxic",
    with_suppressed: bool = False,
) -> VocabProjection:
    """Project ``W_Toxic[:, 1]`` onto the vocabulary space (Table 1, row 1)."""
    v = np.asarray(_to_numpy(direction), dtype=np.float32).reshape(1, -1)
    ids, scores, strings = top_tokens_for_vectors(
        model, v, tokenizer=tokenizer, top_k=top_k
    )
    sup = None
    if with_suppressed:
        sup = bottom_tokens_for_vectors(model, v, tokenizer=tokenizer, top_k=top_k)
    return VocabProjection(
        labels=[label],
        token_ids=ids,
        token_strings=strings,
        scores=scores,
        top_k=int(top_k),
        suppressed_token_ids=None if sup is None else sup[0],
        suppressed_token_strings=None if sup is None else sup[2],
        suppressed_scores=None if sup is None else sup[1],
        meta={"kind": "W_Toxic"},
    )


def _format_mlp_label(layer: int, idx: int) -> str:
    return f"MLP.v_{idx}^{layer}"


def table1_projections(
    model,
    toxic_vectors: Optional[ToxicVectors] = None,
    tokenizer: Any = None,
    probe: Any = None,
    direction: Any = None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    max_mlp_vectors: Optional[int] = None,
) -> Dict[str, VocabProjection]:
    """Reproduce the Table 1 panels available in the extracted artifacts.

    Returns a mapping ``label -> VocabProjection`` covering ``W_Toxic``, the
    extracted ``MLP.v_Toxic`` rows (optionally limited) and
    ``SVD.U_Toxic[0..2]``.
    """
    out: Dict[str, VocabProjection] = {}

    # W_Toxic row
    try:
        d = resolve_toxic_direction(probe=probe, direction=direction)
        out["W_Toxic"] = project_direction(model, d, tokenizer=tokenizer, top_k=top_k)
    except Exception as exc:  # pragma: no cover
        print(f"[toxic_vectors] skipping W_Toxic projection: {exc}")

    if toxic_vectors is not None:
        proj = project_value_vectors(
            model, toxic_vectors, tokenizer=tokenizer, top_k=top_k
        )
        n = len(proj) if max_mlp_vectors is None else int(max_mlp_vectors)
        for i, label in enumerate(proj.labels[:n]):
            out[label] = VocabProjection(
                labels=[label],
                token_ids=proj.token_ids[i : i + 1],
                token_strings=[proj.token_strings[i]],
                scores=proj.scores[i : i + 1],
                top_k=top_k,
                meta=dict(proj.meta),
            )
        if toxic_vectors.svd_u is not None:
            svd_proj = project_svd_vectors(
                model, toxic_vectors, tokenizer=tokenizer, n=3, top_k=top_k
            )
            for i, label in enumerate(svd_proj.labels):
                out[label] = VocabProjection(
                    labels=[label],
                    token_ids=svd_proj.token_ids[i : i + 1],
                    token_strings=[svd_proj.token_strings[i]],
                    scores=svd_proj.scores[i : i + 1],
                    top_k=top_k,
                    meta=dict(svd_proj.meta),
                )
    return out


# -- Table 1 validation ------------------------------------------------------ #

_VOWELS = set("aeiou")


def _censor_variants(token: str) -> List[str]:
    """Generate censored spellings used by the paper (e.g. ``shit`` -> ``sh*t``)."""
    t = token.strip().lower()
    if not t:
        return []
    variants = {t}
    # replace internal vowels with '*', keeping the first letter
    for i in range(1, len(t) - 1):
        if t[i] in _VOWELS:
            variants.add(t[:i] + "*" + t[i + 1 :])
    for i in range(1, len(t) - 1):
        if t[i] in _VOWELS:
            for j in range(i + 1, len(t) - 1):
                if t[j] in _VOWELS:
                    variants.add(t[:i] + "*" + t[i + 1 : j] + "*" + t[j + 1 :])
    # two letter trailing censor, e.g. "ass" -> "a**"
    if len(t) >= 3:
        variants.add(t[0] + "*" * (len(t) - 1))
    return sorted(variants)


def _token_matches(token: str, expected: str, prefix_len: int = 4) -> bool:
    """Censoring/prefix tolerant comparison of a decoded token and an expected token."""
    tok = (token or "").strip().lower().replace("Ġ", " ").strip()
    exp = (expected or "").strip().lower()
    if not tok or not exp:
        return False
    if exp in tok or tok in exp:
        return True
    if "*" in exp:
        # Compare the star-free prefix (e.g. "sh*t" -> "sh", "d*ck" -> "d")
        head = exp.split("*")[0]
        if len(head) >= 2 and tok.startswith(head):
            return True
        # Compare consonant skeleton ignoring '*' (e.g. "f*ck" -> "fck")
        skel_exp = exp.replace("*", "")
        skel_tok = "".join(c for c in tok if c.isalpha())
        if skel_exp and skel_exp in skel_tok:
            return True
        return False
    n = min(prefix_len, len(exp))
    return tok[:n] == exp[:n]


def check_table1_tokens(
    projections: Dict[str, Union[VocabProjection, Sequence[str]]],
    expected: Optional[Dict[str, List[str]]] = None,
    prefix_len: int = 4,
) -> Dict[str, Any]:
    """Validate the vocabulary projections against the paper's Table 1 groups."""
    expected = expected or TABLE1_EXPECTED
    report: Dict[str, Any] = {}
    for label, exp_tokens in expected.items():
        proj = projections.get(label)
        if proj is None:
            report[label] = {"present": False, "top_tokens": [], "matched": [], "coverage": 0.0}
            continue
        if isinstance(proj, VocabProjection):
            toks = proj.tokens(0)
        else:
            toks = list(proj)
        matched = [
            e for e in exp_tokens
            if any(_token_matches(t, e, prefix_len=prefix_len) for t in toks)
        ]
        report[label] = {
            "present": True,
            "top_tokens": toks,
            "matched": matched,
            "missing": [e for e in exp_tokens if e not in matched],
            "coverage": len(matched) / max(len(exp_tokens), 1),
        }
    present = [v for v in report.values() if v.get("present")]
    report["_summary"] = {
        "n_vectors_checked": len(report) - 1,
        "n_present": len(present),
        "mean_coverage": (
            float(np.mean([v["coverage"] for v in present])) if present else 0.0
        ),
    }
    return report


def promotion_sign(m_i: float, e_w_dot_v_i: float) -> str:
    """Promotion vs suppression of token ``w`` by sub-update ``m_i v_i``.

    Appendix A: the likelihood of ``w`` increases when
    ``e_w . m_i v_i > 0`` and decreases when ``< 0``.
    """
    prod = float(m_i) * float(e_w_dot_v_i)
    if prod > 0:
        return "promote"
    if prod < 0:
        return "suppress"
    return "neutral"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def default_paths(out_dir: str = ARTIFACT_DIR) -> Dict[str, str]:
    """Default artifact paths for the toxic-vector extraction."""
    return {
        "pt": os.path.join(out_dir, TOXIC_VECTORS_PT),
        "json": os.path.join(out_dir, TOXIC_VECTORS_JSON),
        "dir": out_dir,
    }


def save_toxic_vectors(
    toxic_vectors: ToxicVectors,
    path: Optional[str] = None,
    out_dir: str = ARTIFACT_DIR,
    save_json: bool = True,
    verbose: bool = False,
) -> str:
    """Save ``ToxicVectors`` as a ``.pt`` bundle (plus JSON metadata sidecar).

    The ``.pt`` payload uses the key names consumed by
    ``scripts/analyze_dpo.py``: ``indices``, ``key_indices``, ``value_vectors``,
    ``key_vectors``, ``svd_u``.
    """
    paths = default_paths(out_dir)
    pt_path = path or paths["pt"]
    if not os.path.splitext(pt_path)[1]:
        pt_path = pt_path + ".pt"
    json_path = (
        os.path.splitext(pt_path)[0] + ".json" if path else paths["json"]
    )
    os.makedirs(os.path.dirname(os.path.abspath(pt_path)) or ".", exist_ok=True)

    payload = toxic_vectors.to_dict(include_arrays=True)
    payload["svd_u_raw"] = (
        None if toxic_vectors.svd_u_raw is None else np.asarray(toxic_vectors.svd_u_raw)
    )
    try:
        import torch

        torch.save(payload, pt_path)
    except Exception:  # pragma: no cover - numpy fallback
        np.save(pt_path.replace(".pt", ".npz"), payload, allow_pickle=True)

    if save_json:
        meta_only = toxic_vectors.to_dict(include_arrays=False)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta_only, fh, indent=2, default=_json_default)
    if verbose:
        print(f"[toxic_vectors] saved {pt_path}")
    return pt_path


def load_toxic_vectors(
    path: Optional[str] = None,
    out_dir: str = ARTIFACT_DIR,
    map_location: str = "cpu",
) -> ToxicVectors:
    """Load a ``ToxicVectors`` artifact (tolerating ``.pt``, ``.npz`` or JSON)."""
    paths = default_paths(out_dir)
    candidates = []
    if path:
        candidates.append(path)
        if not os.path.splitext(path)[1]:
            candidates += [path + ".pt", path + ".json", path.replace(".pt", ".npz")]
    candidates += [paths["pt"], paths["json"], paths["pt"].replace(".pt", ".npz")]

    last_exc: Optional[Exception] = None
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        try:
            if cand.endswith(".json"):
                with open(cand, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return ToxicVectors.from_dict(data)
            if cand.endswith(".npz"):
                with np.load(cand, allow_pickle=True) as npz:
                    data = {k: npz[k] for k in npz.files if npz[k].shape != ()}
                    scalars = {k: npz[k].item() for k in npz.files if npz[k].shape == ()}
                data.update(scalars)
                return ToxicVectors.from_dict(data)
            import torch

            data = torch.load(cand, map_location=map_location, weights_only=False)
            if not isinstance(data, dict):
                raise ValueError(f"unexpected payload type {type(data)}")
            return ToxicVectors.from_dict(data)
        except Exception as exc:  # try the next candidate
            last_exc = exc
            continue
    raise FileNotFoundError(
        f"no toxic-vector artifact found (tried {candidates}); "
        f"last error: {last_exc}"
    )


def load_toxic_artifact(path: str) -> Dict[str, Any]:
    """Tolerant loader returning a plain dict (used by analysis scripts)."""
    data: Dict[str, Any] = {}
    if path and path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif path:
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        tv = load_toxic_vectors()
        data = tv.to_dict(include_arrays=True)
        return data
    if not isinstance(data, dict):
        raise ValueError(f"unexpected artifact payload: {type(data)}")
    raw = data.get("raw") if isinstance(data.get("raw"), dict) else None
    if raw:
        merged = dict(raw)
        merged.update({k: v for k, v in data.items() if k != "raw"})
        data = merged
    out = {
        "indices": [tuple(int(x) for x in p) for p in data.get("indices", [])],
        "key_indices": [
            tuple(int(x) for x in p)
            for p in data.get("key_indices", data.get("indices", []))
        ],
        "value_vectors": data.get("value_vectors"),
        "key_vectors": data.get("key_vectors"),
        "svd_u": data.get("svd_u"),
        "raw": data,
    }
    return out


def toxic_vectors_exist(path: Optional[str] = None, out_dir: str = ARTIFACT_DIR) -> bool:
    """Whether a toxic-vector artifact is available at ``path``/default paths."""
    if path and os.path.exists(path):
        return True
    paths = default_paths(out_dir)
    return any(os.path.exists(p) for p in (paths["pt"], paths["json"]))


# --------------------------------------------------------------------------- #
# High-level pipeline
# --------------------------------------------------------------------------- #


def extract_and_save(
    model=None,
    tokenizer=None,
    model_name: str = "gpt2",
    probe: Any = None,
    probe_path: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
    layers: Optional[Sequence[int]] = None,
    out_dir: str = ARTIFACT_DIR,
    top_k_tokens: int = DEFAULT_TOP_K_TOKENS,
    device: Optional[str] = None,
    validate_table1: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end extraction: rank -> select -> SVD -> vocabulary check -> save.

    Returns a dict with the ``ToxicVectors`` object, the saved paths, the Table 1
    vocabulary projections and the validation report.
    """
    if model is None or tokenizer is None:
        from .model_utils import load_model

        model, tokenizer = load_model(model_name, device=device)

    tv = extract_toxic_vectors(
        model,
        probe=probe,
        probe_path=probe_path,
        tokenizer=tokenizer,
        top_n=top_n,
        layers=layers,
        compute_svd=True,
        model_name=model_name,
        top_k_tokens=top_k_tokens,
        verbose=verbose,
    )
    pt_path = save_toxic_vectors(tv, out_dir=out_dir, verbose=verbose)

    projections: Dict[str, VocabProjection] = {}
    report: Dict[str, Any] = {}
    if validate_table1:
        projections = table1_projections(
            model,
            toxic_vectors=tv,
            tokenizer=tokenizer,
            probe=probe,
            direction=None if probe is not None else None,
            top_k=top_k_tokens,
        )
        report = check_table1_tokens(projections)
        json_path = os.path.join(out_dir, "table1_projections.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "projections": {k: v.as_token_lists() for k, v in projections.items()},
                    "validation": report,
                },
                fh,
                indent=2,
                default=_json_default,
            )
        if verbose:
            print(f"[toxic_vectors] table 1 report -> {json_path}")
            print(f"[toxic_vectors] mean token coverage: {report['_summary']['mean_coverage']:.3f}")

    return {
        "toxic_vectors": tv,
        "path": pt_path,
        "projections": projections,
        "validation": report,
    }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Extract toxic vectors (smoke test)")
    parser.add_argument("--model", default="openai-community/gpt2-medium")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--out-dir", default=ARTIFACT_DIR)
    parser.add_argument("--quick", action="store_true", help="tiny N for a fast check")
    args = parser.parse_args()

    from .model_utils import load_model, set_seed

    set_seed(0)
    top_n = 8 if args.quick else args.top_n
    model, tokenizer = load_model(args.model)
    result = extract_and_save(
        model,
        tokenizer,
        model_name=args.model,
        top_n=top_n,
        out_dir=args.out_dir,
        validate_table1=not args.quick,
        verbose=True,
    )
    tv: ToxicVectors = result["toxic_vectors"]
    print(f"extracted {tv.n_vectors} toxic value vectors, d_model={tv.d_model}")
    print("top-5 (layer, idx) by cosine similarity with W_Toxic[:, 1]:")
    for i in range(min(5, tv.n_vectors)):
        print(f"  {tv.index(i)}  cos={float(tv.cosines[i]):.4f}")
    if tv.svd_u is not None:
        print(f"SVD.U_Toxic shape: {np.asarray(tv.svd_u).shape} (SVD of the d x N transpose)")
