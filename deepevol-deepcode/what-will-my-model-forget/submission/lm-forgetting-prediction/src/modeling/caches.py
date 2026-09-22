"""Caching layer for cheap (cache-only) forgetting forecasting.

Paper specification
-------------------
Sec. 3.2 "Efficient Inference":

    "We note that the method does not require repetitive inference with the LM f: the logits of
     pretraining examples before model updates f_0(x_j) can be computed once and cached for
     different online learning examples x_i; similarly, the representations h(x_i, y_i) required
     by the trained kernel Theta_tilde can also be cached. In practice, we only cache top k=100
     largest logits for each token in y_j."

Sec. 3.3 adds the cached frequency prior ``b_j`` for every upstream example
``<x_j, y_j> in D_PT`` (Algorithms 3 and 4, Appendix F), where

    b_j = log(|{<x_i,y_i> in D_R^train | z_ij = 1}| / |D_R^train|)
        - log(|{<x_i,y_i> in D_R^train | z_ij = 0}| / |D_R^train|)

and ``z_ij = 1[f_i(x_j) != y_j]`` (Sec. 2; note Appendix F Algorithms 1/3 write ``x_i`` in that
line which is treated here as a typo -- see ``src/forgetting/ground_truth.py``).

What is cached (all keyed by the upstream example index ``j`` into ``D_PT_hat``)
-------------------------------------------------------------------------------
1. ``LogitCache``          -> top-k (k=100) logits of ``f_0(x_j)`` per output token of ``y_j``.
2. ``RepresentationCache`` -> ``h(x_j, y_j)``: token-level ``[T, d]`` (logit-kernel mode) and
                              mean-pooled ``[d]`` (representation mode).
3. ``PriorCache``          -> frequency prior ``b_j``.

With these, forecasting a given online example over the whole ``D_PT_hat`` costs
``O(N_PT)`` cheap operations (inner products / sigmoid) and no additional PTLM forward pass at
forecast time -- exactly the "cheap forecaster" requirement of the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:  # torch is a core dependency, but keep the module importable without it.
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


logger = logging.getLogger(__name__)

DEFAULT_TOPK = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOGIT_FILENAME = "logit_cache.pt"
DEFAULT_REPR_FILENAME = "representation_cache.pt"
DEFAULT_PRIOR_FILENAME = "frequency_prior.json"

__all__ = [
    "LogitCache",
    "RepresentationCache",
    "PriorCache",
    "ForecastCache",
    "CacheBuilder",
    "build_logit_cache",
    "build_representation_cache",
    "build_prior_cache",
    "build_caches",
    "load_caches",
    "cache_paths",
    "extract_token_topk",
    "extract_representations",
    "target_of",
    "DEFAULT_TOPK",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LOGIT_FILENAME",
    "DEFAULT_REPR_FILENAME",
    "DEFAULT_PRIOR_FILENAME",
]


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def target_of(example: Mapping[str, Any]) -> str:
    """Best-effort extraction of the gold output string of one example."""
    if not isinstance(example, Mapping):
        return str(example)
    for key in ("target", "targets", "output", "label", "answer"):
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                value = value[0]
            return str(value)
    refs = example.get("references")
    if refs:
        if isinstance(refs, (list, tuple)):
            return str(refs[0])
        return str(refs)
    return ""


def _input_of(example: Mapping[str, Any]) -> str:
    if not isinstance(example, Mapping):
        return str(example)
    for key in ("input", "inputs", "question", "prompt", "text"):
        if key in example and example[key] is not None:
            return str(example[key])
    return ""


def _rows_from(container: Any) -> List[Any]:
    """Normalise the many shapes a top-k container can take into a list of rows."""
    if container is None:
        return []
    if isinstance(container, Mapping):
        for key in ("values", "logits", "topk_values", "indices", "topk_indices", "data"):
            if key in container:
                return _rows_from(container[key])
        return list(container.values())
    if isinstance(container, (list, tuple)):
        return list(container)
    # torch tensor
    return list(container)


def _to_tensor(x: Any, dtype: Any = None) -> Any:
    if not _HAS_TORCH or x is None:
        return x
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if dtype is not None else x
    try:
        t = torch.as_tensor(x)
        return t.to(dtype) if dtype is not None else t
    except Exception:  # pragma: no cover - ragged input
        return x


def _as_topk(indices: Any, values: Any) -> Any:
    """Wrap (indices, values) into the project's ``TopKLogits`` container when available."""
    try:
        from ..forgetting.ground_truth import TopKLogits  # type: ignore
    except Exception:  # pragma: no cover - standalone usage
        TopKLogits = _LocalTopKLogits  # type: ignore
    return TopKLogits(indices=indices, values=values)


class _LocalTopKLogits:
    """Fallback container with the same public surface as ``ground_truth.TopKLogits``."""

    def __init__(self, indices: Optional[Sequence[Sequence[int]]] = None,
                 values: Optional[Sequence[Sequence[float]]] = None):
        self.indices = [list(map(int, row)) for row in (indices or [])]
        self.values = [list(map(float, row)) for row in (values or [])]

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {"indices": self.indices, "values": self.values}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "_LocalTopKLogits":
        d = d or {}
        return cls(indices=d.get("indices"), values=d.get("values"))


def topk_from_logits(logits: Any, k: int = DEFAULT_TOPK) -> Any:
    """Top-k extraction from ``[T, V]`` / ``[B, T, V]`` tensors or nested lists."""
    rows = _rows_from(logits)
    if len(rows) == 0:
        return _as_topk([], [])
    if _HAS_TORCH and isinstance(logits, torch.Tensor):
        t = logits
        if t.dim() == 3:
            t = t[0]
        if t.dim() != 2:
            t = t.reshape(-1, t.shape[-1])
        kk = min(int(k), int(t.shape[-1]))
        values, indices = torch.topk(t.float(), kk, dim=-1)
        return _as_topk(indices.tolist(), values.tolist())
    # nested python lists
    idx_rows: List[List[int]] = []
    val_rows: List[List[float]] = []
    for row in rows:
        if _HAS_TORCH and isinstance(row, torch.Tensor):
            row = row.detach().float().tolist()
        if len(row) and isinstance(row[0], (list, tuple)):
            row = row[0]
        pairs = sorted(enumerate(row), key=lambda p: -float(p[1]))[:k]
        idx_rows.append([int(i) for i, _ in pairs])
        val_rows.append([float(v) for _, v in pairs])
    return _as_topk(idx_rows, val_rows)


# ---------------------------------------------------------------------------------------------
# extraction utilities (model -> cache payload)
# ---------------------------------------------------------------------------------------------
def extract_token_topk(model: Any, inputs: Sequence[str], targets: Sequence[str],
                       k: int = DEFAULT_TOPK, batch_size: int = DEFAULT_BATCH_SIZE,
                       **gen_kwargs: Any) -> List[Any]:
    """Return one ``TopKLogits`` (top-k per output token of ``y``) per input example.

    Tries the project's ``ground_truth.iter_token_logits`` helper first (it knows every accepted
    model API), then ``model.token_logits``, then ``model.hidden_states``-free last resort.
    """
    inputs = list(inputs)
    targets = list(targets)
    if len(inputs) != len(targets):
        raise ValueError("inputs/targets length mismatch: %d != %d" % (len(inputs), len(targets)))
    if len(inputs) == 0:
        return []

    try:
        from ..forgetting.ground_truth import iter_token_logits  # type: ignore

        out = list(iter_token_logits(model, inputs, targets, k=k, batch_size=batch_size))
        if len(out) == len(inputs):
            return out
    except Exception as exc:  # pragma: no cover - fall back to direct API below
        logger.debug("iter_token_logits unavailable (%s); falling back to token_logits", exc)

    out: List[Any] = []
    for start in range(0, len(inputs), max(1, batch_size)):
        bi = inputs[start:start + batch_size]
        bt = targets[start:start + batch_size]
        logits = None
        if hasattr(model, "token_logits"):
            try:
                res = model.token_logits(bi, bt, k=k, **gen_kwargs)
            except TypeError:
                res = model.token_logits(bi, bt, **gen_kwargs)
            logits = res.get("logits") if isinstance(res, Mapping) else res
        if logits is None:
            for name in ("batch_token_logits", "logits"):
                fn = getattr(model, name, None)
                if callable(fn):
                    logits = fn(bi, bt)
                    break
        if logits is None:
            raise AttributeError("model exposes no usable token-logits API")
        rows = _rows_from(logits)
        for row in rows:
            out.append(topk_from_logits(row, k=k))
    return out


def extract_representations(encoder: Any, inputs: Sequence[str], targets: Sequence[str],
                            batch_size: int = DEFAULT_BATCH_SIZE, mean_pool: bool = True,
                            token_level: bool = True, **kwargs: Any) -> List[Dict[str, Any]]:
    """Return ``[{"token": [T,d] tensor or None, "mean": [d] tensor or None}, ...]``."""
    inputs = list(inputs)
    targets = list(targets)
    if len(inputs) != len(targets):
        raise ValueError("inputs/targets length mismatch: %d != %d" % (len(inputs), len(targets)))
    if len(inputs) == 0:
        return []

    token_fn = None
    for name in ("encode_token_level", "encode_token", "token_level_encode"):
        fn = getattr(encoder, name, None)
        if callable(fn):
            token_fn = fn
            break
    mean_fn = None
    for name in ("encode_mean", "encode_pooled", "mean_encode"):
        fn = getattr(encoder, name, None)
        if callable(fn):
            mean_fn = fn
            break
    generic_fn = getattr(encoder, "encode", None) if callable(getattr(encoder, "encode", None)) else None

    records: List[Dict[str, Any]] = []
    for start in range(0, len(inputs), max(1, batch_size)):
        bi = inputs[start:start + batch_size]
        bt = targets[start:start + batch_size]
        token_out = None
        mean_out = None
        if token_level and token_fn is not None:
            token_out = token_fn(bi, bt, **kwargs)
        if mean_pool and mean_fn is not None:
            mean_out = mean_fn(bi, bt, **kwargs)
        if generic_fn is not None and (token_out is None or mean_out is None):
            try:
                generic = generic_fn(bi, bt, mean_pool=mean_pool, **kwargs)
            except TypeError:
                generic = generic_fn(bi, bt, **kwargs)
            if isinstance(generic, Mapping):
                token_out = token_out if token_out is not None else generic.get("token")
                token_out = token_out if token_out is not None else generic.get("token_level")
                mean_out = mean_out if mean_out is not None else generic.get("mean")
                mean_out = mean_out if mean_out is not None else generic.get("pooled")
            elif token_out is None:
                token_out = generic

        n = len(bi)
        token_rows = _rows_from(token_out)
        mean_rows = _rows_from(mean_out)
        if isinstance(token_out, Mapping):
            token_rows = _rows_from(token_out.get("token") or token_out.get("token_level"))
        if isinstance(mean_out, Mapping):
            mean_rows = _rows_from(mean_out.get("mean") or mean_out.get("pooled"))
        for pos in range(n):
            tok = token_rows[pos] if pos < len(token_rows) else None
            mn = mean_rows[pos] if pos < len(mean_rows) else None
            if mn is None and tok is not None and _HAS_TORCH and isinstance(tok, torch.Tensor):
                mn = tok.float().mean(dim=0) if tok.dim() >= 2 else tok.float()
            records.append({"token": tok, "mean": mn})
    return records


# ---------------------------------------------------------------------------------------------
# LogitCache
# ---------------------------------------------------------------------------------------------
class LogitCache:
    """Upstream top-k logit cache: ``j -> TopKLogits`` for ``f_0(x_j)`` (k=100 by default)."""

    def __init__(self, topk: int = DEFAULT_TOPK, entries: Optional[Mapping[Any, Any]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.topk = int(topk)
        self.entries: Dict[int, Any] = {}
        self.meta: Dict[str, Any] = dict(meta or {})
        for key, value in (entries or {}).items():
            self.add(key, value)

    # -- container protocol -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, index: Any) -> bool:
        return int(index) in self.entries

    def __getitem__(self, index: Any) -> Any:
        return self.entries[int(index)]

    def keys(self) -> List[int]:
        return sorted(self.entries.keys())

    def indices(self) -> List[int]:
        return self.keys()

    def items(self):
        return self.entries.items()

    # -- mutation ---------------------------------------------------------------------------
    def add(self, index: Any, topk: Any, values: Any = None, indices: Any = None) -> None:
        """Store an entry; accepts a ``TopKLogits`` object or raw arrays."""
        if values is not None or indices is not None:
            payload = _as_topk(indices, values)
        elif topk is None:
            return
        elif hasattr(topk, "to_dict") and hasattr(topk, "indices"):
            payload = topk
        elif isinstance(topk, Mapping):
            payload = _as_topk(topk.get("indices"), topk.get("values"))
        elif isinstance(topk, (list, tuple)) and len(topk) == 2 and not _is_number(topk[0]):
            payload = _as_topk(topk[0], topk[1])
        else:  # raw logits tensor/list -> top-k
            payload = topk_from_logits(topk, k=self.topk)
        self.entries[int(index)] = payload

    def add_raw_logits(self, index: Any, logits: Any) -> None:
        self.entries[int(index)] = topk_from_logits(logits, k=self.topk)

    def update(self, other: "LogitCache") -> None:
        self.entries.update(other.entries)

    def subset(self, indices: Iterable[Any]) -> "LogitCache":
        out = LogitCache(topk=self.topk, meta=dict(self.meta))
        for idx in indices:
            if int(idx) in self.entries:
                out.entries[int(idx)] = self.entries[int(idx)]
        return out

    def get(self, index: Any, default: Any = None) -> Any:
        return self.entries.get(int(index), default)

    def values_tensor(self, index: Any):
        entry = self.get(index)
        if entry is None:
            return None
        return _to_tensor(getattr(entry, "values", None))

    def indices_tensor(self, index: Any):
        entry = self.get(index)
        if entry is None:
            return None
        return _to_tensor(getattr(entry, "indices", None))

    # -- serialisation ----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topk": self.topk,
            "meta": self.meta,
            "entries": {str(k): v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LogitCache":
        d = d or {}
        cache = cls(topk=int(d.get("topk", DEFAULT_TOPK)), meta=d.get("meta"))
        for key, value in (d.get("entries") or {}).items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                idx = key
            cache.entries[idx] = _as_topk((value or {}).get("indices"), (value or {}).get("values"))
        return cache

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if path.endswith(".pt") and _HAS_TORCH:
            payload = {
                "topk": self.topk,
                "meta": self.meta,
                "indices": {int(k): getattr(v, "indices", []) for k, v in self.entries.items()},
                "values": {int(k): _to_tensor(getattr(v, "values", []), torch.float32)
                           for k, v in self.entries.items()},
            }
            torch.save(payload, path)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh)
        logger.info("saved logit cache (%d entries, top-%d) -> %s", len(self), self.topk, path)
        return path

    @classmethod
    def load(cls, path: str) -> "LogitCache":
        if path.endswith(".pt") and _HAS_TORCH:
            payload = torch.load(path, map_location="cpu")
            cache = cls(topk=int(payload.get("topk", DEFAULT_TOPK)), meta=payload.get("meta"))
            idx_map = payload.get("indices", {})
            val_map = payload.get("values", {})
            for key in idx_map:
                indices = _rows_from(idx_map[key])
                values = _rows_from(val_map[key])
                values = [v.detach().float().tolist() if _HAS_TORCH and isinstance(v, torch.Tensor) else v
                          for v in values]
                cache.entries[int(key)] = _as_topk(indices, values)
            return cache
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def stats(self) -> Dict[str, Any]:
        lengths = [len(getattr(v, "values", []) or []) for v in self.entries.values()]
        return {
            "n_entries": len(self),
            "topk": self.topk,
            "mean_output_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        }


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------------------------
# RepresentationCache
# ---------------------------------------------------------------------------------------------
class RepresentationCache:
    """``h(x_j, y_j)`` cache: token-level ``[T, d]`` and/or mean-pooled ``[d]`` per upstream ``j``."""

    def __init__(self, mean: Optional[Mapping[Any, Any]] = None,
                 token: Optional[Mapping[Any, Any]] = None,
                 dim: Optional[int] = None, meta: Optional[Dict[str, Any]] = None,
                 store_token_level: bool = True, store_mean: bool = True):
        self.mean_map: Dict[int, Any] = {}
        self.token_map: Dict[int, Any] = {}
        self.dim = dim
        self.meta: Dict[str, Any] = dict(meta or {})
        self.store_token_level = bool(store_token_level)
        self.store_mean = bool(store_mean)
        for k, v in (mean or {}).items():
            self.mean_map[int(k)] = v
        for k, v in (token or {}).items():
            self.token_map[int(k)] = v
        if self.dim is None:
            self.dim = self._infer_dim()

    # -- container protocol -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(set(self.mean_map) | set(self.token_map))

    def __contains__(self, index: Any) -> bool:
        idx = int(index)
        return idx in self.mean_map or idx in self.token_map

    def keys(self) -> List[int]:
        return sorted(set(self.mean_map) | set(self.token_map))

    def indices(self) -> List[int]:
        return self.keys()

    # -- mutation ---------------------------------------------------------------------------
    def add(self, index: Any, mean: Any = None, token: Any = None) -> None:
        idx = int(index)
        if mean is not None and self.store_mean:
            self.mean_map[idx] = _to_tensor(mean, torch.float32) if _HAS_TORCH else mean
        if token is not None and self.store_token_level:
            self.token_map[idx] = _to_tensor(token, torch.float32) if _HAS_TORCH else token
        if self.dim is None:
            self.dim = self._infer_dim()

    def get(self, index: Any, mode: str = "mean", default: Any = None) -> Any:
        idx = int(index)
        if mode in ("mean", "pooled", "repr"):
            return self.mean_map.get(idx, default)
        if mode in ("token", "token_level", "seq"):
            return self.token_map.get(idx, default)
        raise ValueError("unknown mode %r (expected 'mean' or 'token')" % mode)

    def mean(self, index: Any, default: Any = None) -> Any:
        return self.mean_map.get(int(index), default)

    def token(self, index: Any, default: Any = None) -> Any:
        return self.token_map.get(int(index), default)

    def subset(self, indices: Iterable[Any]) -> "RepresentationCache":
        keep = [int(i) for i in indices]
        return RepresentationCache(
            mean={i: self.mean_map[i] for i in keep if i in self.mean_map},
            token={i: self.token_map[i] for i in keep if i in self.token_map},
            dim=self.dim, meta=dict(self.meta),
            store_token_level=self.store_token_level, store_mean=self.store_mean,
        )

    def stack_mean(self, indices: Optional[Iterable[Any]] = None, device: Any = None):
        """Stack mean-pooled representations into ``[N, d]`` (order follows ``indices``)."""
        keys = [int(i) for i in indices] if indices is not None else [
            k for k in self.keys() if k in self.mean_map]
        rows = [self.mean_map[k] for k in keys if k in self.mean_map]
        if not rows:
            return None
        if not _HAS_TORCH:
            return rows
        rows = [_to_tensor(r, torch.float32) for r in rows]
        out = torch.stack([r.reshape(-1) for r in rows], dim=0)
        return out.to(device) if device is not None else out

    def _infer_dim(self) -> Optional[int]:
        for mapping in (self.mean_map, self.token_map):
            for value in mapping.values():
                if _HAS_TORCH and isinstance(value, torch.Tensor):
                    return int(value.shape[-1])
                if isinstance(value, (list, tuple)) and len(value):
                    inner = value
                    while isinstance(inner, (list, tuple)) and len(inner) and not _is_number(inner[0]):
                        inner = inner[0]
                    return int(len(inner)) if hasattr(inner, "__len__") else None
        return None

    # -- serialisation ----------------------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if path.endswith(".pt") and _HAS_TORCH:
            torch.save({
                "mean": {int(k): _to_tensor(v, torch.float32) for k, v in self.mean_map.items()},
                "token": {int(k): _to_tensor(v, torch.float32) for k, v in self.token_map.items()},
                "dim": self.dim,
                "meta": self.meta,
                "store_token_level": self.store_token_level,
                "store_mean": self.store_mean,
            }, path)
        else:
            payload = {
                "dim": self.dim,
                "meta": self.meta,
                "store_token_level": self.store_token_level,
                "store_mean": self.store_mean,
                "mean": {str(k): _tolist(v) for k, v in self.mean_map.items()},
                "token": {str(k): _tolist(v) for k, v in self.token_map.items()},
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        logger.info("saved representation cache (%d entries, dim=%s) -> %s", len(self), self.dim, path)
        return path

    @classmethod
    def load(cls, path: str) -> "RepresentationCache":
        if path.endswith(".pt") and _HAS_TORCH:
            payload = torch.load(path, map_location="cpu")
            return cls(mean=payload.get("mean"), token=payload.get("token"),
                       dim=payload.get("dim"), meta=payload.get("meta"),
                       store_token_level=payload.get("store_token_level", True),
                       store_mean=payload.get("store_mean", True))
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        mean = {int(k): v for k, v in (payload.get("mean") or {}).items()}
        token = {int(k): v for k, v in (payload.get("token") or {}).items()}
        if _HAS_TORCH:
            mean = {k: _to_tensor(v, torch.float32) for k, v in mean.items()}
            token = {k: _to_tensor(v, torch.float32) for k, v in token.items()}
        return cls(mean=mean, token=token, dim=payload.get("dim"), meta=payload.get("meta"),
                   store_token_level=payload.get("store_token_level", True),
                   store_mean=payload.get("store_mean", True))

    def stats(self) -> Dict[str, Any]:
        token_lens = []
        for value in self.token_map.values():
            if _HAS_TORCH and isinstance(value, torch.Tensor) and value.dim() >= 2:
                token_lens.append(int(value.shape[0]))
        return {
            "n_entries": len(self),
            "n_mean": len(self.mean_map),
            "n_token": len(self.token_map),
            "dim": self.dim,
            "mean_output_len": (sum(token_lens) / len(token_lens)) if token_lens else 0.0,
        }


def _tolist(value: Any) -> Any:
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_tolist(v) for v in value]
    return value


# ---------------------------------------------------------------------------------------------
# PriorCache
# ---------------------------------------------------------------------------------------------
class PriorCache:
    """Frequency-prior cache ``b_j`` for upstream examples (Sec. 3.3)."""

    def __init__(self, prior: Any = None, default: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        self.prior = prior
        self.default = float(default)
        self.meta: Dict[str, Any] = dict(meta or {})

    def __len__(self) -> int:
        try:
            return len(self.prior)  # type: ignore[arg-type]
        except Exception:
            return 0

    def get(self, index: Any, default: Optional[float] = None) -> float:
        try:
            from ..forgetting.frequency_prior import prior_for_upstream  # type: ignore
        except Exception:  # pragma: no cover
            prior_for_upstream = None  # type: ignore
        dflt = self.default if default is None else default
        if prior_for_upstream is not None and self.prior is not None:
            return float(prior_for_upstream(self.prior, index, default=dflt))
        if self.prior is not None:
            try:
                return float(self.prior.get(index, dflt))
            except Exception:
                try:
                    return float(self.prior[int(index)])
                except Exception:
                    return dflt
        return dflt

    def __getitem__(self, index: Any) -> float:
        return self.get(index)

    def as_tensor(self, indices: Optional[Iterable[Any]] = None, default: Optional[float] = None,
                  device: Any = None):
        keys = [int(i) for i in indices] if indices is not None else sorted(self.keys())
        values = [self.get(k, default) for k in keys]
        if not _HAS_TORCH:
            return values
        t = torch.tensor(values, dtype=torch.float32) if values else torch.zeros(0)
        return t.to(device) if device is not None else t

    def keys(self) -> List[int]:
        try:
            return sorted(int(k) for k in self.prior.keys())  # type: ignore[union-attr]
        except Exception:
            return []

    def items(self):
        return [(k, self.get(k)) for k in self.keys()]

    # -- serialisation ----------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "PriorCache":
        try:
            from ..forgetting.frequency_prior import load_prior  # type: ignore

            return cls(prior=load_prior(path), meta={"path": path})
        except Exception:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            priors = payload.get("priors", payload)
            priors = {int(k): float(v) for k, v in priors.items()} if isinstance(priors, Mapping) else {}
            return cls(prior=priors, meta={"path": path})

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if self.prior is not None and hasattr(self.prior, "save"):
            self.prior.save(path)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"priors": {str(k): self.get(k) for k in self.keys()}}, fh)
        logger.info("saved prior cache (%d entries) -> %s", len(self), path)
        return path


# ---------------------------------------------------------------------------------------------
# ForecastCache: the bundle consumed at forecast time
# ---------------------------------------------------------------------------------------------
@dataclass
class ForecastCache:
    """Bundle of the three caches making forecast-time inference O(N_PT) and LM-free."""

    logits: Optional[LogitCache] = None
    representations: Optional[RepresentationCache] = None
    priors: Optional[PriorCache] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def upstream_indices(self) -> List[int]:
        keys: set = set()
        if self.logits is not None:
            keys |= set(int(k) for k in self.logits.keys())
        if self.representations is not None:
            keys |= set(int(k) for k in self.representations.keys())
        if self.priors is not None:
            keys |= set(int(k) for k in self.priors.keys())
        return sorted(keys)

    def __len__(self) -> int:
        return len(self.upstream_indices())

    def f0_topk(self, j: Any) -> Any:
        return None if self.logits is None else self.logits.get(j)

    def h_token(self, j: Any) -> Any:
        if self.representations is None:
            return None
        return self.representations.token(j)

    def h_mean(self, j: Any) -> Any:
        if self.representations is None:
            return None
        return self.representations.mean(j)

    def b_j(self, j: Any, default: Optional[float] = None) -> float:
        if self.priors is None:
            return 0.0 if default is None else float(default)
        return self.priors.get(j, default)

    def stack_mean(self, indices: Optional[Iterable[Any]] = None, device: Any = None):
        if self.representations is None:
            return None
        return self.representations.stack_mean(indices, device=device)

    def stack_token(self, indices: Optional[Iterable[Any]] = None) -> List[Any]:
        if self.representations is None:
            return []
        keys = [int(i) for i in indices] if indices is not None else self.representations.keys()
        return [self.representations.token(k) for k in keys if self.representations.token(k) is not None]

    def save(self, out_dir: str, logit_filename: str = DEFAULT_LOGIT_FILENAME,
             repr_filename: str = DEFAULT_REPR_FILENAME,
             prior_filename: str = DEFAULT_PRIOR_FILENAME) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        paths: Dict[str, str] = {}
        if self.logits is not None:
            paths["logits"] = self.logits.save(os.path.join(out_dir, logit_filename))
        if self.representations is not None:
            paths["representations"] = self.representations.save(os.path.join(out_dir, repr_filename))
        if self.priors is not None:
            paths["priors"] = self.priors.save(os.path.join(out_dir, prior_filename))
        meta_path = os.path.join(out_dir, "cache_meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"paths": paths, "meta": self.meta,
                       "stats": {
                           "logits": self.logits.stats() if self.logits is not None else None,
                           "representations": (self.representations.stats()
                                               if self.representations is not None else None),
                           "priors": len(self.priors) if self.priors is not None else None,
                       }}, fh, indent=2)
        paths["meta"] = meta_path
        return paths

    @classmethod
    def load(cls, out_dir: str, logit_filename: str = DEFAULT_LOGIT_FILENAME,
             repr_filename: str = DEFAULT_REPR_FILENAME,
             prior_filename: str = DEFAULT_PRIOR_FILENAME) -> "ForecastCache":
        cache = cls(meta={"dir": out_dir})
        lp = os.path.join(out_dir, logit_filename)
        if os.path.exists(lp):
            cache.logits = LogitCache.load(lp)
        rp = os.path.join(out_dir, repr_filename)
        if os.path.exists(rp):
            cache.representations = RepresentationCache.load(rp)
        pp = os.path.join(out_dir, prior_filename)
        if os.path.exists(pp):
            cache.priors = PriorCache.from_file(pp)
        return cache


# ---------------------------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------------------------
def _iter_batches(indices: Sequence[int], batch_size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(indices), max(1, batch_size)):
        yield indices[start:start + batch_size]


def build_logit_cache(model: Any, examples: Sequence[Mapping[str, Any]],
                      indices: Optional[Sequence[int]] = None, k: int = DEFAULT_TOPK,
                      batch_size: int = DEFAULT_BATCH_SIZE, save_to: Optional[str] = None,
                      meta: Optional[Dict[str, Any]] = None, desc: str = "logit cache") -> LogitCache:
    """Cache ``f_0(x_j)`` top-k logits (k=100) for each upstream example ``j``.

    ``examples`` is the upstream pool (``D_PT_hat``); ``indices`` optionally maps position ->
    upstream index (default: position in the list).
    """
    idx_list = list(range(len(examples))) if indices is None else [int(i) for i in indices]
    if len(idx_list) != len(examples):
        raise ValueError("indices length %d != examples length %d" % (len(idx_list), len(examples)))
    cache = LogitCache(topk=k, meta=dict(meta or {}))
    if len(examples) == 0:
        return cache

    try:
        from tqdm import tqdm as _tqdm
    except Exception:  # pragma: no cover
        _tqdm = None

    iterator = _iter_batches(list(range(len(examples))), batch_size)
    if _tqdm is not None:
        iterator = _tqdm(iterator, total=(len(examples) + batch_size - 1) // batch_size, desc=desc)
    for positions in iterator:
        inputs = [_input_of(examples[p]) for p in positions]
        targets = [target_of(examples[p]) for p in positions]
        topks = extract_token_topk(model, inputs, targets, k=k, batch_size=len(positions))
        for pos, topk in zip(positions, topks):
            cache.add(idx_list[pos], topk)
    if save_to:
        cache.save(save_to)
    return cache


def build_representation_cache(encoder: Any, examples: Sequence[Mapping[str, Any]],
                               indices: Optional[Sequence[int]] = None,
                               batch_size: int = DEFAULT_BATCH_SIZE,
                               token_level: bool = True, mean_pool: bool = True,
                               save_to: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                               desc: str = "h cache") -> RepresentationCache:
    """Cache ``h(x_j, y_j)`` (token-level and/or mean-pooled) for each upstream example."""
    idx_list = list(range(len(examples))) if indices is None else [int(i) for i in indices]
    if len(idx_list) != len(examples):
        raise ValueError("indices length %d != examples length %d" % (len(idx_list), len(examples)))
    cache = RepresentationCache(meta=dict(meta or {}), store_token_level=token_level,
                                store_mean=mean_pool)
    if len(examples) == 0:
        return cache

    try:
        from tqdm import tqdm as _tqdm
    except Exception:  # pragma: no cover
        _tqdm = None

    iterator = _iter_batches(list(range(len(examples))), batch_size)
    if _tqdm is not None:
        iterator = _tqdm(iterator, total=(len(examples) + batch_size - 1) // batch_size, desc=desc)
    for positions in iterator:
        inputs = [_input_of(examples[p]) for p in positions]
        targets = [target_of(examples[p]) for p in positions]
        reps = extract_representations(encoder, inputs, targets, batch_size=len(positions),
                                       mean_pool=mean_pool, token_level=token_level)
        for pos, rep in zip(positions, reps):
            cache.add(idx_list[pos], mean=rep.get("mean"), token=rep.get("token"))
    if save_to:
        cache.save(save_to)
    return cache


def build_prior_cache(pair_records: Optional[Iterable[Any]] = None, gt_dir: Optional[str] = None,
                      n_online: Optional[int] = None, n_upstream: Optional[int] = None,
                      eps: float = 1e-8, smoothing: float = 0.0,
                      save_to: Optional[str] = None) -> PriorCache:
    """Build the frequency-prior cache ``b_j`` (Sec. 3.3 / Algorithm 3).

    Either pass labelled pair records (``<i, j, z_ij>``) directly or point at a ground-truth
    directory containing ``pairs.jsonl``.
    """
    try:
        from ..forgetting.frequency_prior import (  # type: ignore
            estimate_frequency_prior,
            estimate_prior_from_ground_truth_dir,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError("frequency_prior module is required to build the prior cache: %s" % exc)

    if pair_records is None:
        if gt_dir is None:
            raise ValueError("either pair_records or gt_dir must be provided")
        prior = estimate_prior_from_ground_truth_dir(gt_dir, n_upstream=n_upstream, eps=eps,
                                                     smoothing=smoothing)
    else:
        prior = estimate_frequency_prior(list(pair_records), n_online=n_online,
                                         n_upstream=n_upstream, eps=eps, smoothing=smoothing)
    cache = PriorCache(prior=prior)
    if save_to:
        cache.save(save_to)
    return cache


def build_caches(base_lm: Any = None, encoder: Any = None,
                 upstream_examples: Optional[Sequence[Mapping[str, Any]]] = None,
                 upstream_indices: Optional[Sequence[int]] = None,
                 pair_records: Optional[Iterable[Any]] = None, gt_dir: Optional[str] = None,
                 k: int = DEFAULT_TOPK, batch_size: int = DEFAULT_BATCH_SIZE,
                 out_dir: Optional[str] = None, build_logits: bool = True,
                 build_representations: bool = True, build_priors: bool = True,
                 token_level: bool = True, mean_pool: bool = True,
                 n_online: Optional[int] = None,
                 meta: Optional[Dict[str, Any]] = None) -> ForecastCache:
    """Build the full forecasting cache bundle (top-k logits, h, priors) and persist it."""
    cache = ForecastCache(meta=dict(meta or {}))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if build_logits:
        if base_lm is None or upstream_examples is None:
            raise ValueError("build_logits requires base_lm and upstream_examples")
        cache.logits = build_logit_cache(
            base_lm, upstream_examples, indices=upstream_indices, k=k, batch_size=batch_size,
            save_to=os.path.join(out_dir, DEFAULT_LOGIT_FILENAME) if out_dir else None,
            meta=meta)

    if build_representations:
        if encoder is None or upstream_examples is None:
            raise ValueError("build_representations requires encoder and upstream_examples")
        cache.representations = build_representation_cache(
            encoder, upstream_examples, indices=upstream_indices, batch_size=batch_size,
            token_level=token_level, mean_pool=mean_pool,
            save_to=os.path.join(out_dir, DEFAULT_REPR_FILENAME) if out_dir else None,
            meta=meta)

    if build_priors:
        if pair_records is not None or gt_dir is not None:
            cache.priors = build_prior_cache(
                pair_records=pair_records, gt_dir=gt_dir, n_online=n_online,
                n_upstream=(len(upstream_examples) if upstream_examples is not None else None),
                save_to=os.path.join(out_dir, DEFAULT_PRIOR_FILENAME) if out_dir else None)
        else:
            logger.warning("no pair records / gt dir given: skipping frequency-prior cache")

    if out_dir:
        cache.meta["paths"] = cache.save(out_dir)
    return cache


def load_caches(out_dir: str, logit_filename: str = DEFAULT_LOGIT_FILENAME,
                repr_filename: str = DEFAULT_REPR_FILENAME,
                prior_filename: str = DEFAULT_PRIOR_FILENAME) -> ForecastCache:
    """Load a previously built cache bundle (used by the forecast/stream scripts)."""
    return ForecastCache.load(out_dir, logit_filename=logit_filename,
                              repr_filename=repr_filename, prior_filename=prior_filename)


def cache_paths(out_dir: str) -> Dict[str, str]:
    """Canonical cache file paths inside ``out_dir``."""
    return {
        "logits": os.path.join(out_dir, DEFAULT_LOGIT_FILENAME),
        "representations": os.path.join(out_dir, DEFAULT_REPR_FILENAME),
        "priors": os.path.join(out_dir, DEFAULT_PRIOR_FILENAME),
    }


# ---------------------------------------------------------------------------------------------
# self test / CLI
# ---------------------------------------------------------------------------------------------
class _DummyLM:
    """Tiny offline stand-in exposing the ``token_logits`` contract (self-test only)."""

    def token_logits(self, inputs, targets, k=DEFAULT_TOPK, **kwargs):
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for the self-test")
        rows = []
        for target in targets:
            n_tok = max(1, len(str(target).split()))
            logits = torch.randn(n_tok, 50)
            rows.append(logits)
        return {"logits": torch.stack(rows, dim=0), "target_ids": None}


class _DummyEncoder:
    """Tiny offline stand-in exposing ``encode_token_level``/``encode_mean`` (self-test only)."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode_token_level(self, inputs, targets, **kwargs):
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for the self-test")
        rows = []
        for target in targets:
            n_tok = max(1, len(str(target).split()))
            rows.append(torch.randn(n_tok, self.dim))
        return rows

    def encode_mean(self, inputs, targets, **kwargs):
        toks = self.encode_token_level(inputs, targets, **kwargs)
        return torch.stack([t.mean(dim=0) for t in toks], dim=0)


def _self_test() -> int:
    if not _HAS_TORCH:
        print("torch unavailable: self-test skipped")
        return 0
    examples = [
        {"input": "q1", "target": "a b c"},
        {"input": "q2", "target": "d e"},
        {"input": "q3", "target": "f"},
    ]
    logit_cache = build_logit_cache(_DummyLM(), examples, k=5, batch_size=2)
    assert len(logit_cache) == 3, len(logit_cache)
    assert len(logit_cache.get(0).values[0]) == 5
    repr_cache = build_representation_cache(_DummyEncoder(), examples, batch_size=2)
    assert len(repr_cache) == 3
    assert repr_cache.mean(0).shape[-1] == 16
    assert repr_cache.stack_mean().shape == (3, 16)

    # Frequency prior: example 1 is forgotten by 3 of 4 online examples.
    class _Rec:
        def __init__(self, i, j, z):
            self.i, self.j, self.z = i, j, z

    pairs = [_Rec(i, j, int(j == 1 and i < 3)) for i in range(4) for j in range(3)]
    prior_cache = build_prior_cache(pairs, n_online=4, n_upstream=3)
    b1 = prior_cache.get(1)
    assert abs(b1 - (__import__("math").log(0.75) - __import__("math").log(0.25))) < 1e-6, b1

    bundle = ForecastCache(logits=logit_cache, representations=repr_cache, priors=prior_cache)
    assert bundle.upstream_indices() == [0, 1, 2]
    assert bundle.f0_topk(1) is not None and bundle.h_mean(2) is not None

    # round-trip serialisation
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        paths = bundle.save(tmp)
        assert os.path.exists(paths["logits"]) and os.path.exists(paths["representations"])
        assert os.path.exists(paths["priors"]) and os.path.exists(paths["meta"])
        reloaded = load_caches(tmp)
        assert len(reloaded.logits) == 3
        assert abs(reloaded.logits.get(0).values[0][0] - logit_cache.get(0).values[0][0]) < 1e-5
        assert reloaded.representations.mean(0).shape[-1] == 16
        assert abs(reloaded.b_j(1) - b1) < 1e-6
    print("caches self-test OK")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/load the forgetting-forecasting caches")
    p.add_argument("--self-test", action="store_true", help="run the offline cache self-test")
    p.add_argument("--model", default=None, help="base LM key for the top-k logit cache")
    p.add_argument("--encoder-model", default=None, help="model key whose h backbone is used")
    p.add_argument("--examples", default=None, help="JSONL file of upstream examples (D_PT_hat)")
    p.add_argument("--gt-dir", default=None, help="ground-truth dir containing pairs.jsonl")
    p.add_argument("--out-dir", default=None, help="directory where caches are written")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-logits", action="store_true")
    p.add_argument("--no-representations", action="store_true")
    p.add_argument("--no-priors", action="store_true")
    p.add_argument("--no-token-level", action="store_true", help="skip token-level h (kernel mode)")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test or (args.examples is None and args.out_dir is None):
        return _self_test()

    import json as _json

    examples: List[Dict[str, Any]] = []
    with open(args.examples, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(_json.loads(line))
    logger.info("loaded %d upstream examples from %s", len(examples), args.examples)

    base_lm = None
    if not args.no_logits:
        from .base_lm import load_base_lm  # type: ignore

        base_lm = load_base_lm(args.model or "BART0_L", device=args.device)

    encoder = None
    if not args.no_representations:
        from .encoder_h import load_encoder_h  # type: ignore

        encoder = load_encoder_h(args.encoder_model or args.model or "BART0_L",
                                 backbone=base_lm, device=args.device)

    build_caches(
        base_lm=base_lm, encoder=encoder, upstream_examples=examples, gt_dir=args.gt_dir,
        k=args.topk, batch_size=args.batch_size, out_dir=args.out_dir,
        build_logits=not args.no_logits, build_representations=not args.no_representations,
        build_priors=not args.no_priors, token_level=not args.no_token_level,
        meta={"model": args.model, "encoder_model": args.encoder_model},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
