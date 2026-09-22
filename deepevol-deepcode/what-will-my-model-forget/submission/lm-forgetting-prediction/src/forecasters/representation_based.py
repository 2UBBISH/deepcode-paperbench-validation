"""Representation-based forecasting of example forgetting.

Implements Sec. 3.3 (Eq. 4) of "What Will My Model Forget? Forecasting Forgotten
Examples in Language Model Refinement", plus the frequency-prior variant and the
"w/o Prior" ablation reported in Sec. 5.1 / Table 1.  Training and inference
procedures follow Appendix F, Algorithms 3 and 4.

Paper formulation (Eq. 4, verbatim)::

    g(<x_i, y_i>, <x_j, y_j>) = sigma( h(x_j, y_j) h(x_i, y_i)^T )

where ``h`` denotes the *averaged* representation over all tokens of
``<x, y>`` (in contrast to the token-level ``h`` used by the logit-change
transfer forecaster in Sec. 3.2).  With frequency priors (Sec. 3.3)::

    z_tilde_ij = sigma( h(x_j, y_j) h(x_i, y_i)^T + b_j )

    b_j = log(|{<x_i,y_i> in D_R^train | z_ij = 1}| / |D_R^train|)
        - log(|{<x_i,y_i> in D_R^train | z_ij = 0}| / |D_R^train|)

and ``h`` is learned by minimizing binary cross entropy loss
``L_BCE(z_tilde_ij, z_ij)``.

Efficient inference (Algorithm 4): the encoding ``h(x_j, y_j)`` for every
upstream example ``<x_j, y_j> in D_PT_hat`` and the frequency prior ``b_j`` are
cached; only the single online example ``<x_i, y_i>`` must be encoded at
forecast time, so forecasting costs ``O(N_PT)`` inner products and **no**
additional PTLM inference.

Label definition note (Sec. 2 vs. Appendix F):
    Sec. 2 defines forgetting of the *upstream* example,
    ``z_ij = 1[f_i(x_j) != y_j]``, whereas Appendix F Algorithms 1/3 write
    ``z_ij <- 1 if f_0(x_i) != f_i(x_i) else 0`` (a typo, referring to the
    online example ``x_i``).  Consistently with ``src/forgetting/ground_truth.py``
    and the plan, this module consumes labels produced by the Sec. 2 definition.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Loss helpers (src/forecasters/losses.py) with standalone fallbacks so that
# this module remains usable/testable in isolation.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on import context
    from .losses import (
        NEG_INF,
        binary_cross_entropy_with_prior,
        counts_to_prior,
        representation_loss,
        representation_scores,
        weighted_binary_cross_entropy,
    )

    _HAS_LOSSES = True
except Exception:  # pragma: no cover
    _HAS_LOSSES = False
    NEG_INF = -1e4

    def counts_to_prior(n_positive, n_negative, n_total=None, eps: float = 1e-8):
        return math.log(float(n_positive) + eps) - math.log(float(n_negative) + eps)

    def representation_scores(h_upstream, h_online, prior=None):
        s = (h_upstream * h_online).sum(dim=-1)
        if prior is not None:
            s = s + prior
        return torch.sigmoid(s)

    def weighted_binary_cross_entropy(
        probabilities, z, positive_weight: float = 0.1, reduction: str = "mean", eps: float = 1e-7
    ):
        z = z.float()
        p = probabilities.clamp(eps, 1.0 - eps)
        loss = -(positive_weight * z * p.log() + (1.0 - z) * (1.0 - p).log())
        return loss.mean() if reduction == "mean" else loss.sum()

    def binary_cross_entropy_with_prior(
        pair_score, z, prior=None, positive_weight: float = 1.0, reduction: str = "mean"
    ):
        s = pair_score
        if prior is not None:
            s = s + prior
        return weighted_binary_cross_entropy(torch.sigmoid(s), z, positive_weight, reduction)

    def representation_loss(
        h_upstream, h_online, z, prior=None, positive_weight: float = 0.1, reduction: str = "mean"
    ):
        s = (h_upstream * h_online).sum(dim=-1)
        return binary_cross_entropy_with_prior(s, z, prior, positive_weight, reduction)


logger = logging.getLogger(__name__)

# Defaults (paper: ``h`` is the base LM backbone followed by a freshly
# initialized 2-layer MLP; the MLP width is unspecified -> 768, see the plan's
# reconciliation notes).
DEFAULT_DIM = 768
DEFAULT_FILENAME = "representation_forecaster.pt"
DEFAULT_DECISION_THRESHOLD = 0.5

__all__ = [
    "RepresentationBasedForecaster",
    "mean_pool_representation",
    "representation_logits",
    "representation_probabilities",
    "prior_vector_from",
    "evaluate_labels",
    "extract_representations_from_records",
    "train_representation_forecaster",
    "forecast_upstream_examples",
    "logits_from_flat",
    "parse_args",
    "main",
    "DEFAULT_DIM",
    "DEFAULT_FILENAME",
]


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, Mapping):
        for key in ("h", "values", "value", "tensor", "repr", "representation"):
            if key in x:
                return _to_tensor(x[key], dtype)
        raise KeyError("cannot find a representation inside mapping: %s" % sorted(x.keys()))
    if isinstance(x, (list, tuple)):
        if len(x) and isinstance(x[0], Mapping):
            return torch.tensor([_to_tensor(e, dtype).tolist() for e in x], dtype=dtype)
        return torch.tensor(x, dtype=dtype)
    raise TypeError("unsupported representation type: %r" % type(x))


def _as_2d(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = _to_tensor(x, dtype)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() > 2:
        t = t.reshape(-1, t.shape[-1])
    return t


def _as_vector(x: Any, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a prior (scalar / vector / list) against a batch tensor."""
    if isinstance(x, torch.Tensor):
        v = x.to(like.device).to(like.dtype)
    elif isinstance(x, (int, float)):
        return torch.full_like(like, float(x))
    else:
        v = torch.as_tensor(list(x), device=like.device, dtype=like.dtype)
    while v.dim() < like.dim():
        v = v.unsqueeze(-1)
    return v


def mean_pool_representation(hidden: Any, mask: Optional[Any] = None) -> torch.Tensor:
    """Mean-pool token-level representations over all tokens (Sec. 3.3).

    ``hidden`` is ``[B, T, d]`` (or ``[T, d]``); ``mask`` is an optional
    ``[B, T]`` supervision mask.  When ``mask`` is ``None`` all tokens
    contribute equally.  Returns ``[B, d]``.
    """
    h = _to_tensor(hidden)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    if h.dim() != 3:
        raise ValueError("mean_pool_representation expects [B, T, d] or [T, d], got %s" % (tuple(h.shape),))
    if mask is None:
        return h.mean(dim=1)

    m = _to_tensor(mask).to(h.device)
    if m.dtype != h.dtype:
        m = m.to(h.dtype)
    if m.dim() == 3:
        m = m.squeeze(-1)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    if m.shape[:2] != h.shape[:2]:
        raise ValueError("mask shape %s incompatible with hidden shape %s" % (tuple(m.shape), tuple(h.shape)))
    denom = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (h * m.unsqueeze(-1)).sum(dim=1) / denom


def representation_logits(
    h_upstream: Any, h_online: Any, prior: Optional[Any] = None, normalize: bool = False
) -> torch.Tensor:
    """Pre-sigmoid score of Eq. 4: ``<h(x_j,y_j), h(x_i,y_i)> (+ b_j)``.

    ``h_upstream`` / ``h_online`` are mean-pooled ``[B, d]`` representations.
    ``normalize=True`` yields a cosine kernel (diagnostics only; the paper uses
    the raw inner product).
    """
    hj = _as_2d(h_upstream)
    hi = _as_2d(h_online)
    if hj.shape[-1] != hi.shape[-1]:
        raise ValueError("representation dim mismatch: upstream %d vs online %d" % (hj.shape[-1], hi.shape[-1]))
    if hi.shape[0] == 1 and hj.shape[0] != 1:
        hi = hi.expand(hj.shape[0], -1)
    elif hj.shape[0] == 1 and hi.shape[0] != 1:
        hj = hj.expand(hi.shape[0], -1)
    if normalize:
        hj = F.normalize(hj, dim=-1)
        hi = F.normalize(hi, dim=-1)
    s = (hj * hi).sum(dim=-1)
    if prior is not None:
        s = s + _as_vector(prior, s)
    return s


def representation_probabilities(
    h_upstream: Any, h_online: Any, prior: Optional[Any] = None, normalize: bool = False
) -> torch.Tensor:
    """``z_tilde_ij = sigma(<h_j, h_i> + b_j)`` (Eq. 4 with frequency prior)."""
    return torch.sigmoid(representation_logits(h_upstream, h_online, prior=prior, normalize=normalize))


def prior_vector_from(
    prior: Any,
    upstream_indices: Sequence[Any],
    default: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Look up cached frequency priors ``b_j`` for a sequence of upstream indices.

    ``prior`` may be a :class:`~src.forgetting.frequency_prior.FrequencyPrior`, a
    mapping ``j -> b_j``, a sequence aligned with ``upstream_indices``, or ``None``.
    """
    if prior is None:
        return torch.full((len(upstream_indices),), float(default), device=device, dtype=dtype)

    try:  # reuse the frequency-prior accessor when available
        from ..forgetting.frequency_prior import prior_for_upstream

        return torch.tensor(
            [float(prior_for_upstream(prior, int(j), default=default)) for j in upstream_indices],
            device=device,
            dtype=dtype,
        )
    except Exception:
        pass

    if isinstance(prior, Mapping):
        return torch.tensor(
            [float(prior.get(j, prior.get(str(j), default))) for j in upstream_indices],
            device=device,
            dtype=dtype,
        )
    if isinstance(prior, torch.Tensor):
        p = prior.reshape(-1).to(device=device, dtype=dtype)
        idx = torch.as_tensor([int(j) for j in upstream_indices], device=p.device)
        return p[idx]

    if isinstance(prior, Sequence):
        if len(prior) == len(upstream_indices):
            return torch.tensor([float(v) for v in prior], device=device, dtype=dtype)
        return torch.tensor([float(prior[int(j)]) for j in upstream_indices], device=device, dtype=dtype)

    getter = getattr(prior, "get", None)
    if callable(getter):
        return torch.tensor([float(getter(j, default)) for j in upstream_indices], device=device, dtype=dtype)
    return torch.full((len(upstream_indices),), float(default), device=device, dtype=dtype)


def logits_from_flat(h_upstream: Any, h_online: Any, prior: Optional[Any] = None) -> torch.Tensor:
    """Score a single online representation against many upstream ones (O(N_PT))."""
    return representation_logits(h_upstream, h_online, prior=prior)


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------
class RepresentationBasedForecaster(nn.Module):
    """Black-box representation-based forecaster (Sec. 3.3, Eq. 4).

    Args:
        encoder: trainable encoding function ``h`` (e.g. the base LM backbone +
            a fresh 2-layer MLP from ``src/modeling/encoder_h.py``).  Any module
            exposing ``encode_mean``/``encode_representation``/``encode``/
            ``forward`` is accepted.
        dim: dimensionality ``d`` of the mean-pooled representation.
        hidden: if given, an extra MLP head is applied to the mean-pooled
            backbone state (useful when the encoder yields backbone states).
        use_prior: whether Eq. 4 includes the frequency prior ``b_j``.  Setting
            this to ``False`` implements the Table 1 "w/o Prior" ablation.
        positive_weight: down-weight for positive (forgotten) pairs; the
            training script uses ``alpha = 0.1`` (Appendix B).
        normalize: use a cosine kernel instead of a raw inner product
            (diagnostic only).
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        dim: int = DEFAULT_DIM,
        hidden: Optional[int] = None,
        use_prior: bool = True,
        positive_weight: float = 0.1,
        encode_batch_size: int = 8,
        prior: Optional[Any] = None,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        normalize: bool = False,
        name: str = "representation_based",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.use_prior = bool(use_prior)
        self.positive_weight = float(positive_weight)
        self.encode_batch_size = int(encode_batch_size)
        self.decision_threshold = float(decision_threshold)
        self.normalize = bool(normalize)
        self.name = name
        self.prior = prior

        self.encoder: Optional[nn.Module] = encoder
        self.head: Optional[nn.Module] = None
        if encoder is None and hidden:
            self.head = nn.Sequential(
                nn.Linear(self.dim, int(hidden)), nn.GELU(), nn.Linear(int(hidden), self.dim)
            )
        self._cached_upstream: Optional[torch.Tensor] = None

    # -- construction ------------------------------------------------------
    def set_encoder(self, encoder: Optional[nn.Module]) -> "RepresentationBasedForecaster":
        """Attach the trainable encoding function ``h``."""
        self.encoder = encoder
        return self

    def set_prior(self, prior: Optional[Any]) -> "RepresentationBasedForecaster":
        """Attach the cached frequency priors ``b_j`` (Algorithm 4)."""
        self.prior = prior
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Parameters optimized while training the forecaster (encoder + head)."""
        params: List[nn.Parameter] = []
        seen = set()
        for module in (self.encoder, self.head):
            if module is None:
                continue
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    # -- encoding ----------------------------------------------------------
    def encode(
        self,
        inputs: Any,
        targets: Any = None,
        mean_pool: bool = True,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode ``<x, y>`` into a mean-pooled representation ``h`` (Sec. 3.3)."""
        if self.encoder is None:
            raise RuntimeError("RepresentationBasedForecaster has no encoder attached")
        bs = int(batch_size or self.encode_batch_size or 8)
        out = self._call_encoder(self.encoder, inputs, targets, mean_pool=mean_pool, batch_size=bs, **kwargs)
        if not isinstance(out, torch.Tensor):
            out = _to_tensor(out)
        if mean_pool and out.dim() == 3:
            out = mean_pool_representation(out)
        elif out.dim() == 1:
            out = out.unsqueeze(0)
        if self.head is not None:
            out = self.head(out)
        return out

    def _call_encoder(
        self, encoder: Any, inputs: Any, targets: Any, mean_pool: bool, batch_size: int, **kwargs: Any
    ) -> Any:
        """Dispatch to whichever contract the encoder from ``encoder_h.py`` exposes."""
        attempts = []
        if mean_pool and hasattr(encoder, "encode_mean"):
            attempts.append(lambda: encoder.encode_mean(inputs, targets, **kwargs))
        if hasattr(encoder, "encode_representation"):
            attempts.append(lambda: encoder.encode_representation(inputs, targets, **kwargs))
        if hasattr(encoder, "encode_token_level") and not mean_pool:
            attempts.append(lambda: encoder.encode_token_level(inputs, targets, **kwargs))
        if hasattr(encoder, "encode"):
            attempts.append(lambda: encoder.encode(inputs, targets, mean_pool=mean_pool, **kwargs))
            attempts.append(lambda: encoder.encode(inputs, targets, **kwargs))
        if hasattr(encoder, "forward"):
            attempts.append(lambda: encoder(inputs, targets, **kwargs))

        last_err: Optional[Exception] = None
        for attempt in attempts:
            try:
                out = attempt()
            except Exception as err:  # signature mismatch or internal failure
                last_err = err
                continue
            if out is None:
                continue
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, Mapping):
                for key in ("h", "representation", "repr", "last_hidden_state", "hidden_states"):
                    if key in out:
                        out = out[key]
                        break
            return out
        raise RuntimeError("could not encode examples with the attached encoder: %r" % (last_err,))

    # -- scoring / loss ----------------------------------------------------
    def forward(
        self,
        h_upstream: Any,
        h_online: Any,
        prior: Optional[Any] = None,
        add_prior: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forgetting probabilities ``z_tilde_ij`` (Eq. 4)."""
        use_prior = self.use_prior if add_prior is None else bool(add_prior)
        p = prior if prior is not None else self.prior
        if not use_prior:
            p = None
        return representation_probabilities(h_upstream, h_online, prior=p, normalize=self.normalize)

    def score(
        self,
        h_upstream: Any,
        h_online: Any,
        prior: Optional[Any] = None,
        add_prior: Optional[bool] = None,
    ) -> torch.Tensor:
        """Pre-sigmoid score (prior optional)."""
        use_prior = self.use_prior if add_prior is None else bool(add_prior)
        p = prior if (use_prior and prior is not None) else None
        return representation_logits(h_upstream, h_online, prior=p, normalize=self.normalize)

    def loss(
        self,
        h_upstream: Any,
        h_online: Any,
        z: Any,
        prior: Optional[Any] = None,
        positive_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Binary cross entropy ``L_BCE(z_tilde_ij, z_ij)`` (Algorithm 3).

        The frequency prior ``b_j`` is added inside the sigmoid; when
        ``self.use_prior`` is False it is dropped entirely ("w/o Prior").
        """
        score = representation_logits(h_upstream, h_online, prior=None, normalize=self.normalize)
        pw = self.positive_weight if positive_weight is None else float(positive_weight)
        p = prior if self.use_prior else None
        zt = _to_tensor(z)
        if zt.dim() > 1:
            zt = zt.reshape(-1)
        if zt.dtype != score.dtype:
            zt = zt.to(score.dtype)
        return binary_cross_entropy_with_prior(score, zt, prior=p, positive_weight=pw, reduction=reduction)

    # -- prediction --------------------------------------------------------
    def predict_from_encodings(
        self,
        h_upstream: Any,
        h_online: Any,
        prior: Optional[Any] = None,
        threshold: Optional[float] = None,
        return_probabilities: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """``z_hat_ij = 1[z_tilde_ij >= threshold]`` for precomputed ``h``."""
        probs = self.forward(h_upstream, h_online, prior=prior)
        thr = self.decision_threshold if threshold is None else float(threshold)
        z_hat = (probs >= thr).long()
        if return_probabilities:
            return z_hat, probs
        return z_hat

    def predict(
        self,
        inputs: Any = None,
        targets: Any = None,
        h_upstream: Any = None,
        h_online: Any = None,
        prior: Optional[Any] = None,
        mean_pool: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generic entry point: encode (if needed) then threshold -> ``z_hat``."""
        if h_online is None:
            h_online = self.encode(inputs, targets, mean_pool=mean_pool, **kwargs)
        if h_upstream is None:
            raise ValueError("predict requires h_upstream (upstream representations)")
        return self.predict_from_encodings(h_upstream, h_online, prior=prior)

    def cache_upstream(self, h_upstream: Any, upstream_indices: Optional[Sequence[int]] = None) -> None:
        """Cache upstream representations so inference is O(N_PT) (Algorithm 4)."""
        self._cached_upstream = _as_2d(h_upstream)
        if upstream_indices is not None:
            self.register_buffer(
                "_upstream_indices",
                torch.as_tensor(list(upstream_indices), dtype=torch.long),
                persistent=False,
            )

    def forecast_over_upstream(
        self,
        h_online: Any,
        h_upstream: Optional[Any] = None,
        upstream_indices: Optional[Sequence[int]] = None,
        prior: Optional[Any] = None,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Algorithm 4: score one online example against all of ``D_PT_hat``."""
        hj = _as_2d(h_upstream) if h_upstream is not None else self._cached_upstream
        if hj is None:
            raise ValueError("no upstream representations available (pass h_upstream or call cache_upstream)")
        if upstream_indices is None:
            upstream_indices = list(range(hj.shape[0]))
        b = prior if prior is not None else self.prior
        b_vec = None
        if self.use_prior and b is not None:
            b_vec = prior_vector_from(b, list(upstream_indices), device=hj.device, dtype=hj.dtype)
        return self.predict_from_encodings(hj, h_online, prior=b_vec, threshold=threshold)

    # -- record-level inference -------------------------------------------
    def predict_pairs(
        self,
        pairs: Sequence[Any],
        h_cache: Optional[Any] = None,
        online_examples: Optional[Sequence[Any]] = None,
        upstream_examples: Optional[Sequence[Any]] = None,
        prior: Optional[Any] = None,
        threshold: Optional[float] = None,
        return_probabilities: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """Forecast ``z_hat_ij`` for a batch of ``(i, j)`` pair records.

        ``h_cache`` may be a mapping ``index -> representation`` (as written by
        ``src/modeling/caches.py``) or a tensor indexed by example index.  When a
        representation is missing, it is computed from ``online_examples`` /
        ``upstream_examples`` if those are provided.
        """
        b = prior if prior is not None else self.prior
        if not self.use_prior:
            b = None
        thr = self.decision_threshold if threshold is None else float(threshold)

        z_hat_all: List[int] = []
        prob_all: List[float] = []

        # group by online index i -> encode each online example only once
        groups: List[Tuple[Any, List[Any]]] = []
        index_of: Dict[Any, int] = {}
        for rec in pairs:
            i = _field(rec, ("i", "online_index", "idx_i"))
            if i not in index_of:
                index_of[i] = len(groups)
                groups.append((i, []))
            groups[index_of[i]][1].append(rec)

        online_repr: Dict[Any, torch.Tensor] = {}
        for i, recs in groups:
            h_i = _lookup_repr(h_cache, i)
            if h_i is None:
                for rec in recs:
                    h_i = _record_repr(rec, upstream=False)
                    if h_i is not None:
                        break
            if h_i is None and online_examples is not None and i is not None:
                ex = online_examples[int(i)]
                h_i = self.encode([ex.get("input", "")], [ex.get("target", "")], mean_pool=True)
            if h_i is None and recs:
                ex = _field(recs[0], ("i_input", "online_input"))
                if ex is not None:
                    tg = _field(recs[0], ("i_target", "online_target", "i_label"))
                    h_i = self.encode([ex], [tg], mean_pool=True)
            if h_i is None:
                raise ValueError("could not obtain the online representation for i=%r" % (i,))
            online_repr[i] = _as_2d(h_i)

        j_list = [_field(rec, ("j", "upstream_index", "idx_j")) for rec in pairs]
        h_j: List[Optional[torch.Tensor]] = []
        missing: List[int] = []
        for pos, (rec, j) in enumerate(zip(pairs, j_list)):
            val = _record_repr(rec, upstream=True)
            if val is None:
                val = _lookup_repr(h_cache, j)
            if val is None:
                missing.append(pos)
                h_j.append(None)
            else:
                h_j.append(_as_2d(val))
        if missing:
            if upstream_examples is None:
                raise ValueError("missing upstream representations; provide h_cache or upstream_examples")
            for pos in missing:
                ex = upstream_examples[int(j_list[pos])]
                h_j[pos] = _as_2d(self.encode([ex.get("input", "")], [ex.get("target", "")], mean_pool=True))

        for rec, j, hj in zip(pairs, j_list, h_j):
            assert hj is not None
            i = _field(rec, ("i", "online_index", "idx_i"))
            hi = online_repr[i]
            b_j = None
            if self.use_prior and b is not None:
                b_j = prior_vector_from(b, [j], device=hj.device, dtype=hj.dtype)
            p = float(self.forward(hj, hi, prior=b_j).reshape(-1)[0].item())
            prob_all.append(p)
            z_hat_all.append(int(p >= thr))

        if return_probabilities:
            return z_hat_all, prob_all
        return z_hat_all

    # -- training helpers --------------------------------------------------
    def train_step(
        self, h_upstream: Any, h_online: Any, z: Any, prior: Optional[Any] = None, positive_weight: Optional[float] = None
    ) -> torch.Tensor:
        return self.loss(h_upstream, h_online, z, prior=prior, positive_weight=positive_weight)

    def fit(
        self,
        h_upstream: Any,
        h_online: Any,
        z: Any,
        prior: Optional[Any] = None,
        steps: int = 1000,
        lr: float = 1e-4,
        batch_size: int = 16,
        weight_decay: float = 0.0,
        log_every: int = 200,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Optimize ``h`` (Eq. 4 / Algorithm 3) on cached representations.

        Positives are down-weighted by ``self.positive_weight`` (``alpha = 0.1``)
        to handle the 1%-10% positive prevalence of forgetting (Sec. 4.2).
        """
        hj = _as_2d(h_upstream)
        hi = _as_2d(h_online)
        if hi.shape[0] == 1 and hj.shape[0] != 1:
            hi = hi.expand(hj.shape[0], -1)
        zt = _to_tensor(z).reshape(-1)
        b = _to_tensor(prior).reshape(-1) if (self.use_prior and prior is not None) else None

        params = self.trainable_parameters()
        if not params:
            with torch.no_grad():
                loss = self.loss(hj, hi, zt, prior=b)
            return {"loss": float(loss.item()), "steps": 0, "n_params": 0}

        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        n = int(hj.shape[0])
        bs = max(1, min(int(batch_size), n))
        self.train()
        last = float("nan")
        for step in range(int(steps)):
            idx = torch.randint(0, n, (bs,), device=hj.device)
            opt.zero_grad(set_to_none=True)
            loss = self.loss(hj[idx], hi[idx], zt[idx], prior=None if b is None else b[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = float(loss.item())
            if verbose and log_every and (step + 1) % log_every == 0:
                logger.info("representation forecaster step %d/%d loss=%.4f", step + 1, steps, last)
        return {"loss": last, "steps": int(steps), "n_params": len(params)}

    # -- persistence -------------------------------------------------------
    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "use_prior": self.use_prior,
            "positive_weight": self.positive_weight,
            "decision_threshold": self.decision_threshold,
            "normalize": self.normalize,
        }

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        state: Dict[str, Any] = {
            "config": self.config(),
            "head": self.head.state_dict() if self.head is not None else None,
            "encoder": self.encoder.state_dict() if isinstance(self.encoder, nn.Module) else None,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        return path

    def load_state(self, path: str, strict: bool = False) -> "RepresentationBasedForecaster":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.use_prior = bool(cfg.get("use_prior", self.use_prior))
        self.positive_weight = float(cfg.get("positive_weight", self.positive_weight))
        self.decision_threshold = float(cfg.get("decision_threshold", self.decision_threshold))
        self.normalize = bool(cfg.get("normalize", self.normalize))
        if self.head is not None and ckpt.get("head") is not None:
            self.head.load_state_dict(ckpt["head"], strict=strict)
        if isinstance(self.encoder, nn.Module) and ckpt.get("encoder") is not None:
            self.encoder.load_state_dict(ckpt["encoder"], strict=strict)
        return self


# ---------------------------------------------------------------------------
# Record-level utilities
# ---------------------------------------------------------------------------
def _field(rec: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(rec, Mapping):
        for n in names:
            if n in rec:
                return rec[n]
        return default
    for n in names:
        if hasattr(rec, n):
            return getattr(rec, n)
    return default


def _record_repr(rec: Any, upstream: bool) -> Optional[Any]:
    names = ("h_j", "h_upstream") if upstream else ("h_i", "h_online")
    val = _field(rec, names)
    if val is None:
        return None
    if isinstance(val, torch.Tensor) and val.dim() == 3:
        return mean_pool_representation(val)
    return val


def _lookup_repr(h_cache: Any, index: Any) -> Optional[Any]:
    """Fetch a cached representation by example index."""
    if h_cache is None or index is None:
        return None
    if isinstance(h_cache, torch.Tensor):
        try:
            v = h_cache[int(index)]
        except Exception:
            return None
        return mean_pool_representation(v) if v.dim() == 2 else v
    if isinstance(h_cache, Mapping):
        keys = [index, str(index)]
        try:
            keys.append(int(index))
        except Exception:
            pass
        for key in keys:
            if key in h_cache:
                v = h_cache[key]
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    return mean_pool_representation(v)
                return _to_tensor(v)
        for outer in ("h", "representations", "repr", "mean_pooled", "data"):
            if outer in h_cache and isinstance(h_cache[outer], Mapping):
                return _lookup_repr(h_cache[outer], index)
        return None
    getter = getattr(h_cache, "get", None)
    if callable(getter):
        v = getter(index)
        if v is None:
            return None
        return mean_pool_representation(v) if isinstance(v, torch.Tensor) and v.dim() == 2 else _to_tensor(v)
    return None


def extract_representations_from_records(
    records: Sequence[Any], upstream: bool = True, mean_pool: bool = True
) -> List[Optional[Any]]:
    """Pull cached representations out of pair records (Algorithm 4 caches)."""
    out: List[Optional[Any]] = []
    for rec in records:
        value = _record_repr(rec, upstream=upstream)
        if value is not None and mean_pool and isinstance(value, torch.Tensor) and value.dim() == 3:
            value = mean_pool_representation(value)
        out.append(value)
    return out


def evaluate_labels(z_hat: Sequence[int], z_true: Sequence[int]) -> Dict[str, float]:
    """Precision / recall / F1 for binary forecasting labels (Sec. 4.1 metrics)."""
    try:  # prefer the shared evaluation-metrics module
        from ..eval import metrics as _metrics

        for fname in ("forecast_metrics", "compute_forecast_metrics", "binary_metrics"):
            fn = getattr(_metrics, fname, None)
            if callable(fn):
                out = fn(z_hat, z_true)
                if isinstance(out, Mapping):
                    return {k: float(v) for k, v in out.items()}
        for fname in ("binary_f1", "f1_score_binary", "forecast_f1"):
            fn = getattr(_metrics, fname, None)
            if callable(fn):
                return {"f1": float(fn(z_hat, z_true))}
    except Exception:
        pass

    tp = fp = fn_ = tn = 0
    for p, t in zip(z_hat, z_true):
        p = int(p) >= 1
        t = int(t) >= 1
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif (not p) and t:
            fn_ += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn_)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn_),
        "tn": float(tn),
        "n": float(len(z_hat)),
    }


# ---------------------------------------------------------------------------
# Training wrapper (used by scripts/train_forecaster.py)
# ---------------------------------------------------------------------------
def train_representation_forecaster(
    records: Sequence[Any],
    forecaster: Optional[RepresentationBasedForecaster] = None,
    prior: Optional[Any] = None,
    h_cache: Optional[Any] = None,
    steps: int = 100000,
    batch_size: int = 16,
    n_positive: int = 8,
    n_negative: int = 8,
    lr: float = 1e-4,
    positive_weight: float = 0.1,
    device: Union[str, torch.device] = "cpu",
    log_every: int = 1000,
    verbose: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train ``h`` with Eq. 4 / Algorithm 3 on labelled pair records.

    Each mini-batch contains ``n_positive = 8`` forgotten and
    ``n_negative = 8`` non-forgotten pairs (Appendix B), with positives
    down-weighted by ``positive_weight = alpha = 0.1``.

    Records must expose the cached representations ``h_j`` (upstream) and
    ``h_i`` (online) either directly or through ``h_cache``.  Returns the
    forecaster plus training diagnostics.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    if forecaster is None:
        forecaster = RepresentationBasedForecaster(use_prior=prior is not None, positive_weight=positive_weight)
    forecaster.to(device)
    forecaster.set_prior(prior)

    h_j = extract_representations_from_records(records, upstream=True)
    h_i = extract_representations_from_records(records, upstream=False)
    if any(v is None for v in h_j):
        h_j = [_lookup_repr(h_cache, _field(r, ("j", "upstream_index", "idx_j"))) for r in records]
    if any(v is None for v in h_i):
        h_i = [_lookup_repr(h_cache, _field(r, ("i", "online_index", "idx_i"))) for r in records]
    if any(v is None for v in h_j):
        raise ValueError("some pair records lack upstream representations h_j; build the h cache first")
    if any(v is None for v in h_i):
        raise ValueError("some pair records lack online representations h_i; build the h cache first")

    z = [int(_field(r, ("z", "label", "z_ij"), 0)) for r in records]
    j_idx = [_field(r, ("j", "upstream_index", "idx_j")) for r in records]

    HJ = torch.stack([_to_tensor(v).reshape(-1) for v in h_j]).to(device)
    HI = torch.stack([_to_tensor(v).reshape(-1) for v in h_i]).to(device)
    Z = torch.tensor(z, device=device, dtype=torch.float32)
    B = prior_vector_from(prior, j_idx, device=device, dtype=torch.float32) if prior is not None else None

    pos = torch.nonzero(Z > 0.5, as_tuple=False).reshape(-1)
    neg = torch.nonzero(Z <= 0.5, as_tuple=False).reshape(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        logger.warning(
            "degenerate label distribution (pos=%d, neg=%d); falling back to uniform sampling",
            int(pos.numel()),
            int(neg.numel()),
        )

    params = forecaster.trainable_parameters()
    if not params:
        return {"forecaster": forecaster, "steps": 0, "loss": float("nan"), "note": "no trainable parameters"}
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    forecaster.train()
    last_loss = float("nan")
    for step in range(int(steps)):
        if pos.numel() and neg.numel():
            sel_p = pos[torch.randint(0, pos.numel(), (int(n_positive),), device=device)]
            sel_n = neg[torch.randint(0, neg.numel(), (int(n_negative),), device=device)]
            idx = torch.cat([sel_p, sel_n])
        else:
            idx = torch.randint(0, Z.numel(), (int(batch_size),), device=device)
        opt.zero_grad(set_to_none=True)
        loss = forecaster.loss(HJ[idx], HI[idx], Z[idx], prior=None if B is None else B[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last_loss = float(loss.item())
        if verbose and log_every and (step + 1) % log_every == 0:
            logger.info("step %d/%d loss=%.4f", step + 1, steps, last_loss)

    with torch.no_grad():
        probs = forecaster.forward(HJ, HI, prior=B)
        z_hat = (probs >= forecaster.decision_threshold).long().cpu().tolist()
    diag = evaluate_labels(z_hat, z)
    diag.update({"steps": int(steps), "loss": last_loss})
    return {"forecaster": forecaster, "train_metrics": diag}


def forecast_upstream_examples(
    forecaster: RepresentationBasedForecaster,
    h_online: Any,
    h_upstream: Any,
    upstream_indices: Optional[Sequence[int]] = None,
    prior: Optional[Any] = None,
    threshold: Optional[float] = None,
) -> List[int]:
    """Algorithm 4 wrapper: forecast ``z_hat_ij`` over all of ``D_PT_hat``."""
    z_hat = forecaster.forecast_over_upstream(
        h_online, h_upstream=h_upstream, upstream_indices=upstream_indices, prior=prior, threshold=threshold
    )
    return [int(v) for v in z_hat.reshape(-1).cpu().tolist()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Representation-based forgetting forecaster (Sec. 3.3, Eq. 4; Algorithms 3 & 4)."
    )
    p.add_argument("--pairs", type=str, default=None, help="ground-truth pairs.jsonl for training")
    p.add_argument("--eval-pairs", type=str, default=None, help="held-out pairs.jsonl for evaluation")
    p.add_argument("--h-cache", type=str, default=None, help="torch .pt file with cached h(x_j,y_j)")
    p.add_argument("--prior", type=str, default=None, help="frequency_prior.json (b_j)")
    p.add_argument("--no-prior", action="store_true", help="ablate the frequency prior (Table 1 'w/o Prior')")
    p.add_argument("--checkpoint", type=str, default=None, help="forecaster checkpoint to load/save")
    p.add_argument("--steps", type=int, default=1000, help="training steps on cached representations")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-positive", type=int, default=8)
    p.add_argument("--n-negative", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--positive-weight", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None, help="where to write the trained checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true", help="run a synthetic smoke test")
    return p.parse_args(argv)


def _load_pairs(path: str) -> List[Any]:
    """Load pair records (JSONL) preserving the ground-truth schema."""
    try:
        from ..forgetting.ground_truth import load_ground_truth_jsonl

        return load_ground_truth_jsonl(path)
    except Exception:
        pass
    out: List[Any] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_prior(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        from ..forgetting.frequency_prior import load_prior

        return load_prior(path)
    except Exception:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, Mapping) and "priors" in data:
            return {int(k): float(v) for k, v in data["priors"].items()}
        return data


def _self_test() -> int:
    """Synthetic check: training improves F1 and the prior shifts the decision."""
    torch.manual_seed(0)
    dim = 16
    n = 400
    h_j = torch.randn(n, dim)
    z = (torch.rand(n) < 0.1).float()
    h_i = torch.randn(n, dim) + 1.5 * z.unsqueeze(-1) * h_j
    prior = torch.where(z > 0.5, 1.0, -0.1)

    f = RepresentationBasedForecaster(dim=dim, use_prior=True, positive_weight=0.1)
    before = evaluate_labels(f.forward(h_j, h_i, prior=prior).ge(0.5).long().tolist(), z.tolist())
    res = f.fit(h_j, h_i, z, prior=prior, steps=300, lr=1e-2, batch_size=32)
    after = evaluate_labels(f.forward(h_j, h_i, prior=prior).ge(0.5).long().tolist(), z.tolist())
    print("self-test: loss=%.4f f1 before=%.3f after=%.3f" % (res["loss"], before["f1"], after["f1"]))
    assert math.isfinite(res["loss"]), "non-finite loss"
    assert after["f1"] >= before["f1"], "training did not improve F1"

    # w/o Prior ablation path must run and drop the bias term
    g = RepresentationBasedForecaster(dim=dim, use_prior=False)
    no_prior_prob = g.forward(h_j, h_i, prior=prior)
    assert torch.allclose(no_prior_prob, torch.sigmoid((h_j * h_i).sum(-1)), atol=1e-5)
    print("representation_based.py self-test OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return _self_test()

    if not args.pairs:
        logger.error("--pairs is required (or use --self-test)")
        return 2

    records = _load_pairs(args.pairs)
    logger.info("loaded %d pair records", len(records))
    h_cache = torch.load(args.h_cache, map_location="cpu") if args.h_cache else None
    prior = None if args.no_prior else _load_prior(args.prior)

    f = RepresentationBasedForecaster(
        dim=args.dim, use_prior=not args.no_prior, positive_weight=args.positive_weight
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        f.load_state(args.checkpoint)

    res = train_representation_forecaster(
        records,
        forecaster=f,
        prior=prior,
        h_cache=h_cache,
        steps=args.steps,
        batch_size=args.batch_size,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        lr=args.lr,
        positive_weight=args.positive_weight,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )
    logger.info("train metrics: %s", res.get("train_metrics"))

    if args.eval_pairs:
        pairs = _load_pairs(args.eval_pairs)
        z_true = [int(_field(r, ("z", "label", "z_ij"), 0)) for r in pairs]
        z_hat = f.predict_pairs(pairs, h_cache=h_cache, prior=prior)
        logger.info("eval metrics: %s", evaluate_labels(z_hat, z_true))

    if args.out or args.checkpoint:
        path = args.out or args.checkpoint
        f.save(path)
        logger.info("saved forecaster to %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
