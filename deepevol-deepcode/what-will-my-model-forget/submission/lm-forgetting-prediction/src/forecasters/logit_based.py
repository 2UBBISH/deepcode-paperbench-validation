"""Logit-change-transfer forecasting (Sec. 3.2, Eq. 2 and Eq. 3 of the paper).

Paper recap
-----------
For one gradient step on the online learned example ``<x_i, y_i>`` the logit change of an upstream
pretraining example ``<x_j, y_j>`` is approximated by a first-order Taylor expansion as

    f_i(x_j) - f_0(x_j) = Theta(x_j, x_i) Theta^{-1}(x_i, x_i) [ f_i(x_i) - f_0(x_i) ]      (Eq. 2)

The trainable variant replaces ``Theta(x_j, x_i) Theta^{-1}(x_i, x_i)`` with a low dimensional
trainable kernel

    Theta_tilde(x_j, x_i) = h(x_j, y_j) h(x_i, y_i)^T   in  R^{T x T}

where ``h: x, y -> R^{T x d}`` encodes the concatenation of input and output tokens (``T`` is the
output length) and the huge ``30k-50k`` dimensional vocabulary space is removed from the kernel.
The predicted logits of ``x_j`` under the updated model are then

    f_hat_i(x_j) = Theta_tilde(x_j, x_i) [ f_hat_i(x_i) - f_hat_0(x_i) ] + f_hat_0(x_j)

and ``h`` is trained with the margin loss (Eq. 3)

    L = max(0, 1 + (-1)^{z_ij} ( max_{v != y_j} f_hat_i(x_j)[v] - f_hat_i(x_j)[y_j] ))

so that the correct token ``y_j`` exceeds the runner-up candidate by a preset margin if
``<x_j, y_j>`` is *not* forgotten, and the ordering is reversed otherwise.

Fixed variant (Sec. 4.2)
------------------------
``FixedLogitForecaster`` is the non-trained baseline that "replaces trainable encoding function h
with the frozen final layer representation of the base PTLM".  The resulting kernel is identical to
the ground-truth kernel when only the final LM heads are tuned (Sec. 3.2), because
``grad_{W_Head} f_0(x_j)`` is simply the representation of ``x_j`` before the LM head.

Efficient inference (Sec. 3.2 + Algorithm 2)
-------------------------------------------
No repeated full inference with the LM ``f`` is required: the logits of pretraining examples before
model updates ``f_0(x_j)`` can be computed once and cached for different online examples ``x_i``,
and so can the representations ``h(x_i, y_i)``.  "In practice, we only cache top k = 100 largest
logits for each token in y_j".  Accordingly, every helper below works over a per-output-token
*candidate set* of vocabulary ids (the union of the cached top-k indices and the gold token ids);
logits outside the set are treated as ``NEG_INF`` for ``f_0(x_j)`` and as a zero delta for the change
matrix, which is the neutral reading of a top-k cache.

Forgetting definition (flagged discrepancy)
-------------------------------------------
Algorithms 1/3 in Appendix F write ``z_ij = 1 if f_0(x_i) != f_i(x_i) else 0`` (whether the error of
the *online* example was fixed), while Sec. 2 defines the forecasting target as whether the
*upstream* example got forgotten: ``z_ij = 1[f_i(x_j) != y_j]`` with
``D_PT^Fgt,i = {<x_j,y_j> in D_PT_hat | f_i(x_j) != y_j}``.  Following the reproduction plan (and the
plan's explicit note that Appendix F contains a typo), all ``z`` labels here are supplied by the
caller and come from ``src.forgetting.ground_truth`` which implements the Sec. 2 definition.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:  # losses.py lives in the same package
    from .losses import (  # type: ignore
        NEG_INF,
        margin_loss_from_topk,
    )
except Exception:  # pragma: no cover - tolerate standalone import
    NEG_INF = -1e4

    def margin_loss_from_topk(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError(
            "src.forecasters.losses is required for the Eq. 3 margin loss "
            "(margin_loss_from_topk)."
        )

logger = logging.getLogger("logit_based")

DEFAULT_TOPK = 100  # Sec. 3.2: "we only cache top k = 100 largest logits for each token in y_j"
DEFAULT_FILENAME = "logit_forecaster.pt"

__all__ = [
    "DEFAULT_TOPK",
    "DEFAULT_FILENAME",
    "NEG_INF",
    "LogitChangeTransferForecaster",
    "FixedLogitForecaster",
    "logit_change_kernel",
    "forecast_logits",
    "build_candidate_indices",
    "densify_topk",
    "make_delta_matrix",
    "predicted_label",
    "forecast_pair_from_cache",
    "pair_margin_objective",
    "pair_margin_objective_from_topk",
    "to_topk",
    "evaluate_labels",
]


# ======================================================================================
# sparse <-> dense helpers (the top-k cache is a sparse per-output-token logit view)
# ======================================================================================
def _rows_from(container: Any) -> List[List[Any]]:
    """Normalise a per-output-token container into ``List[List[Any]]`` (tensors -> lists)."""
    if container is None:
        return []
    if torch.is_tensor(container):
        if container.dim() == 1:
            return [[v for v in container.cpu().tolist()]]
        return [[v for v in row] for row in container.cpu().tolist()]
    if isinstance(container, (list, tuple)):
        out: List[List[Any]] = []
        for row in container:
            if row is None:
                out.append([])
            elif torch.is_tensor(row):
                out.append([v for v in row.cpu().tolist()])
            elif isinstance(row, (list, tuple)):
                out.append(list(row))
            else:
                out.append([row])
        return out
    return [[container]]


def to_topk(values: torch.Tensor, k: int = DEFAULT_TOPK) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` (values, indices) along the last (vocabulary) dimension.

    ``values`` of shape ``[..., V]`` returns ``([..., k], [..., k])``.  Implements the caching rule
    of Sec. 3.2.
    """
    k = int(min(k, values.shape[-1]))
    return torch.topk(values, k=k, dim=-1, largest=True, sorted=True)


def build_candidate_indices(
    topk_indices: Sequence[Any],
    target_ids: Optional[Sequence[Any]] = None,
    max_candidates: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Sorted unique vocabulary ids forming the candidate space of one example.

    Parameters
    ----------
    topk_indices:
        Iterable of per-output-token index containers (lists or tensors), e.g. the cached top-k
        indices of ``f_0(x_j)``, ``f_0(x_i)`` and ``f_i(x_i)``.
    target_ids:
        Gold token ids (for ``y_i`` and/or ``y_j``) which must always be in the candidate set so that
        ``f_hat_i(x_j)[y_j]`` is recoverable from a top-k cache.
    max_candidates:
        Optional safety cap (keeps the lowest ids).
    vocab_size:
        Optional upper bound so that padding labels (e.g. ``-100``) or out-of-range ids are dropped.
    """
    cand: set = set()
    for container in topk_indices:
        if container is None:
            continue
        for row in _rows_from(container):
            for v in row:
                if v is None:
                    continue
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                if v < 0:
                    continue
                if vocab_size is not None and v >= int(vocab_size):
                    continue
                cand.add(v)
    if target_ids is not None:
        if torch.is_tensor(target_ids):
            target_ids = target_ids.cpu().tolist()
            if isinstance(target_ids, int):
                target_ids = [target_ids]
        for t in target_ids:
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t < 0:
                continue
            if vocab_size is not None and t >= int(vocab_size):
                continue
            cand.add(t)
    if not cand:
        return torch.empty(0, dtype=torch.long)
    idx = sorted(cand)
    if max_candidates is not None and len(idx) > int(max_candidates):
        idx = idx[: int(max_candidates)]
    return torch.tensor(idx, dtype=torch.long)


def densify_topk(
    topk_indices: Any,
    topk_values: Any,
    candidates: torch.Tensor,
    fill_value: float = NEG_INF,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scatter a sparse per-token top-k view onto a dense ``[T, S]`` tensor over ``candidates``.

    ``candidates`` must be sorted ascending and unique (see :func:`build_candidate_indices`); ids
    absent from the cache receive ``fill_value``.
    """
    idx_rows = _rows_from(topk_indices)
    val_rows = _rows_from(topk_values)
    T = len(idx_rows)
    S = int(candidates.numel())
    out = torch.full((T, max(S, 0)), float(fill_value), dtype=dtype)
    if S == 0 or T == 0:
        return out
    cand = candidates.to(torch.long)
    # vectorised searchsorted per row
    for t in range(T):
        row_idx = idx_rows[t] if t < len(idx_rows) else []
        row_val = val_rows[t] if t < len(val_rows) else []
        if not row_idx:
            continue
        ii = torch.tensor([int(v) for v in row_idx], dtype=torch.long)
        if len(row_val) == len(row_idx):
            vv = torch.tensor([float(v) for v in row_val], dtype=dtype)
        else:
            vv = torch.full((ii.numel(),), float(fill_value), dtype=dtype)
        valid = (ii >= 0) & (ii <= int(cand.max()))
        ii, vv = ii[valid], vv[valid]
        if ii.numel() == 0:
            continue
        pos = torch.searchsorted(cand, ii)
        pos = torch.clamp(pos, 0, S - 1)
        hit = cand[pos] == ii
        if bool(hit.any()):
            out[t, pos[hit]] = vv[hit]
    return out


def make_delta_matrix(
    f0_topk_indices: Any,
    f0_topk_values: Any,
    fi_topk_indices: Any,
    fi_topk_values: Any,
    candidates: torch.Tensor,
    zero_missing: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``delta = f_i(x_i) - f_0(x_i)`` as a dense ``[T_i, S]`` tensor over ``candidates``.

    Cached logits only keep the per-token top-k, so the change is unknown for ids missing from
    either stream.  ``zero_missing=True`` (default) treats those changes as zero, the neutral choice
    for a sparse top-k approximation; set ``False`` to mark them ``NEG_INF`` instead.
    """
    f0 = densify_topk(f0_topk_indices, f0_topk_values, candidates, fill_value=NEG_INF, dtype=dtype)
    fi = densify_topk(fi_topk_indices, fi_topk_values, candidates, fill_value=NEG_INF, dtype=dtype)
    known = (f0 > NEG_INF / 2) & (fi > NEG_INF / 2)
    delta = torch.zeros_like(f0)
    delta[known] = fi[known] - f0[known]
    if not zero_missing:
        delta[~known] = float(NEG_INF)
    return delta


# ======================================================================================
# kernel and logit prediction (Eq. 2)
# ======================================================================================
def logit_change_kernel(
    h_j: torch.Tensor,
    h_i: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """``Theta_tilde(x_j, x_i) = h(x_j, y_j) h(x_i, y_i)^T`` of shape ``R^{T_j x T_i}``.

    ``h_j``: ``[..., T_j, d]`` and ``h_i``: ``[..., T_i, d]``.  Optional L2 row normalisation gives
    a cosine kernel; the paper uses the plain inner product (``normalize=False`` default).
    """
    if normalize:
        h_j = nn.functional.normalize(h_j.float(), dim=-1)
        h_i = nn.functional.normalize(h_i.float(), dim=-1)
    return torch.matmul(h_j.float(), h_i.float().transpose(-1, -2))


def forecast_logits(
    theta: torch.Tensor,
    delta_xi: torch.Tensor,
    f0_xj: torch.Tensor,
) -> torch.Tensor:
    """``f_hat_i(x_j) = Theta_tilde(x_j, x_i) [f_hat_i(x_i) - f_hat_0(x_i)] + f_hat_0(x_j)``.

    ``theta``: ``[..., T_j, T_i]``; ``delta_xi``: ``[..., T_i, S]``; ``f0_xj``: ``[..., T_j, S]``,
    where ``S`` is the candidate vocabulary size.
    """
    return torch.matmul(theta.float(), delta_xi.float()) + f0_xj.float()


def predicted_label(
    pred_logits: torch.Tensor,
    candidates: torch.Tensor,
    target_ids: Sequence[Any],
    target_mask: Optional[Sequence[Any]] = None,
) -> torch.Tensor:
    """``z_hat = 1[argmax_v f_hat_i(x_j)[v] != y_j]`` (Algorithm 2), over the candidate set."""
    cand = candidates.to(torch.long)
    arg = torch.argmax(pred_logits, dim=-1)
    top_id = cand[arg]
    tg = torch.tensor(
        [int(t) if t is not None else -1 for t in target_ids],
        dtype=torch.long,
        device=top_id.device,
    )
    n = min(int(top_id.numel()), int(tg.numel()))
    wrong = (top_id[:n] != tg[:n]).to(torch.long)
    if target_mask is not None:
        mask = torch.tensor(
            [int(m) for m in target_mask], dtype=torch.long, device=wrong.device
        )
        n = min(int(wrong.numel()), int(mask.numel()))
        wrong = wrong[:n] * mask[:n]
    return wrong


# ======================================================================================
# forecaster modules
# ======================================================================================
class LogitChangeTransferForecaster(nn.Module):
    """Partially interpretable trainable logit-change-transfer forecaster (Sec. 3.2).

    The model is *partially interpretable* in that it explains how the logit change of
    ``<x_i, y_i>`` is transferred to ``<x_j, y_j>`` depending on their learned similarity, i.e. the
    kernel ``Theta_tilde`` (Figure 2(b)).

    Parameters
    ----------
    encoder:
        Trainable encoding function ``h``.  Either an ``nn.Module`` exposing one of
        ``encode_token_level`` / ``encode_token`` / ``encode`` / ``forward`` mapping
        ``(inputs, targets) -> [T, d]``, or a plain callable.  May be ``None`` when the caller
        supplies pre-computed (cached) representations.
    dim:
        Dimensionality ``d`` of the encoding (introspection / defaults only).
    fixed:
        ``True`` builds the non-trained fixed-logit baseline: ``h`` is the frozen final layer
        representation of the base PTLM and its parameters have ``requires_grad=False``.
    normalize:
        Use a cosine kernel instead of the plain inner product (default ``False``).
    topk:
        Cache size ``k`` for the logits (default 100, as in the paper).
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        dim: int = 768,
        fixed: bool = False,
        normalize: bool = False,
        encode_batch_size: int = 8,
        topk: int = DEFAULT_TOPK,
        name: str = "logit_based",
    ) -> None:
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.Module) else None
        self._encoder_fn = None if isinstance(encoder, nn.Module) else encoder
        self.fixed = bool(fixed)
        self.normalize = bool(normalize)
        self.dim = int(dim)
        self.encode_batch_size = int(encode_batch_size)
        self.topk = int(topk)
        self.name = name
        if self.fixed and self.encoder is not None:
            for p in self.encoder.parameters():  # frozen base-PTLM representation
                p.requires_grad_(False)

    # -------------------------------- encoding ---------------------------------------
    def encode(self, inputs: Any, targets: Any = None, **kwargs: Any) -> torch.Tensor:
        """Call the encoding function ``h(x, y) -> [T, d]`` (token level).

        For the fixed variant with a base LM, ``src.modeling.base_lm`` exposes the frozen final
        layer representation through one of the recognised method names; for the trainable variant
        this dispatches to ``src.modeling.encoder_h`` (base LM + 2-layer MLP).
        """
        if self._encoder_fn is not None:
            return self._encoder_fn(inputs, targets, **kwargs)
        if self.encoder is None:
            raise ValueError(
                "LogitChangeTransferForecaster was built without an encoder; pass an encoder to the "
                "constructor or use `set_encoder`."
            )
        for attr in ("encode_token_level", "encode_token", "encode", "forward"):
            fn = getattr(self.encoder, attr, None)
            if callable(fn):
                try:
                    return fn(inputs, targets, **kwargs)
                except TypeError:
                    return fn(inputs, targets=targets, **kwargs)
        if callable(self.encoder):
            return self.encoder(inputs, targets, **kwargs)
        raise TypeError(
            "encoder exposes none of encode_token_level/encode_token/encode/forward and is not "
            "callable"
        )

    def set_encoder(self, encoder: Any) -> None:
        """Attach / replace the encoding function ``h``."""
        self.encoder = encoder if isinstance(encoder, nn.Module) else None
        self._encoder_fn = None if isinstance(encoder, nn.Module) else encoder
        if self.fixed and self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    # ------------------------------ forward pieces -----------------------------------
    def kernel(self, h_j: torch.Tensor, h_i: torch.Tensor) -> torch.Tensor:
        """``Theta_tilde(x_j, x_i)`` — the learned similarity measurement of Figure 2(b)."""
        return logit_change_kernel(h_j, h_i, normalize=self.normalize)

    def forward(
        self,
        h_j: torch.Tensor,
        h_i: torch.Tensor,
        delta_xi: torch.Tensor,
        f0_xj: torch.Tensor,
    ) -> torch.Tensor:
        """Return the predicted logits ``f_hat_i(x_j)`` given encodings and cached logits."""
        return forecast_logits(self.kernel(h_j, h_i), delta_xi, f0_xj)

    def forecast_from_encodings(
        self,
        h_j: torch.Tensor,
        h_i: torch.Tensor,
        delta_xi: torch.Tensor,
        f0_xj: torch.Tensor,
    ) -> torch.Tensor:
        """Alias of :meth:`forward` for explicit call sites."""
        return self.forward(h_j, h_i, delta_xi, f0_xj)

    # ---------------------------------- loss ------------------------------------------
    def loss(
        self,
        pred_logits: torch.Tensor,
        candidates: torch.Tensor,
        target_ids: Sequence[Any],
        z: Union[int, float, Sequence[float], torch.Tensor],
        margin: float = 1.0,
        reduction: str = "mean",
        target_mask: Optional[Sequence[Any]] = None,
    ) -> torch.Tensor:
        """Eq. 3 margin loss (delegates to :func:`pair_margin_objective`)."""
        return pair_margin_objective(
            pred_logits,
            candidates,
            target_ids,
            z,
            margin=margin,
            reduction=reduction,
            target_mask=target_mask,
        )

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Parameters to optimise: ``h`` only (never the frozen PTLM of the fixed variant)."""
        if self.fixed:
            return iter(())
        if self.encoder is not None:
            return (p for p in self.encoder.parameters() if p.requires_grad)
        return iter(())

    # -------------------------------- persistence ------------------------------------
    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "fixed": self.fixed,
            "normalize": self.normalize,
            "topk": self.topk,
            "encode_batch_size": self.encode_batch_size,
        }

    def save(self, path: str, extra: Optional[Mapping[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload: Dict[str, Any] = {"config": self.config()}
        if extra:
            payload.update(dict(extra))
        if self.encoder is not None:
            payload["state_dict"] = self.encoder.state_dict()
        torch.save(payload, path)
        logger.info("saved logit forecaster to %s", path)
        return path

    def load_state(self, path: str, strict: bool = False) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        if self.encoder is not None and isinstance(payload, dict) and "state_dict" in payload:
            missing, unexpected = self.encoder.load_state_dict(payload["state_dict"], strict=strict)
            if missing:
                logger.warning("missing keys when loading h: %d", len(missing))
            if unexpected:
                logger.warning("unexpected keys when loading h: %d", len(unexpected))
        if isinstance(payload, dict):
            payload.pop("config", None)
        return payload


class FixedLogitForecaster(LogitChangeTransferForecaster):
    """Non-trained fixed-logit baseline (Sec. 4.2).

    Replaces the trainable encoding function ``h`` with the *frozen final layer representation of
    the base PTLM*; the resulting kernel is identical to the ground-truth kernel when only the final
    LM heads are tuned.
    """

    def __init__(self, encoder: Optional[Any] = None, dim: int = 768, **kwargs: Any) -> None:
        kwargs.pop("fixed", None)
        kwargs.setdefault("name", "fixed_logit")
        super().__init__(encoder=encoder, dim=dim, fixed=True, **kwargs)

    @classmethod
    def from_base_lm(cls, base_lm: Any, dim: Optional[int] = None) -> "FixedLogitForecaster":
        """Build from a base LM object exposing a token-level encoder.

        Recognised attributes: ``encoder_h`` / ``token_encoder`` / ``representation_encoder``, or
        ``encode_token_level`` / ``final_hidden_states`` on the LM itself.
        """
        encoder = None
        for attr in ("encoder_h", "token_encoder", "representation_encoder"):
            cand = getattr(base_lm, attr, None)
            if cand is not None:
                encoder = cand
                break
        if encoder is None and (
            hasattr(base_lm, "encode_token_level") or hasattr(base_lm, "final_hidden_states")
        ):
            encoder = base_lm
        if encoder is None:
            logger.warning(
                "could not locate a token-level encoder on the base LM; the fixed forecaster will "
                "require explicitly supplied h(x, y)."
            )
        d = dim
        if d is None:
            for attr in ("dim", "hidden_size", "d_model"):
                val = getattr(encoder, attr, None) if encoder is not None else None
                if val is None and encoder is not None:
                    val = getattr(getattr(encoder, "config", None), attr, None)
                if val:
                    d = int(val)
                    break
        return cls(encoder=encoder, dim=d or 768)


# ======================================================================================
# Eq. 3 objective helpers
# ======================================================================================
def pair_margin_objective(
    pred_logits: torch.Tensor,
    candidates: torch.Tensor,
    target_ids: Union[Sequence[Any], torch.Tensor],
    z: Union[int, float, Sequence[float], torch.Tensor],
    margin: float = 1.0,
    reduction: str = "mean",
    target_mask: Optional[Sequence[Any]] = None,
) -> torch.Tensor:
    """Eq. 3 margin loss over a dense ``[T, S]`` candidate logit matrix.

    ``L = max(0, 1 + (-1)^z ( max_{v != y_j} f_hat_i(x_j)[v] - f_hat_i(x_j)[y_j] ))``
    """
    if margin_loss_from_topk is None:  # pragma: no cover
        raise ImportError("src.forecasters.losses is required for the Eq. 3 margin loss")
    cand = candidates.to(torch.long)
    k = int(min(max(pred_logits.shape[-1], 1), max(int(cand.numel()), 1)))
    vals, idxs = torch.topk(pred_logits, k=k, dim=-1)
    topk_indices = cand[idxs]
    return margin_loss_from_topk(
        vals,
        topk_indices,
        target_ids,
        z,
        margin=margin,
        default_logit=NEG_INF,
        reduction=reduction,
    )


def pair_margin_objective_from_topk(
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    target_ids: Union[Sequence[Any], torch.Tensor],
    z: Union[int, float, Sequence[float], torch.Tensor],
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq. 3 directly from cached top-k logits (``losses.margin_loss_from_topk``)."""
    return margin_loss_from_topk(
        topk_values,
        topk_indices,
        target_ids,
        z,
        margin=margin,
        default_logit=NEG_INF,
        reduction=reduction,
    )


# ======================================================================================
# end-to-end helper over cached artifacts (Algorithm 2)
# ======================================================================================
def forecast_pair_from_cache(
    forecaster: Optional[LogitChangeTransferForecaster],
    h_j: torch.Tensor,
    h_i: torch.Tensor,
    f0_xi_topk: Tuple[Any, Any],
    fi_xi_topk: Tuple[Any, Any],
    f0_xj_topk: Tuple[Any, Any],
    target_ids_i: Sequence[Any] = (),
    target_ids_j: Sequence[Any] = (),
    return_logits: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Forecast one ``(x_i, x_j)`` pair using only cached top-k logits and encodings.

    ``f0_xi_topk`` / ``fi_xi_topk`` / ``f0_xj_topk`` are ``(indices, values)`` pairs of
    per-output-token top-k logits.  Returns the per-token label
    ``z_hat = 1[argmax f_hat_i(x_j) != y_j]`` (Algorithm 2); with ``return_logits=True`` it also
    returns the candidate ids and the dense predicted logits.

    ``forecaster`` may be ``None``, in which case ``Theta_tilde`` is the identity-like plain inner
    product of the provided encodings (useful for the fixed variant when ``h`` is pre-computed).
    """
    f0_i_idx, f0_i_val = f0_xi_topk
    fi_i_idx, fi_i_val = fi_xi_topk
    f0_j_idx, f0_j_val = f0_xj_topk

    candidates = build_candidate_indices(
        [f0_i_idx, fi_i_idx, f0_j_idx],
        target_ids=list(target_ids_i) + list(target_ids_j),
    )
    delta = make_delta_matrix(f0_i_idx, f0_i_val, fi_i_idx, fi_i_val, candidates, zero_missing=True)
    f0_j = densify_topk(f0_j_idx, f0_j_val, candidates, fill_value=NEG_INF)

    if forecaster is None:
        theta = logit_change_kernel(h_j, h_i)
    else:
        theta = forecaster.kernel(h_j, h_i)
    pred = forecast_logits(theta, delta, f0_j)
    z_hat = predicted_label(pred, candidates, target_ids_j)
    if return_logits:
        return z_hat, candidates, pred
    return z_hat


def evaluate_labels(
    z_hat: Iterable[float],
    z_true: Iterable[float],
) -> Dict[str, float]:
    """Precision / recall / F1 of predicted forgetting labels.

    Delegates to ``src.eval.metrics`` when available and otherwise computes the same quantities
    locally (binary F1 is what Tables 1 & 2 report).
    """
    try:  # normal path once src/eval/metrics.py exists
        from ..eval import metrics as _m  # type: ignore

        for fn_name in ("forecast_metrics", "compute_forecast_metrics", "binary_metrics"):
            fn = getattr(_m, fn_name, None)
            if callable(fn):
                try:
                    out = dict(fn(list(z_true), list(z_hat)))
                    return {k: float(v) for k, v in out.items()}
                except Exception:
                    continue
        for fn_name in ("binary_f1", "f1_score_binary", "forecast_f1"):
            fn = getattr(_m, fn_name, None)
            if callable(fn):
                return {"f1": float(fn(list(z_true), list(z_hat)))}
    except Exception:
        pass
    tp = fp = fn_ = 0
    for zt, zh in zip(list(z_true), list(z_hat)):
        zi, zj = int(zt), int(zh)
        if zi == 1 and zj == 1:
            tp += 1
        elif zi == 0 and zj == 1:
            fp += 1
        elif zi == 1 and zj == 0:
            fn_ += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn_)}


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logit-change-transfer forecaster (Sec. 3.2, Eq. 2/3)")
    p.add_argument("--pairs", type=str, default=None,
                   help="pairs.jsonl produced by scripts/generate_ground_truth.py")
    p.add_argument("--h-cache", type=str, default=None,
                   help="cached h(x,y) token-level encodings (.pt) keyed by example index")
    p.add_argument("--checkpoint", type=str, default=None, help="trained h checkpoint")
    p.add_argument("--fixed", action="store_true",
                   help="use the frozen base-PTLM representation (non-trained baseline)")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--out", type=str, default=None, help="optional JSON output path")
    p.add_argument("--self-test", action="store_true", help="run a synthetic tensor smoke test")
    return p.parse_args(argv)


def _self_test() -> int:  # pragma: no cover - smoke test
    torch.manual_seed(0)
    Tj, Ti, d, S, V = 4, 3, 8, 16, 64
    h_j = torch.randn(1, Tj, d)
    h_i = torch.randn(1, Ti, d)
    f0_j = torch.randn(Tj, S)

    f0_j_topk = torch.topk(f0_j, k=5, dim=-1)
    f0_i_topk = torch.topk(torch.randn(Ti, V), k=5, dim=-1)
    fi_i_topk = torch.topk(torch.randn(Ti, V), k=5, dim=-1)

    forecaster = LogitChangeTransferForecaster(dim=d)
    z_hat, cand, pred = forecast_pair_from_cache(
        forecaster,
        h_j,
        h_i,
        (f0_i_topk.indices, f0_i_topk.values),
        (fi_i_topk.indices, fi_i_topk.values),
        (f0_j_topk.indices, f0_j_topk.values),
        target_ids_i=[1, 2, 3],
        target_ids_j=[0, 1, 2, 3],
        return_logits=True,
    )
    assert cand.numel() > 0 and pred.shape == (Tj, cand.numel())
    assert z_hat.numel() == Tj
    loss = pair_margin_objective(pred, cand, [0, 1, 2, 3], z=1)
    assert torch.isfinite(loss), "margin loss is not finite"
    loss0 = pair_margin_objective(pred, cand, [0, 1, 2, 3], z=0)
    assert torch.isfinite(loss0)
    logger.info("self-test ok: z_hat=%s loss(z=1)=%.4f loss(z=0)=%.4f",
                z_hat.tolist(), float(loss), float(loss0))

    # kernel is [T_j, T_i] and the zero-change prediction reduces to f_0(x_j)
    assert forecaster.kernel(h_j, h_i).shape == (1, Tj, Ti)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.self_test:
        return _self_test()
    if not args.pairs:
        logger.error("nothing to do: provide --pairs or --self-test")
        return 2

    from ..forgetting.ground_truth import load_ground_truth_jsonl  # local import

    pairs = load_ground_truth_jsonl(args.pairs)
    logger.info("loaded %d pair records from %s", len(pairs), args.pairs)

    forecaster: Optional[LogitChangeTransferForecaster] = None
    if args.fixed:
        forecaster = FixedLogitForecaster(dim=args.dim, topk=args.topk)
    elif args.checkpoint or args.h_cache:
        forecaster = LogitChangeTransferForecaster(dim=args.dim, topk=args.topk)
    if forecaster is not None and args.checkpoint and os.path.exists(args.checkpoint):
        forecaster.load_state(args.checkpoint)

    h_cache: Dict[Any, Any] = {}
    if args.h_cache and os.path.exists(args.h_cache):
        h_cache = torch.load(args.h_cache, map_location="cpu")

    def _h(key: Any) -> Optional[Any]:
        entry = h_cache.get(key)
        if entry is None and isinstance(key, str) and key.isdigit():
            entry = h_cache.get(int(key))
        if entry is None:
            return None
        return entry.get("h") if isinstance(entry, Mapping) else entry

    z_true: List[float] = []
    z_hat: List[float] = []
    for rec in pairs:
        i_key = getattr(rec, "i", None)
        j_key = getattr(rec, "j", None)
        h_i = _h(i_key)
        h_j = _h(j_key)
        if h_i is None or h_j is None:
            continue
        try:
            out = forecast_pair_from_cache(
                forecaster,
                _to_tensor(h_j),
                _to_tensor(h_i),
                (rec.f0_i_token_logits.indices, rec.f0_i_token_logits.values),
                (rec.fi_i_token_logits.indices, rec.fi_i_token_logits.values),
                (rec.f0_j_token_logits.indices, rec.f0_j_token_logits.values),
                target_ids_i=[],
                target_ids_j=[],
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("skipping pair (%s, %s): %s", i_key, j_key, exc)
            continue
        z_hat.append(float(int(out.sum()) > 0))
        z_true.append(float(int(getattr(rec, "z", 0))))
    metrics = evaluate_labels(z_hat, z_true)
    logger.info("forecast metrics on %d pairs: %s", len(z_hat), metrics)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    else:
        print(json.dumps(metrics, indent=2))
    return 0


def _to_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, Mapping):
        for k in ("h", "tensor", "values"):
            if k in x:
                return torch.as_tensor(x[k])
    return torch.as_tensor(x)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
