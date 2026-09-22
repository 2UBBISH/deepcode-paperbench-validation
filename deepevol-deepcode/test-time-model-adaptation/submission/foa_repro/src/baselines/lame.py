"""LAME (Logits Affinity-based Model Adaptation) baseline for FOA comparisons.

Reference
---------
Boudiaf, Mueller, Ben Ayed, Bertinetto.
"Parameter-free Online Test-time Adaptation", CVPR 2022  (fiveai/LAME).

LAME is a *post-hoc*, gradient-free test-time adaptation method: the frozen
source model is never modified.  Instead, the source posterior probabilities
``p = softmax(z)`` of the test batch are refined by a Laplacian-style
regularisation that encourages neighbouring samples (in feature space) to share
their predicted soft-assignments.

Objective (LAME, Eq. 4)::

    max_Q   sum_i sum_c q_ic * log p_ic  +  w * sum_{i,j} A_ij * <q_i, q_j>
    s.t.    q_i in the probability simplex

whose exponentiated-gradient fixed point is the iterative update::

    q_i <- softmax( log p_i + w * sum_j A_ij q_j )

with ``A`` the (row-normalised) k-nearest-neighbour cosine affinity matrix of
the penultimate features.  Number of neighbours ``knn = 5`` and the
probabilities are refined in place for ``num_iters`` steps (Appendix B.2 of the
FOA paper gives ``kNN k=5``, ``BS=64`` for LAME).

This module provides a self-contained adapter (no external repo required) that
exposes the same duck-typed protocol as the other FOA baseline wrappers
(``step`` / ``adapt`` / ``predict`` / ``reset`` / ``state_dict``) so that
``scripts/run_baselines.py`` can drive it alongside TENT/SAR.

By default ``step`` returns *log-posteriors* (``log q``) so that a downstream
``softmax`` recovers exactly the refined posterior ``q`` (keeps accuracy and ECE
consistent with the other baselines that emit logits).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Appendix B.2 (FOA paper) tuned hyper-parameters for LAME
# ---------------------------------------------------------------------------
DEFAULT_KNN = 5
DEFAULT_AFFINITY = "knn"
DEFAULT_KERNEL = "cosine"
DEFAULT_SIGMA = 1.0
DEFAULT_NUM_ITERS = 5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_W = 1.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_CLASSES = 1000
DEFAULT_EPS = 1e-8

__all__ = [
    "LAME",
    "LAMEConfig",
    "build_lame",
    "build_baseline",
    "cosine_similarity_matrix",
    "normalize_features",
    "knn_affinity",
    "normalize_affinity",
    "compute_affinity",
    "refine_posteriors",
    "lame_objective",
    "sinkhorn_balance",
    "DEFAULT_KNN",
    "DEFAULT_BATCH_SIZE",
]


# ---------------------------------------------------------------------------
# small config helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup tolerant to nested dicts/``Config``/attribute objects."""
    if cfg is None:
        return default
    node = cfg
    for key in keys:
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(key, default)
        else:
            node = getattr(node, key, default)
    return default if node is None else node


# ---------------------------------------------------------------------------
# affinity maths
# ---------------------------------------------------------------------------
def normalize_features(features: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """L2-normalise each feature row (cosine geometry)."""
    if features.dim() == 1:
        features = features.unsqueeze(0)
    return features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def cosine_similarity_matrix(
    features: torch.Tensor,
    *,
    normalize: bool = True,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Pairwise cosine similarity matrix ``[N, N]`` of ``features``."""
    if normalize:
        features = normalize_features(features, eps=eps)
    sim = features @ features.t()
    # guard against floating point drift outside [-1, 1]
    return sim.clamp(-1.0, 1.0)


def knn_affinity(sim: torch.Tensor, k: int = DEFAULT_KNN, include_self: bool = False) -> torch.Tensor:
    """Sparsify a similarity matrix to its ``k`` largest entries per row.

    The result is symmetrised with a max-reduction (an edge is kept if either
    endpoint ranks it in its own kNN), which is the usual LAME/graph practice.
    """
    n = sim.shape[0]
    if k is None or k <= 0 or k >= n:
        return sim
    kk = int(k) + (0 if include_self else 1)
    kk = min(kk, n)
    values, indices = torch.topk(sim, k=kk, dim=-1, largest=True, sorted=False)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, indices, 1.0)
    if not include_self:
        # never keep the self loop when kNN sparsification is requested
        mask.fill_diagonal_(0.0)
    affinity = sim * mask
    affinity = torch.maximum(affinity, affinity.t())
    return affinity


def normalize_affinity(affinity: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Row-normalise the affinity matrix into a stochastic transition matrix."""
    denom = affinity.sum(dim=-1, keepdim=True).clamp_min(eps)
    return affinity / denom


def compute_affinity(
    features: torch.Tensor,
    *,
    knn: Optional[int] = DEFAULT_KNN,
    kernel: str = DEFAULT_KERNEL,
    sigma: float = DEFAULT_SIGMA,
    normalize: bool = True,
    symmetrize: bool = True,
    include_self: bool = False,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Build the LAME affinity matrix from penultimate features.

    ``kernel``:
      * ``"cosine"``  -> ``exp(-(1 - cos) / sigma)``   (equivalent to a Gaussian
        kernel on the squared euclidean distance of L2-normalised features)
      * ``"rbf"``     -> ``exp(-||x_i - x_j||^2 / sigma)``
      * ``"linear"``  -> raw cosine similarity moved into ``[0, 1]``
    """
    kernel = (kernel or "cosine").lower()
    normed = normalize_features(features, eps=eps)
    if kernel == "cosine":
        sim = cosine_similarity_matrix(normed, normalize=False, eps=eps)
        affinity = torch.exp(-(1.0 - sim) / max(float(sigma), eps))
    elif kernel in ("rbf", "gaussian"):
        sq = torch.cdist(normed, normed, p=2.0).pow(2)
        affinity = torch.exp(-sq / max(float(sigma), eps))
    elif kernel in ("linear", "dot"):
        sim = cosine_similarity_matrix(normed, normalize=False, eps=eps)
        affinity = 0.5 * (sim + 1.0)
    else:
        raise ValueError(f"unknown LAME kernel: {kernel!r}")

    if normalize:
        affinity = normalize_affinity(affinity, eps=eps)
    if knn:
        affinity = knn_affinity(affinity, k=knn, include_self=include_self)
    if symmetrize:
        affinity = 0.5 * (affinity + affinity.t())
    return affinity


# ---------------------------------------------------------------------------
# LAME refinement
# ---------------------------------------------------------------------------
def sinkhorn_balance(q: torch.Tensor, num_iters: int = 3, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Balance column marginals of ``q`` towards the uniform prior (Sinkhorn)."""
    qc = q
    n = qc.shape[0]
    for _ in range(max(int(num_iters), 1)):
        col = qc.sum(dim=0, keepdim=True).clamp_min(eps)
        target = float(n) / float(qc.shape[1])
        qc = qc * (target / col)
        row = qc.sum(dim=-1, keepdim=True).clamp_min(eps)
        qc = qc / row
    return qc


def refine_posteriors(
    logits: torch.Tensor,
    affinity: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    temperature: float = DEFAULT_TEMPERATURE,
    w: float = DEFAULT_W,
    force_balanced: bool = False,
    balance_iters: int = 3,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """LAME posterior refinement (exponentiated-gradient fixed point).

    ``q_i <- softmax( log p_i + w * sum_j A_ij q_j )`` with ``p = softmax(z/T)``.
    Returns the refined posterior ``q`` of shape ``[N, C]``.
    """
    temperature = max(float(temperature), eps)
    p = F.softmax(logits / temperature, dim=-1)
    log_p = torch.log(p + eps)
    q = p
    for _ in range(max(int(num_iters), 0)):
        message = affinity @ q
        q = F.softmax(log_p + float(w) * message, dim=-1)
        if force_balanced:
            q = sinkhorn_balance(q, num_iters=balance_iters, eps=eps)
    return q


def lame_objective(
    q: torch.Tensor,
    logits: torch.Tensor,
    affinity: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    w: float = DEFAULT_W,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """LAME objective value (to be *maximised*), useful for sanity checks."""
    temperature = max(float(temperature), eps)
    p = F.softmax(logits / temperature, dim=-1)
    log_p = torch.log(p + eps)
    fit = (q * log_p).sum()
    smooth = (affinity * (q @ q.t())).sum()
    return fit + float(w) * smooth


# ---------------------------------------------------------------------------
# config container
# ---------------------------------------------------------------------------
@dataclass
class LAMEConfig:
    """Hyper-parameters of the LAME baseline (FOA Appendix B.2 defaults)."""

    knn: Optional[int] = DEFAULT_KNN          # k = 5 nearest neighbours
    affinity: str = DEFAULT_AFFINITY          # "knn" | "full"
    kernel: str = DEFAULT_KERNEL              # "cosine" | "rbf" | "linear"
    sigma: float = DEFAULT_SIGMA
    num_iters: int = DEFAULT_NUM_ITERS
    temperature: float = DEFAULT_TEMPERATURE
    w: float = DEFAULT_W                      # affinity regularisation weight
    force_balanced: bool = False              # class-balanced variant
    balance_iters: int = 3
    batch_size: int = DEFAULT_BATCH_SIZE      # BS = 64
    num_classes: int = DEFAULT_NUM_CLASSES
    normalize_features: bool = True
    symmetrize: bool = True
    include_self: bool = False
    return_logits: bool = True                # emit log-posteriors by default
    support_mode: str = "batch"               # "batch" | "queue"
    max_support: Optional[int] = None
    eps: float = DEFAULT_EPS

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_config(cls, cfg: Any) -> "LAMEConfig":
        """Build from a FOA config, reading ``baselines.lame`` (or ``lame``)."""
        source = _cfg_get(cfg, "baselines", "lame", default=None)
        if source is None:
            source = _cfg_get(cfg, "lame", default=None)
        data = cls()
        if source is None:
            return data
        for key in cls().to_dict().keys():
            val = _cfg_get(source, key, default=None)
            if val is not None:
                setattr(data, key, val)
        # inherit class count from the eval/data section when not overridden
        if _cfg_get(source, "num_classes", default=None) is None:
            nc = _cfg_get(cfg, "data", "num_classes_eval", default=None)
            if nc is None:
                nc = _cfg_get(cfg, "model", "num_classes", default=None)
            if nc is not None:
                data.num_classes = int(nc)
        if _cfg_get(source, "batch_size", default=None) is None:
            bs = _cfg_get(cfg, "data", "batch_size", default=None)
            if bs is not None:
                data.batch_size = int(bs)
        return data


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------
class LAME(nn.Module):
    """Parameter-free post-hoc LAME adapter.

    Parameters
    ----------
    model:
        Optional frozen backbone.  When provided, ``step(images=...)`` performs
        the forward pass itself; otherwise features/logits must be supplied by
        the caller (the usual path in ``run_baselines.py`` for post-hoc
        methods).
    config:
        :class:`LAMEConfig` (or any object/dict with the same fields).
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[LAMEConfig] = None,
        *,
        knn: Optional[int] = None,
        affinity: Optional[str] = None,
        kernel: Optional[str] = None,
        sigma: Optional[float] = None,
        num_iters: Optional[int] = None,
        temperature: Optional[float] = None,
        w: Optional[float] = None,
        force_balanced: Optional[bool] = None,
        balance_iters: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        normalize_features: Optional[bool] = None,
        symmetrize: Optional[bool] = None,
        include_self: Optional[bool] = None,
        return_logits: Optional[bool] = None,
        support_mode: Optional[str] = None,
        max_support: Optional[int] = None,
        eps: Optional[float] = None,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        cfg = config if config is not None else LAMEConfig()
        self.config = cfg
        overrides = dict(
            knn=knn,
            affinity=affinity,
            kernel=kernel,
            sigma=sigma,
            num_iters=num_iters,
            temperature=temperature,
            w=w,
            force_balanced=force_balanced,
            balance_iters=balance_iters,
            batch_size=batch_size,
            num_classes=num_classes,
            normalize_features=normalize_features,
            symmetrize=symmetrize,
            include_self=include_self,
            return_logits=return_logits,
            support_mode=support_mode,
            max_support=max_support,
            eps=eps,
        )
        self.knn = cfg.knn if overrides["knn"] is None else overrides["knn"]
        self.affinity = (cfg.affinity if overrides["affinity"] is None else overrides["affinity"])
        self.kernel = cfg.kernel if overrides["kernel"] is None else overrides["kernel"]
        self.sigma = float(cfg.sigma if overrides["sigma"] is None else overrides["sigma"])
        self.num_iters = int(cfg.num_iters if overrides["num_iters"] is None else overrides["num_iters"])
        self.temperature = float(
            cfg.temperature if overrides["temperature"] is None else overrides["temperature"]
        )
        self.w = float(cfg.w if overrides["w"] is None else overrides["w"])
        self.force_balanced = bool(
            cfg.force_balanced if overrides["force_balanced"] is None else overrides["force_balanced"]
        )
        self.balance_iters = int(
            cfg.balance_iters if overrides["balance_iters"] is None else overrides["balance_iters"]
        )
        self.batch_size = int(cfg.batch_size if overrides["batch_size"] is None else overrides["batch_size"])
        self.num_classes = int(
            cfg.num_classes if overrides["num_classes"] is None else overrides["num_classes"]
        )
        self.normalize_features = bool(
            cfg.normalize_features if overrides["normalize_features"] is None else overrides["normalize_features"]
        )
        self.symmetrize = bool(cfg.symmetrize if overrides["symmetrize"] is None else overrides["symmetrize"])
        self.include_self = bool(
            cfg.include_self if overrides["include_self"] is None else overrides["include_self"]
        )
        self.return_logits = bool(
            cfg.return_logits if overrides["return_logits"] is None else overrides["return_logits"]
        )
        self.support_mode = str(
            cfg.support_mode if overrides["support_mode"] is None else overrides["support_mode"]
        )
        self.max_support = cfg.max_support if overrides["max_support"] is None else overrides["max_support"]
        self.eps = float(cfg.eps if overrides["eps"] is None else overrides["eps"])

        # "full" affinity means no kNN sparsification
        if str(self.affinity).lower() in ("full", "none", "dense"):
            self.knn = None

        self.model = model
        if self.model is not None:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.device = device if device is not None else _infer_device(self.model)

        # queue support (only used when support_mode == "queue")
        self._support_features: List[torch.Tensor] = []
        self._support_logits: List[torch.Tensor] = []
        self.num_seen = 0

    # -- state management ---------------------------------------------------
    def reset(self) -> None:
        """Clear the accumulated support and counters (episodic protocol)."""
        self._support_features = []
        self._support_logits = []
        self.num_seen = 0

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "config": self.config.to_dict(),
            "num_seen": self.num_seen,
            "support_features": [t.detach().clone() for t in self._support_features],
            "support_logits": [t.detach().clone() for t in self._support_logits],
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.num_seen = int(state.get("num_seen", 0))
        self._support_features = [t.detach().clone() for t in state.get("support_features", [])]
        self._support_logits = [t.detach().clone() for t in state.get("support_logits", [])]

    # -- model plumbing -----------------------------------------------------
    def _unwrap(self) -> Optional[nn.Module]:
        """Return the inner torch module of a FOA ``ViTWithCLSFeatures`` wrapper."""
        if self.model is None:
            return None
        inner = getattr(self.model, "model", None)
        return inner if isinstance(inner, nn.Module) else self.model

    @staticmethod
    def _split_output(output: Any) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Normalise a model output into ``(features_or_None, logits)``."""
        if isinstance(output, dict):
            logits = output.get("logits", None)
            feats = output.get("features", None)
            if feats is None:
                feats = output.get("cls_features", None)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1] if len(feats) else None
            return feats, logits
        if isinstance(output, (list, tuple)):
            if len(output) == 1:
                return None, output[0]
            return output[-1], output[0] if output[0].dim() == 2 else output[1]
        return None, output

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frozen forward pass returning ``(penultimate_features, logits)``."""
        if self.model is None:
            raise RuntimeError("LAME has no model attached; pass features/logits explicitly.")
        images = images.to(self.device)
        if hasattr(self.model, "forward_with_features"):
            out = self.model.forward_with_features(images)
            feats = out.get("final_cls", None)
            logits = out.get("logits", None)
            if feats is None:
                cls_feats = out.get("cls_features", None)
                feats = cls_feats[-1] if isinstance(cls_feats, (list, tuple)) else cls_feats
            if logits is None:  # pragma: no cover - defensive
                raise RuntimeError("model.forward_with_features did not return logits")
            return feats, logits
        try:
            out = self.model(images)
        except TypeError:
            out = self.model.forward_features(images)
        feats, logits = self._split_output(out)
        if feats is None:  # fall back to the logits themselves as the key space
            feats = logits
        return feats, logits

    # -- core computation ---------------------------------------------------
    def affinities(self, features: torch.Tensor) -> torch.Tensor:
        """Build the (kNN) affinity matrix for the given feature batch."""
        return compute_affinity(
            features,
            knn=self.knn,
            kernel=self.kernel,
            sigma=self.sigma,
            normalize=self.normalize_features,
            symmetrize=self.symmetrize,
            include_self=self.include_self,
            eps=self.eps,
        )

    def refine(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the refined posterior ``q`` for the given logits/features."""
        if features is None:
            # with no feature space available, LAME degenerates to plain softmax
            return F.softmax(logits / max(self.temperature, self.eps), dim=-1)
        affinity = self.affinities(features)
        return refine_posteriors(
            logits,
            affinity,
            num_iters=self.num_iters,
            temperature=self.temperature,
            w=self.w,
            force_balanced=self.force_balanced,
            balance_iters=self.balance_iters,
            eps=self.eps,
        )

    @torch.no_grad()
    def step(
        self,
        features: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adapt + predict for one batch (LAME is post-hoc: no parameters change).

        Returns log-posteriors when ``return_logits`` is ``True`` (default), so a
        downstream ``softmax`` reproduces the refined posterior exactly.
        """
        del targets, weights  # LAME is unsupervised / unweighted
        if logits is None or features is None:
            if images is None:
                raise ValueError("LAME.step requires (features, logits) or images.")
            features, logits = self.forward_features(images)
        features = features.to(self.device).float()
        logits = logits.to(self.device).float()

        if self.support_mode == "queue" and self.max_support:
            self._support_features.append(features.detach().clone())
            self._support_logits.append(logits.detach().clone())
            if len(self._support_features) > 1:
                feats = torch.cat(self._support_features, dim=0)
                logs = torch.cat(self._support_logits, dim=0)
            else:  # pragma: no cover - single batch
                feats, logs = features, logits
            keep = int(self.max_support)
            if feats.shape[0] > keep:
                feats = feats[-keep:]
                logs = logs[-keep:]
                self._support_features = [feats.clone()]
                self._support_logits = [logs.clone()]
            else:
                self._support_features = [feats.clone()]
                self._support_logits = [logs.clone()]
            q = self.refine(logs, feats)[-features.shape[0]:]
        else:
            q = self.refine(logits, features)

        self.num_seen += int(logits.shape[0])
        if self.return_logits:
            return torch.log(q + self.eps)
        return q

    # aliases expected by the generic baseline runner
    @torch.no_grad()
    def adapt(
        self,
        features: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.step(features=features, logits=logits, images=images, **kwargs)

    @torch.no_grad()
    def predict(
        self,
        features: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refined predictions without registering the batch as "seen"."""
        if logits is None or features is None:
            if images is None:
                raise ValueError("LAME.predict requires (features, logits) or images.")
            features, logits = self.forward_features(images)
        features = features.to(self.device).float()
        logits = logits.to(self.device).float()
        q = self.refine(logits, features)
        return torch.log(q + self.eps) if self.return_logits else q

    @torch.no_grad()
    def __call__(  # type: ignore[override]
        self,
        features: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.step(features=features, logits=logits, images=images, **kwargs)

    # -- diagnostics --------------------------------------------------------
    def trainable_parameter_names(self) -> List[str]:
        """LAME adapts no parameters (parameter-free)."""
        return []

    def trainable_parameter_count(self) -> int:
        return 0

    def extra_repr(self) -> str:
        return (
            f"knn={self.knn}, kernel={self.kernel}, sigma={self.sigma}, "
            f"num_iters={self.num_iters}, w={self.w}, temperature={self.temperature}, "
            f"force_balanced={self.force_balanced}, batch_size={self.batch_size}"
        )


def _infer_device(model: Optional[nn.Module]) -> torch.device:
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------
def build_lame(
    model: Optional[nn.Module] = None,
    cfg: Any = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> LAME:
    """Build a LAME wrapper from a FOA config (``baselines.lame`` block)."""
    config = LAMEConfig.from_config(cfg) if cfg is not None else LAMEConfig()
    return LAME(model, config, device=device, **kwargs)


# alias used by the generic baseline runner
build_baseline = build_lame


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    feats = torch.randn(8, 32)
    logits = torch.randn(8, 5)
    lame = LAME()
    out = lame.step(feats, logits)
    print("LAME output shape:", tuple(out.shape))
    print("max prob diff:", (torch.softmax(out, -1) - lame.refine(logits, feats)).abs().max().item())
