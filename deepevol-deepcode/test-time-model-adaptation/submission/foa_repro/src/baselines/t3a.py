"""T3A baseline adapter for the FOA reproduction.

Implements the T3A test-time classifier adjustment module (Iwasawa & Matsuo,
NeurIPS 2021) as a self-contained, parameter-free adapter comparable to
``src/baselines/lame.py`` and ``src/baselines/tent.py``.

T3A keeps a *support set* of pseudo-labelled prototype features
(``M = 20`` supports per class, see Appendix B.2 of the FOA paper, "T3A
(supports M=20, BS=64)") and classifies every test sample with a
similarity-weighted kNN vote over the top-``K`` (``K = 50``) most similar
supports:

1. compute the cosine similarity between the (L2-normalised) query feature
   and every stored support feature;
2. keep the top-``K`` similarities and turn them into weights with a softmax
   (``a_i = exp(sim_i) / sum_j exp(sim_j)``);
3. sum the weights per class, ``w_c = sum_{i: y_i = c} a_i``, and *filter out*
   classes whose aggregate weight is below ``1 / K`` (i.e. classes that gather
   less than the average weight of a uniform softmax over the ``K``
   candidates);
4. predict the posterior by renormalising the surviving class weights, and
   fall back to the (source) softmax whenever no class survives;
5. afterwards insert the sample into the support set of its pseudo-labelled
   class, replacing the least-confident support once the per-class buffer is
   full.

No model parameter is ever modified and no gradient is ever computed: the
adapter only needs the frozen features plus the logits of the source
classifier.

The wrapper exposes the same duck-typed protocol as the other FOA baselines
(``step(features, logits, ...)``, ``reset()``, ``state_dict()``,
``build_t3a`` / ``build_baseline`` factories) so ``scripts/run_baselines.py``
can drive it unchanged.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "T3A",
    "T3ABaseline",
    "T3AConfig",
    "build_t3a",
    "build_baseline",
    "normalize_features",
    "cosine_similarity_matrix",
    "t3a_refine",
    "update_support_set",
    "DEFAULT_NUM_SUPPORTS",
    "DEFAULT_FILTER_K",
    "DEFAULT_LAM",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPS",
]

# ---------------------------------------------------------------------------
# Paper / Appendix B.2 defaults
# ---------------------------------------------------------------------------
DEFAULT_NUM_SUPPORTS = 20     # M: supports kept per class
DEFAULT_FILTER_K = 50         # K: number of nearest supports considered
DEFAULT_LAM = 1.0             # similarity scale inside the softmax
DEFAULT_TEMPERATURE = 1.0     # softmax temperature over the top-K similarities
DEFAULT_NUM_CLASSES = 1000
DEFAULT_BATCH_SIZE = 64       # BS = 64 (same online stream as FOA)
DEFAULT_EPS = 1e-8
_NEG_INF = -1e4               # safe replacement for -inf before softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_features(features: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """L2-normalise the feature rows (cosine kernel)."""
    return F.normalize(features, dim=-1, eps=eps)


def cosine_similarity_matrix(
    features: torch.Tensor,
    supports: torch.Tensor,
    normalize: bool = True,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Pairwise cosine similarity ``[B, N]`` between features and supports."""
    if normalize:
        features = normalize_features(features, eps=eps)
        supports = normalize_features(supports, eps=eps)
    return features @ supports.t()


def t3a_refine(
    features: torch.Tensor,
    support_features: Optional[torch.Tensor],
    support_labels: Optional[torch.Tensor],
    filter_K: int = DEFAULT_FILTER_K,
    lam: float = DEFAULT_LAM,
    temperature: float = DEFAULT_TEMPERATURE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    support_valid: Optional[torch.Tensor] = None,
    filter_threshold: Optional[float] = None,
    normalize: bool = True,
    eps: float = DEFAULT_EPS,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """T3A prediction rule.

    Returns ``(probs, valid_mask)`` where ``probs`` is ``[B, C]`` (``None`` when
    the support set is empty) and ``valid_mask`` marks the rows for which the
    refined posterior is usable (at least one class survived the filter).
    """
    if support_features is None or support_features.shape[0] == 0:
        return None, torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)

    device = features.device
    batch_size = features.shape[0]
    num_support = support_features.shape[0]

    sim = cosine_similarity_matrix(features, support_features, normalize=normalize, eps=eps)
    if support_valid is not None:
        sim = sim.masked_fill(~support_valid.unsqueeze(0).to(device), _NEG_INF)

    k = int(min(int(filter_K), num_support))
    k = max(k, 1)
    topk_sim, topk_idx = sim.topk(k, dim=1)

    finite = torch.isfinite(topk_sim) & (topk_sim > _NEG_INF / 2)
    safe_sim = torch.where(finite, topk_sim, torch.full_like(topk_sim, _NEG_INF))
    logits = (safe_sim * float(lam)) / max(float(temperature), eps)
    weights = torch.softmax(logits, dim=1)
    weights = weights * finite.to(weights.dtype)
    row_sum = weights.sum(dim=1, keepdim=True)
    usable = row_sum.squeeze(1) > 0
    weights = weights / row_sum.clamp_min(eps)

    # Aggregate the weights per class.
    labels = support_labels.to(device)
    topk_labels = labels[topk_idx]                       # [B, k]
    class_weights = torch.zeros(batch_size, num_classes, device=device, dtype=weights.dtype)
    class_weights.scatter_add_(1, topk_labels.clamp(0, num_classes - 1), weights)

    threshold = (1.0 / float(k)) if filter_threshold is None else float(filter_threshold)
    class_weights = class_weights * (class_weights >= threshold).to(class_weights.dtype)

    total = class_weights.sum(dim=1, keepdim=True)
    valid = usable & (total.squeeze(1) > 0)
    probs = class_weights / total.clamp_min(eps)
    return probs, valid


def update_support_set(
    support_features: torch.Tensor,
    support_conf: torch.Tensor,
    support_counts: torch.Tensor,
    support_valid: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    num_supports: int = DEFAULT_NUM_SUPPORTS,
    strategy: str = "confidence",
    sample_weights: Optional[torch.Tensor] = None,
) -> None:
    """Insert samples into the per-class support buffers (in place).

    ``support_features`` has shape ``[C, M, d]``, ``support_conf``/``support_valid``
    have shape ``[C, M]`` and ``support_counts`` has shape ``[C]``.  A sample is
    appended while its class buffer is not full; once full, the least-confident
    stored support is replaced when the incoming sample is more confident.
    """
    num_classes, cap, _ = support_features.shape
    cap = min(cap, int(num_supports))
    for b in range(features.shape[0]):
        c = int(labels[b].item())
        if c < 0 or c >= num_classes:
            continue
        conf = float(confidences[b].item())
        if sample_weights is not None:
            conf = conf * float(sample_weights[b].item())
        count = int(support_counts[c].item())
        if count < cap:
            support_features[c, count] = features[b].detach()
            support_conf[c, count] = conf
            support_valid[c, count] = True
            support_counts[c] = count + 1
            continue
        # Buffer full -> replace the least confident support (if beneficial).
        if count == 0:
            continue
        stored = support_conf[c, :cap].clone()
        stored[~support_valid[c, :cap]] = float("inf")
        j = int(torch.argmin(stored).item())
        if strategy == "confidence":
            if conf > float(support_conf[c, j].item()):
                support_features[c, j] = features[b].detach()
                support_conf[c, j] = conf
                support_valid[c, j] = True
        elif strategy == "fifo":
            # round-robin replacement
            j = int(support_counts[c].item()) % cap
            support_features[c, j] = features[b].detach()
            support_conf[c, j] = conf
            support_valid[c, j] = True
            support_counts[c] = support_counts[c] + 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class T3AConfig:
    """Hyper-parameters of the T3A baseline (FOA Appendix B.2)."""

    num_supports: int = DEFAULT_NUM_SUPPORTS           # M = 20 per class
    filter_K: int = DEFAULT_FILTER_K                   # K = 50 nearest supports
    lam: float = DEFAULT_LAM
    temperature: float = DEFAULT_TEMPERATURE
    num_classes: int = DEFAULT_NUM_CLASSES
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize_features: bool = True
    update_supports: bool = True
    replace_strategy: str = "confidence"               # "confidence" | "fifo"
    filter_threshold: Optional[float] = None           # None -> 1 / k
    pseudo_label_source: bool = True                   # pseudo-labels from source logits
    return_logits: bool = True                         # return log-posteriors
    eps: float = DEFAULT_EPS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: Any) -> "T3AConfig":
        """Build from a FOA config object (``baselines.t3a`` or legacy ``t3a``)."""
        spec = cls()
        block = _cfg_get(cfg, "baselines", "t3a")
        if not isinstance(block, dict):
            block = _cfg_get(cfg, "t3a")
        if isinstance(block, dict):
            for key, value in block.items():
                if hasattr(spec, key):
                    setattr(spec, key, value)
        num_classes = _cfg_get(cfg, "data", "num_classes_eval")
        if num_classes is None:
            num_classes = _cfg_get(cfg, "data", "num_classes")
        if num_classes is None:
            num_classes = _cfg_get(cfg, "model", "num_classes")
        if num_classes is not None:
            spec.num_classes = int(num_classes)
        batch_size = _cfg_get(cfg, "data", "batch_size")
        if batch_size is not None:
            spec.batch_size = int(batch_size)
        return spec


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup working for dict-like and attribute-like configs."""
    if cfg is None:
        return default
    node: Any = cfg
    for key in keys:
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(key, None)
        else:
            node = getattr(node, key, None)
    return default if node is None else node


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------
class T3A(nn.Module):
    """Post-hoc T3A test-time classifier adjustment (parameter-free)."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[T3AConfig] = None,
        *,
        num_supports: int = DEFAULT_NUM_SUPPORTS,
        filter_K: int = DEFAULT_FILTER_K,
        lam: float = DEFAULT_LAM,
        temperature: float = DEFAULT_TEMPERATURE,
        num_classes: int = DEFAULT_NUM_CLASSES,
        normalize_features: bool = True,
        update_supports: bool = True,
        replace_strategy: str = "confidence",
        filter_threshold: Optional[float] = None,
        pseudo_label_source: bool = True,
        return_logits: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        feature_dim: Optional[int] = None,
        eps: float = DEFAULT_EPS,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        spec = config or T3AConfig()
        spec.num_supports = int(num_supports)
        spec.filter_K = int(filter_K)
        spec.lam = float(lam)
        spec.temperature = float(temperature)
        spec.num_classes = int(num_classes)
        spec.normalize_features = bool(normalize_features)
        spec.update_supports = bool(update_supports)
        spec.replace_strategy = str(replace_strategy)
        spec.filter_threshold = filter_threshold
        spec.pseudo_label_source = bool(pseudo_label_source)
        spec.return_logits = bool(return_logits)
        spec.batch_size = int(batch_size)
        spec.eps = float(eps)

        self.config = spec
        self.model = model
        self.feature_dim = feature_dim
        if device is None:
            device = _infer_device(model)
        self.device = device

        # Support-set state (lazily allocated once the feature width is known).
        self.supports: Optional[torch.Tensor] = None       # [C, M, d]
        self.support_conf: Optional[torch.Tensor] = None   # [C, M]
        self.support_counts: Optional[torch.Tensor] = None  # [C]
        self.support_valid: Optional[torch.Tensor] = None  # [C, M] bool
        self.num_refined = 0
        self.num_fallback = 0
        if feature_dim is not None:
            self._init_supports(feature_dim, device=self.device)

    # -- state management -------------------------------------------------
    def _init_supports(
        self,
        feature_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        device = device if device is not None else self.device
        self.feature_dim = int(feature_dim)
        cap = int(self.config.num_supports)
        num_classes = int(self.config.num_classes)
        self.supports = torch.zeros(num_classes, cap, self.feature_dim, device=device, dtype=dtype)
        self.support_conf = torch.zeros(num_classes, cap, device=device, dtype=torch.float32)
        self.support_counts = torch.zeros(num_classes, device=device, dtype=torch.long)
        self.support_valid = torch.zeros(num_classes, cap, device=device, dtype=torch.bool)

    def reset(self) -> None:
        """Clear the support set (episodic / per-corruption protocol)."""
        if self.supports is None:
            return
        d = self.supports.shape[-1]
        self._init_supports(d, device=self.supports.device, dtype=self.supports.dtype)
        self.num_refined = 0
        self.num_fallback = 0

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "supports": None if self.supports is None else self.supports.detach().clone(),
            "support_conf": None if self.support_conf is None else self.support_conf.detach().clone(),
            "support_counts": None if self.support_counts is None else self.support_counts.detach().clone(),
            "support_valid": None if self.support_valid is None else self.support_valid.detach().clone(),
            "feature_dim": self.feature_dim,
            "config": self.config.to_dict(),
            "num_refined": self.num_refined,
            "num_fallback": self.num_fallback,
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        if not state:
            return
        self.supports = state.get("supports", None)
        self.support_conf = state.get("support_conf", None)
        self.support_counts = state.get("support_counts", None)
        self.support_valid = state.get("support_valid", None)
        self.feature_dim = state.get("feature_dim", self.feature_dim)
        self.num_refined = int(state.get("num_refined", 0))
        self.num_fallback = int(state.get("num_fallback", 0))

    # -- model plumbing ---------------------------------------------------
    def _unwrap(self) -> nn.Module:
        model = self.model
        if model is None:
            raise ValueError("T3A requires a model to extract features/logits.")
        inner = getattr(model, "model", None)
        if isinstance(inner, nn.Module):
            return inner
        return model

    def _split_output(self, out: Any) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Normalise the many output contracts into ``(features, logits)``."""
        if isinstance(out, dict):
            logits = out.get("logits")
            if logits is None:
                logits = out.get("output")
            feats = out.get("final_cls")
            if feats is None:
                cf = out.get("cls_features")
                if isinstance(cf, (list, tuple)) and len(cf):
                    feats = cf[-1]
            return feats, logits
        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                cls_features, logits, final = out
                return final, logits
            if len(out) == 2:
                feats, logits = out
                return feats, logits
            if len(out) == 1:
                return None, out[0]
        return None, out

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Return ``(final-cls feature, logits)`` for a batch of images."""
        model = self.model
        if model is None:
            raise ValueError("T3A requires a model, but `features`/`logits` may be passed directly to step().")
        images = images.to(self.device)
        if hasattr(model, "forward_with_features"):
            out = model.forward_with_features(images)
        elif hasattr(model, "forward_features"):
            out = model.forward_features(images)
        else:
            out = model(images)
        feats, logits = self._split_output(out)
        if logits is None:
            raise RuntimeError("Could not locate logits in the model output.")
        return None if feats is None else feats.float(), logits.float()

    # -- T3A core ---------------------------------------------------------
    def refine(
        self,
        logits: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the T3A posterior ``[B, C]`` (falls back to the source softmax)."""
        logits = logits.float()
        source_probs = torch.softmax(logits, dim=-1)
        if (
            features is None
            or self.supports is None
            or int(self.support_valid.sum().item()) == 0
        ):
            self.num_fallback += 1
            return source_probs
        with torch.no_grad():
            flat_features = self.supports.reshape(-1, self.supports.shape[-1])
            flat_valid = self.support_valid.reshape(-1)
            flat_labels = torch.arange(self.config.num_classes, device=flat_valid.device)
            flat_labels = flat_labels.repeat_interleave(self.supports.shape[1])
            probs, valid = t3a_refine(
                features.float(),
                flat_features,
                flat_labels,
                filter_K=self.config.filter_K,
                lam=self.config.lam,
                temperature=self.config.temperature,
                num_classes=self.config.num_classes,
                support_valid=flat_valid,
                filter_threshold=self.config.filter_threshold,
                normalize=self.config.normalize_features,
                eps=self.config.eps,
            )
        if probs is None:
            self.num_fallback += 1
            return source_probs
        probs = probs.to(source_probs.dtype)
        self.num_refined += 1
        self.num_fallback += int((~valid).sum().item())
        probs = torch.where(valid.unsqueeze(1), probs, source_probs)
        return probs

    @torch.no_grad()
    def step(
        self,
        features: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One online TTA step: refine the batch, then update the support set.

        Returns log-posteriors when ``config.return_logits`` is set (so a plain
        ``softmax`` recovers the refined posterior), otherwise probabilities.
        """
        if features is None or logits is None:
            if images is None:
                raise ValueError("step() needs either (features, logits) or images.")
            feats, logits_out = self.forward_features(images)
            features = feats if features is None else features
            logits = logits_out if logits is None else logits
        logits = logits.float()
        if features is None:
            probs = torch.softmax(logits, dim=-1)
        else:
            features = features.float()
            if self.supports is None:
                self._init_supports(features.shape[-1], device=features.device, dtype=features.dtype)
            probs = self.refine(logits, features)
            if self.config.update_supports:
                self._update(features, logits, probs, weights=weights)
        if self.config.return_logits:
            return torch.log(probs.clamp_min(self.config.eps))
        return probs

    def adapt(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.step(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.step(*args, **kwargs)

    def _update(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        probs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        source_probs = torch.softmax(logits.float(), dim=-1)
        if self.config.pseudo_label_source:
            pseudo = source_probs.argmax(dim=1)
            conf = source_probs.max(dim=1).values
        else:
            pseudo = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
        update_support_set(
            self.supports,
            self.support_conf,
            self.support_counts,
            self.support_valid,
            features.detach(),
            pseudo.detach(),
            conf.detach(),
            num_supports=self.config.num_supports,
            strategy=self.config.replace_strategy,
            sample_weights=None if weights is None else weights.detach().float(),
        )

    def forward(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        return self.step(images=images, targets=targets)

    # -- introspection ----------------------------------------------------
    @property
    def num_supports_per_class(self) -> int:
        return int(self.config.num_supports)

    def stored_supports(self) -> int:
        if self.support_valid is None:
            return 0
        return int(self.support_valid.sum().item())

    def trainable_parameter_names(self) -> List[str]:
        return []

    def trainable_parameter_count(self) -> int:
        return 0

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.config.num_classes}, M={self.config.num_supports}, "
            f"K={self.config.filter_K}, lam={self.config.lam}, "
            f"temperature={self.config.temperature}, "
            f"update_supports={self.config.update_supports}, "
            f"stored={self.stored_supports()}"
        )


# Alias matching the class name used by scripts/run_baselines.py internals.
T3ABaseline = T3A


def _infer_device(model: Optional[nn.Module]) -> torch.device:
    if model is not None:
        try:
            param = next(model.parameters())
            return param.device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_t3a(
    model: Optional[nn.Module] = None,
    cfg: Any = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> T3A:
    """Config-driven factory (reads ``baselines.t3a`` / ``t3a`` from ``cfg``)."""
    spec = T3AConfig.from_config(cfg) if cfg is not None else T3AConfig()
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return T3A(
        model,
        config=spec,
        num_supports=kwargs.pop("num_supports", kwargs.pop("M", spec.num_supports)),
        filter_K=kwargs.pop("filter_K", kwargs.pop("K", spec.filter_K)),
        lam=kwargs.pop("lam", spec.lam),
        temperature=kwargs.pop("temperature", spec.temperature),
        num_classes=kwargs.pop("num_classes", spec.num_classes),
        normalize_features=kwargs.pop("normalize_features", spec.normalize_features),
        update_supports=kwargs.pop("update_supports", spec.update_supports),
        replace_strategy=kwargs.pop("replace_strategy", spec.replace_strategy),
        filter_threshold=kwargs.pop("filter_threshold", spec.filter_threshold),
        pseudo_label_source=kwargs.pop("pseudo_label_source", spec.pseudo_label_source),
        return_logits=kwargs.pop("return_logits", spec.return_logits),
        batch_size=kwargs.pop("batch_size", spec.batch_size),
        feature_dim=kwargs.pop("feature_dim", None),
        eps=kwargs.pop("eps", spec.eps),
        device=device,
        **kwargs,
    )


# Alias expected by the generic baseline runner.
build_baseline = build_t3a
