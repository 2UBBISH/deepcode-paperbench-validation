"""Baseline runner for the FOA reproduction (paper Table 2 / Table 3 / Table 16 / Table 17).

Implements the *comparison* half of the FOA experiments: the gradient-free TTA methods
(LAME, T3A) and the gradient-based TTA methods (TENT, SAR, CoTTA, MEMO) are evaluated on
exactly the same ordered, single-pass online streams (``BS=64``) used by FOA, so that the
numbers are directly comparable with the paper's Tables 2/3.

All baseline hyper-parameters are taken verbatim from Appendix B.2 ("More Evaluation
Protocols") of the paper:

  LAME   : kNN affinity with k chosen from {1,5,10,20}; k=5 for all experiments. BS=64.
  T3A    : all hyper-parameters of T3A; BS=64; number of supports to restore M chosen
           from {1,5,20,50,100}; M=20 for all experiments.
  TENT   : SGD, momentum 0.9, BS=64, lr 0.001; trainable params = affine params of all
           layer-normalization layers.
  SAR    : SGD, momentum 0.9, BS=64, lr 0.001; entropy threshold E0 = 0.4 * ln C;
           trainable params = affine params of the layer-norm layers of blocks 1..8.
  CoTTA  : SGD, momentum 0.9, BS=64, lr 0.05; augmentation threshold p_th = 0.1;
           32 augmentations (color jitter, random affine, Gaussian blur, random horizontal
           flip, Gaussian noise) for images below the threshold; restoration probability
           0.01; EMA factor alpha = 0.999 for the teacher; trainable = all ViT-Base params.
  MEMO   : optional (marginal entropy minimization, single-sample augmentation ensemble).

The wrappers live in ``src/baselines/<name>.py``.  If a wrapper module is not importable the
runner degrades gracefully to a self-contained reference implementation (documented below)
so that the comparison table can still be produced; LAME/T3A are post-hoc (they only read
logits and features and never update the backbone), TENT/SAR/CoTTA/MEMO adapt parameters.

No part of this script ever needs the source-statistics checkpoint used by FOA.

Usage
-----
    python scripts/run_baselines.py --config configs/foa_imagenetc.yaml \
        --methods tent sar cotta t3a lame --all-corruptions
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:  # allow `python scripts/run_baselines.py` from anywhere
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.utils.config import FOA_DEFAULTS, Config, config_to_dict, load_config  # noqa: E402

# --------------------------------------------------------------------------------------
# Optional imports from sibling modules (kept duck-typed / degradable on purpose)
# --------------------------------------------------------------------------------------
try:  # metrics are pure numpy/torch post-processing
    from src.eval.metrics import (  # type: ignore
        DEFAULT_ECE_BINS,
        PAPER_REFERENCES,
        MetricAccumulator,
        compute_accuracy,
        compute_ece,
        softmax as _softmax_np,
    )
except Exception:  # pragma: no cover - defensive
    DEFAULT_ECE_BINS = 15
    PAPER_REFERENCES: Dict[str, Any] = {}
    MetricAccumulator = None  # type: ignore
    compute_accuracy = None  # type: ignore
    compute_ece = None  # type: ignore
    _softmax_np = None  # type: ignore

try:  # reuse the FOA driver's stream/metric machinery verbatim
    from scripts.run_foa import (  # type: ignore
        IMAGENET_C_CORRUPTIONS,
        ResultAccumulator,
        build_test_loader,
        run_foa,
    )
except Exception:  # pragma: no cover - direct execution fallback
    try:
        from run_foa import (  # type: ignore
            IMAGENET_C_CORRUPTIONS,
            ResultAccumulator,
            build_test_loader,
            run_foa,
        )
    except Exception:
        IMAGENET_C_CORRUPTIONS = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
            "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression",
        ]
        ResultAccumulator = None  # type: ignore
        build_test_loader = None  # type: ignore
        run_foa = None  # type: ignore

try:
    from src.models.vit_loader import FOA_CHECKPOINT_URL, build_vit  # type: ignore
except Exception:  # pragma: no cover
    FOA_CHECKPOINT_URL = ""
    build_vit = None  # type: ignore

LOGGER = logging.getLogger("foa.baselines")


# ======================================================================================
# Appendix B.2 hyper-parameters (verbatim from the paper)
# ======================================================================================
BASELINE_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    # Gradient-free / post-hoc ---------------------------------------------------------
    "lame": {
        "batch_size": 64,
        "knn": 5,                 # k chosen from {1,5,10,20}; 5 best on ImageNet-C
        "affinity": "knn",
        "kernel": "cosine",
        "sigma": None,            # None -> LAME's median-distance heuristic
        "force_balanced": False,
        "num_iters": 5,
        "temperature": 1.0,
    },
    "t3a": {
        "batch_size": 64,
        "num_supports": 20,       # M chosen from {1,5,20,50,100}; 20 best on ImageNet-C
        "filter_K": 50,
        "lam": 1.0,
        "temperature": 1.0,
    },
    # Gradient-based -------------------------------------------------------------------
    "tent": {
        "batch_size": 64,
        "lr": 1e-3,               # SGD, momentum 0.9, lr 0.001
        "momentum": 0.9,
        "weight_decay": 0.0,
        "episodic": False,
        "trainable": "norm_affine",
    },
    "sar": {
        "batch_size": 64,
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "rho": 0.05,
        "eta": 0.01,
        "entropy_factor": 0.4,    # E0 = 0.4 * ln C
        "num_blocks": 8,          # affine params of blocks 1..8
        "block_start": 1,
        "weighting": "exp",
        "restore": True,
        "episodic": False,
    },
    "cotta": {
        "batch_size": 64,
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "aug_threshold": 0.1,     # p_th
        "num_augmentations": 32,
        "restore_probability": 0.01,
        "ema_alpha": 0.999,       # teacher EMA factor
        "episodic": False,
        "trainable": "all",
    },
    "memo": {
        "batch_size": 64,
        "lr": 1e-3,
        "momentum": 0.9,
        "num_augmentations": 32,
        "steps": 1,
        "episodic": True,         # MEMO is per-sample / episodic by construction
    },
    "noadapt": {
        "batch_size": 64,
    },
}

POSTHOC_BASELINES: Tuple[str, ...] = ("noadapt", "lame", "t3a")
GRADIENT_BASELINES: Tuple[str, ...] = ("tent", "sar", "cotta", "memo")
KNOWN_BASELINES: Tuple[str, ...] = tuple(BASELINE_HYPERPARAMS.keys())

# Paper reference numbers (Table 2 / Table 3) used only for printing comparisons.
PAPER_TABLE2 = {
    "noadapt": (55.5, 10.5),
    "lame": (54.1, 11.0),
    "t3a": (56.9, 26.8),
    "tent": (59.6, 18.5),
    "sar": (62.7, 7.0),
    "cotta": (61.7, 6.5),
    "foa": (66.3, 3.2),
}
PAPER_TABLE3 = {  # ImageNet-R / -V2(MF) / -Sketch accuracy (%) and average ECE (%)
    "noadapt": {"r": 50.1, "v2": 61.9, "sketch": 42.0},
    "tent": {"r": 55.3, "v2": 66.9, "sketch": 45.2},
    "sar": {"r": 60.0, "v2": 71.2, "sketch": 47.6},
    "cotta": {"r": 60.1, "v2": 71.6, "sketch": 46.7},
    "foa": {"r": 63.8, "v2": 75.4, "sketch": 49.9},
}


# ======================================================================================
# Self-contained reference implementations (fallbacks)
# ======================================================================================
class NoAdaptBaseline:
    """Frozen-backbone reference: use the raw source logits (no adaptation at all)."""

    def __init__(self, num_classes: int = 1000, **_: Any) -> None:
        self.num_classes = num_classes

    def reset(self) -> None:  # pragma: no cover - stateless
        return None

    def step(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)


class LAMEBaseline:
    """Reference LAME (Boudiaf et al., 2022): post-hoc probability refinement.

    ``P_i <- sum_j W_ij P_j`` followed by column normalization, iterated a few times,
    where ``W`` is a (row-normalized) kNN affinity matrix computed in the frozen backbone's
    final-layer ``[CLS]`` feature space.  ``sigma`` follows LAME's median-distance heuristic
    when not provided.  Only the model's *output probabilities* are touched.
    """

    def __init__(
        self,
        knn: int = 5,
        affinity: str = "knn",
        kernel: str = "cosine",
        sigma: Optional[float] = None,
        force_balanced: bool = False,
        num_iters: int = 5,
        temperature: float = 1.0,
        **_: Any,
    ) -> None:
        self.knn = int(knn)
        self.affinity = affinity
        self.kernel = kernel
        self.sigma = sigma
        self.force_balanced = force_balanced
        self.num_iters = int(num_iters)
        self.temperature = float(temperature)

    # -- kernel -------------------------------------------------------------------
    def _affinity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        z = F.normalize(features.float(), dim=-1)
        sim = z @ z.t()                                  # cosine similarity in [-1, 1]
        if self.affinity == "knn" and 0 < self.knn < sim.shape[0]:
            k = self.knn
            topk_vals, topk_idx = torch.topk(sim, k=k + 1, dim=-1)  # include self, drop it
            W = torch.zeros_like(sim)
            rows = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1).expand(-1, k + 1)
            W[rows.reshape(-1), topk_idx.reshape(-1)] = topk_vals.reshape(-1)
            W.fill_diagonal_(0.0)
            # symmetrize like the reference implementation
            W = 0.5 * (W + W.t())
        else:
            W = sim.clamp_min(0.0)
        if self.kernel in ("rbf", "gaussian"):
            dist = (1.0 - sim).clamp_min(0.0)
            sigma = self.sigma
            if sigma is None:  # median-distance heuristic
                iu = torch.triu_indices(dist.shape[0], dist.shape[1], offset=1)
                sigma = float(dist[iu[0], iu[1]].median().clamp_min(1e-6)) if dist.shape[0] > 1 else 1.0
            W = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return W

    @torch.no_grad()
    def step(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float() / max(self.temperature, 1e-6), dim=-1)
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        if features is None or features.shape[0] < 2:
            return probs
        W = self._affinity_matrix(features)
        for _ in range(max(self.num_iters, 1)):
            probs = W @ probs
            probs = probs.clamp_min(1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if self.force_balanced:
            # marginal-preserving prior correction (optional in the official code)
            prior = probs.mean(dim=0, keepdim=True).clamp_min(1e-12)
            probs = probs / prior
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def reset(self) -> None:
        return None


class T3ABaseline:
    """Reference T3A (Iwasawa & Matsuo, 2021): prototype-based test-time classifier.

    Supports are the batch's own L2-normalized features (pseudo-labelled by the source
    classifier) after T3A's filter step; they are then augmented with the L2-normalized
    source classifier weights and classification is a temperature-scaled cosine similarity.
    ``num_supports`` is the paper's ``M`` (20).  Model parameters are never modified.
    """

    def __init__(
        self,
        num_supports: int = 20,
        filter_K: int = 50,
        lam: float = 1.0,
        temperature: float = 1.0,
        num_classes: int = 1000,
        **_: Any,
    ) -> None:
        self.num_supports = int(num_supports)
        self.filter_K = int(filter_K)
        self.lam = float(lam)
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)

    # -- filter / augment (T3A Sec. 3.2-3.3) ---------------------------------------
    def _build_supports(
        self, features: torch.Tensor, pseudo_labels: torch.Tensor, weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        z = F.normalize(features.float(), dim=-1)
        n = z.shape[0]
        if self.num_supports > 0 and n > self.num_supports:
            keep = min(self.num_supports, n)
            # T3A keeps the supports closest to their class prototype
            protos = torch.zeros(self.num_classes, z.shape[1], device=z.device, dtype=z.dtype)
            counts = torch.zeros(self.num_classes, device=z.device, dtype=z.dtype)
            protos.index_add_(0, pseudo_labels, z)
            counts.index_add_(0, pseudo_labels, torch.ones_like(pseudo_labels, dtype=z.dtype))
            proto_norm = F.normalize(protos.clamp_min(0.0), dim=-1)
            sim = (z * proto_norm[pseudo_labels]).sum(dim=-1)
            order = torch.argsort(sim, descending=True)[:keep]
            z = z[order]
            pseudo_labels = pseudo_labels[order]
        # augment with the normalized source classifier weights
        if weights is not None:
            w = F.normalize(weights.float(), dim=-1)
            z = torch.cat([z, w], dim=0)
            pseudo_labels = torch.cat(
                [
                    pseudo_labels,
                    torch.arange(self.num_classes, device=z.device, dtype=pseudo_labels.dtype),
                ],
                dim=0,
            )
        return z, pseudo_labels

    @torch.no_grad()
    def step(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features is None:
            return F.softmax(logits.float(), dim=-1)
        pseudo = logits.float().argmax(dim=-1)
        z, labels = self._build_supports(features, pseudo, weights)
        zn = F.normalize(features.float(), dim=-1)
        sim = zn @ z.t()                                   # [B, S]
        sim = (sim / max(self.temperature, 1e-6)) * self.lam
        # aggregate per class with log-sum-exp (equivalent to a prototype classifier)
        out = torch.full(
            (zn.shape[0], self.num_classes), -1e4, device=logits.device, dtype=torch.float32
        )
        out.scatter_reduce_(1, labels.unsqueeze(0).expand(zn.shape[0], -1), sim, reduce="amax")
        return F.softmax(out, dim=-1)

    def reset(self) -> None:
        return None


# ======================================================================================
# Wrapper construction
# ======================================================================================
def _resolve_device(cfg: Any = None, device: Optional[str] = None) -> torch.device:
    if device is not None:
        dev = torch.device(device)
    else:
        name = None
        try:
            name = cfg.model.device  # type: ignore[attr-defined]
        except Exception:
            name = None
        dev = torch.device(name or "cuda")
    if dev.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable - falling back to CPU.")
        dev = torch.device("cpu")
    return dev


def merged_baseline_kwargs(name: str, cfg: Any = None, **overrides: Any) -> Dict[str, Any]:
    """Appendix B.2 defaults + optional ``baselines.<name>`` YAML block + CLI overrides."""
    kwargs: Dict[str, Any] = dict(BASELINE_HYPERPARAMS.get(name, {}))
    cfg_block: Dict[str, Any] = {}
    for key in (name, "default"):
        try:
            block = cfg.baselines[key]  # type: ignore[attr-defined]
        except Exception:
            try:
                block = (cfg.get("baselines", {}) or {}).get(key)  # type: ignore[union-attr]
            except Exception:
                block = None
        if isinstance(block, dict):
            cfg_block = dict(block)
            break
    # legacy top-level block (e.g. top-level `sar:`)
    if not cfg_block:
        try:
            block = cfg[name]  # type: ignore[attr-defined]
            if isinstance(block, dict):
                cfg_block = dict(block)
        except Exception:
            pass
    kwargs.update(cfg_block)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    # resolve E0 = 0.4 * ln C for SAR when a class count is available
    num_classes = kwargs.get("num_classes")
    if num_classes is None and cfg is not None:
        for path in (("data", "num_classes_eval"), ("model", "num_classes")):
            try:
                num_classes = cfg[path[0]][path[1]]  # type: ignore[index]
                break
            except Exception:
                continue
    if name == "sar" and num_classes and kwargs.get("entropy_threshold") is None:
        factor = kwargs.get("entropy_factor", 0.4)
        kwargs["num_classes"] = int(num_classes)
        kwargs["entropy_threshold"] = float(factor) * math.log(int(num_classes))
    elif num_classes and "num_classes" not in kwargs:
        kwargs["num_classes"] = int(num_classes)
    return kwargs


def load_baseline_module(name: str):
    """Import ``src.baselines.<name>`` (returns ``None`` when unavailable)."""
    try:
        import importlib

        return importlib.import_module(f"src.baselines.{name}")
    except Exception as exc:  # pragma: no cover - graceful degradation
        LOGGER.info("Baseline module src.baselines.%s unavailable (%s); using built-in fallback.", name, exc)
        return None


def build_baseline_wrapper(
    name: str,
    model: Optional[torch.nn.Module] = None,
    cfg: Any = None,
    device: Optional[torch.device] = None,
    **overrides: Any,
):
    """Instantiate the baseline wrapper for ``name``.

    Prefers ``src/baselines/<name>.py`` (``build_<name>`` / ``build_baseline`` / ``build``),
    falling back to the self-contained reference classes above.
    """
    kwargs = merged_baseline_kwargs(name, cfg, **overrides)
    module = load_baseline_module(name)
    if module is not None:
        for factory_name in (f"build_{name}", "build_baseline", "build"):
            factory = getattr(module, factory_name, None)
            if callable(factory):
                try:
                    return factory(model=model, cfg=cfg, **kwargs)
                except TypeError:
                    try:
                        return factory(model, cfg, **kwargs)
                    except TypeError:
                        continue
        klass = getattr(module, name.upper(), None) or getattr(module, "".join(
            [p.capitalize() for p in name.split("_")]
        ), None)
        if klass is not None:
            try:
                return klass(model, **kwargs)
            except TypeError:
                return klass(**kwargs)
    # ---- built-in fallbacks --------------------------------------------------------
    if name == "noadapt":
        return NoAdaptBaseline(**kwargs)
    if name == "lame":
        return LAMEBaseline(**kwargs)
    if name == "t3a":
        return T3ABaseline(**kwargs)
    LOGGER.warning("No implementation available for baseline '%s' - skipping.", name)
    return None


def build_baseline_model(cfg: Any, device: torch.device) -> torch.nn.Module:
    """Frozen ViT-Base backbone (shared feature extractor for every baseline)."""
    if build_vit is None:
        raise RuntimeError("src.models.vit_loader.build_vit is unavailable.")
    name = "vit_base_patch16_224"
    checkpoint = None
    try:
        name = cfg.model.name
        checkpoint = cfg.model.checkpoint
    except Exception:
        pass
    model = build_vit(model_name=name, checkpoint=checkpoint, pretrained=True, device=str(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ======================================================================================
# Per-batch stepping
# ======================================================================================
def _batch_images_labels(batch: Any, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    images = None
    labels = None
    if isinstance(batch, dict):
        for key in ("image", "images", "x", "inputs", "pixel_values"):
            if key in batch:
                images = batch[key]
                break
        for key in ("label", "labels", "y", "target", "targets"):
            if key in batch:
                labels = batch[key]
                break
    elif isinstance(batch, (list, tuple)):
        images = batch[0] if len(batch) > 0 else None
        labels = batch[1] if len(batch) > 1 else None
    if images is None:
        raise ValueError("Could not find images in batch.")
    if isinstance(images, (list, tuple)):
        images = torch.stack([im if torch.is_tensor(im) else torch.as_tensor(im) for im in images])
    images = images.to(device, non_blocking=True)
    if labels is not None and not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    if labels is not None:
        labels = labels.to(device, non_blocking=True)
    return images, labels


def _features_and_logits(model: torch.nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (final-layer CLS feature, logits) for post-hoc baselines."""
    with torch.no_grad():
        if hasattr(model, "forward_with_features"):
            out = model.forward_with_features(images)  # type: ignore[attr-defined]
            if isinstance(out, dict):
                feats = out.get("final_cls", None)
                if feats is None:
                    cls = out.get("cls_features")
                    feats = cls[-1] if isinstance(cls, (list, tuple)) else None
                logits = out.get("logits")
                if logits is None:
                    logits = model.head(feats) if hasattr(model, "head") else None
                if logits is not None:
                    return feats, logits
        logits = model(images)
        if isinstance(logits, dict):
            logits = logits.get("logits")
    return None, logits  # type: ignore[return-value]


def _classifier_weight(model: torch.nn.Module) -> Optional[torch.Tensor]:
    for attr in ("head",):
        head = getattr(model, attr, None)
        if head is None:
            continue
        weight = getattr(head, "weight", None)
        if weight is not None:
            return weight.detach()
        inner = getattr(head, "model", None)  # FOA wrapper stores the timm model
        if inner is not None:
            w = getattr(inner, "weight", None)
            if w is not None:
                return w.detach()
    inner = getattr(model, "model", None)
    if inner is not None:
        h = getattr(inner, "head", None)
        w = getattr(h, "weight", None)
        if w is not None:
            return w.detach()
    return None


def _to_probs(output: Any, num_classes: Optional[int] = None) -> torch.Tensor:
    """Normalize a wrapper output into a probability tensor."""
    if isinstance(output, dict):
        for key in ("probs", "probabilities", "softmax", "logits", "pred"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (list, tuple)):
        output = output[-1]
    output = output.float()
    if output.min() < 0.0 or not torch.allclose(
        output.sum(dim=-1), torch.ones(output.shape[0], device=output.device), atol=1e-3
    ):
        output = F.softmax(output, dim=-1)
    return output


# ======================================================================================
# Main runner
# ======================================================================================
def run_baseline(
    name: str,
    cfg: Any = None,
    loader: Optional[Iterable] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    corruption: Optional[str] = None,
    class_subset: Optional[Sequence[int]] = None,
    verbose: bool = True,
    log_every: int = 20,
) -> Dict[str, Any]:
    """Run one baseline over one ordered online stream and report accuracy/ECE.

    Mirrors ``run_foa``'s protocol: single forward pass per batch, never shuffles, and the
    reported prediction is the method's own output (no candidate averaging).
    """
    name = (name or "").lower()
    device = _resolve_device(cfg, str(device) if device is not None else None)
    if model is None:
        model = build_baseline_model(cfg, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if loader is None:
        if build_test_loader is None:
            raise RuntimeError("src/data/datasets.py loader is unavailable; pass `loader=` explicitly.")
        loader = build_test_loader(cfg, corruption=corruption, device=device)

    wrapper = build_baseline_wrapper(name, model=model, cfg=cfg, device=device)
    if wrapper is None:
        raise ValueError(f"Unknown baseline '{name}'. Known: {KNOWN_BASELINES}")

    ece_bins = DEFAULT_ECE_BINS
    try:
        ece_bins = int(cfg.eval.ece_bins)
    except Exception:
        pass
    if ResultAccumulator is not None:
        accumulator = ResultAccumulator(ece_bins=ece_bins)
    else:  # minimal fallback accumulator (accuracy + ECE, 15 equal-width bins)
        accumulator = _FallbackAccumulator(ece_bins)

    posthoc = name in POSTHOC_BASELINES
    weights = _classifier_weight(model) if name == "t3a" else None
    per_batch: List[Dict[str, Any]] = []
    t0 = time.time()
    n_batches = 0
    for step_idx, batch in enumerate(loader):
        images, targets = _batch_images_labels(batch, device)
        if posthoc:
            feats, logits = _features_and_logits(model, images)
            if name == "t3a":
                probs = wrapper.step(feats, logits, weights) if feats is not None else F.softmax(logits, -1)
            else:
                probs = wrapper.step(feats, logits)
        else:
            try:
                out = wrapper.step(images, targets)  # TENT/SAR/CoTTA/MEMO protocol
            except TypeError:
                out = wrapper.step(images)
            probs = _to_probs(out)
        if class_subset is not None:
            probs = probs[:, list(class_subset)]
        logits_for_metrics = torch.log(probs.clamp_min(1e-12))
        if targets is not None:
            acc, ece = accumulator.update(logits_for_metrics, targets)
        else:
            acc, ece = float("nan"), float("nan")
        n_batches += 1
        if verbose and log_every and (n_batches % log_every == 0):
            LOGGER.info("[%s%s] batch %d acc=%.2f ece=%.2f", name,
                        f"/{corruption}" if corruption else "", n_batches, acc, ece)
        per_batch.append({"batch": n_batches, "accuracy": acc, "ece": ece})
    summary = accumulator.compute()
    elapsed = time.time() - t0
    result: Dict[str, Any] = {
        "method": name,
        "corruption": corruption,
        "accuracy": float(summary.get("accuracy", float("nan"))),
        "ece": float(summary.get("ece", float("nan"))),
        "num_samples": int(summary.get("num_samples", 0)),
        "num_batches": n_batches,
        "posthoc": bool(posthoc),
        "batch_size": int(merged_baseline_kwargs(name, cfg).get("batch_size", 64)),
        "hyperparameters": merged_baseline_kwargs(name, cfg),
        "wall_clock_s": elapsed,
        "per_batch": per_batch,
    }
    if verbose:
        LOGGER.info(
            "Baseline %s%s: accuracy=%.2f%%  ECE=%.2f%%  (%d samples, %.1fs)",
            name, f"/{corruption}" if corruption else "", result["accuracy"], result["ece"],
            result["num_samples"], elapsed,
        )
    return result


class _FallbackAccumulator:
    """Minimal accuracy/ECE accumulator used when ``scripts.run_foa`` is unavailable."""

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = int(n_bins)
        self._probs: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        logits_np = logits.detach().cpu().float().numpy()
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        if _softmax_np is not None:
            probs = _softmax_np(logits_np, axis=-1)
        else:
            shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
            e = np.exp(shifted)
            probs = e / e.sum(axis=-1, keepdims=True)
        self._probs.append(probs.astype(np.float32))
        self._labels.append(targets_np.astype(np.int64))
        return self._running(probs, targets_np)

    def _running(self, probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        preds = probs.argmax(axis=-1)
        acc = 100.0 * float((preds == labels).mean()) if labels.size else float("nan")
        conf = probs.max(axis=-1)
        correct = (preds == labels).astype(np.float64)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, self.n_bins - 1)
        ece = 0.0
        for b in range(self.n_bins):
            m = idx == b
            if m.any():
                ece += (m.sum() / conf.size) * abs(correct[m].mean() - conf[m].mean())
        return acc, 100.0 * float(ece)

    def compute(self) -> Dict[str, Any]:
        if not self._probs:
            return {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
        probs = np.concatenate(self._probs, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        acc, ece = self._running(probs, labels)
        return {"accuracy": acc, "ece": ece, "num_samples": int(labels.size)}


class _MaybeLimited:
    """Wrap a stream/loader so at most ``limit`` batches are consumed (smoke tests)."""

    def __init__(self, loader: Iterable, limit: Optional[int] = None) -> None:
        self.loader = loader
        self.limit = limit
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.limit is not None and i >= self.limit:
                break
            yield batch

    def __len__(self) -> int:
        try:
            total = len(self.loader)  # type: ignore[arg-type]
        except Exception:
            total = 0
        return min(total, self.limit) if (self.limit is not None and total) else (self.limit or total)


def run_over_corruptions(
    name: str,
    cfg: Any,
    corruptions: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Average one baseline over the 15 ImageNet-C corruptions (Table 2 / 16 protocol)."""
    if corruptions is None:
        try:
            corruptions = list(cfg.data.corruptions)
        except Exception:
            corruptions = list(IMAGENET_C_CORRUPTIONS)
    severity = None
    try:
        severity = int(cfg.data.severity)
    except Exception:
        severity = 5
    per_corruption: Dict[str, Dict[str, Any]] = {}
    for corruption in corruptions:
        seed = None
        try:
            seed = int(cfg.seed)
        except Exception:
            seed = 0
        _seed_everything(seed)
        loader = build_test_loader(cfg, corruption=corruption, severity=severity, device=device)
        if limit_batches is not None:
            loader = _MaybeLimited(loader, limit=limit_batches)
        per_corruption[corruption] = run_baseline(
            name, cfg, loader=loader, device=device, corruption=corruption, verbose=verbose
        )
    accs = [v["accuracy"] for v in per_corruption.values() if not math.isnan(v["accuracy"])]
    eces = [v["ece"] for v in per_corruption.values() if not math.isnan(v["ece"])]
    return {
        "method": name,
        "dataset": "imagenet-c",
        "severity": severity,
        "num_corruptions": len(per_corruption),
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "ece": float(np.mean(eces)) if eces else float("nan"),
        "per_corruption": per_corruption,
    }


# ======================================================================================
# CLI
# ======================================================================================
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_corruptions(cfg: Any, args) -> List[str]:
    if args.all_corruptions:
        try:
            return list(cfg.data.corruptions)
        except Exception:
            return list(IMAGENET_C_CORRUPTIONS)
    if args.corruptions:
        return [c.strip() for c in args.corruptions.split(",") if c.strip()]
    try:
        if cfg.data.corruption:
            return [cfg.data.corruption]
        if cfg.data.corruptions:
            return list(cfg.data.corruptions)
    except Exception:
        pass
    return [IMAGENET_C_CORRUPTIONS[0]]


def _apply_overrides(cfg: Any, args) -> Any:
    if getattr(args, "dataset", None):
        cfg.data.dataset = args.dataset
    if getattr(args, "severity", None):
        cfg.data.severity = int(args.severity)
    if getattr(args, "batch_size", None):
        cfg.data.batch_size = int(args.batch_size)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "device", None):
        cfg.model.device = args.device
    return cfg


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TTA baselines for the FOA reproduction.")
    parser.add_argument("--config", required=True, help="Base YAML config (e.g. configs/foa_imagenetc.yaml).")
    parser.add_argument("--extra-config", default=None, help="Optional second YAML merged on top.")
    parser.add_argument("--methods", default="tent,sar,cotta,t3a,lame,noadapt",
                        help="Comma-separated baseline names.")
    parser.add_argument("--dataset", default=None, help="Override data.dataset.")
    parser.add_argument("--corruptions", default=None, help="Comma-separated corruption names.")
    parser.add_argument("--all-corruptions", action="store_true", help="Average over all corruptions.")
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="JSON output path.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Smoke-test cap on batches.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def print_summary(results: Dict[str, Dict[str, Any]], dataset: str = "imagenet-c") -> None:
    """Print an accuracy/ECE comparison table including the paper's reference rows."""
    print("\n" + "=" * 78)
    print(f"TTA baseline comparison ({dataset}, severity "
          f"{next(iter(results.values()))['severity'] if results else 5})")
    print("=" * 78)
    print(f"{'Method':<14}{'Accuracy (%)':>14}{'ECE (%)':>12}{'Samples':>10}")
    print("-" * 78)
    for name, res in results.items():
        print(f"{name:<14}{res['accuracy']:>14.2f}{res['ece']:>12.2f}{res['num_samples']:>10}")
    print("-" * 78)
    ref = PAPER_TABLE2 if dataset.startswith("imagenet-c") else {}
    if ref:
        print("Paper reference (Table 2, ImageNet-C severity 5):")
        for key, (acc, ece) in ref.items():
            print(f"  {key:<12}{acc:>14.2f}{ece:>12.2f}")
    print("=" * 78 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    configs = [args.config] + ([args.extra_config] if args.extra_config else [])
    cfg = load_config(*configs)
    cfg = _apply_overrides(cfg, args)
    seed = int(getattr(args, "seed", None) if args.seed is not None else
               (cfg.seed if hasattr(cfg, "seed") else 0))
    _seed_everything(seed)
    device = _resolve_device(cfg, args.device)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    corruptions = _resolve_corruptions(cfg, args)
    dataset = str(getattr(cfg.data, "dataset", "imagenet-c"))
    results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        if method not in KNOWN_BASELINES:
            LOGGER.warning("Skipping unknown baseline '%s'. Known: %s", method, KNOWN_BASELINES)
            continue
        if len(corruptions) > 1:
            results[method] = run_over_corruptions(
                method, cfg, corruptions=corruptions, device=device,
                verbose=not args.quiet, limit_batches=args.limit_batches,
            )
        else:
            loader = build_test_loader(cfg, corruption=corruptions[0], device=device)
            if args.limit_batches is not None:
                loader = _MaybeLimited(loader, limit=args.limit_batches)
            res = run_baseline(method, cfg, loader=loader, device=device,
                               corruption=corruptions[0], verbose=not args.quiet,
                               log_every=args.log_every)
            res["severity"] = int(getattr(cfg.data, "severity", 5))
            results[method] = res

    print_summary(results, dataset=dataset)

    out_path = args.output
    if out_path is None:
        out_dir = str(getattr(cfg, "output_dir", "./outputs"))
        os.makedirs(out_dir, exist_ok=True)
        exp = str(getattr(getattr(cfg, "experiment", {}), "get", lambda *_: "baselines")("name", "baselines"))
        out_path = os.path.join(out_dir, f"{exp}_baselines.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"config": config_to_dict(cfg), "results": results}, fh, indent=2, default=str)
    LOGGER.info("Wrote baseline results to %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
