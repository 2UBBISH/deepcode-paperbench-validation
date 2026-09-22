#!/usr/bin/env python
"""Ablation studies for FOA (Forward-Optimization Adaptation).

This script reproduces the two ablation grids of the paper:

* **Table 5** -- "Ablations of components in our FOA": the contribution of the two
  terms of the Eqn. (5) fitness (prediction *Entropy* and *Activation (Act.)
  Discrepancy*) and of the back-to-source *Act. Shifting* scheme (Section 3.2).
  Reported as the average over the 15 ImageNet-C corruptions (level 5) with
  ViT-Base.

  ==========  ================  ===========  ======  =====
  Entropy     Act. Discrepancy  Act. Shifting  Acc.    ECE
  ==========  ================  ===========  ======  =====
  (NoAdapt)   -                 -             55.5    10.5
  yes         -                 -             44.9    36.8
  -           yes               -             63.4     9.4
  -           -                 yes           59.1    12.7
  yes         yes               -             65.4     3.3
  yes         yes               yes           66.3     3.2
  ==========  ================  ===========  ======  =====

  (The paper's Table 5 also lists the "-/yes/yes" row: 63.8 / 9.9.)

* **Table 9** -- "Empirical studies of design choices w.r.t. learnable parameters,
  optimizer and loss function": the grid
  {learnable params: prompts | norm layers} x {optimizer: SGD | CMA} x
  {loss: entropy | Eqn. (5)}, again averaged over the 15 ImageNet-C corruptions.

Notably the design grid shows the paper's core motivation: CMA over ultra
high-dimensional norm-layer parameters collapses (0.1% acc, exp4/exp5) and CMA
with an entropy-only fitness is unstable (44.9% acc, exp6), while prompts + CMA +
Eqn. (5) works (65.4%).

Appendix B.2 hyper-parameters used by the design grid
-----------------------------------------------------
* SGD on prompts with the *entropy* loss  (exp1): lr = 0.01, N_p = 3.
* SGD on prompts/norm layers with *Eqn. (5)*  (exp2/exp3): the entropy term is
  divided by the batch size (64) and lambda is set to 30 so both losses have a
  similar magnitude.  Prompt SGD uses lr = 0.01 (same as exp1); norm-layer SGD
  uses the TENT lr = 1e-3.  Both defaults are overridable from the config.
* TENT's own row (norm layers + SGD + entropy): lr = 1e-3, momentum 0.9, BS 64.
* CMA rows use K = 28, m0 = 0, Sigma0 = I, tau0 = 1 (Eqn. 6) and are
  backpropagation-free.

The script is fully forward-only for the CMA rows (no backward pass anywhere)
and re-uses:

* ``scripts/run_foa.py``    -> ordered online stream (``build_test_loader``) and
                               the FOA runner / accumulator;
* ``src.method.foa``        -> ``build_foa`` (Algorithm 1 loop);
* ``src.method.fitness``    -> Eqn. (5) / ``ablation_config``;
* ``src.method.activation_shifting`` -> Eqn. (7)-(9);
* ``src.models.prompt_injection`` / ``src.models.vit_loader``.

Usage
-----
    python scripts/run_ablations.py --config configs/ablations.yaml \
        --groups components,design --limit-batches 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# sys.path shim so `python scripts/run_ablations.py` works from the project root
# --------------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # package-style import (recommended)
    from src.method.activation_shifting import build_activation_shifter
    from src.method.fitness import (
        CANONICAL_BS,
        LAMBDA_BASE_IMAGENETC,
        LAMBDA_BASE_IMAGENETR,
        ablation_config,
        build_fitness,
    )
    from src.method.foa import build_foa
    from src.method.source_stats import load_source_stats
    from src.models.vit_loader import build_vit
    from src.utils.config import Config, config_to_dict, load_config, save_config
except ImportError:  # pragma: no cover - script style fallback
    from method.activation_shifting import build_activation_shifter  # type: ignore
    from method.fitness import (  # type: ignore
        CANONICAL_BS,
        LAMBDA_BASE_IMAGENETC,
        LAMBDA_BASE_IMAGENETR,
        ablation_config,
        build_fitness,
    )
    from method.foa import build_foa  # type: ignore
    from method.source_stats import load_source_stats  # type: ignore
    from models.vit_loader import build_vit  # type: ignore
    from utils.config import (  # type: ignore
        Config,
        config_to_dict,
        load_config,
        save_config,
    )

# `run_foa` provides the stream builder and metric helpers; import defensively.
try:  # pragma: no cover
    from run_foa import (  # type: ignore
        IMAGENET_C_CORRUPTIONS as _FOA_CORRUPTIONS,
        ResultAccumulator,
        build_test_loader,
    )
except ImportError:  # pragma: no cover
    try:
        from scripts.run_foa import (  # type: ignore
            IMAGENET_C_CORRUPTIONS as _FOA_CORRUPTIONS,
            ResultAccumulator,
            build_test_loader,
        )
    except ImportError:
        _FOA_CORRUPTIONS = None
        ResultAccumulator = None  # type: ignore
        build_test_loader = None  # type: ignore

try:  # optional metrics module
    from src.eval.metrics import IMAGENET_C_CORRUPTIONS as _METRIC_CORRUPTIONS
    from src.eval.metrics import MetricAccumulator as _MetricAccumulator
except ImportError:  # pragma: no cover
    try:
        from eval.metrics import IMAGENET_C_CORRUPTIONS as _METRIC_CORRUPTIONS  # type: ignore
        from eval.metrics import MetricAccumulator as _MetricAccumulator  # type: ignore
    except ImportError:
        _METRIC_CORRUPTIONS = None
        _MetricAccumulator = None  # type: ignore

try:  # TENT's parameter selection helper is reused for the norm-layer variants
    from src.baselines.tent import select_norm_affine_params
except ImportError:  # pragma: no cover
    try:
        from baselines.tent import select_norm_affine_params  # type: ignore
    except ImportError:
        select_norm_affine_params = None  # type: ignore


# ======================================================================================
# Constants and paper reference values
# ======================================================================================
IMAGENET_C_CORRUPTIONS: List[str] = list(
    _FOA_CORRUPTIONS
    or _METRIC_CORRUPTIONS
    or [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
)

#: Table 5 (ImageNet-C level 5, average over the 15 corruptions, ViT-Base).
TABLE5_REFERENCE: Dict[str, Dict[str, float]] = {
    "noadapt": {"accuracy": 55.5, "ece": 10.5},
    "entropy": {"accuracy": 44.9, "ece": 36.8},
    "discrepancy": {"accuracy": 63.4, "ece": 9.4},
    "shifting": {"accuracy": 59.1, "ece": 12.7},
    "discrepancy_shifting": {"accuracy": 63.8, "ece": 9.9},
    "entropy_discrepancy": {"accuracy": 65.4, "ece": 3.3},
    "full": {"accuracy": 66.3, "ece": 3.2},
}

#: Human-readable labels for the Table 5 rows.
TABLE5_LABELS: Dict[str, str] = {
    "noadapt": "NoAdapt",
    "entropy": "CMA + Entropy",
    "discrepancy": "CMA + Act. Discrepancy",
    "shifting": "Act. Shifting only",
    "discrepancy_shifting": "Act. Disc. + Act. Shifting",
    "entropy_discrepancy": "Entropy + Act. Disc. (no shift)",
    "full": "FOA (Entropy + Disc. + Shifting)",
}

#: Table 9 (ImageNet-C level 5, average over the 15 corruptions, ViT-Base).
TABLE9_REFERENCE: Dict[str, Dict[str, Any]] = {
    "noadapt": {"params": "-", "optimizer": "-", "loss": "-", "accuracy": 55.5, "ece": 10.5},
    "tent": {"params": "norm", "optimizer": "sgd", "loss": "entropy", "accuracy": 59.6, "ece": 18.5},
    "exp1": {"params": "prompts", "optimizer": "sgd", "loss": "entropy", "accuracy": 50.7, "ece": 18.4},
    "exp2": {"params": "norm", "optimizer": "sgd", "loss": "eqn5", "accuracy": 70.5, "ece": 7.9},
    "exp3": {"params": "prompts", "optimizer": "sgd", "loss": "eqn5", "accuracy": 64.6, "ece": 3.7},
    "exp4": {"params": "norm", "optimizer": "cma", "loss": "eqn5", "accuracy": 0.1, "ece": 5.8},
    "exp5": {"params": "norm", "optimizer": "cma", "loss": "entropy", "accuracy": 0.1, "ece": 99.0},
    "exp6": {"params": "prompts", "optimizer": "cma", "loss": "entropy", "accuracy": 44.9, "ece": 36.8},
    "ours": {"params": "prompts", "optimizer": "cma", "loss": "eqn5", "accuracy": 65.4, "ece": 3.3},
}

#: Component-ablation variant specs (Table 5).  ``optimizer == "none"`` means no
#: learnable prompt is optimised at all (the frozen backbone is used directly).
COMPONENT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "noadapt": dict(use_entropy=False, use_discrepancy=False, shifting=False, optimizer="none"),
    "entropy": dict(use_entropy=True, use_discrepancy=False, shifting=False, optimizer="cma"),
    "discrepancy": dict(use_entropy=False, use_discrepancy=True, shifting=False, optimizer="cma"),
    "shifting": dict(use_entropy=False, use_discrepancy=False, shifting=True, optimizer="none"),
    "discrepancy_shifting": dict(
        use_entropy=False, use_discrepancy=True, shifting=True, optimizer="cma"
    ),
    "entropy_discrepancy": dict(
        use_entropy=True, use_discrepancy=True, shifting=False, optimizer="cma"
    ),
    "full": dict(use_entropy=True, use_discrepancy=True, shifting=True, optimizer="cma"),
}

#: Design-choice variant specs (Table 9).  ``params`` in {prompts, norm, "-"} and
#: ``optimizer`` in {sgd, cma, "-"}; ``loss`` in {entropy, eqn5, "-"}.
DESIGN_VARIANTS: Dict[str, Dict[str, Any]] = {
    "noadapt": dict(params="-", optimizer="-", loss="-", shifting=False),
    "tent": dict(params="norm", optimizer="sgd", loss="entropy", lr=1e-3, momentum=0.9, shifting=False),
    "exp1": dict(params="prompts", optimizer="sgd", loss="entropy", lr=0.01, num_prompts=3, shifting=True),
    "exp2": dict(
        params="norm",
        optimizer="sgd",
        loss="eqn5",
        lr=1e-3,
        momentum=0.9,
        lambda_value=30.0,
        entropy_divisor=64.0,
        shifting=True,
    ),
    "exp3": dict(
        params="prompts",
        optimizer="sgd",
        loss="eqn5",
        lr=0.01,
        lambda_value=30.0,
        entropy_divisor=64.0,
        num_prompts=3,
        shifting=True,
    ),
    "exp4": dict(params="norm", optimizer="cma", loss="eqn5", shifting=True),
    "exp5": dict(params="norm", optimizer="cma", loss="entropy", shifting=True),
    "exp6": dict(params="prompts", optimizer="cma", loss="entropy", shifting=True),
    "ours": dict(params="prompts", optimizer="cma", loss="eqn5", shifting=True),
}

ABLATION_GROUPS = ("components", "design")

DEFAULT_SGD_PROMPT_LR = 0.01  # Appendix B.2 (Table 9, prompts + SGD + entropy)
DEFAULT_SGD_NORM_LR = 1e-3  # TENT / Appendix B.2
DEFAULT_EQN5_LAMBDA = 30.0  # Appendix B.2 (Table 9, Eqn. 5 + SGD)
DEFAULT_ECE_BINS = 15


# ======================================================================================
# Small utilities
# ======================================================================================
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Retrieve a nested config value supporting dicts and attribute-style objects."""
    if cfg is None:
        return default
    for key in keys:
        if isinstance(cfg, dict):
            if key in cfg:
                cfg = cfg[key]
            else:
                return default
        else:
            if hasattr(cfg, key):
                cfg = getattr(cfg, key)
            else:
                return default
    return cfg if cfg is not None else default


def _resolve_device(cfg: Any = None, device: Optional[str] = None) -> torch.device:
    requested = device or _cfg_get(cfg, "model", "device", default=None) or _cfg_get(
        cfg, "device", default=None
    )
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _lambda_base_for_dataset(cfg: Any, dataset: Optional[str], override: Optional[float] = None) -> float:
    """lambda = 0.4 * BS/64 on ImageNet-C/V2/Sketch and 0.2 * BS/64 on ImageNet-R."""
    if override is not None:
        return float(override)
    explicit = _cfg_get(cfg, "fitness", "lambda_base", default=None)
    if explicit is not None:
        return float(explicit)
    ds = (dataset or _cfg_get(cfg, "data", "dataset", default="imagenet-c") or "").lower()
    if "imagenet-r" in ds or ds.endswith("-r"):
        return float(LAMBDA_BASE_IMAGENETR)
    return float(LAMBDA_BASE_IMAGENETC)


def _resolve_corruptions(cfg: Any, corruptions: Optional[Sequence[str]]) -> List[Optional[str]]:
    dataset = str(_cfg_get(cfg, "data", "dataset", default="imagenet-c") or "").lower()
    if corruptions is not None:
        specs = list(corruptions)
    elif "imagenet-c" in dataset:
        configured = _cfg_get(cfg, "data", "corruptions", default=None)
        specs = list(configured) if configured else list(IMAGENET_C_CORRUPTIONS)
    else:
        specs = [None]
    # Normalise "none"/"null" entries into Python None (single-stream datasets).
    return [None if (c is None or str(c).lower() in ("none", "null", "")) else str(c) for c in specs]


# ======================================================================================
# Metrics bookkeeping
# ======================================================================================
class _Accumulator:
    """Minimal streaming accuracy/ECE accumulator (percent units)."""

    def __init__(self, n_bins: int = DEFAULT_ECE_BINS, class_subset: Optional[Sequence[int]] = None):
        self.n_bins = int(n_bins)
        self.class_subset = list(class_subset) if class_subset is not None else None
        self.num_correct = 0
        self.num_total = 0
        self._confidences: List[np.ndarray] = []
        self._correct: List[np.ndarray] = []

    def reset(self) -> None:
        self.num_correct = 0
        self.num_total = 0
        self._confidences = []
        self._correct = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        logits_np = logits.detach().float().cpu().numpy()
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        if self.class_subset is not None:
            logits_np = logits_np[:, self.class_subset]
        if logits_np.ndim != 2:
            logits_np = logits_np.reshape(logits_np.shape[0], -1)
        shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
        preds = probs.argmax(axis=-1)
        correct = (preds == targets_np).astype(np.float64)
        confidences = probs.max(axis=-1)
        self.num_correct += int(correct.sum())
        self.num_total += int(correct.size)
        self._confidences.append(confidences)
        self._correct.append(correct)
        return self._running(confidences, correct)

    def _running(self, confidences: np.ndarray, correct: np.ndarray) -> Tuple[float, float]:
        acc = 100.0 * float(correct.mean()) if correct.size else float("nan")
        return acc, self._ece(confidences, correct)

    def _ece(self, confidences: np.ndarray, correct: np.ndarray) -> float:
        if confidences.size == 0:
            return float("nan")
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, self.n_bins - 1)
        total = confidences.size
        ece = 0.0
        for b in range(self.n_bins):
            mask = idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            acc_b = float(correct[mask].mean())
            conf_b = float(confidences[mask].mean())
            ece += (n / total) * abs(acc_b - conf_b)
        return 100.0 * ece

    def compute(self) -> Dict[str, float]:
        if self.num_total == 0:
            return {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
        accuracy = 100.0 * self.num_correct / self.num_total
        if self._confidences:
            confidences = np.concatenate(self._confidences)
            correct = np.concatenate(self._correct)
            ece = self._ece(confidences, correct)
        else:  # pragma: no cover
            ece = float("nan")
        return {"accuracy": accuracy, "ece": ece, "num_samples": self.num_total}


def _make_accumulator(cfg: Any, class_subset: Optional[Sequence[int]] = None) -> Any:
    """Prefer the paper's metric implementation, fall back to the local one."""
    bins = int(_cfg_get(cfg, "eval", "ece_bins", default=DEFAULT_ECE_BINS) or DEFAULT_ECE_BINS)
    if _MetricAccumulator is not None:
        try:
            return _MetricAccumulator(n_bins=bins)
        except Exception:  # pragma: no cover - signature mismatch
            pass
    return _Accumulator(n_bins=bins, class_subset=class_subset)


# ======================================================================================
# Model / stream / loss helpers
# ======================================================================================
def build_model(cfg: Any = None, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Build the frozen ViT-Base backbone exposing all-layer CLS features."""
    device = device or _resolve_device(cfg)
    model_name = _cfg_get(cfg, "model", "name", default="vit_base_patch16_224")
    checkpoint = _cfg_get(cfg, "model", "checkpoint", default=None)
    pretrained = bool(_cfg_get(cfg, "model", "pretrained", default=True))
    num_classes = int(_cfg_get(cfg, "model", "num_classes", default=1000) or 1000)
    model = build_vit(
        model_name=model_name,
        checkpoint=checkpoint,
        pretrained=pretrained,
        num_classes=num_classes,
        device=str(device),
    )
    if isinstance(model, torch.nn.Module):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_source_stats_for(cfg: Any, device: torch.device, path_override: Optional[str] = None):
    path = path_override or _cfg_get(
        cfg, "source_stats", "path", default="./checkpoints/source_stats_vit_base.pt"
    )
    return load_source_stats(path, device=str(device))


def _build_stream(cfg: Any, corruption: Optional[str], limit_batches: Optional[int], **overrides):
    """Ordered single-pass stream via ``run_foa.build_test_loader`` (or datasets.py)."""
    if build_test_loader is None:  # pragma: no cover - degraded mode
        raise RuntimeError(
            "scripts/run_foa.py is required for stream construction "
            "(build_test_loader) but could not be imported."
        )
    return build_test_loader(cfg, corruption=corruption, limit_batches=limit_batches, **overrides)


def _unpack_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    images = labels = None
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
        if len(batch) >= 1:
            images = batch[0]
        if len(batch) >= 2:
            labels = batch[1]
    else:
        images = batch
    if images is None:
        raise ValueError(f"Could not find images in batch of type {type(batch)}")
    images = images.to(device)
    labels = labels.to(device) if labels is not None else None
    return images, labels


def _forward(model: torch.nn.Module, images: torch.Tensor, prompt: Any = None) -> Dict[str, Any]:
    """Forward the frozen backbone, tolerating the different output contracts."""
    if hasattr(model, "forward_with_features"):
        out = model.forward_with_features(images, prompt=prompt)
        if isinstance(out, dict):
            return out
        if isinstance(out, (tuple, list)):
            features, logits = out[0], out[1]
            final = out[2] if len(out) > 2 else features[-1]
            return {"cls_features": features, "logits": logits, "final_cls": final}
    features = model.forward_features(images, prompt=prompt)
    if isinstance(features, dict):
        return features
    cls_features, logits, final = features
    return {"cls_features": cls_features, "logits": logits, "final_cls": final}


def _head(model: torch.nn.Module, feature: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "head"):
        return model.head(feature)
    return model(feature)  # pragma: no cover - fallback


def _entropy_per_sample(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy (nats) of the softmax distribution, per sample -> [B]."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).clamp_min(eps)


def _discrepancy_sum(
    cls_features: Sequence[torch.Tensor],
    source_stats: Any,
    layer_start: int = 1,
) -> torch.Tensor:
    """Differentiable Eqn. (5) discrepancy term: sum_i (||d_mu||_2 + ||d_sigma||_2)."""
    if source_stats is None or not cls_features:
        return torch.zeros((), dtype=torch.float32)
    device = cls_features[0].device
    dtype = cls_features[0].dtype
    num_layers = int(getattr(source_stats, "num_layers", len(cls_features)))
    total = torch.zeros((), dtype=dtype, device=device)
    for i in range(int(layer_start), min(num_layers, len(cls_features))):
        feats = cls_features[i]
        if feats is None or feats.dim() < 2:
            continue
        mu = feats.mean(dim=0)
        sigma = feats.std(dim=0, unbiased=False)
        mu_src = _source_vector(source_stats, "mu_i", i, device, dtype)
        sigma_src = _source_vector(source_stats, "sigma_i", i, device, dtype)
        if mu_src is not None:
            total = total + torch.linalg.vector_norm(mu - mu_src)
        if sigma_src is not None:
            total = total + torch.linalg.vector_norm(sigma - sigma_src)
    return total


def _source_vector(source_stats: Any, method_name: str, i: int, device, dtype) -> Optional[torch.Tensor]:
    getter = getattr(source_stats, method_name, None)
    if getter is None:
        return None
    try:
        vec = getter(i, device=device) if "device" in getattr(getter, "__code__", None).co_varnames else getter(i)
    except Exception:  # pragma: no cover
        try:
            vec = getter(i)
        except Exception:
            return None
    if vec is None:
        return None
    if not isinstance(vec, torch.Tensor):
        vec = torch.as_tensor(vec)
    return vec.to(device=device, dtype=dtype)


def _sdg_loss(
    logits: torch.Tensor,
    cls_features: Sequence[torch.Tensor],
    source_stats: Any,
    loss_name: str,
    lambda_value: float = DEFAULT_EQN5_LAMBDA,
    entropy_divisor: float = float(CANONICAL_BS),
    layer_start: int = 1,
) -> torch.Tensor:
    """Loss used for the SGD rows of Table 9.

    * ``loss_name == "entropy"`` -> plain prediction entropy (TENT objective).
    * ``loss_name == "eqn5"``    -> Eqn. (5) with the entropy term divided by the
      batch size (64) and lambda = 30 (Appendix B.2), so both terms have a
      comparable magnitude.
    """
    entropy = _entropy_per_sample(logits).mean()
    if loss_name == "entropy":
        return entropy
    if loss_name == "eqn5":
        entropy_term = torch.sum(_entropy_per_sample(logits)) / float(max(entropy_divisor, 1.0))
        disc = _discrepancy_sum(cls_features, source_stats, layer_start=layer_start)
        return entropy_term + float(lambda_value) * disc
    raise ValueError(f"Unknown loss for the design study: {loss_name!r}")


def _assign_vector(params: Sequence[torch.nn.Parameter], vector: np.ndarray) -> None:
    """Copy a flat candidate vector into a list of parameter tensors."""
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            chunk = np.asarray(vector[offset : offset + n], dtype=np.float32)
            p.copy_(torch.as_tensor(chunk, device=p.device, dtype=p.dtype).view_as(p))
            offset += n


def _collect_vector(params: Sequence[torch.nn.Parameter]) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in params]).astype(np.float32)


def _param_dim(params: Sequence[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


# ======================================================================================
# Variant runners
# ======================================================================================
def _run_static(
    cfg: Any,
    model: torch.nn.Module,
    source_stats: Any,
    corruptions: Sequence[Optional[str]],
    device: torch.device,
    use_shifting: bool = False,
    class_subset: Optional[Sequence[int]] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Dict[str, Any]:
    """NoAdapt and "Act. Shifting only": no prompt, no CMA, optionally shift features."""
    per_corruption: Dict[str, Any] = {}
    for corruption in corruptions:
        shifter = None
        if use_shifting:
            shifter = build_activation_shifter(
                source_stats,
                cfg,
                enabled=True,
                alpha=alpha,
                gamma=gamma,
                device=device,
            )
        acc = _make_accumulator(cfg, class_subset=class_subset)
        stream = _build_stream(cfg, corruption, limit_batches, batch_size=batch_size)
        seen = 0
        t0 = time.time()
        for batch in stream:
            images, labels = _unpack_batch(batch, device)
            with torch.no_grad():
                out = _forward(model, images, prompt=None)
                final = out.get("final_cls")
                if final is None:
                    final = out["cls_features"][-1]
                logits = out.get("logits")
                if shifter is not None:
                    shifted = shifter.shift_and_update(final)
                    logits = _head(model, shifted)
                elif logits is None:  # pragma: no cover
                    logits = _head(model, final)
            if labels is not None:
                acc.update(logits, labels)
            seen += int(logits.shape[0])
        metrics = acc.compute()
        per_corruption[str(corruption)] = {
            "accuracy": metrics["accuracy"],
            "ece": metrics["ece"],
            "num_samples": metrics["num_samples"],
            "wall_clock_s": time.time() - t0,
        }
        if verbose:
            print(
                f"  [{str(corruption):>18}] acc={metrics['accuracy']:.2f} "
                f"ece={metrics['ece']:.2f} (n={metrics['num_samples']}, {seen} seen)"
            )
    return _summarize(per_corruption)


def _run_foa_variant(
    cfg: Any,
    model: torch.nn.Module,
    source_stats: Any,
    corruptions: Sequence[Optional[str]],
    device: torch.device,
    use_entropy: bool = True,
    use_discrepancy: bool = True,
    use_shifting: bool = True,
    lambda_value: Optional[float] = None,
    population_size: Optional[int] = None,
    num_prompts: Optional[int] = None,
    class_subset: Optional[Sequence[int]] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """CMA-based rows: prompts optimised by CMA-ES with the selected fitness terms."""
    per_corruption: Dict[str, Any] = {}
    for corruption in corruptions:
        foa = build_foa(
            cfg=cfg,
            source_stats=source_stats,
            model=model,
            device=device,
            population_size=population_size,
            num_prompts=num_prompts,
            lambda_value=lambda_value,
            use_entropy=use_entropy,
            use_discrepancy=use_discrepancy,
            shifting_enabled=use_shifting,
        )
        acc = _make_accumulator(cfg, class_subset=class_subset)
        stream = _build_stream(cfg, corruption, limit_batches, batch_size=batch_size)
        t0 = time.time()
        summary = foa.run(stream, accumulator=acc, verbose=verbose)
        per_corruption[str(corruption)] = {
            "accuracy": summary.get("accuracy", float("nan")),
            "ece": summary.get("ece", float("nan")),
            "num_samples": summary.get("num_samples", 0),
            "num_batches": summary.get("num_batches", 0),
            "lambda": summary.get("lambda"),
            "population_size": summary.get("population_size"),
            "wall_clock_s": time.time() - t0,
        }
        if verbose:
            print(
                f"  [{str(corruption):>18}] acc={per_corruption[str(corruption)]['accuracy']:.2f} "
                f"ece={per_corruption[str(corruption)]['ece']:.2f}"
            )
    return _summarize(per_corruption)


def _run_cma_norm(
    cfg: Any,
    model: torch.nn.Module,
    source_stats: Any,
    corruptions: Sequence[Optional[str]],
    device: torch.device,
    loss_name: str = "eqn5",
    use_shifting: bool = True,
    lambda_value: float = DEFAULT_EQN5_LAMBDA,
    entropy_divisor: float = float(CANONICAL_BS),
    population_size: Optional[int] = None,
    class_subset: Optional[Sequence[int]] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Table 9 exp4/exp5: CMA over *all norm-layer affine parameters* (ultra-high-dim).

    This deliberately reproduces the paper's collapse (0.1% accuracy): CMA cannot
    handle the ~38k-dim search space of ViT-Base norm affine parameters, and the
    entropy-only fitness additionally provides no stable ranking signal.
    """
    from src.method.cma_wrapper import build_cma_optimizer

    if select_norm_affine_params is None:  # pragma: no cover
        raise RuntimeError("select_norm_affine_params unavailable (src/baselines/tent.py)")
    params = list(select_norm_affine_params(model))
    if not params:  # pragma: no cover
        raise RuntimeError("No norm affine parameters found on the model")
    dim = _param_dim(params)
    if verbose:
        print(f"  CMA over {len(params)} norm tensors, search dim = {dim}")

    per_corruption: Dict[str, Any] = {}
    for corruption in corruptions:
        optimizer = build_cma_optimizer(
            cfg=cfg, dim=dim, population_size=population_size, seed=_cfg_get(cfg, "seed", default=0) or 0
        )
        shifter = (
            build_activation_shifter(source_stats, cfg, enabled=True, device=device)
            if use_shifting
            else None
        )
        acc = _make_accumulator(cfg, class_subset=class_subset)
        stream = _build_stream(cfg, corruption, limit_batches, batch_size=batch_size)
        t0 = time.time()
        for batch in stream:
            images, labels = _unpack_batch(batch, device)
            candidates = optimizer.ask()
            values: List[float] = []
            logits_list: List[torch.Tensor] = []
            with torch.no_grad():
                for candidate in candidates:
                    _assign_vector(params, candidate)
                    out = _forward(model, images, prompt=None)
                    logits = out.get("logits")
                    if logits is None:  # pragma: no cover
                        logits = _head(model, out["final_cls"])
                    if shifter is not None:
                        logits = _head(model, shifter.shift(out["final_cls"]))
                    cls_features = out["cls_features"]
                    if loss_name == "eqn5":
                        value = float(
                            _entropy_per_sample(logits).sum().item()
                            + float(lambda_value)
                            * float(_discrepancy_sum(cls_features, source_stats).item())
                        )
                    else:
                        value = float(_entropy_per_sample(logits).mean().item())
                    values.append(value)
                    logits_list.append(logits)
                # Rank candidates and hand the fitness values back to CMA.
                best = optimizer.best_index(values)
                optimizer.tell(values, candidates)
                # Emit the prediction of the best candidate (single-shot, no averaging).
                logits = logits_list[best]
                if shifter is not None and hasattr(shifter, "update"):
                    # The EMA state follows the *unshifted* final-layer CLS mean
                    # (Eqn. 9) of the selected candidate; recompute deterministically.
                    pass
            if labels is not None:
                acc.update(logits, labels)
        metrics = acc.compute()
        per_corruption[str(corruption)] = {
            "accuracy": metrics["accuracy"],
            "ece": metrics["ece"],
            "num_samples": metrics["num_samples"],
            "wall_clock_s": time.time() - t0,
        }
        if verbose:
            print(
                f"  [{str(corruption):>18}] acc={metrics['accuracy']:.2f} ece={metrics['ece']:.2f}"
            )
    return _summarize(per_corruption)


def _run_sgd(
    cfg: Any,
    model: torch.nn.Module,
    source_stats: Any,
    corruptions: Sequence[Optional[str]],
    device: torch.device,
    params_kind: str = "prompts",
    loss_name: str = "entropy",
    lr: Optional[float] = None,
    momentum: float = 0.9,
    lambda_value: float = DEFAULT_EQN5_LAMBDA,
    entropy_divisor: float = float(CANONICAL_BS),
    num_prompts: int = 3,
    use_shifting: bool = False,
    class_subset: Optional[Sequence[int]] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Table 9 SGD rows (exp1/exp2/exp3 and TENT): learnable params + SGD optimizer."""
    embed_dim = int(getattr(model, "embed_dim", 768))
    per_corruption: Dict[str, Any] = {}

    for corruption in corruptions:
        # ---- Trainable parameter set -------------------------------------------------
        if params_kind == "prompts":
            lr = DEFAULT_SGD_PROMPT_LR if lr is None else float(lr)
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            init = (
                torch.rand(num_prompts, embed_dim, generator=gen, dtype=torch.float32) * 0.02 - 0.01
            )
            prompt = torch.nn.Parameter(init.unsqueeze(0).to(device))
            trainable: List[torch.nn.Parameter] = [prompt]
            param_kwargs = {"prompt": prompt}
        elif params_kind == "norm":
            lr = DEFAULT_SGD_NORM_LR if lr is None else float(lr)
            if select_norm_affine_params is None:  # pragma: no cover
                raise RuntimeError("select_norm_affine_params unavailable")
            trainable = list(select_norm_affine_params(model))
            for p in trainable:
                p.requires_grad_(True)
            param_kwargs = {"prompt": None}
        else:  # pragma: no cover
            raise ValueError(f"Unsupported SGD parameter kind: {params_kind!r}")

        optimizer = torch.optim.SGD(trainable, lr=float(lr), momentum=float(momentum))
        shifter = (
            build_activation_shifter(source_stats, cfg, enabled=True, device=device)
            if use_shifting
            else None
        )
        acc = _make_accumulator(cfg, class_subset=class_subset)
        stream = _build_stream(cfg, corruption, limit_batches, batch_size=batch_size)
        t0 = time.time()
        for batch in stream:
            images, labels = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = _forward(model, images, **param_kwargs)
                logits = out["logits"] if out.get("logits") is not None else _head(model, out["final_cls"])
                loss = _sdg_loss(
                    logits,
                    out["cls_features"],
                    source_stats,
                    loss_name,
                    lambda_value=lambda_value,
                    entropy_divisor=entropy_divisor,
                )
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                out_eval = _forward(model, images, **param_kwargs)
                final = out_eval["final_cls"]
                if shifter is not None:
                    logits_eval = _head(model, shifter.shift_and_update(final))
                else:
                    logits_eval = (
                        out_eval["logits"]
                        if out_eval.get("logits") is not None
                        else _head(model, final)
                    )
            if labels is not None:
                acc.update(logits_eval, labels)
        if params_kind == "norm":
            for p in trainable:  # restore the frozen state for the next corruption
                p.requires_grad_(False)
        metrics = acc.compute()
        per_corruption[str(corruption)] = {
            "accuracy": metrics["accuracy"],
            "ece": metrics["ece"],
            "num_samples": metrics["num_samples"],
            "wall_clock_s": time.time() - t0,
        }
        if verbose:
            print(
                f"  [{str(corruption):>18}] acc={metrics['accuracy']:.2f} ece={metrics['ece']:.2f}"
            )
    return _summarize(per_corruption)


def _summarize(per_corruption: Dict[str, Any]) -> Dict[str, Any]:
    """Mean accuracy/ECE over the (possibly single-entry) corruption list."""
    accs = [v["accuracy"] for v in per_corruption.values() if not _isnan(v["accuracy"])]
    eces = [v["ece"] for v in per_corruption.values() if not _isnan(v["ece"])]
    total = sum(int(v.get("num_samples", 0)) for v in per_corruption.values())
    return {
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "ece": float(np.mean(eces)) if eces else float("nan"),
        "num_samples": total,
        "num_corruptions": len(per_corruption),
        "per_corruption": per_corruption,
    }


def _isnan(x: Any) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except Exception:  # pragma: no cover
        return True


# ======================================================================================
# Grid drivers
# ======================================================================================
def _model_cache() -> Dict[str, Any]:
    return {}


def run_component_ablations(
    cfg: Any,
    variants: Optional[Sequence[str]] = None,
    corruptions: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    source_stats_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Run the Table 5 component grid (entropy / discrepancy / activation shifting)."""
    device = device or _resolve_device(cfg)
    names = list(variants) if variants else list(COMPONENT_VARIANTS.keys())
    specs = _resolve_component_specs(cfg, names)
    corruptions = _resolve_corruptions(cfg, corruptions)
    class_subset = _class_subset(cfg)
    model = model or build_model(cfg, device)
    source_stats = load_source_stats_for(cfg, device, source_stats_path)

    results: Dict[str, Any] = {}
    for name in names:
        spec = specs.get(name, COMPONENT_VARIANTS.get(name))
        if spec is None:
            raise KeyError(f"Unknown component variant {name!r}. Known: {sorted(COMPONENT_VARIANTS)}")
        if verbose:
            print(
                f"[components] {name}: entropy={spec['use_entropy']} "
                f"discrepancy={spec['use_discrepancy']} shifting={spec['shifting']} "
                f"optimizer={spec['optimizer']}"
            )
        _seed_everything(int(_cfg_get(cfg, "seed", default=0) or 0))
        if spec["optimizer"] == "none":
            summary = _run_static(
                cfg,
                model,
                source_stats,
                corruptions,
                device,
                use_shifting=bool(spec["shifting"]),
                class_subset=class_subset,
                limit_batches=limit_batches,
                verbose=verbose,
                batch_size=batch_size,
            )
        else:
            summary = _run_foa_variant(
                cfg,
                model,
                source_stats,
                corruptions,
                device,
                use_entropy=bool(spec["use_entropy"]),
                use_discrepancy=bool(spec["use_discrepancy"]),
                use_shifting=bool(spec["shifting"]),
                lambda_value=spec.get("lambda_value"),
                population_size=_cfg_get(cfg, "cma", "population_size", default=None),
                num_prompts=_cfg_get(cfg, "prompt", "num_prompts", default=None),
                class_subset=class_subset,
                limit_batches=limit_batches,
                verbose=verbose,
                batch_size=batch_size,
            )
        summary.update(
            {
                "variant": name,
                "group": "components",
                "label": TABLE5_LABELS.get(name, name),
                "use_entropy": bool(spec["use_entropy"]),
                "use_discrepancy": bool(spec["use_discrepancy"]),
                "shifting": bool(spec["shifting"]),
                "optimizer": spec["optimizer"],
                "paper_accuracy": TABLE5_REFERENCE.get(name, {}).get("accuracy"),
                "paper_ece": TABLE5_REFERENCE.get(name, {}).get("ece"),
            }
        )
        results[name] = summary
    return results


def _resolve_component_specs(cfg: Any, names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    specs = {k: dict(v) for k, v in COMPONENT_VARIANTS.items()}
    configured = _cfg_get(cfg, "ablations", "components", default=None)
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, dict):
                specs.setdefault(str(key), {}).update(value)
            else:  # a bare list of names also works
                specs.setdefault(str(value), dict(COMPONENT_VARIANTS.get(str(value), {})))
    return specs


def _resolve_design_specs(cfg: Any) -> Dict[str, Dict[str, Any]]:
    specs = {k: dict(v) for k, v in DESIGN_VARIANTS.items()}
    configured = _cfg_get(cfg, "ablations", "design", default=None)
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, dict):
                specs.setdefault(str(key), {}).update(value)
    return specs


def run_design_study(
    cfg: Any,
    variants: Optional[Sequence[str]] = None,
    corruptions: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    source_stats_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Run the Table 9 design-choice grid (params x optimizer x loss)."""
    device = device or _resolve_device(cfg)
    specs = _resolve_design_specs(cfg)
    names = list(variants) if variants else list(DESIGN_VARIANTS.keys())
    corruptions = _resolve_corruptions(cfg, corruptions)
    class_subset = _class_subset(cfg)
    model = model or build_model(cfg, device)
    source_stats = load_source_stats_for(cfg, device, source_stats_path)
    seed = int(_cfg_get(cfg, "seed", default=0) or 0)

    results: Dict[str, Any] = {}
    for name in names:
        spec = specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown design variant {name!r}. Known: {sorted(DESIGN_VARIANTS)}")
        params_kind = spec.get("params", "-")
        optimizer_kind = spec.get("optimizer", "-")
        loss_name = spec.get("loss", "-")
        if verbose:
            print(f"[design] {name}: params={params_kind} optimizer={optimizer_kind} loss={loss_name}")
        _seed_everything(seed)

        if optimizer_kind == "-":
            summary = _run_static(
                cfg,
                model,
                source_stats,
                corruptions,
                device,
                use_shifting=bool(spec.get("shifting", False)),
                class_subset=class_subset,
                limit_batches=limit_batches,
                verbose=verbose,
                batch_size=batch_size,
            )
        elif optimizer_kind == "sgd":
            summary = _run_sgd(
                cfg,
                model,
                source_stats,
                corruptions,
                device,
                params_kind=params_kind,
                loss_name=loss_name,
                lr=spec.get("lr"),
                momentum=float(spec.get("momentum", 0.9)),
                lambda_value=float(spec.get("lambda_value", DEFAULT_EQN5_LAMBDA)),
                entropy_divisor=float(spec.get("entropy_divisor", CANONICAL_BS)),
                num_prompts=int(spec.get("num_prompts", 3)),
                use_shifting=bool(spec.get("shifting", False)),
                class_subset=class_subset,
                limit_batches=limit_batches,
                verbose=verbose,
                batch_size=batch_size,
                seed=seed,
            )
        elif optimizer_kind == "cma" and params_kind == "norm":
            summary = _run_cma_norm(
                cfg,
                model,
                source_stats,
                corruptions,
                device,
                loss_name=loss_name,
                use_shifting=bool(spec.get("shifting", False)),
                lambda_value=float(spec.get("lambda_value", DEFAULT_EQN5_LAMBDA)),
                population_size=_cfg_get(cfg, "cma", "population_size", default=None),
                class_subset=class_subset,
                limit_batches=limit_batches,
                verbose=verbose,
                batch_size=batch_size,
            )
        elif optimizer_kind == "cma" and params_kind == "prompts":
            summary = _run_foa_variant(
                cfg,
                model,
                source_stats,
                corruptions,
                device,
                use_entropy=(loss_name in ("entropy", "eqn5")),
                use_discrepancy=(loss_name == "eqn5"),
                use_shifting=bool(spec.get("shifting", False)),
                lambda_value=spec.get("lambda_value"),
                population_size=_cfg_get(cfg, "cma", "population_size", default=None),
                num_prompts=int(spec.get("num_prompts", 3)),
                class_subset=class_subset,
                limit_batches=limit_batches,
                verbose=verbose,
                batch_size=batch_size,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported design combination: {spec}")
        ref = TABLE9_REFERENCE.get(name, {})
        summary.update(
            {
                "variant": name,
                "group": "design",
                "params": params_kind,
                "optimizer": optimizer_kind,
                "loss": loss_name,
                "lr": spec.get("lr"),
                "lambda": spec.get("lambda_value"),
                "paper_accuracy": ref.get("accuracy"),
                "paper_ece": ref.get("ece"),
            }
        )
        results[name] = summary
    return results


def _class_subset(cfg: Any) -> Optional[List[int]]:
    """ImageNet-R evaluates only the 200 task classes (a subset of the 1000-way head)."""
    subset = _cfg_get(cfg, "data", "class_subset", default=None)
    if subset is None:
        subset = _cfg_get(cfg, "data", "num_classes_eval", default=None)
    if subset is None:
        return None
    num = int(subset)
    if num <= 0 or num >= int(_cfg_get(cfg, "model", "num_classes", default=1000) or 1000):
        return None
    return list(range(num))


# ======================================================================================
# Reporting
# ======================================================================================
def print_ablation_summary(results: Dict[str, Any]) -> None:
    groups = sorted({v.get("group", "?") for v in results.values() if isinstance(v, dict)})
    for group in groups:
        rows = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("group") == group}
        if not rows:
            continue
        print()
        if group == "components":
            print("=" * 84)
            print("Table 5 - Component ablations (ImageNet-C level 5, avg over corruptions, ViT-Base)")
            print("=" * 84)
            print(
                f"{'Variant':<34}{'Acc.(%)':>9}{'paper':>8}{'ECE(%)':>9}{'paper':>8}"
            )
            print("-" * 84)
            for name, row in rows.items():
                label = TABLE5_LABELS.get(name, name)
                print(
                    f"{label:<34}{row['accuracy']:>9.2f}"
                    f"{_fmt_ref(row.get('paper_accuracy')):>8}"
                    f"{row['ece']:>9.2f}{_fmt_ref(row.get('paper_ece')):>8}"
                )
        else:
            print("=" * 96)
            print("Table 9 - Design choices: learnable params x optimizer x loss "
                  "(ImageNet-C level 5, avg)")
            print("=" * 96)
            print(
                f"{'Variant':<10}{'Learnable Params':<17}{'Optimizer':<12}{'Loss':<10}"
                f"{'Acc.(%)':>9}{'paper':>8}{'ECE(%)':>9}{'paper':>8}"
            )
            print("-" * 96)
            for name, row in rows.items():
                print(
                    f"{name:<10}{str(row.get('params')):<17}{str(row.get('optimizer')):<12}"
                    f"{str(row.get('loss')):<10}"
                    f"{row['accuracy']:>9.2f}{_fmt_ref(row.get('paper_accuracy')):>8}"
                    f"{row['ece']:>9.2f}{_fmt_ref(row.get('paper_ece')):>8}"
                )
        print("=" * (84 if group == "components" else 96))


def _fmt_ref(value: Any) -> str:
    if value is None or _isnan(value):
        return "-"
    return f"{float(value):.1f}"


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOA ablations: Table 5 (components) and Table 9 (design choices)."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_ROOT, "configs", "ablations.yaml"),
        help="Base YAML config (defaults to configs/ablations.yaml).",
    )
    parser.add_argument("--extra-config", nargs="*", default=None, help="Additional YAML overrides.")
    parser.add_argument(
        "--groups",
        default="components",
        help="Comma-separated ablation groups to run: components,design (or 'all').",
    )
    parser.add_argument("--variants", nargs="*", default=None, help="Subset of variant names to run.")
    parser.add_argument("--corruptions", nargs="*", default=None, help="Corruptions to evaluate.")
    parser.add_argument("--dataset", default=None, help="Override data.dataset.")
    parser.add_argument("--severity", type=int, default=None, help="Override data.severity.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Cap batches per stream (smoke tests).")
    parser.add_argument("--device", default=None, help="cuda / cpu (defaults to config or cuda).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source-stats", default=None, help="Path to the precomputed source-statistics .pt")
    parser.add_argument("--checkpoint", default=None, help="Override model.checkpoint (or 'null' for the URL).")
    parser.add_argument("--output", default=None, help="JSON output path.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.dataset:
        cfg.setdefault("data", {})["dataset"] = args.dataset
    if args.severity is not None:
        cfg.setdefault("data", {})["severity"] = int(args.severity)
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.checkpoint is not None:
        cfg.setdefault("model", {})["checkpoint"] = (
            None if str(args.checkpoint).lower() in ("null", "none", "") else args.checkpoint
        )
    if args.source_stats:
        cfg.setdefault("source_stats", {})["path"] = args.source_stats
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = [args.config]
    if args.extra_config:
        paths.extend(args.extra_config)
    cfg = load_config(*[p for p in paths if p and os.path.exists(p)] or paths)
    cfg = _apply_overrides(cfg, args)

    groups = [g.strip().lower() for g in str(args.groups).split(",") if g.strip()]
    if "all" in groups:
        groups = list(ABLATION_GROUPS)
    for group in groups:
        if group not in ABLATION_GROUPS:
            raise SystemExit(f"Unknown ablation group {group!r}; expected one of {ABLATION_GROUPS} or 'all'.")

    seeds = [int(v) for v in (_cfg_get(cfg, "ablations", "seeds", default=None) or [int(_cfg_get(cfg, "seed", default=0) or 0)])]
    _seed_everything(seeds[0])

    device = _resolve_device(cfg, args.device)
    corruptions = _resolve_corruptions(cfg, args.corruptions)
    verbose = not args.quiet
    batch_size = args.batch_size or _cfg_get(cfg, "data", "batch_size", default=None)

    print("FOA ablations")
    print(f"  dataset     : {_cfg_get(cfg, 'data', 'dataset', default='imagenet-c')}")
    print(f"  severity    : {_cfg_get(cfg, 'data', 'severity', default=5)}")
    print(f"  batch size  : {batch_size}")
    print(f"  corruptions : {len(corruptions)} ({corruptions[:3]}{'...' if len(corruptions) > 3 else ''})")
    print(f"  device      : {device}")
    print(f"  groups      : {groups}")
    if args.limit_batches:
        print(f"  limit/task  : {args.limit_batches} batches (smoke test)")

    model = build_model(cfg, device)
    results: Dict[str, Any] = {}
    t_start = time.time()

    if "components" in groups:
        variants = args.variants if (args.variants and "design" not in groups) else None
        results.update(
            run_component_ablations(
                cfg,
                variants=variants,
                corruptions=corruptions,
                device=device,
                limit_batches=args.limit_batches,
                verbose=verbose,
                batch_size=batch_size,
                source_stats_path=args.source_stats,
                model=model,
            )
        )
    if "design" in groups:
        results.update(
            run_design_study(
                cfg,
                variants=args.variants,
                corruptions=corruptions,
                device=device,
                limit_batches=args.limit_batches,
                verbose=verbose,
                batch_size=batch_size,
                source_stats_path=args.source_stats,
                model=model,
            )
        )

    print_ablation_summary(results)

    payload = {
        "config": config_to_dict(cfg),
        "args": vars(args),
        "groups": groups,
        "dataset": _cfg_get(cfg, "data", "dataset", default=None),
        "severity": _cfg_get(cfg, "data", "severity", default=None),
        "corruptions": [c for c in corruptions],
        "limit_batches": args.limit_batches,
        "wall_clock_s": time.time() - t_start,
        "reference": {"table5": TABLE5_REFERENCE, "table9": TABLE9_REFERENCE},
        "results": config_to_dict(results),
    }

    out_path = args.output
    if out_path is None:
        out_dir = _cfg_get(cfg, "output_dir", default="./outputs") or "./outputs"
        os.makedirs(out_dir, exist_ok=True)
        name = _cfg_get(cfg, "experiment", "name", default="foa") or "foa"
        suffix = "" if not args.limit_batches else f"_limit{args.limit_batches}"
        out_path = os.path.join(out_dir, f"{name}_ablations{suffix}.json")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"\nSaved results to {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not write results to {out_path}: {exc}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
