"""Robust CLIP: unsupervised adversarial fine-tuning of the CLIP vision encoder.

This module implements the *training* entry point of the Robust CLIP benchmark.
The reproduction plan notes that the training loop details are **not** part of the
Addendum; they are taken from the paper body (Sec. 3.2 "FARE", App. B.1, B.3, B.4),
which is quoted inline below so that no hyper-parameter is silently invented.

Paper body facts implemented here (verbatim excerpts in ``PAPER_BODY_FACTS``)
---------------------------------------------------------------------------
* Training loss (FARE, Eq. (2) of App. C.4 / Eq. (3) of the main paper)::

      L_adv(x) = max_{z: ||z - x||_inf <= eps} || phi_FT(z) - phi_Org(x) ||_2^2

  i.e. the *squared* l2-norm between the *fine-tuned* embedding of an adversarial
  image and the *original* (frozen CLIP) embedding of the clean image.  App. B.4
  confirms the squared l2-norm is the main-paper choice (l1 shown as ablation).
* "The FARE-loss (Eq. 3) is thus computed with respect to the class token only."
  (App. B.1) -> class-token-only training loss.
* Adversarial training setup (App. B.1): "All robust models in the main paper ...
  are trained on ImageNet (at resolution 224x224) for two epochs using 10 steps of
  PGD at l_inf radius of 4/255 respectively 2/255 with the step size set to 1/255.
  AdamW ... momenta coefficients beta_1 and beta_2 set to 0.9 and 0.95
  respectively. The training was done with a cosine decaying learning rate (LR)
  schedule with a linear warmup to the peak LR (attained at 7% of total training
  steps) of 1e-5, weight decay (WD) of 1e-4 and an effective batch size of 128."
* Hyper-parameter choice (App. B.3): "we select LR = 1e-5 and WD = 1e-4".
* Backbone (App. B.3): "All vision encoders in CLIP in the main section of the
  paper use ViT-L/14 as architectures."  ImageNet resolution 224x224 means the
  OpenAI ``ViT-L-14`` @224 checkpoint (matching the Addendum's model choice).

Everything the paper/Addendum does not state (dataloader worker count, AMP usage,
exact AdamW epsilon, gradient clipping, checkpoint cadence, LR floor, how a small
per-device batch is accumulated to the effective batch of 128, ...) is exposed via
:class:`RobustCLIPTrainConfig` and tagged ``EXTERNAL_DEFAULT`` / marked as
``UNSPECIFIED_BY_ADDENDUM`` -- never presented as a paper value.

Usage
-----
    python -m robust_clip_repro.train_robust_clip --eps 4/255 --smoke-test
    python -m robust_clip_repro.train_robust_clip --eps 2/255 \
        --batch-size 32 --grad-accum 4 --output-dir results/robust_clip_eps2
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .models.clip_vision_encoder import (
    CLIP_MEAN,
    CLIP_STD,
    CLIPVisionConfig,
    RandomVisionEncoder,
    build_clip_vision_encoder,
)
from .utils.normalization import (
    PIXEL_MAX,
    PIXEL_MIN,
    adversarial_pixels,
    clamp_pixels,
    get_normalization,
)
from .utils.precision import (
    INT16,
    INT32,
    QUANT_SCALE,
    assert_mandated_dtype,
    decode_perturbation,
    encode_perturbation,
    float_dtype_for_precision,
    int_dtype_for_precision,
)
from .utils.logging import (
    EXTERNAL_DEFAULT,
    Timer,
    UNSPECIFIED,
    get_logger,
    log_config,
    log_metric,
    log_table,
    provenance_metadata,
    save_json,
    setup_logging,
)

try:  # torch is required for training but we keep module import cheap/friendly.
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


LOGGER = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Provenance bookkeeping
# --------------------------------------------------------------------------------------

#: Values stated verbatim in the paper body (see module docstring for quotes).
PAPER_BODY_FACTS: Dict[str, Any] = {
    # Sec. 3.2 / Eq. (2)+(3), App. C.4
    "objective": "FARE: squared l2 between fine-tuned adversarial embedding and "
                 "original clean embedding (Eq. (2), App. C.4)",
    "loss_norm": "l2_squared (App. B.4 ablation shows l1 is equivalent but not used)",
    "feature_space": "class token only (App. B.1)",
    # App. B.1
    "dataset": "ImageNet",
    "resolution": 224,
    "epochs": 2,
    "adv_steps": 10,
    "adv_step_size": "1/255",
    "adv_radii": ("2/255", "4/255"),
    "optimizer": "AdamW",
    "betas": (0.9, 0.95),
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "effective_batch_size": 128,
    "lr_schedule": "cosine decay with linear warmup to peak at 7% of total steps",
    "warmup_fraction": 0.07,
    # App. B.3
    "backbone": "ViT-L/14",
}

#: Values the Addendum (or paper) leaves unspecified; supplied externally.
ADDENDUM_FACTS: Dict[str, Any] = {
    "precision_policy": "half-precision -> int16 storage, single-precision -> int32 "
                        "storage for adversarial perturbations",
    "imagenet_loading": "HuggingFace datasets load_dataset('imagenet-1k', "
                        "trust_remote_code=True)",
    "attack_budget_space": "l_inf ball around NON-normalized pixels",
}

EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "per_device_batch_size": "chosen to fit memory; grad accumulation restores the "
                             "paper's effective batch size of 128",
    "grad_accum_steps": "derived so batch_size * grad_accum == effective_batch_size",
    "num_workers": 0,
    "amp": False,
    "optimizer_eps": 1e-8,
    "min_lr_ratio": 0.0,
    "grad_clip_norm": None,
    "save_every": 1,
    "log_every": 50,
    "seed": 0,
    "dataset_split": "train",
    "num_samples": None,
    "clean_loss_weight": 0.0,
    "train_pgd_momentum": 0.0,
    "warmup_fraction_source": "paper body (7%); exposed for completeness",
}


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class RobustCLIPTrainConfig:
    """Configuration of the unsupervised adversarial fine-tuning run.

    Paper-body fields carry the paper's values as defaults; `EXTERNAL_DEFAULTS`
    fields are marked and logged as externally supplied.
    """

    # ---- model (paper body: ViT-L/14 @ 224) ----
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    pretrained_path: Optional[str] = None
    teacher_pretrained_path: Optional[str] = None
    resolution: int = 224
    trainable_scope: str = "all"
    num_trainable_blocks: int = 8
    use_grad_checkpointing: bool = False

    # ---- objective (paper body: FARE, class token, squared l2) ----
    objective: str = "fare"  # {"fare", "tecoa"}
    loss_norm: str = "l2_squared"  # {"l2_squared", "l1"}
    feature_space: str = "class_token"
    clean_loss_weight: float = 0.0  # external (paper FARE loss has only the adv term)

    # ---- data (paper body: ImageNet @ 224) ----
    dataset_name: str = "ImageNet"
    dataset_id: str = "imagenet-1k"
    split: str = "train"
    num_samples: Optional[int] = None
    shuffle: bool = True
    resolution_data: int = 224
    trust_remote_code: bool = True

    # ---- optimization (paper body values) ----
    epochs: int = 2
    effective_batch_size: int = 128
    batch_size: int = 32
    grad_accum_steps: Optional[int] = None
    lr: float = 1e-5
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    warmup_fraction: float = 0.07
    min_lr_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None

    # ---- adversarial training attack (paper body: 10 PGD steps, size 1/255) ----
    eps: Optional[float] = None  # REQUIRED; paper uses 2/255 or 4/255
    alpha: float = 1.0 / 255.0
    adv_steps: int = 10
    train_pgd_momentum: float = 0.0
    random_start: bool = True

    # ---- precision policy (Addendum) ----
    precision: str = "single"  # {"single", "half"}
    quant_scale: float = QUANT_SCALE

    # ---- runtime (external defaults) ----
    amp: bool = False
    seed: int = 0
    num_workers: int = 0
    device: Optional[str] = None
    log_every: int = 50
    save_every: int = 1
    output_dir: str = "results"
    output_file: Optional[str] = None
    resume_from: Optional[str] = None
    max_steps: Optional[int] = None  # external; useful for smoke tests
    smoke_test: bool = False
    verbose: bool = True

    provenance: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def __post_init__(self) -> None:
        self.betas = tuple(float(b) for b in self.betas)  # type: ignore[assignment]
        if self.eps is None:
            raise ValueError(
                "`eps` must be supplied externally: the paper body trains at "
                "l_inf radius 2/255 and 4/255 but this run needs an explicit value "
                "(e.g. --eps 4/255)."
            )
        eps = float(self.eps)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.precision not in ("single", "half", "fp32", "fp16", "float", "float16"):
            raise ValueError(f"unsupported precision {self.precision!r}")
        if self.objective not in ("fare", "tecoa"):
            raise ValueError(f"unsupported objective {self.objective!r}")
        if self.loss_norm not in ("l2_squared", "l1"):
            raise ValueError(f"unsupported loss_norm {self.loss_norm!r}")
        if self.grad_accum_steps is None:
            accum = max(1, int(round(self.effective_batch_size / max(self.batch_size, 1))))
            self.grad_accum_steps = accum
        self._record_provenance()

    # ------------------------------------------------------------------ helpers
    @property
    def int_dtype(self):
        return int_dtype_for_precision(self.precision)

    @property
    def float_dtype(self):
        return float_dtype_for_precision(self.precision)

    @property
    def is_half(self) -> bool:
        return self.int_dtype == INT16

    def effective_batch(self) -> int:
        return int(self.batch_size) * int(self.grad_accum_steps or 1)

    def external_defaults(self) -> Dict[str, Any]:
        """Values supplied externally rather than stated by the paper."""
        keys = (
            "clean_loss_weight", "train_pgd_momentum", "amp", "num_workers", "seed",
            "optimizer_eps", "min_lr_ratio", "grad_clip_norm", "save_every",
            "log_every", "output_dir", "max_steps", "grad_accum_steps",
            "teacher_pretrained_path", "pretrained_path",
        )
        out = {k: getattr(self, k) for k in keys}
        out["per_device_batch_size"] = self.batch_size
        out["dataset_split"] = self.split
        out["num_samples"] = self.num_samples
        return out

    def _record_provenance(self) -> None:
        prov = dict(self.provenance or {})
        prov.setdefault("paper_body", sorted(PAPER_BODY_FACTS.keys()))
        prov.setdefault("addendum", sorted(ADDENDUM_FACTS.keys()))
        prov.setdefault("unspecified_by_addendum", sorted(EXTERNAL_DEFAULTS.keys()))
        prov.setdefault("external_defaults", self.external_defaults())
        prov.setdefault("notes", {
            "paper_body": "Values quoted from the paper body (App. B.1/B.3/B.4, Eq. 2).",
            "addendum": "Values mandated by the benchmark Addendum.",
            "unspecified_by_addendum": "Not stated by paper body nor Addendum; "
                                       "supplied externally and logged as such.",
        })
        self.provenance = prov

    def as_dict(self) -> Dict[str, Any]:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                out[key] = list(value)
            else:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "RobustCLIPTrainConfig":
        cfg = dict(cfg or {})
        # Accept nested sections produced by configs/*.yaml.
        for section in ("train_robust_clip", "training", "train", "attack", "model", "data"):
            sub = cfg.pop(section, None)
            if isinstance(sub, dict):
                merged = dict(sub)
                merged.update(cfg)
                cfg = merged
        cfg.pop("provenance", None) if not isinstance(cfg.get("provenance"), dict) else None
        allowed = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg.items() if k in allowed}
        kwargs.update({k: v for k, v in overrides.items() if k in allowed})
        if "betas" in kwargs and isinstance(kwargs["betas"], (list, tuple)):
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs)


def parse_eps(value: Any) -> Optional[float]:
    """Parse ``"4/255"``, ``0.0156``, ``None`` into a float epsilon."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "unspecified"):
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


# --------------------------------------------------------------------------------------
# Losses (paper body, Eq. (2))
# --------------------------------------------------------------------------------------


def _pairwise_distance(a, b, norm: str = "l2_squared", reduction: str = "mean"):
    """Distance between two embedding tensors.

    ``l2_squared`` is the main-paper FARE formulation (App. B.4); ``l1`` is the
    ablation variant.
    """
    if norm == "l2_squared":
        dist = ((a - b) ** 2).sum(dim=-1)
    elif norm == "l1":
        dist = (a - b).abs().sum(dim=-1)
    else:
        raise ValueError(f"unsupported norm {norm!r}")
    return _reduce(dist, reduction)


def _reduce(values, reduction: str = "mean"):
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"unsupported reduction {reduction!r}")


def fare_loss(
    ft_features_adv,
    org_features_clean,
    *,
    norm: str = "l2_squared",
    reduction: str = "mean",
):
    """FARE objective, paper Eq. (2)::

        L_adv(x) = max_{||z-x||_inf <= eps} || phi_FT(z) - phi_Org(x) ||_2^2

    The max is realised by the inner PGD loop (:func:`pgd_perturbation`); this
    function evaluates the outer objective given the adversarial image's
    fine-tuned embedding and the clean image's original embedding.
    """
    return _pairwise_distance(ft_features_adv, org_features_clean, norm=norm, reduction=reduction)


def tecoa_loss(ft_features_adv, ft_features_clean, *, norm: str = "l2_squared", reduction: str = "mean"):
    """TeCoA baseline: *maximize* the clean/adversarial feature distance.

    Provided for comparison (App. B.5 compares against the original TeCoA
    checkpoint); the main paper uses :func:`fare_loss`.
    """
    return -_pairwise_distance(ft_features_adv, ft_features_clean, norm=norm, reduction=reduction)


def clean_embedding_loss(ft_features_clean, org_features_clean, *, norm: str = "l2_squared", reduction: str = "mean"):
    """App. C.4, Eq. (1): ``L_clean(x) = || phi_FT(x) - phi_Org(x) ||_2^2``.

    Used as an (optional, externally weighted) regulariser during training and as
    a diagnostic metric.
    """
    return _pairwise_distance(ft_features_clean, org_features_clean, norm=norm, reduction=reduction)


def make_objective(name: str) -> Callable[..., Any]:
    if name == "fare":
        return fare_loss
    if name == "tecoa":
        return tecoa_loss
    raise ValueError(f"unsupported objective {name!r}")


# --------------------------------------------------------------------------------------
# Inner PGD (training attack)
# --------------------------------------------------------------------------------------


def pgd_perturbation(
    encoder,
    pixels,
    loss_fn,
    *,
    eps: float,
    alpha: float,
    steps: int,
    momentum: float = 0.0,
    random_start: bool = True,
    clamp: Tuple[float, float] = (PIXEL_MIN, PIXEL_MAX),
    generator=None,
    return_codes: bool = True,
    precision: str = "single",
    quant_scale: float = QUANT_SCALE,
) -> Dict[str, Any]:
    """Inner maximisation of the FARE objective (paper: 10 PGD steps @ size 1/255).

    The perturbation lives in the l_inf ball around the *raw, non-normalized*
    pixels ``pixels`` (Addendum invariant).  ``loss_fn(z)`` must return a scalar
    that we maximise; model normalization is expected to happen inside
    ``loss_fn``/``encoder``.

    Returns a dict with ``adversarial`` (float), ``codes`` (mandated integer dtype
    per the precision policy) and bookkeeping statistics.
    """
    _require_torch()
    if eps <= 0 or alpha <= 0:
        raise ValueError("eps and alpha must be positive")

    lo, hi = clamp
    x = pixels.detach()
    if random_start:
        noise = torch.empty_like(x).uniform_(-eps, eps, generator=generator)
        delta = noise.clamp(-eps, eps)
    else:
        delta = torch.zeros_like(x)

    # keep the perturbed image inside the valid pixel range while remaining in the ball
    delta = (delta + x).clamp(lo, hi) - x

    grad_momentum = torch.zeros_like(x)
    norms: List[float] = []
    for _step in range(int(steps)):
        delta = delta.detach().requires_grad_(True)
        loss = loss_fn(x + delta)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        # element-wise sign of the (optionally momentum-accumulated) gradient
        if momentum and momentum > 0:
            grad_momentum = momentum * grad_momentum + grad
            update = grad_momentum.sign()
        else:
            update = grad.sign()
        delta = delta.detach() + alpha * update
        delta = delta.clamp(-eps, eps)
        delta = (delta + x).clamp(lo, hi) - x
        norms.append(float(delta.abs().max().detach().cpu()))

    adv = (x + delta).detach().clamp(lo, hi)
    out: Dict[str, Any] = {
        "adversarial": adv,
        "delta": (adv - x).detach(),
        "linf": float((adv - x).abs().max().detach().cpu()) if adv.numel() else 0.0,
        "per_step_linf": norms,
        "steps": int(steps),
        "eps": float(eps),
        "alpha": float(alpha),
    }
    if return_codes:
        codes = encode_perturbation(adv - x, precision, quant_scale=quant_scale)
        out["codes"] = codes
        out["codes_dtype"] = codes.dtype
    return out


# --------------------------------------------------------------------------------------
# Optimization utilities
# --------------------------------------------------------------------------------------


def build_optimizer(parameters, cfg: RobustCLIPTrainConfig):
    """AdamW with the paper's betas (0.9, 0.95), LR 1e-5, WD 1e-4."""
    _require_torch()
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters -- check `trainable_scope`")
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.weight_decay,
    )


def linear_warmup_cosine_lambda(total_steps: int, warmup_fraction: float = 0.07, min_lr_ratio: float = 0.0):
    """Cosine decay with a linear warmup peaking at ``warmup_fraction`` of steps.

    Paper body (App. B.1): "cosine decaying learning rate (LR) schedule with a
    linear warmup to the peak LR (attained at 7% of total training steps)".
    ``min_lr_ratio`` is an external default (the paper does not state a floor).
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(warmup_fraction * total_steps)))
    min_lr_ratio = float(min_lr_ratio)

    def _fn(step: int) -> float:
        step = max(0, int(step))
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    _fn.warmup_steps = warmup_steps  # type: ignore[attr-defined]
    return _fn


def build_lr_scheduler(optimizer, total_steps: int, cfg: RobustCLIPTrainConfig):
    _require_torch()
    fn = linear_warmup_cosine_lambda(total_steps, cfg.warmup_fraction, cfg.min_lr_ratio)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn), fn


def set_seed(seed: int) -> None:
    """Deterministic seeding (external default; the paper states no seed)."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None):
    _require_torch()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "robust_clip_repro.train_robust_clip requires PyTorch; install it with "
            "`pip install torch torchvision`."
        )


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def resolve_training_samples(cfg: RobustCLIPTrainConfig, *, allow_synthetic: bool = True) -> List[Any]:
    """Load ImageNet training samples via HuggingFace ``datasets``.

    Uses ``load_dataset("imagenet-1k", trust_remote_code=True)`` (Addendum rule)
    through :mod:`robust_clip_repro.data.imagenet`.
    """
    try:
        from .data.imagenet import load_imagenet_dataset, synthetic_samples  # local import

        samples = load_imagenet_dataset(
            dataset_name=cfg.dataset_name,
            split=cfg.split,
            num_samples=cfg.num_samples,
            dataset_id=cfg.dataset_id,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
            resolution=cfg.resolution_data,
            trust_remote_code=cfg.trust_remote_code,
            verbose=cfg.verbose,
        )
        if samples:
            return list(samples)
        if allow_synthetic:
            LOGGER.warning(
                "No ImageNet samples loaded; falling back to synthetic samples "
                "(EXTERNAL_DEFAULT, smoke-test only)."
            )
            return list(synthetic_samples(n=cfg.num_samples or 8, resolution=cfg.resolution_data, seed=cfg.seed))
        return []
    except Exception as exc:  # pragma: no cover - depends on environment
        LOGGER.warning("ImageNet loading failed (%s); falling back to synthetic samples.", exc)
        if not allow_synthetic:
            raise
        from .data.imagenet import synthetic_samples

        return list(synthetic_samples(n=cfg.num_samples or 8, resolution=cfg.resolution_data, seed=cfg.seed))


class _PixelSampleDataset:
    """Minimal dataset wrapper turning samples into raw pixel tensors ``[0, 1]``.

    Raw (non-normalized) pixels are the space in which the l_inf ball is defined
    (Addendum); normalization happens inside the encoder.
    """

    def __init__(self, samples: Sequence[Any], resolution: int = 224, device=None):
        self.samples = list(samples)
        self.resolution = int(resolution)
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from .data.imagenet import image_to_pixels, normalize_imagenet_sample

        sample = self.samples[index]
        pixels = None
        if hasattr(sample, "pixels") and getattr(sample, "pixels") is not None:
            pixels = getattr(sample, "pixels")
            if pixels.dim() == 3:
                pixels = pixels.unsqueeze(0)
        if pixels is None:
            sample = normalize_imagenet_sample(sample, index=index, with_pixels=True)
            pixels = getattr(sample, "pixels", None)
        if pixels is None:
            image = getattr(sample, "image", sample)
            pixels = image_to_pixels(image, resolution=self.resolution, in01=True)
        if self.device is not None:
            pixels = pixels.to(self.device)
        label = getattr(sample, "label", None)
        return {"pixels": pixels.float(), "label": label, "index": index}


def build_dataloader(samples: Sequence[Any], cfg: RobustCLIPTrainConfig, device=None):
    _require_torch()
    from torch.utils.data import DataLoader

    dataset = _PixelSampleDataset(samples, resolution=cfg.resolution_data, device=device)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
        generator=generator,
        collate_fn=_collate_pixels,
    )


def _collate_pixels(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    _require_torch()
    pixels = torch.cat([item["pixels"] for item in batch], dim=0)
    labels = [item["label"] for item in batch]
    return {
        "pixels": pixels,
        "labels": labels,
        "index": [item["index"] for item in batch],
    }


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class RobustCLIPTrainer:
    """Unsupervised adversarial fine-tuning of the CLIP vision encoder (FARE)."""

    def __init__(
        self,
        config: RobustCLIPTrainConfig,
        *,
        student=None,
        teacher=None,
        samples: Optional[Sequence[Any]] = None,
        optimizer=None,
        scheduler=None,
    ):
        _require_torch()
        self.cfg = config
        self.device = resolve_device(config.device)

        self.student = student if student is not None else self._build_student()
        self.teacher = teacher if teacher is not None else self._build_teacher()
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad_(False)
        self.student.train()

        self.samples = list(samples) if samples is not None else None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._scheduler_lambda = None
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []
        self.objective = make_objective(config.objective)
        self.external_defaults = config.external_defaults()

    # ------------------------------------------------------------------ builders
    def _encoder_config(self, trainable: Any) -> CLIPVisionConfig:
        return CLIPVisionConfig(
            model_name=self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            pretrained_path=self.cfg.pretrained_path,
            resolution=self.cfg.resolution,
            device=str(self.device),
            trainable=trainable,
            trainable_scope=self.cfg.trainable_scope if trainable != "none" else "none",
            num_trainable_blocks=self.cfg.num_trainable_blocks,
            use_grad_checkpointing=self.cfg.use_grad_checkpointing,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            seed=self.cfg.seed,
        )

    def _build_student(self):
        encoder = build_clip_vision_encoder(self._encoder_config(self.cfg.trainable_scope))
        encoder.to(self.device)
        return encoder

    def _build_teacher(self):
        """Frozen *original* CLIP encoder providing ``phi_Org`` (Eq. (2))."""
        teacher_cfg = self._encoder_config("none")
        if self.cfg.teacher_pretrained_path:
            teacher_cfg.pretrained_path = self.cfg.teacher_pretrained_path
        try:
            teacher = build_clip_vision_encoder(teacher_cfg)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not build teacher encoder (%s); FARE will fall back "
                           "to the student's own initial embeddings.", exc)
            return None
        teacher.to(self.device)
        return teacher

    def trainable_parameters(self):
        fn = getattr(self.student, "trainable_parameters", None)
        if callable(fn):
            params = list(fn())
            if params:
                return params
        return [p for p in self.student.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    # ------------------------------------------------------------------ helpers
    def features(self, pixels, *, normalize: bool = True):
        """Fine-tuned (student) class-token features ``phi_FT``."""
        out = self.student(pixels, normalize=normalize, return_features=True)
        return out if not isinstance(out, dict) else out["features"]

    @torch.no_grad()
    def teacher_features(self, pixels, *, normalize: bool = True):
        """Original (teacher) class-token features ``phi_Org``."""
        if self.teacher is None:
            return None
        out = self.teacher(pixels, normalize=normalize, return_features=True)
        return out if not isinstance(out, dict) else out["features"]

    def _loss_at(self, x, org_features):
        """Objective evaluated at ``x`` (a differentiable pixel tensor)."""
        ft = self.features(x)
        if self.cfg.objective == "fare":
            return self.objective(ft, org_features, norm=self.cfg.loss_norm)
        ft_clean = self.features_target_clean
        return self.objective(ft, ft_clean, norm=self.cfg.loss_norm)

    # ------------------------------------------------------------------ steps
    def _train_micro_batch(self, pixels):
        """One micro-batch: inner PGD maximisation + FARE gradient step."""
        pixels = pixels.to(self.device)
        with torch.no_grad():
            org_features = self.teacher_features(pixels)
            if org_features is None:
                # No teacher available: use the student's current clean embedding
                # (documented fallback; the paper always uses the original CLIP).
                org_features = self.features(pixels).detach()
            self.features_target_clean = self.features(pixels).detach() if self.cfg.objective == "tecoa" else None

        loss_at = self._loss_at

        def _loss_fn(z):
            return loss_at(z, org_features)

        perturb = pgd_perturbation(
            self.student,
            pixels,
            _loss_fn,
            eps=float(self.cfg.eps),
            alpha=float(self.cfg.alpha),
            steps=int(self.cfg.adv_steps),
            momentum=float(self.cfg.train_pgd_momentum),
            random_start=bool(self.cfg.random_start),
            precision=self.cfg.precision,
            quant_scale=self.cfg.quant_scale,
        )
        adv = perturb["adversarial"]
        if perturb.get("codes") is not None:
            # honour the Addendum int16/int32 storage policy and verify it
            assert_mandated_dtype(perturb["codes"], self.cfg.precision)
            adv = decode_perturbation(
                perturb["codes"], float_dtype=float_dtype_for_precision(self.cfg.precision),
                quant_scale=self.cfg.quant_scale,
            )
            adv = adversarial_pixels(pixels, adv - pixels)

        adv_features = self.features(adv)
        if self.cfg.objective == "fare":
            loss = fare_loss(adv_features, org_features, norm=self.cfg.loss_norm)
        else:
            loss = tecoa_loss(adv_features, self.features_target_clean, norm=self.cfg.loss_norm)

        metrics = {
            "loss": float(loss.detach().cpu()),
            "adv_linf": float(perturb["linf"]),
            "adv_distance": float(
                _pairwise_distance(adv_features.detach(), org_features, norm=self.cfg.loss_norm, reduction="mean")
                .detach().cpu()
            ),
            "codes_dtype": str(perturb.get("codes_dtype")),
        }
        if self.cfg.clean_loss_weight:
            clean_features = self.features(pixels)
            metrics["clean_loss"] = float(
                clean_embedding_loss(clean_features, org_features, norm=self.cfg.loss_norm).detach().cpu()
            )
            loss = loss + float(self.cfg.clean_loss_weight) * clean_embedding_loss(
                clean_features, org_features, norm=self.cfg.loss_norm
            )
        return loss, metrics

    def _step(self, pixels) -> Dict[str, float]:
        loss, metrics = self._train_micro_batch(pixels)
        scaled = loss / max(1, int(self.cfg.grad_accum_steps or 1))
        scaled.backward()
        return metrics

    # ------------------------------------------------------------------ epoch
    def train_epoch(self, dataloader, epoch: int, *, max_steps: Optional[int] = None) -> Dict[str, Any]:
        accum = max(1, int(self.cfg.grad_accum_steps or 1))
        self.optimizer.zero_grad(set_to_none=True)
        running: Dict[str, float] = {}
        num_micro = 0
        started = time.time()
        for micro_index, batch in enumerate(dataloader):
            metrics = self._step(batch["pixels"])
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    running[key] = running.get(key, 0.0) + float(value)
            num_micro += 1
            if num_micro % accum == 0 or (micro_index + 1) == len(dataloader):
                if self.cfg.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), float(self.cfg.grad_clip_norm))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                if self.global_step % max(1, int(self.cfg.log_every)) == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    log_metric("train/loss", running.get("loss", float("nan")) / max(1, num_micro), logger=LOGGER,
                               step=self.global_step)
                    log_metric("train/lr", lr, logger=LOGGER, step=self.global_step)
                    log_metric("train/adv_linf", running.get("adv_linf", float("nan")) / max(1, num_micro),
                               logger=LOGGER, step=self.global_step)
                if max_steps is not None and self.global_step >= int(max_steps):
                    break
        summary = {k: v / max(1, num_micro) for k, v in running.items()}
        summary["epoch"] = epoch
        summary["global_step"] = self.global_step
        summary["lr"] = float(self.optimizer.param_groups[0]["lr"])
        summary["seconds"] = time.time() - started
        summary["num_micro_batches"] = num_micro
        self.history.append(summary)
        return summary

    # ------------------------------------------------------------------ full run
    def train(self) -> Dict[str, Any]:
        cfg = self.cfg
        samples = self.samples if self.samples is not None else resolve_training_samples(cfg)
        if not samples:
            raise RuntimeError("no training samples available")
        if cfg.verbose:
            LOGGER.info("Training on %d samples (effective batch %d = %d x %d)",
                        len(samples), cfg.effective_batch(), cfg.batch_size, cfg.grad_accum_steps)
            if cfg.effective_batch() != cfg.effective_batch_size:
                LOGGER.warning(
                    "Effective batch %d differs from the paper body's 128 "
                    "(batch_size=%d, grad_accum=%d). Logged as EXTERNAL_DEFAULT.",
                    cfg.effective_batch(), cfg.batch_size, cfg.grad_accum_steps,
                )
        dataloader = build_dataloader(samples, cfg, device=None)
        steps_per_epoch = max(1, math.ceil(max(1, len(dataloader)) / max(1, cfg.grad_accum_steps or 1)))
        total_steps = steps_per_epoch * max(1, int(cfg.epochs))
        if cfg.max_steps is not None:
            total_steps = min(total_steps, int(cfg.max_steps))
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.trainable_parameters(), cfg)
        if self.scheduler is None:
            self.scheduler, self._scheduler_lambda = build_lr_scheduler(self.optimizer, total_steps, cfg)
        if cfg.verbose and self._scheduler_lambda is not None:
            LOGGER.info("LR schedule: linear warmup over %d steps (%.0f%% of %d), then cosine decay "
                        "to min_lr_ratio=%.3f (EXTERNAL_DEFAULT)",
                        self._scheduler_lambda.warmup_steps, 100 * cfg.warmup_fraction,
                        total_steps, cfg.min_lr_ratio)

        timer = Timer("train", logger=LOGGER if cfg.verbose else None)
        for epoch in range(1, int(cfg.epochs) + 1):
            summary = self.train_epoch(dataloader, epoch, max_steps=cfg.max_steps)
            if cfg.verbose:
                LOGGER.info("epoch %d/%d done in %.1fs: loss=%.4f lr=%.3e linf=%.4f",
                            epoch, cfg.epochs, summary.get("seconds", 0.0),
                            summary.get("loss", float("nan")), summary.get("lr", float("nan")),
                            summary.get("adv_linf", float("nan")))
            if epoch % max(1, int(cfg.save_every)) == 0:
                path = self.checkpoint_path(epoch=epoch)
                self.save(path, epoch=epoch)
            if cfg.max_steps is not None and self.global_step >= int(cfg.max_steps):
                LOGGER.info("Reached max_steps=%d; stopping.", cfg.max_steps)
                break
        elapsed = timer.stop(log=False)
        final_path = self.checkpoint_path(epoch=None)
        self.save(final_path, epoch=int(cfg.epochs))
        result = {
            "checkpoint": final_path,
            "epochs": int(cfg.epochs),
            "global_step": self.global_step,
            "num_samples": len(samples),
            "history": self.history,
            "elapsed_seconds": elapsed,
            "config": cfg.as_dict(),
            "provenance": provenance_metadata(cfg.as_dict(), external_defaults=self.external_defaults),
        }
        if cfg.verbose:
            LOGGER.info("Saved Robust CLIP vision encoder to %s", final_path)
        return result

    # ------------------------------------------------------------------ io
    def checkpoint_path(self, epoch: Optional[int] = None) -> str:
        cfg = self.cfg
        eps_tag = f"eps{int(round(float(cfg.eps) * 255))}" if cfg.eps else "epsNA"
        name = f"robust_clip_{cfg.model_name}_{eps_tag}_{cfg.objective}"
        if epoch is not None:
            name += f"_epoch{epoch}"
        name += ".pt"
        if cfg.output_file and epoch is None:
            name = cfg.output_file
        return str(Path(cfg.output_dir) / name)

    def save(self, path: str, *, epoch: Optional[int] = None) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = None
        if hasattr(self.student, "state_dict"):
            try:
                state = self.student.state_dict()
            except Exception:  # pragma: no cover
                state = None
        payload = {
            "vision_encoder": state,
            "config": self.cfg.as_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "history": self.history,
            "provenance": provenance_metadata(self.cfg.as_dict(), external_defaults=self.external_defaults),
            "paper_body_facts": PAPER_BODY_FACTS,
            "addendum_facts": ADDENDUM_FACTS,
        }
        torch.save(payload, path)
        meta_path = str(Path(path).with_suffix(".json"))
        save_json({k: v for k, v in payload.items() if k != "vision_encoder"}, meta_path)
        return path


# --------------------------------------------------------------------------------------
# Smoke test / self test
# --------------------------------------------------------------------------------------


def run_smoke_test(verbose: bool = True) -> Dict[str, Any]:
    """Model-free end-to-end training check.

    Uses :class:`RandomVisionEncoder` students/teachers and synthetic ImageNet
    samples so that no weights or data downloads are needed.  Validates:
      * FARE loss follows Eq. (2) (squared l2, class token);
      * inner PGD honours the eps ball, step count and step size 1/255;
      * perturbation storage follows the int16 (half) / int32 (single) policy;
      * AdamW betas, LR and WD match the paper body;
      * LR schedule warms up linearly to the peak at 7% of total steps;
      * checkpoint save/load round-trip.
    """
    _require_torch()
    results: Dict[str, Any] = {}

    # ---- objective checks (paper Eq. (2)) ----
    a = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
    fare = fare_loss(a, b)
    assert abs(float(fare) - ((1.0 + 0.0 + 0.0 + 1.0) / 2)) < 1e-6, float(fare)
    assert abs(float(fare_loss(a, b, norm="l1")) - ((1.0 + 1.0) / 2)) < 1e-6
    assert abs(float(clean_embedding_loss(a, b)) - float(fare)) < 1e-6  # Eq. (1)
    results["fare_matches_eq2"] = True

    # ---- LR schedule (7% warmup, cosine) ----
    fn = linear_warmup_cosine_lambda(total_steps=100, warmup_fraction=0.07)
    assert fn.warmup_steps == 7, fn.warmup_steps
    assert abs(fn(0) - 1 / 7) < 1e-9
    assert abs(fn(6) - 1.0) < 1e-9
    assert abs(fn(7) - 1.0) < 1e-6  # peak right after warmup
    assert fn(99) < 1e-6  # cosine decays to ~0
    results["warmup_7pct"] = True

    # ---- PGD ball / precision policy ----
    for precision, expected_dtype in (("single", INT32), ("half", INT16)):
        enc = RandomVisionEncoder(CLIPVisionConfig(resolution=32, device="cpu"))
        enc.to("cpu")
        x = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        eps = 4.0 / 255.0

        calls = {"n": 0}

        def loss_fn(z, _enc=enc):
            calls["n"] += 1
            return _enc(z, normalize=True, return_features=True).pow(2).sum()

        out = pgd_perturbation(
            enc, x, loss_fn, eps=eps, alpha=1.0 / 255.0, steps=10,
            precision=precision, quant_scale=QUANT_SCALE,
        )
        assert calls["n"] == 10, "paper body: 10 PGD steps"
        assert out["linf"] <= eps + 1e-5, out["linf"]
        assert out["codes"].dtype == expected_dtype, out["codes"].dtype
        assert_mandated_dtype(out["codes"], precision)
        # decode round-trip stays inside the ball
        delta = decode_perturbation(out["codes"], float_dtype=torch.float32)
        assert float(delta.abs().max()) <= eps + 1e-5
        adv = (x + delta.clamp(-eps, eps)).clamp(0.0, 1.0)
        assert float((adv - x).abs().max()) <= eps + 1e-5
        results[f"precision_{precision}_dtype"] = str(out["codes"].dtype)

    # ---- optimizer hyperparameters (paper body) ----
    cfg = RobustCLIPTrainConfig(eps=4.0 / 255.0, batch_size=32, epochs=1, max_steps=2,
                                device="cpu", num_samples=4, log_every=1, save_every=1,
                                output_dir="results/_smoke_train", verbose=bool(verbose))
    assert cfg.effective_batch() == 128, cfg.effective_batch()
    assert cfg.betas == (0.9, 0.95)
    student = RandomVisionEncoder(CLIPVisionConfig(resolution=32, device="cpu"))
    teacher = RandomVisionEncoder(CLIPVisionConfig(resolution=32, device="cpu"))
    teacher.load_state_dict(student.state_dict())
    opt = build_optimizer(list(student.parameters()), cfg)
    groups = opt.param_groups[0]
    assert abs(groups["lr"] - 1e-5) < 1e-12 and abs(groups["weight_decay"] - 1e-4) < 1e-12
    assert tuple(groups["betas"]) == (0.9, 0.95)
    sched, lam = build_lr_scheduler(opt, total_steps=100, cfg=cfg)
    results["optimizer_matches_paper"] = True

    # ---- full training loop on synthetic data ----
    from .data.imagenet import synthetic_samples

    samples = synthetic_samples(n=4, resolution=32, seed=0)
    trainer = RobustCLIPTrainer(
        replace(cfg, resolution=32, resolution_data=32, num_samples=4),
        student=student, teacher=teacher, samples=samples, optimizer=opt, scheduler=sched,
    )
    result = trainer.train()
    assert result["global_step"] >= 1, result["global_step"]
    assert Path(result["checkpoint"]).exists(), result["checkpoint"]
    payload = torch.load(result["checkpoint"], map_location="cpu")
    assert "config" in payload and "provenance" in payload
    results["checkpoint"] = result["checkpoint"]
    results["history"] = result["history"]
    results["ok"] = True

    if verbose:
        log_table(
            [{"check": k, "value": str(v)} for k, v in results.items() if k != "history"],
            title="train_robust_clip smoke test",
        )
    return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "train_robust_clip.yaml")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        LOGGER.warning("Config file %s not found; using CLI/defaults.", path)
        return {}
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not parse config %s (%s); using CLI/defaults.", path, exc)
        return {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robust_clip_repro.train_robust_clip",
        description="Unsupervised adversarial fine-tuning (FARE) of the CLIP vision encoder.",
    )
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--model-name", default=None, help="open_clip architecture (paper: ViT-L-14)")
    parser.add_argument("--pretrained", default=None, help="open_clip pretrained tag (paper: openai)")
    parser.add_argument("--pretrained-path", default=None, help="local checkpoint for the student init")
    parser.add_argument("--teacher-pretrained-path", default=None, help="local checkpoint for phi_Org")
    parser.add_argument("--resolution", type=int, default=None, help="input resolution (paper: 224)")
    parser.add_argument("--trainable-scope", default=None,
                        choices=["all", "last_blocks", "projection", "none"])
    parser.add_argument("--num-trainable-blocks", type=int, default=None)
    parser.add_argument("--objective", default=None, choices=["fare", "tecoa"])
    parser.add_argument("--loss-norm", default=None, choices=["l2_squared", "l1"])
    parser.add_argument("--dataset-id", default=None, help="HuggingFace dataset id (paper: imagenet-1k)")
    parser.add_argument("--split", default=None, help="dataset split (ImageNet train)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="paper body: 2")
    parser.add_argument("--batch-size", type=int, default=None, help="per-device batch (EXTERNAL_DEFAULT)")
    parser.add_argument("--effective-batch-size", type=int, default=None, help="paper body: 128")
    parser.add_argument("--grad-accum", dest="grad_accum_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="paper body: 1e-5")
    parser.add_argument("--weight-decay", type=float, default=None, help="paper body: 1e-4")
    parser.add_argument("--betas", nargs=2, type=float, default=None, help="paper body: 0.9 0.95")
    parser.add_argument("--warmup-fraction", type=float, default=None, help="paper body: 0.07")
    parser.add_argument("--min-lr-ratio", type=float, default=None, help="EXTERNAL_DEFAULT")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="EXTERNAL_DEFAULT")
    parser.add_argument("--eps", default=None, help="REQUIRED l_inf radius, e.g. 4/255 or 2/255")
    parser.add_argument("--alpha", default=None, help="PGD step size (paper body: 1/255)")
    parser.add_argument("--adv-steps", type=int, default=None, help="paper body: 10")
    parser.add_argument("--train-pgd-momentum", type=float, default=None, help="EXTERNAL_DEFAULT (paper: unstated)")
    parser.add_argument("--precision", default=None, choices=["single", "half"])
    parser.add_argument("--amp", action="store_true", default=None, help="EXTERNAL_DEFAULT mixed precision")
    parser.add_argument("--seed", type=int, default=None, help="EXTERNAL_DEFAULT")
    parser.add_argument("--num-workers", type=int, default=None, help="EXTERNAL_DEFAULT")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="EXTERNAL_DEFAULT (smoke tests)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="model-free orchestration check")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser


def _namespace_to_config(args: argparse.Namespace, cfg_dict: Dict[str, Any]) -> RobustCLIPTrainConfig:
    overrides: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is None or key in ("config", "smoke_test", "quiet", "log_file"):
            continue
        if key in ("betas",):
            overrides[key] = tuple(value)
        else:
            overrides[key] = value
    if "eps" in overrides:
        overrides["eps"] = parse_eps(overrides["eps"])
    if "alpha" in overrides:
        overrides["alpha"] = parse_eps(overrides["alpha"])
    if "amp" in overrides:
        overrides["amp"] = bool(overrides["amp"])
    merged = dict(cfg_dict or {})
    merged.update(overrides)
    return RobustCLIPTrainConfig.from_dict(merged)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=not args.quiet, log_file=args.log_file)

    if args.smoke_test or parse_eps(args.eps) is None and not (args.config or os.environ.get("ROBUST_CLIP_EPS")):
        if not args.smoke_test:
            # No budget supplied: refuse silently inventing eps.
            LOGGER.error(
                "No l_inf radius supplied. The paper body trains at 2/255 and 4/255; "
                "pass --eps 4/255 (or --config). Nothing is assumed."
            )
            if not args.smoke_test:
                return 2
        return 0 if run_smoke_test(verbose=run_smoke_test_succeeded() if False else not args.quiet) else 1

    cfg_dict = load_config(args.config)
    if parse_eps(args.eps) is None and os.environ.get("ROBUST_CLIP_EPS"):
        cfg_dict["eps"] = os.environ["ROBUST_CLIP_EPS"]
    try:
        config = _namespace_to_config(args, cfg_dict)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    if config.verbose:
        log_config(config.as_dict(), logger=LOGGER, title="robust CLIP training configuration")

    trainer = RobustCLIPTrainer(config)
    result = trainer.train()
    payload = {
        "checkpoint": result["checkpoint"],
        "epochs": result["epochs"],
        "global_step": result["global_step"],
        "num_samples": result["num_samples"],
        "elapsed_seconds": result["elapsed_seconds"],
        "history": result["history"],
        "provenance": result["provenance"],
    }
    out_path = Path(config.output_dir) / (config.output_file or "train_robust_clip_result.json")
    save_json(payload, str(out_path))
    if not config.quiet if hasattr(config, "quiet") else True:
        print(json.dumps(payload, indent=2, default=str))
    return 0


def run_smoke_test_succeeded() -> bool:  # pragma: no cover - trivial helper
    return True


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
