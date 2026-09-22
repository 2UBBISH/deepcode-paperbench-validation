"""CoTTA baseline adapter for the FOA reproduction.

Self-contained implementation of *Continual Test-Time Adaptation* (CoTTA,
Wang et al., CVPR 2022) used as a comparison method against FOA.  The recipe
follows the hyper-parameters of the FOA paper (Appendix B.2):

    SGD, momentum 0.9, lr 0.05, pseudo-label confidence threshold 0.1,
    32 augmentations (augmentation ensemble), stochastic restoration
    probability 0.01, teacher EMA factor 0.999, batch size 64.

CoTTA maintains:

* a **student** network updated online with SGD using a consistency loss
  between its predictions and the (augmented-view) predictions of
* a **teacher** network whose weights are an exponential moving average of the
  student's weights (``theta_t = alpha * theta_t + (1 - alpha) * theta_s``),
* plus a **stochastic restoration** step that resets a random ``p = 0.01``
  subset of parameters back to the source (pretrained) values to avoid
  error accumulation / catastrophic forgetting.

The adapter exposes the same duck-typed protocol as the other FOA baselines
(``step`` / ``reset`` / ``state_dict`` / ``load_state_dict`` / ``predict`` and a
``build_cotta`` / ``build_baseline`` factory), so ``scripts/run_baselines.py``
can drive it unchanged.

This module never optimises anything with the FOA machinery; it is a plain
gradient-based TTA baseline and therefore *does* call ``backward()``.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CoTTA",
    "CoTTABaseline",
    "CoTTAConfig",
    "AugmentationEnsemble",
    "build_cotta",
    "build_baseline",
    "consistency_loss",
    "stochastic_restore",
    "ema_update",
    "select_trainable_params",
    "DEFAULT_LR",
    "DEFAULT_MOMENTUM",
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_THRESHOLD",
    "DEFAULT_N_AUG",
    "DEFAULT_RESTORATION_PROB",
    "DEFAULT_BATCH_SIZE",
]

# --------------------------------------------------------------------------- #
# Paper (Appendix B.2) hyper-parameters
# --------------------------------------------------------------------------- #
DEFAULT_LR = 0.05
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_EMA_ALPHA = 0.999
DEFAULT_THRESHOLD = 0.1
DEFAULT_N_AUG = 32
DEFAULT_RESTORATION_PROB = 0.01
DEFAULT_AUG_LEVEL = 0.3
DEFAULT_CHUNK = 8
DEFAULT_NUM_CLASSES = 1000
DEFAULT_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup working for both dict-like and attribute-like configs."""
    if cfg is None:
        return default
    for key in keys:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            cfg = cfg.get(key, None)
        else:
            cfg = getattr(cfg, key, None)
    return default if cfg is None else cfg


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the inner module that computes logits with gradients enabled.

    The FOA ``ViTWithCLSFeatures`` wrapper separates prompt injection from the
    timm ViT and runs its forward under ``torch.no_grad()``.  Gradient-based
    baselines therefore always work on the inner timm model (``.model``).
    """
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    return model


def _logits_from_output(output: Any) -> torch.Tensor:
    """Normalise the many output contracts used across the code base."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "output", "out", "pred", "predictions"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    if isinstance(output, (list, tuple)):
        tensors = [o for o in output if isinstance(o, torch.Tensor)]
        if tensors:
            # ``forward_features`` returns (cls_features, logits, final_cls);
            # the logits are the only 2-D tensor.
            for t in tensors:
                if t.dim() == 2:
                    return t
            return tensors[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _infer_device(model: Optional[nn.Module]) -> torch.device:
    if model is not None:
        for p in model.parameters():
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_trainable_params(model: nn.Module, which: str = "all") -> List[nn.Parameter]:
    """Select the parameter set CoTTA updates.

    ``which``:
      * ``"all"``        - every parameter (CoTTA default: full-model update)
      * ``"norm"``       - LayerNorm/BatchNorm affine parameters only
      * ``"head"``       - classification head only
    """
    which = (which or "all").lower()
    params: List[nn.Parameter] = []
    if which in ("all", "full", "model"):
        params = [p for p in model.parameters() if p.requires_grad or True]
        for p in params:
            p.requires_grad_(True)
        return params
    if which in ("norm", "affine", "norm_affine"):
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                for p in module.parameters(recurse=False):
                    p.requires_grad_(True)
                    params.append(p)
        # ViT blocks use timm LayerNorm variants; fall back to name matching.
        if not params:
            for name, p in model.named_parameters():
                if "norm" in name.lower() or "ln" in name.lower():
                    p.requires_grad_(True)
                    params.append(p)
        return params
    if which in ("head", "classifier"):
        for name, p in model.named_parameters():
            if any(tok in name.lower() for tok in ("head", "classifier", "fc", "pre_logits")):
                p.requires_grad_(True)
                params.append(p)
        return params
    raise ValueError(f"Unknown trainable parameter selection: {which!r}")


def ema_update(teacher: nn.Module, student: nn.Module, alpha: float = DEFAULT_EMA_ALPHA) -> None:
    """In-place EMA: ``teacher <- alpha * teacher + (1 - alpha) * student``."""
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            if tp.dtype.is_floating_point:
                tp.mul_(alpha).add_(sp.detach(), alpha=1.0 - alpha)
            else:
                tp.copy_(sp.detach())
        for tb, sb in zip(teacher.buffers(), student.buffers()):
            if tb.dtype.is_floating_point:
                tb.mul_(alpha).add_(sb.detach(), alpha=1.0 - alpha)
            else:
                tb.copy_(sb.detach())


@torch.no_grad()
def stochastic_restore(
    model: nn.Module,
    source_state: Dict[str, torch.Tensor],
    prob: float = DEFAULT_RESTORATION_PROB,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Restore a random fraction ``prob`` of parameters to the source values.

    Implements CoTTA's stochastic restoration: each parameter element is
    independently reset to its source value with probability ``prob``::

        m ~ Bernoulli(prob);  theta = m * theta_0 + (1 - m) * theta

    Returns the number of restored elements.
    """
    if prob <= 0.0:
        return 0
    restored = 0
    for name, param in model.named_parameters():
        src = source_state.get(name, None)
        if src is None or src.shape != param.shape:
            continue
        if not param.dtype.is_floating_point:
            continue
        mask = (torch.rand(param.shape, device=param.device, generator=generator) < prob)
        if not bool(mask.any()):
            continue
        restored += int(mask.sum().item())
        param.data[mask] = src.to(param.device, non_blocking=True)[mask]
    return restored


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CoTTA consistency (soft cross-entropy) loss with confidence filtering.

    ``teacher_probs`` are the softmax probabilities of the EMA teacher on the
    same augmented view; samples whose maximum teacher confidence is below
    ``threshold`` are masked out (they carry no useful supervision).

    Returns ``(loss, valid_mask)``.
    """
    log_student = F.log_softmax(student_logits.float(), dim=1)
    per_sample = -(teacher_probs.detach().float() * log_student).sum(dim=1)
    valid = teacher_probs.detach().max(dim=1).values >= float(threshold)
    if valid.any():
        if reduction == "sum":
            loss = per_sample[valid].sum()
        else:
            loss = per_sample[valid].mean()
    else:
        loss = student_logits.sum() * 0.0
    return loss, valid


# --------------------------------------------------------------------------- #
# Augmentation ensemble (32 augmented views)
# --------------------------------------------------------------------------- #
class AugmentationEnsemble:
    """CoTTA augmentation ensemble.

    Each view applies an independent random chain of ``chain_depth``
    augmentations drawn from {color jitter, gaussian blur, gaussian noise,
    random affine (rotation/translation/scale), cutout}, with magnitudes scaled
    by ``level``.  ``n_views`` defaults to 32 per Appendix B.2.

    The ensemble is *stochastic* (fresh draws at every call), matching the
    official CoTTA implementation which re-samples augmentations online.
    """

    OPS = ("color_jitter", "gaussian_blur", "gaussian_noise", "affine", "cutout")

    def __init__(
        self,
        n_views: int = DEFAULT_N_AUG,
        level: float = DEFAULT_AUG_LEVEL,
        chain_depth: int = 3,
        image_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        seed: Optional[int] = None,
    ) -> None:
        self.n_views = int(n_views)
        self.level = float(level)
        self.chain_depth = max(1, int(chain_depth))
        self.image_size = int(image_size)
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1)
        self._rng = random.Random(seed)

    # -- individual operations (operate on normalised tensors) -------------- #
    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)).clamp(0.0, 1.0)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def _color_jitter(self, x: torch.Tensor, mag: float) -> torch.Tensor:
        img = self._denorm(x)
        n = img.shape[0]
        # brightness / contrast / saturation style perturbations
        brightness = 1.0 + (torch.rand(n, 1, 1, 1, device=img.device) * 2 - 1) * mag
        contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=img.device) * 2 - 1) * mag
        img = img * brightness
        gray = img.mean(dim=1, keepdim=True)
        img = (img - gray) * contrast + gray
        img = img.clamp(0.0, 1.0)
        return self._norm(img)

    def _gaussian_blur(self, x: torch.Tensor, mag: float) -> torch.Tensor:
        img = self._denorm(x)
        kernel = 2 * int(1 + round(2 * mag)) + 1  # odd kernel, >= 3
        sigma = 0.1 + 2.0 * mag
        coords = torch.arange(kernel, dtype=img.dtype, device=img.device) - (kernel - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, 1, kernel)
        weight_h = g.expand(img.shape[1], 1, 1, kernel)
        weight_w = g.view(1, 1, kernel, 1).expand(img.shape[1], 1, kernel, 1)
        pad = kernel // 2
        img = F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="reflect"), weight_h, groups=img.shape[1])
        img = F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="reflect"), weight_w, groups=img.shape[1])
        return self._norm(img)

    def _gaussian_noise(self, x: torch.Tensor, mag: float) -> torch.Tensor:
        return x + torch.randn_like(x) * (0.05 + 0.15 * mag)

    def _affine(self, x: torch.Tensor, mag: float) -> torch.Tensor:
        n = x.shape[0]
        angle = (torch.rand(n, device=x.device) * 2 - 1) * (30.0 * mag)
        translate = (torch.rand(n, 2, device=x.device) * 2 - 1) * (0.15 * mag)
        scale = 1.0 + (torch.rand(n, device=x.device) * 2 - 1) * (0.3 * mag)
        out = []
        for i in range(n):
            theta = torch.zeros(1, 2, 3, device=x.device, dtype=x.dtype)
            rad = angle[i] * math.pi / 180.0
            cos, sin = torch.cos(rad), torch.sin(rad)
            theta[0, 0, 0] = cos / scale[i]
            theta[0, 0, 1] = -sin / scale[i]
            theta[0, 1, 0] = sin / scale[i]
            theta[0, 1, 1] = cos / scale[i]
            theta[0, 0, 2] = translate[i, 0]
            theta[0, 1, 2] = translate[i, 1]
            grid = F.affine_grid(theta, (1,) + tuple(x.shape[1:]), align_corners=False)
            out.append(F.grid_sample(x[i: i + 1], grid, align_corners=False, padding_mode="reflection"))
        return torch.cat(out, dim=0)

    def _cutout(self, x: torch.Tensor, mag: float) -> torch.Tensor:
        n, _, h, w = x.shape
        out = x.clone()
        size = int(round(0.25 * mag * min(h, w)))
        if size <= 0:
            return out
        for i in range(n):
            top = self._rng.randint(0, max(0, h - size))
            left = self._rng.randint(0, max(0, w - size))
            out[i, :, top: top + size, left: left + size] = 0.0
        return out

    # -- ensembles --------------------------------------------------------- #
    def _random_view(self, x: torch.Tensor) -> torch.Tensor:
        ops = {name: getattr(self, f"_{name}") for name in self.OPS}
        out = x
        depth = self.chain_depth if self._rng.random() < 0.5 else 1
        for _ in range(depth):
            name = self._rng.choice(self.OPS)
            mag = self.level * self._rng.choice((0.1, 0.2, 0.3, 0.4, 0.5))
            out = ops[name](out, mag)
        return out

    def views(self, x: torch.Tensor, n_views: Optional[int] = None) -> List[torch.Tensor]:
        """Return ``n_views`` independently augmented copies of the batch ``x``."""
        k = int(n_views if n_views is not None else self.n_views)
        return [self._random_view(x) for _ in range(k)]

    def __call__(self, x: torch.Tensor, n_views: Optional[int] = None) -> List[torch.Tensor]:
        return self.views(x, n_views=n_views)

    def extra_repr(self) -> str:
        return f"n_views={self.n_views}, level={self.level}, chain_depth={self.chain_depth}"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CoTTAConfig:
    """CoTTA hyper-parameters (FOA Appendix B.2 defaults)."""

    lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    ema_alpha: float = DEFAULT_EMA_ALPHA
    threshold: float = DEFAULT_THRESHOLD
    n_aug: int = DEFAULT_N_AUG
    chain_depth: int = 3
    aug_level: float = DEFAULT_AUG_LEVEL
    chunk_size: int = DEFAULT_CHUNK
    restoration_prob: float = DEFAULT_RESTORATION_PROB
    train_which: str = "all"
    num_classes: int = DEFAULT_NUM_CLASSES
    eps: float = DEFAULT_EPS
    episodic: bool = False
    grad_clip: Optional[float] = None
    adapt_after_loss: bool = True
    seed: int = 0
    image_size: int = 224
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: Any) -> "CoTTAConfig":
        """Build from a FOA YAML config (``baselines.cotta`` block)."""
        block = _cfg_get(cfg, "baselines", "cotta", default=None) or _cfg_get(cfg, "cotta", default=None) or {}
        kwargs: Dict[str, Any] = {}
        fields = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        for key in list(fields):
            value = _cfg_get(block, key, default=None)
            if value is not None:
                kwargs[key] = value
        # aliases used in some configs
        for alias, target in (
            ("lr", "lr"), ("learning_rate", "lr"), ("ema", "ema_alpha"),
            ("alpha", "ema_alpha"), ("p", "restoration_prob"),
            ("restoration", "restoration_prob"), ("n_views", "n_aug"),
            ("num_aug", "n_aug"), ("confidence_threshold", "threshold"),
        ):
            value = _cfg_get(block, alias, default=None)
            if value is not None:
                kwargs[target] = value
        num_classes = _cfg_get(cfg, "data", "num_classes_eval", default=None) or _cfg_get(
            cfg, "data", "num_classes", default=None
        ) or _cfg_get(cfg, "model", "num_classes", default=None)
        if num_classes:
            kwargs.setdefault("num_classes", int(num_classes))
        batch_size = _cfg_get(cfg, "data", "batch_size", default=None)
        if batch_size:
            kwargs.setdefault("batch_size", int(batch_size))
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Main adapter
# --------------------------------------------------------------------------- #
class CoTTA(nn.Module):
    """Test-time adaptation with a student network + EMA teacher (CoTTA).

    Per test batch the adapter

    1. draws ``n_aug`` augmented views of the batch,
    2. (teacher, no grad) predicts soft pseudo-labels for each view,
    3. (student, grad) minimises the confidence-filtered consistency loss,
    4. takes an SGD step (momentum 0.9, lr 0.05),
    5. updates the teacher by EMA (alpha 0.999),
    6. applies stochastic restoration (p 0.01) to the student.

    Predictions are read from the teacher on the *un-augmented* batch.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[CoTTAConfig] = None,
        *,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        threshold: float = DEFAULT_THRESHOLD,
        n_aug: int = DEFAULT_N_AUG,
        chain_depth: int = 3,
        aug_level: float = DEFAULT_AUG_LEVEL,
        chunk_size: int = DEFAULT_CHUNK,
        restoration_prob: float = DEFAULT_RESTORATION_PROB,
        train_which: str = "all",
        num_classes: int = DEFAULT_NUM_CLASSES,
        eps: float = DEFAULT_EPS,
        episodic: bool = False,
        grad_clip: Optional[float] = None,
        adapt_after_loss: bool = True,
        seed: int = 0,
        image_size: int = 224,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        cfg = config or CoTTAConfig()
        # explicit kwargs win over dataclass defaults, config wins over nothing
        self.config = CoTTAConfig(
            lr=cfg.lr if lr == DEFAULT_LR else lr,
            momentum=cfg.momentum if momentum == DEFAULT_MOMENTUM else momentum,
            weight_decay=cfg.weight_decay if weight_decay == DEFAULT_WEIGHT_DECAY else weight_decay,
            batch_size=cfg.batch_size if batch_size == DEFAULT_BATCH_SIZE else batch_size,
            ema_alpha=cfg.ema_alpha if ema_alpha == DEFAULT_EMA_ALPHA else ema_alpha,
            threshold=cfg.threshold if threshold == DEFAULT_THRESHOLD else threshold,
            n_aug=cfg.n_aug if n_aug == DEFAULT_N_AUG else n_aug,
            chain_depth=cfg.chain_depth if chain_depth == 3 else chain_depth,
            aug_level=cfg.aug_level if aug_level == DEFAULT_AUG_LEVEL else aug_level,
            chunk_size=cfg.chunk_size if chunk_size == DEFAULT_CHUNK else chunk_size,
            restoration_prob=(
                cfg.restoration_prob if restoration_prob == DEFAULT_RESTORATION_PROB else restoration_prob
            ),
            train_which=cfg.train_which if train_which == "all" else train_which,
            num_classes=cfg.num_classes if num_classes == DEFAULT_NUM_CLASSES else num_classes,
            eps=cfg.eps if eps == DEFAULT_EPS else eps,
            episodic=cfg.episodic if episodic is False else episodic,
            grad_clip=cfg.grad_clip if grad_clip is None else grad_clip,
            adapt_after_loss=cfg.adapt_after_loss if adapt_after_loss is True else adapt_after_loss,
            seed=cfg.seed if seed == 0 else seed,
            image_size=cfg.image_size if image_size == 224 else image_size,
        )

        self.model = _unwrap(model)                     # student (inner timm/ViT)
        self.device = torch.device(device) if device is not None else _infer_device(self.model)
        self.model.to(self.device)

        # Teacher = frozen copy of the source (student) network.
        self.teacher = copy.deepcopy(self.model).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Keep an immutable copy of the source state for stochastic restoration.
        self.source_state: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in self.model.state_dict().items()
        }

        self.trainable = select_trainable_params(self.model, self.config.train_which)
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        self.augment = AugmentationEnsemble(
            n_views=self.config.n_aug,
            level=self.config.aug_level,
            chain_depth=self.config.chain_depth,
            image_size=self.config.image_size,
            seed=self.config.seed,
        )
        self._generator = None
        self.num_steps = 0
        self.last_loss: Optional[float] = None
        self.last_valid_fraction: float = 0.0

    # ------------------------------------------------------------------ #
    # Introspection / state
    # ------------------------------------------------------------------ #
    def trainable_parameter_names(self) -> List[str]:
        trainable_ids = {id(p) for p in self.trainable}
        return [n for n, p in self.model.named_parameters() if id(p) in trainable_ids]

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.trainable))

    def reset(self) -> None:
        """Restore student + teacher to the source weights (episodic protocol)."""
        self.model.load_state_dict(self.source_state, strict=False)
        self.teacher.load_state_dict(self.source_state, strict=False)
        select_trainable_params(self.model, self.config.train_which)
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self.num_steps = 0
        self.last_loss = None
        self.last_valid_fraction = 0.0

    def load_source_state(self, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Load the source checkpoint used as the restoration anchor."""
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.teacher.load_state_dict(state_dict, strict=False)
        self.source_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "student": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "teacher": {k: v.detach().clone() for k, v in self.teacher.state_dict().items()},
            "num_steps": self.num_steps,
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        if "student" in state:
            self.model.load_state_dict(state["student"], strict=strict)
        if "teacher" in state:
            self.teacher.load_state_dict(state["teacher"], strict=strict)
        self.num_steps = int(state.get("num_steps", 0))

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _teacher_logits(self, images: torch.Tensor) -> torch.Tensor:
        return _logits_from_output(self.teacher(images.to(self.device)))

    def forward_features(self, images: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Teacher logits on the raw batch (+ penultimate features when available)."""
        output = self.teacher(images.to(self.device))
        features = None
        if isinstance(output, dict):
            features = output.get("final_cls", output.get("cls_features", None))
            if isinstance(features, (list, tuple)):
                features = features[-1]
        logits = _logits_from_output(output)
        return features, logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Teacher predictions (no adaptation) for the given batch."""
        return self._teacher_logits(images)

    # ------------------------------------------------------------------ #
    # Loss / adaptation
    # ------------------------------------------------------------------ #
    def loss(self, images: torch.Tensor, n_views: Optional[int] = None) -> torch.Tensor:
        """CoTTA consistency loss for a batch (no parameter update)."""
        return self._compute_loss(images, n_views=n_views, backward=False)

    def _compute_loss(
        self,
        images: torch.Tensor,
        n_views: Optional[int] = None,
        backward: bool = False,
    ) -> torch.Tensor:
        images = images.to(self.device)
        views = self.augment.views(images, n_views=n_views)
        chunk = max(1, int(self.config.chunk_size))
        total_loss = 0.0
        total_valid = 0
        total_seen = 0
        self.model.train()
        for start in range(0, len(views), chunk):
            batch_views = torch.cat(views[start: start + chunk], dim=0)
            with torch.no_grad():
                teacher_logits = self._teacher_logits(batch_views)
                teacher_probs = F.softmax(teacher_logits.float(), dim=1)
            student_logits = _logits_from_output(self.model(batch_views))
            loss, valid = consistency_loss(
                student_logits, teacher_probs, threshold=self.config.threshold, reduction="sum"
            )
            n_valid = int(valid.sum().item())
            total_valid += n_valid
            total_seen += int(valid.numel())
            if n_valid > 0:
                scaled = loss / float(n_valid)
                total_loss = total_loss + scaled if isinstance(total_loss, torch.Tensor) else scaled
            del student_logits, teacher_logits, teacher_probs
        self.model.eval()

        if isinstance(total_loss, torch.Tensor):
            loss_out = total_loss / max(1, len(views) // chunk)
        else:  # pragma: no cover - defensive
            loss_out = torch.zeros((), device=self.device, requires_grad=True)
        self.last_valid_fraction = (total_valid / total_seen) if total_seen else 0.0
        if backward and isinstance(loss_out, torch.Tensor) and loss_out.requires_grad:
            loss_out.backward()
        return loss_out

    def adapt(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Perform one CoTTA update step on the batch (student SGD + teacher EMA)."""
        images = images.to(self.device)
        loss = self._compute_loss(images, backward=True)
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.config.grad_clip
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        ema_update(self.teacher, self.model, alpha=self.config.ema_alpha)
        with torch.no_grad():
            restored = stochastic_restore(
                self.model, self.source_state, prob=self.config.restoration_prob, generator=self._generator
            )
        ema_update(self.teacher, self.model, alpha=self.config.ema_alpha) if False else None
        self.num_steps += 1
        self.last_loss = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        return {
            "loss": self.last_loss,
            "valid_fraction": self.last_valid_fraction,
            "restored": restored,
            "step": self.num_steps,
        }

    @torch.no_grad()
    def step(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adapt on ``images`` then return teacher logits for the same batch."""
        if self.config.adapt_after_loss:
            self.adapt(images)
        else:
            # predict first, then adapt (older TTA protocol variant)
            logits = self._teacher_logits(images)
            self.adapt(images)
            return logits
        return self._teacher_logits(images)

    __call__ = step  # type: ignore[assignment]

    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"lr={cfg.lr}, momentum={cfg.momentum}, ema_alpha={cfg.ema_alpha}, "
            f"threshold={cfg.threshold}, n_aug={cfg.n_aug}, "
            f"restoration_prob={cfg.restoration_prob}, train_which={cfg.train_which}"
        )


# alias matching the generic baseline runner's naming
CoTTABaseline = CoTTA


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_cotta(
    model: Optional[nn.Module] = None,
    cfg: Any = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> CoTTA:
    """Config-driven CoTTA factory (``baselines.cotta`` block of the YAML)."""
    config = kwargs.pop("config", None)
    if config is None:
        config = CoTTAConfig.from_config(cfg) if cfg is not None else CoTTAConfig()
    # drop None-valued overrides so dataclass values survive
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    if device is not None:
        config.device = str(device)
    if model is None:
        raise ValueError("build_cotta requires a model instance")
    return CoTTA(model, config=config, device=device, **overrides)


def build_baseline(model: Optional[nn.Module] = None, cfg: Any = None, **kwargs: Any) -> CoTTA:
    """Alias used by ``scripts/run_baselines.py``."""
    return build_cotta(model=model, cfg=cfg, **kwargs)
