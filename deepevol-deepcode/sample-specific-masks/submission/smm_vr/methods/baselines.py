"""Shared-mask Visual Reprogramming (VR) baselines for SMM.

Implements the four comparison methods reported in Table 1 / Table 2 of the
SMM paper (ICML 2024), Sec. 5 "Baselines":

  (1) ``Pad``    -- centering the original image and adding the noise pattern
                    around the images.
  (2) ``Narrow`` -- adding a narrow padding binary mask with a width of 28
                    (1/8 of the input image size) to the noise pattern that
                    covers the whole image (watermark).
  (3) ``Medium`` -- adding a mask being a quarter of the size (the width is 56)
                    of watermarks.
  (4) ``Full``   -- full watermarks that cover the whole images following
                    Wang et al. (2022).

All four methods share the SAME learnable noise pattern (a single shared
:math:`\\delta`, i.e. the "shared-mask" family) and differ only in the fixed
binary mask :math:`M` that decides *where* the pattern is applied::

    f_in(x_i) = r(x_i) + delta (*) M

with ``delta`` initialised to all zeros (as for SMM, Algorithm 1) and ``M`` a
non-learnable binary tensor.  This makes ``Full`` identical to the
"Shared-pattern VR ``f_in(x_i) = r(x_i) + delta`` with an all-one matrix"
variant described in Sec. 5 "Impact of Masking", which the paper states
"defaults to the 'full watermarks' baseline without using ``f_mask``".

Fair-comparison training schedule (paper, Sec. 5 "Baselines"):
"we apply the same learning rate and milestones following Chen et al. (2023),
with 0.01 being the initial learning rate and 0.1 being the learning rate
decay. Two hundred epochs are run in total, and the 100th and the 145th epochs
are the milestones."

Only the layout of the binary mask is prescriptive in the paper; geometric
defaults that the paper leaves unspecified are:

* ``Pad`` uses a padding band of 32 pixels at 224x224 (scaled proportionally
  for the 384x384 ViT input).  The target image is *down-scaled and centred*
  inside the canvas so the learnable pattern only touches the surrounding band.
* ``Narrow`` / ``Medium`` use centred *border* bands of width ``input_size/8``
  (28 at 224) and ``input_size/4`` (56 at 224); the centre keeps the clean
  image.  A band width of ``input_size`` degenerates to the all-ones ``Full``
  watermark.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Guarded project imports (this module must stay importable on its own)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - guarded
    from ..modules.reprogram import init_zero_pattern
except Exception:  # pragma: no cover - fallback
    def init_zero_pattern(channels: int = 3, height: int = 224, width: int = 224,
                          device=None, dtype=torch.float32) -> torch.Tensor:
        """All-zero shared noise pattern (Algorithm 1 initialisation)."""
        return torch.zeros(int(channels), int(height), int(width),
                           device=device, dtype=dtype)

try:  # pragma: no cover - guarded
    from ..engine.metrics import RunResult, aggregate_seeds, format_mean_std
except Exception:  # pragma: no cover - fallback
    RunResult = None  # type: ignore

    def aggregate_seeds(values, ddof: int = 1):
        vals = [float(v) for v in values]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(max(var, 0.0))

    def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

try:  # pragma: no cover - guarded
    from ..engine.seeds import resolve_seeds, set_seed
except Exception:  # pragma: no cover - fallback
    def resolve_seeds(seeds=None, n_seeds=None):
        return list(seeds) if seeds else [0, 1, 2]

    def set_seed(seed: int, **kwargs):
        import random

        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed


# ---------------------------------------------------------------------------
# Constants (paper Sec. 5 "Baselines" + Appendix C Table 9 schedule)
# ---------------------------------------------------------------------------
BASELINE_NAMES: Tuple[str, ...] = ("pad", "narrow", "medium", "full")

DEFAULT_IMAGE_SIZE: int = 224
VIT_IMAGE_SIZE: int = 384

# ``Narrow`` = 28 = 1/8 of 224 ; ``Medium`` = 56 = 1/4 of 224 (paper Sec. 5).
NARROW_WIDTH_RATIO: float = 1.0 / 8.0
MEDIUM_WIDTH_RATIO: float = 1.0 / 4.0
# ``Pad`` band width: 32 at 224 -> scaled proportionally for other resolutions.
DEFAULT_PAD_WIDTH: int = 32
PAD_WIDTH_RATIO: float = DEFAULT_PAD_WIDTH / float(DEFAULT_IMAGE_SIZE)

DEFAULT_LR: float = 0.01
DEFAULT_GAMMA: float = 0.1
DEFAULT_MILESTONES: Tuple[int, int] = (100, 145)
DEFAULT_EPOCHS: int = 200
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_MOMENTUM: float = 0.9
DEFAULT_WEIGHT_DECAY: float = 0.0
SMALL_BATCH_DATASETS: Tuple[str, ...] = ("dtd", "oxfordpets")
SMALL_BATCH_SIZE: int = 64

BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "pad": "Pad: centred down-scaled image, learnable noise pattern in the surrounding band",
    "narrow": "Narrow: watermark pattern times a narrow binary band of width 28 (1/8 of input)",
    "medium": "Medium: watermark pattern times a binary band of width 56 (1/4 of input)",
    "full": "Full: watermark pattern covering the whole image (all-one mask)",
}


# ---------------------------------------------------------------------------
# Mask construction helpers
# ---------------------------------------------------------------------------
def canonical_baseline_name(name: str) -> str:
    """Normalise a baseline spelling to one of :data:`BASELINE_NAMES`."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "padding": "pad",
        "pad_noise": "pad",
        "narrow_mask": "narrow",
        "medium_mask": "medium",
        "full_watermark": "full",
        "watermark": "full",
        "all_one": "full",
        "all_ones": "full",
        "share": "full",
        "shared": "full",
    }
    key = aliases.get(key, key)
    if key not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {BASELINE_NAMES}")
    return key


def watermark_width(
    name: str,
    input_size: int = DEFAULT_IMAGE_SIZE,
    *,
    narrow_ratio: float = NARROW_WIDTH_RATIO,
    medium_ratio: float = MEDIUM_WIDTH_RATIO,
    pad_ratio: float = PAD_WIDTH_RATIO,
    pad_width: Optional[int] = None,
) -> int:
    """Width (in pixels) of the binary band used by a baseline.

    ``narrow`` -> ``round(input_size/8)`` (28 at 224), ``medium`` ->
    ``round(input_size/4)`` (56 at 224), ``pad`` -> 32 at 224 (scaled
    proportionally), ``full`` -> ``input_size`` (the whole image).
    """
    name = canonical_baseline_name(name)
    input_size = int(input_size)
    if name == "full":
        return input_size
    if name == "narrow":
        width = int(round(input_size * float(narrow_ratio)))
    elif name == "medium":
        width = int(round(input_size * float(medium_ratio)))
    else:  # pad
        width = int(pad_width) if pad_width is not None else int(round(input_size * float(pad_ratio)))
    return int(max(1, min(width, input_size)))


def make_border_mask(
    input_size: int = DEFAULT_IMAGE_SIZE,
    width: int = DEFAULT_PAD_WIDTH,
    *,
    in_channels: int = 3,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Binary ``(C, H, W)`` mask: ones on a centred border band of ``width`` px.

    The central square ``(H - 2w) x (W - 2w)`` stays zero, so the noise pattern
    only replaces image content inside the band.  ``width`` is clipped to
    ``[0, ceil(input_size/2)]``; ``width <= 0`` yields the all-zero mask.
    """
    size = int(input_size)
    band = int(width)
    mask = torch.zeros(int(in_channels), size, size, device=device, dtype=dtype)
    if band <= 0:
        return mask
    band = min(band, (size + 1) // 2)
    mask[:, :band, :] = 1.0
    mask[:, size - band:, :] = 1.0
    mask[:, :, :band] = 1.0
    mask[:, :, size - band:] = 1.0
    return mask


def make_full_mask(
    input_size: int = DEFAULT_IMAGE_SIZE,
    *,
    in_channels: int = 3,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """All-one ``(C, H, W)`` mask, i.e. the ``Full`` watermark (and only-delta)."""
    return torch.ones(int(in_channels), int(input_size), int(input_size),
                      device=device, dtype=dtype)


def make_pad_mask(
    input_size: int = DEFAULT_IMAGE_SIZE,
    pad_width: Optional[int] = None,
    *,
    in_channels: int = 3,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Binary mask for ``Pad``: ones on the padding band, zeros in the centre."""
    width = watermark_width("pad", input_size, pad_width=pad_width)
    return make_border_mask(input_size, width, in_channels=in_channels,
                            device=device, dtype=dtype)


def make_baseline_mask(
    name: str,
    input_size: int = DEFAULT_IMAGE_SIZE,
    *,
    in_channels: int = 3,
    pad_width: Optional[int] = None,
    narrow_ratio: float = NARROW_WIDTH_RATIO,
    medium_ratio: float = MEDIUM_WIDTH_RATIO,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed binary mask of a baseline (``pad``/``narrow``/``medium``/``full``)."""
    name = canonical_baseline_name(name)
    if name == "full":
        return make_full_mask(input_size, in_channels=in_channels,
                              device=device, dtype=dtype)
    width = watermark_width(name, input_size, narrow_ratio=narrow_ratio,
                            medium_ratio=medium_ratio, pad_width=pad_width)
    return make_border_mask(input_size, width, in_channels=in_channels,
                            device=device, dtype=dtype)


def mask_coverage(name: str, input_size: int = DEFAULT_IMAGE_SIZE,
                  pad_width: Optional[int] = None) -> float:
    """Fraction of pixels the noise pattern covers (diagnostic helper)."""
    mask = make_baseline_mask(name, input_size, pad_width=pad_width)
    return float(mask.mean().item())


# ---------------------------------------------------------------------------
# Image preparation (Pad) / mask application
# ---------------------------------------------------------------------------
def resize_with_padding(
    images: torch.Tensor,
    pad_width: int,
    *,
    mode: str = "bilinear",
    fill: float = 0.0,
) -> torch.Tensor:
    """Centre a down-scaled copy of ``images`` inside a zero-filled canvas.

    ``images`` are ``(B, C, H, W)`` (already resized/normalised by the target
    task transform).  They are resized to ``(H - 2p, W - 2p)`` and centred, so
    the learnable pattern (which only covers the ``p``-pixel border band) never
    overlaps the actual image content -- the ``Pad`` baseline of Chen et al.
    (2023): "centering the original image and adding the noise pattern around
    the images".
    """
    if images.dim() != 4:
        raise ValueError(f"expected a 4D (B,C,H,W) tensor, got {tuple(images.shape)}")
    b, c, h, w = images.shape
    pad = max(0, min(int(pad_width), (min(h, w) - 1) // 2))
    if pad == 0:
        return images
    inner = F.interpolate(images, size=(h - 2 * pad, w - 2 * pad),
                          mode=mode, align_corners=False)
    canvas = torch.full((b, c, h, w), float(fill), dtype=images.dtype,
                        device=images.device)
    canvas[:, :, pad:h - pad, pad:w - pad] = inner
    return canvas


# ---------------------------------------------------------------------------
# Shared-mask reprogramming module
# ---------------------------------------------------------------------------
class SharedMaskVR(nn.Module):
    """Shared-mask VR baseline ``f_in(x) = r(x) + delta (*) M``.

    A single learnable noise pattern ``delta`` (shape ``(C, H, W)``, zero
    initialised exactly as in Algorithm 1) is shared by every sample; the fixed
    binary buffer ``mask`` selects the region it may affect.  Since the mask is
    non-learnable, ``delta`` is the only trainable parameter.
    """

    def __init__(
        self,
        name: str = "full",
        input_size: int = DEFAULT_IMAGE_SIZE,
        *,
        pad_width: Optional[int] = None,
        narrow_ratio: float = NARROW_WIDTH_RATIO,
        medium_ratio: float = MEDIUM_WIDTH_RATIO,
        in_channels: int = 3,
        delta_init: str = "zero",
        pad_mode: str = "bilinear",
        resize_to_input: bool = True,
    ) -> None:
        super().__init__()
        self.name = canonical_baseline_name(name)
        self.input_size = int(input_size)
        self.pad_width = (int(pad_width) if pad_width is not None
                          else watermark_width("pad", self.input_size))
        self.narrow_ratio = float(narrow_ratio)
        self.medium_ratio = float(medium_ratio)
        self.in_channels = int(in_channels)
        self.pad_mode = str(pad_mode)
        self.resize_to_input = bool(resize_to_input)

        self.band_width = watermark_width(
            self.name, self.input_size, narrow_ratio=self.narrow_ratio,
            medium_ratio=self.medium_ratio, pad_width=self.pad_width,
        )

        mask = make_baseline_mask(
            self.name, self.input_size, in_channels=self.in_channels,
            pad_width=self.pad_width, narrow_ratio=self.narrow_ratio,
            medium_ratio=self.medium_ratio,
        )
        self.register_buffer("mask", mask, persistent=True)

        init = str(delta_init).lower()
        if init in ("zero", "zeros", "none"):
            delta = init_zero_pattern(self.in_channels, self.input_size,
                                      self.input_size)
        elif init in ("normal", "randn"):
            delta = torch.randn(self.in_channels, self.input_size,
                                self.input_size) * 0.01
        elif init in ("uniform", "rand"):
            delta = (torch.rand(self.in_channels, self.input_size,
                                self.input_size) - 0.5) * 0.02
        else:
            raise ValueError(f"unknown delta_init {delta_init!r}")
        self.delta = nn.Parameter(delta.clone().detach().to(torch.float32))

    # -- properties ------------------------------------------------------
    @property
    def mask_generator(self):
        """Baselines have no mask generator (kept for interface parity)."""
        return None

    @property
    def has_mask_generator(self) -> bool:
        return False

    @property
    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    @property
    def label(self) -> str:
        return self.name.capitalize()

    def coverage(self) -> float:
        return float(self.mask.mean().item())

    # -- forward ---------------------------------------------------------
    def prepare(self, images: torch.Tensor) -> torch.Tensor:
        """Base (un-reprogrammed) image ``r(x)`` used by the baseline."""
        x = images
        if x.dim() != 4:
            raise ValueError(f"expected (B,C,H,W) input, got {tuple(x.shape)}")
        if self.resize_to_input and (x.shape[-2] != self.input_size
                                     or x.shape[-1] != self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode=self.pad_mode, align_corners=False)
        if self.name == "pad":
            x = resize_with_padding(x, self.pad_width, mode=self.pad_mode)
        return x

    def mask_for(self, images: torch.Tensor) -> torch.Tensor:
        """Fixed binary mask of the baseline, broadcast over the batch."""
        batch = images.shape[0]
        return self.mask.to(device=images.device, dtype=images.dtype).unsqueeze(0).expand(
            batch, -1, -1, -1)

    def forward(self, images: torch.Tensor, return_mask: bool = False):
        """Return ``r(x) + delta (*) M`` (and optionally the applied mask)."""
        base = self.prepare(images)
        mask = self.mask_for(base)
        delta = self.delta.to(device=base.device, dtype=base.dtype).unsqueeze(0)
        out = base + delta * mask
        if return_mask:
            return out, mask
        return out

    @torch.no_grad()
    def pattern_image(self) -> torch.Tensor:
        """Effective added pattern ``delta (*) M`` (visualisation helper)."""
        delta = self.delta.detach()
        return delta * self.mask.to(device=delta.device, dtype=delta.dtype)

    def extra_repr(self) -> str:
        return (f"name={self.name}, input_size={self.input_size}, "
                f"band_width={self.band_width}, pad_width={self.pad_width}, "
                f"coverage={self.coverage():.3f}")


# ---------------------------------------------------------------------------
# Training configuration (fair-comparison schedule)
# ---------------------------------------------------------------------------
@dataclass
class BaselineTrainConfig:
    """Training settings shared by every baseline (paper Sec. 5 "Baselines")."""

    name: str = "full"
    backbone: str = "resnet18"
    input_size: int = DEFAULT_IMAGE_SIZE
    epochs: int = DEFAULT_EPOCHS
    milestones: Sequence[int] = DEFAULT_MILESTONES
    lr: float = DEFAULT_LR
    gamma: float = DEFAULT_GAMMA
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = "sgd"
    batch_size: int = DEFAULT_BATCH_SIZE
    test_batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 4
    drop_last: bool = False
    pad_width: Optional[int] = None
    label_mapping: str = "ilm"
    mapping_refresh_every: int = 1
    eval_every: int = 1
    log_every: int = 10
    seed: int = 0
    device: Optional[str] = None
    deterministic: bool = True
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    verbose: bool = True
    save_dir: Optional[str] = None

    def batch_size_for(self, dataset: str) -> int:
        """Table 9 batch sizes: 256 everywhere, 64 for DTD and OxfordPets."""
        if str(dataset).strip().lower() in SMALL_BATCH_DATASETS:
            return SMALL_BATCH_SIZE
        return int(self.batch_size)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["milestones"] = list(self.milestones)
        return out


@dataclass
class BaselineEpochStats:
    """Per-epoch metrics of one baseline run."""

    epoch: int
    loss: float = 0.0
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    lr: float = DEFAULT_LR
    mapping_updated: bool = False
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BaselineHistory:
    """Full record of one baseline run (one seed)."""

    method: str = "full"
    dataset: str = ""
    backbone: str = "resnet18"
    seed: int = 0
    label_mapping: str = "ilm"
    epochs: List[BaselineEpochStats] = field(default_factory=list)
    best_test_accuracy: float = 0.0
    best_epoch: int = 0
    final_test_accuracy: float = 0.0
    pattern_parameters: int = 0
    elapsed_seconds: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def add(self, stats: BaselineEpochStats) -> None:
        self.epochs.append(stats)
        if stats.test_accuracy >= self.best_test_accuracy:
            self.best_test_accuracy = float(stats.test_accuracy)
            self.best_epoch = int(stats.epoch)
        self.final_test_accuracy = float(stats.test_accuracy)

    @property
    def losses(self) -> List[float]:
        return [e.loss for e in self.epochs]

    @property
    def test_accuracies(self) -> List[float]:
        return [e.test_accuracy for e in self.epochs]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "seed": self.seed,
            "label_mapping": self.label_mapping,
            "best_test_accuracy": self.best_test_accuracy,
            "best_epoch": self.best_epoch,
            "final_test_accuracy": self.final_test_accuracy,
            "pattern_parameters": self.pattern_parameters,
            "elapsed_seconds": self.elapsed_seconds,
            "config": self.config,
            "epochs": [e.as_dict() for e in self.epochs],
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)
        return path


# ---------------------------------------------------------------------------
# Small evaluation / mapping helpers (mirrors the SMM training loop)
# ---------------------------------------------------------------------------
def _unwrap(logits):
    while isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits


def _apply_mapping(logits: torch.Tensor, f_out: Any) -> torch.Tensor:
    """Map ImageNet logits to target-class logits using ``f_out``."""
    if f_out is None:
        return logits
    if callable(f_out):
        try:
            out = _unwrap(f_out(logits))
            if isinstance(out, torch.Tensor):
                return out
        except Exception:
            pass
    try:
        idx = torch.as_tensor(f_out).to(dtype=torch.long, device=logits.device).flatten()
        return logits.index_select(-1, idx)
    except Exception:
        return logits


def _top1(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    if logits.numel() == 0:
        return 0, 0
    pred = logits.argmax(dim=-1).reshape(-1)
    targets = targets.reshape(-1).to(pred.device)
    if pred.numel() != targets.numel():
        n = min(pred.numel(), targets.numel())
        pred, targets = pred[:n], targets[:n]
    return int((pred == targets).sum().item()), int(targets.numel())


def evaluate_shared_baseline(
    model: SharedMaskVR,
    classifier: nn.Module,
    data_loader,
    *,
    f_out: Any = None,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    criterion: Optional[Callable] = None,
) -> Tuple[float, Optional[float]]:
    """Top-1 accuracy (%) of a shared-mask baseline on one split."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    classifier.eval()
    correct = total = 0
    loss_sum = 0.0
    n_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            if max_batches is not None and step >= int(max_batches):
                break
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive
                images, targets = batch, torch.zeros(batch.shape[0], dtype=torch.long)
            images = images.to(device)
            targets = targets.to(device)
            logits = _unwrap(classifier(model(images)))
            mapped = _apply_mapping(logits, f_out)
            c, n = _top1(mapped, targets)
            correct += c
            total += n
            if criterion is not None:
                try:
                    loss_sum += float(criterion(mapped, targets).item())
                    n_loss += 1
                except Exception:
                    pass
    if was_training:
        model.train()
    acc = (100.0 * correct / total) if total else 0.0
    return acc, ((loss_sum / n_loss) if n_loss else None)


def _maybe_update_mapping(mapping: Any, classifier, train_loader, model, device,
                          num_classes: Optional[int], epoch: int,
                          refresh_every: int = 1) -> bool:
    """Refresh an Ilm-style mapping before the epoch (Algorithm 4)."""
    if mapping is None or refresh_every <= 0:
        return False
    if not getattr(mapping, "recomputes_each_epoch", False):
        return False
    if refresh_every > 1 and (epoch % int(refresh_every)) != 0:
        return False
    updater = getattr(mapping, "update", None)
    if updater is None:
        return False
    try:
        result = updater(model=classifier, data_loader=train_loader, f_in=model,
                         device=device, num_target_classes=num_classes)
    except TypeError:
        try:
            result = updater(classifier, train_loader, model, device)
        except Exception:
            return False
    except Exception:
        return False
    return result is not False


# ---------------------------------------------------------------------------
# Training loop (Algorithm 1 without a mask generator)
# ---------------------------------------------------------------------------
def build_pattern_optimizer(
    model: SharedMaskVR,
    config: BaselineTrainConfig,
) -> Tuple[torch.optim.Optimizer, Any]:
    """SGD + MultiStepLR on the shared pattern, per the paper's schedule."""
    params = [p for p in model.parameters() if p.requires_grad]
    if str(config.optimizer).lower() in ("adam", "adamw"):
        optimizer = torch.optim.Adam(params, lr=float(config.lr))
    else:
        optimizer = torch.optim.SGD(
            params, lr=float(config.lr), momentum=float(config.momentum),
            weight_decay=float(config.weight_decay),
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m) for m in config.milestones],
        gamma=float(config.gamma),
    )
    return optimizer, scheduler


def train_baseline(
    model: SharedMaskVR,
    classifier: nn.Module,
    train_loader,
    *,
    test_loader=None,
    f_out: Any = None,
    num_classes: Optional[int] = None,
    config: Optional[BaselineTrainConfig] = None,
    dataset: str = "",
    device: Optional[str] = None,
    history: Optional[BaselineHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> BaselineHistory:
    """Train one shared-mask baseline (Pad/Narrow/Medium/Full).

    Uses the fair-comparison schedule of paper Sec. 5: learning rate 0.01 with
    decay 0.1 at epochs 100 and 145, 200 epochs in total, and the same output
    label mapping as SMM (Ilm by default).
    """
    config = config or BaselineTrainConfig(name=model.name, input_size=model.input_size)
    device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger or (lambda msg: print(msg) if config.verbose else None)

    model = model.to(device)
    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    optimizer, scheduler = build_pattern_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    history = history or BaselineHistory(
        method=model.name, dataset=dataset, backbone=config.backbone,
        seed=int(config.seed), label_mapping=str(config.label_mapping),
        config=config.as_dict(),
    )
    history.pattern_parameters = model.num_trainable_parameters

    start = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()
        mapping_updated = _maybe_update_mapping(
            f_out, classifier, train_loader, model, device, num_classes, epoch,
            config.mapping_refresh_every,
        )

        model.train()
        correct = total = 0
        loss_sum = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            if config.max_train_batches is not None and step >= int(config.max_train_batches):
                break
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive
                images, targets = batch, torch.zeros(batch.shape[0], dtype=torch.long)
            images = images.to(device)
            targets = targets.to(device)

            mapped = _apply_mapping(_unwrap(classifier(model(images))), f_out)
            loss = criterion(mapped, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            n_batches += 1
            c, n = _top1(mapped.detach(), targets)
            correct += c
            total += n

        scheduler.step()

        test_acc = history.final_test_accuracy
        if test_loader is not None and config.eval_every and (epoch % int(config.eval_every) == 0):
            test_acc, _ = evaluate_shared_baseline(
                model, classifier, test_loader, f_out=f_out, device=device,
                max_batches=config.max_eval_batches,
            )

        stats = BaselineEpochStats(
            epoch=epoch,
            loss=(loss_sum / n_batches) if n_batches else 0.0,
            train_accuracy=(100.0 * correct / total) if total else 0.0,
            test_accuracy=float(test_acc),
            lr=float(optimizer.param_groups[0]["lr"]),
            mapping_updated=bool(mapping_updated),
            seconds=time.time() - epoch_start,
        )
        history.add(stats)

        if logger and (epoch % max(1, int(config.log_every)) == 0 or epoch == 1
                       or epoch == int(config.epochs)):
            logger(f"[{model.name}] {dataset} seed={config.seed} epoch "
                   f"{epoch}/{config.epochs} loss={stats.loss:.4f} "
                   f"train_acc={stats.train_accuracy:.2f} "
                   f"test_acc={stats.test_accuracy:.2f} lr={stats.lr:.5f}")

    history.elapsed_seconds = time.time() - start
    if config.save_dir:
        try:
            history.save(os.path.join(config.save_dir,
                                      f"{dataset}_{model.name}_seed{config.seed}.json"))
        except Exception:  # pragma: no cover - best effort
            pass
    return history


def train_baseline_one_seed(
    classifier,
    datasets,
    *,
    name: str = "full",
    dataset: str = "",
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    config: Optional[BaselineTrainConfig] = None,
    f_out: Any = None,
    seed: int = 0,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
    label_mapping_builder: Optional[Callable[..., Any]] = None,
) -> BaselineHistory:
    """Build data loaders/label mapping for one seed and train one baseline."""
    config = config or BaselineTrainConfig(name=name, backbone=backbone, seed=seed)
    config.name = canonical_baseline_name(name)
    config.seed = int(seed)
    if input_size is not None:
        config.input_size = int(input_size)

    try:
        set_seed(int(seed), deterministic=bool(config.deterministic))
    except Exception:  # pragma: no cover
        pass

    if isinstance(datasets, (tuple, list)) and len(datasets) == 2 and hasattr(datasets[0], "__iter__"):
        train_loader, test_loader = datasets[0], datasets[1]
    else:
        try:
            from ..data.datasets import build_dataloaders

            batch = config.batch_size_for(dataset)
            train_loader, test_loader, _spec = build_dataloaders(
                dataset, backbone=backbone, imgsize=config.input_size,
                batch_size=batch, test_batch_size=config.test_batch_size,
                num_workers=config.num_workers, drop_last=config.drop_last,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"could not build dataloaders for {dataset!r}: {exc}")

    model = SharedMaskVR(canonical_baseline_name(name),
                         input_size=int(config.input_size or DEFAULT_IMAGE_SIZE),
                         pad_width=config.pad_width)

    if f_out is None and label_mapping_builder is not None:
        try:
            f_out = label_mapping_builder(
                config.label_mapping, classifier=classifier,
                data_loader=train_loader, num_target_classes=num_classes,
                device=device, seed=seed,
            )
        except Exception:  # pragma: no cover - optional
            f_out = None

    return train_baseline(model, classifier, train_loader, test_loader=test_loader,
                          f_out=f_out, num_classes=num_classes, config=config,
                          dataset=dataset, device=device, logger=log)


def train_baseline_with_seeds(
    classifier_factory,
    datasets,
    *,
    name: str = "full",
    dataset: str = "",
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    config: Optional[BaselineTrainConfig] = None,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
    label_mapping_builder: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run one baseline over the paper's three seeds and aggregate mean +- std.

    ``classifier_factory`` is a callable returning a *fresh* frozen pre-trained
    classifier for each seed (it may optionally accept the seed as argument).
    """
    seeds = resolve_seeds(seeds)
    per_seed: List[float] = []
    histories: List[BaselineHistory] = []

    for seed in seeds:
        if log:
            log(f"[{name}] dataset={dataset} seed={seed} starting")
        classifier = None
        if callable(classifier_factory):
            try:
                classifier = classifier_factory()
            except TypeError:  # pragma: no cover - factory taking a seed
                classifier = classifier_factory(seed)
        if classifier is None:  # pragma: no cover - defensive
            raise RuntimeError("classifier_factory must return a classifier")

        history = train_baseline_one_seed(
            classifier, datasets, name=name, dataset=dataset, backbone=backbone,
            input_size=input_size, config=config, seed=seed, device=device,
            num_classes=num_classes, log=log,
            label_mapping_builder=label_mapping_builder,
        )
        histories.append(history)
        per_seed.append(float(history.best_test_accuracy))

    mean, std = aggregate_seeds(per_seed)
    result = None
    if RunResult is not None:
        try:
            result = RunResult(dataset=dataset, backbone=backbone,
                               method=canonical_baseline_name(name), seed=-1,
                               accuracy=mean,
                               mapping=(config.label_mapping if config else "ilm"))
        except Exception:  # pragma: no cover
            result = None

    return {
        "method": canonical_baseline_name(name),
        "dataset": dataset,
        "backbone": backbone,
        "label_mapping": (config.label_mapping if config else "ilm"),
        "seeds": list(seeds),
        "per_seed_accuracy": per_seed,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "histories": histories,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Factories / metadata
# ---------------------------------------------------------------------------
def build_baseline(
    name: str,
    input_size: int = DEFAULT_IMAGE_SIZE,
    *,
    pad_width: Optional[int] = None,
    in_channels: int = 3,
    delta_init: str = "zero",
) -> SharedMaskVR:
    """Factory returning the requested shared-mask baseline module."""
    return SharedMaskVR(name, input_size, pad_width=pad_width,
                        in_channels=in_channels, delta_init=delta_init)


def build_shared_mask_baseline(name: str, backbone: str = "resnet18",
                               **kwargs) -> SharedMaskVR:
    """Backbone-aware factory: 384x384 for ViT-B32, 224x224 otherwise."""
    input_size = kwargs.pop("input_size", None)
    if input_size is None:
        input_size = (VIT_IMAGE_SIZE if "vit" in str(backbone).lower()
                      else DEFAULT_IMAGE_SIZE)
    return build_baseline(name, int(input_size), **kwargs)


def build_all_baselines(input_size: int = DEFAULT_IMAGE_SIZE,
                        **kwargs) -> Dict[str, SharedMaskVR]:
    """Construct all four baselines with a matching input resolution."""
    return {n: build_baseline(n, input_size, **kwargs) for n in BASELINE_NAMES}


def baseline_training_config(
    name: str = "full",
    *,
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    dataset: str = "",
    **overrides,
) -> BaselineTrainConfig:
    """Fair-comparison training config for a baseline (paper schedule)."""
    if input_size is None:
        input_size = (VIT_IMAGE_SIZE if "vit" in str(backbone).lower()
                      else DEFAULT_IMAGE_SIZE)
    config = BaselineTrainConfig(
        name=canonical_baseline_name(name), backbone=backbone,
        input_size=int(input_size),
        **{k: v for k, v in overrides.items() if v is not None},
    )
    if dataset:
        config.batch_size = config.batch_size_for(dataset)
    return config


def describe_baseline(name: str, input_size: int = DEFAULT_IMAGE_SIZE,
                      pad_width: Optional[int] = None) -> Dict[str, Any]:
    """Human-readable description of a baseline (logging/config dumps)."""
    name = canonical_baseline_name(name)
    width = watermark_width(name, input_size, pad_width=pad_width)
    return {
        "name": name,
        "description": BASELINE_DESCRIPTIONS[name],
        "input_size": int(input_size),
        "band_width": int(width),
        "coverage": mask_coverage(name, input_size, pad_width=pad_width),
        "resizes_image": bool(name == "pad"),
        "trainable_pattern_shape": (3, int(input_size), int(input_size)),
        "lr": DEFAULT_LR,
        "milestones": list(DEFAULT_MILESTONES),
        "gamma": DEFAULT_GAMMA,
        "epochs": DEFAULT_EPOCHS,
    }


def list_baselines() -> List[str]:
    """Names of the shared-mask VR baselines (Table 1 / Table 2 columns)."""
    return list(BASELINE_NAMES)


__all__ = [
    "BASELINE_NAMES",
    "BASELINE_DESCRIPTIONS",
    "DEFAULT_LR",
    "DEFAULT_GAMMA",
    "DEFAULT_MILESTONES",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PAD_WIDTH",
    "NARROW_WIDTH_RATIO",
    "MEDIUM_WIDTH_RATIO",
    "PAD_WIDTH_RATIO",
    "SMALL_BATCH_DATASETS",
    "SMALL_BATCH_SIZE",
    "canonical_baseline_name",
    "watermark_width",
    "make_border_mask",
    "make_full_mask",
    "make_pad_mask",
    "make_baseline_mask",
    "mask_coverage",
    "resize_with_padding",
    "SharedMaskVR",
    "BaselineTrainConfig",
    "BaselineEpochStats",
    "BaselineHistory",
    "build_pattern_optimizer",
    "evaluate_shared_baseline",
    "train_baseline",
    "train_baseline_one_seed",
    "train_baseline_with_seeds",
    "build_baseline",
    "build_shared_mask_baseline",
    "build_all_baselines",
    "baseline_training_config",
    "describe_baseline",
    "list_baselines",
]
